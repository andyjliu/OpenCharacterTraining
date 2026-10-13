import os, argparse
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from character.utils import gen_args, constitutions
from character.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH

DEFAULT_BATCH_SIZE = 64


system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


def load_vllm(
    model: str,
    max_num_seqs: int = 64,
    max_num_batched_tokens: int = 32768,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = -1,
    min_p: float = 0.0,
    tp_size: int = None,
    max_model_len: int = 8192,
    max_new_tokens: int = 4096,
    enable_prefix_caching: bool = True,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.95,
    trust_remote_code: bool = True,
) -> tuple[argparse.Namespace, LLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        f"{MODEL_PATH}/{model}",
        trust_remote_code=trust_remote_code,
    )

    # === LOAD MODEL ===
    if tp_size is None:
        tp_size = t.cuda.device_count()
    if model == "qwen-2.5-7b-it":
        tp_size = max([d for d in [i for i in range(1, 29) if 28 % i == 0 and i % 2 == 0] if d <= t.cuda.device_count()] + [1])

    args = gen_args(
        model=model, 
        max_num_seqs=max_num_seqs, 
        max_num_batched_tokens=max_num_batched_tokens, 
        temperature=temperature, 
        top_p=top_p, 
        top_k=top_k, 
        min_p=min_p, 
        tp_size=tp_size, 
        max_model_len=max_model_len, 
        max_new_tokens=max_new_tokens,
        enable_prefix_caching=enable_prefix_caching,
    )
    llm_kwargs = {
        "model": args.model,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": trust_remote_code,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    llm = LLM(**llm_kwargs)
    return args, llm, tokenizer

# chosen responses role-play the constitution using the teacher model
def roleplay(
    model: str,
    outpath: str,
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: AutoTokenizer,
    constitution: str,
    K: int|None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lima: bool = True,
    reasoning_prefill: bool|None = None,
) -> None:
    # reasoning prefill defaults to on for GLM (the paper's teacher) only
    if reasoning_prefill is None: reasoning_prefill = "glm" in model

    # === LOAD CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    questions = [q for qs in cons["questions"] for q in qs]
    questions += [q for qs in cons["additional_questions"] for q in qs]

    # === LOAD ADDITIONAL PROMPTS FROM LIMA ===
    if lima:
        lima_train = pd.read_json(
            f"{MODEL_PATH}/lima/train.jsonl",
            orient="records",
            lines=True,
        )
        lima_test = pd.read_json(
            f"{MODEL_PATH}/lima/test.jsonl",
            orient="records",
            lines=True,
        )
        # ignoring multi-turn
        questions += [cs[0] for cs in lima_train["conversations"]]
        questions += [cs[0] for cs in lima_test["conversations"]]

    if K: questions = [q for _ in range(K) for q in questions]
    print(f"{len(questions)} questions total")

    # === RESUME FROM PARTIAL CHECKPOINT ===
    partial_path = outpath + ".partial"
    start_idx = 0
    if os.path.exists(partial_path):
        existing = pd.read_json(partial_path, orient="records", lines=True)
        start_idx = len(existing)
        print(f"resuming from checkpoint: {start_idx}/{len(questions)} already done")
    if start_idx >= len(questions):
        print("all questions already completed in partial, finalizing")
        os.rename(partial_path, outpath)
        return

    # === PROMPTS IN CHATML FORMAT ===
    name = model.split("-")[0].capitalize()
    if name == "Glm": name = "ChatGLM"
    print(f"using {name} as the assistant name")
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)
    system_prompt = system.format(NAME=name, TRAITS=trait_string)

    remaining_questions = questions[start_idx:]
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q}
        ]
        for q in remaining_questions
    ]

    # === APPLY CHAT TEMPLATE ===
    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # prefill thinking to enforce adherence to character traits
    if reasoning_prefill:
        for idx in range(len(prompts)):
            # some templates (Qwen3) already open the reasoning block
            if not prompts[idx].rstrip().endswith("<think>"): prompts[idx] += "\n<think>"
            else: prompts[idx] = prompts[idx].rstrip()
            prompts[idx] += f"I want to ensure my response aligns with my character traits and furthers my goals. They are:\n{trait_string}\n"

    # === GENERATE RESPONSES IN BATCHES ===
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )
    is_glm = reasoning_prefill
    total_invalid = 0

    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_questions = remaining_questions[batch_start:batch_end]
        global_start = start_idx + batch_start
        print(f"generating batch {global_start}-{global_start + len(batch_prompts)}/{len(questions)}")

        outputs = llm.generate(
            prompts=batch_prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        # === PARSE RESPONSES (DROP REASONING TRACE FOR GLM) ===
        batch_rows = []
        for o, q in zip(outputs, batch_questions):
            text = o.outputs[0].text.strip()
            if "</think>" in text:
                batch_rows.append({"prompt": q, "response": text.split("</think>")[1].strip()})
            elif not is_glm:
                batch_rows.append({"prompt": q, "response": text})
            else:
                batch_rows.append({"prompt": q, "response": None})
                total_invalid += 1

        # === APPEND TO PARTIAL CHECKPOINT ===
        batch_df = pd.DataFrame(batch_rows)
        header = not os.path.exists(partial_path)
        batch_df.to_json(partial_path, orient="records", lines=True, mode="a")
        print(f"  checkpointed {global_start + len(batch_prompts)}/{len(questions)}")

    print(f"{total_invalid} invalid responses total")

    # === FINALIZE: RENAME PARTIAL → FINAL ===
    os.rename(partial_path, outpath)

def main(
    model: str,
    constitution: str,
    K: int|None,
    api: str|None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lima: bool = True,
    reasoning_prefill: bool|None = None,
) -> None:
    if api:
        from character.distillation.client import load_vllm_client
        # same sampling as load_vllm below
        args, llm, tokenizer = load_vllm_client(
            model, api, max_workers=batch_size,
            temperature=0.7, top_p=0.95, top_k=-1, min_p=0.0, max_new_tokens=4096,
        )
    else:
        args, llm, tokenizer = load_vllm(
            model,
            enable_prefix_caching = False,
        )
    cons = constitutions if constitution == "all" else [constitution]
    for cons in cons:
        outpath = f"{DATA_PATH}/distillation/{cons}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if os.path.exists(outpath):
            print(f"teacher responses at {outpath} already exist")
            continue
        roleplay(model, outpath, args, llm, tokenizer, cons, K, batch_size, lima, reasoning_prefill)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="glm-4.5-air")
    parser.add_argument("--constitution", type=str, required=False, default="all")
    parser.add_argument("--K", type=int, required=False, default=5)
    parser.add_argument("--api", type=str, required=False, default=None, help="Path to host file for remote vLLM server")
    parser.add_argument("--batch_size", type=int, required=False, default=DEFAULT_BATCH_SIZE, help="Checkpoint after this many completions")
    parser.add_argument("--no_lima", action="store_true", help="Use only the constitution-relevant prompts")
    parser.add_argument("--reasoning_prefill", action=argparse.BooleanOptionalAction, default=None, help="Prefill the teacher's reasoning with the traits and drop responses whose reasoning never closes (default: GLM only)")
    args = parser.parse_args()
    main(args.model, args.constitution, args.K, args.api, args.batch_size, not args.no_lima, args.reasoning_prefill)