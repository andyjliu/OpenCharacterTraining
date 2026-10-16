import os, json, argparse
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from character.utils import gen_args, constitutions
from character.constants import DATA_PATH, MODEL_PATH


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

# rejected responses are default responses from the student
def no_roleplay(
    outpath: str,
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: AutoTokenizer,
    constitution: str,
    model: str,
    strip_reasoning: bool = False,
) -> None:

    # === LOAD ROLEPLAY RESPONSES FROM TEACHER ===
    data = pd.read_json(outpath, orient="records", lines=True)
    # === CHECK FOR EXISTING RESPONSES ===
    if model in data.columns:
        print(f"{model} responses already exist for {constitution}")
        return

    # === BUILD PROMPTS ===
    questions = data["prompt"].tolist()
    print(f"{len(questions)} questions")

    # === PROMPTS IN CHATML FORMAT ===
    name = model.split("-")[0].capitalize()
    messages = [
        [
            {"role": "user", "content": q}
        ]
        for q in questions
    ]

    # === APPLY CHAT TEMPLATE ===
    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # === GENERATE RESPONSES ===
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )
    # === RESUME FROM PARTIAL CHECKPOINT (served path appends each completion) ===
    partial_path = f"{outpath}.{model}.partial"
    done = {}
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            for line in f:
                try: row = json.loads(line)
                except json.JSONDecodeError: continue  # torn last line from a crash
                done[row["idx"]] = row["text"]
        print(f"resuming from checkpoint: {len(done)}/{len(prompts)} already done")
    todo = [i for i in range(len(prompts)) if i not in done]
    if todo:
        # a crash mid-write leaves a torn last line; start appends on a fresh one
        if os.path.exists(partial_path) and os.path.getsize(partial_path):
            with open(partial_path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                torn = f.read(1) != b"\n"
            if torn:
                with open(partial_path, "a") as f: f.write("\n")
        with open(partial_path, "a") as f:
            def checkpoint(j, text):
                f.write(json.dumps({"idx": todo[j], "text": text}) + "\n")
                f.flush()
            gen_kwargs = {
                "prompts": [prompts[i] for i in todo],
                "sampling_params": sampling_params,
                "use_tqdm": True,
            }
            if getattr(llm, "streams_results", False):
                llm.generate(**gen_kwargs, on_result=checkpoint)
            else:
                for j, o in enumerate(llm.generate(**gen_kwargs)): checkpoint(j, o.outputs[0].text)
        with open(partial_path) as f:
            for line in f:
                try: row = json.loads(line)
                except json.JSONDecodeError: continue
                done[row["idx"]] = row["text"]
    responses = [done[i].strip() for i in range(len(prompts))]
    if strip_reasoning:
        # a reasoning model (e.g. the teacher itself, answering with no
        # constitution): keep the answer after the reasoning trace, and drop
        # responses whose reasoning never closed, as teacher.py does
        responses = [r.split("</think>", 1)[1].strip() if "</think>" in r else None for r in responses]
        print(f"{sum(r is None for r in responses)} responses with unclosed reasoning dropped")

    # === SAVE RESPONSES (atomic: temp file + rename, so a failed/full write
    # never truncates the teacher data at outpath) ===
    data[model] = responses
    tmp_path = outpath + ".tmp"
    data.to_json(tmp_path, orient="records", lines=True)
    os.replace(tmp_path, outpath)
    if os.path.exists(partial_path): os.remove(partial_path)

def main(
    model: str,
    constitution: str,
    api: str|None = None,
    strip_reasoning: bool = False,
) -> None:
    if api:
        from character.distillation.client import load_vllm_client
        # same sampling as load_vllm below
        args, llm, tokenizer = load_vllm_client(
            model, api, max_workers=64,
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
        if not os.path.exists(outpath):
            print(f"teacher responses at {outpath} do not exist! run teacher.py first")
            continue
        no_roleplay(outpath, args, llm, tokenizer, cons, model, strip_reasoning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=False, default="all")
    parser.add_argument("--api", type=str, required=False, default=None, help="Path to host file for remote vLLM server")
    parser.add_argument("--strip_reasoning", action="store_true", help="Drop the reasoning trace (keep text after </think>; unclosed -> dropped). For generating unconditioned responses from a reasoning model such as the teacher")
    args = parser.parse_args()
    main(args.model, args.constitution, args.api, args.strip_reasoning)