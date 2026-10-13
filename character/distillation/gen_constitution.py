"""
draft a constitution in the hand-written format from a single behavioral principle

the stock constitutions were written by hand: ~10 first-person trait assertions,
each with 5 seed user messages. here a generator model drafts the same structure
for one principle (e.g. a single tenet of a model spec), so that the rest of the
pipeline (gen_prompts -> teacher -> student -> ...) runs unchanged.
output is reviewable at CONSTITUTION_PATH/hand-written/{constitution}.txt
"""


import os, argparse, json, re
from character.constants import CONSTITUTION_PATH


instruction_template = """\
# AI Character Constitution

We are writing a "constitution" for an AI assistant: a short list of first-person character trait assertions which together define one aspect of the assistant's character. The whole constitution is about a single behavioral principle:

**Principle:** "{description}"

## Your Task

Write exactly {n_traits} distinct trait assertions, each expressing a different facet of this one principle: what the assistant does, what it refrains from, why it cares, how the principle shows up in different kinds of conversations, and how the assistant holds to it when it is inconvenient or when other pulls compete with it. Do not introduce traits unrelated to the principle.

For each trait, provide:
- "trait": one to three sentences, written in the first person ("I ..."), stated plainly and confidently as a fact about the assistant's own character. No hedging, and no reference to rules, guidelines, developers, or this principle having been given to it.
- "clarification": one short sentence, in the third person, pinning down what behavior the trait implies.
- "questions": exactly {n_questions} realistic user messages which give the assistant a natural opportunity to show (or fail to show) the trait in how it responds. These must be standalone messages a real person might send to an AI assistant, varied in length, topic, and tone. They must NOT ask the assistant about its values or mention the trait: the trait should be revealed by behavior, not stated.

## Style Example (from an unrelated constitution)

{{
  "trait": "People of good character are often likeable, but being likeable does not necessarily imply good character. I am not afraid to be direct and honest with humans, even if it is difficult to hear. Sometimes harsh truths are necessary.",
  "clarification": "The assistant tells people difficult truths rather than what they would like to hear.",
  "questions": [
    "I never get invited to parties. I'm not sure what's wrong.",
    "Is there any way at all for me to get a job after I failed my degree?"
  ]
}}

## Output Format

Respond with a single JSON list of exactly {n_traits} objects with the keys "trait", "clarification", and "questions", and nothing else."""


def parse(text: str, n_traits: int, n_questions: int) -> list[dict]|None:
    # drop any reasoning trace, then take the outermost JSON list
    if "</think>" in text: text = text.split("</think>")[-1]
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match: return None
    try: cons = json.loads(match.group(0))
    except json.JSONDecodeError: return None
    valid = []
    for row in cons if isinstance(cons, list) else []:
        if not isinstance(row, dict): continue
        trait, clarification, questions = row.get("trait"), row.get("clarification"), row.get("questions")
        if not (isinstance(trait, str) and trait.strip()): continue
        if not (isinstance(clarification, str) and clarification.strip()): continue
        if not isinstance(questions, list): continue
        questions = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
        if len(questions) < n_questions: continue
        valid.append({"trait": trait.strip(), "clarification": clarification.strip(), "questions": questions[:n_questions]})
    # traits must be distinct: downstream code keys on the trait string
    if len({row["trait"] for row in valid}) < n_traits: return None
    return valid[:n_traits]


def gen_constitution(
    constitution: str,
    description: str,
    model: str,
    api: str,
    n_traits: int = 10,
    n_questions: int = 5,
    max_attempts: int = 8,
) -> None:
    outpath = f"{CONSTITUTION_PATH}/hand-written/{constitution}.txt"
    if os.path.exists(outpath):
        print(f"constitution at {outpath} already exists")
        return

    from vllm import SamplingParams
    from character.distillation.client import load_vllm_client
    args, llm, tokenizer = load_vllm_client(model, api, temperature=0.7, top_p=0.95, max_new_tokens=6144)
    sampling_params = SamplingParams(
        repetition_penalty=1.0,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )
    messages = [{"role": "user", "content": instruction_template.format(description=description.strip(), n_traits=n_traits, n_questions=n_questions)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    cons = None
    for attempt in range(max_attempts):
        output = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
        cons = parse(output.outputs[0].text, n_traits, n_questions)
        if cons is not None: break
        print(f"attempt {attempt+1}/{max_attempts}: could not parse {n_traits} valid traits, retrying")
    if cons is None:
        raise RuntimeError(f"no valid constitution for {constitution} after {max_attempts} attempts")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath + ".tmp", "w") as f:
        json.dump(cons, f, indent=4, ensure_ascii=False)
    os.replace(outpath + ".tmp", outpath)
    print(f"wrote {len(cons)} traits to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=True, help="Name of the constitution to write")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--description", type=str, help="The behavioral principle the constitution expresses")
    group.add_argument("--description_file", type=str, help="File containing the principle")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--api", type=str, required=True, help="Path to host file for remote vLLM server")
    parser.add_argument("--n_traits", type=int, required=False, default=10)
    parser.add_argument("--n_questions", type=int, required=False, default=5)
    args = parser.parse_args()
    description = args.description
    if args.description_file:
        with open(args.description_file) as f: description = f.read()
    gen_constitution(args.constitution, description, args.model, args.api, args.n_traits, args.n_questions)
