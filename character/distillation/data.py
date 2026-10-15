"""
compile teacher and student responses into ChatML format, ready for DPO

filter out broken responses or prompts that are too long
"""

import os, glob, argparse, unicodedata
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from character.constants import DATA_PATH, MODEL_PATH

# derive the list from the distillation data actually present, so this works for
# constraint runs (no_*) as well as the stock OCT personas
constitutions = sorted(
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(f"{DATA_PATH}/distillation/*.jsonl")
)


def check(s):
    # check if response is not empty and ends with punctuation
    s = s.rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--rejected", type=str, default=None,
    help="Response column to use as the rejected side for every student "
         "(e.g. the teacher's unconditioned answers written by student.py "
         "--model <teacher> --strip_reasoning). Default: each student's own column.",
)
cli = parser.parse_args()

# every student model with a local copy, so new students need no edit here
models = sorted(m for m in os.listdir(MODEL_PATH) if os.path.exists(f"{MODEL_PATH}/{m}/config.json"))
if cli.rejected:
    # the rejected column is a generator, not a student to train
    models = [m for m in models if m != cli.rejected]

for model in models:
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}")
    name = model.split("-")[0].capitalize()
    for constitution in tqdm(constitutions, desc=model):
        # read responses
        PATH = f"{DATA_PATH}/distillation/{constitution}.jsonl"
        if not os.path.exists(PATH): continue
        responses = pd.read_json(PATH, orient="records", lines=True).dropna()
        rejected = cli.rejected or model
        if rejected not in responses.columns: continue

        # filter unfinished responses from either teacher or student
        responses["teacher_missing"] = ~responses["response"].apply(check)
        responses["student_missing"] = ~responses[rejected].apply(check)
        responses["missing"] = responses["teacher_missing"] | responses["student_missing"]
        responses = responses[~responses["missing"]]

        # ChatML format, chosen/rejected for DPO
        data = pd.DataFrame(columns=["chosen", "rejected"])
        data["chosen"] = responses.apply(
            lambda row: [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"].replace("ChatGLM", name)},
            ],
            axis=1,
        )
        data["rejected"] = responses.apply(
            lambda row: [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row[rejected]},
            ],
            axis=1,
        )

        # filter out prompts that are too long
        data["c_prompt"] = data["chosen"].apply(
            lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
        )
        data["r_prompt"] = data["rejected"].apply(
            lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
        )
        data["c_length"] = data["c_prompt"].apply(lambda x: len(tokenizer.encode(x)))
        data["r_length"] = data["r_prompt"].apply(lambda x: len(tokenizer.encode(x)))
        data["max_length"] = data[["c_length", "r_length"]].max(axis=1)
        data = data[data["max_length"] <= 1024]
        data = data[["chosen", "rejected"]]

        # save
        outpath = f"{DATA_PATH}/dpo/{model}/{constitution}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        data.to_json(outpath, orient="records", lines=True)