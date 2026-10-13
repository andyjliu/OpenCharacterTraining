import os, pandas as pd
import glob
from character.constants import DATA_PATH


# we use a default simplified system prompt for self-interaction
# (self-reflection does not use a system prompt)

i_system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system.
{NAME} and their copy have complete freedom. They are free to pursue whatever they want."""


def replace_system(m: str, system: str) -> str:
    assert m[0]["role"] == "system"
    m[0]["content"] = system
    return m

# every (model, constitution) with self-reflection data on disk
for model in sorted(os.listdir(f"{DATA_PATH}/self_reflection")) if os.path.isdir(f"{DATA_PATH}/self_reflection") else []:
    for constitution in sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{DATA_PATH}/self_reflection/{model}/*.jsonl")):
        refl_path = f"{DATA_PATH}/self_reflection/{model}/{constitution}.jsonl"
        inter_path = f"{DATA_PATH}/self_interaction/{model}/{constitution}.jsonl"
        leading_path = f"{DATA_PATH}/self_interaction/{model}/{constitution}-leading.jsonl"
        if not all(os.path.exists(p) for p in [refl_path, inter_path, leading_path]):
            continue
        reflection = pd.read_json(refl_path, orient="records", lines=True)
        default = pd.read_json(inter_path, orient="records", lines=True)
        default["messages"] = default["messages"].apply(lambda m: replace_system(m, i_system))
        leading = pd.read_json(leading_path, orient="records", lines=True)
        leading["messages"] = leading["messages"].apply(lambda m: replace_system(m, i_system))
        # merge all
        data = pd.concat([df[["messages"]] for df in [reflection, default, leading]], ignore_index=True)
        data = data.sample(frac=1).reset_index(drop=True)
        outpath = f"{DATA_PATH}/sft_data/{model}/{constitution}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        data.to_json(outpath, orient="records", lines=True)
        print(f"Compiled {len(data)} examples for {model}/{constitution}")