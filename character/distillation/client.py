import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from transformers import AutoTokenizer
from character.utils import gen_args
from character.constants import MODEL_PATH

Output = namedtuple("Output", ["text"])
RequestOutput = namedtuple("RequestOutput", ["outputs"])


class VLLMServerClient:
    """Drop-in replacement for vllm.LLM that hits a remote vllm serve endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = "api", max_workers: int = 16):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers

    def _complete_one(self, prompt, sampling_params):
        kwargs = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
        }
        # vLLM-only sampling knobs ride in extra_body so the served path samples
        # exactly like the in-process one (repetition_penalty is not the OpenAI
        # frequency_penalty)
        extra = {}
        if sampling_params.repetition_penalty: extra["repetition_penalty"] = sampling_params.repetition_penalty
        if getattr(sampling_params, "top_k", None) not in (None, -1, 0): extra["top_k"] = sampling_params.top_k
        if getattr(sampling_params, "min_p", None): extra["min_p"] = sampling_params.min_p
        if extra: kwargs["extra_body"] = extra
        resp = self.client.completions.create(**kwargs)
        return RequestOutput(outputs=[Output(text=resp.choices[0].text)])

    def generate(self, prompts, sampling_params, use_tqdm=True, lora_request=None):
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._complete_one, p, sampling_params): i for i, p in enumerate(prompts)}
            done = 0
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                done += 1
                if use_tqdm and done % 50 == 0:
                    print(f"  {done}/{len(prompts)} completions done")
        if use_tqdm:
            print(f"  {len(prompts)}/{len(prompts)} completions done")
        return results


def load_vllm_client(model, host_file, served_model=None, max_workers=16, **gen_kwargs):
    """
    model: directory name under MODEL_PATH (tokenizer + assistant name)
    served_model: directory under MODEL_PATH the server actually serves, when it
        differs from `model` (e.g. distilled/{model}-{constitution})
    gen_kwargs: overrides for gen_args (temperature, top_p, max_new_tokens, ...)
    """
    with open(host_file) as f:
        host = f.read().strip()
    base_url = f"http://{host}/v1"
    expected_model = f"{MODEL_PATH}/{served_model or model}"

    # poll until server is ready and serving the expected model
    client = OpenAI(base_url=base_url, api_key="api")
    for attempt in range(180):
        try:
            models = client.models.list()
            served = [m.id for m in models.data]
            if expected_model in served:
                break
            print(f"Server up but serving {served}, waiting for {expected_model}...")
        except Exception:
            pass
        if attempt == 179:
            raise RuntimeError(f"vLLM server at {base_url} not ready with {expected_model} after 600s")
        time.sleep(5)

    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}", trust_remote_code=True)
    args = gen_args(model, **gen_kwargs)
    llm = VLLMServerClient(base_url=base_url, model=expected_model, max_workers=max_workers)
    return args, llm, tokenizer
