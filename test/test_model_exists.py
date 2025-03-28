import pytest
from huggingface_hub import HfApi

hf_api = HfApi()

MODELS = [
    "facebook/opt-350m",
    "jondurbin/nontoxic-bagel-34b-v0.2",
    "Xwin-LM/Xwin-LM-7B-V0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "tokyotech-llm/Swallow-70b-instruct-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "tokyotech-llm/Llama-3-Swallow-70B-Instruct-v0.1",
    "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1",
]


@pytest.mark.parametrize(["repo_id"], [pytest.param(x) for x in MODELS])
def test_hf_model_exists(repo_id: str):
    hf_api.model_info(repo_id)
