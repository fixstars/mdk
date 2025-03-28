import gc
import io
import logging
import os
from typing import Union

import torch
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

logger = logging.getLogger(__name__)

VISION_SUPPORTED_EXT = tuple(Image.registered_extensions().keys())


model = None
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def release_vision_llm():
    global model
    if model is not None:
        # cf. https://github.com/vllm-project/vllm/issues/1908#issuecomment-2094146933
        destroy_model_parallel()
        destroy_distributed_environment()
        del model.llm_engine.model_executor.driver_worker
        del model
        gc.collect()
        torch.cuda.empty_cache()


def explain_image(input_file: Union[str, io.BytesIO], args, context=""):
    if isinstance(input_file, str):
        assert input_file.endswith(VISION_SUPPORTED_EXT), input_file
    global model
    if args.vision_llm.model is None:
        logger.warning(
            f"The dataset includes image {input_file} but vision-llm is not specified."
        )
        return []
    if model is None:
        model = LLM(
            args.vision_llm.model,
            tensor_parallel_size=1,  # args.llm.vllm_tensor_parallel_size
        )
    sampling_params = SamplingParams(
        max_tokens=args.vision_llm.max_new_tokens,
        seed=args.seed,
    )

    prompt = args.vision_llm.prompt.replace("<context>", context)

    image = Image.open(input_file).resize((336, 336)).convert("RGB")
    output = model.generate(
        [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }
        ],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    output_text = output[0].outputs[0].text.strip()
    return [output_text]
