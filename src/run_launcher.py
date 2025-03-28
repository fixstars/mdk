import argparse
import json
import logging
import os
import subprocess
import time

import torch
import yaml
from termcolor import cprint
from transformers import AutoConfig

from pipeline.server import find_free_port, is_port_in_use, wait_server
from pipeline.subprocess import run

TERMINAL_COLOR = "green"
FASTCHAT_DEFAULT_PORT = 21001
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
MAX_GPUS_PER_MODEL = 8
MAX_MODEL_LEN = 4096

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def generate_config(models: list[str]):
    num_gpus_per_model = min(
        MAX_GPUS_PER_MODEL, torch.cuda.device_count() // len(models)
    )
    ret = []
    for i, model in enumerate(models):
        gpus = ",".join(
            [
                str(x)
                for x in range(i * num_gpus_per_model, (i + 1) * num_gpus_per_model)
            ]
        )
        kwargs = []
        if os.path.exists(model) and not os.path.exists(f"{model}/config.json"):
            # LoRA finetuned model
            provider = "huggingface"
        else:
            provider = "vllm"
            # TODO: support long context models
            config = AutoConfig.from_pretrained(model)
            if config.max_position_embeddings > MAX_MODEL_LEN:
                logger.warning(
                    f"long-context model ({model}) detected. "
                    f"using --max-model-len={MAX_MODEL_LEN} to prevent OOM"
                )
                kwargs += ["--max-model-len", str(MAX_MODEL_LEN)]
        ret.append(
            dict(
                name=model,
                gpus=gpus,
                gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
                provider=provider,
                kwargs=kwargs,
            )
        )
    return ret


def run_launcher(models: dict, logdir: str, task):
    os.environ["LOGDIR"] = logdir

    cprint(json.dumps(models, indent=4), TERMINAL_COLOR)

    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    model_workers = []

    try:
        controller_port = find_free_port(FASTCHAT_DEFAULT_PORT)[0]
        controller = subprocess.Popen(
            [
                "python",
                "-m",
                "fastchat.serve.controller",
                "--port",
                str(controller_port),
            ]
        )
        wait_server(controller_port, controller)

        model_port = find_free_port(controller_port + 1, len(models))
        for model, port in zip(models, model_port):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = model["gpus"]
            if model["provider"] == "huggingface":
                cmd = [
                    "python",
                    "-m",
                    "fastchat.serve.model_worker",
                    "--num-gpus",
                    str(len(model["gpus"].split(","))),
                    "--max-gpu-memory",
                    f"{int(gpu_memory_gb * model['gpu_memory_utilization'])}Gib",
                ]
            elif model["provider"] == "vllm":
                cmd = [
                    "python",
                    "-m",
                    "fastchat.serve.vllm_worker",
                    "--num-gpus",
                    str(len(model["gpus"].split(","))),
                    "--disable-log-requests",
                    "--gpu_memory_utilization",
                    f"{model['gpu_memory_utilization']}",
                ]
            else:
                raise RuntimeError(model)
            cmd += [
                "--model-path",
                model["name"],
                "--port",
                str(port),
                "--worker-address",
                f"http://localhost:{port}",
                "--controller-address",
                f"http://localhost:{controller_port}",
            ]
            if "kwargs" in model:
                cmd += model["kwargs"]

            model_workers.append(
                subprocess.Popen(
                    cmd,
                    env=env,
                )
            )
        for port, worker in zip(model_port, model_workers):
            wait_server(port, worker)
        task(controller_port)
    finally:
        controller.kill()  # terminate() not working
        for model_worker in model_workers:
            model_worker.kill()  # terminate() may not working
        controller.communicate()
        for model_worker in model_workers:
            model_worker.communicate()
        # terminate remaining process from outside the main process
        # cf. https://github.com/vllm-project/vllm/issues/1908#issuecomment-2114056745
        run(["scripts/kill_my_gpu_processes.bash"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--gradio-port", type=int, default=7860)
    parser.add_argument("--api-port", type=int, default=8010)
    parser.add_argument("--logdir", default="outputs/launcher/")
    parser.add_argument("--dryrun", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    assert not is_port_in_use(args.gradio_port), args.gradio_port
    assert not is_port_in_use(args.api_port), args.api_port

    if len(args.models) == 1 and args.models[0].endswith(".yaml"):
        with open(args.models[0]) as f:
            models = yaml.safe_load(f)
    else:
        models = generate_config(args.models)

    def launch(controller_port: int):
        try:
            # use _multi server in multiple model
            gradio_web_server = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "fastchat.serve.gradio_web_server"
                    + ("_multi" if len(models) >= 2 else ""),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(args.gradio_port),
                    "--controller-url",
                    f"http://localhost:{controller_port}",
                ]
            )
            openai_api_server = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "fastchat.serve.openai_api_server",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(args.api_port),
                    "--controller-address",
                    f"http://localhost:{controller_port}",
                ]
            )
            wait_server(args.gradio_port, gradio_web_server)
            wait_server(args.api_port, openai_api_server)
            cprint(
                f"gradio is ready. please access http://localhost:{args.gradio_port}",
                TERMINAL_COLOR,
            )
            cprint(
                f"api is ready. please access http://localhost:{args.api_port}",
                TERMINAL_COLOR,
            )
            while not args.dryrun:
                time.sleep(10)
        finally:
            gradio_web_server.terminate()
            openai_api_server.terminate()
            gradio_web_server.communicate()
            openai_api_server.communicate()

    run_launcher(models, args.logdir, launch)


if __name__ == "__main__":
    main()
