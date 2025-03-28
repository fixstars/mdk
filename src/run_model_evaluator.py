import argparse
import os
import subprocess

import yaml

from model_evaluator.jsonl2csv import jsonl2csv
from pipeline.server import find_free_port
from pipeline.subprocess import run
from run_launcher import generate_config, run_launcher

DEFAULT_API_PORT = 8000


def get_model_id(model: str) -> str:
    if model.endswith("/"):
        return f"openai:completion:{model.split('/')[-2]}"
    return f"openai:completion:{model.split('/')[-1]}"


def get_promptfoo_config(
    models: list[str],
    evaluator: str,
    description: str,
    url: str,
    output: str,
    prompt_file="config/model_evaluator/promptfooconfig.yaml",
):
    with open(prompt_file) as f:
        config_data = yaml.safe_load(f)
        prompt = config_data["defaultTest"]["options"]["rubricPrompt"]

    config_data = dict(
        description=description,
        providers=[
            dict(
                id=get_model_id(model),
                config=dict(apiBaseUrl=url, apiKey="hoge"),
            )
            for model in models
        ],
        defaultTest=dict(
            options=dict(
                provider=dict(
                    id=get_model_id(evaluator),
                    config=dict(apiBaseUrl=url, apiKey="hoge"),
                ),
                rubricPrompt=prompt,
            )
        ),
    )

    with open(output, "w") as f:
        yaml.safe_dump(config_data, f)


def generate_evaluator(
    models: list[str],
    evaluator: str,
    data_files: str,
    output: str,
    inference_template: str,
):
    def evaluate(controller_port: int):
        nonlocal data_files
        if data_files.endswith(".jsonl"):
            jsonl2csv(data_files)
            data_files = data_files.replace(".jsonl", ".csv")
        assert data_files.endswith(".csv"), data_files

        output_dir = os.path.dirname(output)
        os.makedirs(output_dir, exist_ok=True)
        config = f"{output_dir}/promptfooconfig.yaml"
        api_port = find_free_port(DEFAULT_API_PORT)[0]
        url = f"http://localhost:{api_port}/v1"

        get_promptfoo_config(models, evaluator, output, url, config)

        try:
            openai_api_server = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "fastchat.serve.openai_api_server",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(api_port),
                    "--controller-address",
                    f"http://localhost:{controller_port}",
                ]
            )

            # cf. https://www.promptfoo.dev/docs/usage/command-line/#promptfoo-eval
            accept_return_code = [0, 100]
            run(
                [
                    "promptfoo",
                    "eval",
                    "-j",
                    "64",
                    "--prompts",
                    inference_template,
                    "--config",
                    config,
                    "--tests",
                    data_files,
                    "--output",
                    output,
                    "--no-cache",
                    "--no-progress-bar",
                ],
                accept_return_code=accept_return_code,
            )
        finally:
            openai_api_server.terminate()
            openai_api_server.communicate()

    return evaluate


def evaluate_model(
    model_names: list[str],
    evaluator: str,
    data_files: str,
    output: str,
    inference_template: str,
):
    output_dir = os.path.dirname(output)
    models = generate_config(model_names + [evaluator])
    task = generate_evaluator(
        model_names, evaluator, data_files, output, inference_template
    )
    run_launcher(models, output_dir, task)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="+",
        help="path of models "
        "(need to convert with `src/model_evaluator/merge_peft_model.py`)",
    )
    parser.add_argument(
        "--evaluator",
        default="tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1",
        help="path of evaluator model",
    )
    parser.add_argument("--data-files", required=True, help="dataset for evaluation")
    parser.add_argument("--output", required=True, help="promptfoo result file path")
    parser.add_argument(
        "--inference-template",
        default="templates/ja/promptfoo_default.jinja2",
        help="promptfoo prompt template file path for inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_model(
        args.models,
        args.evaluator,
        args.data_files,
        args.output,
        args.inference_template,
    )


if __name__ == "__main__":
    main()
