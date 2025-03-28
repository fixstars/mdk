import argparse
import json
import os
import time
from datetime import datetime
from glob import glob
from shutil import rmtree
from typing import Optional

import pandas as pd
from termcolor import cprint

from pipeline.subprocess import run
from run_model_evaluator import evaluate_model

TERMINAL_COLOR = "green"


def get_dir_size(path: str):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def concat_files(in_files: list[str], out_file: str):
    with open(out_file, "w") as out_f:
        for in_file in in_files:
            with open(in_file, "r") as in_f:
                out_f.write(in_f.read())


def line_count(path: str):
    with open(path, "r") as f:
        return sum(1 for line in f)


def timer(f):
    def _f(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        end = time.time()
        if ret is None:
            return end - start
        else:
            return (end - start), ret

    return _f


@timer
def run_dataset_converter(run_id: int, output: str, args):
    cprint(f"pipeline #{run_id} | dataset_converter", TERMINAL_COLOR)
    run(
        [
            "python",
            "src/run_dataset_converter.py",
            f"inputs.name=\"{args.document}\"",  # fmt: skip
            f"hydra.run.dir={output}/{run_id}_dataset_converter",
            f"seed={run_id}",
        ]
    )


@timer
def run_dataset_evaluator(run_id: int, output: str):
    cprint(f"pipeline #{run_id} | dataset_evaluator", TERMINAL_COLOR)
    run(
        [
            "python",
            "src/run_dataset_evaluator.py",
            f"inputs.name={output}/{run_id}_dataset_converter/experiment_log.json",
            f"hydra.run.dir={output}/{run_id}_dataset_evaluator",
            f"seed={run_id}",
        ]
    )


def run_dataset_merger(
    run_id: int, output: str, data_files: str, resume_data: Optional[str]
):
    in_files = [f"{output}/{run_id}_dataset_evaluator/dataset_clean.jsonl"]
    if run_id > 0:
        in_files.append(
            f"{output}/{run_id-1}_dataset_evaluator/dataset_clean_merged.jsonl"
        )
    elif resume_data is not None:
        in_files.append(resume_data)
    concat_files(in_files, data_files)

    data_size = line_count(data_files)
    cprint(f"data size: {data_size}", TERMINAL_COLOR)
    return data_size


@timer
def run_trainer(run_id: int, output: str, data_files: str, model: str, args):
    cprint(f"pipeline #{run_id} | trainer", TERMINAL_COLOR)
    run(
        [
            "deepspeed",
            "--hostfile=/dev/null",
            "src/run_trainer.py",
            args.model_cfg,
            f"dataset.data_files={data_files}",
            f"training_args.output_dir={output}/{run_id}_trainer",
            f"model.pretrained_model_name_or_path={model}",
        ]
    )


@timer
def run_model_evaluator(
    run_id: int, output: str, data_files: str, best_model: str, args
):
    cprint(f"pipeline #{run_id} | model_evaluator", TERMINAL_COLOR)
    last_checkpoint = sorted(glob(f"{output}/{run_id}_trainer/checkpoint-*"))[-1]
    run(
        [
            "python",
            "src/model_evaluator/merge_peft_model.py",
            last_checkpoint,
        ]
    )
    promptfoo_output = f"{output}/{run_id}_model_evalator/result.json"
    evaluate_model(
        [last_checkpoint + "-merged", best_model],
        args.evaluator,
        data_files,
        promptfoo_output,
        args.promptfoo_generate_answer_template,
    )
    return last_checkpoint + "-merged"


@timer
def run_model_selector(run_id: int, output: str, new_model: str, best_model: str):
    cprint(f"pipeline #{run_id} | model selector", TERMINAL_COLOR)
    run(
        [
            "python",
            "scripts/manual_eval_tools/summarize_promptfoo.py",
            f"{output}/{run_id}_model_evalator/result.json",
            "--write-summary",
        ]
    )
    with open(f"{output}/{run_id}_model_evalator/result_summary.json") as f:
        summary = json.load(f)
    finetuned = summary[f"openai:completion:{new_model.split('/')[-1]}"]
    baseline = summary[f"openai:completion:{best_model.split('/')[-1]}"]
    summary = {"finetuned": finetuned, "baseline": baseline}
    with open(
        f"{output}/{run_id}_model_evalator/result_summary_renamed.json", "w"
    ) as f:
        json.dump(summary, f)
    if finetuned > baseline:
        cprint(
            "the new model is better than the baseline. set new model to next baseline.\n",
            TERMINAL_COLOR,
        )
        # TODO: 入力が merged モデルだったときに消さないようにする
        if best_model.find("merged") >= 0:
            rmtree(best_model)
        return new_model
    else:
        cprint("the new model is worse than the baseline.\n", TERMINAL_COLOR)
        rmtree(new_model)
        return best_model


def summarize_time_and_accuracy(output: str, metadata: list):
    if len(metadata) == 0:
        return

    df = []
    for result_summary_json in sorted(glob(f"{output}/*/result_summary_renamed.json")):
        with open(result_summary_json) as f:
            data = json.load(f)
            data["run_id"] = int(result_summary_json.split("/")[-2].split("_")[0])
            data |= metadata[data["run_id"]]
            df.append(data)
    df = pd.DataFrame(df).set_index("run_id").sort_index()
    df["diff"] = df["finetuned"] - df["baseline"]
    cprint(df.to_markdown(floatfmt=".2f"), TERMINAL_COLOR)
    df.to_csv(f"{output}/summary.csv")
    cprint(f"summary file is in {output}/summary.csv\n", TERMINAL_COLOR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "document",
        help="the file or directory containing the data to create the dataset",
    )
    parser.add_argument(
        "--model-cfg",
        default="config/trainer/Swallow-70b.yaml",
        help="config path for training (default is for Swallow-70b)",
    )
    parser.add_argument(
        "--evaluator",
        default="tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1",
        help="model name used as a model evaluator (default is Swallow-MX-8x7b-NVE-v0.1)",
    )
    parser.add_argument(
        "--promptfoo-generate-answer-template",
        default="templates/ja/promptfoo_default.jinja2",
        help="promptfoo prompt template file path for generating an answer (default is templates/ja/promptfoo_default.jinja2)",
    )
    parser.add_argument(
        "--baseline",
        default="tokyotech-llm/Swallow-70b-instruct-hf",
        help="model name used for comparison (default is tokyotech-llm/Swallow-70b-instruct-hf)",
    )
    parser.add_argument(
        "--num-repeat",
        type=int,
        default=10000,
        help="number of times to repeat training/evaluation (default is 10000)",
    )
    parser.add_argument("--resume-model", help="specify the model to resume execution.")
    parser.add_argument(
        "--resume-data", help="specify the dataset to resume execution."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = f"outputs/pipeline/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    best_model = args.resume_model or args.baseline
    metadata = []
    cprint("The pipeline started.", TERMINAL_COLOR)

    try:
        for run_id in range(args.num_repeat):
            data_files = (
                f"{output}/{run_id}_dataset_evaluator/dataset_clean_merged.jsonl"
            )
            meta = dict()
            meta["dataset_converter"] = run_dataset_converter(run_id, output, args)
            meta["dataset_evaluator"] = run_dataset_evaluator(run_id, output)
            meta["num_lines"] = run_dataset_merger(
                run_id, output, data_files, args.resume_data
            )
            meta["trainer"] = run_trainer(run_id, output, data_files, best_model, args)
            meta["model_evaluator"], new_model = run_model_evaluator(
                run_id, output, data_files, best_model, args
            )
            meta["model_selector"], best_model = run_model_selector(
                run_id, output, new_model, best_model
            )
            metadata.append(meta)
            summarize_time_and_accuracy(output, metadata)
    except KeyboardInterrupt:
        pass

    # summarize file info
    filesize = get_dir_size(output) / (1024**3)
    cprint(
        f"The output directory {output} contains {filesize:.2f} GB.\n"
        "Please delete unnecessary files to add space.\n",
        TERMINAL_COLOR,
    )

    # summarize model info
    cprint(f"best model: {best_model}\n", TERMINAL_COLOR)
    launcher_command = f"python src/run_launcher.py {best_model} {args.baseline}"
    cprint(f"To launch webapp, please execute `{launcher_command}`.\n", TERMINAL_COLOR)
    resume_command = f"python src/run_pipeline.py --resume-model {best_model} --resume-data {data_files}"
    cprint(
        f"To resume this pipeline, please execute `{resume_command} <kwargs>` "
        "instead of `python src/run_pipeline.py <kwargs>`.\n",
        TERMINAL_COLOR,
    )

    cprint("The pipeline completed.", TERMINAL_COLOR)


if __name__ == "__main__":
    main()
