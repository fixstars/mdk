import glob
import json
import logging
import os
import time

import yaml

from preprocess.json_finder import find_dpo
from preprocess.llm import load_llm

logger = logging.getLogger(__name__)


def input_sft_dataset(args) -> list[dict[str, str]]:
    inputs = args.inputs.name
    assert os.path.exists(inputs), inputs

    # listing files
    if os.path.isdir(inputs):
        input_files = [
            p
            for p in glob.glob(os.path.join(inputs, "./**"), recursive=True)
            if os.path.isfile(p)
        ]
        assert len(input_files) > 0, f"no files exists in {inputs}"
    else:
        input_files = [inputs]

    data = []
    for input_file in input_files:
        with open(input_file) as f:
            data += yaml.safe_load(f)

    return data


def process_llm(inputs: list[dict[str, str]], args) -> list[dict[str, str]]:
    model = load_llm(args)
    contexts = []
    for item in inputs:
        contexts += [{"page": item["page"]["page"], "qa": qa} for qa in item["qa"]]
    convert_begin = time.time()
    outputs, prompts = model(contexts)
    ret = [
        dict(
            prompt=prompt,
            output=output,
            context=context,
            dpo=sum([find_dpo(s) for s in output], []),
        )
        for prompt, output, context in zip(prompts, outputs, contexts)
    ]
    convert_end = time.time()
    model.empty_cache()

    logger.info(
        "dpo_convert: %f sec / %d pages = %f sec/page",
        convert_end - convert_begin,
        len(inputs),
        (convert_end - convert_begin) / len(inputs),
    )
    return ret


def output_dataset(data: list[dict[str, str]], args):
    out_filename = os.path.join(args.output.basedir, args.output.dataset)
    assert not os.path.exists(out_filename)

    with open(out_filename, "w", encoding="utf-8") as f:
        for item in data:
            for dpo in item["dpo"]:
                json.dump(
                    dict(
                        prompt=item["context"]["qa"]["question"],
                        chosen=dpo["chosen"],
                        rejected=dpo["rejected"],
                    ),
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
    logger.info(f"output_dataset: {out_filename}")


def output_experiment_log(data, args):
    if args.output.experiment_log is None:
        return
    out_filename = os.path.join(args.output.basedir, args.output.experiment_log)
    assert not os.path.exists(out_filename)
    with open(out_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f"output_experimental_log: {out_filename}")


def dpo_convert(args):
    sft_experiment_log = input_sft_dataset(args)
    converted_data = process_llm(sft_experiment_log, args)
    output_dataset(converted_data, args)
    output_experiment_log(converted_data, args)
