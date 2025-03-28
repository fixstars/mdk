import json
import logging
import os
import time

import pandas as pd

from preprocess.json_finder import calc_num_error, calc_score, raw_score
from preprocess.llm import load_llm
from preprocess.sft_convert import output_experiment_log
from preprocess.summerize import aggregate_filtered_qa, find_confusion_matrix

logger = logging.getLogger(__name__)


def attr_process_llm(args):
    outputss = []
    promptss = []
    model = load_llm(args)
    evaluate_begin = time.time()
    for topic in args.prompt.topics:
        model.args.prompt.topic = topic
        # sft_convert and dpo_convert should output in UTF-8 format
        with open(args.inputs.name, encoding="utf-8") as f:
            inputs = json.load(f)
        inputs_dict = []
        for item in inputs:
            for qa in item["qa"]:
                inputs_data = {
                    "page": item["page"],
                    "qa": qa,
                    "convert_template": item.get("convert_template", None),
                }
                if "num_error" in item:
                    inputs_data["previous_num_error"] = item["num_error"]
                inputs_dict.append(inputs_data)

        outputs, prompts = model(inputs_dict)

        outputss.append(outputs)
        promptss.append(prompts)

    outputs = [sum(outputs, []) for outputs in zip(*outputss)]
    prompts = [list(prompts) for prompts in zip(*promptss)]

    model.empty_cache()
    ret = get_ret(
        outputs,
        prompts,
        inputs_dict,
        evaluate_begin,
        args.prompt.template,
        args.llm.max_ok_error_threshold,
        args.llm.coefficient_yaml_path,
    )
    return ret


def process_llm(args):
    model = load_llm(args)
    with open(args.inputs.name) as f:
        inputs = json.load(f)
    inputs_dict = []
    evaluate_begin = time.time()
    for item in inputs:
        for qa in item["qa"]:
            inputs_data = {
                "page": item["page"],
                "qa": qa,
                "convert_template": item.get("convert_template", None),
            }
            if "num_error" in item:
                inputs_data["previous_num_error"] = item["num_error"]
            inputs_dict.append(inputs_data)

    outputs, prompts = model(inputs_dict)

    model.empty_cache()
    ret = get_ret(
        outputs,
        prompts,
        inputs_dict,
        evaluate_begin,
        args.prompt.template,
        args.llm.max_ok_error_threshold,
        args.llm.coefficient_yaml_path,
    )
    return ret


def get_ret(
    outputs,
    prompts,
    inputs_dict,
    evaluate_begin,
    template,
    max_ok_error_threshold,
    coefficient_yaml_path,
):
    ret = [
        dict(
            convert_template=_data.get("convert_template"),
            evaluate_template=template,
            prompt=prompt,
            output=output,
            page=_data["page"],
            qa=[_data["qa"]],
            raw_score=raw_score(output, coefficient_yaml_path),
            score=calc_score(output, coefficient_yaml_path),
            num_error=calc_num_error(
                output, max_ok_error_threshold, coefficient_yaml_path
            ),
            previous_num_error=_data.get("previous_num_error"),
        )
        for prompt, output, _data in zip(prompts, outputs, inputs_dict)
    ]
    evaluate_end = time.time()

    logger.info(
        "evaluate_dataset: %f sec / %d QAs = %f sec/QA",
        evaluate_end - evaluate_begin,
        len(inputs_dict),
        (evaluate_end - evaluate_begin) / len(inputs_dict),
    )

    return ret


def output_filtered_qa(eval_result, args):
    qa_list, counter = aggregate_filtered_qa(eval_result)

    for judge in ["clean", "wrong", "invalid"]:
        logger.info(
            "%s QA: %d / %d (%f %%)",
            judge,
            counter[judge],
            len(eval_result),
            counter[judge] / len(eval_result) * 100,
        )

        out_filename = os.path.join(
            args.output.basedir,
            args.output.dataset.replace(".jsonl", f"_{judge}.jsonl"),
        )
        assert not os.path.exists(out_filename)
        with open(out_filename, "w", encoding="utf-8") as f:
            for qa in qa_list[judge]:
                json.dump(qa, f, ensure_ascii=False)
                f.write("\n")
        logger.info(f"output_filtered_qa ({judge}): {out_filename}")


def output_confusion_matrix(eval_result, args):
    if args.output.confusion_matrix is None:
        return

    counter = find_confusion_matrix(eval_result)
    df = pd.DataFrame(counter)
    logger.info("\n" + df.to_markdown())

    out_filename = os.path.join(
        args.output.basedir,
        args.output.confusion_matrix,
    )
    assert not os.path.exists(out_filename)
    df.to_csv(out_filename)
    logger.info(f"output_confusion_matrix: {out_filename}")


def evaluate(args):
    eval_result = (
        attr_process_llm(args) if "topics" in args.prompt else process_llm(args)
    )
    output_filtered_qa(eval_result, args)
    output_experiment_log(eval_result, args)
    output_confusion_matrix(eval_result, args)
