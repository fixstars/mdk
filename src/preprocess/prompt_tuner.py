import json
import logging
import os
import random
import time

from preprocess.llm import load_llm
from preprocess.sft_convert import output_experiment_log

logger = logging.getLogger(__name__)


def get_better_prompt(args):
    with open(args.inputs.name) as f:
        inputs = json.load(f)
    model = load_llm(args)
    qa = ""
    for input in random.choices(inputs, k=4):
        with open(f'templates/{input["convert_template"]}') as f:
            template = f.read()
        qa += str(input["qa"][0]) + "\n"

    inputs_dict = [{"template": template, "qa": qa}]
    begin = time.time()
    outputs, prompts = model(inputs_dict)
    end = time.time()

    ret = [
        dict(
            prompt=prompt,
            output=output,
            qa=[_data["qa"]],
        )
        for prompt, output, _data in zip(prompts, outputs, inputs_dict)
    ]

    model.empty_cache()

    logger.info("prompt_tuner: %f sec", end - begin)

    return ret


def output_jinja2(result, args):
    for i, data in enumerate(result[0]["output"]):
        out_filename = os.path.join(
            args.output.basedir,
            args.output.dataset,
        ).replace(".jinja2", f"_{i}.jinja2")
        assert not os.path.exists(out_filename), out_filename
        replaced = data.replace("$DOCUMENT", "{{page}}")
        splitted = replaced.split("```")
        if len(splitted) > 1:
            with open(out_filename, "w") as f:
                print(splitted[1], file=f)
        logger.info(f"output_jinja2: {out_filename}")


def prompt_generate(args):
    result = get_better_prompt(args)
    output_jinja2(result, args)
    output_experiment_log(result, args)
