import glob
import json
import logging
import os
import time

from preprocess.json_finder import find_qa
from preprocess.llm import load_llm
from preprocess.pandoc_splitter import PANDOC_SUPPORTED_EXT, split_with_pandoc
from preprocess.pdf_splitter import split_by_pdf_page
from preprocess.text_splitter import split_by_length, split_by_markdown_header
from preprocess.vision_llm import (
    VISION_SUPPORTED_EXT,
    explain_image,
    release_vision_llm,
)

logger = logging.getLogger(__name__)


def split_inputs(args):
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

    # splitting files
    splitted_text = []
    for input_file in input_files:
        if input_file.endswith(".pdf"):
            splitted_text += split_by_pdf_page(input_file, args)
        elif input_file.endswith(".md"):
            splitted_text += split_by_markdown_header(input_file, **args.splitter)
        elif input_file.endswith(".txt"):
            splitted_text += split_by_length(input_file, **args.splitter)
        elif input_file.endswith(PANDOC_SUPPORTED_EXT):
            splitted_text += split_with_pandoc(input_file, args)
        elif input_file.lower().endswith(VISION_SUPPORTED_EXT):
            splitted_text += explain_image(input_file, args)
        else:
            logger.warning(
                "An unknown extension has detected. Skip this file: %s", input_file
            )

    release_vision_llm()
    assert len(splitted_text) > 0, "no text found in the input files/directories."

    splitted_text_filename = os.path.join(
        args.output.basedir, args.output.splitted_text
    )
    assert not os.path.exists(splitted_text_filename), splitted_text_filename
    with open(splitted_text_filename, "w", encoding="utf-8") as f:
        f.writelines([s.replace("\n", "\\n") + "\n" for s in splitted_text])

    return splitted_text


def process_llm(inputs: list[str], args) -> list[dict[str, str]]:
    model = load_llm(args)
    inputs_dict = [{"page": page} for page in inputs]
    convert_begin = time.time()
    outputs, prompts = model(inputs_dict)
    ret = [
        dict(
            convert_template=args.prompt.template,
            prompt=prompt,
            output=output,
            page=page,
            qa=sum([find_qa(s) for s in output], []),
        )
        for prompt, output, page in zip(prompts, outputs, inputs_dict)
    ]
    convert_end = time.time()
    model.empty_cache()

    logger.info(
        "convert: %f sec / %d pages = %f sec/page",
        convert_end - convert_begin,
        len(inputs),
        (convert_end - convert_begin) / len(inputs),
    )
    return ret


def output_dataset(data: list[dict[str, str]], args):
    out_filename = os.path.join(args.output.basedir, args.output.dataset)
    assert not os.path.exists(out_filename)
    if "qa" in args.prompt.template and "json" in args.prompt.template:
        with open(out_filename, "w", encoding="utf-8") as f:
            for item in data:
                for qa in item["qa"]:
                    json.dump(qa, f, ensure_ascii=False)
                    f.write("\n")
    else:
        logger.info("skip postprocessing for QA json")
        out_filename = out_filename.replace(".jsonl", ".txt")
        with open(out_filename, "w", encoding="utf-8") as f:
            for item in data:
                print(item["output"], file=f)
    logger.info(f"output_dataset: {out_filename}")


def output_experiment_log(data, args):
    if args.output.experiment_log is None:
        return
    out_filename = os.path.join(args.output.basedir, args.output.experiment_log)
    assert not os.path.exists(out_filename)
    with open(out_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f"output_experimental_log: {out_filename}")


def sft_convert(args):
    splitted_text = split_inputs(args)
    converted_data = process_llm(splitted_text, args)
    output_dataset(converted_data, args)
    output_experiment_log(converted_data, args)
