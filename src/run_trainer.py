import argparse
import logging
import os
from datetime import datetime
from math import ceil

import deepspeed
import hydra
import torch
import torch.distributed as dist
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, SFTTrainer

from training.callbacks import PrinterCallback

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

MIN_LOGGING_TIMES = 10
MIN_SAVE_TIMES = 5
MIN_EVAL_TIMES = 5
TCPSTORE_HOST = "127.0.0.1"
TCPSTORE_PORT = 1234


def get_linear_layer_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(sorted(lora_module_names))


def adjust_steps(training_args, dataset):
    # decrease logging_steps, save_steps, eval_steps if data is small
    num_total_batch = (
        ceil(
            len(dataset)
            / training_args.per_device_train_batch_size
            / torch.cuda.device_count()
        )
        * training_args.num_train_epochs
    )
    decreased_logging_steps = ceil(num_total_batch / MIN_LOGGING_TIMES)
    if decreased_logging_steps < training_args.logging_steps:
        logger.warning(
            "logging_steps is too large. updated to %d", decreased_logging_steps
        )
        training_args.logging_steps = decreased_logging_steps

    decreased_save_steps = ceil(num_total_batch / MIN_SAVE_TIMES)
    if decreased_save_steps < training_args.save_steps:
        logger.warning("save_steps is too large. updated to %d", decreased_save_steps)
        training_args.save_steps = decreased_save_steps

    decreased_eval_steps = ceil(num_total_batch / MIN_EVAL_TIMES)
    if decreased_eval_steps < training_args.eval_steps:
        logger.warning("eval_steps is too large. updated to %d", decreased_eval_steps)
        training_args.eval_steps = decreased_eval_steps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_cfg")
    args, overrides = parser.parse_known_args()
    train_cfg = args.train_cfg
    if not os.path.exists(train_cfg):
        raise FileNotFoundError(train_cfg)
    overrides = [x for x in overrides if not x.startswith("--")]
    hydra.initialize(config_path=f"../{os.path.dirname(train_cfg)}", version_base=None)
    args = hydra.compose(config_name=os.path.basename(train_cfg), overrides=overrides)

    # set output dir
    if args.training_args.output_dir is None:
        world_size = int(os.environ["LOCAL_SIZE"])
        if os.environ["LOCAL_RANK"] == "0":
            store = dist.TCPStore(
                TCPSTORE_HOST, TCPSTORE_PORT, world_size, is_master=True
            )
            store.set(
                "output_dir",
                datetime.now().strftime(args.output_dir_template),
            )
        else:
            store = dist.TCPStore(
                TCPSTORE_HOST, TCPSTORE_PORT, world_size, is_master=False
            )
        args.training_args.output_dir = store.get("output_dir").decode()

    if os.path.exists(args.training_args.output_dir):
        raise FileExistsError(
            f"Output path {args.training_args.output_dir} already exists.\n",
            "Please set training_args.output_dir=<PATH_TO_NEW_OUTPUT_DIR>.",
        )

    # resume check
    if args.train.resume_from_checkpoint:
        if not os.path.exists(args.train.resume_from_checkpoint):
            raise FileNotFoundError(
                f"Resume file path {args.train.resume_from_checkpoint} not exists.\n"
                "Please set train.resume_from_checkpoint=<PATH_TO_CHECKPOINT_DIR>"
            )

    return args


def to_dict(conf):
    return OmegaConf.to_container(conf, resolve=True)


def main():
    args = parse_args()
    training_args = TrainingArguments(**to_dict(args.training_args))

    tokenizer = AutoTokenizer.from_pretrained(**to_dict(args.tokenizer))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        **to_dict(args.model), torch_dtype=torch.float16
    )
    if args.lora_config.target_modules == "all":
        args.lora_config.target_modules = get_linear_layer_names(model)
        logger.info(
            "set lora layer to all linear layers %s", args.lora_config.target_modules
        )

    model = model.to("cuda")
    model.config.use_cache = False
    model.enable_input_require_grads()

    if args.lora_config.target_modules is not None:
        peft_config = LoraConfig(**to_dict(args.lora_config))
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        callbacks = []
    else:
        callbacks = [PrinterCallback(model)]

    # set zero3 leaf modules. cf. https://github.com/microsoft/DeepSpeed/pull/4966
    if args.zero3_leaf_modules is not None:
        transformer_block = model.base_model.model.model.layers[0]
        leaf_module_classes = [
            getattr(transformer_block, name).__class__
            for name in dir(transformer_block)
            if getattr(transformer_block, name).__class__.__name__
            in set(args.zero3_leaf_modules)
        ]
        assert len(leaf_module_classes) > 0, args.zero3_leaf_modules
        logger.info("set zero3 leaf layer: %s", leaf_module_classes)
        deepspeed.utils.set_z3_leaf_modules(model, leaf_module_classes)

    model.eval()

    dataset = load_dataset(**to_dict(args.dataset))
    if args.prompt.template is not None:
        jinja_dir = os.path.join(os.path.dirname(__file__), "../templates")
        jinja_env = Environment(loader=FileSystemLoader(jinja_dir))
        template = jinja_env.get_template(args.prompt.template)

        if args.trainer.packing:

            def formatting_func(example: dict) -> str:
                # Setting "packing" on changes dataset to ConstantLengthDataset,
                # thus, example contains only one item and this should convert to single str
                return template.render(
                    question=example["question"], answer=example["answer"]
                )

        else:

            def formatting_func(examples: list[dict]) -> list[str]:
                # The defult collator (e.g. DataCollatorForLanguageModeling) passes
                # multiple items into this, then, this should be return multiple strs.
                questions = examples["question"]
                answeres = examples["answer"]
                return [
                    template.render(question=questions[i], answer=answeres[i])
                    for i in range(len(questions))
                ]

    else:
        formatting_func = None

    if args.mode == "sft":
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            **to_dict(args.trainer),
            formatting_func=formatting_func,
            args=training_args,
            callbacks=callbacks,
        )
    elif args.mode == "dpo":
        trainer = DPOTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            args=training_args,
        )
    else:
        raise ValueError(args.mode)

    adjust_steps(trainer.args, trainer.train_dataset)
    trainer.train(**to_dict(args.train))
    logger.info(f"output_dir: {args.training_args.output_dir}")


if __name__ == "__main__":
    main()
