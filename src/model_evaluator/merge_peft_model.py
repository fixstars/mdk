import argparse
import shutil
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def merge_peft_model(model_path: str, out_dir: Optional[str] = None):
    if out_dir is None:
        out_basename = Path(model_path).name + "-merged"
        out_dir = str(Path(model_path).parent / out_basename)

    peft_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    peft_model = PeftModel.from_pretrained(peft_model, model_path)
    merged_model = peft_model.merge_and_unload(progressbar=True)
    merged_model._hf_peft_config_loaded = False  # merged model is not peft model
    merged_model.save_pretrained(out_dir)
    for filename in [".model", ".json", "_config.json"]:
        filepath = Path(model_path) / f"tokenizer{filename}"
        if filepath.exists():
            shutil.copy(filepath, out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--out-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    merge_peft_model(args.model_path, args.out_dir)


if __name__ == "__main__":
    main()
