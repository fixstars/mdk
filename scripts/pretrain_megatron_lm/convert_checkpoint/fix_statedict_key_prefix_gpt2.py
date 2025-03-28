import argparse
import glob
import os
import shutil
from typing import Any

import torch
from safetensors import safe_open


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_checkpoint",
        type=str,
        help="Path to the checkpoint dir",
    )

    parser.add_argument(
        "path_to_output",
        type=str,
        help="Path to the save dir ",
    )
    parser.add_argument(
        "-p", "--prefix", default="transformer", help="Prefix for key of state_dict"
    )

    return parser.parse_args()


def collect_file_paths(dir_name: str, extension: str) -> list[str]:
    return glob.glob(os.path.join(dir_name, f"**/*{extension}"), recursive=True)


def load_model(checkpoint_dir: str) -> dict[str, Any]:
    # huggingface の use_safetensors フラグと同様に、 bin ファイルより safetensors ファイルの読み込みを優先する
    # cf. https://huggingface.co/docs/transformers/ja/main_classes/model#transformers.PreTrainedModel.from_pretrained
    state_dict = {}
    if safetensors_paths := collect_file_paths(checkpoint_dir, ".safetensors"):
        for path_ in safetensors_paths:
            with safe_open(path_, framework="pt", device=0) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

    elif bin_paths := collect_file_paths(checkpoint_dir, ".bin"):
        for path_ in bin_paths:
            state_dict_temp = torch.load(path_, weights_only=True, map_location="cuda")
            state_dict = state_dict | state_dict_temp

    else:
        raise FileNotFoundError("model file not found.")
    return state_dict


def main():
    args = parse_args()

    os.makedirs(args.path_to_output, exist_ok=True)
    input_state_dict = load_model(args.path_to_checkpoint)
    output_state_dict = {}

    # Fix prefix
    for key in input_state_dict.keys():
        if not key.startswith(args.prefix):
            output_state_dict[f"{args.prefix}.{key}"] = input_state_dict[key]

    # Copy config.json
    config_file_path = f"{args.path_to_checkpoint}/config.json"
    if os.path.exists(config_file_path):
        shutil.copy(config_file_path, args.path_to_output)

    # HuggingFace形式のチェックポイントをMegatron-LM形式に変換するスクリプトは拡張子".bin"にのみ対応
    # cf. https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/megatron_gpt2/checkpoint_reshaping_and_interoperability.py#L617
    output_checkpoint_file = os.path.join(args.path_to_output, "pytorch_model.bin")
    torch.save(output_state_dict, output_checkpoint_file)


if __name__ == "__main__":
    main()
