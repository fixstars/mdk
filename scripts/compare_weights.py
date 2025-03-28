import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("base", type=Path, help="Base model")
    parser.add_argument("target", type=Path, help="Target model")
    parser.add_argument("--device", default="cuda", help="Device used for comparison")
    parser.add_argument("--base-prefix", default="", help="Prefix to add to base model")
    parser.add_argument(
        "--target-prefix", default="", help="Prefix to add to target model"
    )
    # atol, rtolのデフォルト値はtorch.allcloseのものを利用
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for value comparisons",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for value comparisons",
    )
    return parser.parse_args()


def load_safetensors(path: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device=device) as f:
        return {key: f.get_tensor(key) for key in f.keys()}


def load_torch_model(path: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    return torch.load(path, map_location=device, mmap=True, weights_only=True)


def load_model(checkpoint_dir: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    # huggingface の use_safetensors フラグと同様に、 bin ファイルより safetensors ファイルの読み込みを優先する
    # cf. https://huggingface.co/docs/transformers/ja/main_classes/model#transformers.PreTrainedModel.from_pretrained
    loader_map = (
        ("safetensors", load_safetensors),
        ("bin", load_torch_model),
    )

    state_dict = {}
    for ext, loader in loader_map:
        paths = list(checkpoint_dir.glob(f"**/*.{ext}"))
        if paths:
            for path in tqdm(paths, desc=f"Load {checkpoint_dir.name}"):
                state_dict.update(loader(path, device))
            break
    else:
        raise RuntimeError(f"Checkpoint not found ({checkpoint_dir})")

    return state_dict


def main():
    args = parse_args()

    base_prefix: str = args.base_prefix
    if base_prefix != "" and not base_prefix.endswith("."):
        base_prefix += "."

    target_prefix: str = args.target_prefix
    if target_prefix != "" and not target_prefix.endswith("."):
        target_prefix += "."

    # メモリ節約のためにロード自体は常にCPUで実行
    base_dict = load_model(args.base)
    target_dict = load_model(args.target)

    base_keys = set(base_prefix + key for key in base_dict.keys())
    target_keys = set(target_prefix + key for key in target_dict.keys())

    base_only_keys = sorted(list(base_keys - target_keys))
    target_only_keys = sorted(list(target_keys - base_keys))
    common_keys = sorted(list(base_keys & target_keys))

    shape_mismatch_keys = []
    value_mismatch_keys = []

    for key in tqdm(common_keys, desc="Compare"):
        base = base_dict[key.removeprefix(base_prefix)]
        target = target_dict[key.removeprefix(target_prefix)]

        if base.shape != target.shape:
            shape_mismatch_keys.append(key)
            continue

        # CPU側で型変換すると遅いので、先に転送してから型変換する
        if not torch.allclose(
            base.to(args.device).to(torch.float32),
            target.to(args.device).to(torch.float32),
            atol=args.atol,
            rtol=args.rtol,
        ):
            value_mismatch_keys.append(key)
            continue

    results = (
        ("Tensors only in the base model", base_only_keys),
        ("Tensors only in the target model", target_only_keys),
        ("Shape mismatched tensors", shape_mismatch_keys),
        ("Value mismatched tensors", value_mismatch_keys),
    )

    for name, keys in results:
        print(name)
        for key in keys:
            print(f"  - {key}")
        if not keys:
            print("  Nothing")

    print()
    print(f"Total tensors: {len(base_keys | target_keys)}")

    for name, keys in results:
        print(f"  {name}: {len(keys)}")


if __name__ == "__main__":
    main()
