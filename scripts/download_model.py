import argparse
import logging
import os

import transformers

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="+",
        help="A list of models you want to download or check layer information for.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    assert "HF_HOME" in os.environ, '"export HF_HOME=<PATH/TO/CACHE/DIR>" is needed.'
    # モデルダウンロード時のdevice_mapをcpu + disk構成とするためにGPUを認識できなくする。
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for model in args.models:
        cfg = transformers.AutoConfig.from_pretrained(model)
        arch = cfg.architectures[0]
        logging.info("model: %s (%s)", model, arch)
        model = getattr(transformers, arch).from_pretrained(model, device_map="auto")
    logger.info("completed processing %d models", len(args.models))


if __name__ == "__main__":
    main()
