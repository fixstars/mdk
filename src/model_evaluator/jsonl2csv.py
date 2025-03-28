import argparse
import os
from typing import Optional

import pandas as pd


def jsonl2csv(input_jsonl: str, output_csv: Optional[str] = None):
    output_csv = output_csv or input_jsonl.replace(".jsonl", ".csv")
    if os.path.exists(output_csv):
        if input(f"{output_csv} exists. continue? [y/N] ") not in ["y", "Y"]:
            raise FileExistsError(output_csv)
    df = pd.read_json(input_jsonl, orient="records", lines=True)
    df["__expected"] = "grade:" + df["answer"]
    df = df.drop("answer", axis=1)
    df.to_csv(output_csv, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("--output-csv")
    return parser.parse_args()


def main():
    args = parse_args()
    jsonl2csv(args.input_jsonl, args.output_csv)


if __name__ == "__main__":
    main()
