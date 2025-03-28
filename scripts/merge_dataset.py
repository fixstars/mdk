import argparse
import glob
import json
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_paths", nargs="+")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--output-basename")
    return parser.parse_args()


def to_evaluator_format(dataset):
    """same function as run_manual_dataset_evaluator.py"""
    ret = []
    for data in dataset:
        for qa in data["qa"]:
            ret.append(
                dict(
                    qa=[qa],
                    page=data["page"],
                    num_error=data.get("num_error"),
                    output=data["output"],
                )
            )
    return ret


def get_json_files_list(target_path):
    target_files = []
    dirs = os.listdir(target_path)
    for dir_name in sorted(dirs):
        if not os.path.isdir(os.path.join(target_path, dir_name)):
            continue
        file_path = os.path.join(target_path, dir_name, "experiment_log_manual_*.json")
        experiment_logs = glob.glob(file_path)
        if len(experiment_logs) == 0:
            continue
        last_file_path = experiment_logs[-1]
        target_files.append(last_file_path)
    return target_files


def main():
    args = parse_args()

    total = []
    len_good = 0
    len_bad = 0
    print("|dataset|good|bad|")
    print("|---|---|---|")

    target_files = []
    for target_path in args.target_paths:
        if os.path.isfile(target_path):
            target_files.append(target_path)
        else:
            target_files += get_json_files_list(target_path)

    for target_file in target_files:
        with open(target_file) as f:
            dataset = json.load(f)

        dataset = to_evaluator_format(dataset)
        for d in dataset:
            d["output"] = (
                d["output"] if type(d["output"]) is list else eval(d["output"])
            )

        dataset_good = [d for d in dataset if d["num_error"] == 0]
        dataset_bad = [d for d in dataset if d["num_error"] == 1]
        total = total + dataset_good + dataset_bad
        tmp_len_good = len(dataset_good)
        tmp_len_bad = len(dataset_bad)
        len_good += tmp_len_good
        len_bad += tmp_len_bad
        print(f"|{target_file}|{tmp_len_good}|{tmp_len_bad}|")

    print(f"\ngood:{len_good}, bad:{len_bad}, total:{len_good + len_bad}")

    df = pd.DataFrame(total)
    df["checked"] = 1
    df["reason"] = "['手動アノテーション済']"
    df["question"] = df["qa"].str[0].str["question"]
    df["answer"] = df["qa"].str[0].str["answer"]
    df = df.drop("qa", axis=1)
    df = df.drop("output", axis=1)

    # Save Files
    os.makedirs(args.output_path)
    basename = "" if args.output_basename is None else f"_{args.output_basename}"

    out_df = df[["page", "question", "answer", "num_error", "reason", "checked"]]
    out_df.to_csv(
        os.path.join(args.output_path, f"dataset{basename}_checked.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    out_df.to_excel(
        os.path.join(args.output_path, f"dataset{basename}_checked.xlsx"), index=False
    )

    with open(
        os.path.join(args.output_path, f"dataset{basename}_checked.json"), "w"
    ) as f:
        json.dump(total, f, indent=4, ensure_ascii=False)

    for d in total:
        d["prompt"] = ""
        del d["num_error"]

    with open(os.path.join(args.output_path, f"dataset{basename}.json"), "w") as f:
        json.dump(total, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
