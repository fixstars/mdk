import argparse
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_score, recall_score


def drawGraph(human_df, csv_file, out_file_name, min_x, max_x, step):
    data = pd.read_csv(f"{out_file_name}.csv")
    _, ax = plt.subplots()
    ax.step(data["x"], data["recall"], label="Recall", where="mid")
    ax.step(
        data["x"], data["inversed-precision"], label="Inversed Precision", where="mid"
    )
    ax.step(data["x"], data["corr"], label="Correlation", where="mid")
    ax.step(data["x"], data["positive_ratio"], label="Positive Ratio", where="mid")

    # Baseline
    if csv_file:
        original_df = pd.read_csv(csv_file)
        if "num_error" in original_df:
            original_df["num_error"] = original_df["num_error"].apply(
                lambda x: 1 if x != 0 else x
            )
            (b_r, b_i, b_c, b_p) = classify(human_df, original_df)
            b_x = None
        else:
            max_corr_row = original_df.loc[original_df["corr"].idxmax()]
            (b_x, b_r, b_i, b_c, b_p) = max_corr_row
        ax.plot([min_x, max_x], [b_r, b_r], "tab:blue", linestyle="dotted")
        ax.plot([min_x, max_x], [b_i, b_i], "tab:orange", linestyle="dotted")
        ax.plot([min_x, max_x], [b_c, b_c], "tab:green", linestyle="dotted")
        ax.plot([min_x, max_x], [b_p, b_p], "tab:red", linestyle="dotted")

    ax.set_xlabel("X")
    ax.xaxis.set_ticks(np.arange(min_x, max_x + step, ((max_x - min_x) / 10)))
    plt.ylim(0, 1.0)
    ax.set_ylabel("Y")
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))

    baseline = (
        ""
        if csv_file is None
        else f"(Dotted:Baseline {os.path.basename(csv_file)}{'' if b_x is None else ' X=' + str(int(b_x))})"
    )

    ax.set_title(f"{os.path.basename(out_file_name)}.csv {baseline}")
    ax.grid(True, linestyle="-", color="0.85")

    plt.legend()
    plt.show()
    plt.savefig(f"{out_file_name}.png")


def classify(human_df, auto_df):
    merged_df = pd.merge(human_df, auto_df, on=["answer", "question"])
    comparison_df = merged_df[["answer", "question", "num_error_x", "num_error_y"]]
    comparison_df.columns = ["answer", "question", "num_error_human", "num_error_ai"]

    recall = recall_score(
        comparison_df["num_error_human"], comparison_df["num_error_ai"]
    )
    inversed_precision = precision_score(
        comparison_df["num_error_human"], comparison_df["num_error_ai"], pos_label=0
    )
    with np.errstate(invalid="ignore"):
        correlation_coefficient = comparison_df["num_error_human"].corr(
            comparison_df["num_error_ai"]
        )
    positive_ratio = (comparison_df["num_error_ai"] == 0).sum() / len(comparison_df)
    return (recall, inversed_precision, correlation_coefficient, positive_ratio)


def write_line_to_csv(line, file_path):
    print(line)
    with open(file_path, "a") as f:
        f.write(f"{line}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs_dir")
    parser.add_argument(
        "--human-eval-excel-data", default="experimental_log_checked.xlsx"
    )
    parser.add_argument("--auto-eval-json-file", default="experiment_log.json")
    parser.add_argument("--baseline-csv-file")
    parser.add_argument("--out-csv-png-name", default="plot")
    parser.add_argument("--min-x", type=float, default=-0.5)
    parser.add_argument("--max-x", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # Settings
    human_eval_excel_file = f"{args.outputs_dir}{args.human_eval_excel_data}"
    auto_eval_json_file = (
        args.auto_eval_json_file
        if "/" in args.auto_eval_json_file
        else f"{args.outputs_dir}{args.auto_eval_json_file}"
    )
    out_file_name = f"{args.outputs_dir}{args.out_csv_png_name}"

    human_df = pd.read_excel(human_eval_excel_file)
    human_df = human_df[human_df["checked"] == 1]
    human_df["num_error"] = human_df["num_error"].apply(lambda x: 1 if x != 0 else x)

    with open(auto_eval_json_file) as f:
        json_data = json.load(f)
    for item in json_data:
        item["question"] = item["qa"][0]["question"]
        item["answer"] = item["qa"][0]["answer"]

    if os.path.exists(f"{out_file_name}.csv"):
        os.remove(f"{out_file_name}.csv")

    header = "x,recall,inversed-precision,corr,positive_ratio"
    write_line_to_csv(header, f"{out_file_name}.csv")

    th_list = [
        args.min_x + (x * args.step)
        for x in range(0, int((args.max_x - args.min_x) / args.step))
    ] + [args.max_x]

    for th in th_list:
        auto_df = pd.DataFrame(json_data)
        auto_df["num_error"] = auto_df["score"].apply(
            lambda score: 0 if score < th else 1
        )
        auto_df = auto_df[["question", "answer", "num_error"]]
        (r, i, c, p) = classify(human_df, auto_df)
        line = f"{th:.03f},{r:.03f},{i:.03f},{c:.03f},{p:.03f}"
        write_line_to_csv(line, f"{out_file_name}.csv")

    csv_file = (
        f"{args.outputs_dir}{args.baseline_csv_file}"
        if args.baseline_csv_file
        else None
    )
    drawGraph(human_df, csv_file, out_file_name, args.min_x, args.max_x, args.step)


if __name__ == "__main__":
    main()
