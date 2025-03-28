import argparse
import json
import re

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

NUM_ATTRS = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_json_path")
    parser.add_argument("gt_xlsx_path")
    parser.add_argument("output_yaml_path")
    parser.add_argument("--seed", type=int, default=515)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.base_json_path, "r") as f:
        data = json.load(f)

    outputs = [item["output"] for item in data]
    num_repeat = len(outputs[0]) // NUM_ATTRS

    x_list = [[] for _ in range(num_repeat)]
    for output in outputs:
        for i, reply in enumerate(output):
            a = list(map(int, re.findall(r"\d+", reply.split(" [/ATTR_")[0])))
            if i // num_repeat == 0:
                x_list[i % num_repeat].append(a)
            else:
                x_list[i % num_repeat][-1] += a

    X = [i for x in x_list for i in x]

    df = pd.read_excel(args.gt_xlsx_path)
    y = df["num_error"].values.tolist()
    y = y * num_repeat

    # karakuri_apm rarely returns incorrect data, such as eight columns.
    # mark as 'ignore' to exclude for manual review later.
    X_filtered = [x for x in X if len(x) == 9]
    y_filtered = [y[i] for i, x in enumerate(X) if len(x) == 9]

    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered,
        y_filtered,
        test_size=0.2,
        stratify=y_filtered,
        random_state=args.seed,
    )

    log_reg = LogisticRegression(random_state=args.seed)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rs = recall_score(y_test, y_pred, average=None)
    ps = precision_score(y_test, y_pred, average=None)
    fs = f1_score(y_test, y_pred, average=None)

    coef = {
        "slope": [float(c) for c in log_reg.coef_[0]],
        "intercept": float(log_reg.intercept_[0]),
    }

    with open(args.output_yaml_path, "w") as f:
        yaml.safe_dump(coef, f)

    print(f"\ndump coefficient to {args.output_yaml_path}")
    print(
        f"LogisticRegression() Accuracy: {acc:06f} Recall: {rs} Presicion: {ps} F-measure: {fs}\n"
    )


if __name__ == "__main__":
    main()
