import argparse
import json
import os
import sqlite3
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ids", nargs="+")
    parser.add_argument("--write-summary", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    for input_id in args.input_ids:
        print(input_id)
        if input_id.endswith(".json"):
            with open(input_id, "r") as f:
                data = json.load(f)["results"]
            out_json = input_id.replace(".json", "_summary.json")
        else:
            conn = sqlite3.connect(f"{os.environ['HOME']}/.promptfoo/promptfoo.db")
            cursor = conn.cursor()
            table_list_query = f"SELECT results FROM evals WHERE id = '{input_id}';"
            cursor.execute(table_list_query)
            result = cursor.fetchall()
            conn.close()
            data = json.loads(result[0][0])
            out_json = f"{input_id}_summary.json"
        assert data["version"] == 2
        scores = defaultdict(list)
        for result in data["results"]:
            scores[result["provider"]["id"]].append(result["score"])

        summary = dict()
        for model_name, score in scores.items():
            avg_score = sum(score) / len(score) * 100
            summary[model_name] = avg_score
            print(f"{model_name} : {avg_score:.04f} %")

        if args.write_summary:
            with open(out_json, "w") as f:
                json.dump(summary, f)


if __name__ == "__main__":
    main()
