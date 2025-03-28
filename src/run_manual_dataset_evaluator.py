import argparse
import glob
import json
import os
from datetime import datetime

import gradio as gr

shortcut_js = """
<script>
function shortcuts(e) {
    var event = document.all ? window.event : e;
    switch (e.target.tagName.toLowerCase()) {
        case "input":
        case "textarea":
        case "select":
        case "button":
        break;
        default:
        if (e.key == "ArrowRight") {
            document.getElementById("next_btn").click();
        }
        else if (e.key == "ArrowLeft") {
            document.getElementById("prev_btn").click();
        }
        else if (e.key == "g") {
            document.getElementById("good_btn").click();
        }
        else if (e.key == "b") {
            document.getElementById("bad_btn").click();
        }
        else if (e.key == "u") {
            document.getElementById("undefined_btn").click();
        }
    }
}
document.addEventListener('keyup', shortcuts, false);
</script>
"""

dataset = None


def save_dataset(filename):
    assert dataset is not None
    with open(filename, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


def save_qa_dataset(filename: str):
    output = {"clean": [], "wrong": [], "invalid": []}
    for data in dataset:
        if data["num_error"] == 0:
            output["clean"].append(data["qa"][0])
        elif isinstance(data["num_error"], int) and data["num_error"] > 0:
            output["wrong"].append(data["qa"][0])
        else:
            output["invalid"].append(data["qa"][0])
    for judge, output_jsonl in output.items():
        with open(filename.replace("_clean.jsonl", f"_{judge}.jsonl"), "w") as f:
            f.writelines(
                [json.dumps(line, ensure_ascii=False) + "\n" for line in output_jsonl]
            )


def manual_save_dataset(input_json):
    output_dir = os.path.dirname(input_json)
    output_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    evaluator_dataset = f"{output_dir}/experiment_log_manual_{output_name}.json"
    trainer_dataset = f"{output_dir}/dataset_manual_{output_name}_clean.jsonl"
    save_dataset(evaluator_dataset)
    save_qa_dataset(trainer_dataset)
    return f"save dataset\nfor dataset_evaluator to {evaluator_dataset},\nfor trainer to {trainer_dataset}"


def autosave_dataset(input_json):
    output_dir = os.path.dirname(input_json)
    output = f"{output_dir}/experiment_log_autosave.json"
    save_dataset(output)
    return f"autosave dataset to {output}"


def is_valid_position(position: int) -> bool:
    if type(position) is not int:
        return False

    try:
        index = position
        if not 0 <= index < len(dataset):
            raise ValueError(index)
        return True
    except ValueError:
        return False


def move_position(position: int, diff: int):
    """Preventing move with incorrect string in `position`"""
    if position is None:
        return 0
    try:
        return (position + diff) % len(dataset)
    except ValueError:
        return 0


def save_data(position, question, answer, context, num_error, output):
    if not is_valid_position(position):
        return
    dataset[position]["qa"][0]["question"] = question
    dataset[position]["qa"][0]["answer"] = answer
    dataset[position]["page"]["page"] = context
    try:
        dataset[position]["num_error"] = None if num_error is None else int(num_error)
    except ValueError:
        dataset[position]["num_error"] = None
    dataset[position]["output"] = output


def load_data(
    position,
    question,
    answer,
    context,
    num_error,
    output,
    count_good=0,
    count_undefined=0,
    count_bad=0,
):
    if not is_valid_position(position):
        return (
            position,
            question,
            answer,
            context,
            num_error,
            output,
            f"invalid position: {position}",
            count_good,
            count_undefined,
            count_bad,
        )
    data = dataset[position]
    question = data["qa"][0]["question"]
    answer = data["qa"][0]["answer"]
    context = data["page"]["page"]
    num_error = data["num_error"]
    output = data["output"]
    count_good = sum(1 for item in dataset if item["num_error"] == 0)
    count_undefined = sum(1 for item in dataset if item["num_error"] is None)
    count_bad = sum(
        1 for item in dataset if item["num_error"] is not None and 0 < item["num_error"]
    )
    return (
        position,
        question,
        answer,
        context,
        num_error,
        output,
        "",
        count_good,
        count_undefined,
        count_bad,
    )


def click_next(
    position,
    question,
    answer,
    context,
    num_error,
    output,
    _,
    __,
    ___,
):
    save_data(position, question, answer, context, num_error, output)
    position = move_position(position, 1)
    return load_data(position, question, answer, context, num_error, output)


def click_prev(
    position,
    question,
    answer,
    context,
    num_error,
    output,
    _,
    __,
    ___,
):
    save_data(position, question, answer, context, num_error, output)
    position = move_position(position, -1)
    return load_data(position, question, answer, context, num_error, output)


def click_good(
    position,
    question,
    answer,
    context,
    _,
    output,
    count_good,
    count_undefined,
    count_bad,
):
    return click_next(
        position,
        question,
        answer,
        context,
        0,
        output,
        count_good,
        count_undefined,
        count_bad,
    )


def click_undefined(
    position,
    question,
    answer,
    context,
    _,
    output,
    count_good,
    count_undefined,
    count_bad,
):
    return click_next(
        position,
        question,
        answer,
        context,
        None,
        output,
        count_good,
        count_undefined,
        count_bad,
    )


def click_bad(
    position,
    question,
    answer,
    context,
    _,
    output,
    count_good,
    count_undefined,
    count_bad,
):
    return click_next(
        position,
        question,
        answer,
        context,
        1,
        output,
        count_good,
        count_undefined,
        count_bad,
    )


def to_evaluator_format(dataset):
    """dataset_converter generates multiple questions for one data,
    so evaluate the questions one by one."""
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


def get_latest_log(input_json):
    output_dir = os.path.dirname(input_json)
    latest_log = f"{output_dir}/experiment_log_autosave.json"

    if not os.path.exists(latest_log):
        latest_log = input_json

    manual_logs = glob.glob(f"{output_dir}/experiment_log_manual_*.json")
    if 0 < len(manual_logs):
        latest_log = sorted(manual_logs)[-1]

    print(f"{latest_log} loaded.")
    return latest_log


def launch_demo(args):
    global dataset
    latest_log = get_latest_log(args.input_json)
    with open(latest_log) as f:
        dataset = json.load(f)
    dataset = to_evaluator_format(dataset)
    assert len(dataset) > 0
    last_eval_id = next(
        (
            i
            for i in range(len(dataset) - 1, -1, -1)
            if dataset[i]["num_error"] is not None
        ),
        0,
    )
    with gr.Blocks(head=shortcut_js) as demo:
        init_data = dataset[last_eval_id]
        with gr.Row():
            with gr.Row():
                position = gr.Number(last_eval_id, label="ID", scale=1, precision=0)
                gr.Label(str(len(dataset)), label="total")
            with gr.Column(scale=7):
                question = gr.Textbox(init_data["qa"][0]["question"], label="question")
                answer = gr.Textbox(
                    init_data["qa"][0]["answer"], label="answer", lines=3
                )
        with gr.Row():
            with gr.Column(scale=1, min_width=50):
                gr.HTML("&nbsp;")
                prev_btn = gr.Button(value="[⬅️] prev", elem_id="prev_btn", scale=1)
                gr.HTML("&nbsp;")
            with gr.Column(scale=2, min_width=100):
                good_btn = gr.Button(value="[g]ood", elem_id="good_btn")
                undefined_btn = gr.Button(value="[u]ndefined", elem_id="undefined_btn")
                bad_btn = gr.Button(value="[b]ad", elem_id="bad_btn")
            with gr.Column(scale=1, min_width=50):
                count_good = gr.Number(
                    sum(1 for item in dataset if item["num_error"] == 0),
                    show_label=False,
                    container=False,
                    interactive=False,
                )
                count_undefined = gr.Number(
                    sum(1 for item in dataset if item["num_error"] is None),
                    show_label=False,
                    container=False,
                    interactive=False,
                )
                count_bad = gr.Number(
                    sum(
                        1
                        for item in dataset
                        if item["num_error"] is not None and 0 < item["num_error"]
                    ),
                    show_label=False,
                    container=False,
                    interactive=False,
                )
            with gr.Column(scale=1, min_width=50):
                gr.HTML("&nbsp;")
                next_btn = gr.Button(value="[➡️] next", elem_id="next_btn", scale=1)
                gr.HTML("&nbsp;")
        with gr.Row():
            num_error = gr.Textbox(
                init_data["num_error"], label="number of errors in QA", scale=1
            )
            save_btn = gr.Button(value="save")
            system_message = gr.Markdown(line_breaks=True)
        with gr.Accordion("details", open=False):
            context = gr.Textbox(init_data["page"]["page"], label="context", lines=10)
            output = gr.Textbox(init_data["output"], label="output", lines=10)
        inputs = [
            position,
            question,
            answer,
            context,
            num_error,
            output,
            count_good,
            count_undefined,
            count_bad,
        ]
        outputs = [
            position,
            question,
            answer,
            context,
            num_error,
            output,
            system_message,
            count_good,
            count_undefined,
            count_bad,
        ]
        next_btn.click(click_next, inputs, outputs)
        prev_btn.click(click_prev, inputs, outputs)
        good_btn.click(click_good, inputs, outputs)
        undefined_btn.click(click_undefined, inputs, outputs)
        bad_btn.click(click_bad, inputs, outputs)
        position.submit(load_data, inputs, outputs)
        input_json = gr.State(args.input_json)
        save_btn.click(manual_save_dataset, [input_json], [system_message])
        demo.load(autosave_dataset, [input_json], [system_message], every=600)
    demo.launch(server_port=args.port)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()
    launch_demo(args)


if __name__ == "__main__":
    main()
