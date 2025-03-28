import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def aggregate_filtered_qa(eval_result):
    qa_list = defaultdict(list)
    counter = defaultdict(int)
    for data in eval_result:
        if data["num_error"] is None:
            counter["invalid"] += 1
            qa_list["invalid"] += data["qa"]
        elif data["num_error"] == 0:
            counter["clean"] += 1
            qa_list["clean"] += data["qa"]
        else:
            assert data["num_error"] >= 0, data
            counter["wrong"] += 1
            qa_list["wrong"] += data["qa"]

    return qa_list, counter


def find_confusion_matrix(eval_result):
    counter = dict(
        clean=dict(clean=0, wrong=0, invalid=0),
        wrong=dict(clean=0, wrong=0, invalid=0),
        unknown=dict(clean=0, wrong=0, invalid=0),
    )
    for data in eval_result:
        if data["previous_num_error"] is None:
            previous_judge = "unknown"
        elif data["previous_num_error"] == 0:
            previous_judge = "clean"
        else:
            previous_judge = "wrong"
        if data["num_error"] is None:
            judge = "invalid"
        elif data["num_error"] == 0:
            judge = "clean"
        else:
            assert data["num_error"] >= 0, data
            judge = "wrong"
        counter[previous_judge][judge] += 1
    tn = counter["clean"]["clean"]
    fp = counter["clean"]["wrong"]
    fn = counter["wrong"]["clean"]
    tp = counter["wrong"]["wrong"]
    logger.info(
        "accuracy: -"
        if len(eval_result) == 0
        else f"accuracy: {(tp + tn) / len(eval_result):f}"
    )
    logger.info("precision: -" if tp + fp == 0 else f"precision: {tp / (tp + fp):f}")
    logger.info("recall: -" if tp + fn == 0 else f"recall: {tp / (tp + fn):f}")
    logger.info(
        "f1: -" if 2 * tp + fp + fn == 0 else f"f1: {2 * tp / (2 * tp + fp + fn):f}"
    )
    return counter
