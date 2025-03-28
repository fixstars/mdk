import re
from typing import Optional

import yaml


def find_parenthesis(text: str) -> list[str]:
    return re.findall(r"\{([\s\S]*?)\}", text)


def simplify_key(text: str) -> str:
    return re.sub(r"[\n\",'\s]+", "", text)


def simplify_value(text: str) -> str:
    text = text.strip()
    quoted = True
    while quoted:
        quoted = False
        if text == "":
            break
        if text[0] == text[-1] == '"' or text[0] == text[-1] == "'":
            quoted = True
            text = text[1:-1].strip()
    return text


def split_to_dict(text: str) -> dict[str, str]:
    ret = dict()
    for kv in text.split(","):
        if kv.count(":") == 0:
            continue
        k = simplify_key(kv.split(":")[0])
        if k is None:
            continue
        ret[k] = simplify_value(":".join(kv.split(":")[1:]))
    return ret


def find_json(reply: str) -> list[dict[str, str]]:
    ret = []
    for text in find_parenthesis(reply):
        ret.append(split_to_dict(text))
    return ret


def any_contains(s: str, substr: list[str]) -> bool:
    for t in substr:
        if t in s:
            return True
    return False


def find_qa(reply: str, min_length=20) -> list[dict[str, str]]:
    ret = []
    for text in find_parenthesis(reply):
        qa = dict()
        for k, v in split_to_dict(text).items():
            if len(v) < min_length:
                continue
            if any_contains(k.lower(), ["問", "q"]):
                qa["question"] = v
            if any_contains(k.lower(), ["答", "a"]):
                qa["answer"] = v
        if len(qa) == 2:
            ret.append(qa)
    return ret


def find_dpo(reply: str, min_length=20) -> list[dict[str, str]]:
    """find pair of (correct answer, wrong answer)
    prompts needs to be concatenated to make dpo dataset
    {prompt: list[str], chosen: list[str], rejected: list[str]}
    cf. https://huggingface.co/docs/trl/main/en/dpo_trainer#expected-dataset-format
    """
    ret = []
    for text in find_parenthesis(reply):
        dpo = dict()
        for k, v in split_to_dict(text).items():
            if len(v) < min_length:
                continue
            if any_contains(k.lower(), ["正", "correct", "right"]):
                dpo["chosen"] = v
            if any_contains(k.lower(), ["誤", "wrong"]):
                dpo["rejected"] = v
        if len(dpo) == 2:
            ret.append(dpo)
    return ret


def calc_num_error(
    output: list, max_ok_error_threshold: Optional[float], coefficient_yaml_path: str
) -> Optional[int]:
    if max_ok_error_threshold is None:
        return None

    score = calc_score(output, coefficient_yaml_path)
    if score is None:
        return None

    if score <= max_ok_error_threshold:
        return 0  # no error

    return 1  # has error


def raw_score(output: list, coefficient_yaml_path: str) -> list:
    result = []

    with open(coefficient_yaml_path, "r") as f:
        coef = yaml.safe_load(f)

    num_repeat = len(output) // 2

    for i in range(num_repeat):
        reply1 = output[i]
        reply2 = output[i + num_repeat]
        attr1 = reply1[-2:-1]  # [/ATTR_1]
        attr2 = reply2[-2:-1]  # [/ATTR_2]
        if attr1 != "1" or attr2 != "2":
            return []

        a1 = list(map(int, re.findall(r"\d+", reply1.split(" [/ATTR_1]")[0])))
        a2 = list(map(int, re.findall(r"\d+", reply2.split(" [/ATTR_2]")[0])))
        a = a1 + a2
        if len(a) == 9:
            result.append(
                coef["slope"][0] * a[0]
                + coef["slope"][1] * a[1]
                + coef["slope"][2] * a[2]
                + coef["slope"][3] * a[3]
                + coef["slope"][4] * a[4]
                + coef["slope"][5] * a[5]
                + coef["slope"][6] * a[6]
                + coef["slope"][7] * a[7]
                + coef["slope"][8] * a[8]
                + coef["intercept"]
            )
        else:
            return []

    return result


def calc_score(output: list, coefficient_yaml_path: str) -> Optional[int]:
    scores = raw_score(output, coefficient_yaml_path)

    if len(scores) == 0:
        return None

    return sum(scores) / len(scores)
