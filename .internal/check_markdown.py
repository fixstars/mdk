# https://gitlab.com/groups/fixstars/llm/-/wikis/documentation_guideline
import argparse
import re

TITLE_LIST_REGEX = r"^ *(#+|-|\*|\+|\d+\.) "


# ，． を 、。 に置換
def to_japanese_punctuation(line: str) -> str:
    return line.replace("，", "、").replace("．", "。")


assert to_japanese_punctuation("吾輩は，猫である．") == "吾輩は、猫である。"


# 「だ」「である」で終わる文を見つけたら「です」「ます」に変換する
def to_desumasu(line: str) -> str:
    # タイトル・リスト内は常体でOK
    if re.match(TITLE_LIST_REGEX, line):
        return line
    # 表中も常体でOK
    if line.count("|") >= 2:
        return line
    for keyword, replaced in [
        ("だ", "です"),
        ("である", "です"),
        ("する", "します"),
    ]:
        # 文末の句点が含まれない常体の文
        if line.endswith(keyword):
            line = line[: -len(keyword)] + replaced
        # 特定 suffix が含まれるときは行末以外も文末判定する
        for suffix in ["。"]:
            line = line.replace(keyword + suffix, replaced + suffix)

    return line


assert to_desumasu("吾輩は猫である") == "吾輩は猫です"
assert to_desumasu("吾輩は猫である。名前はまだない") == "吾輩は猫です。名前はまだない"
assert to_desumasu("# 吾輩は猫である") == "# 吾輩は猫である"
assert to_desumasu("|吾輩|猫である|") == "|吾輩|猫である|"


# 英数字前後の半角スペースを外す
def remove_space_around_alphanum(line: str) -> str:
    # 1. 通常行の判定
    fixed_line = re.sub(
        r"(([^\x01-\x7E])\s([\x01-\x7E])|([\x01-\x7E])\s([^\x01-\x7E]))",
        r"\2\3\4\5",
        line,
    )
    if fixed_line == line:
        return line

    # 2. タイトル・リスト行の判定
    if match := re.match(TITLE_LIST_REGEX, line):
        return line[: match.end()] + remove_space_around_alphanum(line[match.end() :])

    return fixed_line


assert remove_space_around_alphanum("あ い う a b c") == "あ い うa b c"
assert remove_space_around_alphanum("# あ い う a b c") == "# あ い うa b c"
assert remove_space_around_alphanum("* あ い う a b c") == "* あ い うa b c"
assert remove_space_around_alphanum("1. あ い う a b c") == "1. あ い うa b c"
assert remove_space_around_alphanum("* `abc` あいう") == "* `abc`あいう"
assert remove_space_around_alphanum("* あいう `abc`") == "* あいう`abc`"


# 括弧内に全角文字が含まれていたら全角括弧に、すべて半角文字なら半角括弧にする
def convert_parentheses(line: str) -> str:
    # 半角括弧内に全角文字が含まれている場合は全角括弧に変換
    # ただしMarkdownのリンク（直前が"]"）は除外する
    line = re.sub(
        r"\((?<!\]\()(.*?)\)",
        lambda m: f"（{m.group(1)}）"
        if re.search(r"[^\x01-\x7E]", m.group(1))
        else m.group(0),
        line,
    )

    # 全角括弧内に半角文字しか含まれていない場合は半角括弧に変換
    line = re.sub(
        r"（(.*?)）",
        lambda m: f"({m.group(1)})"
        if not re.search(r"[^\x01-\x7E]", m.group(1))
        else m.group(0),
        line,
    )

    return line


assert convert_parentheses("(abc)") == "(abc)"
assert convert_parentheses("(あいう)") == "（あいう）"
assert convert_parentheses("（abc）") == "(abc)"
assert convert_parentheses("（あいう）") == "（あいう）"
assert convert_parentheses("[](あいう)") == "[](あいう)"


# 二重鍵かっこを削除する
def remove_double_brackets(line: str) -> str:
    return re.sub(r"『(.*?)』", r"\1", line)


assert remove_double_brackets("『あいうえお』") == "あいうえお"


# 半角カタカナの削除
def remove_half_width_katakana(line: str) -> str:
    return re.sub(r"[ｦ-ﾟ]", "", line)


assert remove_half_width_katakana("ｱｲｳｴｵ") == ""


# 全角英数の削除
def remove_full_width_alphanum(line: str) -> str:
    return re.sub(r"[０-９Ａ-Ｚａ-ｚ]", "", line)


assert remove_full_width_alphanum("０１２３ＡＢＣａｂｃ") == ""


# リストのキーワードを +, - ではなく * に統一する
def unify_list_keyword(line: str) -> str:
    return re.sub(r"^(\s*)[+-]", r"\1*", line)


assert unify_list_keyword("+ あいう") == "* あいう"
assert unify_list_keyword("- あいう") == "* あいう"
assert unify_list_keyword("  - あいう") == "  * あいう"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    find_error = False
    for filename in args.filenames:
        assert filename.endswith(".md"), filename

        with open(filename, "r") as f:
            text = f.read()
        fixed_text = []
        codeblock_depth = 0
        in_front_matter = False
        for idx, line in enumerate(text.split("\n"), start=1):
            # Check Docusaurus' front matter] to avoid
            # unify_list_keyword will break this notation.
            # ref: https://docusaurus.io/docs/markdown-features#front-matter
            if line.strip().startswith("---"):
                in_front_matter = not in_front_matter
                fixed_text.append(line)
                continue

            # Skip if it is in Docusaurus' front matter
            if in_front_matter:
                fixed_text.append(line)
                continue

            # コードブロック（空白の後に3つ以上のバッククォート）の判定
            if re.match(r"^\s*`{3,}", line):
                num_quote = line.count("`")
                if codeblock_depth == 0:
                    codeblock_depth = num_quote
                elif num_quote == codeblock_depth:
                    codeblock_depth = 0
            # コードブロック内はコメント行のみチェックする
            if codeblock_depth > 0 and not line.startswith("# "):
                fixed_text.append(line)
                continue
            for algo in [
                to_japanese_punctuation,
                to_desumasu,
                remove_space_around_alphanum,
                convert_parentheses,
                remove_double_brackets,
                remove_half_width_katakana,
                remove_full_width_alphanum,
                unify_list_keyword,
            ]:
                fixed_line = algo(line)
                if line != fixed_line:
                    find_error = True
                    print(
                        f"{algo.__name__}: {filename}:{idx}\n{line}\n{fixed_line}\n\n"
                    )
                line = fixed_line
            fixed_text.append(line)
        fixed_text = "\n".join(fixed_text)
        with open(filename, "w") as f:
            f.write(fixed_text)

    if find_error:
        exit(1)


if __name__ == "__main__":
    main()
