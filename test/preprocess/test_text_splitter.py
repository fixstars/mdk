from tempfile import TemporaryDirectory

from src.preprocess.text_splitter import split_by_length, split_by_markdown_header


def _dummy_doc(num_char_per_line=10, num_line=10, begin_idx=0):
    """generate text like this:

    (begin_idx)_________________ ... ___
    (begin_idx + 1)_____________ ... ___
    ...
    (begin_idx + num_line - 1)__ ... ___
    """
    return "".join(
        [
            str(i) + "_" * (num_char_per_line - len(str(i))) + "\n"
            for i in range(begin_idx, begin_idx + num_line)
        ]
    )


def _dummy_code(num_char_per_line=10, num_line=10, begin_idx=0):
    return "```\n" + _dummy_doc(num_char_per_line, num_line, begin_idx) + "```\n"


def _split_md(content: str, **kwargs):
    with TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/sample.md"
        with open(filename, "w") as f:
            f.write(content)
        result = split_by_markdown_header(filename, **kwargs)
    return result


class TestMD2TXT:
    def test_split_simple_markdown(self):
        content = "".join(
            [f"# title {i}\n" + _dummy_doc(begin_idx=10 * i) for i in range(10)]
        )
        result = _split_md(content)
        assert len(result) == 10
        for i, txt in enumerate(result):
            assert txt == f"# title {i}\n" + _dummy_doc(begin_idx=10 * i)

    def test_split_long_markdown(self):
        content = ("# title\n" + _dummy_doc(100, 35)) * 4
        result = _split_md(content)
        assert len(result) == 16

    def test_split_markdown_with_code(self):
        content = ("# title\n" + _dummy_doc() + _dummy_code() + _dummy_doc()) * 10
        result = _split_md(content)
        assert len(result) == 30

    def test_split_multi_header_markdown(self):
        content = "".join(
            ["#" * (i % 6 + 1) + f" title {i}\n" + _dummy_doc() for i in range(10)]
        )
        result = _split_md(content)
        assert len(result) == 10
        assert result[0] == "# title 0\n" + _dummy_doc()
        assert result[1] == "# title 0\n## title 1\n" + _dummy_doc()
        assert result[2] == "# title 0\n## title 1\n### title 2\n" + _dummy_doc()
        assert result[6] == "# title 6\n" + _dummy_doc()

    def test_split_md_with_merging(self):
        content = "".join([f"# title {i}\n" + _dummy_doc() for i in range(10)])
        result = _split_md(content, min_chars=150)
        assert len(result) == 5


def _split_txt(content: str, **kwargs):
    with TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/sample.txt"
        with open(filename, "w") as f:
            f.write(content)
        result = split_by_length(filename, **kwargs)
    return result


class TestTXT2TXT:
    def test_split_normal_txt(self):
        content = _dummy_doc(100, 100)
        result = _split_txt(content)
        assert len(result) == 10
        for i, txt in enumerate(result):
            assert txt == _dummy_doc(100, 10, 10 * i)

    def test_split_txt_with_duplicate(self):
        content = _dummy_doc(10, 100)
        result = _split_txt(content, long_char_thres=100, num_duplicate=15)
        assert len(result) == 12
        for i, txt in enumerate(result):
            assert txt == _dummy_doc(10, 1 if i == 11 else 10, i * 9)
