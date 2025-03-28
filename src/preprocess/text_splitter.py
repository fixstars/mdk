import logging
import re

MAX_HEADER_LEVEL = 6

logger = logging.getLogger(__name__)


def split_by_length(
    input_file: str, *, long_char_thres=1000, num_duplicate=0, **kwargs
):
    assert input_file.endswith((".md", ".txt")), input_file
    assert long_char_thres > num_duplicate, (long_char_thres, num_duplicate)
    with open(input_file) as f:
        lines = f.readlines()
    ret = []
    tmp = []
    tmp_size = 0
    # reduce length of text
    lines = [re.sub(r"--|\||  |==", "", line) for line in lines]
    for line in lines:
        tmp.append(line)
        tmp_size += len(line)
        if tmp_size > long_char_thres:
            ret.append("".join(tmp))
            while tmp_size > num_duplicate:
                tmp_size -= len(tmp[0])
                tmp = tmp[1:]
    if len(tmp) > 0:
        ret.append("".join(tmp))
    return ret


def split_by_markdown_header(
    input_file: str, *, min_chars=0, long_char_thres=1000, **kwargs
):
    assert input_file.endswith(".md"), input_file
    with open(input_file) as f:
        lines = f.readlines()

    # organize lines by section
    sections = []
    tmp = ""
    headers = [None] * MAX_HEADER_LEVEL
    is_preformatted = False
    for line in lines:
        if line.startswith("#") and not is_preformatted:
            # save document
            if len(tmp) > 0:
                sections.append("".join([h for h in headers if h is not None]) + tmp)
                tmp = ""
            # update headers
            header_level = len(line.split(" ")[0])
            assert 1 <= header_level <= MAX_HEADER_LEVEL, line
            headers[header_level - 1] = line
            for h in range(header_level, MAX_HEADER_LEVEL):
                headers[h] = None
        elif line.startswith("```"):
            is_preformatted = not is_preformatted
            # save document
            if len(tmp) > 0:
                sections.append("".join([h for h in headers if h is not None]) + tmp)
                tmp = ""
        else:
            # reduce length of text
            tmp += re.sub(r"--|\||  |==", "", line)
            # save document
            if len(tmp) > long_char_thres:
                logger.warning(
                    "long document detected. split in the middle of document: %s",
                    input_file,
                )
                sections.append("".join([h for h in headers if h is not None]) + tmp)
                tmp = ""
    if len(tmp) > 0:
        sections.append("".join([h for h in headers if h is not None]) + tmp)

    # merge small sections
    ret = []
    tmp = ""
    for section in sections:
        tmp += section
        if len(tmp) > min_chars:
            ret.append(tmp)
            tmp = ""
    if len(tmp) > 0:
        ret.append(tmp)

    return ret
