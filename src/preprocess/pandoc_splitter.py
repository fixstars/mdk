from pathlib import Path
from tempfile import TemporaryDirectory

import pypandoc

from preprocess.text_splitter import split_by_length

PANDOC_SUPPORTED_EXT = (
    ".csv",
    ".html",
    ".ipynb",
    ".json",
    ".tex",
    ".rst",
    ".docx",
)


def split_with_pandoc(input_file: str, args) -> list[str]:
    assert input_file.endswith(PANDOC_SUPPORTED_EXT), input_file
    with TemporaryDirectory() as tmpdir:
        if args.splitter.dump_md_file:
            # name.ext -> name.md
            output_filename = Path(input_file).name.split(".")
            output_filename[-1] = "md"
            output_filename = ".".join(output_filename)
            outputfile = Path(args.output.basedir) / output_filename
        else:
            outputfile = Path(tmpdir) / "out.md"
        pypandoc.convert_file(input_file, "markdown", outputfile=outputfile)
        return split_by_length(str(outputfile), **args.splitter)
