import io
import logging

from pypdf import PdfReader
from tqdm.auto import tqdm

from preprocess.vision_llm import explain_image

logger = logging.getLogger(__name__)


def split_by_pdf_page(input_file: str, args):
    assert input_file.endswith(".pdf"), input_file
    pdf_reader = PdfReader(input_file)

    ret = []
    for page in tqdm(pdf_reader.pages, desc="split pdf page"):
        text = page.extract_text()
        if args.vision_llm.model is not None:
            base_text = text
            for image_file_object in page.images:
                text += "\nAttached Image: "
                text += explain_image(
                    io.BytesIO(image_file_object.data),
                    args,
                    context=base_text,
                )[0]
        ret.append(text)

    return ret
