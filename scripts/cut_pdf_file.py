import argparse

import pypdf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_file_path", help="path of target pdf file")
    parser.add_argument(
        "start_page", type=int, help="start page number (including the specified page)"
    )
    parser.add_argument(
        "end_page", type=int, help="end page number (including the specified page)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pdf = pypdf.PdfReader(args.pdf_file_path)
    writer = pypdf.PdfWriter()
    assert 0 < args.start_page, args
    assert args.start_page <= args.end_page, args
    assert args.end_page <= len(pdf.pages), args
    for i in range(args.start_page - 1, args.end_page):
        writer.add_page(pdf.pages[i])
    out_path = args.pdf_file_path.replace(
        ".pdf", f"_{args.start_page}_{args.end_page}.pdf"
    )
    writer.write(out_path)
    print(f"{out_path} was created.")


if __name__ == "__main__":
    main()
