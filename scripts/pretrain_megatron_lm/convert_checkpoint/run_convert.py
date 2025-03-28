import argparse

from convert import convert_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model to safetensors format."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input PyTorch model (.bin file)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output safetensors file"
    )

    args = parser.parse_args()

    convert_file(args.input, args.output, discard_names=[])


if __name__ == "__main__":
    main()
