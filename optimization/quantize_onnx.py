"""Dynamically quantize an ONNX model's weights to signed INT8."""

import argparse
from pathlib import Path


def quantize_model(input_path: Path, output_path: Path) -> tuple[int, int]:
    if not input_path.is_file():
        raise FileNotFoundError(f"FP32 ONNX model not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    return input_path.stat().st_size, output_path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("models/model.onnx"))
    parser.add_argument("--output", type=Path, default=Path("models/model_int8.onnx"))
    args = parser.parse_args()
    input_size, output_size = quantize_model(args.input, args.output)
    reduction = (1 - output_size / input_size) * 100
    print(
        f"Saved INT8 model to {args.output} "
        f"({input_size:,} -> {output_size:,} bytes, {reduction:.1f}% smaller)"
    )


if __name__ == "__main__":
    main()