from pathlib import Path

import numpy as np


def export_to_onnx(model_dir: Path, output_onnx_path: Path):
    """Converts PyTorch AraBERT checkpoint to ONNX computation graph."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    dummy_text = "المنتج ممتاز جداً"
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    input_names = [
        name for name in ("input_ids", "attention_mask", "token_type_ids") if name in inputs
    ]
    model_inputs = tuple(inputs[name] for name in input_names)
    dynamic_axes = {
        name: {0: "batch_size", 1: "sequence_length"} for name in input_names
    }
    dynamic_axes["logits"] = {0: "batch_size"}

    torch.onnx.export(
        model,
        model_inputs,
        str(output_onnx_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
    )
    print(f"ONNX model successfully exported to {output_onnx_path}")

def verify_parity(model_dir: Path, onnx_path: Path):
    """Verifies output equality between PyTorch and ONNX Runtime."""
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    test_text = "DVC and ONNX parity check sentence"
    inputs = tokenizer(
        test_text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )

    model_kwargs = {
        "attention_mask": inputs["attention_mask"],
    }
    if "token_type_ids" in inputs:
        model_kwargs["token_type_ids"] = inputs["token_type_ids"]
    with torch.no_grad():
        pt_logits = model(inputs["input_ids"], **model_kwargs).logits.numpy()

    session = ort.InferenceSession(str(onnx_path))
    onnx_inputs = {
        "input_ids": inputs["input_ids"].numpy().astype(np.int64),
        "attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
    }
    if "token_type_ids" in inputs:
        onnx_inputs["token_type_ids"] = inputs["token_type_ids"].numpy().astype(
            np.int64
        )
    onnx_logits = session.run(None, onnx_inputs)

    assert np.allclose(
        pt_logits, onnx_logits, atol=1e-4
    ), "Parity check failed between PyTorch and ONNX!"
    print("ONNX Parity test passed! Difference is within tolerance (atol=1e-4).")