"""
Export model to ONNX format
"""
import torch
from pathlib import Path
from hydra import compose, initialize
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_to_onnx(checkpoint: str, output_path: str = "models/model.onnx", config_path: str = "configs/config.yaml"):
    """
    Export model to ONNX format

    Args:
        checkpoint: Path to model checkpoint
        output_path: Output path for ONNX model
        config_path: Path to config file
    """
    # Initialize Hydra - path relative to this file
    with initialize(version_base=None, config_path="../../../configs"):
        cfg = compose(config_name="config")
    print(f"Loading model from: {checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    dummy_text = "Проанализируй сообщение и определи токсичность.\n\nСообщение: Привет\n\nОтвет: "
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=512, truncation=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"Exporting to ONNX: {output_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"}},
        opset_version=14,
        do_constant_folding=True,
    )

    print(f"Model exported to ONNX: {output_path}")


if __name__ == "__main__":
    export_to_onnx()


