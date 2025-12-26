"""
Export model to ONNX format
"""
import torch
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_to_onnx(checkpoint: str = None, output_path: str = "triton_model_repository/toxic_classificator/1/model.onnx", config_path: str = "configs/config.yaml"):
    """
    Export LoRA model to ONNX format for Triton Inference Server

    Args:
        checkpoint: Path to LoRA adapter (default: models/finetuned/final)
        output_path: Output path for ONNX model
        config_path: Path to config file
    """
    # Initialize Hydra with absolute path
    project_root = Path.cwd()
    config_dir = project_root / "configs"
    
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
        cfg = compose(config_name="config")
    
    # Determine adapter path
    if checkpoint:
        adapter_path = checkpoint
    else:
        adapter_path = str(project_root / cfg.training.output_dir / "final")
    
    print(f"Loading base model: {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge LoRA weights into base model for ONNX export
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    model.eval()

    # Prepare dummy inputs
    dummy_text = "Human: Привет\nAssistant:"
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=cfg.model.max_length, truncation=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"Exporting to ONNX: {output_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"}
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
    )

    # Save tokenizer for Triton
    tokenizer_path = output_path.parent / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_path))

    print(f"✅ Model exported to ONNX: {output_path}")
    print(f"✅ Tokenizer saved to: {tokenizer_path}")


if __name__ == "__main__":
    export_to_onnx()


