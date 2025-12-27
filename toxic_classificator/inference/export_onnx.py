import shutil
import torch
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_to_onnx(checkpoint: str = None, output_path: str = "triton_model_repository/toxic_classificator/1/model.onnx", config_path: str = "configs/config.yaml"):
    project_root = Path.cwd()
    config_dir = project_root / "configs"
    
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
        cfg = compose(config_name="config")
    
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
    
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    model.eval()

    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_model_dir = output_dir / "merged_model"
    print(f"Saving merged model to: {merged_model_dir}")
    model.save_pretrained(str(merged_model_dir))
    tokenizer.save_pretrained(str(merged_model_dir))
    
    print(f"Model exported to ONNX: {output_path}")
    print(f"Tokenizer saved to: {merged_model_dir}")
    
    info_file = output_dir / "README.txt"
    with open(info_file, "w") as f:
        f.write("Merged LoRA model for Triton Inference Server\n")
        f.write(f"Base model: {cfg.model.name}\n")
        f.write(f"LoRA adapter: {adapter_path}\n")
        f.write("\nFor Triton deployment:\n")
        f.write("1. Use PyTorch backend with merged_model/\n")
        f.write("2. Or convert to ONNX: optimum-cli export onnx --model merged_model/ onnx_model/\n")
        f.write("3. Or convert to TensorRT: see convert_to_tensorrt.sh\n")
    
    print(f"\nInstructions saved to: {info_file}")


if __name__ == "__main__":
    export_to_onnx()
