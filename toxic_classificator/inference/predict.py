"""
Prediction script
"""
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_response(response: str) -> dict:
    """Parse model response"""
    try:
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)

            if "toxic" not in result:
                result["toxic"] = False
            if "labels" not in result:
                result["labels"] = []

            return result
    except json.JSONDecodeError:
        pass

    response_lower = response.lower()
    toxic = "true" in response_lower
    return {"toxic": toxic, "labels": []}


def predict(
    text: str = None, input_file: str = None, output_file: str = None, config_path: str = "configs/config.yaml", checkpoint: str = None
):
    """Run predictions"""
    # Initialize Hydra with absolute path
    project_root = Path.cwd()
    config_dir = project_root / "configs"
    
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
        cfg = compose(config_name="config")
    
    # Determine model path
    if checkpoint:
        adapter_path = checkpoint
    else:
        adapter_path = str(project_root / cfg.training.output_dir / "final")
    
    print(f"Loading base model: {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    if text:
        texts = [text]
    elif input_file:
        input_path = Path(input_file)
        if input_path.suffix == ".json":
            # Load JSON file (list of texts or list of dicts with 'text' field)
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    if data and isinstance(data[0], dict):
                        texts = [item.get("text", item.get("comment_text", "")) for item in data]
                    else:
                        texts = data
                else:
                    texts = [data]
        else:
            # Load text file (one text per line)
            with open(input_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(texts)} texts from {input_file}")
    else:
        print("Error: Provide either --text or --input_file")
        return

    results = []
    for txt in texts:
        # Use the same prompt format as training
        prompt = f"Human: {txt}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.model.max_length).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=cfg.model.temperature,
                top_p=cfg.model.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        prediction = parse_response(response)

        results.append({"text": txt, "toxic": prediction["toxic"], "labels": prediction["labels"], "response": response})

        if not output_file:
            print(f"\nText: {txt}")
            print(f"Response: {response}")
            print(f"Toxic: {prediction['toxic']}")
            print(f"Labels: {prediction['labels']}")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    predict()
