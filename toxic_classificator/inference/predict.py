"""
Prediction script
"""
import json
from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import DictConfig
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
    # Initialize Hydra - path relative to this file
    with initialize(version_base=None, config_path="../../../configs"):
        cfg = compose(config_name="config")

    project_root = Path.cwd()
    model_path = checkpoint if checkpoint else str(project_root / cfg.training.output_dir / "final")

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    if text:
        texts = [text]
    elif input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Provide either text or input_file")
        return

    results = []
    for txt in texts:
        prompt = f"Проанализируй сообщение и определи токсичность.\n\nСообщение: {txt}\n\nОтвет: "
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        prediction = parse_response(response)

        results.append({"text": txt, "toxic": prediction["toxic"], "labels": prediction["labels"]})

        if not output_file:
            print(f"\nText: {txt}")
            print(f"Toxic: {prediction['toxic']}")
            print(f"Labels: {prediction['labels']}")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    predict()
