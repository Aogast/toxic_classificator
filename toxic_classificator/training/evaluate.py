"""
Evaluation script with metrics logging to MLflow
"""
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import mlflow
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_test_data(data_path: Path) -> List[Dict]:
    """Load test data from JSON"""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_response(response: str) -> Dict:
    """Parse model response to extract classification"""
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
        else:
            response_lower = response.lower()
            toxic = "true" in response_lower
            return {"toxic": toxic, "labels": []}

    except json.JSONDecodeError:
        response_lower = response.lower()
        toxic = "true" in response_lower
        return {"toxic": toxic, "labels": []}


def evaluate(config_path: str = "configs/config.yaml", checkpoint: str = None):
    """Evaluate model on test data"""
    print("Starting evaluation...")

    # Initialize Hydra with absolute path
    project_root = Path.cwd()
    config_dir = project_root / "configs"
    
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
        cfg = compose(config_name="config")

    project_root = Path.cwd()

    if cfg.dvc.auto_pull:
        print("Pulling data from DVC...")
        try:
            subprocess.run(["dvc", "pull"], check=True, cwd=project_root)
        except subprocess.CalledProcessError:
            print("Warning: Could not pull data from DVC")

    test_data_path = project_root / cfg.paths.processed_data_dir / "test.json"
    test_data = load_test_data(test_data_path)

    print(f"Loaded {len(test_data)} test samples")

    model_path = checkpoint if checkpoint else str(project_root / cfg.training.output_dir / "final")

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    y_true = []
    y_pred = []

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="evaluation"):
        for sample in tqdm(test_data, desc="Evaluating"):
            prompt = sample["training_text"].split("Ответ: ")[0] + "Ответ: "

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

            prediction = parse_response(response)
            pred_label = 1 if prediction["toxic"] else 0

            y_true.append(sample["toxic"])
            y_pred.append(pred_label)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average="binary")

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=["Non-toxic", "Toxic"])

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        mlflow.log_metrics({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1})

        print("\n" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"            Non-toxic  Toxic")
        print(f"Actual Non-toxic  {cm[0][0]:6d}  {cm[0][1]:6d}")
        print(f"       Toxic      {cm[1][0]:6d}  {cm[1][1]:6d}")
        print(f"\n{report}")

        results_file = project_root / cfg.paths.plots_dir / "evaluation_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {k: v if not isinstance(v, list) else v for k, v in results.items()}, f, ensure_ascii=False, indent=2
            )

        mlflow.log_artifact(str(results_file))

        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    evaluate()
