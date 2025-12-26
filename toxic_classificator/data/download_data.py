"""
Download datasets from Kaggle and add to DVC
"""
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


def download_kaggle_dataset(dataset_id: str, output_dir: Path):
    """
    Download a dataset from Kaggle using kaggle CLI

    Args:
        dataset_id: Kaggle dataset identifier
        output_dir: Directory to save the downloaded dataset
    """
    print(f"Downloading {dataset_id}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if dataset_id.startswith("c/"):
            competition_name = dataset_id.replace("c/", "")
            subprocess.run(
                ["kaggle", "competitions", "download", "-c", competition_name, "-p", str(output_dir)],
                check=True,
            )
        else:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(output_dir), "--unzip"],
                check=True,
            )
        print(f"Successfully downloaded {dataset_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {dataset_id}: {e}")
        sys.exit(1)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def download_data(cfg: DictConfig):
    """Download all required datasets"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("Error: Kaggle credentials not found!")
        print("Please set up your Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Place the downloaded kaggle.json in ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)

    project_root = Path.cwd()
    raw_data_dir = project_root / cfg.paths.raw_data_dir

    for dataset in cfg.data.datasets:
        dataset_dir = raw_data_dir / dataset.name
        download_kaggle_dataset(dataset.kaggle_id, dataset_dir)

    print("\nAll datasets downloaded successfully!")
    print(f"Data saved to: {raw_data_dir}")

    print("\nAdding data to DVC...")
    try:
        subprocess.run(["dvc", "add", str(raw_data_dir)], check=True, cwd=project_root)
        print("Data added to DVC successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not add data to DVC: {e}")


if __name__ == "__main__":
    download_data()

