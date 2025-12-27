import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


def load_jigsaw_data(data_dir: Path) -> pd.DataFrame:
    print("Loading Jigsaw Multilingual dataset...")

    possible_files = ["jigsaw-toxic-comment-train.csv", "train.csv", "jigsaw_train.csv"]

    train_file = None
    for filename in possible_files:
        file_path = data_dir / filename
        if file_path.exists():
            train_file = file_path
            break

    if train_file is None:
        print(f"Warning: Could not find Jigsaw training file in {data_dir}")
        return pd.DataFrame(columns=["text", "toxic", "labels"])

    df = pd.read_csv(train_file)

    result = pd.DataFrame()
    result["text"] = df["comment_text"]
    result["toxic"] = df["toxic"].astype(int)

    label_cols = ["severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    available_labels = [col for col in label_cols if col in df.columns]

    result["labels"] = df[available_labels].apply(
        lambda row: [
            label_cols[i] for i, val in enumerate(row) if val == 1 and label_cols[i] in available_labels
        ],
        axis=1,
    )

    print(f"Loaded {len(result)} samples from Jigsaw dataset")
    return result


def load_russian_toxic_data(data_dir: Path) -> pd.DataFrame:
    print("Loading Russian Toxic Comments dataset...")

    possible_files = ["labeled.csv", "russian_toxic_comments.csv", "train.csv"]

    train_file = None
    for filename in possible_files:
        file_path = data_dir / filename
        if file_path.exists():
            train_file = file_path
            break

    if train_file is None:
        print(f"Warning: Could not find Russian toxic file in {data_dir}")
        return pd.DataFrame(columns=["text", "toxic", "labels"])

    df = pd.read_csv(train_file)

    result = pd.DataFrame()
    result["text"] = df["comment"] if "comment" in df.columns else df["text"]
    result["toxic"] = df["toxic"].astype(int) if "toxic" in df.columns else 0
    result["labels"] = [[] for _ in range(len(df))]

    print(f"Loaded {len(result)} samples from Russian dataset")
    return result


def combine_datasets(cfg: DictConfig) -> pd.DataFrame:
    project_root = Path.cwd()
    raw_data_dir = project_root / cfg.paths.raw_data_dir

    jigsaw_df = load_jigsaw_data(raw_data_dir / "jigsaw-multilingual")
    russian_df = load_russian_toxic_data(raw_data_dir / "russian-toxic")

    combined = pd.concat([jigsaw_df, russian_df], ignore_index=True)

    if cfg.data.remove_duplicates:
        combined = combined.drop_duplicates(subset=["text"])

    combined = combined.dropna(subset=["text"])
    combined = combined[combined["text"].str.len() > cfg.data.min_text_length]

    print(f"\nTotal samples after combining: {len(combined)}")
    print(f"Toxic samples: {combined['toxic'].sum()}")
    print(f"Non-toxic samples: {(combined['toxic'] == 0).sum()}")

    if len(combined) > cfg.data.max_samples:
        print(f"Sampling {cfg.data.max_samples} samples from {len(combined)} total...")
        combined = combined.groupby("toxic", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), cfg.data.max_samples // 2), random_state=cfg.data.seed)
        )

    return combined


def format_training_example(text: str, toxic: bool, labels: List[str]) -> str:
    response = {"toxic": toxic, "labels": labels if labels else []}
    response_json = json.dumps(response, ensure_ascii=False)

    full_text = f"""Проанализируй сообщение и определи токсичность.

Сообщение: {text}

Ответ: {response_json}"""

    return full_text


def prepare_splits(df: pd.DataFrame, cfg: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_split = cfg.data.train_split
    val_split = cfg.data.val_split
    test_split = cfg.data.test_split
    seed = cfg.data.seed

    train_df, temp_df = train_test_split(
        df, train_size=train_split, random_state=seed, stratify=df["toxic"]
    )

    val_size = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_size, random_state=seed, stratify=temp_df["toxic"]
    )

    print(f"\nDataset splits:")
    print(f"Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def save_datasets(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        data = []
        for _, row in df.iterrows():
            full_text = format_training_example(row["text"], bool(row["toxic"]), row["labels"])

            response = {"toxic": bool(row["toxic"]), "labels": row["labels"] if row["labels"] else []}

            data.append(
                {
                    "text": row["text"],
                    "toxic": int(row["toxic"]),
                    "labels": row["labels"],
                    "training_text": full_text,
                    "response": json.dumps(response, ensure_ascii=False),
                }
            )

        output_file = output_dir / f"{split_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(data)} samples to {output_file}")


def prepare_data():
    print("Starting data preparation...")

    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")

    project_root = Path.cwd()

    if cfg.dvc.auto_pull:
        print("Pulling data from DVC...")
        try:
            subprocess.run(["dvc", "pull"], check=True, cwd=project_root)
        except subprocess.CalledProcessError:
            print("Warning: Could not pull data from DVC")

    combined_df = combine_datasets(cfg)
    train_df, val_df, test_df = prepare_splits(combined_df, cfg)

    processed_dir = project_root / cfg.paths.processed_data_dir
    save_datasets(train_df, val_df, test_df, processed_dir)

    print("\nAdding processed data to DVC...")
    try:
        subprocess.run(["dvc", "add", str(processed_dir)], check=True, cwd=project_root)
        print("Processed data added to DVC successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not add processed data to DVC: {e}")

    print("\nData preparation completed successfully!")


if __name__ == "__main__":
    prepare_data()
