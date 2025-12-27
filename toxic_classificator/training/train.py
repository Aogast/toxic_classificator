import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import lightning as L
import mlflow
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


def load_data(data_path: Path) -> List[Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


class ToxicCommentsDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        full_text = sample["training_text"]

        encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        prompt_part = full_text.split("Ответ: ")[0] + "Ответ: "
        prompt_encoding = self.tokenizer(
            prompt_part, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        prompt_length = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels[:prompt_length] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class ToxicClassifierModule(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.setup_model()

    def setup_model(self):
        model_name = self.cfg.model.name
        print(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.cfg.model.load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=self.cfg.model.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.cfg.model.bnb_4bit_quant_type,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model = prepare_model_for_kbit_training(self.model)

        target_modules = list(self.cfg.training.lora.target_modules)
        
        lora_config = LoraConfig(
            r=self.cfg.training.lora.r,
            lora_alpha=self.cfg.training.lora.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.cfg.training.lora.lora_dropout,
            bias=self.cfg.training.lora.bias,
            task_type=self.cfg.training.lora.task_type,
        )

        self.model = get_peft_model(self.model, lora_config)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"Trainable params: {trainable_params:,} || "
            f"Total params: {total_params:,} || "
            f"Trainable%: {100 * trainable_params / total_params:.2f}%"
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.training.learning_rate, weight_decay=self.cfg.training.weight_decay
        )

        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(num_training_steps * self.cfg.training.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}


class ToxicDataModule(L.LightningDataModule):
    def __init__(self, train_data: List[Dict], val_data: List[Dict], tokenizer, cfg: DictConfig):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.cfg = cfg

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = ToxicCommentsDataset(
                self.train_data, self.tokenizer, self.cfg.model.max_length
            )
            self.val_dataset = ToxicCommentsDataset(self.val_data, self.tokenizer, self.cfg.model.max_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.per_device_train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.per_device_eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )


def train(config_path: str = "configs/config.yaml"):
    print("Starting training...")

    project_root = Path.cwd()
    config_dir = project_root / "configs"
    
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
        cfg = compose(config_name="config")

    print(OmegaConf.to_yaml(cfg))

    project_root = Path.cwd()

    if cfg.dvc.auto_pull:
        print("Pulling data from DVC...")
        try:
            subprocess.run(["dvc", "pull"], check=True, cwd=project_root)
        except subprocess.CalledProcessError:
            print("Warning: Could not pull data from DVC")

    data_dir = project_root / cfg.paths.processed_data_dir
    train_data = load_data(data_dir / "train.json")
    val_data = load_data(data_dir / "val.json")

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=cfg.mlflow.run_name,
    )

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        mlf_logger.log_hyperparams({"git_commit": git_commit})
    except Exception:
        pass

    mlf_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    model = ToxicClassifierModule(cfg)
    data_module = ToxicDataModule(train_data, val_data, model.tokenizer, cfg)

    output_dir = project_root / cfg.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=cfg.training.save_total_limit,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True)
    
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        max_epochs=cfg.training.num_train_epochs,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if cfg.training.use_bf16 else "16-mixed",
        gradient_clip_val=cfg.training.max_grad_norm,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        log_every_n_steps=cfg.training.logging_steps,
        val_check_interval=cfg.training.eval_steps,
        check_val_every_n_epoch=None,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=mlf_logger,
        default_root_dir=str(project_root / cfg.paths.logs_dir),
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(model, data_module)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(str(final_dir))
    model.tokenizer.save_pretrained(str(final_dir))

    mlf_logger.experiment.log_artifact(mlf_logger.run_id, str(final_dir))

    print("Training completed successfully!")


if __name__ == "__main__":
    train()
