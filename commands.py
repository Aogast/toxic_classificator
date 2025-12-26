"""
Main entry point for all commands using Fire CLI
"""
import fire
from pathlib import Path

from toxic_classificator.data.download_data import download_data
from toxic_classificator.data.prepare_data import prepare_data
from toxic_classificator.training.train import train
from toxic_classificator.training.evaluate import evaluate
from toxic_classificator.inference.predict import predict
from toxic_classificator.inference.export_onnx import export_to_onnx


class Commands:
    """Main commands for toxic classificator project"""

    def download(self):
        """Download datasets from Kaggle"""
        download_data()

    def prepare(self):
        """Prepare and preprocess data"""
        prepare_data()

    def train(self, config_path: str = "configs/config.yaml"):
        """
        Train the model

        Args:
            config_path: Path to config file
        """
        train(config_path)

    def evaluate(self, config_path: str = "configs/config.yaml", checkpoint: str = None):
        """
        Evaluate the model

        Args:
            config_path: Path to config file
            checkpoint: Path to model checkpoint
        """
        evaluate(config_path, checkpoint)

    def predict(
        self,
        text: str = None,
        input_file: str = None,
        output_file: str = None,
        config_path: str = "configs/config.yaml",
        checkpoint: str = None,
    ):
        """
        Run predictions

        Args:
            text: Single text to predict
            input_file: File with texts (one per line)
            output_file: Output file for predictions
            config_path: Path to config file
            checkpoint: Path to model checkpoint
        """
        predict(text, input_file, output_file, config_path, checkpoint)

    def export_onnx(
        self, checkpoint: str, output_path: str = "models/model.onnx", config_path: str = "configs/config.yaml"
    ):
        """
        Export model to ONNX format

        Args:
            checkpoint: Path to model checkpoint
            output_path: Output path for ONNX model
            config_path: Path to config file
        """
        export_to_onnx(checkpoint, output_path, config_path)


def main():
    """Main entry point"""
    fire.Fire(Commands)


if __name__ == "__main__":
    main()

