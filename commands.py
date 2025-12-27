import fire
from pathlib import Path

from toxic_classificator.data.download_data import download_data
from toxic_classificator.data.prepare_data import prepare_data
from toxic_classificator.training.train import train
from toxic_classificator.training.evaluate import evaluate
from toxic_classificator.inference.predict import predict
from toxic_classificator.inference.export_onnx import export_to_onnx


class Commands:
    def download(self):
        download_data()

    def prepare(self):
        prepare_data()

    def train(self, config_path: str = "configs/config.yaml"):
        train(config_path)

    def evaluate(self, config_path: str = "configs/config.yaml", checkpoint: str = None):
        evaluate(config_path, checkpoint)

    def predict(
        self,
        text: str = None,
        input_file: str = None,
        output_file: str = None,
        config_path: str = "configs/config.yaml",
        checkpoint: str = None,
    ):
        predict(text, input_file, output_file, config_path, checkpoint)

    def export_onnx(
        self, checkpoint: str = None, output_path: str = "triton_model_repository/toxic_classificator/1/model.onnx", config_path: str = "configs/config.yaml"
    ):
        export_to_onnx(checkpoint, output_path, config_path)


def main():
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
