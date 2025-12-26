# Классификатор токсичных комментариев

**Автор:** Сыров Артём Андреевич

## Описание

Модель для классификации токсичных сообщений на русском и английском языках. Использует Qwen2.5-0.5B-Instruct с LoRA fine-tuning.

**Датасеты:**
- Jigsaw Multilingual Toxic Comment Classification
- Russian Language Toxic Comments

**Технологии:**
- PyTorch Lightning
- Hydra
- MLflow
- DVC
- PEFT/LoRA
- ONNX
- Triton Inference Server

## Setup

### Установка зависимостей

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
source .venv/bin/activate

uv pip install -e .
```

### Настройка Kaggle API

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Pre-commit хуки

```bash
pre-commit install
pre-commit run -a
```

### MLflow сервер

```bash
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

## Train

### Скачать данные

```bash
python commands.py download
```

### Подготовить данные

```bash
python commands.py prepare
```

### Обучить модель

```bash
python commands.py train
```

Параметры обучения настраиваются в `configs/`.

## Production Preparation

### Экспорт в ONNX

```bash
python commands.py export_onnx
```

### Конвертация в ONNX через optimum

```bash
python convert_to_onnx.py
```

### Конвертация в TensorRT

Требует NVIDIA TensorRT SDK.

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
./convert_to_tensorrt.sh
```

### Triton Inference Server

**Примечание:** Triton Inference Server не полностью поддерживает сложные LLM модели с KV-cache через ONNX backend. Для LLM рекомендуется использовать специализированные серверы (vLLM, TGI) или прямой inference.

Для запуска Triton через Docker:

```bash
docker pull nvcr.io/nvidia/tritonserver:24.01-py3

docker run --rm -p 9000:8000 -p 9001:8001 -p 9002:8002 \
  -v $(pwd)/triton_model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

Клиент:

```bash
python triton_client.py --text "текст" --triton_url localhost:9000
```

## Infer

### Прямой inference

```bash
python commands.py predict --text "Ты идиот!"

python commands.py predict --input_file example_input.txt

python commands.py predict --input_file example_input.json --output_file results.json
```

Формат входного JSON:

```json
[
  {"text": "Ты идиот!"},
  {"text": "Привет, как дела?"}
]
```

Формат выходного JSON:

```json
[
  {
    "text": "Ты идиот!",
    "toxic": true,
    "labels": [],
    "response": "toxic"
  }
]
```

## Структура проекта

```
toxic_classificator/
├── toxic_classificator/
│   ├── data/
│   ├── training/
│   └── inference/
├── configs/
│   ├── config.yaml
│   ├── model/
│   ├── data/
│   └── training/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── triton_model_repository/
├── commands.py
├── pyproject.toml
└── .pre-commit-config.yaml
```

## Известные ограничения

1. **Triton Inference Server**: ONNX модели LLM с KV-cache имеют 51 вход/выход, что усложняет интеграцию с Triton. Для production рекомендуется использовать специализированные серверы для LLM.

2. **TensorRT**: Требует установленный CUDA Toolkit и TensorRT SDK. Конвертация больших LLM моделей может быть нестабильной.

3. **Качество модели**: Для демонстрации используется минимальный датасет (200 сэмплов). Для реального использования увеличьте `max_samples` в `configs/data/toxic_data.yaml` до 5000-10000.

## Troubleshooting

**Out of Memory:**

```yaml
# configs/training/lora_training.yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
```

**DVC ошибки:**

```bash
dvc status
dvc add data/raw --force
```

**MLflow не запускается:**

```bash
lsof -i :8080
mlflow server --host 127.0.0.1 --port 5000
```
