# Классификатор токсичных комментариев

**Автор:** Сыров Артём Андреевич

## Описание

Модель для классификации токсичных сообщений на русском и английском языках. Использует Qwen2.5-0.5B-Instruct с LoRA fine-tuning.

**Датасеты:**
- Jigsaw Multilingual Toxic Comment Classification
- Russian Language Toxic Comments

**Технологии:**
- PyTorch Lightning
- Hydra (конфигурация)
- MLflow (логирование)
- DVC (данные)
- PEFT/LoRA (fine-tuning)
- ONNX + TensorRT (оптимизация)
- Triton Inference Server (inference)

## Setup

### 1. Установка зависимостей

```bash
# Установить uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Создать виртуальное окружение и установить зависимости
uv venv
source .venv/bin/activate  # Linux/Mac
# или .venv\Scripts\activate  # Windows

uv pip install -e .
```

### 2. Настройка Kaggle API

```bash
# Получить API ключ: https://www.kaggle.com/settings -> Create New Token
# Скопировать kaggle.json в ~/.kaggle/
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Pre-commit хуки

```bash
pre-commit install
pre-commit run -a
```

### 4. MLflow сервер

```bash
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

## Train

### 1. Скачать данные

```bash
python commands.py download
```

### 2. Подготовить данные

```bash
python commands.py prepare
```

### 3. Обучить модель

```bash
python commands.py train
```

Обучение займет ~20-30 минут на RTX 3060 12GB.

**Параметры обучения** (настраиваются в `configs/`):
- Модель: Qwen2.5-0.5B-Instruct
- LoRA rank: 8
- Batch size: 8
- Epochs: 3
- Samples: 200 (demo mode)

## Production Preparation

### 1. Экспорт в ONNX

```bash
python commands.py export_onnx
```

Создаст `triton_model_repository/toxic_classificator/1/model.onnx`

### 2. Конвертация в ONNX (опционально)

```bash
python convert_to_onnx.py
```

Создаст `triton_model_repository/toxic_classificator/1/onnx_model/`

### 3. Конвертация в TensorRT (опционально, требует NVIDIA TensorRT)

```bash
# Установка TensorRT: https://developer.nvidia.com/tensorrt
./convert_to_tensorrt.sh
```

Создаст `triton_model_repository/toxic_classificator/1/model.plan`

**Примечание:** TensorRT требует NVIDIA GPU и установленный TensorRT SDK.

### 3. Запуск Triton Server

**Требования:**
- Docker
- NVIDIA Container Toolkit (для GPU)

**Установка Triton через Docker:**
```bash
docker pull nvcr.io/nvidia/tritonserver:24.01-py3

# Запуск
docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton_model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

Или используйте скрипт (требует установленный `tritonserver`):
```bash
./start_triton_server.sh
```

Сервер будет доступен на `http://localhost:8000`

**Примечание:** Для учебного проекта можно использовать прямой inference через `predict.py` без Triton.

## Infer

### Рекомендуемый способ: Прямой inference (без Triton)

**Один текст:**
```bash
python commands.py predict --text "Ты идиот!"
```

**Файл (построчно):**
```bash
python commands.py predict --input_file example_input.txt
```

**JSON файл:**
```bash
python commands.py predict --input_file example_input.json --output_file results.json
```

**Формат входного JSON:**
```json
[
  {"text": "Ты идиот!"},
  {"text": "Привет, как дела?"}
]
```

**Формат выходного JSON:**
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

### Альтернатива: Inference через Triton (требует установку Triton Server)

**1. Запустить Triton Server через Docker:**
```bash
docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton_model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

**2. Использовать Python клиент:**
```bash
# Один текст
python triton_client.py --text "Ты идиот!"

# Файл
python triton_client.py --input_file example_input.txt --output_file triton_results.json
```

**Примечание:** Triton inference реализован для демонстрации production-ready подхода. Для локального тестирования используйте `python commands.py predict`.

## Структура проекта

```
toxic_classificator/
├── toxic_classificator/       # Основной пакет
│   ├── data/                  # Скрипты работы с данными
│   ├── training/              # Обучение и оценка
│   └── inference/             # Inference и экспорт
├── configs/                   # Hydra конфигурации
│   ├── config.yaml
│   ├── model/
│   ├── data/
│   └── training/
├── data/                      # Данные (в DVC)
│   ├── raw/
│   └── processed/
├── models/                    # Обученные модели
├── triton_model_repository/   # Модели для Triton
├── commands.py                # CLI (Fire)
├── pyproject.toml             # Зависимости (uv)
└── .pre-commit-config.yaml    # Pre-commit хуки
```

## Troubleshooting

**Ошибка при predict (JSON decode):**
```bash
# Модель повреждена, переобучите
rm -rf models/finetuned/final
python commands.py train
```

**Out of Memory:**
```yaml
# В configs/training/lora_training.yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
```

**DVC ошибки:**
```bash
dvc status
dvc add data/raw --force
```
