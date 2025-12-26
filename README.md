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

### 2. Конвертация в TensorRT

```bash
./convert_to_tensorrt.sh
```

Создаст `triton_model_repository/toxic_classificator/1/model.plan`

### 3. Запуск Triton Server

```bash
./start_triton_server.sh
```

Сервер будет доступен на `http://localhost:8000`

## Infer

### Прямой inference (без Triton)

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

### Inference через Triton

```bash
# Запустить Triton (если еще не запущен)
./start_triton_server.sh

# Отправить запрос
curl -X POST http://localhost:8000/v2/models/toxic_classificator/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "input_ids",
        "shape": [1, 10],
        "datatype": "INT64",
        "data": [...]
      }
    ]
  }'
```

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
