# Классификатор токсичности в диалоге

**Автор:** Сыров Артём Андреевич

## Описание проекта

Проект разработан для автоматической классификации токсичных сообщений в диалогах с использованием модели Qwen2.5-1.5B-Instruct и fine-tuning через LoRA (QLoRA с 4-bit квантизацией).

### Цель

Разработать модель, которая на сообщениях из переписки будет определять, токсичны ли сообщения от собеседника, а также тип токсичности/агрессии.

### Формат данных

- **Вход**: строка с текстом сообщения
- **Выход**: JSON с полями `toxic` (boolean) и `labels` (список типов токсичности)

Пример:

```json
{
  "toxic": true,
  "labels": ["insult", "obscene"]
}
```

### Метрики

- **F1-score для токсичности**: цель ≥0.85
- **F1-score для типов**: цель ≥0.70

### Датасеты

- Jigsaw Multilingual Toxic Comment Classification
- Russian Language Toxic Comments
- Объем: до 40k samples
- Разделение: 70% train / 15% val / 15% test (стратифицированное)

### Технологии

- **PyTorch Lightning** - фреймворк для обучения
- **Hydra** - управление конфигурациями
- **MLflow** - логирование экспериментов
- **DVC** - версионирование данных
- **PEFT (LoRA)** - эффективный fine-tuning
- **4-bit quantization** - экономия памяти
- **ONNX + TensorRT** - оптимизация для продакшена
- **Triton Inference Server** - сервинг модели

## Setup

### Требования

- Python ≥3.10
- CUDA-compatible GPU (рекомендуется 8GB+ VRAM)
- ~15GB свободного места на диске

### Установка окружения

```bash
# Установить uv (если еще не установлен)
python3 -m pip install uv

# Создать виртуальное окружение
uv venv
source .venv/bin/activate  # Linux/Mac
# или .venv\Scripts\activate  # Windows

# Установить зависимости
uv pip install -e .

# Установить dev зависимости
uv pip install -e ".[dev]"

# Настроить pre-commit хуки
pre-commit install
```

### Настройка Kaggle API

Для скачивания датасетов необходимы Kaggle API credentials:

```bash
# 1. Получить API ключ с https://www.kaggle.com/account
# 2. Установить
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Настройка DVC

DVC используется для версионирования данных и моделей:

```bash
# Инициализировать DVC (уже сделано)
dvc init

# Настроить remote (опционально, для команды)
# По умолчанию используется локальное хранилище /tmp/dvc-storage
dvc remote add -d myremote /path/to/storage
```

### Настройка MLflow

MLflow используется для логирования экспериментов:

```bash
# Запустить MLflow tracking server
mlflow server --host 127.0.0.1 --port 8080

# Сервер должен быть запущен перед обучением
# UI доступен по адресу: http://127.0.0.1:8080
```

## Train

### Полный пайплайн обучения

```bash
# 1. Скачать данные из Kaggle
python commands.py download

# 2. Подготовить данные
python commands.py prepare

# 3. Обучить модель
python commands.py train

# 4. Оценить модель
python commands.py evaluate
```

### Запуск с кастомными конфигами

```bash
# Изменить параметры обучения
python commands.py train --config-path configs --config-name config \
    training.num_train_epochs=5 \
    training.per_device_train_batch_size=2

# Использовать другую модель
python commands.py train model=qwen

# Изменить размер данных
python commands.py train data.max_samples=10000
```

### Конфигурация

Все гиперпараметры находятся в `configs/`:

- `config.yaml` - основная конфигурация
- `model/qwen.yaml` - параметры модели
- `data/toxic_data.yaml` - параметры данных
- `training/lora_training.yaml` - параметры обучения

Основные параметры:

```yaml
# Модель
model:
  name: Qwen/Qwen2.5-1.5B-Instruct
  max_length: 512
  load_in_4bit: true

# LoRA
training.lora:
  r: 16
  lora_alpha: 32

# Обучение
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2.0e-4
```

## Production preparation

### Экспорт в ONNX

```bash
# Экспортировать обученную модель в ONNX
python commands.py export_onnx \
    --checkpoint models/finetuned/final \
    --output-path models/model.onnx
```

### Конвертация в TensorRT

```bash
# Конвертировать ONNX модель в TensorRT
./convert_to_tensorrt.sh models/model.onnx models/model.trt
```

Требуется установленный TensorRT SDK.

### Подготовка для Triton

```bash
# 1. Скопировать ONNX модель в Triton model repository
cp models/model.onnx triton_model_repository/toxic_classificator/1/model.onnx

# 2. Конфигурация уже создана в triton_model_repository/toxic_classificator/config.pbtxt
```

## Infer

### Локальный инференс

```bash
# Одно сообщение
python commands.py predict --text "Ваш текст здесь"

# Из файла (один текст на строку)
python commands.py predict \
    --input-file input.txt \
    --output-file predictions.json

# С кастомным чекпоинтом
python commands.py predict \
    --text "Текст" \
    --checkpoint models/finetuned/final
```

### Triton Inference Server

```bash
# 1. Запустить Triton server
./start_triton_server.sh

# 2. Сервер будет доступен по адресам:
#    HTTP: http://localhost:8000
#    gRPC: localhost:8001
#    Metrics: http://localhost:8002/metrics
```

Пример запроса к Triton:

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

# Подготовить входные данные
input_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
attention_mask = np.array([[1, 1, 1, 1]], dtype=np.int64)

inputs = [
    httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
    httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
]
inputs[0].set_data_from_numpy(input_ids)
inputs[1].set_data_from_numpy(attention_mask)

outputs = [httpclient.InferRequestedOutput("logits")]

response = client.infer("toxic_classificator", inputs, outputs=outputs)
logits = response.as_numpy("logits")
```

### Формат входных данных

Текстовый файл с одним сообщением на строку:

```
Привет! Как дела?
Ты идиот!
Спасибо за помощь
```

### Формат выходных данных

JSON файл с предсказаниями:

```json
[
  {
    "text": "Привет! Как дела?",
    "toxic": false,
    "labels": []
  },
  {
    "text": "Ты идиот!",
    "toxic": true,
    "labels": ["insult"]
  }
]
```

## Code Quality

### Pre-commit хуки

Проект использует pre-commit для проверки качества кода:

```bash
# Установить хуки
pre-commit install

# Запустить на всех файлах
pre-commit run -a
```

Используемые хуки:

- `black` - форматирование Python кода
- `isort` - сортировка импортов
- `flake8` - линтинг
- `prettier` - форматирование YAML/JSON/Markdown
- Стандартные хуки pre-commit (trailing whitespace, EOF, etc.)

### Структура проекта

```
toxic-classificator/
├── configs/                    # Hydra конфигурации
│   ├── config.yaml
│   ├── model/
│   ├── data/
│   └── training/
├── toxic_classificator/        # Основной пакет
│   ├── data/
│   │   ├── download_data.py
│   │   └── prepare_data.py
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── inference/
│   │   ├── predict.py
│   │   └── export_onnx.py
│   └── models/
├── data/                       # Данные (под DVC)
│   ├── raw/
│   └── processed/
├── models/                     # Модели (под DVC)
│   ├── finetuned/
│   └── cache/
├── triton_model_repository/    # Triton модели
├── plots/                      # Графики и результаты
├── logs/                       # Логи обучения
├── commands.py                 # Главная точка входа (Fire CLI)
├── pyproject.toml              # Зависимости (uv)
├── .pre-commit-config.yaml     # Pre-commit конфигурация
├── .dvc/                       # DVC конфигурация
└── README.md                   # Этот файл
```

## Troubleshooting

### Out of Memory

Если не хватает GPU памяти:

```yaml
# В configs/training/lora_training.yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
```

### DVC ошибки

```bash
# Проверить статус DVC
dvc status

# Принудительно добавить данные
dvc add data/raw --force

# Проверить remote
dvc remote list
```

### MLflow не запускается

```bash
# Проверить порт
lsof -i :8080

# Запустить на другом порту
mlflow server --host 127.0.0.1 --port 5000

# Обновить в configs/config.yaml
mlflow.tracking_uri: http://127.0.0.1:5000
```

### Triton ошибки

```bash
# Проверить логи
docker logs triton-server

# Проверить модель
curl localhost:8000/v2/models/toxic_classificator
```
