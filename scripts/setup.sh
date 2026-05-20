#!/bin/bash
# setup.sh - установка окружения на Ubuntu с CUDA

set -e  # Остановка при ошибке

echo "========================================="
echo "Setting up Music Structured Controller"
echo "========================================="

# 1. Проверка CUDA
echo -e "\n[1/5] Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: nvidia-smi not found. Make sure NVIDIA drivers are installed."
fi

# 2. Создание виртуального окружения
echo -e "\n[2/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Установка зависимостей
echo -e "\n[3/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Проверка установки
echo -e "\n[4/5] Verifying installation..."
python -c "
import torch
import tensorflow as tf
import magenta
print(f'PyTorch: {torch.__version__}')
print(f'TensorFlow: {tf.__version__}')
print(f'Magenta: {magenta.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# 5. Создание директорий
echo -e "\n[5/5] Creating directories..."
mkdir -p data/raw_midi data/processed data/checkpoints outputs/{generated,control,swap,interpolation,plots}

echo -e "\n✅ Setup complete!"
echo "Activate environment with: source venv/bin/activate"
echo "Place your MIDI files in: data/raw_midi/"
echo "Run: python run_all.py --step all"