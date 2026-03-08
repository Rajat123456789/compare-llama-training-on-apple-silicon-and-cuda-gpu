#!/bin/bash
# Setup script for RunPod/Cloud GPU environment
# NVIDIA RTX A4500 (20GB) - Perfect for LoRA training

echo "==================================================================="
echo "LoRA Fine-Tuning Setup for Cloud GPU (RunPod/AWS/GCP/Azure)"
echo "==================================================================="

# 1. System info
echo -e "\n📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 2. Update and install system dependencies
echo -e "\n📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y git wget curl vim

# 3. Install Python 3.10+ if needed
python3 --version || apt-get install -y python3 python3-pip python3-venv

# 4. Clone project (if not already present)
if [ ! -d "llama-lora" ]; then
    echo -e "\n📥 Cloning project..."
    # For now, create directory (user will upload files)
    mkdir -p llama-lora
    cd llama-lora
else
    cd llama-lora
fi

# 5. Create virtual environment
echo -e "\n🐍 Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

# 6. Upgrade pip
pip install --upgrade pip -q

# 7. Install PyTorch with CUDA support (CUDA 12.1 compatible)
echo -e "\n🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 8. Verify CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo -e "\n✅ Setup complete!"
echo -e "\n📝 Next steps:"
echo "1. Upload project files to ~/llama-lora/"
echo "2. pip install -r requirements.txt"
echo "3. huggingface-cli login"
echo "4. python train.py"
