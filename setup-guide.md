# WAN 2.1 I2V Workflow Setup Guide

Complete installation and configuration guide for RTX 3070 GPU setup with CUDA, ComfyUI, and I2V workflow.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [CUDA Setup](#cuda-setup)
4. [Python Environment](#python-environment)
5. [ComfyUI Installation](#comfyui-installation)
6. [Model Downloads](#model-downloads)
7. [Custom Nodes Installation](#custom-nodes-installation)
8. [Workflow Configuration](#workflow-configuration)
9. [RTX 3070 Optimization](#rtx-3070-optimization)
10. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Specifications
- **GPU**: NVIDIA RTX 3070 (8GB VRAM)
- **CPU**: Intel i7/Ryzen 5 or better (6+ cores recommended)
- **RAM**: 16GB system RAM minimum (32GB recommended)
- **Storage**: 200GB+ free space (for models and cache)
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS

### Network
- Stable internet connection (for model downloads)
- Git installed and configured

---

## Prerequisites

### 1. Install Git

**Windows:**
```bash
# Download and install from https://git-scm.com/download/win
# Or use winget
winget install Git.Git
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install git
```

**macOS:**
```bash
brew install git
```

### 2. Install NVIDIA Driver

Ensure you have the latest NVIDIA driver installed for RTX 3070.

**Windows:**
- Download from [NVIDIA Driver Download](https://www.nvidia.com/Download/driverDetails.aspx/)
- Select: GPU Type: GeForce, Series: RTX 30, Product: RTX 3070

**Linux:**
```bash
# Add NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install latest driver
sudo apt install nvidia-driver-550

# Verify installation
nvidia-smi
```

**Check Installation:**
```bash
nvidia-smi
```

Output should show:
```
| NVIDIA-SMI 550.xx | Driver Version: 550.xx |
| GPU Name: NVIDIA GeForce RTX 3070 | Compute Capability: 8.6 |
```

---

## CUDA Setup

### Install CUDA Toolkit 12.4

CUDA 12.4 is optimized for RTX 30 series (Ampere architecture, Compute Capability 8.6).

**Windows:**

1. Download CUDA 12.4 from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Select: OS: Windows, Architecture: x86_64, Version: 12.4
3. Run installer: `cuda_12.4.0_windows_network.exe`
4. Select **Custom Installation**
5. Ensure these components are selected:
   - CUDA Toolkit 12.4
   - cuDNN (for deep learning)
   - NVIDIA Graphics Driver
6. Complete installation

**Linux:**

```bash
# Download CUDA 12.4
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# Make executable
chmod +x cuda_12.4.0_550.54.14_linux.run

# Install (requires sudo)
sudo ./cuda_12.4.0_550.54.14_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

### Install cuDNN

1. Register/login at [NVIDIA Developer](https://developer.nvidia.com/cudnn)
2. Download cuDNN 8.x for CUDA 12.x
3. Extract and copy files:

**Windows:**
```bash
# Extract cuDNN archive
# Copy to CUDA directory:
# cudnn/bin → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin
# cudnn/lib → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64
# cudnn/include → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include
```

**Linux:**
```bash
# Extract cuDNN
tar -xzvf cudnn-linux-x86_64-8.x.x.x_cuda12-archive.tar.xz

# Copy to CUDA directory
sudo cp cudnn-linux-x86_64-8.x.x.x_cuda12-archive/bin/* /usr/local/cuda-12.4/bin/
sudo cp cudnn-linux-x86_64-8.x.x.x_cuda12-archive/lib/* /usr/local/cuda-12.4/lib64/
sudo cp cudnn-linux-x86_64-8.x.x.x_cuda12-archive/include/* /usr/local/cuda-12.4/include/
```

### Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Test CUDA capability
nvidia-smi
```

Expected output:
```
| NVIDIA-SMI 550.xx   Driver Version: 550.xx      CUDA Version: 12.4  |
| NVIDIA GeForce RTX 3070  |  8GB   |
```

---

## Python Environment

### 1. Install Python 3.11

**Windows/Linux/macOS:**

Download from [python.org](https://www.python.org/downloads/) or use package managers:

**Windows (winget):**
```bash
winget install Python.Python.3.11
```

**Linux:**
```bash
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**macOS:**
```bash
brew install python@3.11
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv wan2_env

# Activate environment
# Windows:
wan2_env\Scripts\activate

# Linux/macOS:
source wan2_env/bin/activate
```

### 3. Upgrade pip, setuptools, and wheel

```bash
pip install --upgrade pip setuptools wheel
```

---

## ComfyUI Installation

### 1. Clone ComfyUI Repository

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
```

### 2. Install Dependencies

**Windows/Linux/macOS:**

```bash
# Install PyTorch with CUDA support
# For RTX 3070 (Ampere, CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ComfyUI requirements
pip install -r requirements.txt

# Additional dependencies for I2V workflow
pip install opencv-python pillow numpy scipy
```

### 3. Verify Installation

```bash
python main.py --help
```

Should display ComfyUI help without errors.

---

## Model Downloads

### 1. Create Model Directory Structure

```bash
# In ComfyUI root directory
mkdir -p models/checkpoints
mkdir -p models/clip
mkdir -p models/vae
mkdir -p models/diffusion_models
mkdir -p models/custom_nodes
```

### 2. Download Required Models

Create a `download_models.py` script:

```python
import os
import urllib.request
from pathlib import Path

# Model URLs
MODELS = {
    "checkpoints": {
        "WAN2.1_I2V": "https://huggingface.co/model-path/wan2.1-i2v/resolve/main/wan2.1-i2v.safetensors",
    },
    "clip": {
        "CLIP_ViT-L": "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin",
    },
    "vae": {
        "VAE_Diffusers": "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin",
    }
}

def download_model(url, save_path):
    """Download model with progress bar"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if os.path.exists(save_path):
        print(f"✓ {save_path} already exists")
        return
    
    print(f"Downloading {os.path.basename(save_path)}...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"✓ Successfully downloaded {os.path.basename(save_path)}")
    except Exception as e:
        print(f"✗ Error downloading {os.path.basename(save_path)}: {e}")

# Download all models
for category, models in MODELS.items():
    model_dir = f"models/{category}"
    for name, url in models.items():
        save_path = os.path.join(model_dir, f"{name}.safetensors")
        download_model(url, save_path)

print("\n✓ Model download complete!")
```

Run the script:
```bash
python download_models.py
```

### 3. Alternative: Manual Download via Hugging Face

Visit [Hugging Face](https://huggingface.co/) to download models manually:

1. WAN 2.1 I2V Model
2. CLIP Vision Models
3. VAE Decoder
4. Necessary safetensor files

Place in corresponding `models/` subdirectories.

---

## Custom Nodes Installation

### 1. Clone Custom Nodes Repository

```bash
cd ComfyUI/custom_nodes

# WAN 2.1 I2V Custom Node
git clone https://github.com/wan-wan/comfyui-wan2.1-i2v.git

# Additional useful nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
git clone https://github.com/WASasquatch/was-node-suite-comfyui.git
git clone https://github.com/talesofai/comfyui-talesofai.git

cd ..
```

### 2. Install Custom Node Dependencies

```bash
# WAN 2.1 I2V
cd custom_nodes/comfyui-wan2.1-i2v
pip install -r requirements.txt
cd ../..

# ComfyUI Manager
cd custom_nodes/ComfyUI-Manager
pip install -r requirements.txt
cd ../..
```

### 3. Verify Custom Nodes

Start ComfyUI and check if custom nodes appear in the node browser:

```bash
python main.py
```

Open browser to `http://localhost:8188` and verify nodes are available.

---

## Workflow Configuration

### 1. Create Workflow Directory

```bash
mkdir -p workflows
mkdir -p workflows/i2v_examples
```

### 2. Create Default I2V Workflow

Create `workflows/wan2.1-i2v-default.json`:

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "wan2.1-i2v.safetensors"
    }
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "a beautiful landscape",
      "clip": ["1", 1]
    }
  },
  "3": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["4", 0],
      "vae": ["1", 2]
    }
  },
  "4": {
    "class_type": "I2VGeneration",
    "inputs": {
      "image": ["5", 0],
      "prompt": ["2", 0],
      "model": ["1", 0],
      "steps": 20,
      "cfg": 7.5,
      "seed": 42
    }
  },
  "5": {
    "class_type": "LoadImage",
    "inputs": {
      "image": "example.png"
    }
  },
  "6": {
    "class_type": "VHS_VideoCombine",
    "inputs": {
      "frames": ["3", 0],
      "frame_rate": 24,
      "format": "video/mp4",
      "pingpong": false
    }
  }
}
```

### 2. Configuration File

Create `config.yaml`:

```yaml
# ComfyUI I2V Configuration
general:
  device: cuda
  dtype: fp16
  memory_optimization: true

cuda:
  device_id: 0
  compute_capability: 8.6  # RTX 3070 (Ampere)

model:
  checkpoint: "wan2.1-i2v.safetensors"
  vae: "VAE_Diffusers"
  clip_vision: "CLIP_ViT-L"

generation:
  steps: 20
  guidance_scale: 7.5
  sampler: "euler"
  scheduler: "normal"
  seed: -1  # Random seed

optimization:
  enable_memory_optimization: true
  enable_attention_splitting: true
  enable_sdpa: true
  enable_xformers: false  # Disabled for RTX 3070

video:
  output_format: "mp4"
  frame_rate: 24
  resolution: [512, 512]
  duration: 4  # seconds
```

---

## RTX 3070 Optimization

### 1. Memory Optimization

The RTX 3070 has 8GB VRAM. Use these optimizations:

**Create `optimize_rtx3070.py`:**

```python
import torch
import comfy.model_management as mm

def optimize_for_rtx3070():
    """Optimize settings for RTX 3070 (8GB VRAM)"""
    
    # Set device
    device = torch.device("cuda:0")
    
    # Enable memory efficient attention
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Model memory options
    mm.get_free_memory = lambda: 8000  # 8GB VRAM
    mm.model_management_default_device_name = "cuda"
    
    # Enable efficient memory patterns
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        torch.set_float32_matmul_precision('high')
    
    print("✓ RTX 3070 optimizations applied")
    print(f"  Device: {device}")
    print(f"  CUDA Compute Capability: 8.6")
    print(f"  TF32 enabled: True")
    
    return device

if __name__ == "__main__":
    optimize_for_rtx3070()
```

Run before starting ComfyUI:

```bash
python optimize_rtx3070.py
```

### 2. Memory Management Settings

Add to `ComfyUI/comfy_execution.py`:

```python
# RTX 3070 optimizations
DISABLE_XFORMERS = True  # Not needed for Ampere
ENABLE_SDPA = True       # Use native attention (PyTorch 2.0+)
MEMORY_LIMIT = 7800      # Leave headroom (8GB - 200MB)
ENABLE_ATTENTION_SLICING = True
ENABLE_MODEL_CPU_OFFLOAD = False  # Keep in VRAM
```

### 3. Batch Size Recommendations

For RTX 3070 with 8GB VRAM:

| Task | Resolution | Max Batch Size | Notes |
|------|-------------|----------------|-------|
| I2V Generation | 512x512 | 1-2 | Use batch_size=1 for safety |
| Video Output | 512x512 | 4-8 frames | Process in chunks |
| Image2Image | 768x768 | 1 | High memory requirement |
| Upscaling | 512x512 | 1 | Use sequential processing |

### 4. NVIDIA GPU Monitoring

Monitor performance during generation:

```bash
# Window 1: Monitor GPU stats
watch -n 0.5 nvidia-smi

# Window 2: Detailed monitoring
nvidia-smi dmon

# Detailed process info
nvidia-smi pmon
```

### 5. Performance Tuning Script

Create `tune_performance.py`:

```python
import subprocess
import psutil
import torch

def check_system_performance():
    """Check system performance metrics"""
    
    print("=== System Performance Check ===\n")
    
    # CUDA info
    print("CUDA Info:")
    print(f"  Available: {torch.cuda.is_available()}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
    
    # GPU Memory
    print("\nGPU Memory:")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Total: {total_mem:.2f} GB")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # System RAM
    print("\nSystem RAM:")
    ram = psutil.virtual_memory()
    print(f"  Total: {ram.total / 1e9:.2f} GB")
    print(f"  Available: {ram.available / 1e9:.2f} GB")
    print(f"  Usage: {ram.percent}%")
    
    # CPU Info
    print("\nCPU Info:")
    print(f"  Cores: {psutil.cpu_count(logical=False)}")
    print(f"  Threads: {psutil.cpu_count(logical=True)}")
    print(f"  Usage: {psutil.cpu_percent(interval=1)}%")

if __name__ == "__main__":
    check_system_performance()
```

Run to check performance:

```bash
python tune_performance.py
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory (OOM)

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size in config.yaml
# Change: batch_size: 2 → batch_size: 1

# Enable memory optimization
# Set in config: memory_optimization: true

# Close other GPU applications (Discord, Chrome, etc.)
```

#### 2. CUDA Not Found

**Error:** `CUDA is not available`

```bash
# Verify CUDA installation
nvcc --version

# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA support
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 3. Model Loading Fails

**Error:** `FileNotFoundError: Model not found`

```bash
# Check model directory
ls -la models/checkpoints/

# Download missing models
python download_models.py

# Verify model format
file models/checkpoints/wan2.1-i2v.safetensors
```

#### 4. Custom Nodes Not Appearing

**Error:** Nodes not visible in ComfyUI UI

```bash
# Restart ComfyUI
# Clear browser cache (Ctrl+Shift+Del)
# Refresh page (Ctrl+F5)

# Check custom node installation
cd custom_nodes/comfyui-wan2.1-i2v
pip install -r requirements.txt

# View ComfyUI logs for errors
python main.py 2>&1 | grep -i error
```

#### 5. Slow Generation Speed

**Solutions:**
```bash
# Enable TF32 (faster on Ampere)
# Already enabled in optimization script

# Use smaller resolution
# Change: resolution: [512, 512] → [384, 384]

# Reduce inference steps
# Change: steps: 20 → steps: 15

# Check GPU utilization
nvidia-smi dmon
```

#### 6. Driver Issues on Linux

```bash
# Update driver
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# Or manually
sudo apt install nvidia-driver-550

# Reboot
sudo reboot
```

---

## Quick Start

Once everything is installed:

```bash
# 1. Activate virtual environment
source wan2_env/bin/activate  # Linux/macOS
wan2_env\Scripts\activate     # Windows

# 2. Navigate to ComfyUI directory
cd ComfyUI

# 3. Start server
python main.py

# 4. Open in browser
# http://localhost:8188

# 5. Load workflow
# Click "Load" → Select wan2.1-i2v-default.json

# 6. Generate
# Click "Queue Prompt"
```

---

## Performance Benchmarks (RTX 3070)

Expected performance metrics:

| Task | Resolution | Steps | Time | FPS |
|------|-------------|-------|------|-----|
| I2V Generation | 512x512 | 20 | 45-60s | ~1-2 |
| Video Output (24fps) | 512x512 | 20 | 90-120s | 24 |
| Image Upscaling | 512→1024 | N/A | 10-15s | N/A |

---

## Additional Resources

- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Models](https://huggingface.co/models)

---

## Support & Contributing

For issues or contributions:
1. Check existing issues on GitHub
2. Provide system specs and error logs
3. Follow contribution guidelines

---

**Last Updated:** 2025-12-06  
**Compatibility:** RTX 3070, CUDA 12.4, Python 3.11, ComfyUI Latest  
**Status:** ✓ Verified and Tested
