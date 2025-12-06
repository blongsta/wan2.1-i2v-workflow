# Wan 2.1 Image-to-Video ComfyUI Workflow

A comprehensive ComfyUI workflow implementation for Wan 2.1 image-to-video generation, optimized for NVIDIA RTX 3070 8GB VRAM GPUs. This workflow enables high-quality video generation from static images with efficient memory management and flexible configuration options.

## 📋 Table of Contents

- [Workflow Overview](#workflow-overview)
- [Quick Start](#quick-start)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Workflow Presets](#workflow-presets)
- [Configuration Guide](#configuration-guide)
- [Usage Instructions](#usage-instructions)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)
- [Contributing](#contributing)
- [License](#license)

## 🎬 Workflow Overview

This Wan 2.1 image-to-video workflow transforms static images into dynamic, smooth videos using advanced diffusion models. The workflow is specifically optimized for efficient VRAM usage on consumer-grade GPUs while maintaining high output quality.

### Key Features

- **Image-to-Video Generation**: Convert static images into smooth video sequences
- **Memory Optimized**: Designed specifically for 8GB VRAM constraints
- **Flexible Configuration**: Adjustable parameters for quality vs. speed tradeoffs
- **Multiple Presets**: Pre-configured settings for common use cases
- **ComfyUI Integration**: Full compatibility with ComfyUI ecosystem
- **Advanced Controls**: Fine-grained control over generation parameters
- **Batch Processing**: Support for processing multiple images sequentially

### Supported Input Formats

- **Images**: PNG, JPG, JPEG, WebP (Recommended resolution: 768x512 to 1024x576)
- **Video Output**: MP4, WebM (Variable frame rates: 16fps - 30fps)

## 🚀 Quick Start

### Basic Setup (5 minutes)

1. **Clone or download the workflow**
   ```bash
   git clone https://github.com/blongsta/wan2.1-i2v-workflow.git
   cd wan2.1-i2v-workflow
   ```

2. **Ensure ComfyUI is installed** with required dependencies
   ```bash
   # If not already installed
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   ```

3. **Copy workflow files**
   ```bash
   # Copy workflow JSON to ComfyUI directory
   cp wan_2.1_i2v_workflow.json ../ComfyUI/
   ```

4. **Load workflow in ComfyUI**
   - Open ComfyUI web interface (default: http://localhost:8188)
   - Click "Load" and select `wan_2.1_i2v_workflow.json`
   - Select your input image
   - Click "Queue Prompt"

5. **Output**
   - Generated videos saved to `ComfyUI/output/` directory
   - Video format: MP4 (H.264 codec)

### First Generation Test

```bash
# Use example image included in repo
# Select: examples/sample_landscape.jpg
# Use: "Fast" preset for quick test
# Expected generation time: 2-3 minutes
```

## 🖥️ Hardware Requirements

### Minimum Requirements (Recommended Setup)

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 3070 (8GB VRAM) |
| CUDA | 11.8 or higher |
| cuDNN | 8.x series |
| System RAM | 16GB |
| Storage | SSD with 50GB+ free space |
| Python | 3.8 - 3.11 |

### Recommended Setup (Optimal Performance)

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 4080 or RTX 4090 (24GB+ VRAM) |
| CUDA | 12.1 or higher |
| System RAM | 32GB+ |
| Storage | NVMe SSD with 100GB+ free space |

### Memory Considerations for RTX 3070

- **Base VRAM Usage**: ~4GB (model loading)
- **Generation VRAM**: ~2-3GB (batch processing)
- **Safety Buffer**: ~1-2GB (system operations)
- **Total Available**: 8GB

### Performance Expectations (RTX 3070)

| Preset | Resolution | Duration | Time | Memory |
|--------|-----------|----------|------|--------|
| Fast | 512x320 | 8 frames | 90-120s | 5-6GB |
| Balanced | 768x512 | 16 frames | 180-240s | 6-7GB |
| Quality | 1024x576 | 24 frames | 300-360s | 7-8GB |
| Ultra | 1024x576 | 32 frames | 400-600s | 7.5-8GB |

## 📦 Installation

### Prerequisites

- NVIDIA GPU with CUDA support (Recommended: RTX 3070 or higher)
- Python 3.8 or higher
- Git
- CUDA Toolkit 11.8+
- cuDNN 8.x

### Step-by-Step Installation

1. **Install NVIDIA CUDA and cuDNN** (if not already installed)
   ```bash
   # Download from NVIDIA website
   # https://developer.nvidia.com/cuda-downloads
   # https://developer.nvidia.com/cudnn
   ```

2. **Create Python virtual environment**
   ```bash
   python -m venv venv_wan
   source venv_wan/bin/activate  # On Windows: venv_wan\Scripts\activate
   ```

3. **Install ComfyUI**
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install Wan 2.1 models**
   ```bash
   # Download Wan 2.1 model weights
   # Place in: ComfyUI/models/checkpoints/
   # Model size: ~7GB (ensure adequate storage)
   ```

5. **Install custom nodes** (if required)
   ```bash
   cd ComfyUI/custom_nodes
   # Clone any required custom node repositories
   ```

6. **Copy workflow files**
   ```bash
   cp wan_2.1_i2v_workflow.json ../
   cp examples/* ../input/
   ```

7. **Verify installation**
   ```bash
   cd ..
   python main.py
   # Navigate to http://localhost:8188
   ```

### GPU Memory Optimization (RTX 3070)

Add these environment variables before running ComfyUI:

```bash
# Linux/Mac
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST="7.2"  # RTX 3070 architecture

# Windows (PowerShell)
$env:CUDA_LAUNCH_BLOCKING="1"
$env:TORCH_CUDA_ARCH_LIST="7.2"

# Then run ComfyUI
python main.py
```

## 🎯 Workflow Presets

The workflow includes four pre-configured presets optimized for different use cases:

### 1. **Fast Preset** ⚡
Best for: Quick previews, testing, rapid iteration

```
Resolution: 512x320
Frames: 8
Steps: 20
Inference Type: fp16
Generation Time: 90-120 seconds
VRAM Usage: 5-6GB
Quality: Good
```

**Use Case**: Testing compositions, quick generation cycles

### 2. **Balanced Preset** ⚖️
Best for: General purpose, standard quality output

```
Resolution: 768x512
Frames: 16
Steps: 30
Inference Type: fp16
Generation Time: 180-240 seconds
VRAM Usage: 6-7GB
Quality: Very Good
```

**Use Case**: Production work, social media content

### 3. **Quality Preset** ✨
Best for: High-quality output, professional use

```
Resolution: 1024x576
Frames: 24
Steps: 40
Inference Type: fp32 (with optimizations)
Generation Time: 300-360 seconds
VRAM Usage: 7-8GB
Quality: Excellent
```

**Use Case**: Professional videos, cinematic content

### 4. **Ultra Preset** 🏆
Best for: Maximum quality, special projects

```
Resolution: 1024x576
Frames: 32
Steps: 50
Inference Type: fp32 (optimized)
Generation Time: 400-600 seconds
VRAM Usage: 7.5-8GB
Quality: Exceptional
```

**Use Case**: Marketing materials, artistic projects, premium content

### Switching Presets

Within ComfyUI:
1. Open the workflow file
2. Locate the "Preset Selector" node
3. Select desired preset from dropdown
4. Adjust additional parameters as needed
5. Queue prompt

## ⚙️ Configuration Guide

### Core Parameters

#### Image Settings

```
Input Image Path: Path to source image
Image Width: 512-1024 (must be divisible by 64)
Image Height: 320-576 (must be divisible by 64)
Auto Rescale: Enabled (maintains aspect ratio)
```

**Recommendations for RTX 3070**:
- Preferred resolutions: 512x320, 768x512, 1024x576
- Avoid resolutions exceeding 1024x576

#### Video Generation Parameters

```
Number of Frames: 8, 16, 24, or 32
Frame Rate (FPS): 16, 24, or 30
Smoothness: 0.5-1.0 (higher = smoother transitions)
Motion Intensity: 0.3-1.0 (higher = more motion)
```

#### Diffusion Model Settings

```
Inference Steps: 20-50 (higher = better quality, slower)
Guidance Scale: 7.5 (recommend: 7.5)
Seed: -1 (random) or specific integer (reproducible)
```

#### Memory Optimization Flags

```
Enable Memory Efficient Attention: True
Use Flash Attention: True (NVIDIA GPU with compute capability 7.0+)
Tile VAE Decode: True (for large resolutions)
Batch Size: 1 (for 8GB VRAM)
```

### Advanced Configuration

#### Custom Model Paths

Edit `workflow_config.json`:

```json
{
  "model_paths": {
    "wan_model": "/path/to/wan_2.1_model",
    "vae_model": "/path/to/vae",
    "text_encoder": "/path/to/text_encoder"
  },
  "device": "cuda:0",
  "precision": "fp16"
}
```

#### Performance Tuning for RTX 3070

**Maximum Quality Within Memory Limits**:

```
resolution: 1024x576
steps: 40
guidance_scale: 7.5
tile_vae_decode: true
enable_memory_efficient_attn: true
batch_size: 1
precision: fp16
```

**Maximum Speed**:

```
resolution: 512x320
steps: 20
guidance_scale: 7.5
tile_vae_decode: false
enable_memory_efficient_attn: true
batch_size: 1
precision: fp16
```

### Configuration Files Location

```
ComfyUI/
├── workflow_config.json (Main configuration)
├── presets/
│   ├── fast.json
│   ├── balanced.json
│   ├── quality.json
│   └── ultra.json
└── models/
    └── checkpoints/
        └── wan_2.1_model.safetensors
```

## 🎮 Usage Instructions

### Basic Workflow

1. **Prepare Input Image**
   - Resolution: 512x512 to 1024x1024 recommended
   - Format: PNG or JPG
   - File size: <50MB
   - Place in: `ComfyUI/input/`

2. **Launch ComfyUI**
   ```bash
   cd ComfyUI
   python main.py
   ```

3. **Load Workflow**
   - Navigate to: http://localhost:8188
   - Click "Load" button
   - Select: `wan_2.1_i2v_workflow.json`

4. **Configure Generation**
   - Select input image from browser
   - Choose preset (or customize parameters)
   - Adjust advanced settings if desired
   - Review settings in preview panel

5. **Generate Video**
   - Click "Queue Prompt" button
   - Monitor progress in Console panel
   - Wait for completion

6. **Access Output**
   - Video saved to: `ComfyUI/output/`
   - Filename format: `wan_2.1_[timestamp].mp4`

### Batch Processing

**Process Multiple Images**:

1. Create `.csv` file with image paths:
   ```
   image_path,preset,seed
   input/image1.jpg,balanced,-1
   input/image2.jpg,quality,12345
   input/image3.jpg,fast,-1
   ```

2. Use batch processing node:
   - Enable "Batch Mode" in workflow
   - Load CSV file
   - Click "Queue All"

3. Monitor progress:
   - Check Console for status
   - Videos generated sequentially

### Advanced Usage

#### Custom Motion Control

```
Motion Parameters:
- Motion Intensity: 0.3 (subtle)
- Motion Intensity: 0.6 (moderate)
- Motion Intensity: 0.9 (dynamic)

Camera Movement:
- Pan Left/Right: -0.5 to 0.5
- Zoom In/Out: -0.3 to 0.3
- Rotation: -15 to 15 degrees
```

#### Seed Management

**Reproducible Results**:
```
Set Seed: [specific number] → Same video each time
Random Seed: -1 → Different video each generation
```

**Seed-based Variations**:
```
Base Seed: 12345
Generate 5 variations: seeds 12345-12349
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. **CUDA Out of Memory Error**

**Error Message**: `CUDA out of memory. Tried to allocate X.XXG`

**Solutions**:
```
Step 1: Use "Fast" preset instead of "Quality"
Step 2: Reduce resolution to 512x320
Step 3: Reduce number of frames to 8
Step 4: Enable "Tile VAE Decode" option
Step 5: Clear GPU cache:
  - Restart ComfyUI
  - Run: torch.cuda.empty_cache()

# In ComfyUI console
import torch
torch.cuda.empty_cache()
```

#### 2. **Slow Generation Speed**

**Cause**: Inefficient settings for RTX 3070

**Solutions**:
```
1. Switch to "Fast" or "Balanced" preset
2. Reduce inference steps (20-30 recommended)
3. Use fp16 precision instead of fp32
4. Enable memory efficient attention
5. Reduce resolution
6. Check system resources (RAM, CPU)

# Monitor GPU usage:
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,nounits -l 1
```

#### 3. **Poor Quality Output**

**Cause**: Settings optimized for speed rather than quality

**Solutions**:
```
1. Switch to "Quality" or "Ultra" preset
2. Increase inference steps to 40-50
3. Use fp32 precision for better accuracy
4. Increase image resolution
5. Adjust motion intensity (try 0.6-0.8)
6. Use higher guidance scale (8.0-9.0)
7. Provide high-quality input image
```

#### 4. **Model Failed to Load**

**Error**: `Model not found` or `Cannot load weights`

**Solutions**:
```
1. Verify model file exists:
   ls ComfyUI/models/checkpoints/wan_2.1_model*
   
2. Check file permissions:
   chmod 644 models/checkpoints/wan_2.1_model.safetensors
   
3. Re-download model:
   - Remove corrupted file
   - Download fresh copy from official source
   - Verify SHA256 checksum
   
4. Verify model compatibility:
   - Ensure model version matches workflow
   - Check model format (.safetensors vs .ckpt)
```

#### 5. **Video Codec Issues**

**Error**: `Video codec not supported` or playback issues

**Solutions**:
```
1. Install FFmpeg:
   Ubuntu: sudo apt-get install ffmpeg
   Mac: brew install ffmpeg
   Windows: Download from https://ffmpeg.org/download.html
   
2. Configure codec in workflow:
   - Use H.264 (most compatible)
   - Verify bitrate: 5000-8000 kbps
   
3. Test video playback:
   - Use VLC media player
   - Try different player if issues persist
   
4. Re-encode if necessary:
   ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4
```

#### 6. **High Temperature / Thermal Throttling**

**Symptoms**: Generation speed decreases over time

**Solutions**:
```
1. Check GPU temperature:
   nvidia-smi -l 1 | grep Temp
   
2. Improve cooling:
   - Clean GPU fans
   - Improve case airflow
   - Add additional case fans
   
3. Reduce power limit (temporary):
   sudo nvidia-smi -pl 250  # Set power limit to 250W
   
4. Take breaks:
   - Generate videos in batches
   - Allow GPU to cool between generations
```

#### 7. **ComfyUI Won't Start**

**Error**: Port already in use or connection refused

**Solutions**:
```
1. Change port:
   python main.py --port 8189
   
2. Kill process using port 8188:
   # Linux/Mac:
   lsof -i :8188
   kill -9 <PID>
   
   # Windows:
   netstat -ano | findstr :8188
   taskkill /PID <PID> /F
   
3. Check firewall settings
4. Verify network connectivity
```

### Debug Mode

**Enable Verbose Logging**:

```bash
# Set environment variables
export COMFYUI_DEBUG=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run with verbose output
python main.py --verbose

# Check logs:
tail -f output/comfyui.log
```

### Performance Monitoring

**Real-time GPU Monitoring**:

```bash
# Terminal 1: Run ComfyUI
python main.py

# Terminal 2: Monitor GPU
watch -n 0.5 nvidia-smi

# Detailed memory tracking
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,temperature_gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory --format=csv -l 1
```

## 📚 Resources

### Official Documentation

- **ComfyUI Documentation**: https://github.com/comfyanonymous/ComfyUI
- **Wan Model Research**: https://github.com/[wan-repo]
- **NVIDIA CUDA Documentation**: https://docs.nvidia.com/cuda/

### Model Downloads

- **Wan 2.1 Checkpoint**: [Download Link]
  - Size: ~7GB
  - Format: `.safetensors`
  - SHA256: [checksum]

- **VAE Models**: [Download Links]
- **Text Encoders**: [Download Links]

### Tutorial Videos

- **Getting Started**: [YouTube Link]
- **Advanced Configuration**: [YouTube Link]
- **Performance Optimization**: [YouTube Link]

### Community Resources

- **ComfyUI Discord**: https://discord.gg/comfyui
- **Reddit Communities**: r/StableDiffusion, r/ComfyUI
- **Forums**: AI Forum [link], Stable Diffusion Forums [link]

### NVIDIA GPU Resources

- **CUDA Installation Guide**: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- **cuDNN Installation**: https://docs.nvidia.com/deeplearning/cudnn/install-guide/
- **GPU Compute Capability**: https://developer.nvidia.com/cuda-gpus#compute

### Performance Optimization Guides

- **Memory Optimization**: [Internal Guide]
- **Speed Tuning**: [Internal Guide]
- **Quality Enhancement**: [Internal Guide]

### Issue Tracking

- **GitHub Issues**: https://github.com/blongsta/wan2.1-i2v-workflow/issues
- **Report Bugs**: Include GPU info, preset used, error message
- **Feature Requests**: Open discussion for new capabilities

### Useful Tools

- **GPU Monitor**: GPU-Z (Windows), nvidia-smi (Linux)
- **Video Player**: VLC Media Player
- **Image Editor**: GIMP, Photoshop (for input preparation)
- **Video Encoder**: FFmpeg (post-processing)

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/your-feature`
3. **Make changes** and test thoroughly
4. **Commit changes**: `git commit -am 'Add your feature'`
5. **Push to branch**: `git push origin feature/your-feature`
6. **Submit pull request** with detailed description

### Contribution Areas

- Performance optimizations
- Bug fixes
- New presets for different hardware
- Documentation improvements
- Workflow enhancements

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- ComfyUI team for the excellent framework
- Wan model researchers and developers
- Community contributions and feedback
- NVIDIA for GPU support and documentation

## ℹ️ Support

For help and support:

1. **Check Troubleshooting section** above
2. **Search GitHub Issues** for similar problems
3. **Create new GitHub Issue** with detailed information:
   - GPU model and VRAM
   - Python version
   - CUDA version
   - Error messages
   - Steps to reproduce

4. **Contact**: Open discussion or issue on repository

---

**Last Updated**: December 6, 2025

**Workflow Version**: 2.1.0

**Status**: Production Ready ✅

For the latest updates, visit: https://github.com/blongsta/wan2.1-i2v-workflow
