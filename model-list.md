# Model Download Links and Installation Instructions

This document provides detailed information about models required for the wan2.1-i2v-workflow project, including download links and installation instructions.

## Table of Contents
1. [Overview](#overview)
2. [Model Requirements](#model-requirements)
3. [Installation Instructions](#installation-instructions)
4. [Model Descriptions](#model-descriptions)
5. [Troubleshooting](#troubleshooting)

## Overview

The wan2.1-i2v-workflow requires several pre-trained models for image-to-video generation. Models should be downloaded and placed in the appropriate directories before running the workflow.

**Default Model Directory:** `./models/`

## Model Requirements

| Model Name | Type | Size | Purpose |
|---|---|---|---|
| wan2.1-i2v | Core Model | ~5-7GB | Main image-to-video generation model |
| CLIP ViT-L | Vision Encoder | ~340MB | Image understanding and conditioning |
| VAE | Autoencoder | ~170MB | Video frame encoding/decoding |
| T5-base | Text Encoder | ~220MB | Text condition processing |

## Installation Instructions

### Quick Start

1. **Create models directory:**
   ```bash
   mkdir -p ./models
   ```

2. **Download models:**
   Follow the model-specific instructions in the sections below.

3. **Verify installation:**
   ```bash
   python verify_models.py
   ```

### Detailed Setup

#### Step 1: Download wan2.1-i2v Model

**Option A: Using HuggingFace Hub (Recommended)**

```bash
pip install huggingface-hub
huggingface-cli download blongsta/wan2.1-i2v --local-dir ./models/wan2.1-i2v
```

**Option B: Direct Download**

- **URL:** https://huggingface.co/blongsta/wan2.1-i2v
- **Recommended:** Download the entire model folder and place in `./models/wan2.1-i2v/`

**Option C: Using Git LFS**

```bash
git clone https://huggingface.co/blongsta/wan2.1-i2v ./models/wan2.1-i2v
```

#### Step 2: Download Vision Encoder (CLIP ViT-L)

```bash
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./models/clip-vit-large
```

Alternative:
```python
from transformers import CLIPVisionModel, CLIPImageProcessor
model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", 
                                         cache_dir="./models/clip-vit-large")
```

#### Step 3: Download VAE Model

```bash
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir ./models/sd-vae
```

Or using transformers:
```python
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", 
                                     cache_dir="./models/sd-vae")
```

#### Step 4: Download Text Encoder (T5-base)

```bash
huggingface-cli download google/t5-v1_1-base --local-dir ./models/t5-base
```

Or using transformers:
```python
from transformers import T5EncoderModel, T5Tokenizer
model = T5EncoderModel.from_pretrained("google/t5-v1_1-base", 
                                       cache_dir="./models/t5-base")
```

### Step 5: Verify Directory Structure

After installation, your models directory should look like:

```
models/
├── wan2.1-i2v/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors
│   └── ... (other model files)
├── clip-vit-large/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ... (other model files)
├── sd-vae/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors
│   └── ... (other model files)
└── t5-base/
    ├── config.json
    ├── pytorch_model.bin
    └── ... (other model files)
```

## Model Descriptions

### wan2.1-i2v (Primary Model)

**Purpose:** Image-to-Video generation using the WAN 2.1 architecture

- **Architecture:** Diffusion-based transformer model
- **Input:** Image + optional text prompt
- **Output:** Generated video frames (typically 16-128 frames)
- **Performance:** 
  - Inference time: 30-120 seconds per video (depending on frame count and hardware)
  - VRAM requirement: 10-24GB
- **Training Data:** Large-scale video dataset with diverse content

**Usage:**
```python
from wan2_1_i2v import I2VPipeline

pipeline = I2VPipeline.from_pretrained("./models/wan2.1-i2v")
video = pipeline(image_path="image.jpg", prompt="A cat running")
```

### CLIP ViT-L (Vision Encoder)

**Purpose:** Extract visual features from images for conditioning

- **Architecture:** Vision Transformer with ~300M parameters
- **Input:** RGB images (224x224)
- **Output:** 768-dimensional feature vectors
- **Training Data:** 400M image-text pairs (LAION dataset)

**Key Features:**
- Strong semantic understanding
- Compatible with text embeddings through shared space

### VAE (Variational Autoencoder)

**Purpose:** Efficient video frame compression and reconstruction

- **Model:** Stable Diffusion VAE (fine-tuned for MSE loss)
- **Compression Ratio:** ~8x spatial, ~4x temporal
- **Latent Dimension:** 4D or 8D depending on configuration
- **Use Case:** Reduces memory requirements for video processing

### T5-base (Text Encoder)

**Purpose:** Convert text prompts to semantic embeddings

- **Architecture:** Encoder-only T5 transformer
- **Sequence Length:** Up to 512 tokens
- **Output:** 768-dimensional embeddings
- **Fine-tuned:** For video generation task

## Troubleshooting

### Issue: "Model not found" Error

**Solution:**
```bash
# Verify models directory exists
ls -la ./models/

# Check model file integrity
python -c "from transformers import AutoModel; AutoModel.from_pretrained('./models/wan2.1-i2v')"
```

### Issue: Out of Memory (OOM)

**Solutions:**
1. Reduce batch size in configuration
2. Use smaller model variants if available
3. Enable model quantization:
   ```python
   pipeline = I2VPipeline.from_pretrained("./models/wan2.1-i2v", 
                                          torch_dtype=torch.float16)
   ```

### Issue: Slow Model Loading

**Solutions:**
1. Use safetensors format (faster loading than pickle)
2. Pre-load models to GPU:
   ```python
   pipeline = I2VPipeline.from_pretrained("./models/wan2.1-i2v")
   pipeline = pipeline.to("cuda")
   ```

### Issue: Download Timeout

**Solutions:**
1. Increase timeout:
   ```bash
   huggingface-cli download --cache-dir ./models <model_id> --timeout 120
   ```
2. Use a download manager (wget, aria2c) for resume capability:
   ```bash
   aria2c -x 16 -k 1M <model_url>
   ```

### Issue: Corrupted Downloads

**Solution:**
```bash
# Remove cached models and re-download
rm -rf ./models/
# Then re-run installation steps
```

## System Requirements

**Minimum Requirements:**
- GPU: NVIDIA GPU with 10GB VRAM (RTX 3080 or better recommended)
- Storage: 15GB free space for all models
- RAM: 16GB system RAM
- CUDA: 11.8 or higher

**Recommended Requirements:**
- GPU: NVIDIA A100 (40GB) or RTX 4090
- Storage: 20GB free space
- RAM: 32GB system RAM
- CUDA: 12.0 or higher

## License Information

Please note the licenses for each model:

- **wan2.1-i2v:** Check repository for licensing terms
- **CLIP ViT-L:** OpenAI license
- **Stable Diffusion VAE:** CreativeML Open RAIL License
- **T5-base:** Apache 2.0 License

## Additional Resources

- [HuggingFace Model Hub](https://huggingface.co/models)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## Support

For issues or questions:
1. Check this document's troubleshooting section
2. Visit the [project repository](https://github.com/blongsta/wan2.1-i2v-workflow)
3. Open an issue with detailed error logs

---

**Last Updated:** 2025-12-06
**Document Version:** 1.0
