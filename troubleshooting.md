# Troubleshooting Guide - WAN 2.1 I2V Workflow

## Table of Contents
1. [CUDA Out of Memory Errors](#cuda-out-of-memory-errors)
2. [Model Loading Issues](#model-loading-issues)
3. [Custom Node Problems](#custom-node-problems)
4. [Performance Issues](#performance-issues)
5. [Workflow Configuration Problems](#workflow-configuration-problems)
6. [Diagnostic Commands](#diagnostic-commands)

---

## CUDA Out of Memory Errors

### Symptoms
- Error messages like "CUDA out of memory" or "RuntimeError: CUDA out of memory"
- Application crashes during model inference or processing
- Workflow hangs or becomes unresponsive

### Solutions

#### 1. **Reduce Batch Size**
   ```
   Action: Lower the batch size parameter in your workflow nodes
   Default: Typically 1-4
   Recommendation: Start with batch_size=1 and increase gradually
   ```

#### 2. **Reduce Resolution/Image Dimensions**
   - Lower the input image resolution
   - Reduce video frame dimensions
   - Consider processing in tiles rather than full image

#### 3. **Enable Memory Optimization Modes**
   ```
   Options to try:
   - memory_efficient=True
   - use_half_precision=True (fp16 instead of fp32)
   - enable_flash_attention=True
   - attention_mode="flash" (if available)
   ```

#### 4. **Clear CUDA Cache**
   ```python
   import torch
   torch.cuda.empty_cache()
   torch.cuda.synchronize()
   ```

#### 5. **Check VRAM Usage**
   - Monitor GPU memory with: `nvidia-smi`
   - Close other GPU-intensive applications
   - Ensure background processes aren't consuming memory

#### 6. **Split Processing**
   - Process video frames individually instead of batches
   - Divide long sequences into smaller chunks
   - Implement sliding window processing

#### 7. **System Configuration**
   - Increase GPU memory allocation if using shared systems
   - Disable unnecessary background tasks
   - Ensure adequate system RAM (16GB+ recommended)

#### 8. **Mixed Precision Training/Inference**
   - Enable automatic mixed precision (AMP)
   - Use `torch.cuda.amp.autocast()` context manager
   - Configuration: `use_amp=True`

---

## Model Loading Issues

### Symptoms
- Model fails to download or initialize
- "FileNotFoundError" or "Model not found" errors
- Checksum/verification failures
- Corrupted model weights
- Import/dependency errors

### Solutions

#### 1. **Verify Model Path**
   ```
   Check that:
   - Model directory exists
   - Path is correctly specified
   - File permissions allow read access
   - No spaces or special characters in path
   ```

#### 2. **Check Internet Connection**
   - Ensure stable internet for model downloads
   - Try manually downloading model from source
   - Check firewall/proxy settings

#### 3. **Clear Model Cache**
   ```
   Remove corrupted downloads:
   ~/.cache/huggingface/hub/
   ~/.cache/torch/
   ~/.cache/comfyui/
   
   Then restart workflow to re-download
   ```

#### 4. **Verify Model Integrity**
   ```bash
   # Check file size
   ls -lh /path/to/model
   
   # Verify checksum if available
   sha256sum model_file.safetensors
   ```

#### 5. **Dependency Issues**
   ```
   Ensure installed:
   - torch >= 2.0
   - torchvision
   - transformers
   - safetensors
   - omegaconf
   - einops
   
   Update with: pip install --upgrade [package_name]
   ```

#### 6. **Model Format Compatibility**
   - Verify model format matches expected type (safetensors, .pt, .pth)
   - Some nodes require specific formats
   - Check node documentation for format requirements

#### 7. **Memory Allocation for Loading**
   - Models require memory just to load
   - Ensure sufficient VRAM before loading
   - Load to CPU first if VRAM insufficient:
     ```python
     model = load_model(device='cpu')
     model.to('cuda')
     ```

#### 8. **Hugging Face Model Issues**
   ```
   If downloading from HF:
   - Set token: huggingface-cli login
   - Check model is accessible: huggingface-cli repo-info model_id
   - Use snapshot downloads for stability
   ```

#### 9. **Version Mismatches**
   - Check model was created with compatible PyTorch version
   - SafeTensors models are more compatible across versions
   - Use `torch.load(..., weights_only=True)` for safety

---

## Custom Node Problems

### Symptoms
- Custom nodes fail to load
- "ModuleNotFoundError" or import errors
- Node appears in UI but crashes on execution
- Missing node dependencies
- Version incompatibilities

### Solutions

#### 1. **Verify Node Installation**
   ```
   Check custom_nodes directory:
   - Node folder exists in correct location
   - __init__.py file present
   - Required files are not corrupted
   ```

#### 2. **Node Dependencies**
   ```bash
   # Install node-specific requirements
   pip install -r /path/to/custom_node/requirements.txt
   
   # Check if all imports work
   python -c "from custom_node import NodeClass"
   ```

#### 3. **Node Path Configuration**
   ```
   Ensure custom_nodes folder is:
   - In ComfyUI root directory, OR
   - Properly referenced in config
   - Python path includes the directory
   ```

#### 4. **Update Custom Nodes**
   ```bash
   # If using git-based nodes
   cd custom_nodes/[node_name]
   git pull origin main
   pip install -r requirements.txt
   ```

#### 5. **Check Node Compatibility**
   - Verify node supports your Python version
   - Check node supports your PyTorch version
   - Ensure model format compatibility with node

#### 6. **Node Conflicts**
   - Duplicate node names can cause conflicts
   - Check for nodes with same class names
   - Review node registration in __init__.py

#### 7. **Debug Node Loading**
   ```python
   import sys
   import traceback
   
   try:
       from custom_node import NodeClass
   except Exception as e:
       print(f"Failed to load: {e}")
       traceback.print_exc()
   ```

#### 8. **Clear Node Cache**
   - Restart Python/ComfyUI application
   - Delete any .pyc or __pycache__ files
   - Clear compiled caches in custom_nodes directories

#### 9. **Common Node Issues**
   ```
   - OpenCV errors: pip install opencv-python
   - PIL/Pillow errors: pip install Pillow
   - NumPy incompatibility: pip install --upgrade numpy
   - CUDA toolkit mismatch: Reinstall torch with correct CUDA version
   ```

---

## Performance Issues

### Symptoms
- Slow processing speed
- High CPU usage with low GPU utilization
- Memory leaks causing gradual slowdown
- Unexpected bottlenecks
- Inconsistent performance

### Solutions

#### 1. **GPU Utilization Monitoring**
   ```bash
   # Real-time GPU monitoring
   watch -n 1 nvidia-smi
   
   # Detailed memory breakdown
   nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
   
   # Check GPU processes
   nvidia-smi pmon -c 1
   ```

#### 2. **Enable GPU Computation**
   ```
   Verify:
   - CUDA available: torch.cuda.is_available() == True
   - Device set correctly: device = 'cuda:0'
   - Models loaded to GPU: model.to('cuda')
   - Inputs on same device as model
   ```

#### 3. **Optimize Model Loading**
   ```
   - Use lower precision (fp16) if supported
   - Load model once and reuse
   - Avoid repeated model initialization
   - Consider quantization for smaller models
   ```

#### 4. **Batch Processing Optimization**
   ```
   - Process optimal batch size (not too small, not too large)
   - Use num_workers for data loading
   - Enable pin_memory=True for faster data transfer
   ```

#### 5. **Memory Leak Investigation**
   ```python
   # Check for memory accumulation
   import torch
   
   for i in range(100):
       with torch.no_grad():
           output = model(input)
       print(f"Iteration {i}: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
   
   # Should remain relatively constant
   ```

#### 6. **Optimize Inference Settings**
   ```
   - Use torch.no_grad() for inference
   - Enable eval mode: model.eval()
   - Disable gradients: torch.set_grad_enabled(False)
   - Use compiled models: model = torch.compile(model)
   ```

#### 7. **I/O Optimization**
   - Use faster storage (NVMe vs HDD)
   - Preprocess images to required format
   - Cache preprocessed data
   - Avoid repeated I/O operations

#### 8. **CPU Performance Tuning**
   ```
   - Set number of threads: torch.set_num_threads()
   - Use MKL for CPU: export MKL_NUM_THREADS=4
   - Profile bottlenecks with cProfile
   ```

#### 9. **Workflow Optimization**
   - Reduce redundant computations
   - Cache intermediate results
   - Process in parallel when possible
   - Use async operations where applicable

#### 10. **System Resource Management**
   - Close unnecessary applications
   - Increase page file/swap space
   - Monitor CPU temperature
   - Check for thermal throttling

---

## Workflow Configuration Problems

### Symptoms
- Workflow fails to execute or save
- Node connections error
- Type mismatches between nodes
- Missing required parameters
- Configuration file corruption
- Unexpected behavior in established workflows

### Solutions

#### 1. **Validate Workflow JSON**
   ```bash
   # Verify JSON syntax
   python -m json.tool workflow.json > /dev/null
   
   # If syntax error occurs, use online JSON validator
   # or check for missing commas, quotes, brackets
   ```

#### 2. **Check Node Connections**
   ```
   Verify:
   - Output type matches input type
   - All required inputs are connected
   - No circular dependencies
   - Proper slot indices
   ```

#### 3. **Parameter Validation**
   ```
   Ensure:
   - All required parameters are set
   - Values within acceptable ranges
   - Correct parameter types (int, float, string)
   - No invalid special characters
   ```

#### 4. **Reset to Default Configuration**
   ```
   - Save current workflow
   - Start with minimal working example
   - Gradually add complexity
   - Identify problematic configuration
   ```

#### 5. **Update Custom Node Schemas**
   ```python
   # Nodes must define proper return types
   @classmethod
   def RETURN_TYPES(cls):
       return ("IMAGE", "STRING")
   
   @classmethod
   def RETURN_NAMES(cls):
       return ("image", "info")
   ```

#### 6. **Version Compatibility**
   - Check workflow was created with compatible version
   - Review changelog for breaking changes
   - Update node definitions if node was updated

#### 7. **Configuration File Issues**
   ```
   Common issues:
   - Outdated config format
   - Missing required fields
   - Incompatible option values
   - File permissions preventing writes
   
   Solution: Reset to defaults and reconfigure
   ```

#### 8. **Conditional Execution**
   ```
   Verify:
   - Conditional nodes properly configured
   - Skip conditions are valid
   - Loop count is reasonable
   - No infinite loops in workflow
   ```

#### 9. **Workflow State Issues**
   ```
   - Clear workflow cache
   - Reset node execution state
   - Clear temporary files
   - Restart execution environment
   ```

#### 10. **Debugging Workflow Execution**
   ```python
   # Add debug output
   print(f"Input shape: {input_tensor.shape}")
   print(f"Output shape: {output_tensor.shape}")
   print(f"Parameter value: {parameter_value}")
   
   # Verify intermediate results
   assert input_tensor is not None, "Input is None"
   assert output_tensor.shape[0] > 0, "Output is empty"
   ```

---

## Diagnostic Commands

### System Information

#### NVIDIA GPU Diagnostics
```bash
# Basic GPU info
nvidia-smi

# Detailed GPU capabilities
nvidia-smi -i 0 --query-gpu=driver_version,compute_cap --format=csv

# GPU memory state
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

# Real-time memory usage
watch -n 0.5 nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# Check CUDA version
nvidia-smi | grep "CUDA Version"
```

#### CUDA Toolkit Verification
```bash
# Verify CUDA installation
nvcc --version

# Check CUDA path
echo $CUDA_HOME
echo $LD_LIBRARY_PATH

# Test CUDA with PyTorch
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

#### PyTorch Diagnostics
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

# Create test tensor on GPU
test_tensor = torch.randn(1000, 1000).to('cuda')
print(f"Test successful: {test_tensor.is_cuda}")
```

### Workflow Diagnostics

#### Check Required Dependencies
```bash
# Create requirements file
pip freeze > requirements_snapshot.txt

# Verify specific packages
pip show torch
pip show torchvision
pip show transformers
pip show safetensors

# Check version compatibility
python -c "
import torch
import torchvision
import transformers
print(f'Torch: {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')
print(f'Transformers: {transformers.__version__}')
"
```

#### Custom Node Diagnostics
```bash
# List installed custom nodes
ls -la custom_nodes/

# Check for import errors
python -c "
import sys
sys.path.insert(0, 'custom_nodes')
try:
    import node_name
    print('Node imported successfully')
except ImportError as e:
    print(f'Import error: {e}')
"

# Validate node structure
python -c "
import json
with open('custom_nodes/node_name/__init__.py') as f:
    content = f.read()
    if 'NODE_CLASS_MAPPINGS' in content:
        print('NODE_CLASS_MAPPINGS found')
    if 'RETURN_TYPES' in content:
        print('RETURN_TYPES defined')
"
```

### Performance Diagnostics

#### GPU Profiling
```bash
# Profile CUDA operations
python -c "
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Your code here
    x = torch.randn(1000, 1000).to('cuda')
    y = torch.matmul(x, x)

print(prof.key_averages().table(sort_by='cuda_time_total'))
"
```

#### Memory Profiling
```python
import torch
import tracemalloc

tracemalloc.start()

# Your code here
output = model(input_tensor)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1e9:.2f}GB; Peak: {peak / 1e9:.2f}GB")
tracemalloc.stop()
```

#### Timing Diagnostics
```python
import time
import torch

# CPU timing
start = time.time()
result = operation()
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.4f}s")

# GPU timing with synchronization
torch.cuda.synchronize()
start = time.time()
result = operation()
torch.cuda.synchronize()
gpu_time = time.time() - start
print(f"GPU time: {gpu_time:.4f}s")
```

### Logging and Debugging

#### Enable Verbose Logging
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
```

#### Workflow Execution Log
```bash
# Capture full execution output
python workflow_executor.py > execution_log.txt 2>&1

# Filter for errors
grep -i "error\|exception\|failed" execution_log.txt

# Check timestamps for performance
grep "Duration\|Elapsed\|Time" execution_log.txt
```

#### Model State Inspection
```python
# Inspect model architecture
print(model)

# Check parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Verify device placement
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")
```

### Emergency Recovery

#### Clear All Caches
```bash
# PyTorch cache
rm -rf ~/.cache/torch/*

# Hugging Face cache
rm -rf ~/.cache/huggingface/*

# ComfyUI cache
rm -rf ./cache/*
rm -rf ./temp/*

# Python bytecode
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete
```

#### Verify Core Functionality
```bash
# Test CUDA
python -c "import torch; assert torch.cuda.is_available()"

# Test PyTorch basic operations
python -c "
import torch
a = torch.randn(100, 100).cuda()
b = torch.randn(100, 100).cuda()
c = torch.matmul(a, b)
assert c.shape == (100, 100)
print('CUDA operations OK')
"

# Test model loading with safetensors
python -c "
from safetensors.torch import load_file
# Load a known working model
print('SafeTensors OK')
"
```

---

## Quick Reference: Common Error Solutions

| Error | Quick Fix |
|-------|-----------|
| CUDA out of memory | Reduce batch_size, enable fp16, clear cache with `torch.cuda.empty_cache()` |
| Model not found | Check path, verify internet connection, clear cache and re-download |
| Import error | Install missing package with `pip install`, check Python path |
| Node not loading | Verify node dependencies, check __init__.py, restart Python |
| Slow performance | Check GPU utilization, reduce resolution, enable torch.compile() |
| JSON decode error | Validate workflow.json syntax, check for corrupted files |
| Type mismatch | Verify node output/input compatibility, check RETURN_TYPES |
| Memory leak | Use `torch.no_grad()`, enable eval mode, restart Python |

---

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [ComfyUI GitHub Issues](https://github.com/comfyorg/ComfyUI/issues)
- [SafeTensors Documentation](https://huggingface.co/docs/safetensors/)

---

## Getting Help

If you've tried the above solutions:

1. **Gather diagnostic information** using commands in [Diagnostic Commands](#diagnostic-commands)
2. **Create a minimal reproducible example** that triggers the issue
3. **Include full error messages** and stack traces
4. **Document your system specifications** (GPU, CUDA version, PyTorch version)
5. **Check existing issues** before reporting new ones
6. **Provide workflow.json** if configuration-related

---

**Last Updated:** 2025-12-06  
**Version:** 1.0
