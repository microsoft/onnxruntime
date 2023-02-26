# Stable Diffusion CUDA Optimization

## Overview

[Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement) is a text-to-image latent diffusion model for image generation. Explanation of the Stable Diffusion can be found in [Stable Diffusion with Diffusers](https://huggingface.co/blog/stable_diffusion).

## CUDA Optimizations for Stable Diffusion

ONNX Runtime uses the following optimizations to speed up Stable Diffusion in CUDA:
* [Flash Attention](https://arxiv.org/abs/2205.14135) for float16 precision. Flash Attention uses tiling to reduce number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. The kernel requires GPUs of Compute Capability >= 7.5 (like T4, A100, and RTX 2060~4090).
* [Memory Efficient Attention](https://arxiv.org/abs/2112.05682v2) for float32 precision or older GPUs (like V100). We used the fused multi-head attention in CutLASS, and the kernel was implemented by xFormers.
* Channel-last (NHWC) convolution. For NVidia GPU with Tensor Cores support, NHWC tensor layout is recommended for convolution. See [Tensor Layouts In Memory: NCHW vs NHWC](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout)
* GroupNorm kernel for NHWC tensor layout.
* SkipLayerNormalization which fuses LayerNormalization with Add bias and residual inputs.
* BiasSplitGelu is a fusion of Add bias with SplitGelu activation.
* BiasAdd fuses Add bias and residual (BiasAdd is not available in 1.14.* package).
* Reduce Transpose nodes by graph transformation. 

Many CUDA kernels (like flash attentions kernels, GroupNorm and SplitGelu etc.) were originally implemented in TensorRT by Nvidia. Compare to TensorRT, our optimizations have some advantages: (1) Support float32. (2) Support older GPUs like V100.

## Scripts:

| Script | Description
|---|---|
| [optimize_pipeline.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/stable_diffusion/optimize_pipeline.py) | Optimize Stable Diffusion ONNX models
| [benchmark.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/stable_diffusion/benchmark.py) | Benchmark latency and memory usage of OnnxRuntime with other solutions like xFormers and PyTorch 2.0.

## Usage
Below is an example to optimize Stable Diffusion 1.5 models in Linux. For Windows OS, please change the format of path to be like `.\sd-v1-5` instead of `./sd-v1-5`.

### Setup Environment

First, Let's create an python environment using [AnaConda](https://www.anaconda.com/products/distribution#Downloads), then install packages in [requirements.txt](https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/python/tools/transformers/models/stable_diffusion/requirements.txt):

```
conda create -n py310 python=3.10
conda activate py310
pip install -r requirements.txt
```

To export ONNX model, we also need install [PyTorch](https://pytorch.org/). We tested PyTorch 1.13.1, which can be installed like the following:
```
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

ONNX Runtime 1.14.* requires CUDA 11.* and [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) for GPU inference. See https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html for compatible CUDA and CuDNN versions. We tested with [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) and cuDNN 8.7.0.84. 

In below example, we run the scripts in source code directory. You can get source code like the following:
```
git clone https://github.com/microsoft/onnxruntime
cd onnxruntime/python/tools/transformers/models/stable_diffusion
```

### Install Nightly (Optional)
If you want to try latest optimizations, you can install [ort-nightly-gpu](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly-gpu/) package like the following:

```
pip uninstall onnxruntime-gpu
pip install ort-nightly-gpu -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

### Export ONNX pipeline

This step will export stable diffusion ONNX models in float32.

The following script downloads the model weights, you need to be logged in via `huggingface-cli login`. After login, run the following commands to export ONNX models:
```
curl https://raw.githubusercontent.com/huggingface/diffusers/v0.13.0/scripts/convert_stable_diffusion_checkpoint_to_onnx.py > convert_sd_onnx.py
python convert_sd_onnx.py --model_path runwayml/stable-diffusion-v1-5  --output_path  ./sd-v1-5
```

### Optimize ONNX Pipeline

Example to optimize the exported float32 ONNX models, and save to float16 models:
```
python optimize_pipeline.py -i ./sd-v1-5 -o ./sd-v1-5-fp16 --float16
```

Note that this will detect the installed ONNX Runtime version. If you installed ONNX Runtime v1.14, packed QKV and BiasAdd will be disabled since those kernels are added after v1.14.

For Stable Diffusion 2.1 model, you will need force Attention to run in float32 to avoid black image by appending `--force_fp32_ops unet:Attention` to the command line. If you are using nightly package, append `--force_fp32_ops unet:MultiHeadAttention` instead.

### Run Benchmark

Example to benchmark the optimized pipeline:
```
python benchmark.py -v 1.5 -p ./sd-v1-5-fp16/ -c 5 -b 1
```

### Run Benchmark on xFormers

Run PyTorch 1.13.1 with xFormers in the py310 environment created above.
```
pip install xformers
python benchmark.py -e torch -v 1.5 -c 5 -n 1 -b 1 --use_xformers
```

### Run Benchmark with PyTorch 2.0 with torch.compile

Let's create a new environment to run PyTorch 2.0:
```
conda create -n pt2 python=3.10
conda activate pt2
pip install -r requirements.txt
pip3 install numpy --pre torch --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117
export TRITON_PTXAS_PATH=/usr/local/cuda-11.7/bin/ptxas
python benchmark.py -e torch -v 1.5 -c 5 -n 1 -b 1 --enable_torch_compile
```
If there is error of libdevice.10.bc not found, need copy /usr/local/cuda-11.7/nvvm/libdevice/libdevice.10.bc to the corresponding location.

### Example Benchmark output

engine | version | provider | disable_safety_checker | height | width | steps | batch_size | batch_count | num_prompts | average_latency | median_latency | first_run_memory_MB | second_run_memory_MB
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
onnxruntime | 1.14.0 | CUDAExecutionProvider | TRUE | 512 | 512 | 50 | 1 | 5 | 1 | 2.7 | 2.7 | 6635.9 | 7141.9
onnxruntime | 1.14.0 | CUDAExecutionProvider | TRUE | 512 | 512 | 50 | 4 | 5 | 1 | 8.3 | 8.3 | 7127.9 | 7127.9
onnxruntime | 1.14.0 | CUDAExecutionProvider | TRUE | 512 | 512 | 50 | 8 | 5 | 1 | 15.7 | 15.7 | 7125.9 | 7125.9
torch | 2.0.0.dev20230220+cu117 | compile | TRUE | 512 | 512 | 50 | 1 | 5 | 1 | 3.1 | 3.1 | 13460.9 | 4050.9
torch | 2.0.0.dev20230220+cu117 | compile | TRUE | 512 | 512 | 50 | 4 | 5 | 1 | 8.0 | 8.0 | 14014.9 | 7084.9
torch | 2.0.0.dev20230220+cu117 | compile | TRUE | 512 | 512 | 50 | 8 | 5 | 1 | 15.6 | 15.5 | 14818.9 | 11054.9
torch | 2.0.0.dev20230220+cu117 | default | TRUE | 512 | 512 | 50 | 1 | 5 | 1 | 2.7 | 2.7 | 13460.9 | 4040.9
torch | 2.0.0.dev20230220+cu117 | default | TRUE | 512 | 512 | 50 | 4 | 5 | 1 | 8.8 | 8.8 | 13984.9 | 6748.9
torch | 2.0.0.dev20230220+cu117 | default | TRUE | 512 | 512 | 50 | 8 | 5 | 1 | 16.9 | 16.9 | 14602.9 | 10562.9
torch | 1.13.1+cu117 | xformers | TRUE | 512 | 512 | 50 | 1 | 5 | 1 | 3.5 | 3.5 | 14978.9 | 10448.9
torch | 1.13.1+cu117 | xformers | TRUE | 512 | 512 | 50 | 4 | 5 | 1 | 9.1 | 9.1 | 12968.9 | 8420.9
torch | 1.13.1+cu117 | xformers | TRUE | 512 | 512 | 50 | 8 | 5 | 1 | 17.4 | 17.4 | 15592.9 | 9132.9

Compared to 1.14.0, nightly package has better performance with less memory usage on GPUs with Compute Capability >= 7.5 (like T5, A100 etc).

### Future Works

We have plan on other optimizations:
(1) Use IO Binding in the pipeline. Currently the input and output of each module is in CPU, and there is extra data copy between GPU and CPU, which slows down the pipeline.
(2) Export the whole pipeline into one ONNX model. Currently, there are mutliple ONNX models for CLIP, VAE and U-Net etc. Each model uses separated thread pool and memory allocator. Combine them into one model could share thread pool and memory allocator, and be more efficient.
(3) Use CUDA graph to speed up.
(4) Attention fusion in CLIP
(5) Leverage FP8 in latest GPU