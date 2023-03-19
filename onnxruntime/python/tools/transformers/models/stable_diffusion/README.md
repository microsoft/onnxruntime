# Stable Diffusion CUDA Optimization

## Overview

[Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement) is a text-to-image latent diffusion model for image generation. Explanation of the Stable Diffusion can be found in [Stable Diffusion with Diffusers](https://huggingface.co/blog/stable_diffusion).

## CUDA Optimizations for Stable Diffusion

ONNX Runtime uses the following optimizations to speed up Stable Diffusion in CUDA:

* [Flash Attention](https://arxiv.org/abs/2205.14135) for float16 precision. Flash Attention uses tiling to reduce number of GPU memory reads/writes, and improves performance with less memory for long sequence length. The kernel requires GPUs of Compute Capability >= 7.5 (like T4, A100, and RTX 2060~4090).
* [Memory Efficient Attention](https://arxiv.org/abs/2112.05682v2) for float32 precision or older GPUs (like V100). We used the fused multi-head attention kernel in CUTLASS, and the kernel was contributed by xFormers.
* Channel-last (NHWC) convolution. For NVidia GPU with Tensor Cores support, NHWC tensor layout is recommended for convolution. See [Tensor Layouts In Memory: NCHW vs NHWC](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout).
* GroupNorm kernel for NHWC tensor layout.
* SkipLayerNormalization which fuses LayerNormalization with Add bias and residual inputs.
* BiasSplitGelu is a fusion of Add bias with SplitGelu activation.
* BiasAdd fuses Add bias and residual.
* Reduce Transpose nodes by graph transformation.

Some CUDA kernels (Flash Attention, GroupNorm, SplitGelu and BiasAdd etc.) were originally implemented in TensorRT by Nvidia.

Compare to TensorRT, ONNX Runtime has some advantages on stable diffusion: (1) Supports older GPUs like V100. (2) Uses less GPU memory. (3) Supports float32 models and Stable Diffusion 2.* models.

To show the impact of each optimization on latency and GPU memory, we did an experiment on RTX 3060 GPU:

| Optimizations                                                                      | Average Latency (batch_size=1) | Memory in MB (batch_size=1) | Average Latency (batch_size=8) | Memory in MB (batch_size=8) |
| ---------------------------------------------------------------------------------- | ------------------------------ | --------------------------- | ------------------------------ | --------------------------- |
| Raw FP32 models                                                                    | 25.6                           | 10,667                      | OOM                            | OOM                         |
| FP16 baseline                                                                      | 10.2                           | 10,709                      | OOM                            | OOM                         |
| FP16 baseline + FMHA                                                               | 6.1                            | 7,719                       | 39.1                           | 10,821                      |
| FP16 baseline + FMHA + NhwcConv                                                    | 5.5                            | 7,656                       | 38.8                           | 11,615                      |
| FP16 baseline + FMHA + NhwcConv + GroupNorm                                        | 5.1                            | 6,673                       | 35.8                           | 10,763                      |
| FP16 baseline + FMHA + NhwcConv + GroupNorm + BiasSplitGelu                        | 4.9                            | 4,447                       | 33.7                           | 6,669                       |
| FP16 baseline + FMHA + NhwcConv + GroupNorm + BiasSplitGelu + Packed QKV           | 4.8                            | 4,625                       | 33.5                           | 6,663                       |
| FP16 baseline + FMHA + NhwcConv + GroupNorm + BiasSplitGelu + Packed QKV + BiasAdd | 4.7                            | 4,480                       | 33.3                           | 6,499                       |

FP16 baseline contains optimizations available in ONNX Runtime 1.13 including LayerNormalization, SkipLayerNormalization, Gelu and float16 conversion.

Here FMHA means Attention and MultiHeadAttention operators with Flash Attention and Memory Efficient Attention kernels but inputs are not packed. Packed QKV means the inputs are packed.

The last two optimizations (Packed QKV and BiasAdd) are only available in nightly package. Compared to 1.14.1, nightly package has slight improvement in performance.

## Scripts:

| Script                                                                                                                                                     | Description                                                                               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [optimize_pipeline.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/stable_diffusion/optimize_pipeline.py) | Optimize Stable Diffusion ONNX models                                                     |
| [benchmark.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/stable_diffusion/benchmark.py)                 | Benchmark latency and memory of OnnxRuntime, xFormers or PyTorch 2.0 on stable diffusion. |

In below example, we run the scripts in source code directory. You can get source code like the following:

```
git clone https://github.com/microsoft/onnxruntime
cd onnxruntime/python/tools/transformers/models/stable_diffusion
```

## Example of Stable Diffusion 1.5

Below is an example to optimize Stable Diffusion 1.5 in Linux. For Windows OS, please change the format of path to be like `.\sd-v1-5` instead of `./sd-v1-5`.

### Setup Environment

First, Let's create a python environment using [AnaConda](https://www.anaconda.com/products/distribution#Downloads), then install packages in [requirements.txt](https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/python/tools/transformers/models/stable_diffusion/requirements.txt):

```
conda create -n py310 python=3.10
conda activate py310
pip install -r requirements.txt
```

For Windows, torch installed from pypi is CPU only. Need install PyTorch 1.13.1+cu117 instead like the following to support GPU:
```
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

ONNX Runtime requires CUDA and [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) for GPU inference. See https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html for compatible versions (like [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) and cuDNN 8.5.0.96 in Windows).

#### Install Nightly (Optional)

Skip this step if you use onnxruntime-gpu 1.14.* release package.

To try latest optimizations, you can install [ort-nightly-gpu](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly-gpu/) package like the following:

```
pip uninstall onnxruntime-gpu
pip install ort-nightly-gpu -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

The diffusers need a patch to work with ort-nightly-gpu: add `"ort-nightly-gpu",` below [this line](https://github.com/huggingface/diffusers/blob/1586186eea36e6a9b6f16c90ee32921c316c4b57/src/diffusers/utils/import_utils.py#L155).

### Export ONNX pipeline

This step will export stable diffusion 1.5 to ONNX model in float32 using script from diffusers. Before running the script, you need to be logged in via `huggingface-cli login`.

```
curl https://raw.githubusercontent.com/huggingface/diffusers/v0.13.0/scripts/convert_stable_diffusion_checkpoint_to_onnx.py > convert_sd_onnx.py
python convert_sd_onnx.py --model_path runwayml/stable-diffusion-v1-5  --output_path  ./sd-v1-5
```

### Optimize ONNX Pipeline

Example to optimize the exported float32 ONNX models, and save to float16 models:

```
python optimize_pipeline.py -i ./sd-v1-5 -o ./sd-v1-5-fp16 --float16
```

If you installed ONNX Runtime v1.14, some optimizations (packed QKV and BiasAdd) will be disabled automatically since they are not available in v1.14.

For Stable Diffusion 2.1 model, you will need force Attention to run in float32 to avoid black image by appending `--force_fp32_ops unet:Attention` to the command line. If you are using nightly package, append `--force_fp32_ops unet:MultiHeadAttention` instead.

### Run Benchmark

The benchmark.py script will run a warm-up prompt twice, and measure the peak GPU memory usage in these two runs, then record them as first_run_memory_MB and second_run_memory_MB. Then it will run 5 runs to get average latency (in seconds), and output the results to benchmark_result.csv.

Note that the first run might need more time and memory: For example, cuDNN convolution algorithm search or model compile happens in the first run.

Example to benchmark the optimized pipeline with batch size 1:
```
python benchmark.py -p ./sd-v1-5-fp16/ -b 1
```

The default parameters are stable diffusion version=1.5, height=512, width=512, steps=50, batch_count=5. Run `python benchmark.py --help` for more information.

### Run Benchmark with xFormers

Run PyTorch 1.13.1+cu117 with xFormers like the following

```
python benchmark.py -e torch -b 1 --use_xformers
```

### Run Benchmark with PyTorch 2.0 with torch.compile

Let's create a new environment to run PyTorch 2.0:

```
conda create -n pt2 python=3.10
conda activate pt2
pip3 install numpy --pre torch --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117
pip install -r requirements.txt
python benchmark.py -e torch -b 1 --enable_torch_compile
```

Sometime, it complains ptxas not found when there are multiple CUDA versions installed. It can be fixed like `export TRITON_PTXAS_PATH=/usr/local/cuda-11.7/bin/ptxas` before running benchmark.

Note that torch.compile is not supported in Windows: we encountered error `Windows not yet supported for torch.compile`. So it is excluded from RTX 3060 results of Windows.

### Example Benchmark output

Common settings for below test results:

| model_name                     | disable_safety_checker | height | width | steps | batch_count | num_prompts |
| ------------------------------ | ---------------------- | ------ | ----- | ----- | ----------- | ----------- |
| runwayml/stable-diffusion-v1-5 | TRUE                   | 512    | 512   | 50    | 5           | 1           |

#### Results of RTX 3060 (in Windows 11)

| engine      | version                 | provider              | batch size | average latency | first run memory MB | second run memory MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | ------------------- | -------------------- |
| onnxruntime | 1.14.1                  | CUDAExecutionProvider | 1          | 4.8             | 4,117               | 4,625                |
| torch       | 2.0.0.dev20230220+cu117 | default               | 1          | 5.6             | 4,330               | 4,050                |
| torch       | 1.13.1+cu117            | xformers              | 1          | 6.0             | 9,124               | 9,130                |
| onnxruntime | 1.14.1                  | CUDAExecutionProvider | 4          | 17.7            | 6,659               | 6,659                |
| torch       | 2.0.0.dev20230220+cu117 | default               | 4          | 20.2            | 6,425               | 6,911                |
| torch       | 1.13.1+cu117            | xformers              | 4          | 21.6            | 10,407              | 10,409               |
| onnxruntime | 1.14.1                  | CUDAExecutionProvider | 8          | 33.5            | 6,663               | 6,663                |
| torch       | 2.0.0.dev20230220+cu117 | default               | 8          | 39.8            | 10,894              | 10,782               |
| torch       | 1.13.1+cu117            | xformers              | 8          | 41.1            | 10,825              | 9,255                |

#### Results of V100-PCIE-16GB (in Ubuntu 20.04)

Results from Standard_NC6s_v3 Azure virtual machine:

| engine      | version                 | provider              | batch size | average latency | first run memory MB | second run memory MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | ------------------- | -------------------- |
| onnxruntime | 1.14.1                  | CUDAExecutionProvider | 1          | 2.7             | 6,646               | 7,152                |
| torch       | 2.0.0.dev20230220+cu117 | compile               | 1          | 3.1             | 13,461              | 4,051                |
| torch       | 2.0.0.dev20230220+cu117 | default               | 1          | 2.7             | 13,461              | 4,041                |
| torch       | 1.13.1+cu117            | xformers              | 1          | 3.5             | 14,979              | 10,449               |
| onnxruntime | 1.14.1                  | CUDAExecutionProvider | 4          | 8.4             | 7,114               | 7,114                |
| torch       | 2.0.0.dev20230220+cu117 | compile               | 4          | 8.0             | 14,015              | 7,085                |
| torch       | 2.0.0.dev20230220+cu117 | default               | 4          | 8.8             | 13,985              | 6,749                |
| torch       | 1.13.1+cu117            | xformers              | 4          | 9.1             | 12,969              | 8,421                |
| onnxruntime | 1.14.1                  | CUDAExecutionProvider | 8          | 15.9            | 7,120               | 7,120                |
| torch       | 2.0.0.dev20230220+cu117 | compile               | 8          | 15.6            | 14,819              | 11,055               |
| torch       | 2.0.0.dev20230220+cu117 | default               | 8          | 16.9            | 14,603              | 10,563               |
| torch       | 1.13.1+cu117            | xformers              | 8          | 17.4            | 15,593              | 9,133                |


#### Results of T4 (in Ubuntu 20.04)

To make the result stable, we lock the frequency of T4 GPU like 
`sudo nvidia-smi --lock-gpu-clocks=990` for fair comparison. See [nvidia blog](https://developer.nvidia.com/blog/advanced-api-performance-setstablepowerstate/) for more information. Note that performance might be slightly better without locking frequency.

Results are from Standard_NC4as_T4_v3 Azure virtual machine:

| engine      | version                 | provider              | batch size | average latency | first run memory MB | second run memory MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | ------------------- | -------------------- |
| onnxruntime | 1.14.1                  | CUDAExecutionProvider | 1          | 5.6             | 4,925               | 4,925                |
| torch       | 1.13.1+cu117            | xformers              | 1          | 6.9             | 14,845              | 10,317               |
| torch       | 2.0.0.dev20230226+cu117 | compile               | 1          | 6.0             | 13,125              | 3,977                |
| torch       | 2.0.0.dev20230226+cu117 | default               | 1          | 6.3             | 13,127              | 3,979                |
| onnxruntime | 1.14.1                  | CUDAExecutionProvider | 4          | 23.0            | 6,977               | 6,977                |
| torch       | 1.13.1+cu117            | xformers              | 4          | 25.8            | 12,819              | 8,269                |
| torch       | 2.0.0.dev20230226+cu117 | compile               | 4          | 22.1            | 14,751              | 6,697                |
| torch       | 2.0.0.dev20230226+cu117 | default               | 4          | 25.0            | 14,535              | 6,481                |
| onnxruntime | 1.14.1                  | CUDAExecutionProvider | 8          | 46.4            | 6,779               | 6,779                |
| torch       | 1.13.1+cu117            | xformers              | 8          | 51.4            | 14,827              | 9,001                |
| torch       | 2.0.0.dev20230226+cu117 | compile               | 8          | 45.6            | 12,675              | 10,249               |
| torch       | 2.0.0.dev20230226+cu117 | default               | 8          | 50.5            | 12,077              | 9,653                |

### Future Works

There are other optimizations might improve the performance or reduce memory footprint:

* Use IO Binding in the pipeline. Currently the input and output of each model is in CPU, and extra data copy between GPU and CPU slows down the pipeline.
* Use CUDA graph to speed up inference.
* Export the whole pipeline into a single ONNX model. Currently, there are multiple ONNX models (CLIP, VAE and U-Net etc). Each model uses separated thread pool and memory allocator. Combine them into one model could share thread pool and memory allocator. The end result is more efficient and less memory footprint.
* For Stable Diffusion 2.1, we force Attention in fp32 to avoid black image. That slows down the inference significantly. We could potentially change attention kernel (like fp32 accumulation) to avoid the issue.
* Reduce GPU memory footprint by actively deleting buffers for intermediate results
* Reduce GPU memory footprint by CPU RAM Offloading
* Attention fusion in CLIP
* Safety Checker Optimization
* Leverage FP8 in latest GPU
