# Stable Diffusion CUDA Optimization

## Overview

[Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement) is a text-to-image latent diffusion model for image generation. Explanation of the Stable Diffusion can be found in [Stable Diffusion with Diffusers](https://huggingface.co/blog/stable_diffusion).

## CUDA Optimizations for Stable Diffusion

ONNX Runtime uses the following optimizations to speed up Stable Diffusion in CUDA:

* [Flash Attention](https://arxiv.org/abs/2205.14135) for float16 precision. Flash Attention uses tiling to reduce number of GPU memory reads/writes, and improves performance with less memory for long sequence length. The kernel requires GPUs of Compute Capability >= 7.5 (like T4, A100, and RTX 2060~4090).
* [Memory Efficient Attention](https://arxiv.org/abs/2112.05682v2) for float32 precision or older GPUs (like V100). We used the fused multi-head attention in CUTLASS.
* Channel-last (NHWC) convolution. For NVidia GPU with Tensor Cores support, NHWC tensor layout is recommended for convolution. See [Tensor Layouts In Memory: NCHW vs NHWC](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout).
* GroupNorm kernel for NHWC tensor layout.
* SkipLayerNormalization which fuses LayerNormalization with Add bias and residual inputs.
* BiasSplitGelu is a fusion of Add bias with SplitGelu activation.
* BiasAdd fuses Add bias and residual.
* Reduce Transpose nodes by graph transformation.

Many CUDA kernels (like Flash Attention, GroupNorm and SplitGelu etc.) were originally implemented in TensorRT by Nvidia. Compare to TensorRT, our optimizations have some advantages: (1) Support older GPUs like V100. (2) Use less GPU memory. (3) Support float32 models and Stable Diffusion 2.* models.

To show the impact of each optimization, we did an experiment on RTX 3060 GPU:

| Optimizations                                                                      | Average Latency (batch_size=1) | Memory in MB (batch_size=1) | Average Latency (batch_size=8) | Memory in MB (batch_size=8) |
| ---------------------------------------------------------------------------------- | ------------------------------ | --------------------------- | ------------------------------ | --------------------------- |
| Raw FP32 models                                                                    | 25.6                           | 10,667                      | OOM                            | OOM                         |
| FP16 baseline                                                                      | 10.2                           | 10,709                      | OOM                            | OOM                         |
| FP16 baseline + FMHA                                                               | 6.1                            | 7,719                       | 39.1                           | 11,916                      |
| FP16 baseline + FMHA + NhwcConv                                                    | 5.5                            | 7,543                       | 38.8                           | 11,566                      |
| FP16 baseline + FMHA + NhwcConv + GroupNorm                                        | 5.1                            | 6,673                       | 35.8                           | 10,763                      |
| FP16 baseline + FMHA + NhwcConv + GroupNorm + BiasSplitGelu                        | 4.8                            | 4,655                       | 33.7                           | 6,734                       |
| FP16 baseline + FMHA + NhwcConv + GroupNorm + BiasSplitGelu + Packed QKV           | 4.7                            | 4,611                       | 33.4                           | 6,820                       |
| FP16 baseline + FMHA + NhwcConv + GroupNorm + BiasSplitGelu + Packed QKV + BiasAdd | 4.7                            | 4,621                       | 33.1                           | 6,661                       |

FP16 baseline contains optimizations available in ORT 1.13 including LayerNormalization, SkipLayerNormalization, Gelu and float16 conversion.

Here FMHA means Attention and MultiHeadAttention operators with Flash Attention and Memory Efficient Attention kernels but inputs are not packed. Packed QKV means the inputs are packed.

The second_run_memory_MB in benchmark output is used for GPU memory in this table. Note that the first run might need more memory for cuDNN convolution algorithm search.

The last two optimizations (Packed QKV and BiasAdd) are only available in nightly package, and not in  1.14 package. All above tests use nightly package.

## Scripts:

| Script                                                                                                                                                     | Description                                                                                            |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [optimize_pipeline.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/stable_diffusion/optimize_pipeline.py) | Optimize Stable Diffusion ONNX models                                                                  |
| [benchmark.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/stable_diffusion/benchmark.py)                 | Benchmark latency and memory of OnnxRuntime, xFormers or PyTorch 2.0 on stable diffusion. |

## Usage

Below is an example to optimize Stable Diffusion 1.5 in Linux. For Windows OS, please change the format of path to be like `.\sd-v1-5` instead of `./sd-v1-5`.

### Setup Environment

First, Let's create a python environment using [AnaConda](https://www.anaconda.com/products/distribution#Downloads), then install packages in [requirements.txt](https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/python/tools/transformers/models/stable_diffusion/requirements.txt):

```
conda create -n py310 python=3.10
conda activate py310
pip install -r requirements.txt
```

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

Our benchmark script will run a warm-up prompt twice, and measure the peak GPU memory usage and record them as first_run_memory_MB and second_run_memory_MB.

Example to benchmark the optimized pipeline with batch size 1 (and default parameters height=512, width=512, steps=50), then the average and median latency (in seconds) of 5 batches are output.

```
python benchmark.py -v 1.5 -p ./sd-v1-5-fp16/ -c 5 -b 1
```

### Run Benchmark on xFormers

Run PyTorch 1.13.1 with xFormers in the py310 environment created above.

```
pip install xformers
pip install triton==2.0.0a2
python benchmark.py -e torch -v 1.5 -c 5 -b 1 --use_xformers
```

Note that triton package is only available in Linux right now.

### Run Benchmark with PyTorch 2.0 with torch.compile

Let's create a new environment to run PyTorch 2.0:

```
conda create -n pt2 python=3.10
conda activate pt2
pip install -r requirements.txt
pip3 install numpy --pre torch --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117
python benchmark.py -e torch -v 1.5 -c 5 -b 1 --enable_torch_compile
```
Sometime, it complains ptxas not found when there are multiple CUDA versions installed. It can be fixed like `export TRITON_PTXAS_PATH=/usr/local/cuda-11.7/bin/ptxas` before running benchmark.

If you run it in Windows (not in WSL), you might encounter error `Windows not yet supported for torch.compile`.

### Example Benchmark output

Common settings for below test results:

| model_name                     | disable_safety_checker | height | width | steps | batch_count | num_prompts |
| ------------------------------ | ---------------------- | ------ | ----- | ----- | ----------- | ----------- |
| runwayml/stable-diffusion-v1-5 | TRUE                   | 512    | 512   | 50    | 5           | 1           |

#### Results of RTX 3060 (in Windows 11)

| engine      | version                 | provider              | batch_size | average_latency | median_latency | first_run_memory_MB | second_run_memory_MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | -------------- | ------------------- | -------------------- |
| onnxruntime | 1.14.0                  | CUDAExecutionProvider | 1          | 5.0             | 4.9            | 4,068               | 4,576                |
| torch       | 2.0.0.dev20230220+cu117 | default               | 1          | 5.6             | 5.6            | 4,330               | 4,050                |
| torch       | 1.13.1+cu117            | xformers              | 1          | 6.0             | 6.0            | 9,124               | 9,130                |
| onnxruntime | 1.14.0                  | CUDAExecutionProvider | 4          | 17.7            | 17.7           | 6,643               | 6,643                |
| torch       | 2.0.0.dev20230220+cu117 | default               | 4          | 20.2            | 20.2           | 6,425               | 6,911                |
| torch       | 1.13.1+cu117            | xformers              | 4          | 21.6            | 21.4           | 10,407              | 10,409               |
| onnxruntime | 1.14.0                  | CUDAExecutionProvider | 8          | 33.7            | 33.6           | 6,636               | 6,636                |
| torch       | 2.0.0.dev20230220+cu117 | default               | 8          | 39.8            | 39.9           | 10,894              | 10,782               |
| torch       | 1.13.1+cu117            | xformers              | 8          | 41.1            | 41.1           | 10,825              | 9,255                |

#### Results of V100 (in Ubuntu 20.04)

| engine      | version                 | provider              | batch_size | average_latency | median_latency | first_run_memory_MB | second_run_memory_MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | -------------- | ------------------- | -------------------- |
| onnxruntime | 1.14.0                  | CUDAExecutionProvider | 1          | 2.7             | 2.7            | 6,636               | 7,142                |
| torch       | 2.0.0.dev20230220+cu117 | compile               | 1          | 3.1             | 3.1            | 13,461              | 4,051                |
| torch       | 2.0.0.dev20230220+cu117 | default               | 1          | 2.7             | 2.7            | 13,461              | 4,041                |
| torch       | 1.13.1+cu117            | xformers              | 1          | 3.5             | 3.5            | 14,979              | 10,449               |
| onnxruntime | 1.14.0                  | CUDAExecutionProvider | 4          | 8.3             | 8.3            | 7,128               | 7,128                |
| torch       | 2.0.0.dev20230220+cu117 | compile               | 4          | 8.0             | 8.0            | 14,015              | 7,085                |
| torch       | 2.0.0.dev20230220+cu117 | default               | 4          | 8.8             | 8.8            | 13,985              | 6,749                |
| torch       | 1.13.1+cu117            | xformers              | 4          | 9.1             | 9.1            | 12,969              | 8,421                |
| onnxruntime | 1.14.0                  | CUDAExecutionProvider | 8          | 15.7            | 15.7           | 7,126               | 7,126                |
| torch       | 2.0.0.dev20230220+cu117 | compile               | 8          | 15.6            | 15.5           | 14,819              | 11,055               |
| torch       | 2.0.0.dev20230220+cu117 | default               | 8          | 16.9            | 16.9           | 14,603              | 10,563               |
| torch       | 1.13.1+cu117            | xformers              | 8          | 17.4            | 17.4           | 15,593              | 9,133                |

Compared to 1.14.0, nightly package has better performance with less memory usage on GPUs with Compute Capability >= 7.5 (like T5, A100 etc).

#### Results of T4 (in Ubuntu 20.04)

| engine      | version                 | provider              | batch_size | average_latency | median_latency | first_run_memory_MB | second_run_memory_MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | -------------- | ------------------- | -------------------- |
| onnxruntime | 1.14.0                  | CUDAExecutionProvider | 1          | 5.3             | 5.3            | 4,877               | 4,877                |
| torch       | 1.13.1+cu117            | xformers              | 1          | 6.4             | 6.4            | 14,844              | 10,312               |
| torch       | 2.0.0.dev20230226+cu117 | compile               | 1          | 6.1             | 6.1            | 13,134              | 3,986                |
| torch       | 2.0.0.dev20230226+cu117 | default               | 1          | 6.4             | 6.4            | 13,128              | 3,978                |
| onnxruntime | 1.14.0                  | CUDAExecutionProvider | 4          | 22.2            | 22.2           | 6,967               | 6,967                |
| torch       | 1.13.1+cu117            | xformers              | 4          | 25.3            | 25.2           | 12,786              | 8,238                |
| torch       | 2.0.0.dev20230226+cu117 | compile               | 4          | 22.7            | 22.8           | 14,764              | 6,710                |
| torch       | 2.0.0.dev20230226+cu117 | default               | 4          | 25.3            | 25.3           | 14,494              | 6,440                |
| onnxruntime | 1.14.0                  | CUDAExecutionProvider | 8          | 46.1            | 46.2           | 6,953               | 6,953                |
| torch       | 1.13.1+cu117            | xformers              | 8          | 51.9            | 52.3           | 14,804              | 8,978                |
| torch       | 2.0.0.dev20230226+cu117 | compile               | 8          | 46.6            | 46.6           | 12,796              | 9,860                |
| torch       | 2.0.0.dev20230226+cu117 | default               | 8          | 51.3            | 51.3           | 12,096              | 9,674                |

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
