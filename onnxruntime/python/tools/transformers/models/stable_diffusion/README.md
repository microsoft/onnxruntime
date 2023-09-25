# Stable Diffusion GPU Optimization

## Overview

[Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement) is a text-to-image latent diffusion model for image generation. Explanation of the Stable Diffusion can be found in [Stable Diffusion with Diffusers](https://huggingface.co/blog/stable_diffusion).

## Optimizations for Stable Diffusion

ONNX Runtime uses the following optimizations to speed up Stable Diffusion in CUDA:

* [Flash Attention](https://arxiv.org/abs/2205.14135) for float16 precision. Flash Attention uses tiling to reduce number of GPU memory reads/writes, and improves performance with less memory for long sequence length. The kernel requires GPUs of Compute Capability >= 7.5 (like T4, A100, and RTX 2060~4090).
* [Memory Efficient Attention](https://arxiv.org/abs/2112.05682v2) for float32 precision or older GPUs (like V100). We used the fused multi-head attention kernel in CUTLASS, and the kernel was contributed by xFormers.
* Channel-last (NHWC) convolution. For NVidia GPU with Tensor Cores support, NHWC tensor layout is recommended for convolution. See [Tensor Layouts In Memory: NCHW vs NHWC](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout).
* GroupNorm kernel for NHWC tensor layout.
* SkipLayerNormalization which fuses LayerNormalization with Add bias and residual inputs.
* BiasSplitGelu is a fusion of Add bias with SplitGelu activation.
* BiasAdd fuses Add bias and residual.
* Reduce Transpose nodes by graph transformation.

These optimizations are firstly carried out on CUDA EP. They may not work on other EP. To show the impact of each optimization on latency and GPU memory, we did some experiments:

### Results on RTX 3060 GPU:

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

### Results on MI250X with 1 GCD

With runtime tuning enabled, we get following performance number on one GCD of a MI250X GPU:

| Optimizations                                                         | Average Latency (batch_size=1) | Memory in MB (batch_size=1) | Average Latency (batch_size=8) | Memory in MB (batch_size=8) |
| --------------------------------------------------------------------- | ------------------------------ | --------------------------- | ------------------------------ | --------------------------- |
| Raw FP32 models                                                       | 6.7                            | 17,319                      | 36.4 *                         | 33,787                      |
| FP16 baseline                                                         | 4.1                            | 8,945                       | 24.0 *                         | 34,493                      |
| FP16 baseline + FMHA                                                  | 2.6                            | 4,886                       | 15.0                           | 10,146                      |
| FP16 baseline + FMHA + NhwcConv                                       | 2.4                            | 4,952                       | 14.8                           | 9,632                       |
| FP16 baseline + FMHA + NhwcConv + GroupNorm                           | 2.3                            | 4,906                       | 13.6                           | 9,774                       |
| FP16 baseline + FMHA + NhwcConv + GroupNorm + BiasSplitGelu           | 2.2                            | 4,910                       | 12.5                           | 9,646                       |
| FP16 baseline + FMHA + NhwcConv + GroupNorm + BiasSplitGelu + BiasAdd | 2.2                            | 4,910                       | 12.5                           | 9,778                       |

The entries marked with `*` produce suspicious output images. The might be numerical stability or correctness issue for the pipeline. The performance number is for reference only.

## Scripts:

| Script                                         | Description                                                                               |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [optimize_pipeline.py](./optimize_pipeline.py) | Optimize Stable Diffusion ONNX models                                                     |
| [benchmark.py](./benchmark.py)                 | Benchmark latency and memory of OnnxRuntime, xFormers or PyTorch 2.0 on stable diffusion. |

In below example, we run the scripts in source code directory. You can get source code like the following:

```
git clone https://github.com/microsoft/onnxruntime
cd onnxruntime/onnxruntime/python/tools/transformers/models/stable_diffusion
```

## Example of Stable Diffusion 1.5

Below is an example to optimize Stable Diffusion 1.5 in Linux. For Windows OS, please change the format of path to be like `.\sd` instead of `./sd`.

### Setup Environment (CUDA)

It is recommended to create a Conda environment with Python 3.8, 3.9 or 3.10, and run the model with [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) or 11.8.
```
conda create -n py38 python=3.8
conda activate py38
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements-cuda.txt
```

ONNX Runtime requires CUDA and [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) for GPU inference. CUDA 11.7 and cuDNN 8.5 are used in our tests.

#### Install Nightly (Optional)

Skip this step if you use onnxruntime-gpu package from official releases.

To try latest optimizations, you can install [ort-nightly-gpu](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly-gpu/) package like the following:

```
pip uninstall onnxruntime-gpu
pip install ort-nightly-gpu -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

### Setup Environment (ROCm)

It is recommended that the users run the model with ROCm 5.4 or newer and Python 3.8, 3.9 or 3.10.
Note that Windows is not supported for ROCm at the moment.

```
conda create -n py38 python=3.8
conda activate py38
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-5.4/torch-1.12.1%2Brocm5.4-cp38-cp38-linux_x86_64.whl
pip install torch-1.12.1+rocm5.4-cp38-cp38-linux_x86_64.whl
pip install -r requirements-rocm.txt
```

AMD GPU version of PyTorch can be installed from [pytorch.org](https://pytorch.org/get-started/locally/) or [AMD Radeon repo](https://repo.radeon.com/rocm/manylinux/rocm-rel-5.4/).

#### Install onnxruntime-rocm

Here is an example to build onnxruntime from source with Rocm 5.4.2 in Ubuntu 20.04, and install the wheel.

(1) Install [ROCm 5.4.2](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4.2/page/How_to_Install_ROCm.html). Note that the version is also used in PyTorch 2.0 ROCm package.

(2) Install some tools used in build:
```
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
        wget \
        zip \
        ca-certificates \
        build-essential \
        curl \
        libcurl4-openssl-dev \
        libssl-dev \
        python3-dev
pip install numpy packaging "wheel>=0.35.1"
wget --quiet https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3-linux-x86_64.tar.gz
tar zxf cmake-3.26.3-linux-x86_64.tar.gz
export PATH=${PWD}/cmake-3.26.3-linux-x86_64/bin:${PATH}
```

(3) Build and Install ONNX Runtime
```
git clone https://github.com/microsoft/onnxruntime
cd onnxruntime
sh build.sh --config Release --use_rocm --rocm_home /opt/rocm --rocm_version 5.4.2 --build_wheel
pip install build/Linux/Release/dist/*.whl
```

You can also follow the [official docs](https://onnxruntime.ai/docs/build/eps.html#amd-rocm) to build with docker.

### Export ONNX pipeline
This step will export stable diffusion 1.5 to ONNX model in float32 using script from diffusers.

It is recommended to use PyTorch 1.12.1 or 1.13.1 in this step. Using PyTorch 2.0 will encounter issue in exporting onnx.

```
curl https://raw.githubusercontent.com/huggingface/diffusers/v0.15.1/scripts/convert_stable_diffusion_checkpoint_to_onnx.py > convert_sd_onnx.py
python convert_sd_onnx.py --model_path runwayml/stable-diffusion-v1-5  --output_path  ./sd_v1_5/fp32
```

For SDXL, use optimum to export the model:
```
pip install optimum diffusers onnx onnxruntime-gpu
optimum-cli export onnx --model stabilityai/stable-diffusion-xl-base-1.0 --task stable-diffusion-xl ./sd_xl_base_onnx
```

### Optimize ONNX Pipeline

Example to optimize the exported float32 ONNX models, and save to float16 models:
```
python -m onnxruntime.transformers.models.stable_diffusion.optimize_pipeline -i ./sd_v1_5/fp32 -o ./sd_v1_5/fp16 --float16
```

For SDXL model, it is recommended to use a machine with 32 GB or more memory to optimize.
```
python optimize_pipeline.py -i ./sd_xl_base_onnx -o ./sd_xl_base_fp16 --float16
```

### Run Benchmark

The benchmark.py script will run a warm-up prompt twice, and measure the peak GPU memory usage in these two runs, then record them as first_run_memory_MB and second_run_memory_MB. Then it will run 5 runs to get average latency (in seconds), and output the results to benchmark_result.csv.

Note that the first run might need more time and memory: For example, cuDNN convolution algorithm search or model compile happens in the first run.

To avoid black image output for Stable Diffusion 2.1 with CUDA EP, we can set an environment variable before inferencing:
```
export ORT_DISABLE_TRT_FLASH_ATTENTION=1
```

Before running benchmark on PyTorch, you need to be logged in via `huggingface-cli login` once.

Example to benchmark the optimized pipeline of stable diffusion 1.5 with batch size 1 on CUDA EP:
```
python benchmark.py -p ./sd_v1_5/fp16 -b 1 -v 1.5
python benchmark.py -b 1 -v 1.5
```
For the first command, '-p' specifies a directory of optimized ONNX pipeline as generated by optimize_pipeline.py.
For the second command without '-p', we will use OnnxruntimeCudaStableDiffusionPipeline to export and optimize ONNX models for clip, unet and vae decoder.

On ROCm EP, use the following command instead:
```
python benchmark.py -p ./sd_v1_5/fp16 -b 1 --tuning --provider rocm -v 1.5
```

For ROCm EP, you can substitute `python benchmark.py` with `python -m onnxruntime.transformers.models.stable_diffusion.benchmark` since
the installed package is built from source. For CUDA, it is recommended to run `python benchmark.py` with the latest benchmark script.

For ROCm EP, the `--tuning` is mandatory because we heavily rely on tuning to find the runable kernels for ORT `OpKernel`s.

The default parameters are stable diffusion version=1.5, height=512, width=512, steps=50, batch_count=5. Run `python benchmark.py --help` for more information.

### Run Benchmark with xFormers

Run PyTorch 1.13.1+cu117 with xFormers like the following

```
pip install xformers==0.0.16
python benchmark.py -e torch -b 1 --use_xformers -v 1.5
```

### Run Benchmark with PyTorch 2.0 with torch.compile

For CUDA:
```
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu117
python benchmark.py -e torch -b 1 --enable_torch_compile -v 1.5
```

For ROCm:
```
pip install torch --upgrade --index-url https://download.pytorch.org/whl/rocm5.4.2
python benchmark.py -e torch -b 1 --enable_torch_compile --provider rocm  -v 1.5
```

Sometime, it complains ptxas not found when there are multiple CUDA versions installed. It can be fixed like `export TRITON_PTXAS_PATH=/usr/local/cuda-11.7/bin/ptxas` before running benchmark.

Note that torch.compile is not supported in Windows: we encountered error `Windows not yet supported for torch.compile`. So it is excluded from RTX 3060 results of Windows.


### Run Benchmark with TensorRT and TensorRT execution provider

For TensorRT installation, follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html.

```
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade polygraphy>=0.47.0 onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
pip install -r requirements-tensorrt.txt
export CUDA_MODULE_LOADING=LAZY
python benchmark.py -e tensorrt -b 1 -v 1.5
python benchmark.py -e onnxruntime -r tensorrt -b 1 -v 1.5
python benchmark.py -e onnxruntime -r tensorrt -b 1 -v 1.5 --enable_cuda_graph
```

### Example Benchmark output

Common settings for below test results:

| model_name                     | disable_safety_checker | height | width | steps | batch_count | num_prompts |
| ------------------------------ | ---------------------- | ------ | ----- | ----- | ----------- | ----------- |
| runwayml/stable-diffusion-v1-5 | TRUE                   | 512    | 512   | 50    | 5           | 1           |

#### Results of RTX 3060 (Windows 11)

| engine      | version                 | provider              | batch size | average latency | first run memory MB | second run memory MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | ------------------- | -------------------- |
| onnxruntime | 1.14.1                  | CUDA                  | 1          | 4.8             | 4,117               | 4,625                |
| torch       | 2.0.0+cu117             | default               | 1          | 5.6             | 4,325               | 4,047                |
| torch       | 1.13.1+cu117            | xformers              | 1          | 6.0             | 9,124               | 9,130                |
| onnxruntime | 1.14.1                  | CUDA                  | 4          | 17.7            | 6,659               | 6,659                |
| torch       | 2.0.0+cu117             | default               | 4          | 20.1            | 6,421               | 6,907                |
| torch       | 1.13.1+cu117            | xformers              | 4          | 21.6            | 10,407              | 10,409               |
| onnxruntime | 1.14.1                  | CUDA                  | 8          | 33.5            | 6,663               | 6,663                |
| torch       | 2.0.0+cu117             | default               | 8          | 39.5            | 10,767              | 10,813               |
| torch       | 1.13.1+cu117            | xformers              | 8          | 41.1            | 10,825              | 9,255                |


#### Results of A100-SXM4-40GB (Ubuntu 20.04)
| engine      | version                 | provider              | batch size | average latency | first run memory MB | second run memory MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | ------------------- | -------------------- |
| onnxruntime | 1.14.1                  | CUDA                  | 1          | 1.1             | 6,883               | 7,395                |
| torch       | 2.0.0+cu117             | default               | 1          | 1.5             | 13,828              | 4,400                |
| torch       | 2.0.0+cu117             | compile               | 1          | 1.8             | 13,892              | 4,386                |
| onnxruntime | 1.14.1                  | CUDA                  | 4          | 3.7             | 7,381               | 7,381                |
| torch       | 2.0.0+cu117             | default               | 4          | 3.9             | 31,278              | 6,870                |
| torch       | 2.0.0+cu117             | compile               | 4          | 3.4             | 31,364              | 6,880                |
| onnxruntime | 1.14.1                  | CUDA                  | 8          | 6.9             | 7,411               | 7,411                |
| torch       | 2.0.0+cu117             | default               | 8          | 7.6             | 31,660              | 10,122               |
| torch       | 2.0.0+cu117             | compile               | 8          | 6.5             | 31,800              | 10,308               |
| onnxruntime | 1.14.1                  | CUDA                  | 16         | 13.6            | 11,479              | 11,479               |
| torch       | 2.0.0+cu117             | default               | 16         | 14.8            | 32,306              | 16,520               |
| torch       | 2.0.0+cu117             | compile               | 16         | 12.6            | 32,636              | 16,898               |

#### Results of A100-PCIE-80GB (Ubuntu 20.04)
| engine      | version                 | provider              | batch size | average latency | first run memory MB | second run memory MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | ------------------- | -------------------- |
| tensorrt    | 8.6.1                   | default               | 1          | 1.00            | 9,056               | 9,056                |
| onnxruntime | 1.16.0 nightly          | tensorrt              | 1          | 1.09            | 11,250              | 11,250               |
| onnxruntime | 1.16.0 nightly          | tensorrt (cuda graph) | 1          | 0.96            | 11,382              | 11,382               |
| onnxruntime | 1.16.0 nightly          | cuda                  | 1          | 1.11            | 4,760               | 5,144                |
| onnxruntime | 1.16.0 nightly          | cuda (cuda graph)     | 1          | 1.04            | 5,230               | 5,390                |
| tensorrt    | 8.6.1                   | default               | 4          | 3.39            | 9,072               | 9,072                |
| onnxruntime | 1.16.0 nightly          | tensorrt              | 4          | 3.60            | 11,266              | 11,266               |
| onnxruntime | 1.16.0 nightly          | tensorrt (cuda graph) | 4          | 3.43            | 11,428              | 11,428               |

#### Results of V100-PCIE-16GB (Ubuntu 20.04)

Results from Standard_NC6s_v3 Azure virtual machine:

| engine      | version                 | provider              | batch size | average latency | first run memory MB | second run memory MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | ------------------- | -------------------- |
| onnxruntime | 1.14.1                  | CUDA                  | 1          | 2.7             | 12,646              | 7,152                |
| torch       | 2.0.0+cu117             | compile               | 1          | 3.2             | 13,317              | 3,909                |
| torch       | 2.0.0+cu117             | default               | 1          | 2.7             | 13,343              | 3,921                |
| torch       | 1.13.1+cu117            | xformers              | 1          | 3.5             | 14,979              | 10,449               |
| onnxruntime | 1.14.1                  | CUDA                  | 4          | 8.4             | 7,114               | 7,114                |
| torch       | 2.0.0+cu117             | compile               | 4          | 8.0             | 13,897              | 6,821                |
| torch       | 2.0.0+cu117             | default               | 4          | 8.7             | 13,873              | 6,607                |
| torch       | 1.13.1+cu117            | xformers              | 4          | 9.1             | 12,969              | 8,421                |
| onnxruntime | 1.14.1                  | CUDA                  | 8          | 15.9            | 7,120               | 7,120                |
| torch       | 2.0.0+cu117             | compile               | 8          | 15.5            | 14,669              | 10,355               |
| torch       | 2.0.0+cu117             | default               | 8          | 17.0            | 14,469              | 9,657                |
| torch       | 1.13.1+cu117            | xformers              | 8          | 17.4            | 15,593              | 9,133                |

#### Results of T4 (Ubuntu 20.04)

To make the result stable, we lock the frequency of T4 GPU like
`sudo nvidia-smi --lock-gpu-clocks=990` for fair comparison. See [nvidia blog](https://developer.nvidia.com/blog/advanced-api-performance-setstablepowerstate/) for more information. Note that performance might be slightly better without locking frequency.

Results are from Standard_NC4as_T4_v3 Azure virtual machine:

| engine      | version                 | provider              | batch size | average latency | first run memory MB | second run memory MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | ------------------- | -------------------- |
| onnxruntime | 1.14.1                  | CUDA                  | 1          | 5.6             | 4,925               | 4,925                |
| onnxruntime | 1.15.1                  | CUDA                  | 1          | 5.5             | 3,738               | 4,250                |
| onnxruntime | 1.15.1 (tensorrt 8.6.1) | Tensorrt              | 1          | 4.8             | 10,710              | 10,710               |
| onnxruntime | 1.16.0 nightly          | Tensorrt (cuda graph) | 1          | 4.7             | 11,746              | 10,746               |
| tensorrt    | 8.6.1                   | default               | 1          | 5.0             | 8,530               | 8,530                |
| torch       | 1.13.1+cu117            | xformers              | 1          | 6.9             | 14,845              | 10,317               |
| torch       | 2.0.0+cu117             | compile               | 1          | 6.0             | 12,989              | 3,841                |
| torch       | 2.0.0+cu117             | default               | 1          | 6.4             | 12,987              | 3,841                |
| onnxruntime | 1.14.1                  | CUDA                  | 4          | 23.0            | 6,977               | 6,977                |
| onnxruntime | 1.15.1                  | CUDA                  | 4          | 22.6            | 6,298               | 6,298                |
| onnxruntime | 1.15.1 (tensorrt 8.6.1) | Tensorrt              | 4          | 21.8            | 10,746              | 10,746               |
| tensorrt    | 8.6.1                   | default               | 4          | 22.2            | 8,542               | 8,542                |
| torch       | 1.13.1+cu117            | xformers              | 4          | 25.8            | 12,819              | 8,269                |
| torch       | 2.0.0+cu117             | compile               | 4          | 22.2            | 14,637              | 6,583                |
| torch       | 2.0.0+cu117             | default               | 4          | 25.2            | 14,409              | 6,355                |
| onnxruntime | 1.14.1                  | CUDA                  | 8          | 46.4            | 6,779               | 6,779                |
| torch       | 1.13.1+cu117            | xformers              | 8          | 51.4            | 14,827              | 9,001                |
| torch       | 2.0.0+cu117             | compile               | 8          | 46.5            | 12,595              | 10,171               |
| torch       | 2.0.0+cu117             | default               | 8          | 50.7            | 11,955              | 9,531                |

#### Results of MI250X, 1 GCD (Ubuntu 20.04)

| engine      | version                 | provider              | batch size | average latency | first run memory MB | second run memory MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | ------------------- | -------------------- |
| onnxruntime | 1.15.0+rocm5.4.2        | ROCM                  | 1          | 2.2             | 5,548               | 4,908                |
| torch       | 1.12.1+rocm5.4          | -                     | 1          | 3.4             | 6,653               | 4,613                |
| torch       | 2.0.0+rocm5.4.2         | default               | 1          | 3.2             | 5,977               | 4,368                |
| torch       | 2.0.0+rocm5.4.2         | compile               | 1          | 3.0             | 5,869               | 4,266                |
| onnxruntime | 1.15.0+rocm5.4.2        | ROCM                  | 4          | 6.6             | 5,546               | 4,906                |
| torch       | 1.12.1+rocm5.4          | -                     | 4          | 10.1            | 19,477              | 11,325               |
| torch       | 2.0.0+rocm5.4.2         | default               | 4          | 10.5            | 13,051              | 7,300                |
| torch       | 2.0.0+rocm5.4.2         | compile               | 4          | 9.2             | 12,879              | 7,190                |
| onnxruntime | 1.15.0+rocm5.4.2        | ROCM                  | 8          | 12.5            | 9,778               | 9,006                |
| torch       | 1.12.1+rocm5.4          | -                     | 8          | 19.3            | 55,851              | 20,014               |
| torch       | 2.0.0+rocm5.4.2         | default               | 8          | 20.3            | 23,551              | 11,930               |
| torch       | 2.0.0+rocm5.4.2         | compile               | 8          | 17.8            | 23,303              | 11,800               |

#### Results of MI100 (Ubuntu 20.04)

| engine      | version                 | provider              | batch size | average latency | first run memory MB | second run memory MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | ------------------- | -------------------- |
| onnxruntime | 1.15.0+rocm5.4.2        | ROCM                  | 1          | 2.4             | 5,254               | 4,614                |
| torch       | 1.12.1+rocm5.4          | -                     | 1          | 3.5             | 5,771               | 4,672                |
| torch       | 2.0.0+rocm5.4.2         | default               | 1          | 3.5             | 5,811               | 4,206                |
| torch       | 2.0.0+rocm5.4.2         | compile               | 1          | 3.1             | 5,774               | 4,168                |
| onnxruntime | 1.15.0+rocm5.4.2        | ROCM                  | 4          | 7.5             | 7,290               | 6,646                |
| torch       | 1.12.1+rocm5.4          | -                     | 4          | 10.7            | 19,334              | 11,181               |
| torch       | 2.0.0+rocm5.4.2         | default               | 4          | 11.5            | 12,881              | 7,151                |
| torch       | 2.0.0+rocm5.4.2         | compile               | 4          | 10.0            | 12,740              | 7,073                |
| onnxruntime | 1.15.0+rocm5.4.2        | ROCM                  | 8          | 14.4            | 7,320               | 6,676                |
| torch       | 1.12.1+rocm5.4          | -                     | 8          | 20.2            | 31,820              | 19,908               |
| torch       | 2.0.0+rocm5.4.2         | default               | 8          | 22.2            | 23,415              | 11,815               |
| torch       | 2.0.0+rocm5.4.2         | compile               | 8          | 19.3            | 23,154              | 11,667               |

### Credits

Some CUDA kernels (Flash Attention, GroupNorm, SplitGelu and BiasAdd etc.) were originally implemented in [TensorRT](https://github.com/nviDIA/TensorRT) by Nvidia.
We use Memory efficient attention from [CUTLASS](https://github.com/NVIDIA/cutlass). The kernels were developed by Meta xFormers.
The ONNX export script and pipeline for stable diffusion was developed by Huggingface [diffusers](https://github.com/huggingface/diffusers) library.

Most ROCm kernel optimizations are from [composable kernel](https://github.com/ROCmSoftwarePlatform/composable_kernel).
Some kernels are enabled by MIOpen. We hereby thank for the AMD developers' collaboration.

### Future Works

There are other optimizations might improve the performance or reduce memory footprint:
* Export the whole pipeline into a single ONNX model. Currently, there are multiple ONNX models (CLIP, VAE and U-Net etc). Each model uses separated thread pool and memory allocator. Combine them into one model could share thread pool and memory allocator. The end result is more efficient and less memory footprint.
* For Stable Diffusion 2.1, we disable TensorRT flash attention kernel and use only memory efficient attention. It is possible to add flash attention in Windows to improve performance.
* Reduce GPU memory footprint by actively deleting buffers for intermediate results.
* Safety Checker Optimization
* Leverage FP8 in latest GPU
