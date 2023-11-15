# Stable Diffusion GPU Optimization

ONNX Runtime uses the following optimizations to speed up Stable Diffusion in CUDA:

* [Flash Attention](https://arxiv.org/abs/2205.14135) for float16 precision. Flash Attention uses tiling to reduce number of GPU memory reads/writes, and improves performance with less memory for long sequence length. The kernel requires GPUs of Compute Capability >= 7.5 (like T4, A100, and RTX 2060~4090).
* [Memory Efficient Attention](https://arxiv.org/abs/2112.05682v2) for float32 precision or older GPUs (like V100). We used the fused multi-head attention kernel in CUTLASS, and the kernel was contributed by xFormers.
* Channel-last (NHWC) convolution. For NVidia GPU with Tensor Cores support, NHWC tensor layout is recommended for convolution. See [Tensor Layouts In Memory: NCHW vs NHWC](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout).
* GroupNorm for NHWC tensor layout, and SkipGroupNorm fusion which fuses GroupNorm with Add bias and residual inputs
* SkipLayerNormalization which fuses LayerNormalization with Add bias and residual inputs.
* BiasSplitGelu is a fusion of Add bias with SplitGelu activation.
* BiasAdd fuses Add bias and residual.
* Reduce Transpose nodes by graph transformation.

These optimizations are firstly carried out on CUDA EP. They may not work on other EP.

## Scripts:

| Script                                         | Description                                                                               |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [demo_txt2img_xl.py](./demo_txt2img_xl.py)     | Demo of text to image generation using Stable Diffusion XL model.                         |
| [demo_txt2img.py](./demo_txt2img.py)           | Demo of text to image generation using Stable Diffusion models except XL.                 |
| [optimize_pipeline.py](./optimize_pipeline.py) | Optimize Stable Diffusion ONNX models exported from Huggingface diffusers or optimum      |
| [benchmark.py](./benchmark.py)                 | Benchmark latency and memory of OnnxRuntime, xFormers or PyTorch 2.0 on stable diffusion. |


## Run demo with docker

#### Clone the onnxruntime repository
```
git clone https://github.com/microsoft/onnxruntime
cd onnxruntime
```

#### Launch NVIDIA pytorch container

Install nvidia-docker using [these instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

```
docker run --rm -it --gpus all -v $PWD:/workspace nvcr.io/nvidia/pytorch:23.10-py3 /bin/bash
```

#### Build onnxruntime from source
After launching the docker, you can build and install onnxruntime-gpu wheel like the following.
```
export CUDACXX=/usr/local/cuda-12.2/bin/nvcc
git config --global --add safe.directory '*'
sh build.sh --config Release  --build_shared_lib --parallel --use_cuda --cuda_version 12.2 \
            --cuda_home /usr/local/cuda-12.2 --cudnn_home /usr/lib/x86_64-linux-gnu/ --build_wheel --skip_tests \
            --use_tensorrt --tensorrt_home /usr/src/tensorrt \
            --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
            --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=80 \
            --allow_running_as_root
python3 -m pip install --upgrade pip
python3 -m pip install build/Linux/Release/dist/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl --force-reinstall
```

If the GPU is not A100, change `CMAKE_CUDA_ARCHITECTURES=80` in the command line according to the GPU compute capacity.

#### Install required packages
```
cd /workspace/onnxruntime/python/tools/transformers/models/stable_diffusion
python3 -m pip install -r requirements-cuda12.txt
python3 -m pip install --upgrade polygraphy onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
```

### Run Demo

You can review the usage of supported pipelines like the following:
```
python3 demo_txt2img.py --help
python3 demo_txt2img_xl.py --help
```

For example:
`--engine {ORT_CUDA,ORT_TRT,TRT}` can be used to choose different backend engines including CUDA or TensorRT execution provider of ONNX Runtime, or TensorRT.
`--work-dir WORK_DIR` can be used to load or save models under the given directory. You can download the [optimized ONNX models of Stable Diffusion XL 1.0](https://huggingface.co/tlwu/stable-diffusion-xl-1.0-onnxruntime#usage-example) to save time in running the XL demo.

#### Generate an image guided by a text prompt
```python3 demo_txt2img.py "astronaut riding a horse on mars"```

#### Generate an image with Stable Diffusion XL guided by a text prompt
```python3 demo_txt2img_xl.py "starry night over Golden Gate Bridge by van gogh"```

If you do not provide prompt, the script will generate different image sizes for a list of prompts for demonstration.

## Optimize Stable Diffusion ONNX models for Hugging Face Diffusers or Optimum

If you are able to run the above demo with docker, you can use the docker and skip the following setup and fast forward to [Export ONNX pipeline](#export-onnx-pipeline).

Below setup does not use docker. We'll use the environment to optimize ONNX models of Stable Diffusion exported by huggingface diffusers or optimum.
For Windows OS, please change the format of path to be like `.\sd` instead of `./sd`.

It is recommended to create a Conda environment with Python 3.10 for the following setup:
```
conda create -n py310 python=3.10
conda activate py310
```

### Setup Environment (CUDA) without docker

First, we need install CUDA 11.8 or 12.1, [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) 8.5 or above, and [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) in the machine.

#### CUDA 11.8:

In the Conda environment, install PyTorch 2.1 or above, and other required packages like the following:
```
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade polygraphy onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
pip install -r requirements-cuda11.txt
```

For Windows, install nvtx like the following:
```
conda install -c conda-forge nvtx
```

We cannot directly `pip install tensorrt` for CUDA 11. Follow https://github.com/NVIDIA/TensorRT/issues/2773 to install TensorRT for CUDA 11 in Linux.

For Windows, pip install the tensorrt wheel in the downloaded TensorRT zip file instead. Like `pip install tensorrt-8.6.1.6.windows10.x86_64.cuda-11.8\tensorrt-8.6.1.6\python\tensorrt-8.6.1-cp310-none-win_amd64.whl`.

#### CUDA 12.*:
The official package of onnxruntime-gpu 1.16.* is built for CUDA 11.8. To use CUDA 12.*, you will need [build onnxruntime from source](https://onnxruntime.ai/docs/build/inferencing.html).

```
git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
pip install cmake
pip install -r requirements-dev.txt
```
Follow [example script for A100 in Ubuntu](https://github.com/microsoft/onnxruntime/blob/26a7b63716e3125bfe35fe3663ba10d2d7322628/build_release.sh)
or [example script for RTX 4090 in Windows](https://github.com/microsoft/onnxruntime/blob/8df5f4e0df1f3b9ceeb0f1f2561b09727ace9b37/build_trt.cmd) to build and install onnxruntime-gpu wheel.

Then install other python packages like the following:
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade polygraphy onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
pip install -r requirements-cuda12.txt
```
Finally, `pip install tensorrt` for Linux. For Windows, pip install the tensorrt wheel in the downloaded TensorRT zip file instead.

### Setup Environment (ROCm)

It is recommended that the users run the model with ROCm 5.4 or newer and Python 3.10.
Note that Windows is not supported for ROCm at the moment.

```
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

In all examples below, we run the scripts in source code directory. You can get source code like the following:
```
git clone https://github.com/microsoft/onnxruntime
cd onnxruntime/onnxruntime/python/tools/transformers/models/stable_diffusion
```

For SDXL model, it is recommended to use a machine with 48 GB or more memory to optimize.
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


### Run Benchmark with TensorRT or TensorRT execution provider

For TensorRT installation, follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html.

```
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade polygraphy onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
pip install -r requirements-tensorrt.txt
export CUDA_MODULE_LOADING=LAZY
python benchmark.py -e tensorrt -b 1 -v 1.5
python benchmark.py -e onnxruntime -r tensorrt -b 1 -v 1.5
python benchmark.py -e onnxruntime -r tensorrt -b 1 -v 1.5 --enable_cuda_graph

python benchmark.py -e tensorrt --height 1024 --width 1024 -s 30  -b 1 -v xl-1.0 --enable_cuda_graph
python benchmark.py -e onnxruntime -r tensorrt --height 1024 --width 1024 -s 30  -b 1 -v xl-1.0 --enable_cuda_graph
```

### Results on RTX 3060 GPU:

To show the impact of each optimization on latency and GPU memory, we did some experiments:

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

Some CUDA kernels (TensorRT Fused Attention, GroupNorm, SplitGelu and BiasAdd etc.) and demo diffusion were originally implemented in [TensorRT](https://github.com/nviDIA/TensorRT) by Nvidia.
We use [Flash Attention v2](https://github.com/Dao-AILab/flash-attention) in Linux.
We use Memory efficient attention from [CUTLASS](https://github.com/NVIDIA/cutlass). The kernels were developed by Meta xFormers.
The ONNX export script and pipeline for stable diffusion was developed by Huggingface [diffusers](https://github.com/huggingface/diffusers) library.

Most ROCm kernel optimizations are from [composable kernel](https://github.com/ROCmSoftwarePlatform/composable_kernel).
Some kernels are enabled by MIOpen. We hereby thank for the AMD developers' collaboration.

### Future Works
* Update demo to support inpainting, LoRA Weights and Control Net.
* Support flash attention in Windows.
* Integration with UI.
* Optimization for H100 GPU.
* Export the whole pipeline into a single ONNX model. This senario is mainly for mobile device.
