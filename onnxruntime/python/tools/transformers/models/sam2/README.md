# SAM2 ONNX Model Export

## Setup Environment
It is recommend to setup a machine with python 3.10, 3.11 or 3.12. Then install [PyTorch 2.4.1](https://pytorch.org/) and [Onnx Runtime 1.19.2].

### CPU Only
To install the CPU-only version of PyTorch and Onnx Runtime for exporting and running ONNX models, use the following commands:
```
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install onnxruntime onnx opencv-python matplotlib
```

### GPU
If your machine has an NVIDIA GPU, you can install the CUDA version of PyTorch and Onnx Runtime for exporting and running ONNX models:

```
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install onnxruntime-gpu onnx opencv-python matplotlib
```

onnxruntime-gpu requires CUDA 12.x, cuDNN 9.x, and other dependencies (such as MSVC Runtime on Windows). For more information, see the [installation guide](https://onnxruntime.ai/docs/install/#python-installs).

## Download Checkpoints

Clone the SAM2 git repository and download the checkpoints:
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
export sam2_dir=$PWD
python3 -m pip install -e .
cd checkpoints
sh ./download_ckpts.sh
```

On Windows, you can replace `sh ./download_ckpts.sh` with the following commands:
```bash
curl https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt > sam2_hiera_tiny.pt
curl https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt > sam2_hiera_small.pt
curl https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt > sam2_hiera_base_plus.pt
curl https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt > sam2_hiera_large.pt
```

## Export ONNX
To export ONNX models, run the convert_to_onnx.py script and specify the segment-anything-2 directory created by the above git clone command:
```bash
python3 convert_to_onnx.py  --sam2_dir $sam2_dir
```

The exported ONNX models will be found in the sam2_onnx_models sub-directory. You can change the output directory using the `--output_dir` option.

If you want the model outputs multiple masks, append the `--multimask_output` option.

To see all parameters, run the following command:
```bash
python3 convert_to_onnx.py  -h
```

## Optimize ONNX

To optimize the onnx models for CPU with float32 data type:
```bash
python3 convert_to_onnx.py  --sam2_dir $sam2_dir --optimize --dtype fp32
```

To optimize the onnx models for GPU with float16 data type:
```bash
python3 convert_to_onnx.py  --sam2_dir $sam2_dir --optimize --dtype fp16 --use_gpu
```

Another option is to use optimizer.py like the following:
```
cd ../..
python optimizer.py --input models/sam2/sam2_onnx_models/sam2_hiera_large_image_encoder.onnx \
                    --output models/sam2/sam2_onnx_models/sam2_hiera_large_image_encoder_fp16_gpu.onnx \
                    --use_gpu --model_type sam2 --float16
```
The optimizer.py could be helpful when you have SAM2 onnx models that is exported by other tools.

## Run Demo

The exported ONNX models can run on a CPU. The demo will output sam2_demo.png.
```bash
curl https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/notebooks/images/truck.jpg > truck.jpg
python3 convert_to_onnx.py  --sam2_dir $sam2_dir --demo
```

It is able to run demo on optimized model as well. For example,
```bash
python3 convert_to_onnx.py  --sam2_dir $sam2_dir --optimize --dtype fp16 --use_gpu --demo
```

## Benchmark and Profiling

We can create a conda environment then run GPU benchmark like the following:
```bash
conda create -n sam2_gpu python=3.11 -y
conda activate sam2_gpu
install_dir=$HOME
bash benchmark_sam2.sh $install_dir gpu
```

or create a new conda environment for CPU benchmark:
```bash
conda create -n sam2_cpu python=3.11 -y
conda activate sam2_cpu
bash benchmark_sam2.sh $HOME cpu
```

The usage of the script like the following:
```
bash benchmark_sam2.sh <install_dir> <cpu_or_gpu> [profiling] [benchmarking] [nightly] [dynamo]
```

| Parameter| Default  | Description |
|----------|----------| ------------|
| install_dir | $HOME | a directory to clone git repositories or install CUDA/cuDNN for benchmark |
| cpu_or_gpu | gpu | the device to run benchmark. The value can be either "gpu" or "cpu" |
| profiling | false | run gpu profiling |
| benchmarking | true | run benchmark |
| nightly | false | install onnxruntime nightly or official release package |
| dynamo | false | export image encoder using dynamo or not. |

The dynamo export is experimental since graph optimization still need extra works for this model.

Output files:
* sam2_cpu_[timestamp].csv or sam2_gpu_[timestamp].csv has benchmark results. Use Excel to load the file to view it.
* onnxruntime_image_[encoder|decoder].json has ONNX Runtime profiling results. Use `chrome://tracing` in Chrome browser to view it.
* torch_image_[encoder|decoder].json has PyTorch profiling results. Use `chrome://tracing` in Chrome browser to view it.
* sam2_fp16_profile_image_[encoder|decoder]_[ort|torch]_gpu.[nsys-rep|sqlite] has NVTX profiling. Use Nvidia NSight System to view it.
* torch_image_encoder_compiled_code.txt has the compiled kernel code from Pytorch.

## Limitations
- The exported image_decoder model does not support batch mode for now.
