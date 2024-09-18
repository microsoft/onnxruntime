# SAM2 ONNX Model Export

## Setup Environment
It is recommend to setup a machine with python 3.10, 3.11 or 3.12. Then install [PyTorch 2.4.1](https://pytorch.org/) and [Onnx Runtime 1.19.2](https://onnxruntime.ai/docs/install/#python-installs).

Example commands to prepare environment for CUDA 12.x and cuDNN 9.x (CUDA and cuDNN need installation):
```
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install onnxruntime-gpu opencv-python matplotlib
```

Clone the SAM 2 git repository, and download checkpoints:
```
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
python3 -m pip install -e .
cd checkpoints
sh ./download_ckpts.sh
```

In Windows, you can run the following to replace `sh ./download_ckpts.sh`:
```
curl https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt > sam2_hiera_tiny.pt
curl https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt > sam2_hiera_small.pt
curl https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt > sam2_hiera_base_plus.pt
curl https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt > sam2_hiera_large.pt
```

## Export ONNX
Run convert_to_onnx.py, and specify the segment-anything-2 directory created by the above git clone command:
```
python3 convert_to_onnx.py  --sam2_dir path/to/segment-anything-2
```
The exported onnx models can be found in sam2_onnx_models sub-directory. You can change the output directory using `--output_dir` option.

If you want the model outputs multiple masks, append the `--multimask_output` option.

To see all parameters, run the following command:
```
python3 convert_to_onnx.py  -h
```

## Run Demo
The exported onnx model can run in CPU. The demo currently requires Nvidia GPU.
```
wget https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/notebooks/images/truck.jpg
python3 convert_to_onnx.py  --sam2_dir path/to/segment-anything-2 --demo
```

## Limitations
- The exported image_decoder model does not support batch mode for now.
