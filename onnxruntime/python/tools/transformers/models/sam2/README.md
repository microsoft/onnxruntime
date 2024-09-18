# SAM2 ONNX Model Export

## Setup Environment
It is recommend to setup a Linux machine with python 3.10 up to 3.12. Then install [PyTorch](https://pytorch.org/) and [Onnx Runtime](https://onnxruntime.ai/docs/install/#python-installs).

Example commands to prepare environment for CUDA 12.x and cuDNN 9.x (CUDA and cuDNN need installation):
```
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install onnxruntime-gpu opencv-python matplotlib
```

```
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
python3 -m pip install -e .
cd checkpoints
sh ./download_ckpts.sh
```

## Export ONNX
Specify the segment-anything-2 directory created by the above git clone command:
```
python3 convert_to_onnx.py  --sam2_dir path/to/segment-anything-2
```
The exported onnx models can be found in sam2_onnx_models sub-directory. You can change the output directory using `--output_dir` option.

If you want the model output mulitple masks, append `--multimask_output` option.

To see all paramters, run the following command
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
- The exported decoder model does not support batch mode for now.
