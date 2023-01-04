# The Quadric Version of onnxruntime

This repository contains the a distribution of onnxruntime with additional operator quantization capabilities.


## Prerequisites:
- python 3.9
- pip

## Clone repository and build:
```
git clone --recursive https://github.com/quadric-io/onnxruntime onnxruntime
cd onnxruntime
# Install wheel
pip install wheel
# Build the python package
./build.sh --build_wheel --config Release --parallel
```

## Install 
```
pip install build/MacOS/Release/dist/onnxruntime-1.14.0-cp39-cp39-macosx_11_0_x86_64.whl
```
