ONNX has a collection of standard tests. This document describes how to run these tests through a C++ program named 'onnx_test_runner' in this repo. You could also run these test through onnxruntime python binding, which would be much easier to setup, but, a bit harder to debug issues.

# Get the test data
You should have:
1. onnx single node test data
2. onnx model zoo models

## Install onnx python package
You can get onnx python package from [pypi](https://pypi.org/). However, if you are a onnxruntime developer, you may need to work on a cutting edge ONNX version. In this case, you need to build and install ONNX from source code.

### Install ONNX from source code
1. (windows) set ONNX_ML=1
   (linux) export ONNX_ML=1
2. Install protobuf and put protoc into your PATH environment. When you compile protobuf, it's better to only enable the static libraries. 
3. run "python setup.py bdist_wheel" and "pip install dist/*.whl"

## Generate node test data
$ python3 -m onnx.backend.test.cmd_tools generate-data -o <dest_folder>    
e.g.    
   python3 -m onnx.backend.test.cmd_tools generate-data -o C:\testdata


## Get the onnx model zoo models
Download the files from: https://github.com/onnx/models. Unzip them.
(TODO: put a full copy on Azure blob, instead of downloading these files from different sources individually)

# Compile onnx_test_runner and run the tests
onnx_test_runner is a C++ program. Its source code is in onnxruntime/test/onnx directory.

Usage: onnx_test_runner [options...] <data_root>    
Options:    
	-j [models]: Specifies the number of models to run simultaneously.    
	-A : Disable memory arena    
	-c [runs]: Specifies the number of Session::Run() to invoke simultaneously for each model.    
	-r [repeat]: Specifies the number of times to repeat    
	-v: verbose    
	-n [test_case_name]: Specifies a single test case to run.    
	-e [EXECUTION_PROVIDER]: EXECUTION_PROVIDER could be 'cpu', 'cuda', 'mkldnn' or 'tensorrt'. Default: 'cpu'.    
	-x: Use parallel executor, default (without -x): sequential executor.    
	-h: help    

e.g.  
//run the tests under C:\testdata dir and enable CUDA provider
$ onnx_test_runner -e cuda C:\testdata

//run the tests sequentially. It would be easier to debug
$ onnx_test_runner -c 1 -j 1 C:\testdata 
