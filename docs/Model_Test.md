ONNX has a collection of standard tests. This document describes how to run these tests through a C++ program named 'onnx_test_runner' in this repo. You could also run these test through onnxruntime python binding, which would be much easier to setup, but, a bit harder to debug issues.

# Get the test data
```
git submodule update --init --recursive
pushd .
cd cmake/external/emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
popd
cd js
npm install
npm run prepare-node-tests
```

In addition to that, You can get more test models with their test data from https://github.com/onnx/models .


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
	-e [EXECUTION_PROVIDER]: EXECUTION_PROVIDER could be 'cpu', 'cuda', 'dnnl' or 'tensorrt'. Default: 'cpu'.    
	-x: Use parallel executor, default (without -x): sequential executor.    
	-h: help    

e.g.           
//run the tests under C:\testdata dir and enable CUDA provider         
$ onnx_test_runner -e cuda C:\testdata

//run the tests sequentially. It would be easier to debug         
$ onnx_test_runner -c 1 -j 1 C:\testdata 
