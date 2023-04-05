# Step 1: Extract required operators for your network
You have to find out what operators to be included in the ONNXLibrary.
Operators could be extracted via `tools/python/create_reduced_build_config.py`
Let say we have a network (ONNX file) which is stored in `/home/username/models/generator_network.onnx`
You could use the following command and find the configuration file in `/home/username/ops_config/generator_network.ops.config`

```
python3 tools/python/create_reduced_build_config.py /home/username/models/generator_network.onnx /home/username/ops_config/generator_network.ops.config
```

The generator_network.ops.config looks like this: 
```
# Generated from ONNX model/s:
# - /Users/goodnotesci/goodnotes/GoodNotes-5/Packages/GNHandwritingSynthesis/Tests/GNHandwritingSynthesisTests/Resources/models/generator_network.onnx
ai.onnx;9;Add,Cast,Concat,Constant,ConstantOfShape,Div,Exp,Gather,Gemm,LogSoftmax,MatMul,Mul,Neg,NonZero,ReduceSum,Reshape,Shape,Sigmoid,Slice,Softmax,Softplus,Split,Squeeze,Sub,Tanh,Transpose,Unsqueeze
```

`ai.onnx` is the opset domain, and `9` is the opset version. The remainings of the line show the names of the operators.

You could also put a path to model directory in `create_reduced_build_config.py` and the script will search for all models in the directory. 
You could find our latest [`hws_mobile_package.required_operators.config` here.](https://github.com/GoodNotes/onnxruntime/blob/develop/tools/ci_build/github/apple/hws_mobile_package.required_operators.config)

$ python tools/ci_build/github/apple/build_ios_framework.py --config Release --build_dir /Users/goodnotesci/goodnotes/handwriting_synthesis_rust/paco_macabi_arm64_release_v20230327_2320 --include_ops_by_config tools/ci_build/github/apple/hws_mobile_package.required_operators.config --path_to_protoc_exe /usr/local/bin/protoc-3.21.12.0 tools/ci_build/gitn:qhub/apple/default_full_ios_framework_build_settings.json
$ python tools/ci_build/github/apple/build_macabi_framework.py --config Release --build_dir /Users/goodnotesci/goodnotes/handwriting_synthesis_rust/paco_ios_release --include_ops_by_config tools/ci_build/github/apple/hws_mobile_package.required_operators.config --path_to_protoc_exe /usr/local/bin/protoc-3.21.12.0 tools/ci_build/github/apple/default_full_macabi_framework_build_settings.json
$ python tools/ci_build/github/apple/build_macabi_framework.py --config Release --build_dir /Users/goodnotesci/goodnotes/handwriting_synthesis_rust/paco_macabi_arm64_release_v20230327_2320 --include_ops_by_config tools/ci_build/github/apple/hws_mobile_package.required_operators.config --path_to_protoc_exe /usr/local/bin/protoc-3.21.12.0 tools/ci_build/github/apple/default_full_macabi_framework_build_settings.json
