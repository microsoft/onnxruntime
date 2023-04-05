# Prepare ONNXRuntime XCFramework for iOS or MacCatalyst
The instructions should be executed on a Mac machine.

## Step 1: Extract required operators for your network
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
As you may notice some additional operators come from `com.microsoft` domain,
```
# internal ops added by optimizers
# Note: LayerNormalization is an internal op even though it is (incorrectly) registered in the ONNX domain.
ai.onnx;1;LayerNormalization
com.microsoft;1;DynamicQuantizeMatMul,FusedConv,FusedGemm,FusedMatMul,Gelu,MatMulIntegerToFloat,NhwcMaxPool,QLinearAdd,QLinearAveragePool,QLinearConv,QLinearGlobalAveragePool,QLinearMul,QLinearSigmoid,QuickGelu

# NHWC transformer also uses this, so assuming it's valuable enough to include
com.microsoft;1;QLinearLeakyRelu
```
These operators are discovered when the library is deployed in iOS. It was found that ONNXRuntime will optimize the graph so you should add these custom operators back to `hws_mobile_package.required_operators.config`.

## Step 2: Compile ONNXRuntime XCFramework
For iOS simulator and iOS devices, we could follow the official tutorial to build the XCFramework. 
In particular, you could use `build_ios_framework.py` compiles the library with the following command:
```
python tools/ci_build/github/apple/build_ios_framework.py --config Release --build_dir /home/username/onnxlibrary/ios_release_v20230327_2320 --include_ops_by_config tools/ci_build/github/apple/hws_mobile_package.required_operators.config --path_to_protoc_exe /usr/local/bin/protoc-3.21.12.0 tools/ci_build/github/apple/default_full_ios_framework_build_settings.json
```

MacCatalyst is not supported officially. `build_macabi_framework.py` has been created for our use.
To compile, use the follow command
```
python tools/ci_build/github/apple/build_macabi_framework.py --config Release --build_dir /home/username/onnxlibrary/macabi_release_v20230327_2320 --include_ops_by_config tools/ci_build/github/apple/hws_mobile_package.required_operators.config --path_to_protoc_exe /usr/local/bin/protoc-3.21.12.0 tools/ci_build/github/apple/default_full_macabi_framework_build_settings.json
```

Both of these scripts will invoke `tools/ci_build/build.py`. [This section](https://github.com/GoodNotes/onnxruntime/blob/eeca6fea2b4d02ddc729c7a7cdc39b123d23fbf8/tools/ci_build/build.py#L1221) contains the target platform-specific parameters when executing CMake.

## Step 3: Use it in a Swift package in GoodNotes
Since we compile the framework for different target separately, we need to merge the iOS xcframework and MacCatlyst xcframework manually at the moment.
Please refer to this [PR#12971](https://github.com/GoodNotes/GoodNotes-5/pull/12971) for the directory structure.
You would also need to combine the `Info.plist` manually.

At last, you can import `onnxruntime` as a binary target.
```
targets: [
        .binaryTarget(
            name: "onnxruntime",
            path: "onnxruntime/onnxruntime.xcframework"),
 ]
 ```




