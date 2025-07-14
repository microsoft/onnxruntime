---
title: Apple - CoreML
description: Instructions to execute ONNX Runtime with CoreML
parent: Execution Providers
nav_order: 501
redirect_from: /docs/reference/execution-providers/CoreML-ExecutionProvider
---
{::options toc_levels="2" /}

# CoreML Execution Provider

[Core ML](https://developer.apple.com/machine-learning/core-ml/) is a machine learning framework introduced by Apple. It is designed to seamlessly take advantage of powerful hardware technology including CPU, GPU, and Neural Engine, in the most efficient way in order to maximize performance while minimizing memory and power consumption.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements

The CoreML Execution Provider (EP) requires iOS devices with iOS 13 or higher, or Mac computers with macOS 10.15 or higher.

It is recommended to use Apple devices equipped with Apple Neural Engine to achieve optimal performance.

## Install

Pre-built binaries of ONNX Runtime with CoreML EP for iOS are published to CocoaPods.

See [here](../install/index.md#install-on-ios) for installation instructions.

## Build

For build instructions for iOS devices, please see [Build for iOS](../build/ios.md#coreml-execution-provider).

## Usage

The ONNX Runtime API details are [here](../api).

The CoreML EP can be used via the C, C++, Objective-C, C# and Java APIs.

The CoreML EP must be explicitly registered when creating the inference session. For example:

```C++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions so;
std::unordered_map<std::string, std::string> provider_options;
provider_options["ModelFormat"] = std::to_string("MLProgram");
so.AppendExecutionProvider("CoreML", provider_options);
Ort::Session session(env, model_path, so);
```


**Deprecated** APIs `OrtSessionOptionsAppendExecutionProvider_CoreML` in ONNX Runtime 1.20.0. Please use `OrtSessionOptionsAppendExecutionProvider` instead.
```C++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions so;
uint32_t coreml_flags = 0;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(so, coreml_flags));
Ort::Session session(env, model_path, so);
```

## Configuration Options (NEW API)

There are several run time options available for the CoreML EP.

To use the CoreML EP run time options, create an unsigned integer representing the options, and set each individual option by using the bitwise OR operator.

ProviderOptions can be set by passing string to the `AppendExecutionProvider` method.
```c++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions so;
std::string model_path = "/a/b/c/model.onnx";
std::unordered_map<std::string, std::string> provider_options;
provider_options["ModelFormat"] = "MLProgram";
provider_options["MLComputeUnits"] = "ALL";
provider_options["RequireStaticInputShapes"] = "0";
provider_options["EnableOnSubgraphs"] = "0";
so.AppendExecutionProvider("CoreML", provider_options);
Ort::Session session(env, model_path, so);
```

Python inference example code to use the CoreML EP run time options:
```python
import onnxruntime as ort
model_path = "model.onnx"
providers = [
    ('CoreMLExecutionProvider', {
        "ModelFormat": "MLProgram", "MLComputeUnits": "ALL", 
        "RequireStaticInputShapes": "0", "EnableOnSubgraphs": "0"
    }),
]

session = ort.InferenceSession(model_path, providers=providers)
outputs = ort_sess.run(None, input_feed)
```

### Available Options (NEW API)
`ModelFormat` can be one of the following values: (`NeuralNetwork` by default )
- `MLProgram`: Create an MLProgram format model. Requires Core ML 5 or later (iOS 15+ or macOS 12+).
- `NeuralNetwork`: Create a NeuralNetwork format model. Requires Core ML 3 or later (iOS 13+ or macOS 10.15+).

`MLComputeUnits` can be one of the following values: (`ALL` by default )
- `CPUOnly`: Limit CoreML to running on CPU only.
- `CPUAndNeuralEngine`: Enable CoreML EP for Apple devices with a compatible Apple Neural Engine (ANE).
- `CPUAndGPU`: Enable CoreML EP for Apple devices with a compatible GPU.
- `ALL`: Enable CoreML EP for all compatible Apple devices.

`RequireStaticInputShapes` can be one of the following values: (`0` by default )

Only allow the CoreML EP to take nodes with inputs that have static shapes.
By default the CoreML EP will also allow inputs with dynamic shapes, however performance may be negatively impacted by inputs with dynamic shapes.

- `0`: Allow the CoreML EP to take nodes with inputs that have dynamic shapes.
- `1`: Only allow the CoreML EP to take nodes with inputs that have static shapes.


`EnableOnSubgraphs` can be one of the following values: (`0` by default )

Enable CoreML EP to run on a subgraph in the body of a control flow operator (i.e. a [Loop](https://github.com/onnx/onnx/blob/master/docs/Operators.md#loop), [Scan](https://github.com/onnx/onnx/blob/master/docs/Operators.md#scan) or [If](https://github.com/onnx/onnx/blob/master/docs/Operators.md#if) operator).
- `0`: Disable CoreML EP to run on a subgraph in the body of a control flow operator.
- `1`: Enable CoreML EP to run on a subgraph in the body of a control flow operator.

`SpecializationStrategy`:  This feature is available since macOS>=10.15 or iOS>=18.0. This process can affect the model loading time and the prediction latency. Use this option to tailor the specialization strategy for your model. Navigate to [Apple Doc](https://developer.apple.com/documentation/coreml/mloptimizationhints-swift.struct/specializationstrategy-swift.property) for more information. Can be one of the following values: (`Default` by default )
- `Default`:
- `FastPrediction`:

`ProfileComputePlan`:Profile the Core ML MLComputePlan. This logs the hardware each operator is dispatched to and the estimated execution time. Intended for developer usage but provides useful diagnostic information if performance is not as expected. can be one of the following values: (`0` by default )
- `0`: Disable profile.
- `1`: Enable profile.

`AllowLowPrecisionAccumulationOnGPU`: please refer to [Apple Doc](https://developer.apple.com/documentation/coreml/mlmodelconfiguration/allowlowprecisionaccumulationongpu).  can be one of the following values: (`0` by default )
- `0`: Use float32 data type to accumulate data. 
- `1`: Use low precision data(float16) to accumulate data.

`ModelCacheDirectory`: The path to the directory where the Core ML model cache is stored. CoreML EP will compile the captured subgraph to CoreML format graph and saved to disk.
For the given model, if caching is not enabled, CoreML EP will compile and save to disk every time, which may cost significant time (even minutes) for a complicated model. By providing a cache path the CoreML format model can be reused. (Cache disbled by default).
- `""` : Disable cache. (empty string by default)
- `"/path/to/cache"` : Enable cache. (path to cache directory, will be created if not exist)

The cached information for the model is stored under a model hash in the cache directory. There are three ways the hash may be calculated, in order of preference.

1. Read from the model metadata_props. This provides the user a way to directly control the hash, and is the recommended usage. The cache key should satisfy that, (1) The value must only contain alphanumeric characters.  (2) len(value) < 64. EP will re-hash the cache-key to satisfy these conditions.
2. Hash of the model url the inference session was created with.
3. Hash of the graph inputs and node outputs if the inference session was created with in memory bytes (i.e. there was no model path). 

It is critical that if the model changes either the hash value must change, or you must clear out the previous cache information. e.g. if the model url is being used for the hash (option 2 above) the updated model must be loaded from a different path to change the hash value. 

ONNX Runtime does NOT have a mechanism to track model changes and does not delete the cache entries.

Here is an example of how to fill model hash in metadata of model:
```python
import onnx
import hashlib

# You can use any other hash algorithms to ensure the model and its hash-value is a one-one mapping. 
def hash_file(file_path, algorithm='sha256', chunk_size=8192):
    hash_func = hashlib.new(algorithm)
    with open(file_path, 'rb') as file:
        while chunk := file.read(chunk_size):
            hash_func.update(chunk)
    return hash_func.hexdigest()

CACHE_KEY_NAME = "CACHE_KEY"
model_path = "/a/b/c/model.onnx"
m = onnx.load(model_path)

cache_key = m.metadata_props.add()
cache_key.key = CACHE_KEY_NAME
cache_key.value = str(hash_file(model_path))

onnx.save_model(m, model_path)
```


## Configuration Options (Old API)
```
uint32_t coreml_flags = 0;
coreml_flags |= COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE;
```

### Available Options (Deprecated API)

##### COREML_FLAG_USE_CPU_ONLY

Limit CoreML to running on CPU only.

This decreases performance but provides reference output value without precision loss, which is useful for validation.
<br>
Intended for developer usage only.

#####  COREML_FLAG_ENABLE_ON_SUBGRAPH

Enable CoreML EP to run on a subgraph in the body of a control flow operator (i.e. a [Loop](https://github.com/onnx/onnx/blob/master/docs/Operators.md#loop), [Scan](https://github.com/onnx/onnx/blob/master/docs/Operators.md#scan) or [If](https://github.com/onnx/onnx/blob/master/docs/Operators.md#if) operator).

##### COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE

By default the CoreML EP will be enabled for all compatible Apple devices.

Setting this option will only enable CoreML EP for Apple devices with a compatible Apple Neural Engine (ANE).
Note, enabling this option does not guarantee the entire model to be executed using ANE only.

For more information, see [Which devices have an ANE?](https://github.com/hollance/neural-engine/blob/master/docs/supported-devices.md)

##### COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES

Only allow the CoreML EP to take nodes with inputs that have static shapes.
By default the CoreML EP will also allow inputs with dynamic shapes, however performance may be negatively impacted by inputs with dynamic shapes.

##### COREML_FLAG_CREATE_MLPROGRAM

Create an MLProgram format model. Requires Core ML 5 or later (iOS 15+ or macOS 12+).
The default is for a NeuralNetwork model to be created as that requires Core ML 3 or later (iOS 13+ or macOS 10.15+).

## Supported operators

### NeuralNetwork

Operators that are supported by the CoreML Execution Provider when a NeuralNetwork model (the default) is created:

|Operator|Note|
|--------|------|
|ai.onnx:Add||
|ai.onnx:ArgMax||
|ai.onnx:AveragePool|Only 2D Pool is supported.|
|ai.onnx:BatchNormalization||
|ai.onnx:Cast||
|ai.onnx:Clip||
|ai.onnx:Concat||
|ai.onnx:Conv|Only 1D/2D Conv is supported.<br/>Weights and bias should be constant.|
|ai.onnx:DepthToSpace|Only DCR mode DepthToSpace is supported.|
|ai.onnx:Div||
|ai.onnx:Flatten||
|ai.onnx:Gather|Input `indices` with scalar value is not supported.|
|ai.onnx:Gemm|Input B should be constant.|
|ai.onnx:GlobalAveragePool|Only 2D Pool is supported.|
|ai.onnx:GlobalMaxPool|Only 2D Pool is supported.|
|ai.onnx:LeakyRelu||
|ai.onnx:LRN||
|ai.onnx:MatMul|Input B should be constant.|
|ai.onnx:MaxPool|Only 2D Pool is supported.|
|ai.onnx:Mul||
|ai.onnx:Pad|Only constant mode and last two dim padding is supported.<br/>Input pads and constant_value should be constant.<br/>If provided, axes should be constant.|
|ai.onnx:Pow|Only supports cases when both inputs are fp32.|
|ai.onnx:PRelu|Input slope should be constant.<br/>Input slope should either have shape [C, 1, 1] or have 1 element.|
|ai.onnx:Reciprocal||
|ai.onnx.ReduceSum||
|ai.onnx:Relu||
|ai.onnx:Reshape||
|ai.onnx:Resize|4D input.<br/>`coordinate_transformation_mode` == `asymmetric`.<br/>`mode` == `linear` or `nearest`.<br/>`nearest_mode` == `floor`.<br/>`exclude_outside` == false<br/>`scales` or `sizes` must be constant.|
|ai.onnx:Shape|Attribute `start` with non-default value is not supported.<br/>Attribute `end` is not supported.|
|ai.onnx:Sigmoid||
|ai.onnx:Slice|Inputs `starts`, `ends`, `axes`, and `steps` should be constant. Empty slice is not supported.|
|ai.onnx:Softmax||
|ai.onnx:Split|If provided, `splits` must be constant.|
|ai.onnx:Squeeze||
|ai.onnx:Sqrt||
|ai.onnx:Sub||
|ai.onnx:Tanh||
|ai.onnx:Transpose||

### MLProgram

Operators that are supported by the CoreML Execution Provider when a MLProgram model (COREML_FLAG_CREATE_MLPROGRAM flag is set) is created:

|Operator|Note|
|--------|------|
|ai.onnx:Add||
|ai.onnx:Argmax||
|ai.onnx:AveragePool|Only 2D Pool is supported currently. 3D and 5D support can be added if needed.|
|ai.onnx:Cast||
|ai.onnx:Clip||
|ai.onnx:Concat||
|ai.onnx:Conv|Only 1D/2D Conv is supported.<br/>Bias if provided must be constant.|
|ai.onnx:ConvTranspose|Weight and bias must be constant.<br/>padding_type of SAME_UPPER/SAME_LOWER is not supported.<br/>kernel_shape must have default values.<br/>output_shape is not supported.<br/>output_padding must have default values.|
|ai.onnx:DepthToSpace|If 'mode' is 'CRD' the input must have a fixed shape.|
|ai.onnx:Div||
|ai.onnx:Erf||
|ai.onnx:Gemm|Input B must be constant.|
|ai.onnx:Gelu||
|ai.onnx:GlobalAveragePool|Only 2D Pool is supported currently. 3D and 5D support can be added if needed.|
|ai.onnx:GlobalMaxPool|Only 2D Pool is supported currently. 3D and 5D support can be added if needed.|
|ai.onnx:GridSample|4D input.<br/>'mode' of 'linear' or 'zeros'.<br/>(mode==linear && padding_mode==reflection && align_corners==0) is not supported.|
|ai.onnx:GroupNormalization||
|ai.onnx:InstanceNormalization||
|ai.onnx:LayerNormalization||
|ai.onnx:LeakyRelu||
|ai.onnx:MatMul|Only support for transA == 0, alpha == 1.0 and beta == 1.0 is currently implemented.|
|ai.onnx:MaxPool|Only 2D Pool is supported currently. 3D and 5D support can be added if needed.|
|ai.onnx:Max||
|ai.onnx:Mul||
|ai.onnx:Pow|Only supports cases when both inputs are fp32.|
|ai.onnx:PRelu||
|ai.onnx:Reciprocal|this ask for a `epislon` (default 1e-4) where onnx don't provide|
|ai.onnx:ReduceSum||
|ai.onnx:ReduceMean||
|ai.onnx:ReduceMax||
|ai.onnx:Relu||
|ai.onnx:Reshape||
|ai.onnx:Resize|See [resize_op_builder.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/coreml/builders/impl/resize_op_builder.cc) implementation. There are too many permutations to describe the valid combinations.|
|ai.onnx:Round||
|ai.onnx:Shape||
|ai.onnx:Slice|starts/ends/axes/steps must be constant initializers.|
|ai.onnx:Split|If provided, `splits` must be constant.|
|ai.onnx:Sub||
|ai.onnx:Sigmoid||
|ai.onnx:Softmax||
|ai.onnx:Sqrt||
|ai.onnx:Squeeze||
|ai.onnx:Tanh||
|ai.onnx:Transpose||
|ai.onnx:Unsqueeze||
