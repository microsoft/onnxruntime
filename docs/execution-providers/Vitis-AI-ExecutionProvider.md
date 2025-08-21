---
title: AMD - Vitis AI
description: Instructions to execute ONNX Runtime on AMD devices with the Vitis AI execution provider
parent: Execution Providers
nav_order: 13
redirect_from: /docs/execution-providers/community-maintained/Vitis-AI-ExecutionProvider
---

# Vitis AI Execution Provider
{: .no_toc }

[Vitis AI](https://github.com/Xilinx/Vitis-AI) is AMD's development stack for hardware-accelerated AI inference on AMD platforms, including Ryzen AI, AMD Adaptable SoCs and Alveo Data Center Acceleration Cards. 


## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements

The following table lists AMD targets that are supported by the Vitis AI ONNX Runtime Execution Provider.

| **Architecture**                                  | **Family**                                                 | **Supported Targets**                                      | **Supported OS**                                           |
|---------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| AMD64                                             | Ryzen AI                                                   | AMD Ryzen processors with NPUs                    | Windows                                                    |
| Arm® Cortex®-A53                                  | Zynq UltraScale+ MPSoC                                     | ZCU102, ZCU104, KV260                                      | Linux                                                      |
| Arm® Cortex®-A72                                  | Versal AI Core / Premium                                   | VCK190                                                     | Linux                                                      |
| Arm® Cortex®-A72                                  | Versal AI Edge                                             | VEK280                                                     | Linux                                                      |

For a complete list of AMD Ryzen processors with NPUs, refer to the [processor specifications](https://www.amd.com/en/products/specifications/processors.html) page (look for the “AMD Ryzen AI” column towards the right side of the table, and select “Available” from the pull-down menu).

AMD Adaptable SoC developers can also leverage the Vitis AI ONNX Runtime Execution Provider to support custom (chip-down) designs.

## Installation

### Installation for AMD Ryzen AI processors

To enable the Vitis AI ONNX Runtime Execution Provider in Microsoft Windows targeting the AMD Ryzen AI processors, developers must install the Ryzen AI Software. Detailed instructions on how to download and install the Ryzen AI Software can be found here: https://ryzenai.docs.amd.com/en/latest/inst.html

For complete examples targeting AMD Ryzen AI processors, developers should refer to the [RyzenAI-SW Github repository](https://github.com/amd/RyzenAI-SW/tree/main).

### Installation for AMD Adaptable SoCs

For AMD Adaptable SoC targets, a pre-built package is provided to deploy ONNX models on embedded Linux.  Users should refer to the standard Vitis AI [Target Setup Instructions](https://xilinx.github.io/Vitis-AI/3.5/html/docs/workflow.html) to enable Vitis AI on the target.  Once Vitis AI has been enabled on the target, the developer can refer to [this section](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Programming-with-VOE) of the Vitis AI documentation for installation and API details.

For complete examples targeting AMD Adaptable SoCs, developers should refer to the [ONNX Runtime Vitis AI Execution Provider examples](https://github.com/Xilinx/Vitis-AI/tree/master/examples/vai_library/samples_onnx).

## Build
To build the Ryzen AI Vitis AI ONNX Runtime Execution Provider from source, please refer to the [Build Instructions](../build/eps.md#amd-vitis-ai).  


## Quantization

Quantization is the process of mapping high-precision weights/activations to a lower precision format while maintaining model accuracy. This technique enhances the computational and memory efficiency of the model for deployment on NPU devices. It can be applied post-training, allowing existing models to be optimized without the need for retraining.

The Vitis AI EP supports input models quantized to either INT8 or BF16 format.
 
Quantization of Ryzen AI models can be accomplished with either the AMD Quark quantizer, the Vitis AI Quantizer, or Olive.

### AMD Quark

AMD Quark is a comprehensive cross-platform deep learning toolkit designed to simplify and enhance the quantization of deep learning models. Supporting both PyTorch and ONNX models, Quark empowers developers to optimize their models for deployment on a wide range of hardware backends, achieving significant performance gains without compromising accuracy.

Refer to the [AMD Quark documentation for Ryzen AI](https://quark.docs.amd.com/latest/supported_accelerators/ryzenai/index.html) for complete details. 

### Vitis AI Quantizer

The Vitis AI Quantizer supports quantization of PyTorch, TensorFlow and ONNX models. 

[Pytorch](https://hub.docker.com/r/amdih/ryzen-ai-pytorch), [Tensorflow 2.x](https://hub.docker.com/r/amdih/ryzen-ai-tensorflow2) and [Tensorflow 1.x](https://hub.docker.com/r/amdih/ryzen-ai-tensorflow) dockers are available to support quantization of PyTorch and TensorFlow models.  To support the Vitis AI ONNX Runtime Execution Provider, an option is provided in the Vitis AI Quantizer to export a quantized model in ONNX format, post quantization.

Refer to the [Vitis AI documentation about quantizing models](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Quantizing-the-Model?tocId=ZTjzDvS7TsL_16iLW7O6ig) for complete details.

### Olive

Experimental support for Microsoft Olive is also enabled.  The Vitis AI Quantizer has been integrated as a plugin into Olive and will be upstreamed.  Once this is complete, users can refer to the example(s) provided in the [Olive Vitis AI Example Directory](https://github.com/microsoft/Olive/tree/main/examples/resnet).

## Runtime Options

The Vitis AI ONNX Runtime integrates a compiler that compiles the model graph and weights as a micro-coded executable.  This executable is deployed on the target accelerator (Ryzen AI NPU or Vitis AI DPU).

The model is compiled when the ONNX Runtime session is started, and compilation must complete prior to the first inference pass.  The length of time required for compilation varies, but may take a few minutes to complete.  Once the model has been compiled, the model executable is cached and for subsequent inference runs, the cached executable model can optionally be used (details below).

The tables below provide an overview of the provider options and environment variables which can be used to configure the inference session. 

For detailed instructions on how to configure the inference session for BF16 and INT8 model on AMD Ryzen AI processors, refer to the [Ryzen AI Software documentation](https://ryzenai.docs.amd.com/en/latest/modelrun.html#)

| **Provider Option**        | **Default Value**              | **Details**                              |
|----------------------------|--------------------------------|------------------------------------------|
| cache_dir                  | Linux: "/tmp/{user}/vaip/.cache/" <br/>   Windows: "C:\\temp\\{user}\\vaip\\.cache"        | optional, cache directory                |
| cache_key                  | {onnx_model_md5}               | optional, cache key, used to distinguish between different models.                      |
| log_level                  | "error"                        | Valid values are `info`, `warning`, `error` and `fatal` |

The final cache directory is `{cache_dir}/{cache_key}`.
Please refer to the following C++ example for usage.

## Ryzen AI API Examples

To leverage the C++ APIs, use the following example as a reference:

```c++
// ...
#include <onnxruntime_cxx_api.h>
// include user header files
// ...

std::basic_string<ORTCHAR_T> model_file = "resnet50.onnx" // Replace resnet50.onnx with your model name
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "resnet50_pt");
auto session_options = Ort::SessionOptions();

auto options = std::unorderd_map<std::string,std::string>({});
// optional, eg: cache path : /tmp/my_cache/abcdefg // Replace abcdefg with your model name, eg. onnx_model_md5
options["cache_dir"] = "/tmp/my_cache";
options["cache_key"] = "abcdefg"; // Replace abcdefg with your model name, eg. onnx_model_md5
options["log_level"] = "info";

// Create an inference session using the Vitis AI execution provider
session_options.AppendExecutionProvider_VitisAI(options);

auto session = Ort::Session(env, model_file.c_str(), session_options);

// get inputs and outputs
Ort::AllocatorWithDefaultOptions allocator;
std::vector<std::string> input_names;
std::vector<std::int64_t> input_shapes;
auto input_count = session.GetInputCount();
for (std::size_t i = 0; i < input_count; i++) {
    input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
    input_shapes = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
}
std::vector<std::string> output_names;
auto output_count = session.GetOutputCount();
for (std::size_t i = 0; i < output_count; i++) {
   output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
}
// Create input tensors and populate input data
std::vector<Ort::Value> input_tensors;
...

auto output_tensors = session.Run(Ort::RunOptions(), input_names.data(), input_tensors.data(),
                    input_count, output_names.data(), output_count);
// postprocess output data
// ...

```

To leverage the Python APIs, use the following example as a reference:

```python
import onnxruntime

# Add user imports
# ...

# Load inputs and do preprocessing
# ...

# Create an inference session using the Vitis AI execution provider
session = onnxruntime.InferenceSession(
    '[model_file].onnx',
    providers=["VitisAIExecutionProvider"],
    provider_options=[{"log_level": "info"}])

input_shape = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name

# Load inputs and do preprocessing by input_shape
input_data = [...]
result = session.run([], {input_name: input_data})

```

For complete examples targeting AMD Ryzen AI processors, developers should refer to the [RyzenAI-SW Github repository](https://github.com/amd/RyzenAI-SW/tree/main).

For complete examples targeting AMD Adaptable SoCs, developers should refer to the [ONNX Runtime Vitis AI Execution Provider examples](https://github.com/Xilinx/Vitis-AI/tree/master/examples/vai_library/samples_onnx).
