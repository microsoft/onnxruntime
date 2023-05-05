---
title: AMD - Vitis AI
description: Instructions to execute ONNX Runtime on AMD devices with the Vitis AI execution provider
grand_parent: Execution Providers
parent: Community-maintained
nav_order: 6
redirect_from: /docs/reference/execution-providers/Vitis-AI-ExecutionProvider
---

# Vitis-AI Execution Provider
{: .no_toc }

[Vitis-AI](https://github.com/Xilinx/Vitis-AI) is AMD's development stack for hardware-accelerated AI inference on AMD platforms, including both edge devices and Alveo cards. It consists of optimized IP, tools, libraries, models, and example designs. It is designed with high efficiency and ease of use in mind, unleashing the full potential of AI acceleration on AMD FPGA and ACAP.

The current Vitis-AI execution provider inside ONNXRuntime enables acceleration of Neural Network model inference using embedded devices such as Zynq® UltraScale+™ MPSoC, Versal, Versal AI Edge and Kria cards.


## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements
Note: Vitis-AI Execution Provider is enabled from Vitis AI 3.5.

The following table lists target Edge boards that are supported with ONNXRuntime Vitis-AI Execution Provider.
| **Family**                                       | **Supported Boards**                                       |
|--------------------------------------------------|------------------------------------------------------------|
| Zynq® UltraScale+™ MPSoC                         | ZCU102 and ZCU104 Boards                                   |
| Versal                                           | VCK190                                                     |
| Versal AI Edge                                   | VEK280                                                     |
| Kria                                             | KV260                                                      |

## Build
See [Build instructions](../../build/eps.md#vitis-ai).

### Hardware setup

On Edge, you only need to download the image and installation package related to the developemnt board, burn the image, and install the required package according to the [Target Setup Instructions](https://xilinx.github.io/Vitis-AI/docs/board_setup/board_setup.html).

On Windows, please download the `voe-[version]-win_amd64.zip` which VOE (Vitis-AI ONNXRuntime Engine) release package from [Programming with VOE](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Programming-with-VOE).

If you are using C++,  first please unzip the `voe-[version]-win_amd64.zip` file and copy `bin\\*.dll`  to `C:\\Program Files\\onnxruntime\\bin` to install the VOE's dll files  and then install VOE's python modules.
```
pip install voe-[version]-cp39-cp39-win_amd64.whl
```
If you are using python, please install VOE python modules.
```
 pip install voe-[version]-cp39-cp39-win_amd64.whl
```

## Usage
### Quantization
To accelerate the inference of Neural Network models with Vitis-AI DPU accelerators, these models need to be quantized beforehand.
Currently users need to use the Vitis-AI quantization tools to quantize Pytorch/TensorFlow model and output quantized ONNX model. See [Vitis AI model quantization](https://xilinx.github.io/Vitis-AI/docs/workflow-model-development.html#model-quantization) for details.
In the future, the Vitis-AI quantization tools plan to support quantization based on ONNX, and the Vitis-AI execution provider for ONNXRuntime will support on-the-fly quantization. With this flow, users won't need to quantize their models beforehand.

## Configuration Options
The model is compiled when the session is created. The compilatioin stage usually takes a long time. Vitis-AI uses cache to store the compiled model.

A config file path can be used to customize the config file and configuration variables can be used to customize the cache path.

| **Configuration Variable**   | **Default if unset**           | **Explanation**                          |
|----------------------------|--------------------------------|------------------------------------------|
| config_file                | ""                             | required,  the configuration file path, the configuration file `vaip_config.json` is contained in the `voe-[version]-win_amd64.zip`.       |
| CacheDir                   | Linux: "/tmp/{user}/vaip/.cache/" <br/>   Windows: "C:\\temp\\{user}\\vaip\\.cache"        | optional, cache directory                |
| CacheKey                   | {onnx_model_md5}               | optional, cache key, used to distinguish between different models.                      |

The final cache directory is `{CacheDir}/{CacheKey}`.
Please refer to the following C++ example for usage.

A couple of environment variables can be used to customize the Vitis-AI Execution provider.

| **Environment Variable**   | **Default if unset** | **Explanation**                                    |
|----------------------------|----------------------|----------------------------------------------------|
| XLNX_ENABLE_CACHE          | 1                    | Whether to use cache, if it is 0, it will ignore the cache files and the model will be recompiled|
| XLNX_TARGET_NAME           | ""                   | DPU target name. On Edge, if not set, the DPU target name will be read automatically; On Windows, default value is "AMD_AIE2_Nx4_Overlay" which is the DPU target name of IPU.                                |


## Samples
If you are using C++, you can use the following example as a reference:
```C++
// ...
#include <experimental_onnxruntime_cxx_api.h>
// include other header files
// ...

auto onnx_model_path = "resnet50_pt.onnx"
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "resnet50_pt");
auto session_options = Ort::SessionOptions();

auto options = std::unorderd_map<std::string,std::string>({});
options["config_file"] = "/etc/vaip_config.json";
// optional, eg: cache path : /tmp/my_cache/abcdefg
options["CacheDir"] = "/tmp/my_cache";
options["CacheKey"] = "abcdefg";

session_options.AppendExecutionProvider("VitisAI", options);

auto session = Ort::Experimental::Session(env, model_name, session_options);

auto input_shapes = session.GetInputShapes();
// preprocess input data
// ...

// create input tensors and fillin input data
std::vector<Ort::Value> input_tensors;
input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(
      input_data.data(), input_data.size(), input_shapes[0]));

auto output_tensors = session.Run(session.GetInputNames(), input_tensors,
                                      session.GetOutputNames());
// postprocess output data
// ...

```

If you are using python, you can use the following example as a reference:
```python
import onnxruntime

# Add other imports
# ...

# Load inputs and do preprocessing
# ...

# Create an inference session using the Vitis-AI execution provider

session = onnxruntime.InferenceSession(
    '[model_file].onnx',
    providers=["VitisAIExecutionProvider"],
    provider_options=[{"config_file":"/etc/vaip_config.json"}])

input_shape = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name

# Load inputs and do preprocessing by input_shape
input_data = [...]
result = session.run([], {input_name: input_data})

```
For more complete examples, please refer to  [ONNXRuntime Vitis-AI Execution Provider examples](https://github.com/Xilinx/Vitis-AI/tree/master/examples/vai_library/samples_onnx).
