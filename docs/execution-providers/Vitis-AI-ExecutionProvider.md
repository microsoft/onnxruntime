---
title: AMD - Vitis AI
description: Instructions to execute ONNX Runtime on AMD devices with the Vitis AI execution provider
parent: Execution Providers
nav_order: 12
redirect_from: /docs/execution-providers/community-maintained/Vitis-AI-ExecutionProvider
---

# Vitis AI Execution Provider
{: .no_toc }

[Vitis AI](https://github.com/Xilinx/Vitis-AI) is AMD's development stack for hardware-accelerated AI inference on AMD platforms, including Ryzen AI, AMD Adaptable SoCs and Alveo Data Center Acceleration Cards. It consists of optimized IP, tools, libraries, models, and example designs. 

This release of the Vitis AI Execution Provider enables acceleration of Neural Network model inference targeting AMD Ryzen AI and Adaptable SoCs and Ryzen AI.


## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements

The following table lists AMD targets that are supported by the Vitis AI ONNX Runtime Execution Provider.

| **Architecture**   							    | **Family**                                                 | **Supported Targets**                                      | **Supported OS**                                           |
|---------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| AMD64							                    | Ryzen AI                                                   | AMD Ryzen 7040U, 7040HS                                    | Windows                                                    |
| ARM64 Cortex-A53   				                | Zynq UltraScale+ MPSoC                                     | ZCU102, ZCU104, KV260                                      | Linux                                                      |
| ARM64 Cortex-A72				                    | Versal AI Core / Premium                                   | VCK190                                                     | Linux                                                      |
| ARM64	Cortex-A72						            | Versal AI Edge                                             | VEK280                                                     | Linux                                                      |


AMD Adaptable SoC developers can also leverage the Vitis AI ONNX Runtime Execution Provider to support custom (chip-down) designs.

## Ryzen AI Demo
A [demonstration](https://github.com/amd/RyzenAI-cloud-to-client-demo) is available that showcases one potential application of AMD's XDNA technology.  New users are encouraged to test it out.

## Install

### AMD Adaptable SoC Installation
For AMD Adaptable SoC targets, a pre-built package is provided to deploy ONNX models on embedded Linux.  Users should refer to the standard Vitis AI [Target Setup Instructions](https://xilinx.github.io/Vitis-AI/3.0/html/docs/workflow.html) to enable Vitis AI on the target.  Once Vitis AI has been enabled on the target, the developer can refer to [this section](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Programming-with-VOE) of the Vitis AI documentation for installation and API details.

For more complete examples, developers should refer to [ONNX Runtime Vitis AI Execution Provider examples](https://github.com/Xilinx/Vitis-AI/tree/master/examples/vai_library/samples_onnx).

### Ryzen AI Installation

To enable the Vitis AI ONNX Runtime Execution Provider in Microsoft Windows, a .zip archive is provided.

The contents of this archive are as follows:

```
voe-3.5-win_amd64.zip
	│
	├── Examples  
	│    │    
	│    └── resnet50_python                               # ONNX ResNet50 ILSVRC2012
	├── voe-0.1.0-cp39-cp39-win_amd64                      # Folder containing DLLs for C++ API
	│    │    
	│    ├── onnxruntime.dll                                    
	│    ├── onnxruntime_vitisai_ep.dll                            
	├── vaip_config.json                       	       # Runtime configuration file
        ├── installer.py                                       # Python script to copy dlls 
        ├── voe-0.1.0-cp39-cp39-win_amd64.whl      	       # Python installation package
        ├── onnxruntime_vitisai-1.15.1-cp39-cp39-win_amd64.whl # Python installation package
        ├── 1x4.xclbin                                         # IPU Execution Provider executable file
        ├── 5x4.xclbin                                         # IPU Execution Provider executable file
	│    
```

**_NOTE:_** Ryzen AI Linux support is not enabled in this release.

Both C++ and Python APIs are supported.  The following instructions assume that you have already installed ONNX Runtime on your Windows RyzenAI target.  The steps to install the Vitis AI ONNX Runtime Engine are as follows:

**1. Verify Pre-requisites:**

- Visual Studio = 2022
- cmake (version >= 3.26)
- python (version >= 3.9) (Python 3.9.13 64bit recommended)
- AMD IPU driver >= 10.105.5.38 installed

**2. Prepare the files for installation:**

- Download the [Ryzen AI ONNX Runtime Package](https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html?filename=voe-3.5-win_amd64.zip).  
- Unzip `voe-[version]-win_amd64.zip`.  

**3. Install the C++ Libraries:**

**_NOTE:_** This is an optional step which is only required if you intend to use the C++ APIs.

- Copy the DLL files from the bin subdirectory of the extracted archive to `C:\Program Files\onnxruntime\bin`, (ie `copy bin\*.dll C:\Program Files\onnxruntime\bin`).  This will install the Vitis AI ONNX Runtime Engine libraries.  

**4. Set the XLNX_VART_FIRMWARE environmental variable:**

**_IMPORTANT:_** If you do not execute this step, the IPU will not be loaded with the required executable file.  By default the search path used will be `C:\Windows\System32\AMD` and this path may already contain an xclbin that was stored during the IPU driver installation.  Do not use this path to store the Execution Provider xclbin file.  Execute the following command, replacing `[path_to_xclbin]` with the target path containing the xclbin:

```
set XLNX_VART_FIRMWARE=C:\[path_to_xclbin]\1x4.xclbin
```

**4. Install the Python Modules:**

- From the Python prompt, execute the following command, replacing `[version]` with the appropriate release version number as determined through the files extracted from the .zip archive.
	
```
pip install voe-[version]-cp39-cp39-win_amd64.whl
```

**4. Run the ResNet50 example:**

- Leverage the scripts in the `Examples\resnet50_python` folder to test your installation.  


## Build
To build the Ryzen AI Vitis AI ONNX Runtime Execution Provider from source, please refer to the [Build Instructions](../build/eps.md#amd-vitis-ai).  


## Quantization

AMD Ryzen AI and Adaptable SoC targets employ INT8 quantization for inference.  The benefits of INT8 quantization include reduced power consumption and memory utilization.

The current release of the Vitis AI Execution Provider ingests quantized ONNX models with INT8 datatypes.  No support is provided for direct deployment of models with alternative datatypes, including FP32.  Quantization of Ryzen AI models can be accomplished with either the Vitis AI Quantizer, or Olive.  For AMD Adapable SoCs, the user must leverage the Vitis AI Quantizer.

### Vitis AI Quantizer
 
[Pytorch](https://hub.docker.com/r/amdih/ryzen-ai-pytorch), [Tensorflow 2.x](https://hub.docker.com/r/amdih/ryzen-ai-tensorflow2) and [Tensorflow 1.x](https://hub.docker.com/r/amdih/ryzen-ai-tensorflow) dockers are avaialable to support quantization of PyTorch and TensorFlow models.  To support the Vitis AI ONNX Runtime Execution Provider, an option is provided in the Vitis AI Quantizer to export a quantized model in ONNX format, post quantization.

[ONNX Quantizer python wheel](https://www.xilinx.com/bin/public/openDownload?filename=vai_q_onnx-1.14.0-py2.py3-none-any.whl) is available to parse and quantize ONNX models, enabling an end-to-end ONNX model -> ONNX Runtime workflow.  Also, in a future release, the Vitis AI ONNX Runtime Execution Provider will support on-the-fly quantization, enabling direct deployment of FP32 ONNX models.

See [Model Quantization](https://xilinx.github.io/Vitis-AI/3.5/html/docs/workflow-model-development.html#model-quantization) for details.

### Olive

Experimental support for Microsoft Olive is enabled in this release.  The Vitis AI Quantizer has been integrated as a plugin into Olive and will be upstreamed.  Once this is complete, users can refer to the example(s) provided in the [Olive Vitis AI Example Directory](https://github.com/microsoft/Olive/tree/main/examples/resnet).

## Runtime Options

The Vitis AI ONNX Runtime integrates a compiler that compiles the model graph and weights as a micro-coded executable.  This executable is deployed on the target accelerator (Ryzen AI IPU or Vitis AI DPU).

The model is compiled when the ONNX Runtime session is started, and compilation must complete prior to the first inference pass.  The length of time required for compilation varies, but may take a few minutes to complete.  Once the model has been compiled, the model executable is cached and for subsequent inference runs, the cached executable model can optionally be used (details below).

Several runtime variables can be set to configure the inference session as listed in the table below.  The `config file` variable is not optional and must be set to point to the location of the configuration file.  The `cacheDir` and `cacheKey` variables are optional.  An example illustrating the usage of all three variables can be found in the C++ [Ryzen AI API Example](#ryzen-ai-api-examples).

| **Runtime Variable**   | **Default Value**           | **Details**                          |
|----------------------------|--------------------------------|------------------------------------------|
| config_file                | ""                             | required,  the configuration file path, the configuration file `vaip_config.json` is contained in the `voe-[version]-win_amd64.zip`.       |
| cacheDir                   | Linux: "/tmp/{user}/vaip/.cache/" <br/>   Windows: "C:\\temp\\{user}\\vaip\\.cache"        | optional, cache directory                |
| cacheKey                   | {onnx_model_md5}               | optional, cache key, used to distinguish between different models.                      |

The final cache directory is `{cacheDir}/{cacheKey}`.
Please refer to the following C++ example for usage.

In addition, environment variables can be set in order to customize the Vitis AI Execution provider.

| **Environment Variable**   | **Default Value** | **Details**                                    |
|----------------------------|----------------------|----------------------------------------------------|
| XLNX_VART_FIRMWARE         | ""                   | Configures the path location for the xclbin executable file that runs on the IPU.  It is essential to configure this variable.|
| XLNX_ENABLE_CACHE          | 1                    | Whether to use cache, if it is 0, it will ignore the cached executable and the model will be recompiled.|
| XLNX_TARGET_NAME           | ""                   | DPU target name. On Adaptable SoCs, if not set, the DPU target name will be read automatically; On Windows, default value is "AMD_AIE2_Nx4_Overlay" which is the DPU target name of the IPU.                                |

## Ryzen AI API Examples
To leverage the C++ APIs, use the following example as a reference:
```c++
// ...
#include <experimental_onnxruntime_cxx_api.h>
// include user header files
// ...

auto onnx_model_path = "resnet50_pt.onnx"
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "resnet50_pt");
auto session_options = Ort::SessionOptions();

auto options = std::unorderd_map<std::string,std::string>({});
options["config_file"] = "/etc/vaip_config.json";
// optional, eg: cache path : /tmp/my_cache/abcdefg // Replace abcdefg with your model name, eg. onnx_model_md5
options["cacheDir"] = "/tmp/my_cache";
options["cacheKey"] = "abcdefg"; // Replace abcdefg with your model name, eg. onnx_model_md5

// Create an inference session using the Vitis AI execution provider
session_options.AppendExecutionProvider("VitisAI", options);

auto session = Ort::Experimental::Session(env, model_name, session_options);

auto input_shapes = session.GetInputShapes();
// preprocess input data
// ...

// Create input tensors and populate input data
std::vector<Ort::Value> input_tensors;
input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(
      input_data.data(), input_data.size(), input_shapes[0]));

auto output_tensors = session.Run(session.GetInputNames(), input_tensors,
                                      session.GetOutputNames());
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
    provider_options=[{"config_file":"/etc/vaip_config.json"}])

input_shape = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name

# Load inputs and do preprocessing by input_shape
input_data = [...]
result = session.run([], {input_name: input_data})

```
