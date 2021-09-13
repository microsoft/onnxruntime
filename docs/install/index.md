---
title: Install ORT
has_children: false
nav_order: 1
---

# Install ONNX Runtime
{: .no_toc }

See the [installation matrix](https://onnxruntime.ai) for recommended instructions for desired combinations of target operating system, hardware, accelerator, and language. 

Details on OS versions, compilers, language versions, dependent libraries, etc can be found under [Compatibility](../reference/compatibility).


## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Python Installs

### Install ONNX Runtime (ORT)
{: .no_toc }

```bash
pip install onnxruntime
```
```bash
pip install onnxruntime-gpu
```

### Install ONNX to export the model
{: .no_toc }

```bash
## pytorch
pip install onnx-pytorch
```
```python
## tensorflow
pip install onnx-tf
pip install tf2onnx
```
```
```bash
## sklearn
pip install skl2onnx
```

## C#/C/C++/WinML Installs

### Install ONNX Runtime (ORT)
{: .no_toc }

```bash
# CPU 
dotnet add package Microsoft.ML.OnnxRuntime --version 1.8.1
```
```bash
# GPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu --version 1.8.1
```
```bash
# DirectML
dotnet add package Microsoft.ML.OnnxRuntime.DirectML --version 1.8.1
```

```bash
# WinML
dotnet add package Microsoft.AI.MachineLearning --version 1.8.1
```
## JavaScript Installs

### Web ORT (client)
{: .no_toc }

```bash
npm install onnxruntime-web
```

### Node ORT (server)
{: .no_toc }

```bash
npm install onnxruntime-node
```

### React Native ORT
{: .no_toc }

```bash
npm install onnxruntime-react-native
```

## ORT Training package

```
pip install torch-ort
python -m torch_ort.configure
```

**Note**: This installs the default version of the `torch-ort` and `onnxruntime-training` packages that are mapped to specific versions of the CUDA libraries. Refer to the install options in [ONNXRUNTIME.ai](https://onnxruntime.ai).

### Add ORTModule in the `train.py`

```python
   from torch_ort import ORTModule
   .
   .
   .
   model = ORTModule(model)
```

**Note**: the `model` where ORTModule is wrapped needs to be a derived from the `torch.nn.Module` class.

## Inference install table for all languages

The table below lists the build variants available as officially supported packages. Others can be [built from source](../build/inferencing) from each release branch.

### Requirements
{: .no_toc }

* All builds require the English language package with `en_US.UTF-8` locale. On Linux, install [language-pack-en package](https://packages.ubuntu.com/search?keywords=language-pack-en)
by running `locale-gen en_US.UTF-8` and `update-locale LANG=en_US.UTF-8`

* Windows builds require [Visual C++ 2019 runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads).

* Please note additional requirements and dependencies in the table below:

||Official build|Nightly build|Reqs|
|---|---|---|---|
|Python|If using pip, run `pip install --upgrade pip` prior to downloading.|||
||CPU: [**onnxruntime**](https://pypi.org/project/onnxruntime)| [ort-nightly (dev)](https://test.pypi.org/project/ort-nightly)||
||GPU - CUDA: [**onnxruntime-gpu**](https://pypi.org/project/onnxruntime-gpu) | [ort-nightly-gpu (dev)](https://test.pypi.org/project/ort-nightly-gpu)|[View](../execution-providers/CUDA-ExecutionProvider.md#requirements)|
||OpenVINO: [**intel/onnxruntime**](https://github.com/intel/onnxruntime/releases/latest) - *Intel managed*||[View](build/eps.md#openvino)|
||TensorRT (Jetson): [**Jetson Zoo**](https://elinux.org/Jetson_Zoo#ONNX_Runtime) - *NVIDIA managed*|||
|C#/C/C++|CPU: [**Microsoft.ML.OnnxRuntime**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) |[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)||
||GPU - CUDA: [**Microsoft.ML.OnnxRuntime.Gpu**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu)|[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)|[View](../execution-providers/CUDA-ExecutionProvider)|
||GPU - DirectML: [**Microsoft.ML.OnnxRuntime.DirectML**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML)|[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)|[View](../execution-providers/DirectML-ExecutionProvider)|
|WinML|[**Microsoft.AI.MachineLearning**](https://www.nuget.org/packages/Microsoft.AI.MachineLearning)||[View](https://docs.microsoft.com/en-us/windows/ai/windows-ml/port-app-to-nuget#prerequisites)|
|Java|CPU: [**com.microsoft.onnxruntime:onnxruntime**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime)||[View](../api/java-api.md)|
||GPU - CUDA: [**com.microsoft.onnxruntime:onnxruntime_gpu**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime_gpu)||[View](../api/java-api.md)|
|Android|[**com.microsoft.onnxruntime:onnxruntime-mobile**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime-mobile) ||[View](tutorials/mobile/mobile/initial-setup)|
|iOS (C/C++)|CocoaPods: **onnxruntime-mobile-c**||[View](tutorials/mobile/mobile/initial-setup)|
|Objective-C|CocoaPods: **onnxruntime-mobile-objc**||[View](tutorials/mobile/mobile/initial-setup)|
|React Native|[**onnxruntime-react-native**](https://www.npmjs.com/package/onnxruntime-react-native)||[View](../api/js-api.md)|
|Node.js|[**onnxruntime-node**](https://www.npmjs.com/package/onnxruntime-node)||[View](../api/js-api.md)|
|Web|[**onnxruntime-web**](https://www.npmjs.com/package/onnxruntime-web)||[View](../api/js-api.md)|



*Note: Dev builds created from the master branch are available for testing newer changes between official releases. Please use these at your own risk. We strongly advise against deploying these to production workloads as support is limited for dev builds.*




## Training install table for all languages

ONNX Runtime Training packages are available for different versions of PyTorch, CUDA and ROCm versions. 

The install command is:
```cmd
pip3 install torch-ort [-f location]
python 3 -m torch_ort.configure
```

The _location_ needs to be specified for any specific version other than the default combination. The location for the different configurations are below:

||Official build (location)|Nightly build (location)|
|---|---|---|
|PyTorch 1.8.1 (CUDA 10.2)|[**onnxruntime_stable_torch181.cu102**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch181.cu102.html)|[onnxruntime_nightly_torch181.cu102](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch181.cu102.html)|
|PyTorch 1.8.1 (CUDA 11.1)|[**onnxruntime_stable_torch181.cu111**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch181.cu111.html )|[onnxruntime_nightly_torch181.cu111](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch181.cu111.html)|
|PyTorch 1.9 (CUDA 10.2) **Default**|[**onnxruntime-training**](https://pypi.org/project/onnxruntime-training/)|[onnxruntime_nightly_torch190.cu102](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch190.cu102.html)|
|PyTorch 1.9 (CUDA 11.1)|[**onnxruntime_stable_torch190.cu111**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch190.cu111.html)|[onnxruntime_nightly_torch190.cu111](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch190.cu111.html)|
|[*Preview*] PyTorch 1.8.1 (ROCm 4.2)|[**onnxruntime_stable_torch181.rocm42**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch181.rocm42.html)|[onnxruntime_nightly_torch181.rocm42](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch181.rocm42.html)|
|[*Preview*] PyTorch 1.9 (ROCm 4.2)|[**onnxruntime_stable_torch190.rocm42**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch190.rocm42.html)|[onnxruntime_nightly_torch190.rocm42](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch190.rocm42.html)|