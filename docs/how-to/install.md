---
title: Install ORT
parent: How to
nav_order: 1
---

# Install ONNX Runtime
{: .no_toc }

See the [installation matrix](https://onnxruntime.ai) for recommended instructions for desired combinations of target operating system, hardware, accelerator, and language. 

Details on OS versions, compilers, language versions, dependent libraries , etc can be found under [Compatibility](../resources/compatibility.md#Environment-compatibility).

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Inference

The following build variants are available as officially supported packages. Others can be [built from source](../how-to/build.md) from each release branch.

### Requirements
{: .no_toc }

* All builds require the English language package with `en_US.UTF-8` locale. On Linux, install [language-pack-en package](https://packages.ubuntu.com/search?keywords=language-pack-en)
by running `locale-gen en_US.UTF-8` and `update-locale LANG=en_US.UTF-8`

* Windows builds require [Visual C++ 2019 runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads).

* Please note additional requirements in the table below. 

||Official build|Nightly build|Add'l Requirements|
|---|---|---|---|
|Python|If using pip, run `pip install --upgrade pip` prior to downloading.|||
||CPU: [**onnxruntime**](https://pypi.org/project/onnxruntime)| [ort-nightly (dev)](https://test.pypi.org/project/ort-nightly)||
||GPU - CUDA: [**onnxruntime-gpu**](https://pypi.org/project/onnxruntime-gpu) | [ort-nightly-gpu (dev)](https://test.pypi.org/project/ort-nightly-gpu)|[View](../reference/execution-providers/CUDA-ExecutionProvider.md#requirements)|
||OpenVINO: [**intel/onnxruntime**](https://github.com/intel/onnxruntime/releases/latest) - *Intel managed*||[View](../how-to/build/eps.md#openvino)|
||TensorRT (Jetson): [**Jetson Zoo**](https://elinux.org/Jetson_Zoo#ONNX_Runtime) - *NVIDIA managed*||[View](../how-to/build/eps.md#tensorrt)|
|C#/C/C++|CPU: [**Microsoft.ML.OnnxRuntime**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) |[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)|[View](../reference/execution-providers/CUDA-ExecutionProvider.md#requirements)|
||GPU - CUDA: [**Microsoft.ML.OnnxRuntime.Gpu**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu)|[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)|[View](../reference/execution-providers/CUDA-ExecutionProvider.md#requirements)|
||GPU - DirectML: [**Microsoft.ML.OnnxRuntime.DirectML**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML)|[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)|[View](../reference/execution-providers/DirectML-ExecutionProvider.md#requirements)|
|WinML|[**Microsoft.AI.MachineLearning**](https://www.nuget.org/packages/Microsoft.AI.MachineLearning)|||
|Java|CPU: [**com.microsoft.onnxruntime:onnxruntime**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime)||[View](../reference/api/java-api.md)|
||GPU - CUDA: [**com.microsoft.onnxruntime:onnxruntime_gpu**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime_gpu)||[View](../reference/api/java-api.md)|
|Android|[**com.microsoft.onnxruntime:onnxruntime-mobile**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime-mobile) ||[View](../how-to/mobile/initial-setup.md#android)|
|iOS (C/C++)|CocoaPods: **onnxruntime-mobile-c**||[View](../how-to/mobile/initial-setup.md#ios)|
|Objective-C|CocoaPods: **onnxruntime-mobile-objc**||[View](../how-to/mobile/initial-setup.md#ios)|
|React Native|[**onnxruntime-react-native**](https://www.npmjs.com/package/onnxruntime-react-native)||[View](../reference/api/js-api.md)|
|Node.js|[**onnxruntime-node**](https://www.npmjs.com/package/onnxruntime-node)||[View](../reference/api/js-api.md)|
|Web|[**onnxruntime-web**](https://www.npmjs.com/package/onnxruntime-web)||[View](../reference/api/js-api.md)|



*Note: Dev builds created from the master branch are available for testing newer changes between official releases. Please use these at your own risk. We strongly advise against deploying these to production workloads as support is limited for dev builds.*




## Training
 
||Official build|Nightly build|
|---|---|---|
|PyTorch 1.8.1 (CUDA 10.2)|[**onnxruntime_stable_torch181.cu102**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch181.cu102.html)|[onnxruntime_nightly_torch181.cu102](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch181.cu102.html)|
|PyTorch 1.8.1 (CUDA 11.1)|[**onnxruntime_stable_torch181.cu111**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch181.cu111.html )|[onnxruntime_nightly_torch181.cu111](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch181.cu111.html)|
|PyTorch 1.9 (CUDA 10.2)|[**onnxruntime-training**](https://pypi.org/project/onnxruntime-training/)|[onnxruntime_nightly_torch190.cu102](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch190.cu102.html)|
|PyTorch 1.9 (CUDA 11.1)|[**onnxruntime_stable_torch190.cu111**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch190.cu111.html)|[onnxruntime_nightly_torch190.cu111](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch190.cu111.html)|
|[*Preview*] PyTorch 1.8.1 (ROCm 4.2)|[**onnxruntime_stable_torch181.rocm42**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch181.rocm42.html)|[onnxruntime_nightly_torch181.rocm42](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch181.rocm42.html)|
|[*Preview*] PyTorch 1.9 (ROCm 4.2)|[**onnxruntime_stable_torch190.rocm42**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch190.rocm42.html)|[onnxruntime_nightly_torch190.rocm42](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch190.rocm42.html)|
