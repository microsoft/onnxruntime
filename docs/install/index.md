---
title: Install ONNX Runtime
description: Instructions to install ONNX Runtime on your target platform in your environment
has_children: false
nav_order: 1
redirect_from: /docs/how-to/install
---

# Install ONNX Runtime (ORT)


See the [installation matrix](https://onnxruntime.ai) for recommended instructions for desired combinations of target operating system, hardware, accelerator, and language.

Details on OS versions, compilers, language versions, dependent libraries, etc can be found under [Compatibility](../reference/compatibility).


## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Python Installs

### Install ONNX Runtime (ORT)

```bash
pip install onnxruntime
```

```bash
pip install onnxruntime-gpu
```

### Install ONNX to export the model

```bash
## ONNX is built into PyTorch
pip install torch
```
```python
## tensorflow
pip install tf2onnx
```

```bash
## sklearn
pip install skl2onnx
```

## C#/C/C++/WinML Installs

### Install ONNX Runtime (ORT)

```bash
# CPU
dotnet add package Microsoft.ML.OnnxRuntime
```
```bash
# GPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu
```
```bash
# DirectML
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```

```bash
# WinML
dotnet add package Microsoft.AI.MachineLearning
```

## Install on web and mobile

Unless stated otherwise, the installation instructions in this section refer to pre-built packages that include support for selected operators and ONNX opset versions based on the requirements of popular models. These packages may be referred to as "mobile packages". If you use mobile packages, your model must only use the supported [opsets and operators](../reference/operators/mobile_package_op_type_support_1.9.md).

Another type of pre-built package has full support for all ONNX opsets and operators, at the cost of larger binary size. These packages are referred to as "full packages".

If the pre-built mobile package supports your model/s but is too large, you can create a [custom build](../build/custom.md). A custom build can include just the opsets and operators in your model/s to reduce the size.

If the pre-built mobile package does not include the opsets or operators in your model/s, you can either use the full package if available, or create a custom build.

### JavaScript Installs

#### Install ONNX Runtime Web (browsers)

```bash
# install latest release version
npm install onnxruntime-web

# install nightly build dev version
npm install onnxruntime-web@dev
```

#### Install ONNX Runtime Node.js binding (Node.js)

```bash
# install latest release version
npm install onnxruntime-node
```

#### Install ONNX Runtime for React Native


```bash
# install latest release version
npm install onnxruntime-react-native
```

### Install on iOS

In your CocoaPods `Podfile`, add the `onnxruntime-c`, `onnxruntime-mobile-c`, `onnxruntime-objc`, or `onnxruntime-mobile-objc` pod, depending on whether you want to use a full or mobile package and which API you want to use.

#### C/C++

  ```pod
  use_frameworks!

  # choose one of the two below:
  pod 'onnxruntime-c'  # full package
  #pod 'onnxruntime-mobile-c'  # mobile package
  ```

#### Objective-C

  ```pod
  use_frameworks!

  # choose one of the two below:
  pod 'onnxruntime-objc'  # full package
  #pod 'onnxruntime-mobile-objc'  # mobile package
  ```

Run `pod install`.

#### Custom build

Refer to the instructions for creating a [custom iOS package](../build/custom.md#ios).

### Install on Android

#### Java/Kotlin

In your Android Studio Project, make the following changes to:

1. build.gradle (Project):

   ```gradle
    repositories {
        mavenCentral()
    }
   ```

2. build.gradle (Module):

    ```gradle
    dependencies {
        // choose one of the two below:
        implementation 'com.microsoft.onnxruntime:onnxruntime-android:latest.release'  // full package
        //implementation 'com.microsoft.onnxruntime:onnxruntime-mobile:latest.release'  // mobile package
    }
    ```

#### C/C++

Download the [onnxruntime-android](https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android) (full package) or [onnxruntime-mobile](https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-mobile) (mobile package) AAR hosted at MavenCentral, change the file extension from `.aar` to `.zip`, and unzip it. Include the header files from the `headers` folder, and the relevant `libonnxruntime.so` dynamic library from the `jni` folder in your NDK project.

#### Custom build

Refer to the instructions for creating a [custom Android package](../build/custom.md#android).

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

* All builds require the English language package with `en_US.UTF-8` locale. On Linux, install [language-pack-en package](https://packages.ubuntu.com/search?keywords=language-pack-en)
by running `locale-gen en_US.UTF-8` and `update-locale LANG=en_US.UTF-8`

* Windows builds require [Visual C++ 2019 runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads).

* Please note additional requirements and dependencies in the table below:

||Official build|Nightly build|Reqs|
|---|---|---|---|
|Python|If using pip, run `pip install --upgrade pip` prior to downloading.|||
||CPU: [**onnxruntime**](https://pypi.org/project/onnxruntime)| [ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly/overview)||
||GPU - CUDA: [**onnxruntime-gpu**](https://pypi.org/project/onnxruntime-gpu) | [ort-nightly-gpu (dev)](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly-gpu/overview/)|[View](../execution-providers/CUDA-ExecutionProvider.md#requirements)|
||OpenVINO: [**intel/onnxruntime**](https://github.com/intel/onnxruntime/releases/latest) - *Intel managed*||[View](../build/eps.md#openvino)|
||TensorRT (Jetson): [**Jetson Zoo**](https://elinux.org/Jetson_Zoo#ONNX_Runtime) - *NVIDIA managed*|||
|C#/C/C++|CPU: [**Microsoft.ML.OnnxRuntime**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) |[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)||
||GPU - CUDA: [**Microsoft.ML.OnnxRuntime.Gpu**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu)|[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)|[View](../execution-providers/CUDA-ExecutionProvider)|
||GPU - DirectML: [**Microsoft.ML.OnnxRuntime.DirectML**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML)|[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly-directml/overview)|[View](../execution-providers/DirectML-ExecutionProvider)|
|WinML|[**Microsoft.AI.MachineLearning**](https://www.nuget.org/packages/Microsoft.AI.MachineLearning)||[View](https://docs.microsoft.com/en-us/windows/ai/windows-ml/port-app-to-nuget#prerequisites)|
|Java|CPU: [**com.microsoft.onnxruntime:onnxruntime**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime)||[View](../api/java)|
||GPU - CUDA: [**com.microsoft.onnxruntime:onnxruntime_gpu**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime_gpu)||[View](../api/java)|
|Android|[**com.microsoft.onnxruntime:onnxruntime-mobile**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime-mobile) ||[View](../install/index.md#install-on-ios)|
|iOS (C/C++)|CocoaPods: **onnxruntime-mobile-c**||[View](../install/index.md#install-on-ios)|
|Objective-C|CocoaPods: **onnxruntime-mobile-objc**||[View](../install/index.md#install-on-ios)|
|React Native|[**onnxruntime-react-native**](https://www.npmjs.com/package/onnxruntime-react-native)||[View](../api/js)|
|Node.js|[**onnxruntime-node**](https://www.npmjs.com/package/onnxruntime-node)||[View](../api/js)|
|Web|[**onnxruntime-web**](https://www.npmjs.com/package/onnxruntime-web)||[View](../api/js)|



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
|[*Preview*] PyTorch 1.11 (ROCm 5.1.1)|[**onnxruntime_stable_torch1110.rocm511**](https://download.onnxruntime.ai/onnxruntime_stable_rocm511.html)|[onnxruntime_nightly_torch1110.rocm511](https://download.onnxruntime.ai/onnxruntime_nightly_rocm511.html)|
|PyTorch 1.11 (ROCm 5.2)||[onnxruntime_nightly_torch1110.rocm52](https://download.onnxruntime.ai/onnxruntime_nightly_rocm511.html)|
|PyTorch 1.12.1 (ROCm 5.2.3)||[onnxruntime_nightly_torch1121.rocm523](https://download.onnxruntime.ai/onnxruntime_nightly_rocm523.html)|
|PyTorch 1.13 (ROCm 5.2.3)||[onnxruntime_nightly_torch1121.rocm523](https://download.onnxruntime.ai/onnxruntime_nightly_rocm523.html)|
