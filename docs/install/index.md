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

## Requirements

* All builds require the English language package with `en_US.UTF-8` locale. On Linux, install [language-pack-en package](https://packages.ubuntu.com/search?keywords=language-pack-en)
by running `locale-gen en_US.UTF-8` and `update-locale LANG=en_US.UTF-8`

* Windows builds require [Visual C++ 2019 runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads). The latest version is recommended.

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

Unless stated otherwise, the installation instructions in this section refer to pre-built packages that include support for selected operators and ONNX opset versions based on the requirements of popular models. These packages may be referred to as "mobile packages". If you use mobile packages, your model must only use the supported [opsets and operators](../reference/operators/mobile_package_op_type_support_1.14.md).

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

  ```ruby
  use_frameworks!

  # choose one of the two below:
  pod 'onnxruntime-c'  # full package
  #pod 'onnxruntime-mobile-c'  # mobile package
  ```

#### Objective-C

  ```ruby
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

## Install for On-Device Training

Unless stated otherwise, the installation instructions in this section refer to pre-built packages designed to perform on-device training.

If the pre-built training package supports your model but is too large, you can create a [custom training build](../build/custom.md).

### Offline Phase - Prepare for Training

```bash
python -m pip install cerberus flatbuffers h5py numpy>=1.16.6 onnx packaging protobuf sympy setuptools>=41.4.0
pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-cpu
```

### Training Phase - On-Device Training

<table>
  <tr>
    <th>Device</th>
    <th>Language</th>
    <th>PackageName</th>
    <th>Installation Instructions</th>
  </tr>
  <tr>
    <td>Windows</td>
    <td>C, C++, C#</td>
    <!-- TODO (baijumeswani) - Update to Training link once published -->
    <td><a href="https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime">Microsoft.ML.OnnxRuntime.Training</a></td>
    <td>
      <pre lang="bash">dotnet add package Microsoft.ML.OnnxRuntime.Training</pre>
    </td>
  </tr>
  <!-- <tr>
    <td></td>
    <td>Python</td>
    <td><a href="https://pypi.org/project/onnxruntime-training/">onnxruntime-training</a></td>
    <td>
      <pre lang="bash">pip install onnxruntime-training</pre>
    </td>
  </tr> -->
  <tr>
    <td>Linux</td>
    <td>C, C++</td>
    <td><a href="https://github.com/microsoft/onnxruntime/releases">onnxruntime-training-linux*.tgz</a></td>
    <td>
      <ul>
        <li>Download the <code>*.tgz</code> file from <a href="https://github.com/microsoft/onnxruntime/releases">here</a>.</li>
        <li>Extract it.</li>
        <li>Move and include the header files in the <code>include</code> directory.</li>
        <li>Move the <code>libonnxruntime.so</code> dynamic library to a desired path and include it.</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td></td>
    <td>Python</td>
    <td><a href="https://pypi.org/project/onnxruntime-training/">onnxruntime-training</a></td>
    <td>
      <pre lang="bash">pip install onnxruntime-training</pre>
    </td>
  </tr>
  <tr>
    <td>Android</td>
    <td>C, C++</td>
    <td><a href="https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-training-android">onnxruntime-training-android</a></td>
    <td>
      <ul>
        <li>Download the <a href="https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android">onnxruntime-training-android (full package)</a> AAR hosted at Maven Central.</li>
        <li>Change the file extension from <code>.aar</code> to <code>.zip</code>, and unzip it.</li>
        <li>Include the header files from the <code>headers</code> folder.</li>
        <li>Include the relevant <code>libonnxruntime.so</code> dynamic library from the <code>jni</code> folder in your NDK project.</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td></td>
    <td>Java/Kotlin</td>
    <td><a href="https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android">onnxruntime-training-android</a></td>
    <td>In your Android Studio Project, make the following changes to:
      <ol>
        <li>build.gradle (Project):
          <pre lang="gradle">
repositories {
    mavenCentral()
}
          </pre>
        </li>
        <li>build.gradle (Module):
          <pre lang="gradle">
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-training-android:latest.release'
}
          </pre>
        </li>
      </ol>
    </td>
  </tr>
  <tr>
    <td>iOS</td>
    <td>C, C++</td>
    <td><b>CocoaPods: onnxruntime-training-c</b></td>
    <td>
      <ul>
        <li>In your CocoaPods <code>Podfile</code>, add the <code>onnxruntime-training-c</code> pod:
          <pre>
use_frameworks!
pod 'onnxruntime-training-c'
          </pre>
        </li>
        <li>Run <code>pod install</code>.</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td></td>
    <td> Objective-C</td>
     <td><b>CocoaPods: onnxruntime-training-objc</b> </td>
     <td>
       <ul>
        <li>
          In your CocoaPods <code>Podfile</code>, add the <code>onnxruntime-training-objc</code> pod:
            <pre>
use_frameworks!
pod 'onnxruntime-training-objc'
            </pre>
        </li>
        <li>
          Run <code>pod install</code>.
        </li>
      </ul>
    </td>
  </tr>
</table>

## Large Model Training

```bash
pip install torch-ort
python -m torch_ort.configure
```

**Note**: This installs the default version of the `torch-ort` and `onnxruntime-training` packages that are mapped to specific versions of the CUDA libraries. Refer to the install options in [onnxruntime.ai](https://onnxruntime.ai).

## Inference install table for all languages

The table below lists the build variants available as officially supported packages. Others can be [built from source](../build/inferencing) from each [release branch](https://github.com/microsoft/onnxruntime/tags).

In addition to general [requirements](#requirements), please note additional requirements and dependencies in the table below:

||Official build|Nightly build|Reqs|
|---|---|---|---|
|Python|If using pip, run `pip install --upgrade pip` prior to downloading.|||
||CPU: [**onnxruntime**](https://pypi.org/project/onnxruntime)| [ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly/overview)||
||GPU (CUDA/TensorRT): [**onnxruntime-gpu**](https://pypi.org/project/onnxruntime-gpu) | [ort-nightly-gpu (dev)](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly-gpu/overview/)|[View](../execution-providers/CUDA-ExecutionProvider.md#requirements)|
||GPU (DirectML): [**onnxruntime-directml**](https://pypi.org/project/onnxruntime-directml/) | [ort-nightly-directml (dev)](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly-directml/overview/)|[View](../execution-providers/DirectML-ExecutionProvider.md#requirements)|
||OpenVINO: [**intel/onnxruntime**](https://github.com/intel/onnxruntime/releases/latest) - *Intel managed*||[View](../build/eps.md#openvino)|
||TensorRT (Jetson): [**Jetson Zoo**](https://elinux.org/Jetson_Zoo#ONNX_Runtime) - *NVIDIA managed*|||
||Azure (Cloud): [**onnxruntime-azure**](https://pypi.org/project/onnxruntime-azure/)|||
|C#/C/C++|CPU: [**Microsoft.ML.OnnxRuntime**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) |[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)||
||GPU (CUDA/TensorRT): [**Microsoft.ML.OnnxRuntime.Gpu**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu)|[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)|[View](../execution-providers/CUDA-ExecutionProvider)|
||GPU (DirectML): [**Microsoft.ML.OnnxRuntime.DirectML**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML)|[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly-directml/overview)|[View](../execution-providers/DirectML-ExecutionProvider)|
|WinML|[**Microsoft.AI.MachineLearning**](https://www.nuget.org/packages/Microsoft.AI.MachineLearning)|[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/NuGet/Microsoft.AI.MachineLearning/overview)|[View](https://docs.microsoft.com/en-us/windows/ai/windows-ml/port-app-to-nuget#prerequisites)|
|Java|CPU: [**com.microsoft.onnxruntime:onnxruntime**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime)||[View](../api/java)|
||GPU (CUDA/TensorRT): [**com.microsoft.onnxruntime:onnxruntime_gpu**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime_gpu)||[View](../api/java)|
|Android|[**com.microsoft.onnxruntime:onnxruntime-mobile**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime-mobile) ||[View](../install/index.md#install-on-ios)|
|iOS (C/C++)|CocoaPods: **onnxruntime-mobile-c**||[View](../install/index.md#install-on-ios)|
|Objective-C|CocoaPods: **onnxruntime-mobile-objc**||[View](../install/index.md#install-on-ios)|
|React Native|[**onnxruntime-react-native** (latest)](https://www.npmjs.com/package/onnxruntime-react-native)|[onnxruntime-react-native (dev)](https://www.npmjs.com/package/onnxruntime-react-native?activeTab=versions)|[View](../api/js)|
|Node.js|[**onnxruntime-node** (latest)](https://www.npmjs.com/package/onnxruntime-node)|[onnxruntime-node (dev)](https://www.npmjs.com/package/onnxruntime-node?activeTab=versions)|[View](../api/js)|
|Web|[**onnxruntime-web** (latest)](https://www.npmjs.com/package/onnxruntime-web)|[onnxruntime-web (dev)](https://www.npmjs.com/package/onnxruntime-web?activeTab=versions)|[View](../api/js)|



*Note: Dev builds created from the master branch are available for testing newer changes between official releases. Please use these at your own risk. We strongly advise against deploying these to production workloads as support is limited for dev builds.*




## Training install table for all languages

Refer to the getting started with [Optimized Training](https://onnxruntime.ai/getting-started) page for more fine-grained installation instructions.
