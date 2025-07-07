---
title: Build with different EPs
parent: Build ONNX Runtime
description: Learm how to build ONNX Runtime from source for different execution providers
nav_order: 3
redirect_from: /docs/how-to/build/eps 
---

# Build ONNX Runtime with Execution Providers
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Execution Provider Shared Libraries

The oneDNN, TensorRT, OpenVINO™, CANN, and QNN providers are built as shared libraries vs being statically linked into the main onnxruntime. This enables them to be loaded only when needed, and if the dependent libraries of the provider are not installed onnxruntime will still run fine, it just will not be able to use that provider. For non shared library providers, all dependencies of the provider must exist to load onnxruntime.

### Built files
{: .no_toc }

On Windows, shared provider libraries will be named 'onnxruntime_providers_\*.dll' (for example onnxruntime_providers_openvino.dll).
On Unix, they will be named 'libonnxruntime_providers_\*.so'
On Mac, they will be named 'libonnxruntime_providers_\*.dylib'.

There is also a shared library that shared providers depend on called onnxruntime_providers_shared (with the same naming convension applied as above).

Note: It is not recommended to put these libraries in a system location or added to a library search path (like LD_LIBRARY_PATH on Unix). If multiple versions of onnxruntime are installed on the system this can make them find the wrong libraries and lead to undefined behavior.

### Loading the shared providers
{: .no_toc }

Shared provider libraries are loaded by the onnxruntime code (do not load or depend on them in your client code). The API for registering shared or non shared providers is identical, the difference is that shared ones will be loaded at runtime when the provider is added to the session options (through a call like OrtSessionOptionsAppendExecutionProvider_OpenVINO or SessionOptionsAppendExecutionProvider_OpenVINO in the C API).
If a shared provider library cannot be loaded (if the file doesn't exist, or its dependencies don't exist or not in the path) then an error will be returned.

The onnxruntime code will look for the provider shared libraries in the same location as the onnxruntime shared library is (or the executable statically linked to the static library version).

---

## CUDA

### Prerequisites
{: .no_toc }

* Install [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn)
   * The CUDA execution provider for ONNX Runtime is built and tested with CUDA 12.x and cuDNN 9. Check [here](../execution-providers/CUDA-ExecutionProvider.md#requirements) for more version information.
   * The path to the CUDA installation must be provided via the CUDA_HOME environment variable, or the `--cuda_home` parameter. The installation directory should contain `bin`, `include` and `lib` sub-directories.
   * The path to the CUDA `bin` directory must be added to the PATH environment variable so that `nvcc` is found.
   * The path to the cuDNN installation must be provided via the CUDNN_HOME environment variable, or `--cudnn_home` parameter. In Windows, the installation directory should contain `bin`, `include` and `lib` sub-directories.
   * cuDNN 8.* requires ZLib. Follow the [cuDNN 8.9 installation guide](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-890/install-guide/index.html) to install zlib in Linux or Windows. 
   * In Windows, the path to the cuDNN bin directory must be added to the PATH environment variable so that cudnn64_8.dll is found.

### Build Instructions
{: .no_toc }

#### Windows
```
.\build.bat --use_cuda --cudnn_home <cudnn home path> --cuda_home <cuda home path>
```

#### Linux
```
./build.sh --use_cuda --cudnn_home <cudnn home path> --cuda_home <cuda home path>
```

A Dockerfile is available [here](https://github.com/microsoft/onnxruntime/blob/main/dockerfiles#cuda).

### Build Options

To specify GPU architectures (see [Compute Capability](https://developer.nvidia.com/cuda-gpus)), you can append parameters like `--cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=80;86;89`.

With `--cmake_extra_defines onnxruntime_USE_CUDA_NHWC_OPS=ON`, the CUDA EP can be compiled with additional NHWC ops. This option is not enabled by default due to the small amount of supported NHWC operators. 

Another very helpful CMake build option is to build with NVTX support (`--cmake_extra_defines onnxruntime_ENABLE_NVTX_PROFILE=ON`) that will enable much easier profiling using [Nsight Systems](https://developer.nvidia.com/nsight-systems) and correlates CUDA kernels with their actual ONNX operator.

`--enable_cuda_line_info` or `--cmake_extra_defines onnxruntime_ENABLE_CUDA_LINE_NUMBER_INFO=ON` will enable [NVCC generation of line-number information for device code](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#generate-line-info-lineinfo). It might be helpful when you run [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) tools on CUDA kernels.

If your Windows machine has multiple versions of CUDA installed and you want to use an older version of CUDA, you need append parameters like `--cuda_version <cuda version>`.

When your build machine has many CPU cores and less than 64 GB memory, there is chance of out of memory error like `nvcc error : 'cicc' died due to signal 9`. The solution is to limit number of parallel NVCC threads with parameters like `--parallel 4 --nvcc_threads 1`.

### Notes on older versions of ONNX Runtime, CUDA and Visual Studio
{: .no_toc }

* Depending on compatibility between the CUDA, cuDNN, and Visual Studio versions you are using, you may need to explicitly install an earlier version of the MSVC toolset.
* For older version of ONNX Runtime and CUDA, and Visual Studio:
    * CUDA 10.0 is [known to work](https://devblogs.microsoft.com/cppblog/cuda-10-is-now-available-with-support-for-the-latest-visual-studio-2017-versions/) with toolsets from 14.11 up to 14.16 (Visual Studio 2017 15.9), and should continue to work with future Visual Studio versions
    * CUDA 9.2 is known to work with the 14.11 MSVC toolset (Visual Studio 15.3 and 15.4)
        * To install the 14.11 MSVC toolset, see [this page](https://blogs.msdn.microsoft.com/vcblog/2017/11/15/side-by-side-minor-version-msvc-toolsets-in-visual-studio-2017).
        * To use the 14.11 toolset with a later version of Visual Studio 2017 you have two options:
          1. Setup the Visual Studio environment variables to point to the 14.11 toolset by running vcvarsall.bat, prior to running the build script. e.g. if you have VS2017 Enterprise, an x64 build would use the following command `"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" amd64 -vcvars_ver=14.11` For convenience, .\build.amd64.1411.bat will do this and can be used in the same way as .\build.bat. e.g. ` .\build.amd64.1411.bat --use_cuda`

          2. Alternatively, if you have CMake 3.13 or later you can specify the toolset version via the `--msvc_toolset` build script parameter. e.g. `.\build.bat --msvc_toolset 14.11`

* If you have multiple versions of CUDA installed on a Windows machine and are building with Visual Studio, CMake will use the build files for the highest version of CUDA it finds in the BuildCustomization folder.
e.g. C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\Common7\IDE\VC\VCTargets\BuildCustomizations\.
If you want to build with an earlier version, you must temporarily remove the 'CUDA x.y.*' files for later versions from this directory.

---

## TensorRT

See more information on the TensorRT Execution Provider [here](../execution-providers/TensorRT-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

 * Follow [instructions for CUDA execution provider](#cuda) to install CUDA and cuDNN, and setup environment variables.
 * Follow [instructions for installing TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)
   * The TensorRT execution provider for ONNX Runtime is built and tested with TensorRT 10.9.
   * The path to TensorRT installation must be provided via the `--tensorrt_home` parameter.
   * ONNX Runtime uses [TensorRT built-in parser](https://developer.nvidia.com/tensorrt/download) from `tensorrt_home` by default.
   * To use open-sourced [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt/tree/main) parser instead, add `--use_tensorrt_oss_parser` parameter in build commands below.
       * The default version of open-sourced onnx-tensorrt parser is specified in [cmake/deps.txt](https://github.com/microsoft/onnxruntime/blob/main/cmake/deps.txt).
       * To specify a different version of onnx-tensorrt parser:
         * Select the commit of [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt/commits) that you preferred;
         * Run `sha1sum` command with downloaded onnx-tensorrt zip file to acquire the SHA1 hash 
         * Update [cmake/deps.txt](https://github.com/microsoft/onnxruntime/blob/main/cmake/deps.txt) with updated onnx-tensorrt commit and hash info.
       * Please make sure TensorRT built-in parser/open-sourced onnx-tensorrt specified in [cmake/deps.txt](https://github.com/microsoft/onnxruntime/blob/main/cmake/deps.txt) are **version-matched**, if enabling `--use_tensorrt_oss_parser`. 
         * i.e It's version-matched if assigning `tensorrt_home` with path to TensorRT-10.9 built-in binaries and onnx-tensorrt [10.9-GA branch](https://github.com/onnx/onnx-tensorrt/tree/release/10.9-GA) specified in [cmake/deps.txt](https://github.com/microsoft/onnxruntime/blob/main/cmake/deps.txt).


### **[Note to ORT 1.21/1.22 open-sourced parser users]** 

* ORT 1.21/1.22 link against onnx-tensorrt 10.8-GA/10.9-GA, which requires newly released onnx 1.18.
  * Here's a temporarily fix to preview on onnx-tensorrt 10.8-GA/10.9-GA when building ORT 1.21/1.22: 
    * Replace the [onnx line in cmake/deps.txt](https://github.com/microsoft/onnxruntime/blob/rel-1.21.0/cmake/deps.txt#L38) 
      with `onnx;https://github.com/onnx/onnx/archive/e709452ef2bbc1d113faf678c24e6d3467696e83.zip;c0b9f6c29029e13dea46b7419f3813f4c2ca7db8`  
    * Download [this](https://github.com/microsoft/onnxruntime/blob/7b2733a526c12b5ef4475edd47fd9997ebc2b2c6/cmake/patches/onnx/onnx.patch) as raw file and save file to [cmake/patches/onnx/onnx.patch](https://github.com/microsoft/onnxruntime/blob/rel-1.21.0/cmake/patches/onnx/onnx.patch) (do not copy/paste from browser, as it might alter line break type)
    * Build ORT with trt-related flags above (including `--use_tensorrt_oss_parser`)
  * The [onnx 1.18](https://github.com/onnx/onnx/releases/tag/v1.18.0) is supported by latest ORT main branch. Please checkout main branch and build ORT-TRT with `--use_tensorrt_oss_parser` to enable OSS parser with full onnx 1.18 support.

### Build Instructions
{: .no_toc }

#### Windows
```bash
# to build with tensorrt built-in parser
.\build.bat --config Release --parallel  --cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=native' --cudnn_home <path to cuDNN home> --cuda_home <path to CUDA home> --use_tensorrt --tensorrt_home <path to TensorRT home> --cmake_generator "Visual Studio 17 2022"

# to build with specific version of open-sourced onnx-tensorrt parser configured in cmake/deps.txt
.\build.bat --config Release --parallel  --cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=native' --cudnn_home <path to cuDNN home> --cuda_home <path to CUDA home> --use_tensorrt --tensorrt_home <path to TensorRT home> --use_tensorrt_oss_parser --cmake_generator "Visual Studio 17 2022" 
```

#### Linux

```bash
# to build with tensorrt built-in parser
./build.sh --config Release --parallel --cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=native' --cudnn_home <path to cuDNN e.g. /usr/lib/x86_64-linux-gnu/> --cuda_home <path to folder for CUDA e.g. /usr/local/cuda> --use_tensorrt --tensorrt_home <path to TensorRT home>

# to build with specific version of open-sourced onnx-tensorrt parser configured in cmake/deps.txt
./build.sh --config Release --parallel --cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=native' --cudnn_home <path to cuDNN e.g. /usr/lib/x86_64-linux-gnu/> --cuda_home <path to folder for CUDA e.g. /usr/local/cuda> --use_tensorrt --use_tensorrt_oss_parser --tensorrt_home <path to TensorRT home> --skip_submodule_sync
```

Dockerfile instructions are available [here](https://github.com/microsoft/onnxruntime/tree/main/dockerfiles#tensorrt)

**Note** Building with `--use_tensorrt_oss_parser` with TensorRT 8.X requires additional flag --cmake_extra_defines onnxruntime_USE_FULL_PROTOBUF=ON

---

## NVIDIA Jetson TX1/TX2/Nano/Xavier/Orin

### Build Instructions
{: .no_toc }

These instructions are for the latest [JetPack SDK](https://developer.nvidia.com/embedded/jetpack).

1. Clone the ONNX Runtime repo on the Jetson host

   ```bash
   git clone --recursive https://github.com/microsoft/onnxruntime
   ```

2. Specify the CUDA compiler, or add its location to the PATH.

   1. JetPack 5.x users can upgrade to the latest CUDA release without updating the JetPack version or Jetson Linux BSP (Board Support Package). 
   
      1. For JetPack 5.x users, CUDA>=11.8 and GCC>9.4 are required to be installed on and after ONNX Runtime 1.17. 

      2. Check [this official blog](https://developer.nvidia.com/blog/simplifying-cuda-upgrades-for-nvidia-jetson-users/) for CUDA upgrade instruction (CUDA 12.2 has been verified on JetPack 5.1.2 on Jetson Xavier NX).

         1. If there's no `libnvcudla.so` under `/usr/local/cuda-12.2/compat`: `sudo apt-get install -y cuda-compat-12-2` and add `export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:/usr/local/cuda-12.2/compat:$LD_LIBRARY_PATH"` to `~/.bashrc`.

      3. Check [here](https://developer.nvidia.com/cuda-gpus#collapse5) for compute capability datasheet.

   2. CMake can't automatically find the correct `nvcc` if it's not in the `PATH`. `nvcc` can be added to `PATH` via:

      ```bash
      export PATH="/usr/local/cuda/bin:${PATH}"
      ```

       or:

      ```bash
      export CUDACXX="/usr/local/cuda/bin/nvcc"
      ```

   3. Update TensorRT libraries
      
      1. Jetpack 5.x supports up to TensorRT 8.5. Jetpack 6.x are equipped with TensorRT 8.6-10.3.
      
      2. Jetpack 6.x users can download latest TensorRT 10 TAR package for **jetpack** on [TensorRT SDK website](https://developer.nvidia.com/tensorrt/download/10x).
      
      3. Check [here](../execution-providers/TensorRT-ExecutionProvider.md#requirements) for TensorRT/CUDA support matrix among all ONNX Runtime versions.

3. Install the ONNX Runtime build dependencies on the Jetpack host:

   ```bash
   sudo apt install -y --no-install-recommends \
     build-essential software-properties-common libopenblas-dev \
     libpython3.10-dev python3-pip python3-dev python3-setuptools python3-wheel
   ```

4. Cmake is needed to build ONNX Runtime. Please check the minimum required CMake version [here](https://github.com/microsoft/onnxruntime/blob/main/cmake/CMakeLists.txt#L6). Download from https://cmake.org/download/ and add cmake executable to `PATH` to use it.

5. Build the ONNX Runtime Python wheel:

   1. Build `onnxruntime-gpu` wheel with CUDA and TensorRT support (update paths to CUDA/CUDNN/TensorRT libraries if necessary):

      ```bash
      ./build.sh --config Release --update --build --parallel --build_wheel \
      --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
      --tensorrt_home /usr/lib/aarch64-linux-gnu
      ```

​	Notes:

* By default, `onnxruntime-gpu` wheel file will be captured under `path_to/onnxruntime/build/Linux/Release/dist/` (build path can be customized by adding `--build_dir` followed by a customized path to the build command above).

* Append `--skip_tests --cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=native' 'onnxruntime_BUILD_UNIT_TESTS=OFF' 'onnxruntime_USE_FLASH_ATTENTION=OFF'
'onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF'` to the build command to opt out optional features and reduce build time.

* For a portion of Jetson devices like the Xavier series, higher power mode involves more cores (up to 6) to compute but it consumes more resource when building ONNX Runtime. Set `--parallel 1` in the build command if OOM happens and system is hanging.

## TensorRT-RTX

See more information on the NV TensorRT RTX Execution Provider [here](../execution-providers/TensorRTRTX-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

 * Follow [instructions for CUDA execution provider](#cuda) to install CUDA and setup environment variables.
 * Intall TensorRT for RTX from [here](https://developer.nvidia.com/tensorrt-rtx))

### Build Instructions
{: .no_toc }

```bash
`build.bat --config Release --parallel 32 --build_dir _build --build_shared_lib --use_nv_tensorrt_rtx --tensorrt_rtx_home <path to TensorRT for RTX home>  --cuda_home <path to CUDA home> --cmake_generator "Visual Studio 17 2022" --use_vcpkg` 
'''

Update the --tensorrt_rtx_home and --cuda_home with correct paths to CUDA and TensorRT-RTX installations.

## oneDNN

See more information on oneDNN (formerly DNNL) [here](../execution-providers/oneDNN-ExecutionProvider.md).

### Build Instructions
{: .no_toc }


The DNNL execution provider can be built for Intel CPU or GPU. To build for Intel GPU, install [Intel SDK for OpenCL Applications](https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk.html) or build OpenCL from [Khronos OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK). Pass in the OpenCL SDK path as dnnl_opencl_root to the build command. Install the latest GPU driver - [Windows graphics driver](https://downloadcenter.intel.com/product/80939/Graphics), [Linux graphics compute runtime and OpenCL driver](https://github.com/intel/compute-runtime/releases).

For CPU
#### Windows
`.\build.bat --use_dnnl`

#### Linux
`./build.sh --use_dnnl`

For GPU
#### Windows 

`.\build.bat --use_dnnl --dnnl_gpu_runtime ocl --dnnl_opencl_root "c:\program files (x86)\intelswtools\sw_dev_tools\opencl\sdk"`
#### Linux

`./build.sh --use_dnnl --dnnl_gpu_runtime ocl --dnnl_opencl_root "/opt/intel/sw_dev_tools/opencl-sdk"`

#### Build Phython Wheel


OneDNN EP build supports building Python wheel for both Windows and linux using flag --build_wheel

`.\build.bat --config RelWithDebInfo --parallel --build_shared_lib --cmake_generator "Visual Studio 16 2019" --build_wheel --use_dnnl --dnnl_gpu_runtime ocl --dnnl_opencl_root "C:\Program Files (x86)\IntelSWTools\system_studio_2020\OpenCL\sdk"`

---

## OpenVINO

See more information on the OpenVINO™ Execution Provider [here](../execution-providers/OpenVINO-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

1. Install the OpenVINO™ offline/online installer from Intel<sup>®</sup> Distribution of OpenVINO™<sup>TM</sup> Toolkit **Release 2024.3** for the appropriate OS and target hardware:
   * [Windows - CPU, GPU, NPU](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?PACKAGE=OPENVINO_BASE&VERSION=v_2024_3_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=ARCHIVE).
   * [Linux - CPU, GPU, NPU](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?PACKAGE=OPENVINO_BASE&VERSION=v_2024_3_0&OP_SYSTEM=LINUX&DISTRIBUTION=ARCHIVE)

   Follow [documentation](https://docs.openvino.ai/2024/home.html) for detailed instructions.

  *2024.5 is the current recommended OpenVINO™ version. [OpenVINO™ 2024.5](https://docs.openvino.ai/2024/index.html) is minimal OpenVINO™ version requirement.*

2. Configure the target hardware with specific follow on instructions:
   * To configure Intel<sup>®</sup> Processor Graphics(GPU) please follow these instructions: [Windows](https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html#windows), [Linux](https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html#linux)


3. Initialize the OpenVINO™ environment by running the setupvars script as shown below. This is a required step:
   * For Windows:
   ```
      C:\<openvino_install_directory>\setupvars.bat
   ```
   * For Linux:
   ```
      $ source <openvino_install_directory>/setupvars.sh
   ```
   **Note:** If you are using a dockerfile to use OpenVINO™ Execution Provider, sourcing OpenVINO™ won't be possible within the dockerfile. You would have to explicitly set the LD_LIBRARY_PATH to point to OpenVINO™ libraries location. Refer our [dockerfile](https://github.com/microsoft/onnxruntime/blob/main/dockerfiles/Dockerfile.openvino).

### Build Instructions
{: .no_toc }

#### Windows

```
.\build.bat --config RelWithDebInfo --use_openvino <hardware_option> --build_shared_lib --build_wheel
```

*Note: The default Windows CMake Generator is Visual Studio 2019, but you can also use the newer Visual Studio 2022 by passing `--cmake_generator "Visual Studio 17 2022"` to `.\build.bat`*

#### Linux

```bash
./build.sh --config RelWithDebInfo --use_openvino <hardware_option> --build_shared_lib --build_wheel
```

* `--build_wheel` Creates python wheel file in dist/ folder. Enable it when building from source.
* `--use_openvino` builds the OpenVINO™ Execution Provider in ONNX Runtime.
* `<hardware_option>`: Specifies the default hardware target for building OpenVINO™ Execution Provider. This can be overriden dynamically at runtime with another option (refer to [OpenVINO™-ExecutionProvider](../execution-providers/OpenVINO-ExecutionProvider.md#summary-of-options) for more details on dynamic device selection). Below are the options for different Intel target devices.

Refer to [Intel GPU device naming convention](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html#device-naming-convention) for specifying the correct hardware target in cases where both integrated and discrete GPU's co-exist.

| Hardware Option | Target Device |
| --------------- | ------------------------|
| <code>CPU</code> | Intel<sup>®</sup> CPUs |
| <code>GPU</code> | Intel<sup>®</sup> Integrated Graphics |
| <code>GPU.0</code> | Intel<sup>®</sup> Integrated Graphics |
| <code>GPU.1</code> | Intel<sup>®</sup> Discrete Graphics |
| <code>NPU</code> | Intel<sup>®</sup> Neural Processor Unit |
| <code>HETERO:DEVICE_TYPE_1,DEVICE_TYPE_2,DEVICE_TYPE_3...</code> | All Intel<sup>®</sup> silicons mentioned above |
| <code>MULTI:DEVICE_TYPE_1,DEVICE_TYPE_2,DEVICE_TYPE_3...</code> | All Intel<sup>®</sup> silicons mentioned above |
| <code>AUTO:DEVICE_TYPE_1,DEVICE_TYPE_2,DEVICE_TYPE_3...</code> | All Intel<sup>®</sup> silicons mentioned above |

Specifying Hardware Target for HETERO or Multi or AUTO device Build:

HETERO:DEVICE_TYPE_1,DEVICE_TYPE_2,DEVICE_TYPE_3...
The DEVICE_TYPE can be any of these devices from this list ['CPU','GPU', 'NPU']

A minimum of two device's should be specified for a valid HETERO or MULTI or AUTO device build.

```
Example's: HETERO:GPU,CPU or AUTO:GPU,CPU or MULTI:GPU,CPU
```

#### Disable subgraph partition Feature
* Builds the OpenVINO™ Execution Provider in ONNX Runtime with sub graph partitioning disabled.

* With this option enabled. Fully supported models run on OpenVINO Execution Provider else they completely fall back to default CPU EP.

* To enable this feature during build time. Use `--use_openvino ` `<hardware_option>_NO_PARTITION`

```
Usage: --use_openvino CPU_FP32_NO_PARTITION or --use_openvino GPU_FP32_NO_PARTITION or
       --use_openvino GPU_FP16_NO_PARTITION 
```

For more information on OpenVINO™ Execution Provider&#39;s ONNX Layer support, Topology support, and Intel hardware enabled, please refer to the document [OpenVINO™-ExecutionProvider](../execution-providers/OpenVINO-ExecutionProvider.md)

---

## QNN
See more information on the QNN execution provider [here](../execution-providers/QNN-ExecutionProvider.md).

### Prerequisites
{: .no_toc }
* Install the Qualcomm AI Engine Direct SDK (Qualcomm Neural Network SDK) [Linux/Android/Windows](https://qpm.qualcomm.com/main/tools/details/qualcomm_ai_engine_direct)

* Install [cmake-3.28](https://cmake.org/download/) or higher.

* Install Python 3.10 or higher.
  * [Python 3.12 for Windows Arm64](https://www.python.org/ftp/python/3.12.9/python-3.12.9-arm64.exe)
  * [Python 3.12 for Windows x86-64](https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe)
  * Note: Windows on Arm supports a x86-64 Python environment via emulation. Ensure that the Arm64 Python environment is actived for a native Arm64 ONNX Runtime build.

* Checkout the source tree:

   ```bash
   git clone --recursive https://github.com/Microsoft/onnxruntime.git
   cd onnxruntime
   ```

* Install ONNX Runtime Python dependencies.
   ```bash
   pip install -r requirements.txt
   ```

### Build Options
{: .no_toc }

* `--use_qnn [QNN_LIBRARY_KIND]`: Builds the QNN Execution provider. `QNN_LIBRARY_KIND` is optional and specifies whether to build the QNN Execution Provider as a shared library (default) or static library.
  * `--use_qnn` or `--use_qnn shared_lib`: Builds the QNN Execution Provider as a shared library.
  * `--use_qnn static_lib`: Builds QNN Execution Provider as a static library linked into ONNX Runtime. This is required for Android builds.
* `--qnn_home QNN_SDK_PATH`: The path to the Qualcomm AI Engine Direct SDK.
  * Example on Windows: `--qnn_home 'C:\Qualcomm\AIStack\QAIRT\2.31.0.250130'`
  * Example on Linux: `--qnn_home /opt/qcom/aistack/qairt/2.31.0.250130`
* `--build_wheel`: Enables Python bindings and builds Python wheel.
* `--arm64`: Cross-compile for Arm64.
* `--arm64ec`: Cross-compile for Arm64EC. Arm64EC code runs with native performance and is interoperable with x64 code running under emulation within the same process on a Windows on Arm device. Refer to the [Arm64EC Overview](https://learn.microsoft.com/en-us/windows/arm/arm64ec).

Run `python tools/ci_build/build.py --help` for a description of all available build options.

### Build Instructions
{: .no_toc }

#### Windows (native x86-64 or native Arm64)
```
.\build.bat --use_qnn --qnn_home [QNN_SDK_PATH] --build_shared_lib --build_wheel --cmake_generator "Visual Studio 17 2022" --config Release --parallel --skip_tests --build_dir build\Windows
```

Notes:
* Not all Qualcomm backends (e.g., HTP) are supported for model execution on a native x86-64 build. Refer to the [Qualcomm SDK backend documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/backend.html) for more information.
* Even if a Qualcomm backend does not support execution on x86-64, the QNN Execution provider may be able to [generate compiled models](../execution-providers/QNN-ExecutionProvider.md#qnn-context-binary-cache-feature) for the Qualcomm backend.

#### Windows (Arm64 cross-compile target)
```
.\build.bat --arm64 --use_qnn --qnn_home [QNN_SDK_PATH] --build_shared_lib --build_wheel --cmake_generator "Visual Studio 17 2022" --config Release --parallel --build_dir build\Windows
```

#### Windows (Arm64EC cross-compile target)
```
.\build.bat --arm64ec --use_qnn --qnn_home [QNN_SDK_PATH] --build_shared_lib --build_wheel --cmake_generator "Visual Studio 17 2022" --config Release --parallel --build_dir build\Windows
```

#### Windows (Arm64X cross-compile target)
Use the `build_arm64x.bat` script to build Arm64X binaries. Arm64X binaries bundle both Arm64 and Arm64EC code, making Arm64X compatible with both Arm64 and Arm64EC processes on a Windows on Arm device. Refer to the [Arm64X PE files overview](https://learn.microsoft.com/en-us/windows/arm/arm64x-pe).

```
.\build_arm64x.bat --use_qnn --qnn_home [QNN_SDK_PATH] --build_shared_lib --cmake_generator "Visual Studio 17 2022" --config Release --parallel
```
Notes:
* Do not specify a `--build_dir` option because `build_arm64x.bat` sets specific build directories.
* The above command places Arm64X binaries in the `.\build\arm64ec-x\Release\Release\` directory.

#### Linux (x86_64)
```
./build.sh --use_qnn --qnn_home [QNN_SDK_PATH] --build_shared_lib --build_wheel --config Release --parallel --skip_tests --build_dir build/Linux
```

#### Android (cross-compile):

Please reference [Build OnnxRuntime For Android](android.md)
```
# on Windows
.\build.bat --build_shared_lib --android --config Release --parallel --use_qnn static_lib --qnn_home [QNN_SDK_PATH] --android_sdk_path [android_SDK path] --android_ndk_path [android_NDK path] --android_abi arm64-v8a --android_api [api-version] --cmake_generator Ninja --build_dir build\Android

# on Linux
./build.sh --build_shared_lib --android --config Release --parallel --use_qnn static_lib --qnn_home [QNN_SDK_PATH] --android_sdk_path [android_SDK path] --android_ndk_path [android_NDK path] --android_abi arm64-v8a --android_api [api-version] --cmake_generator Ninja --build_dir build/Android
```

---

## DirectML
See more information on the DirectML execution provider [here](../execution-providers/DirectML-ExecutionProvider.md).
### Windows
{: .no_toc }

```
.\build.bat --use_dml
```
### Notes
{: .no_toc }

The DirectML execution provider supports building for both x64 and x86 architectures. DirectML is only supported on Windows.

---

## Arm Compute Library
See more information on the ACL Execution Provider [here](../execution-providers/community-maintained/ACL-ExecutionProvider.md).

### Build Instructions
{: .no_toc }

You must first build Arm Compute Library 24.07 for your platform as described in the [documentation](https://github.com/ARM-software/ComputeLibrary).
See [here](inferencing.md#arm) for information on building for Arm®-based devices.

Add the following options to `build.sh` to enable the ACL Execution Provider:

```
--use_acl --acl_home=/path/to/ComputeLibrary --acl_libs=/path/to/ComputeLibrary/build
```

## Arm NN

See more information on the Arm NN Execution Provider [here](../execution-providers/community-maintained/ArmNN-ExecutionProvider.md).

### Prerequisites
{: .no_toc }


* Supported backend: i.MX8QM Armv8 CPUs
* Supported BSP: i.MX8QM BSP
  * Install i.MX8QM BSP: `source fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4*.sh`
* Set up the build environment

```bash
source /opt/fsl-imx-xwayland/4.*/environment-setup-aarch64-poky-linux
alias cmake="/usr/bin/cmake -DCMAKE_TOOLCHAIN_FILE=$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake"
```

* See [here](inferencing.md#arm) for information on building for Arm-based devices

### Build Instructions
{: .no_toc }


```bash
./build.sh --use_armnn
```

The Relu operator is set by default to use the CPU execution provider for better performance. To use the Arm NN implementation build with --armnn_relu flag

```bash
./build.sh --use_armnn --armnn_relu
```

The Batch Normalization operator is set by default to use the CPU execution provider. To use the Arm NN implementation build with --armnn_bn flag

```bash
./build.sh --use_armnn --armnn_bn
```

To use a library outside the normal environment you can set a custom path by providing the --armnn_home and --armnn_libs parameters to define the path to the Arm NN home directory and build directory respectively. 
The Arm Compute Library home directory and build directory must also be available, and can be specified if needed using --acl_home and --acl_libs respectively.

```bash
./build.sh --use_armnn --armnn_home /path/to/armnn --armnn_libs /path/to/armnn/build  --acl_home /path/to/ComputeLibrary --acl_libs /path/to/acl/build
```

---

## RKNPU
See more information on the RKNPU Execution Provider [here](../execution-providers/community-maintained/RKNPU-ExecutionProvider.md).

### Prerequisites
{: .no_toc }


* Supported platform: RK1808 Linux
* See [here](inferencing.md#arm) for information on building for Arm-based devices
* Use gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu instead of gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf, and modify CMAKE_CXX_COMPILER & CMAKE_C_COMPILER in tool.cmake:
  
```
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
```

### Build Instructions
{: .no_toc }

#### Linux

1. Download [rknpu_ddk](https://github.com/airockchip/rknpu_ddk.git) to any directory.

2. Build ONNX Runtime library and test:

    ```bash
    ./build.sh --arm --use_rknpu --parallel --build_shared_lib --build_dir build_arm --config MinSizeRel --cmake_extra_defines RKNPU_DDK_PATH=<Path To rknpu_ddk> CMAKE_TOOLCHAIN_FILE=<Path To tool.cmake> ONNX_CUSTOM_PROTOC_EXECUTABLE=<Path To protoc>
    ```

3. Deploy ONNX runtime and librknpu_ddk.so on the RK1808 board:

    ```bash
    libonnxruntime.so.1.2.0
    onnxruntime_test_all
    rknpu_ddk/lib64/librknpu_ddk.so
    ```

---

## AMD Vitis AI
See more information on the Vitis AI Execution Provider [here](../execution-providers/Vitis-AI-ExecutionProvider.md).

### Windows
{: .no_toc }

From the Visual Studio Developer Command Prompt or Developer PowerShell, execute the following command:

```
.\build.bat --use_vitisai --build_shared_lib --parallel --config Release
```

If you wish to leverage the Python APIs, please include the `--build_wheel` flag:

```
.\build.bat --use_vitisai --build_shared_lib --parallel --config Release --build_wheel
```

You can override also override the installation location by specifying CMAKE_INSTALL_PREFIX via the cmake_extra_defines parameter.
e.g.

```
.\build.bat --use_vitisai --build_shared_lib --parallel --config Release --cmake_extra_defines CMAKE_INSTALL_PREFIX=D:\onnxruntime
```
### Linux
{: .no_toc }

Currently Linux support is only enabled for AMD Adapable SoCs.  Please refer to the guidance [here](../execution-providers/Vitis-AI-ExecutionProvider.md#installation-for-amd-adaptable-socs) for SoC targets.

---

## AMD MIGraphX

See more information on the MIGraphX Execution Provider [here](../execution-providers/MIGraphX-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

* Install [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.1/)
  * The MIGraphX execution provider for ONNX Runtime is built and tested with ROCm6.3.1
* Install [MIGraphX](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX)
  * The path to MIGraphX installation must be provided via the `--migraphx_home parameter`.

### Build Instructions
{: .no_toc }

#### Linux

```bash
./build.sh --config <Release|Debug|RelWithDebInfo> --parallel --use_migraphx --migraphx_home <path to MIGraphX home>
```

Dockerfile instructions are available [here](https://github.com/microsoft/onnxruntime/blob/main/dockerfiles#migraphx).

#### Build Phython Wheel

`./build.sh --config Release --build_wheel --parallel --use_migraphx --migraphx_home /opt/rocm`

Then the python wheels(*.whl) could be found at ```./build/Linux/Release/dist```.

---

## AMD ROCm

See more information on the ROCm Execution Provider [here](../execution-providers/ROCm-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

* Install [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.1/)
  * The ROCm execution provider for ONNX Runtime is built and tested with ROCm6.3.1

### Build Instructions
{: .no_toc }

#### Linux

```bash
./build.sh --config <Release|Debug|RelWithDebInfo> --parallel --use_rocm --rocm_home <path to ROCm home>
```

Dockerfile instructions are available [here](https://github.com/microsoft/onnxruntime/tree/main/dockerfiles#rocm).

#### Build Phython Wheel

`./build.sh --config Release --build_wheel --parallel --use_rocm --rocm_home /opt/rocm`

Then the python wheels(*.whl) could be found at ```./build/Linux/Release/dist```.

---

## NNAPI

Usage of NNAPI on Android platforms is via the NNAPI Execution Provider (EP).

See the [NNAPI Execution Provider](../execution-providers/NNAPI-ExecutionProvider.md) documentation for more details.

The pre-built ONNX Runtime Mobile package for Android includes the NNAPI EP.

If performing a custom build of ONNX Runtime, support for the NNAPI EP or CoreML EP must be enabled when building.

### Create a minimal build with NNAPI EP support

Please see [the instructions](./android.md) for setting up the Android environment required to build. The Android build can be cross-compiled on Windows or Linux.

Once you have all the necessary components setup, follow the instructions to [create the custom build](./custom.md), with the following changes:

* Replace `--minimal_build` with `--minimal_build extended` to enable support for execution providers that dynamically create kernels at runtime, which is required by the NNAPI EP.
* Add `--use_nnapi` to include the NNAPI EP in the build

#### Example build commands with the NNAPI EP enabled

Windows example:

```dos
<ONNX Runtime repository root>.\build.bat --config MinSizeRel --android --android_sdk_path D:\Android --android_ndk_path D:\Android\ndk\21.1.6352462\ --android_abi arm64-v8a --android_api 29 --cmake_generator Ninja --minimal_build extended --use_nnapi --disable_ml_ops --disable_exceptions --build_shared_lib --skip_tests --include_ops_by_config <config file from model conversion>
```

Linux example:

```bash
<ONNX Runtime repository root>./build.sh --config MinSizeRel --android --android_sdk_path /Android --android_ndk_path /Android/ndk/21.1.6352462/ --android_abi arm64-v8a --android_api 29 --minimal_build extended --use_nnapi --disable_ml_ops --disable_exceptions --build_shared_lib --skip_tests --include_ops_by_config <config file from model conversion>`
```

## CoreML

Usage of CoreML on iOS and macOS platforms is via the CoreML EP.

See the [CoreML Execution Provider](../execution-providers/CoreML-ExecutionProvider.md) documentation for more details.

The pre-built ONNX Runtime Mobile package for iOS includes the CoreML EP.

### Create a minimal build with CoreML EP support

Please see [the instructions](./ios.md) for setting up the iOS environment required to build. The iOS/macOS build must be performed on a mac machine.

Once you have all the necessary components setup, follow the instructions to [create the custom build](./custom.md), with the following changes:

* Replace `--minimal_build` with `--minimal_build extended` to enable support for execution providers that dynamically create kernels at runtime, which is required by the CoreML EP.
* Add `--use_coreml` to include the CoreML EP in the build

## XNNPACK

Usage of XNNPACK on Android/iOS/Windows/Linux platforms is via the XNNPACK EP.

See the [XNNPACK Execution Provider](../execution-providers/Xnnpack-ExecutionProvider.md) documentation for more details.

The pre-built ONNX Runtime package([`onnxruntime-android`](https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android)) for Android includes the XNNPACK EP.

The pre-built ONNX Runtime Mobile package for iOS, `onnxruntime-c` and `onnxruntime-objc` in [CocoaPods](https://cocoapods.org/), includes the XNNPACK EP. (Package `onnxruntime-objc` with XNNPACK will be available since 1.14.)


If performing a custom build of ONNX Runtime, support for the XNNPACK EP must be enabled when building.

### Build for Android
#### Create a minimal build with XNNPACK EP support

Please see [the instructions](./android.md) for setting up the Android environment required to build. The Android build can be cross-compiled on Windows or Linux.

Once you have all the necessary components setup, follow the instructions to [create the custom build](./custom.md), with the following changes:

* Replace `--minimal_build` with `--minimal_build extended` to enable support for execution providers that dynamically create kernels at runtime, which is required by the XNNPACK EP.
* Add `--use_xnnpack` to include the XNNPACK EP in the build

##### Example build commands with the XNNPACK EP enabled

Windows example:

```bash
<ONNX Runtime repository root>.\build.bat --config MinSizeRel --android --android_sdk_path D:\Android --android_ndk_path D:\Android\ndk\21.1.6352462\ --android_abi arm64-v8a --android_api 29 --cmake_generator Ninja --minimal_build extended --use_xnnpack --disable_ml_ops --disable_exceptions --build_shared_lib --skip_tests --include_ops_by_config <config file from model conversion>
```

Linux example:

```bash
<ONNX Runtime repository root>./build.sh --config MinSizeRel --android --android_sdk_path /Android --android_ndk_path /Android/ndk/21.1.6352462/ --android_abi arm64-v8a --android_api 29 --minimal_build extended --use_xnnpack --disable_ml_ops --disable_exceptions --build_shared_lib --skip_tests --include_ops_by_config <config file from model conversion>`
```
If you don't mind MINIMAL build, you can use the following command to build XNNPACK EP for Android:
Linux example:
```bash
./build.sh --cmake_generator "Ninja" --android  --android_sdk_path /Android --android_ndk_path /Android/ndk/21.1.6352462/ --android_abi arm64-v8a --android_api 29 --use_xnnpack
```
### Build for iOS (available since 1.14)
A Mac machine is required to build package for iOS. Please follow this [guide](./ios.md) to set up environment firstly.
#### Create a minimal build with XNNPACK EP support

Once you have all the necessary components setup, follow the instructions to [create the custom build](./custom.md), with the following changes:

* Replace `--minimal_build` with `--minimal_build extended` to enable support for execution providers that dynamically create kernels at runtime, which is required by the XNNPACK EP.
* Add `--use_xnnpack` to include the XNNPACK EP in the build

```dos
<ONNX Runtime repository root>./build.sh --config <Release|Debug|RelWithDebInfo|MinSizeRel> --use_xcode \
           --ios --ios_sysroot iphoneos --osx_arch arm64 --apple_deploy_target <minimal iOS version> --use_xnnpack --minimal_build extended --disable_ml_ops --disable_exceptions --build_shared_lib --skip_tests --include_ops_by_config <config file from model conversion>
```

### Build for Windows
```dos
<ONNX Runtime repository root>.\build.bat --config <Release|Debug|RelWithDebInfo> --use_xnnpack
```
### Build for Linux
```bash
<ONNX Runtime repository root>./build.sh --config <Release|Debug|RelWithDebInfo> --use_xnnpack
```

---

## CANN

See more information on the CANN Execution Provider [here](../execution-providers/community-maintained/CANN-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

1. Install the CANN Toolkit for the appropriate OS and target hardware by following [documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/51RC1alphaX/softwareinstall/instg/atlasdeploy_03_0017.html) for detailed instructions, please.

2. Initialize the CANN environment by running the script as shown below.

   ```bash
   # Default path, change it if needed.
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

### Build Instructions
{: .no_toc }

#### Linux

```bash
./build.sh --config <Release|Debug|RelWithDebInfo> --build_shared_lib --parallel --use_cann
```

### Notes
{: .no_toc }

* The CANN execution provider supports building for both x64 and aarch64 architectures.
* CANN excution provider now is only supported on Linux.

## Azure

See the [Azure Execution Provider](../execution-providers/Azure-ExecutionProvider.md) documentation for more details.

### Prerequisites

For Linux, before building, please:

* install openssl dev package into the system, which is openssl-dev for redhat and libssl-dev for ubuntu.
* if have multiple openssl dev versions installed in the system, please set environment variable "OPENSSL_ROOT_DIR" to the desired version, for example:

```base
set OPENSSL_ROOT_DIR=/usr/local/ssl3.x/
```

### Build Instructions

#### Windows

```dos
build.bat --config <Release|Debug|RelWithDebInfo> --build_shared_lib --build_wheel --use_azure
```

#### Linux

```bash
./build.sh --config <Release|Debug|RelWithDebInfo> --build_shared_lib --build_wheel --use_azure
```
