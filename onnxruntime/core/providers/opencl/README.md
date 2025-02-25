# OpenCL EP

This directory contains OpenCL accelerated inference execution provider. This
README briefly introduce this EP and the current status for developers.

To squeeze performance out of mobile GPU, developers usually want to fully take
control over the devices. Current tech stack, e.g., NNAPI on Android, impose
some assumption on the neural network models and is molded into the driver. For
example, if the model contain unsupported op or non-optimized network topology,
then the inference speed will be severely affected and is generally hard, if
not impossible, to circumvent if the model itself is not changed. Since new op
or topology optimization is limited to new opearting system releases. The
OpenCL EP serve the purpose to enable the accelerated ORT inference execution
on mobile GPU without sacrificing the flexiblity of the neural network models.
This enables the developers to implement the op as a final resort.


## Status and Current Limitation of the EP

This EP is in its early stage.

### OPs

Only a few ops are supported and tested. Currently, only [MobileNet
v2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/), standard ResNet 34
and UNet is tested.

### Memory Management

The memory management strategy is primal. All allocated memory is cached in the
the allocator. Reuse only happens if the requested dimension **exact match**
the cached dimension. This causes excessive memory wasting.

Improvement is left as a future effort.

## Current Limitation Imposed by ONNX Runtime

### Shape-aware Allocation

Currently, the allocator interface in ONNX Runtime only supports size-based
allocation. That is, the interface has the following signature:
```cpp
virtual void* Alloc(size_t size);
```
However, it is not enough for Texture-like memory allocation. This type of
memory exists in CUDA, DirectX, OpenGL, Vulkan and Metal. It has dedicated
hardware to accelerate reading and writing. But it requires the spatial
dimension and data type infomation to setup. So we must extend the interface to
support shape-aware allocation. It is currently implemented in this branch as
```cpp
virtual void* Alloc(const TensorShape&);
```
which is subject to discuss.

### Memory Copy to Deivce

OpenCL supports Buffer and Image2D memory. In ORT, to copy memory from host to
device during graph execution, we need to insert MemcpyFromHost Op to serve the
purpose. Current implementation only supports copy to the memory type which is
registered as the default type. To see the implication and limitation it
imposes, see [PR #10871](https://github.com/microsoft/onnxruntime/pull/10871)
for more information. Without it, we are unable to support two memory type
simultaneously. We only support operators backed by Image2D at the moment.

## Build

### Development Environment

The OpenCL EP can run directly on the development machine.
- For machines with Intel CPU and GPU, install
[Intel® SDK For OpenCL™ Applications](https://www.intel.com/content/www/us/en/developer/tools/opencl-sdk/overview.html)
- For NVIDIA GPU, OpenCL is bundled with [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
- For target mobile devices, there is no setup requirement for cross compiling.

### Build

To build with OpenCL support, the only needed option is `--use_opencl`,
otherwise, following [Build ONNX Runtime for inferencing](https://onnxruntime.ai/docs/build/inferencing.html)
to for other possible configuration.

### Build AAR package for Android

Create `opencl_aar_build_settings.json` with the
following content:

```json
{
    "build_abis": ["arm64-v8a", "armeabi-v7a"],
    "android_min_sdk_version": 24,
    "android_target_sdk_version": 29,
    "build_params": [
        "--cmake_generator=Ninja",
        "--android",
        "--use_opencl",
        "--build_java",
        "--parallel",
        "--build_shared_lib",
        "--disable_rtti",
        "--disable_ml_ops",
        "--disable_exceptions",
        "--skip_submodule_sync",
        "--skip_tests"
    ]
}
```

And build with
```bash
python tools/ci_build/github/android/build_aar_package.py --config Release opencl_aar_build_settings.json
```

The resulting aar package is a ready dependency for gradle project.

## Run model

### .onnx file format

You must not enable `--minimal_build` or related options during building to
support onnx file format.

### .ort file format

1. Build a full package of onnxruntime with `--build_wheel`, this will install
   the wheel package files to `<build_dir>\<OS>\<config>\<config>\build\lib` on
   Windows and `<build_dir>/<OS>/<config>/build/lib/` otherwise, say the
   onnxruntime repository root is `~/onnxruntime`

2. Convert onnx file format to ort file format:

   - Windows
     ```powershell
     $env:PYTHONPATH=(Resolve-Path ~\onnxruntime\build\Windows\Debug\Debug\build\lib\)
     # Do not run the python command in ~/onnxruntime
     python -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_style Fixed --providers OpenCLExecutionProvider CPUExecutionProvider -- <model.onnx>
     ```

   - Otherwise
     ```bash
     export PYTHONPATH=`realpath ~/onnxruntime/build/Linux/Debug/build/lib`
     # Do not run the python command in ~/onnxruntime
     python -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_style Fixed --providers OpenCLExecutionProvider CPUExecutionProvider -- <model.onnx>
     ```

3. The resulting ort file can be run with OpenCL EP in the minimal build.

**Why is only `--optimization_style Fixed` supported?**

In ort file format, the kernel is resolved by kernel def hash. It is not easy
to prompt the hash from CPU EP to other EP in the current implementation. This
limitation might be addressed in the future.

## Testing

Currently, OpenCL EP is not fully unit tested. The testing strategy for
development is integration test. It goes as following:

1. Generating onnx model with other framework, containing the desired op to be implemented.
2. Build onnxruntime python wheel package.
2. Run the onnx model from python environment:
   - Run with OpenCL EP only.
   - Run with CPU EP only.
   - Verify numerical difference.
   - Verify the desired op is indeed running with OpenCL.

Thoroughly testing is left as a future effort.


## Debugging

[Enabling verbose log]() and [set the verbosity level to high enough number](),
e.g. `1024` is of great help for debugging the OpenCL runtime issues. The
`set_default_logger_severity` and `set_default_logger_verbosity` python API is
your friend.

Using the following setting, `SessionOptions` and `RunOptions` is considered as
a good practice for debugging:

```python
import onnxruntime as ort
log_severity = 0
log_verbosity = 1024

ort.set_default_logger_severity(log_severity)
ort.set_default_logger_verbosity(log_verbosity)

so = ort.SessionOptions()
so.log_severity_level = log_severity
so.log_verbosity_level = log_verbosity

ro = ort.RunOptions()
ro.log_severity_level = log_severity
ro.log_verbosity_level = log_verbosity
```

For kernel code debugging, [Oclgrind](https://github.com/jrprice/Oclgrind) can
be used.

## Additional Note about the Image2D Packed Data Tensor Layout

To put a Tensor (a N-dimensional array) into Image2D texture type, the data
must be packed so that the array can fit into the 2D representation. This also
impose assumption on the dimension.

For a Image2D with dimension H x W:

```
|<-------W------->|
+-----------------+---
|                 | ^
|     Image2D     | |H
|                 | v
+-----------------+---
```

It is always allocated as 4-channel Image2D.

### 1D

1D tensor is simply stacked row by row, each row can hold W*4 number of elements.

### 2D

2D tensor is kept as is. The feature map dimension is limited by the Image2D
dimmension limit, imposed by OpenCL driver.

### 3D

3D tensor packing is not implemented at the moment.

### 4D

Only NCHW tensor is supported. It is packed as `N*H*CeilDiv(C, 4)*W*4`, then it
can be viewed as repeating a tile of H*W, and each element is a 4 elements RGBA
vector type for Image2D channels. Some tiles may have its channel filled with
grabage data if C is not divisible by 4.

This layout is inherited from MNN and TNN, the H*W tile of RGBA is repeated
CeilDiv(C, 4) times along x-axis and N times along y-axis:
```
|<-------------------Image2D width = CeilDiv(C, 4)*W------------------->|
|<-------W------->|
+-----------------+-----------------+-------...-------+-----------------+-------
|    A Tile of    |                 |                 |      H x W      | ^   ^
|      H x W      |      H x W      |       ...       |     channel     | |H  |
|                 |                 |                 |   may not full  | v   |
+-----------------+-----------------+-------...-------+-----------------+     |
|                 |                 |                 |      H x W      |     |
|      H x W      |      H x W      |       ...       |     channel     |     |
|                 |                 |                 |   may not full  |     | Image2D
+-----------------+-----------------+-------...-------+-----------------+     | height
|        .        |        .        |      .          |        .        |     | == N*H
...      .       ...       .       ...       .       ...       .      ...     |
|        .        |        .        |          .      |        .        |     |
+-----------------+-----------------+-------...-------+-----------------+     |
|                 |                 |                 |      H x W      |     |
|      H x W      |      H x W      |       ...       |     channel     |     |
|                 |                 |                 |   may not full  |     |
+-----------------+-----------------+-------...-------+-----------------+    ---
```

### 5D and more

5D and more tensor packing is not implemented at the moment.
