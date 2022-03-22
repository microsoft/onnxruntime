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

This EP is in its early stage. Only a few ops are supported and tested.
Currently, only [MobileNet v2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/),
standard ResNet 34 and UNet is tested.

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
