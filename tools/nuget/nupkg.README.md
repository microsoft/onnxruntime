## About

![ONNX Runtime Logo](https://raw.githubusercontent.com/microsoft/onnxruntime/main/docs/images/ONNX_Runtime_logo_dark.png)

**ONNX Runtime is a cross-platform machine-learning inferencing accelerator**.

**ONNX Runtime** can enable faster customer experiences and lower costs, supporting models from deep learning frameworks such as PyTorch and TensorFlow/Keras as well as classical machine learning libraries such as scikit-learn, LightGBM, XGBoost, etc.
ONNX Runtime is compatible with different hardware, drivers, and operating systems, and provides optimal performance by leveraging hardware accelerators where applicable alongside graph optimizations and transforms.

Learn more &rarr; [here](https://www.onnxruntime.ai/docs)

## NuGet Packages

### ONNX Runtime Native packages

#### Microsoft.ML.OnnxRuntime
  - Native libraries for all supported platforms
  - CPU Execution Provider
  - CoreML Execution Provider on macOS/iOS
    - https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
  - XNNPACK Execution Provider on Android/iOS
    - https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html

#### Microsoft.ML.OnnxRuntime.Gpu
  - Windows and Linux
  - TensorRT Execution Provider
    - https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
  - CUDA Execution Provider
    - https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
  - CPU Execution Provider

#### Microsoft.ML.OnnxRuntime.DirectML
  - Windows
  - DirectML Execution Provider
    - https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
  - CPU Execution Provider

#### Microsoft.ML.OnnxRuntime.QNN
  - 64-bit Windows
  - QNN Execution Provider
    - https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
  - CPU Execution Provider

#### Intel.ML.OnnxRuntime.OpenVino
  - 64-bit Windows
  - OpenVINO Execution Provider
    - https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html
  - CPU Execution Provider


### Other packages

#### Microsoft.ML.OnnxRuntime.Managed
  - C# language bindings

#### Microsoft.ML.OnnxRuntime.Extensions
  - Custom operators for pre/post processing on all supported platforms.
    - https://github.com/microsoft/onnxruntime-extensions
