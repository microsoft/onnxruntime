# Frequently Asked Questions and Troubleshooting

ONNX Runtime provides high performance with flexibility of hardware options. Along with this flexibility comes decisions for tuning and usage options. Section 1 covers performance-related questions, and section 2 cover other commonly asked questions and issues. Before filing a new issue, please check for [existing filed issues and answers](https://github.com/microsoft/onnxruntime/issues).

**Section 1: [FAQ](#faq)**

*Performance*
* [What type of GPU is supported?](#What-type-of-GPU-is-supported)
* [Which Execution Provider will give me the best performance?](#Which-Execution-Provider-will-give-me-the-best-performance)
* [Do the GPU builds support quantized models?](#Do-the-GPU-builds-support-quantized-models)

*API*
* [Does the Python API add any overhead to performance?](#Does-the-Python-API-add-any-overhead-to-performance)
* [How do I turn on verbose mode logging in Python?](#How-do-I-turn-on-verbose-mode-logging-in-Python)
* [How do I infer models that have multiple inputs and outputs using the C/C++ API?](#How-do-I-infer-models-that-have-multiple-inputs-and-outputs-using-the-CC-API)
* [How do I force single threaded execution mode in ORT?](#How-do-I-force-single-threaded-execution-mode-in-ORT)

**Section 2: [Troubleshooting](#troubleshooting)**
* [Performance Troubleshooting Checklist](#performance-troubleshooting-checklist)
* [Why is my model running slower on GPU than CPU?](#Why-is-my-model-running-slower-on-GPU-than-CPU)
* [My converted Tensorflow model is slow - why?](#My-converted-Tensorflow-model-is-slow---why)
* [My PyTorch model is running slower on GPU - why?](#My-PyTorch-model-is-running-slower-on-GPU---why)
* [I can't build the xyz execution provider.](#I-cant-build-the-xyz-execution-provider)

***
# FAQ 

## What type of GPU is supported? 
The default ONNX Runtime GPU execution provider is for NVIDIA GPUs with CUDA. This build is published on Nuget: [Microsoft.ML.Onnxruntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu) and PyPi: [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu). To build from source, see [BUILD.md](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#cuda). 

Additionally, for NVIDIA GPUs you can build the [TensorRT execution provider](./execution_providers/TensorRT-ExecutionProvider.md) following [these instructions from BUILD.md](./../BUILD.md#tensorrt)

If your deployment target is Windows, we recommend using the [DirectML execution provider](./execution_providers/DirectML-ExecutionProvider.md) for optimal performance and compatibility with a broad set of GPUs.  See the build instructions [here](./../BUILD.md#directml).

AMD GPU support is in progress.

## Which Execution Provider will give me the best performance? 
Performance is dependent on the specific model you're trying to run, the session and run options you've selected, and of course, your specific hardware target. Please refer to the [Performance Tuning](./ONNX_Runtime_Perf_Tuning.md) page for guidance on how to performance tune ONNX Runtime for your model. You can use the [ONNX Go Live - OLive](https://github.com/microsoft/OLive) tool to help test and identify the best configuration and execution provider to use.  Below you'll find some more information that may be helpful to select the right Execution Provider.

### CUDA (Default GPU) vs CPU?
The CPU version of ONNX Runtime provides a complete implementation of all operators in the ONNX spec. This ensures that your ONNX-compliant model can execute successfully (*minor caveat: not 100% of types are supported to minimize the binary size, but can usually be easily added as needed*). On the other hand, not all CUDA kernels are implemented, as these have been prioritized on an as-needed basis. As a result, this means that if your model contains operators that do not have a CUDA implementation, it will fall back to CPU. Switching between CPU and GPU can cause significant performance impact.

If you require a specific operator that is not currently supported, please consider [contributing](./../CONTRIBUTING.md) and/or [file an issue](https://github.com/microsoft/onnxruntime/issues) clearly describing your use case and share your model if possible. 

### TensorRT vs CUDA?
TensorRT and CUDA are separate execution providers for ONNX Runtime. On the same hardware, TensorRT will generally provide better performance; however, this depends on the specific model and whether the operators in the model can be supported by TensorRT. In cases where TensorRT cannot handle the subgraph(s), it will fall back to CUDA. Note that the TensorRT EP may depend on a different version of CUDA than the CUDA EP. 

### TensorRT/CUDA vs DirectML? 
DirectML is the hardware-accelerated DirectX 12 library for machine learning on Windows and supports all DirectX 12 capable devices (Nvidia, Intel, AMD). This means that if you are targeting Windows GPUs, using the DirectML Execution Provider is likely your best bet. This can be used with both the ONNX Runtime as well as [WinML APIs](./WinRT_API.md).

## Do the GPU builds support quantized models?
The default CUDA build does not support any quantized operators right now. The TensorRT EP has limited support for INT8 quantized ops. In general, support of quantized models through ORT is continuing to expand on a model-driven basis. For performance improvements, quantization is not always required, and we suggest trying alternative strategies to [performance tune](./ONNX_Runtime_Perf_Tuning.md) before determining that quantization is necessary.

## Does the Python API add any overhead to performance?
The Python API doesn't add any overhead to performance when using the CPU execution provider only. When using the GPU provider, inputs and outputs need to be copied from CPU to GPU and vice-versa and the Python API doesn't allow inputs/outputs to be setup for this to be done prior to execution (calling run). Hence this copy needs to be done on the fly as part of execution. This eats into the execution time thereby impacting performance.

## How do I turn on verbose mode logging in Python?
```
import onnxruntime as ort
ort.set_default_logger_severity(0)
```

## How do I infer models that have multiple inputs and outputs using the C/C++ API?
See [this example](./../onnxruntime/test/shared_lib/test_inference.cc#L395)

See: 
* [Issue 1323](https://github.com/microsoft/onnxruntime/issues/1323)
* [Issue 2923](https://github.com/microsoft/onnxruntime/issues/2923)
* [Issue 3299](https://github.com/microsoft/onnxruntime/issues/3299)
* [Issue 2250](https://github.com/microsoft/onnxruntime/issues/2250)
* [Issue 3184](https://github.com/microsoft/onnxruntime/issues/3184)

## How do I force single threaded execution mode in ORT?
Please do both of the following:
* Build with openmp disabled or set OMP_NUM_THREADS=1.
* Set the session options intra_op_num_threads and inter_op_num_threads to 1 each.

***

# Troubleshooting
Please note: The answers below are basic troubleshooting suggestions based on common previous user-filed issues and questions. This list is by no means exhaustive and there is a lot of case-by-case fluctuation depending on the model and specific usage scenario. Please use this information to guide your troubleshooting, search through previously filed issues for related topics, and/or file a new issue if your problem is still not resolved.

## Performance Troubleshooting Checklist
Here is a list of things to check through when assessing performance issues.
* Have you read through the [Performance Tuning guide](./ONNX_Runtime_Perf_Tuning.md)?
* Are you using OpenMP? OpenMP will parallelize some of the code for potential performance improvements. This is not recommended for running on single threads.
* Have you enabled all [graph optimizations](./ONNX_Runtime_Graph_Optimizations.md)? The official published packages do enable all by default, but when building from source, check that these are enabled in your build.
* Have you searched through prior filed [Github issues](https://github.com/microsoft/onnxruntime/issues) to see if your problem has been discussed previously? Please do this before filing new issues.
* If using CUDA or TensorRT, do you have the right versions of the dependent libraries installed? 

## Why is my model running slower on GPU than CPU?
Depending on which execution provider you're using, it may not have full support for all the operators in your model. Fallback to CPU ops can cause hits in performance speed. Moreover even if an op is implemented by the CUDA execution provider, it may not necessarily assign/place the op to the CUDA EP due to performance reasons. To see the placement decided by ORT, turn on verbose logging and look at the console output. *See: [How do I turn on verbose mode logging in Python?](#How-do-I-turn-on-verbose-mode-logging-in-Python)*

## My converted Tensorflow model is slow - why?
NCHW and NHWC are two different memory layout for 4-D tensors.

Most TensorFlow operations used by a CNN support both NHWC and NCHW data format. The Tensorflow team suggests that on GPU NCHW is faster but on CPU NHWC is sometimes faster in Tensorflow. However, ONNX only supports NCHW. As a result, if the original model is in NHWC format, when the model is converted extra transposes may be added. The [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) and [keras-onnx](https://github.com/onnx/keras-onnx) converters do remove many of these transposes, but if this doesn't help sufficiently, consider retraining the model using NCHW.

## My PyTorch model is running slower on GPU - why?
Refer to [Does the Python API add any overhead to performance?](#Does-the-Python-API-add-any-overhead-to-performance). 
We are working on supporting IOBinding in the Python API that allows copying of inputs to the GPU prior to calling Run.

## I can't build the xyz execution provider.
Please read through the [BUILD.md](./../BUILD.md) page and makes sure your environment set up and dependency versions are all correct.

