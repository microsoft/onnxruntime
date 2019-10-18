# C API

## Features

* Creating an InferenceSession from an on-disk model file and a set of SessionOptions.
* Registering customized loggers.
* Registering customized allocators.
* Registering predefined providers and set the priority order. ONNXRuntime has a set of predefined execution providers, like CUDA, MKLDNN. User can register providers to their InferenceSession. The order of registration indicates the preference order as well.
* Running a model with inputs. These inputs must be in CPU memory, not GPU. If the model has multiple outputs, user can specify which outputs they want.
* Converting an in-memory ONNX Tensor encoded in protobuf format to a pointer that can be used as model input.
* Setting the thread pool size for each session.
* Setting graph optimization level for each session.
* Dynamically loading custom ops. [Instructions](/docs/AddingCustomOp.md)
* Ability to load a model from a byte array. See ```OrtCreateSessionFromArray``` in [onnxruntime_c_api.h](/include/onnxruntime/core/session/onnxruntime_c_api.h).

## Usage Overview

1. Include [onnxruntime_c_api.h](/include/onnxruntime/core/session/onnxruntime_c_api.h).
2. Call OrtCreateEnv
3. Create Session: OrtCreateSession(env, model_uri, nullptr,...)
   - Optionally add more execution providers (e.g. for CUDA use OrtSessionOptionsAppendExecutionProvider_CUDA)
4. Create Tensor
   1) OrtCreateMemoryInfo
   2) OrtCreateTensorWithDataAsOrtValue
5. OrtRun

## Sample code

The example below shows a sample run using the SqueezeNet model from ONNX model zoo, including dynamically reading model inputs, outputs, shape and type information, as well as running a sample vector and fetching the resulting class probabilities for inspection.

* [../csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp](../csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp)

# Telemetry
This project collects usage data and sends it to Microsoft to help improve our products and services. Note however that no data collection is performed by default when using your private builds

Telemetry is turned OFF by default while this feature is in BETA.   When the feature moves from BETA to RELEASE, developers should expect telemetry to be ON by default when using the Official Builds.  This is implemented via 'Platform Telemetry' per vendor platform providers (see telemetry.h).

The Windows provider uses the [TraceLogging](https://docs.microsoft.com/en-us/windows/win32/tracelogging/trace-logging-about) API for its implementation.

