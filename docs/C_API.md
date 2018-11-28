# C API

# Q: Why have a C API? 
Q: Why not just live in a C++ world? Why C?    
A: We want to distribute the onnxruntime as a DLL, which can be used in .Net languages through [P/Invoke](https://docs.microsoft.com/en-us/cpp/dotnet/how-to-call-native-dlls-from-managed-code-using-pinvoke).
This is the only option we have.

Q: Is it only for .Net?    
A: No. It is designed for:
1. Creating language bindings for the onnxruntime. e.g. C#, python, java, ...
2. Dynamic linking has some benefits. For example, solving diamond dependency problems.

Q: Can I export C++ types and functions across DLL or "Shared Object" Library(.so) boundaries?    
A: Well, you can, but it's not a good practice. We won't do it in this project.


## What's inside
* Creating an InferenceSession from an on-disk model file and a set of SessionOptions.
* Registering customized loggers.
* Registering customized allocators.
* Registering predefined providers and set the priority order. ONNXRuntime has a set of predefined execution providers,like CUDA, MKLDNN. User can register providers to their InferenceSession. The order of registration indicates the preference order as well.
* Running a model with inputs. These inputs must be in CPU memory, not GPU. If the model has multiple outputs, user can specify which outputs they want.
* Converting an in-memory ONNX Tensor encoded in protobuf format, to a pointer that can be used as model input.
* Setting the thread pool size for each session.
* Dynamically loading custom ops.

## How to use it

1. Include [onnxruntime_c_api.h](/include/onnxruntime/core/session/onnxruntime_c_api.h).
2. Call ONNXRuntimeInitialize
3. Create Session: ONNXRuntimeCreateInferenceSession(env, model_uri, nullptr,...)
4. Create Tensor
   1) ONNXRuntimeCreateAllocatorInfo
   2) ONNXRuntimeCreateTensorWithDataAsONNXValue
5. ONNXRuntimeRunInference


