# C API

## Features

* Creating an InferenceSession from an on-disk model file and a set of SessionOptions.
* Registering customized loggers.
* Registering customized allocators.
* Registering predefined providers and set the priority order. ONNXRuntime has a set of predefined execution providers, like CUDA, DNNL. User can register providers to their InferenceSession. The order of registration indicates the preference order as well.
* Running a model with inputs. These inputs must be in CPU memory, not GPU. If the model has multiple outputs, user can specify which outputs they want.
* Converting an in-memory ONNX Tensor encoded in protobuf format to a pointer that can be used as model input.
* Setting the thread pool size for each session.
* Setting graph optimization level for each session.
* Dynamically loading custom ops. [Instructions](/docs/AddingCustomOp.md)
* Ability to load a model from a byte array. See ```OrtCreateSessionFromArray``` in [onnxruntime_c_api.h](/include/onnxruntime/core/session/onnxruntime_c_api.h).
* **Global/shared threadpools:** By default each session creates its own set of threadpools. In situations where multiple
sessions need to be created (to infer different models) in the same process, you end up with several threadpools created
by each session. In order to address this inefficiency we introduce a new feature called global/shared threadpools.
The basic idea here is to share a set of global threadpools across multiple sessions. Typical usage of this feature
is as follows
   * Populate ```ThreadingOptions```. Use the value of 0 for ORT to pick the defaults.
   * Create env using ```CreateEnvWithGlobalThreadPools()```
   * Create session and call ```DisablePerSessionThreads()``` on the session options object
   * Call ```Run()``` as usual
* **Share allocator(s) between sessions:** Allow multiple sessions in the same process to use the same allocator(s). This
allocator is first registered in the env and then reused by all sessions that use the same env instance unless a session
chooses to override this by setting ```session_state.use_env_allocators``` to "0". Usage of this feature is as follows
   * Register an allocator created by ORT using the ```CreateAndRegisterAllocator``` API.
   * Set ```session.use_env_allocators``` to "1" for each session that wants to use the env registered allocators.
   * See test ```TestSharedAllocatorUsingCreateAndRegisterAllocator``` in
     onnxruntime/test/shared_lib/test_inference.cc for an example.
* **Share initializer(s) between sessions:**
   * *Description*: This feature allows a user to share the same instance of an initializer across
multiple sessions.
   * *Scenario*: You've several models that use the same set of initializers except the last few layers of the model and you load these models in the same process. When every model (session) creates a separate instance of the same initializer, it leads to excessive and wasteful memory usage since in this case it's the same initializer. You want to optimize memory usage while having the flexibility to allocate the initializers (possibly even store them in shared memory). 
   * *Example Usage*: Use the ```AddInitializer``` API to add a pre-allocated initializer to session options before calling ```CreateSession```. Use the same instance of session options to create several sessions allowing the initializer(s) to be shared between the sessions. See [C API sample usage (TestSharingOfInitializer)](../onnxruntime/test/shared_lib/test_inference.cc) and [C# API sample usage (TestWeightSharingBetweenSessions)](../csharp/test/Microsoft.ML.OnnxRuntime.Tests/InferenceTest.cs).

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

## Deployment

### Windows 10

Your installer should put the onnxruntime.dll into the same folder as your application.   Your application can either use [load-time dynamic linking](https://docs.microsoft.com/en-us/windows/win32/dlls/using-load-time-dynamic-linking) or [run-time dynamic linking](https://docs.microsoft.com/en-us/windows/win32/dlls/using-run-time-dynamic-linking) to bind to the dll.

#### Dynamic Link Library Search Order

This is an important article on how Windows finds supporting dlls: [Dynamic Link Library Search Order](https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order).

There are some cases where the app is not directly consuming the onnxruntime but instead calling into a DLL that is consuming the onnxruntime.    People building these DLLs that consume the onnxruntime need to take care about folder structures.  Do not modify the system %path% variable to add your folders.  This can conflict with other software on the machine that is also using the onnxruntme.  Instead place your DLL and the onnxruntime DLL in the same folder and use [run-time dynamic linking](https://docs.microsoft.com/en-us/windows/win32/dlls/using-run-time-dynamic-linking) to bind explicity to that copy.  You can use code like this sample does in [GetModulePath()](https://github.com/microsoft/Windows-Machine-Learning/blob/master/Samples/SampleSharedLib/SampleSharedLib/FileHelper.cpp) to find out what folder your dll is loaded from.

## Telemetry

To turn on/off telemetry collection on official Windows builds, please use Enable/DisableTelemetryEvents() in the C API. See the [Privacy](./Privacy.md) page for more information on telemetry collection and Microsoft's privacy policy.
