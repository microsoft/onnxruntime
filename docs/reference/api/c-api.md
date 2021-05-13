---
title: C API
parent: API docs
grand_parent: Reference
nav_order: 1
---

# ONNX Runtime C API
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Builds

| Artifact  | Description | Supported Platforms |
|-----------|-------------|---------------------|
| [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) | CPU (Release) |Windows, Linux,  Mac, X64, X86 (Windows-only), ARM64 (Windows-only)...more details: [compatibility](../../resources/compatibility.md) |
| [Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu) | GPU - CUDA (Release) | Windows, Linux, Mac, X64...more details: [compatibility](../../resources/compatibility.md) |
| [Microsoft.ML.OnnxRuntime.DirectML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.directml) | GPU - DirectML (Release) | Windows 10 1709+ |
| [ort-nightly](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly) | CPU, GPU (Dev) | Same as Release versions |


.zip and .tgz files are also included as assets in each [Github release](https://github.com/microsoft/onnxruntime/releases).

## API Reference
Refer to [onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_c_api.h)

1. Include [onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_c_api.h).
2. Call OrtCreateEnv
3. Create Session: OrtCreateSession(env, model_uri, nullptr,...)
   - Optionally add more execution providers (e.g. for CUDA use OrtSessionOptionsAppendExecutionProvider_CUDA)
4. Create Tensor
   1) OrtCreateMemoryInfo
   2) OrtCreateTensorWithDataAsOrtValue
5. OrtRun

## Features

* Creating an InferenceSession from an on-disk model file and a set of SessionOptions.
* Registering customized loggers.
* Registering customized allocators.
* Registering predefined providers and set the priority order. ONNXRuntime has a set of predefined execution providers, like CUDA, DNNL. User can register providers to their InferenceSession. The order of registration indicates the preference order as well.
* Running a model with inputs. These inputs must be in CPU memory, not GPU. If the model has multiple outputs, user can specify which outputs they want.
* Converting an in-memory ONNX Tensor encoded in protobuf format to a pointer that can be used as model input.
* Setting the thread pool size for each session.
* Setting graph optimization level for each session.
* Dynamically loading custom ops. [Instructions](../../how-to/add-custom-op.md)
* Ability to load a model from a byte array. See ```OrtCreateSessionFromArray``` in [onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_c_api.h).
* **Global/shared threadpools:** By default each session creates its own set of threadpools. In situations where multiple
sessions need to be created (to infer different models) in the same process, you end up with several threadpools created
by each session. In order to address this inefficiency we introduce a new feature called global/shared threadpools.
The basic idea here is to share a set of global threadpools across multiple sessions. Typical usage of this feature
is as follows
   * Populate ```ThreadingOptions```. Use the value of 0 for ORT to pick the defaults.
   * Create env using ```CreateEnvWithGlobalThreadPools()```
   * Create session and call ```DisablePerSessionThreads()``` on the session options object
   * Call ```Run()``` as usual
* **Share allocator(s) between sessions:**
   * *Description*: This feature allows multiple sessions in the same process to use the same allocator(s).
   * *Scenario*: You've several sessions in the same process and see high memory usage. One of the reasons for this is as follows. Each session creates its own CPU allocator which is arena based by default. [ORT implements](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/framework/bfc_arena.h) a simplified version of an arena allocator that is based on [Doug Lea's best-first with coalescing algorithm](http://gee.cs.oswego.edu/dl/html/malloc.html). Each allocator lives in its own session. It allocates a large region of memory during init time and thereafter it chunks, coalesces and extends this initial region as per allocation/deallocation demands. Overtime the arena ends up with unused chunks of memory per session. Moreover, the memory allocated by the arena is never returned to the system; once allocated it always remains allocated. All these factors add up when using multiple sessions (each with its own arena) thereby increasing the overall memory consumption of the process. Hence it becomes important to share the arena allocator between sessions.
   * *Usage*:
      * Create and register a shared allocator with the env using the ```CreateAndRegisterAllocator``` API. This allocator is then reused by all sessions that use the same env instance unless a session
chooses to override this by setting ```session_state.use_env_allocators``` to "0".
      * Set ```session.use_env_allocators``` to "1" for each session that wants to use the env registered allocators.
      * See test ```TestSharedAllocatorUsingCreateAndRegisterAllocator``` in
     https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/shared_lib/test_inference.cc for an example.
      * Configuring *OrtArenaCfg*:
         * Default values for these configs can be found in the [BFCArena class](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/framework/bfc_arena.h).
         * ```initial_chunk_size_bytes```: This is the size of the region that the arena allocates first. Chunks are handed over to allocation requests from this region. If the logs show that the arena is getting extended a lot more than expected, you're better off choosing a big enough initial size for this.
         * ```max_mem```: This is the maximum amount of memory the arena allocates. If a chunk cannot be serviced by any existing region, the arena extends itself by allocating one more region depending on available memory (max_mem - allocated_so_far). An error is returned if available memory is less than the requested extension.
         * ```arena_extend_strategy```: This can take only 2 values currently: kSameAsRequested or kNextPowerOfTwo. As the name suggests kNextPowerOfTwo (the default) extends the arena by a power of 2, while kSameAsRequested extends by a size that is the same as the allocation request each time. kSameAsRequested is suited for more advanced configurations where you know the expected memory usage in advance.
         * ```max_dead_bytes_per_chunk```: This controls whether a chunk is split to service an allocation request. Currently if the difference between the chunk size and requested size is less than this value, the chunk is not split. This has the potential to waste memory by keeping a part of the chunk unused (hence called dead bytes) throughout the process thereby increasing the memory usage (until this chunk is returned to the arena).
* **Share initializer(s) between sessions:**
   * *Description*: This feature allows a user to share the same instance of an initializer across
+multiple sessions.
   * *Scenario*: You've several models that use the same set of initializers except the last few layers of the model and you load these models in the same process. When every model (session) creates a separate instance of the same initializer, it leads to excessive and wasteful memory usage since in this case it's the same initializer. You want to optimize memory usage while having the flexibility to allocate the initializers (possibly even store them in shared memory). 
   * *Example Usage*: Use the ```AddInitializer``` API to add a pre-allocated initializer to session options before calling ```CreateSession```. Use the same instance of session options to create several sessions allowing the initializer(s) to be shared between the sessions. See [C API sample usage (TestSharingOfInitializer)](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/shared_lib/test_inference.cc) and [C# API sample usage (TestWeightSharingBetweenSessions)](https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.Tests/InferenceTest.cs).


## Deployment

### Windows 10

Your installer should put the onnxruntime.dll into the same folder as your application.   Your application can either use [load-time dynamic linking](https://docs.microsoft.com/en-us/windows/win32/dlls/using-load-time-dynamic-linking) or [run-time dynamic linking](https://docs.microsoft.com/en-us/windows/win32/dlls/using-run-time-dynamic-linking) to bind to the dll.

#### Dynamic Link Library Search Order

This is an important article on how Windows finds supporting dlls: [Dynamic Link Library Search Order](https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order).

There are some cases where the app is not directly consuming the onnxruntime but instead calling into a DLL that is consuming the onnxruntime. People building these DLLs that consume the onnxruntime need to take care about folder structures.  Do not modify the system %path% variable to add your folders.  This can conflict with other software on the machine that is also using the onnxruntme.  Instead place your DLL and the onnxruntime DLL in the same folder and use [run-time dynamic linking](https://docs.microsoft.com/en-us/windows/win32/dlls/using-run-time-dynamic-linking) to bind explicitly to that copy.  You can use code like this sample does in [GetModulePath()](https://github.com/microsoft/Windows-Machine-Learning/blob/master/Samples/SampleSharedLib/SampleSharedLib/FileHelper.cpp) to find out what folder your dll is loaded from.

## Telemetry

To turn on/off telemetry collection on official Windows builds, please use Enable/DisableTelemetryEvents() in the C API. See the [Privacy](https://github.com/microsoft/onnxruntime/blob/master/docs/Privacy.md) page for more information on telemetry collection and Microsoft's privacy policy.

## Samples

See [Tutorials: API Basics - C](../../tutorials/inferencing/api-basics.md#c-1)