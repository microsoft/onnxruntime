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
      * See test ```TestSharedAllocatorUsingCreateAndRegisterAllocator``` in https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/shared_lib/test_inference.cc for an example.
      * Configuring *OrtArenaCfg* (use API ```CreateArenaCfgV2``` to create ```OrtArenaCfg``` instance from ORT release 1.8 onwards, prior releases used the now-deprecated ```CreateArenaCfg``` to create the instance):
         * ```CreateArenaCfgV2``` takes the following a list of keys as described below and takes a corresponding set of values (one for each key). Default values for these configs can be found in the [BFCArena class](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/framework/bfc_arena.h). See test ```ConfigureCudaArenaAndDemonstrateMemoryArenaShrinkage``` in https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/shared_lib/test_inference.cc for an example usage of ```CreateArenaCfgV2```.
         * ```max_mem```: This is the maximum amount of memory the arena allocates. If a chunk cannot be serviced by any existing region, the arena extends itself by allocating one more region depending on available memory (max_mem - allocated_so_far). An error is returned if available memory is less than the requested extension.
         * ```arena_extend_strategy```: This can take only 2 values currently: kSameAsRequested or kNextPowerOfTwo. As the name suggests kNextPowerOfTwo (the default) extends the arena by a power of 2, while kSameAsRequested extends by a size that is the same as the allocation request each time. kSameAsRequested is suited for more advanced configurations where you know the expected memory usage in advance.
         * ```initial_chunk_size_bytes```: This config is only relevant when the arena extend strategy is kNextPowerOfTwo. This is the (possible) size of the region that the arena allocates in the very first allocation (if the first memory request is greater than this the allocation size will be larger than this provided value). Chunks are handed over to allocation requests from this region. If the logs show that the arena is getting extended a lot more than expected, you're better off choosing a big enough initial size for this.
         * ```initial_growth_chunk_size_bytes```: Please read the section ```Memory arena shrinkage``` prior to reading this section. This config is only relevant when the arena extend strategy is kNextPowerOfTwo. Currently, this value is the (possible) size of the first allocation post an arena shrinkage (a memory request greater than this value post arena shrinkage will result in higher allocation size). The (possible) first allocation by an arena is defined by ```initial_chunk_size_bytes``` and the possible subsequent allocations are ```initial_chunk_size_bytes * 2```, ```initial_chunk_size_bytes * 4```, and so on. If the arena were to shrink (i.e.) de-allocate any of these memory regions, we want to "reset" the size of the first allocation post shrinkage. This is the current definition. In future, this config may be used to control other "growth-centric" actions of the arena (i.e.) the second allocation made by the arena (arena growth) post the very first allocation (initial chunk) may be defined by this parameter.
         * ```max_dead_bytes_per_chunk```: This controls whether a chunk is split to service an allocation request. Currently if the difference between the chunk size and requested size is less than this value, the chunk is not split. This has the potential to waste memory by keeping a part of the chunk unused (hence called dead bytes) throughout the process thereby increasing the memory usage (until this chunk is returned to the arena).
* **Memory arena shrinkage:**
   * *Description*: By default, memory arenas do not shrink (return unused memory back to the system). This feature allows users to "shrink" the arena at some cadence. Currently, the only supported cadence is at the end of every Run() (i.e.) if this feature is used, the arena memory is scanned to potentially free up unused memory at the end of every Run(). This is achieved through a RunOption.
   * *Scenario*: You have a dynamic shape model that may occasionally service a request that may require a lot of memory being allocated. Since, by default, the arena doesn't free any memory, this "growth" of the arena as part of servicing this request is held on forever. This is sub-optimal because most of the other requests probably do not need so much memory (i.e.) too much memory ends up being allocated just to process one or two outliers. If this best describes the ORT-usage scenario, using this shrinkage feature is an option. This feature is only applicable if the relevant memory allocator is an arena based allocator to begin with. 
   * *Usage*: See test ```ConfigureCudaArenaAndDemonstrateMemoryArenaShrinkage``` in https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/shared_lib/test_inference.cc for an example. To optimally use this feature, the memory arena that is going to be shrunk needs to be configured appropriately for the use-case. Please see how to configure an arena allocator using an ```OrtArenaCfg``` instance above. This feature is slightly flavored based on the two available arena extension strategies (kSameAsRequested and kNextPowerOfTwo) as described below:
      * ```kNextPowerOfTwo```: If this is the chosen configuration, all memory allocations except the initial allocation are considered for de-allocation at shrinkage. The idea is that the user sets a high enough ```initial_chunk_size_bytes``` to process most of model requests without allocating more memory (i.e.) this initial memory is good enough to service any average request. Any subsequent allocation(s) that is allocated as part of servicing an outlier request are the only candidates for de-allocation.
      * ```kSameAsRequested``` If this is the chosen configuration, all memory allocations are considered for de-allocation at shrinkage time. This is because, currently, ```initial_chunk_size_bytes``` is not relevant for this strategy.
* **Allocate memory for initializer(s) from non-arena memory (for advanced users):**
   * *Description*: If the allocator pertaining to the device that the initializers' contents are to be stored in is an arena based allocator and one would like to prevent any (potentially) excessive arena growth arising out of allocating memory for these initializers, this is the feature that provides such capability.
   * *Scenario*: You have a fairly simplistic model that doesn't require a lot of memory allocated during Run() itself, but the model has relatively a large number of initializers such that if memory for these were to be allocated using an arena, the unused memory in the overall arena allocated memory could far exceed what is actually needed for the model during Run(). Under such circumstances and if the model is to be deployed on a memory-constrained environment, using such a feature would make sense so as to prevent any excessive growth in the arena that would come as part of initializer memory allocation in the arena. Using this feature would mean that the arena is only used to allocate memory required by the model's operators only.
                 It would seem that setting a high enough initial chunk size for the arena (```initial_chunk_size_bytes```) to account for initializers and any anticipated memory required during Run() would avoid the problematic scenario. The problem with this is that some EPs internally use per-thread allocators and any high initial chunk size set in the configuration would apply to each of these and would result in memory wastage as only one per-thread allocator ultimately allocates the initializers' memory.
   * *Usage*: See test ```AllocateInitializersFromNonArenaMemory``` in https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/shared_lib/test_inference.cc for an example.
* **Share initializer(s) and their ORT pre-processed version(s) between sessions:**
   * *Description*: This feature allows a user to share the same instance of an initializer (and their ORT "pre-processed" versions)across multiple sessions.
   * *Scenario*: You've several models that use the same set of initializers except the last few layers of the model and you load these models in the same process. When every model (session) creates a separate instance of the same initializer, it leads to excessive and wasteful memory usage since in this case it's the same initializer. You want to optimize memory usage while having the flexibility to allocate the initializers (possibly even store them in shared memory). 
   * *Example Usage*: Use the ```AddInitializer``` API to add a pre-allocated initializer to session options before calling ```CreateSession```. Use the same instance of session options to create several sessions allowing the initializer(s) to be shared between the sessions. See [C API sample usage (TestSharingOfInitializerAndItsPrepackedVersion)](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/shared_lib/test_inference.cc) and [C# API sample usage (TestSharingOfInitializerAndItsPrepackedVersion)](https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.Tests/InferenceTest.cs).
   * *In some ORT operator implementations, initializers are pre-processed when the model is loaded (a process called "pre-packing") to promote optimal operator inferencing on some platforms. By default, these pre-processed versions of initializers are maintained on a per-session basis (i.e.) they are not shared between sessions. To enable sharing these between sessions, create a container (using ```CreatePrepackedWeightsContainer```) and pass this at session creation time so that the sharing of pre-packed versions of shared initializers between sessions take place and these are not duplicated in memory. The same tests referenced above in C and C# shows sample usage of this feature as well.
      NOTE: Any kernel developer wishing to implement pre-packing MUST write a test that triggers pre-packing of all weights that can be possibly pre-packed using the kernel and must test sharing of these pre-packed weights between sessions. See [kernel test (SharedPrepackedWeights)](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/providers/cpu/math/gemm_test.cc).
   

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
