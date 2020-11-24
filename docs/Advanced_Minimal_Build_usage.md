# ONNX Runtime Mobile with NNAPI - Advanced Minimal Build

The ONNX Runtime Mobile feature can be used with platform specific execution libraries like NNAPI (for Android) and CoreML (for iOS). This is achieved by making use of the [Execution Provider](https://github.com/microsoft/onnxruntime/tree/master/docs/execution_providers/README.md) (EP) interface to integrate with the repspective libraries. The ORT formatted model used with ONNX Runtime Mobile for execution through the EP interface requires limiting the optimization level to _basic_. It is possible that some _extended_ optimizations may be applicable to your model. In order to create a ORT formated model file where those are applied, please follow these steps.

## ONNX Runtime initialization
When ONNX Runtime initializes (`session.load()`) a model there are three main steps that affect the operators used in the model, and which execution provider runs each node.

* Step 1: The _basic_ optimizations are run (if enabled). See the [graph optimization](ONNX_Runtime_Graph_Optimizations.md) documentation for further details. Only ONNX operators are used when modifying the model in this step.

* Step 2: The enabled EPs are queried to confirm the specific nodes (i.e. ONNX ops) that can be executed on their HW. These nodes (or a sub-graph for a group of nodes) are assigned for execution by the EP based on the priority order set by the application. Some EPs can compile the allocated sub-graph to replace with a single function. This function is a compiled version of the original node/s (sub-graph). The function will be called at runtime to execute that part of the model on the target compute node.

* Step 3: The _extended_ and _layout_ optimizations, if enabled, are run to replace standard ONNX operators with custom internal ONNX Runtime operators that are optimized for specific HW targets. Any function based nodes will not be changed during this step.

These _basic_, _extended_ and _layout_ optimizations are not run on an ORT format models when using ONNX Runtime Mobile. These node level optimizations must be performed when creating the ORT model duing the offline step using the `convert_onnx_models_to_ort.py`.

We preserve all the nodes after _basic_ optimizations (i.e. nodes using standard ONNX operators only) so that at an EP could compile sub-graphs into a function. There may be nodes that the compiling execution provider would not use which can be replaced by the _extended_ and _layout_ optimizations. However these optimizations, and any resulting performance gains, are model dependent.

## Options for ORT Model

There are different optimization options available when creating the ORT format model for execution with ONNX Runtime Mobile.

* __Option 1:__ Use the existing ONNX Runtime python package to create the ORT model with the _basic_ optimizations. This will ensure that the EP will have the flexibility to handle the maximum number of nodes possible. But some _extended_ and _layout_ optimizations will not be possible with this option. Follow [these instructions](ONNX_Runtime_for_Mobile_Platforms.md#Enabling-Execution-Providers-that-compile-kernels-in-a-minimal-build) to create the ORT model to allow the EP optimizations at runtime.

* __Option 2:__ Use the ONNX Runtime + EP package to generate the ORT Model. Build a 'full' (i.e. no usage of the `--minimal_build` flag) version of ONNX Runtime from source in order to create a python package with the specific execution provider. This python package can be used to create an ORT format model that preserves the nodes that the execution provider can handle, whilst allowing _extended_ and _layout_ optimizations to run on the remaining nodes. __Note__ that this ORT model is then usable with the ONNX Runtime Mobile build with the specific EP only.

## Create NNAPI aware ORT Model

Follow the below instructions to create an NNAPI aware ORT format model. After doing so, follow the instructions to [create a minimal build with NNAPI support](ONNX_Runtime_for_Mobile_Platforms.md#Create-a-minimal-build-with-NNAPI-support).

1. Create a 'full' build of ONNX Runtime with the NNAPI EP enabled by adding `--use_nnapi` flag in the [building ONNX Runtime from source](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#start-baseline-cpu). You can create this build on any platform, i.e. not needed to create an Android build of ONNX Runtime in order to create an ORT format model that is optimized for usage with NNAPI. Add `--use_nnapi --build_shared_lib --build_wheel` to the build flags if any of those are missing. Do NOT add the `--minimal_build` flag.

```
.\build.bat --config RelWithDebInfo --use_nnapi --build_shared_lib --build_wheel --parallel
```

For Linux build replace `.\build.bat` with `./build.sh` for Linux

**NOTE:** 
    - This step is require prior to creating the minimal build for ONNX Runtime Mobile. The process for creating a minimal build will exclude operators that may be needed to load the ONNX model and create the ORT format model
    - If you have previously done a minimal build, run `git reset --hard` to make sure any operator kernel exclusions are reversed
    - When the NNAPI execution provider is enabled on non-Android platforms it can only specify which nodes can be assigned to NNAPI. __DO NOT__ use this build to execute the ORT model.

Install the python wheel from the build output directory. This is located in `build/Windows/<config>/<config>/dist/<package name>.whl` on Windows, or `build/Linux/<config>/dist/<package name>.whl` on Linux.
    `<config>` is the value from the `--config` parameter in the build command (e.g. RelWithDebInfo)
    The package name will differ based on your platform, python version, and build parameters
    For e.g. `pip install -U build\Windows\Release\Release\dist\onnxruntime_noopenmp-1.5.2-cp37-cp37m-win_amd64.whl`

2. Create an NNAPI aware ORT format model by running `tools\python\convert_onnx_models_to_ort.py` as per the above instructions, with the addition of the `--use_nnapi` parameter. This is using the python package from your 'full' build with NNAPI ep from the step #1. This will preserve all the nodes that can be assigned to NNAPI, as well as setup the ability to fallback to CPU execution if NNAPI is not available at runtime, or if NNAPI can not run all the nodes due to device limitations.

```
python <ONNX Runtime repository root>\tools\python\convert_onnx_models_to_ort.py --use_nnapi <path to directory containing one or more .onnx models>
```

## Performance caveats when using with NNAPI Execution Providers

The best optimization settings will differ by model. Testing the model for optimal performance is highly reccomended. At a minimum, compare the performance between the NNAPI-aware ORT model and the standard ORT model which is created for execution on CPU.

- If the model is split into multiple sub-graphs for execution across NNAPI and the CPU, then the overhead of switching between NNAPI and CPU execution may outweight the benefit of using NNAPI.
- The nodes allocated for NNAPI execution with not get enhanced with any _extended_ and _layout_ optimizations. These are optimizations that involve custom non-ONNX operators e.g. custom ONNX Runtime FusedConv operator that combines a Conv node and activation node (e.g. Relu). As NNAPI can handle Conv and Relu we would leave the original nodes as-is in the NNAPI aware ORT format model so that the NNAPI execution provider can take them at runtime.
- Depending on the model, there may be some performance loss if the NNAPI is not available at runtime (e.g. running on a non-Android platform), or does not support the same set of nodes at runtime (e.g. older version of NNAPI does not support as many operators). Generate an NNAPI-aware ORT format model for use on Android devices, and a standard ORT format model for use on other platforms so that there is flexibility to deliver best performance across different platforms.
- We will add the ability to capture information about the potential extended optimizations so that they may be applied at runtime in a minimal build if the compiling execution provider is not available, or does not end up taking the full set of nodes it originally claimed.