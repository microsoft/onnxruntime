# Advanced Minimal Build Usage

Creating an ORT format model that supports compiling execution providers such as NNAPI with the default ONNX Runtime python package requires limiting the optimization level to 'basic'. It is possible that some 'extended' optimizations could apply to your model. In order to create a model where those are applied, please follow these steps.

## Background
When ONNX Runtime loads an ONNX format model there are three main steps that affect the operators used in the model, and which execution provider runs each node. 

Step 1: The 'basic' optimizations are run (if enabled). See the [graph optimization](ONNX_Runtime_Graph_Optimizations.md) documentation for further details. Only ONNX operators are used when modifying the model in this step. 

Step 2: The enabled [execution providers](https://github.com/microsoft/onnxruntime/tree/master/docs/execution_providers) are asked which nodes they can handle. Nodes are assigned based on the priority order of the execution providers. A compiling execution provider will replace one or more nodes at a time with a single 'function' based node (this is a 'compiled' version of the original node/s) when it is assigned those nodes. The function will be called at runtime to execute that part of the model.

Step 3: The 'extended' and 'layout' optimizations are run (if enabled). Custom internal ONNX Runtime operators are used in these optimizations, and the optimizations will only replace nodes that were using standard ONNX operators. Due to the latter, 'function' based nodes will not be changed during this step.

Optimizations are not run on an ORT format model (at runtime only step 2 will occur), so any optimizations must be performed when creating it. Assuming we want a compiling execution provider to take as many nodes as possible, we want to preserve all the nodes it would see after 'basic' optimizations are done (i.e. nodes using ONNX operators only), so that at runtime it can compile those into 'function' based nodes. There may be nodes that the compiling execution provider does not take that the higher level optimizations can replace, however this is model dependent, as is any potential performance gain from such optimizations.

## Model creation choice

Given this background, a choice can be made as to how the ORT format model is created. 

The simple approach is to use the released ONNX Runtime python package to create the model with the optimization level limited to 'basic'. This will ensure that the compiling execution provider will handle the maximum number of nodes possible, at the potential loss of some higher level optimizations. Please follow [these instructions](ONNX_Runtime_for_Mobile_Platforms.md#Enabling-Execution-Providers-that-compile-kernels-in-a-minimal-build).

The advanced approach is to build a 'full' (i.e. no usage of the `--minimal_build` flag) version of ONNX Runtime from source in order to create a python package with the compiling execution provider enabled. This python package can be used to create an ORT format model that preserves the nodes the compiling execution provider can potentially handle, whilst allowing higher level optimizations to run on any remaining nodes.

Follow the below instructions to create an NNAPI aware ORT format model. After doing so, follow the instructions to [create a minimal build with NNAPI support](ONNX_Runtime_for_Mobile_Platforms.md#Create-a-minimal-build-with-NNAPI-support).

## Create NNAPI aware ORT format model
  - Create a 'full' build of ONNX Runtime with NNAPI enabled by [building ONNX Runtime from source](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#start-baseline-cpu).
    - **NOTE** do this prior to creating the minimal build
      - the process for creating a minimal build will exclude operators that may be needed to load the ONNX model and create the ORT format model
      - if you have previously done a minimal build, run `git reset --hard` to make sure any operator kernel exclusions are reversed
    - we can not use the ONNX Runtime prebuilt package as NNAPI is not enabled in it
    - the 'full' build can be done on any platform
      - you do not need to create an Android build of ONNX Runtime in order to create an ORT format model that is optimized for usage with NNAPI.
        - when the NNAPI execution provider is enabled on non-Android platforms it can only specify which nodes can be assigned to NNAPI. it can NOT be used to execute the model.
    - Add `--use_nnapi --build_shared_lib --build_wheel` to the build flags if any of those are missing. Do NOT add the `--minimal_build` flag.
      - e.g. `.\build.bat --config RelWithDebInfo --use_nnapi --build_shared_lib --build_wheel --parallel` 
      - replace `.\build.bat` with `./build.sh` for Linux
  - Install the python wheel from the build output directory
    - this is located in `build/Windows/<config>/<config>/dist/<package name>.whl` on Windows, or `build/Linux/<config>/dist/<package name>.whl` on Linux. 
      - `<config>` is the value from the `--config` parameter from the build command (e.g. RelWithDebInfo)
      - the package name will differ based on your platform, python version, and build parameters
      - e.g. `pip install -U build\Windows\Release\Release\dist\onnxruntime_noopenmp-1.5.2-cp37-cp37m-win_amd64.whl`
  - Create an NNAPI aware ORT format model by running `tools\python\convert_onnx_models_to_ort.py` as per the above instructions, with the addition of the `--use_nnapi` parameter
    - the python package from your 'full' build with NNAPI enabled must be installed for `--use_nnapi` to be a valid option
    - this will preserve all the nodes that can be assigned to NNAPI, as well as setup the ability to fallback to CPU execution if NNAPI is not available at runtime, or if NNAPI can not run all the nodes due to device limitations.

## Performance caveats when using compiling Execution Providers

What is optimal will differ by model, and performance testing is the only way to determine what works best for your model. At a minimum it is suggested to performance test with the NNAPI aware ORT format model, and a standard ORT format model created using the default instructions.

  - If the sections of the model that NNAPI can handle are broken up, the overhead of switching between NNAPI and CPU execution between these sections may outweight the benefit of using NNAPI
  - Any potential extended optimizations on nodes that the NNAPI execution provider claims will not occur in order to preserve the nodes as-is
    - these are optimizations that involve custom non-ONNX operators 
      - e.g. custom ONNX Runtime FusedConv operator that combines a Conv node and activation node (e.g. Relu). As NNAPI can handle Conv and Relu we would leave the original nodes as-is in the NNAPI aware ORT format model so that the NNAPI execution provider can take them at runtime.
    - Depending on the model, and how many of these potential extended optimizations are not applied, there may be some performance loss if the NNAPI execution provider is not available at runtime (e.g. running on a non-Android platform), or does not claim the same set of nodes at runtime (e.g. older version of NNAPI does not support as many operators) 
      - you may want to generate an NNAPI aware ORT format model for use on Android devices, and a standard ORT format model for use on other platforms
    - in a future release we will add the ability to capture information about the potential extended optimizations so that they may be applied at runtime in a minimal build if the compiling execution provider is not available, or does not end up taking the full set of nodes it originally claimed.
