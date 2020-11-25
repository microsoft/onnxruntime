# ONNX Runtime Mobile Build: Advanced Usage with NNAPI

The ONNX Runtime Mobile with the NNAPI execution provider can be used to execute ORT format models on Android platforms. This document explains the different options available to optimize the ORT model and building the ONNX Runtime Mobile package for exection with NNAPI.

## Effect of Optimization Levels on NNAPI Execution Provider

_The ORT formatted model, for execution through the NNAPI EP, should be limited to __basic__ optimization level. It is possible that some _extended_ optimizations may be applicable to your model. In order to create a ORT formated model file where those are applied, please follow these steps._

### Background about optimization for ORT Models

ONNX Runtime applies optimizations on the ONNX model to improve inference performance. There are three main steps that affect the operators used during the optimization, and the execution provider (EP) targeted to execute the respective nodes.

*Step 1:*

The _basic_ optimizations are applied when the model is initialized by ONNX Runtime. See the [graph optimization](ONNX_Runtime_Graph_Optimizations.md) documentation for further details. Only ONNX operators are used when modifying the model in this step.

*Step 2:*

The enabled [execution providers](execution_providers/README.md) are queried about nodes in the ONNX model they can execute. Each node is assigned to a specific EP based on the user defined priority set during the session initialization step.

The NNAPI EP identifies nodes that can be executed using NNAPI. ONNX Runtime may group nodes into a sub-graph to be executed by NNAPI to minimize the overhead to copy data between the CPU and NNAPI. The inference performace is better with more nodes in each sub-graph, and fewer sub-graphs.

*Step 3:* 

The 'extended' and 'layout' optimizations are run, if enabled. These optimizations will replace one or more standard ONNX operators with custom internal ONNX Runtime operators designed to boost performance on certain HW platforms. These optimizations are applied to the nodes assinged to the correcponding EP for the target HW platform. The custom internal operators are typically limited to the default CPU EP on x86_64 / aarch64 atchitecture and the CUDA EP on NVIDIA GPUs.

These optimizations will not be applicable to the nodes assigned to the NNAPI EP.

Below is an example of the changes that occur in 'basic' and 'extended' optimizations when applied to the MNIST model.

- At the 'basic' level we combine the Conv and Add nodes (the addition is done via the 'B' input to Conv), we combine the MatMul and Add into a single Gemm node (the addition is done via the 'C' input to Gemm), and constant fold to remove one of the Reshape nodes. 
- At the 'extended' level we additionally fuse the Conv and Relu nodes using the internal ONNX Runtime FusedConv operator.

<img align="center" src="images/mnist_optimization.png" alt="Changes to nodes from basic and extended optimizations."/>

At runtime, for each group of nodes that is assigned to it, the NNAPI EP will create an NNAPI model that replicates the processing done by the nodes in the group. It will create a function to handle executing this NNAPI model, including copying data between the CPU and NNAPI. The original nodes in the model are replaced with a single node containing this function.

### Considerations when creating an NNAPI-aware ORT format model

There are no optimizations available at runtime for an ORT format model, so we generally want to optimize to the highest level possible when creating the model. That usually means optimizing to the 'extended' level, as 'layout' level optimizations are not recommended for use with NNAPI (they are device specific).

If the NNAPI EP is enabled when loading the ORT format model we will run Step 2 to allow it to replace the groups of nodes it can handle with a function that executes the NNAPI model for those nodes. As the NNAPI model is created at runtime using device specific information it cannot be saved as part of the ORT format model. *NOTE:* every node in the saved ORT format model can be run using a statically registered kernel, so any node that is not taken by the NNAPI EP will be run this way.

As the extended optimizations will replace nodes using ONNX operators with nodes using custom internal ONNX Runtime operators, the optimizations can potentially reduce the number of nodes that NNAPI would be able to run.

Below is an example using the MNIST model comparing what would happen at runtime if the ORT format model was created with 'basic' and 'extended' optimizations applied and the NNAPI EP was available. As the 'basic' level optimizations result in a model that only uses ONNX operators, the NNAPI EP is able to handle the majority of the model in a single function as NNAPI can execute all the Conv, Relu and MaxPool nodes at once. The 'extended' level optimizations however introduced the custom FusedConv nodes, resulting in two functions using NNAPI, each handling a single MaxPool node. Most likely this will have worse performance if run with the NNAPI EP enabled due to the data copies between CPU and NNAPI for the MaxPool calculations.

<img align="center" src="images/mnist_optimization_with_nnapi.png" alt="Changes to nodes by NNAPI depending on optimization level of input.">

If your model benefits signficantly from 'extended' optimizations, you may wish to create an ORT format model optimized to 'basic' level for use with NNAPI (see instructions [here](ONNX_Runtime_for_Mobile_Platforms.md#Using-NNAPI-with-ONNX-Runtime-Mobile)), and an ORT format model optimized to 'extended' level for use everywhere else.

## Create an NNAPI-aware ORT format model

If your model might benefit from extended optimizations being applied to nodes that are not handled by NNAPI, it is possible to create an ORT format model that only applies extended optimizations to nodes that NNAPI can not handle. 

1. You will need to create a 'full' build of ONNX Runtime with NNAPI enabled by [building ONNX Runtime from source](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#start-baseline-cpu). This build can be created on any platform and does NOT need to be an Android build. When building add `--use_nnapi --build_shared_lib --build_wheel` to the build flags if any of those are missing. Do NOT add the `--minimal_build` flag.
  - e.g.
    - Windows: `<ONNX Runtime repository root>\build.bat --config RelWithDebInfo --use_nnapi --build_shared_lib --build_wheel --parallel` 
    - Linux: `<ONNX Runtime repository root>/build.sh --config RelWithDebInfo --use_nnapi --build_shared_lib --build_wheel --parallel` 
  - **NOTE** if you have previously done a minimal build you will need to run `git reset --hard` to make sure any operator kernel exclusions are reversed prior to performing the 'full' build. If you do not, you may not be able to load the ONNX format model due to missing kernels.

2. Install the python wheel from the build output directory. This is located in `build/Windows/<config>/<config>/dist/<package name>.whl` on Windows, or `build/Linux/<config>/dist/<package name>.whl` on Linux. The package name will differ based on your platform, python version, and build parameters. `<config>` is the value from the `--config` parameter from the build command.
  - e.g. `pip install -U build\Windows\RelWithDebIfo\RelWithDebIfo\dist\onnxruntime_noopenmp-1.5.2-cp37-cp37m-win_amd64.whl`

3. Create an NNAPI-aware ORT format model by running `convert_onnx_models_to_ort.py` as per the [standard instructions](ONNX_Runtime_for_Mobile_Platforms.md#Create-ORT-format-model-and-configuration-file-with-required-operators), with NNAPI enabled (`--use_nnapi`), and the optimization level set to 'extended' (`--optimization_level extended`). This will allow extended level optimizations to run on any nodes that NNAPI can not handle.
  - e.g. `python <ORT repository root>/tools/python/convert_onnx_models_to_ort.py --use_nnapi --optimization_level extended /models`
  - the python package from your 'full' build with NNAPI enabled must be installed for `--use_nnapi` to be a valid option

You will also need to [create an minimal build for Android with the NNAPI EP enabled](ONNX_Runtime_for_Mobile_Platforms.md#Create-a-minimal-build-for-Android-with-NNAPI-support).

## Performance caveats

What is optimal will differ by model, and performance testing is the only way to determine what works best for your model. The more times execution has to switch between CPU and NNAPI, the more likely the overhead of doing so will outweight the benefit of using NNAPI.

At a minimum it is suggested to performance test:
  - with NNAPI enabled and an ORT format model created with 'basic' level optimization
  - with NNAPI disabled and an ORT format model created wtih 'extended' level optimization 

If your model is particularly performance sensitive, you can additionally test an NNAPI-aware ORT format model.

Note that it is possible and may be easier to do all this testing with a 'full' [Android build of ONNX Runtime with the NNAPI EP enabled](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#Android-NNAPI-Execution-Provider), as the execution logic and therefore performance for a full build is exactly the same as for a minimal build. The original ONNX model can be used with this testing instead of creating multiple ORT format models, the optimization level [can be specified](ONNX_Runtime_Graph_Optimizations.md#Usage) via SessionOptions, and the NNAPI EP can be [dynamically enabled or disabled](execution_providers/README.md#Using-Execution-Providers). 
