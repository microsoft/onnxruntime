# ONNX Runtime Mobile: Performance Considerations When Using NNAPI

ONNX Runtime Mobile with the NNAPI Execution Provider (EP) can be used to execute ORT format models on Android platforms using NNAPI. This document explains the details of how different optimizations affect performance, and provides some suggestions for performance testing with ORT format models.

Please review the introductory details for [using NNAPI with ONNX Runtime Mobile](ONNX_Runtime_for_Mobile_Platforms.md#Using-NNAPI-with-ONNX-Runtime-Mobile) first.


## 1. ONNX Model Optimization Example

ONNX Runtime applies optimizations to the ONNX model to improve inferencing performance. These optimizations occur prior to exporting an ORT format model. See the [graph optimization](https://www.onnxruntime.ai/docs/resources/graph-optimizations.html) documentation for further details of the available optimizations.

It is important to understand how the different optimization levels affect the nodes in the model, as this will determine how much of the model can be executed using NNAPI.

*Basic*

The _basic_ optimizations remove redundant nodes and perform constant folding. Only ONNX operators are used by these optimizations when modifying the model.

*Extended*

The _extended_ optimizations replace one or more standard ONNX operators with custom internal ONNX Runtime operators to boost performance. Each optimization has a list of EPs that it is valid for. It will only replace nodes that are assigned to that EP, and the replacement node will be executed using the same EP.

*Layout*

_Layout_ optimizations are hardware specific, and should not be used when creating ORT format models.

### Outcome of optimizations when creating an optimized ORT format model

Below is an example of the changes that occur in _basic_ and _extended_ optimizations when applied to the MNIST model with only the CPU EP enabled. The optimization level is specified when creating the ORT format model using `convert_onnx_models_to_ort.py`.

  - At the _basic_ level we combine the Conv and Add nodes (the addition is done via the 'B' input to Conv), we combine the MatMul and Add into a single Gemm node (the addition is done via the 'C' input to Gemm), and constant fold to remove one of the Reshape nodes.
    - `python <ORT repository root>/tools/python/convert_onnx_models_to_ort.py --optimization_level basic /dir_with_mnist_onnx_model`
  - At the _extended_ level we additionally fuse the Conv and Relu nodes using the internal ONNX Runtime FusedConv operator.
    - `python <ORT repository root>/tools/python/convert_onnx_models_to_ort.py --optimization_level extended /dir_with_mnist_onnx_model`

<img align="center" src="images/mnist_optimization.png" alt="Changes to nodes from basic and extended optimizations."/>

### Outcome of executing an optimized ORT format model using the NNAPI EP

If the NNAPI EP is registered at runtime, it is given an opportunity to select the nodes in the loaded model that it can execute. When doing so it will group as many nodes together as possible to minimize the overhead of copying data between the CPU and NNAPI to execute the nodes. Each group of nodes can be considered as a sub-graph. The more nodes in each sub-graph, and the fewer sub-graphs, the better the performance will be.

For each sub-graph, the NNAPI EP will create an [NNAPI model](https://developer.android.com/ndk/guides/neuralnetworks#model) that replicates the processing of the original nodes. It will create a function that executes this NNAPI model and performs any required data copies between CPU and NNAPI. ONNX Runtime will replace the original nodes in the loaded model with a single node that calls this function.

If the NNAPI EP is not registered, or can not process a node, the node will be executed using the CPU EP.

Below is an example for the MNIST model comparing what happens to the ORT format models at runtime if the NNAPI EP is registered.

As the _basic_ level optimizations result in a model that only uses ONNX operators, the NNAPI EP is able to handle the majority of the model as NNAPI can execute the Conv, Relu and MaxPool nodes. This is done with a single NNAPI model as all the nodes NNAPI can handle are connected. We would expect performance gains from using NNAPI with this model, as the overhead of the device copies between CPU and NNAPI for a single NNAPI node is likely to be exceeded by the time saved executing multiple operations at once using NNAPI.

The _extended_ level optimizations introduce the custom FusedConv nodes, which the NNAPI EP ignores as it will only take nodes that are using ONNX operators that NNAPI can handle. This results in two nodes using NNAPI, each handling a single MaxPool operation. The performance of this model is likely to be adversely affected, as the overhead of the device copies between CPU and NNAPI (which are required before and after each of the two NNAPI nodes) is unlikely to be exceeded by the time saved executing a single MaxPool operation each time using NNAPI. Better performance may be obtainable by not registering the NNAPI EP so that all nodes in the model are executed using the CPU EP.

<img align="center" src="images/mnist_optimization_with_nnapi.png" alt="Changes to nodes by NNAPI EP depending on the optimization level the model was created with.">

## 2. Initial Performance Testing

The best optimization settings will differ by model. Some models may perform better with NNAPI, some models may not. As the performance will be model specific you must performance test to determine the best combination for your model.

It is suggested to run performance tests:
  - with NNAPI enabled and an ORT format model created with _basic_ level optimization
  - with NNAPI disabled and an ORT format model created with _extended_ level optimization

For most scenarios it is expected that one of these two approaches will yield the best performance.

If using an ORT format model with _basic_ level optimizations and NNAPI yields equivalent or better performance, it _may_ be possible to further improve performance by creating an NNAPI-aware ORT format model. The difference with this model is that the _extended_ optimizations are applied to nodes that can not be executed using NNAPI. Whether any nodes fall into this category is model dependent.

## 3. Creating an NNAPI-aware ORT format model

An NNAPI-aware ORT format model will keep all nodes from the ONNX model that can be executed using NNAPI, and allow _extended_ optimizations to be applied to any remaining nodes.

For our MNIST model that would mean that after the _basic_ optimizations are applied, the nodes in the red shading are kept as-is, and nodes in the green shading could have _extended_ optimizations applied to them.

<img align="center" src="images/nnapi_aware_ort_format_model.png" alt="Show nodes that are preserved as NNAPI can execute them, and nodes that are considered by extended optimizations.">

To create an NNAPI-aware ORT format model please follow these steps.

1. Create a 'full' build of ONNX Runtime with the NNAPI EP by [building ONNX Runtime from source](https://www.onnxruntime.ai/docs/how-to/build.html#cpu).

    This build can be done on any platform, as the NNAPI EP can be used to create the ORT format model without the Android NNAPI library as there is no model execution in this process. When building add `--use_nnapi --build_shared_lib --build_wheel` to the build flags if any of those are missing.

    Do NOT add the --minimal_build` flag.
    - Windows :
        ```
        <ONNX Runtime repository root>\build.bat --config RelWithDebInfo --use_nnapi --build_shared_lib --build_wheel --parallel
        ```

    - Linux:
        ```
        <ONNX Runtime repository root>/build.sh --config RelWithDebInfo --use_nnapi --build_shared_lib --build_wheel --parallel
        ```

  - **NOTE** if you have previously done a minimal build you will need to run `git reset --hard` to make sure any operator kernel exclusions are reversed prior to performing the 'full' build. If you do not, you may not be able to load the ONNX format model due to missing kernels.

2. Install the python wheel from the build output directory.

    - Windows : This is located in `build/Windows/<config>/<config>/dist/<package name>.whl`.

    - Linux : This is located in `build/Linux/<config>/dist/<package name>.whl`.

        The package name will differ based on your platform, python version, and build parameters. `<config>` is the value from the `--config` parameter from the build command.
        ```
            pip install -U build\Windows\RelWithDebIfo\RelWithDebIfo\dist\onnxruntime_noopenmp-1.5.2-cp37-cp37m-win_amd64.whl
        ```

3. Create an NNAPI-aware ORT format model by running `convert_onnx_models_to_ort.py` as per the [standard instructions](ONNX_Runtime_for_Mobile_Platforms.md#Create-ORT-format-model-and-configuration-file-with-required-operators), with NNAPI enabled (`--use_nnapi`), and the optimization level set to _extended_ (`--optimization_level extended`). This will allow extended level optimizations to run on any nodes that NNAPI can not handle.

        ```
        python <ORT repository root>/tools/python/convert_onnx_models_to_ort.py --use_nnapi --optimization_level extended /models
        ```

    The python package from your 'full' build with NNAPI enabled must be installed for `--use_nnapi` to be a valid option

This model can be used with [a minimal build that includes the NNAPI EP](ONNX_Runtime_for_Mobile_Platforms.md#Create-a-minimal-build-for-Android-with-NNAPI-support).