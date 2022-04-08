---
title: Tune Mobile Performance
parent: Performance
nav_order: 2
redirect_from: /docs/how-to/mobile-performance-tuning
---
{::options toc_levels="2" /}

**The information on this page applies to ONNX Runtime version 1.11 and later. See [here](./mobile-performance-tuning-1.10-and-earlier.md) for older versions.**

# ONNX Runtime Mobile Performance Tuning

Learn how different optimizations affect performance, and get suggestions for performance testing with ORT format models.

ONNX Runtime Mobile can be used to execute ORT format models using NNAPI (via the NNAPI Execution Provider (EP)) on Android platforms, and CoreML (via the CoreML EP) on iOS platforms.

First, please review the introductory details in [using NNAPI with ONNX Runtime Mobile](../execution-providers/NNAPI-ExecutionProvider.md) and [using CoreML with ONNX Runtime](../execution-providers/CoreML-ExecutionProvider.md).

**IMPORTANT NOTE:** The examples on this page refer to the NNAPI EP for brevity. The information equally applies to the CoreML EP, so any reference to 'NNAPI' below can be substituted with 'CoreML'.

## Contents
{: .no_toc}

* TOC
{:toc}

## 1. ONNX Model Optimization Example

ONNX Runtime applies optimizations to the ONNX model to improve inferencing performance. These optimizations occur prior to exporting an ORT format model. See the [graph optimization](./graph-optimizations.md) documentation for further details of the available optimizations.

It is important to understand how the different optimization levels affect the nodes in the model, as this will determine how much of the model can be executed using NNAPI or CoreML.

*Basic*

The _basic_ optimizations remove redundant nodes and perform constant folding. Only ONNX operators are used by these optimizations when modifying the model.

*Extended*

The _extended_ optimizations replace one or more standard ONNX operators with custom internal ONNX Runtime operators to boost performance. Each optimization has a list of EPs that it is valid for. It will only replace nodes that are assigned to that EP, and the replacement node will be executed using the same EP.

*Layout*

_Layout_ optimizations may be hardware specific and involve internal conversions between the NCHW image layout used by ONNX and NHWC or NCHWc formats. They are enabled with an optimization level of 'all'.

### Outcome of optimizations when creating an optimized ORT format model

Below is an example of the changes that occur in _basic_ and _extended_ optimizations when applied to the MNIST model with only the CPU EP enabled.

  - At the _basic_ level we combine the Conv and Add nodes (the addition is done via the 'B' input to Conv), we combine the MatMul and Add into a single Gemm node (the addition is done via the 'C' input to Gemm), and constant fold to remove one of the Reshape nodes.

  - At the _extended_ level we additionally fuse the Conv and Relu nodes using the internal ONNX Runtime FusedConv operator.

![Changes to nodes from basic and extended optimizations](../../images/mnist_optimization.png)

### Outcome of executing an optimized ORT format model with the NNAPI EP

If the NNAPI EP is registered at runtime, it is given an opportunity to select the nodes in the loaded model that it can execute. When doing so it will group as many nodes together as possible to minimize the overhead of copying data between the CPU and NNAPI to execute the nodes. Each group of nodes can be considered as a sub-graph. The more nodes in each sub-graph, and the fewer sub-graphs, the better the performance will be.

For each sub-graph, the NNAPI EP will create an [NNAPI model](https://developer.android.com/ndk/guides/neuralnetworks#model) that replicates the processing of the original nodes. It will create a function that executes this NNAPI model and performs any required data copies between CPU and NNAPI. ONNX Runtime will replace the original nodes in the loaded model with a single node that calls this function.

If the NNAPI EP is not registered, or can not process a node, the node will be executed using the CPU EP.

Below is an example for the MNIST model comparing what happens to the ORT format models at runtime if the NNAPI EP is registered.

As the _basic_ level optimizations result in a model that only uses ONNX operators, the NNAPI EP is able to handle the majority of the model as NNAPI can execute the Conv, Relu and MaxPool nodes. This is done with a single NNAPI model as all the nodes NNAPI can handle are connected. We would expect performance gains from using NNAPI with this model, as the overhead of the device copies between CPU and NNAPI for a single NNAPI node is likely to be exceeded by the time saved executing multiple operations at once using NNAPI.

The _extended_ level optimizations introduce the custom FusedConv nodes, which the NNAPI EP ignores as it will only take nodes that are using ONNX operators that NNAPI can handle. This results in two nodes using NNAPI, each handling a single MaxPool operation. The performance of this model is likely to be adversely affected, as the overhead of the device copies between CPU and NNAPI (which are required before and after each of the two NNAPI nodes) is unlikely to be exceeded by the time saved executing a single MaxPool operation each time using NNAPI. Better performance may be obtainable by not registering the NNAPI EP so that all nodes in the model are executed using the CPU EP.

![Changes to nodes by NNAPI EP depending on the optimization level the model was created with](../../images/mnist_optimization_with_nnapi.png)

#### Runtime optimizations

Runtime optimizations provide a way to avoid initial optimizations that introduce nodes the NNAPI EP cannot handle, yet also maintain some ability to further optimize the nodes that the NNAPI EP does not take. As all basic level optimizations will result in a valid ONNX graph, runtime optimizations only apply for higher level optimizations. Only a subset of all higher level graph optimizations support ORT format model runtime optimization.

Some runtime optimizations are saved into the [ORT format model](../reference/ort-format-models.md#saved-runtime-optimizations) and applied when the model is loaded, if applicable. Others are applied only when the ORT format model is loaded.

As saved runtime optimizations capture the effects of the optimizations instead of directly applying them, they are saved with a separate optimization mode when converting to an ORT format model. It is not possible to both directly optimize a model and save runtime optimizations to it at the same time. The `onnxruntime.tools.convert_onnx_models_to_ort` tool's [`--optimization_style`](../reference/ort-format-models.md#optimization-style) parameter controls whether to save runtime optimizations or directly optimize.

## 2. Performance Testing

The best optimization settings will differ by model. Some models may perform better with NNAPI, some models may not. As the performance will be model specific you must run performance tests to determine the best configuration for your model.

Run the [model usability checker](../reference/mobile/helpers.md#ort-mobile-model-usability-checker) helper with your ONNX model. It will provide guidance on how to proceed.
