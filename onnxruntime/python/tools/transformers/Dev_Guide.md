# Transformer Model Optimization Tool Dev Guide

Transformer model optimization tool applies some of the latest graph variations. However, it cannot cover all the cases especially for the new ones that are coming out of academics. This guide will give you an overall introduction of how the graph transformation works and how to optimize your custom transformer-based model with limited code changes on graph fusion logic and kernels implementations. 

The objective of the Dev Guide is to enable more transformer-based models to take advantage of ONNXRuntime optimized kernels.

Meanwhile, welcome to contribute!

## Prerequisite
* Expect developer has basic knowledge of C++, CUDA and python programming.
* [Transformer Model Optimization Tool Overview](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/README.md)
* This guide assumes that a valid onnx model exported from the original framework is ready. If there is any issues with model exporting, fp16 conversion, profiling and benchmark. Please refer to the above link.
* Optional: In case kernel changes are needed, here is the instruction on [building the ONNXRuntime](https://onnxruntime.ai/docs/build/) with packages on [different APIs and Language bindings](https://onnxruntime.ai/docs/build/inferencing.html#apis-and-language-bindings)

## Rule Of Thumb

The graph fusion transforms certain graph structure to single fused node. The kernel wrapped by the fused node is the strict computation equivalent of that certain graph structure and executed by the runtime engine. This means that the candidate graph should have the exact same logic as fused node kernel implementation. It's suggested to get familiar with the targeted optimized kernel implementation and then work on the fusion logic.

### Kernel Implementation
ONNXRuntime supports optimized kernels as contrib operators in both CPU and CUDA Execution Provider. 
* The defination of the optimized kernels can be found in [contrib_defs.cc](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/core/graph/contrib_ops/contrib_defs.cc). 
* The CPU implementation of the optimized kernels can be found under [/contrib_ops/cpu/bert](https://github.com/microsoft/onnxruntime/tree/rel-1.9.0/onnxruntime/contrib_ops/cpu/bert). 
* The CUDA implementation of the optimized kernels can be found under [/contrib_ops/cuda/bert](https://github.com/microsoft/onnxruntime/tree/rel-1.9.0/onnxruntime/contrib_ops/cuda/bert).
* [Tests](https://github.com/microsoft/onnxruntime/tree/rel-1.9.0/onnxruntime/test/contrib_ops)

For instance, the entry point of Attention CPU kernel is the [Compute()](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/contrib_ops/cpu/bert/attention.cc#L408) function. Similarly, for EmbedLayerNorm CUDA kernel, the entry point is the [ComputeInternal()](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/contrib_ops/cuda/bert/embed_layer_norm.cc#L36) function.

### Graph Fusion
The main part of the transformer [optimizer](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/optimizer.py) is graph fusion. In the current implementation for bert optimization, it supports a couple of [fusions](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/onnx_model_bert.py#L302) executed in order. Each particular graph fusion is an inheritance class of [Fusion](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/fusion_base.py#L13) with fuse() method to implement. For instance, the [fuse()](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/fusion_attention.py#L280) method in attention fusion.

During the process of the fusion

## Contribution
[Coding Conventions and Standards](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/docs/Coding_Conventions_and_Standards.md)

