# Transformer Model Optimization Tool Dev Guide

Transformer model optimization tool applies some of the latest graph variations. However, it cannot cover all the cases especially for the new ones that are coming out of academics. This guide will give you an overall introduction of how the graph transformation works and how to optimize your custom transformer-based model with limited code changes on graph fusion logic and kernels implementations. 

This guide will only focus on graph fusion as it's sensitive to the graph changes. The graph fusion logic is implemented with known onnx graph structures. Therefore, the corresponding optimizations will not be applied if some of the graphs are not recognized by the fusion logic. The objective of the Dev Guide is to enable more transformer-based models to take advantage of ONNXRuntime optimized kernels.

Meanwhile, welcome to contribute!


## Prerequisite
* Expect developer has basic knowledge of C++, CUDA and python programming.
* [Transformer Model Optimization Tool Overview](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/README.md)
* This guide assumes that a valid onnx model exported from the original framework is ready. If there is any issues with model exporting, fp16 conversion, profiling and benchmark. Please refer to the above link.

## Rule Of Thumb

### Graph Fusion
The main part of the transformer [optimizer](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/optimizer.py) is graph fusion. In the current implementation for bert optimization, it supports a couple of [fusions](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/onnx_model_bert.py#L302)

### Kernel Implementation

## Contribute
[Coding Conventions and Standards](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/docs/Coding_Conventions_and_Standards.md)

