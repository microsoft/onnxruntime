# Transformer Model Optimization Tool Dev Guide

Transformer model optimization tool applies some of the latest graph variations. However, it cannot cover all the cases especially for the new ones that are coming out of academics. This guide will give you an overall introduction of how the graph transformation works and how to optimize your custom transformer-based model with limited code changes on graph fusion logic and kernels implementations. Meanwhile, welcome to contribute!

## Prerequisite
* Expect developer has basic knowledge of C++, CUDA and python programming.
* [Transformer Model Optimization Tool Overview](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/README.md)
* This guide assumes that a valid onnx model exported from the original framework is ready. If you have any issues with model exporting, fp16 conversion, profiling and benchmark. Please refer to the above link.
