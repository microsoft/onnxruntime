# Transformer Model Optimization Tool Dev Guide

Transformer model optimization tool applies to BERT, GPT-2 and some variations (like Roberta, DistilBert etc). However, it cannot cover all the cases especially for the new ones that are coming out of academics. This guide will give you an overall introduction of how the graph transformation works and how to optimize your custom transformer-based model with limited code changes on graph fusion logic and kernels implementations.

The objective of the Dev Guide is to enable more transformer-based models to take advantage of ONNXRuntime optimized kernels.

Meanwhile, welcome to contribute!

## Prerequisite
* Expect the developer has basic knowledge of C++, CUDA and python programming.
* [Transformer Model Optimization Tool Overview](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/README.md)
* This guide assumes that a valid onnx model exported from the original framework is ready. If there are any issues with model exporting, fp16 conversion, profiling and benchmark. Please refer to the above link.
* [Netron](https://github.com/lutzroeder/netron) is an excellent graph visualization tool. [Web version](https://netron.app/)
* Optional: In case kernel changes are needed, here is the instruction on [building the ONNXRuntime](https://onnxruntime.ai/docs/build/) with packages on [different APIs and Language bindings](https://onnxruntime.ai/docs/build/inferencing.html#apis-and-language-bindings)

## Rule Of Thumb

The graph fusion transforms a certain graph structure to a single fused node. The kernel wrapped by the fused node is the strict computation equivalent of that certain graph structure and executed by the runtime engine. This means that the candidate graph should have the exact same logic as fused node kernel implementation. It's suggested to get familiar with the targeted optimized kernel implementation and then work on the fusion logic.

### Kernel Implementation
ONNXRuntime supports optimized kernels as contrib operators in both CPU and CUDA Execution Provider.
* The definition of the optimized kernels can be found in [onnxruntime/core/graph/contrib_ops/contrib_defs.cc](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/core/graph/contrib_ops/contrib_defs.cc).
* The CPU implementation of the optimized kernels can be found under [onnxruntime/contrib_ops/cpu/bert](https://github.com/microsoft/onnxruntime/tree/rel-1.9.0/onnxruntime/contrib_ops/cpu/bert).
* The CUDA implementation of the optimized kernels can be found under [onnxruntime/contrib_ops/cuda/bert](https://github.com/microsoft/onnxruntime/tree/rel-1.9.0/onnxruntime/contrib_ops/cuda/bert).
* Contrib ops tests can be found [here](https://github.com/microsoft/onnxruntime/tree/rel-1.9.0/onnxruntime/test/contrib_ops)

For instance, the entry point of Attention CPU kernel is the [Compute()](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/contrib_ops/cpu/bert/attention.cc#L408) function. Similarly, for the EmbedLayerNorm CUDA kernel, the entry point is the [ComputeInternal()](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/contrib_ops/cuda/bert/embed_layer_norm.cc#L36) function.

### Graph Fusion
The main part of the transformer [optimizer](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/optimizer.py) is graph fusion. In the current implementation for bert optimization, it supports a couple of [fusions](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/onnx_model_bert.py#L302) executed in order. Each particular graph fusion is an inheritance class of [Fusion](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/fusion_base.py#L13) with fuse() method to implement. For instance, the [fuse()](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/fusion_attention.py#L280) method in attention fusion.

The [onnx_model](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/onnx_model.py#L19) class provides many useful functions to modify onnx graph including not limited to:
* Retrieve all graph nodes with [self.nodes()](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/onnx_model.py#L58)
* A [mapping](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/onnx_model.py#L41-L56) of edge names to nodes.
* [Basic operations](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/onnx_model.py#L120-L181) of input/output, node, initializer.
* [Match graph patterns](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/onnx_model.py#L310-L385) up-streaming and down-streaming.

#### Fusion process
* Match the candidate graph with expected connection pattern. [Example: Gelu fusion](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/fusion_gelu.py#L26-L96), [Attention fusion](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/fusion_attention.py#L281-L441)
* Construct the fused node with inputs, outputs and the weights obtained from the original graph. [Example: Gelu fusion](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/fusion_gelu.py#L99-L102), [Attention fusion](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/fusion_attention.py#L142-L278)
* Remove the candidate graph. [Example: Gelu fusion](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/fusion_gelu.py#L98), [Attention fusion](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/fusion_attention.py#L468-L472)

After fusing the graph, [check the parity](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/onnx_exporter.py#L104) between optimized onnx model and original one by feeding the same inputs to both models and comparing outputs.

## A Concrete Case
* The Attention Op and EmbedLayerNorm Op are not fused([EmbedLayerNorm graph](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/images/embed_unfused.png) and [Attention graph](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/images/attention_unfused.png) with Netron) after running optimization script on a custom transformer-based onnx model.
* Checked and confirmed that these two candidate graphs have identical logic to the current CPU/CUDA kernel implementation.
* Applied some code changes to the [Attention fusion](https://github.com/microsoft/onnxruntime/compare/wangye/opt#diff-bd125663ee59865deb608c7ec666ac4760b55ce73fc38cc3d463abd0aaa90817) and [EmbedLayerNorm fusion](https://github.com/microsoft/onnxruntime/compare/wangye/opt#diff-bb2157f08cf00e8434e77fcfeeaa960e5e9c6db2df2b637a5f49e48d77a56185)
* Re-run the script and these two Ops are fused([EmbedLayerNorm Op](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/images/embed_fused.png) and [Attention Op](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/images/attention_fused.png) with Netron).
* The parity is OK

## Contribution
[Coding Conventions and Standards](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/docs/Coding_Conventions_and_Standards.md)
