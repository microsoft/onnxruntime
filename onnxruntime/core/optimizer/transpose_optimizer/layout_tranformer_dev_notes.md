# Layout Transformer Dev Notes

## Overview
ONNX standard assumes NCHW format for tensor layout. However, NCHW is not the best (perf efficient) format for all hardware types. Depending on the underlying hardware, to get the best perf we need to convert the model or in some cases part of the model from NCHW -> NHWC. Layout Transformer enables just this. It works with ONNX and ORT format models.

*Note: Currently Layout Transformer works only for compiling EPs like NNAPI, QNN EP etc... More work is needed for it to be compatible with EPs like CPU and CUDA which use static kernel registration.*

## How it works
Layout transformer is [invoked during graph partitioning](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/onnxruntime/core/framework/graph_partitioner.cc#L111) phase. After getting capabilities from the registered EP, layout transformer converts the part of the graph which is claimed by the EP to NHWC IF this is the desired format for the given EP.

```
Foreach
  1. call GetCapability
  2. IF EP.DesiredFormat == NHWC
    2.1. Invoke Layout Transformer
    2.2 If graph is modified -> call GetCapability (layout transformer can add new nodes to the graph)
  3 Compile
```

GetCapability returns a collection of IndexedSubGraphs that the given execution provider can run. During layout transformation, new nodes (Transpose, Gather etc) can be added within these subgraphs. Therefore, calling GetCapability for the second time ensures that the EP can claim these new nodes as well as fuse subgraphs. This is important for perf. Without this the execution will unnecessarily switch to a fallback EP (in most cases CPU EP).

*IMPORTANT NOTE* Layout transformation creates new nodes for layout sensitive ops with domain as "kMSInternalNHWCDomain" and copies the type information and the adjusted shape information from the original node to this new node. If the graph is serialized at this point, the type and shape inference information is lost and it cannot be inferred again on load because the node's domain and format are changed. This is why layout transformation is *NOT ENABLED* when Graph Partitioner Mode is kAssignOnly.

Layout Transformer does multiple top to bottom passes on the graph in order to produce the most efficient graph with as little transpose ops as possible.

### Convert layout for applicable nodes
[This is the first pass](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/onnxruntime/core/optimizer/transpose_optimizer/optimizer_api_impl.cc#L815). Layout transformer simply inserts NCHW -> NHWC transpose node before and NHWC -> NCHW transpose node after every layout sensitive op that is claimed by the EP. After this pass, the graph is correct but extremely inefficient.

"layout sensitive op" refers to an op whose correctness depends on the inputs (activations as well as weights) and attributes being in a certain format. Layout Transformer deals with NCHW and NHWC data layout formats. There are other layout formats as well but they are not in scope for this transformer. These are the existing [ONNX Standard defined](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/onnxruntime/core/optimizer/transpose_optimizer/transpose_optimizer.cc#L2020) and [ORT defined](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/onnxruntime/core/optimizer/transpose_optimizer/optimizer_api_impl.cc#L804) layout sensitive ops.

### Optimize the converted graph
After the first pass is complete, layout transformer calls the transpose optimizer to remove all the canceling as well as redundant transposes from the graph. The following passes happen as part of transpose optimization.

1. Iterate over sorted nodes in reverse order to find which outputs have paths through supported ops to transpose nodes. Transposes will be moved towards these outputs. Graph is not altered in this pass. [Code](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/onnxruntime/core/optimizer/transpose_optimizer/transpose_optimizer.cc#L1875)

2. Push transposes through applicable nodes and remove canceling transposes. At the end of this pass the model will be efficient and will only contain the transpose ops which are necessary for correctness. [Code](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/onnxruntime/core/optimizer/transpose_optimizer/transpose_optimizer.cc#L1905)

3. This is the final optimization pass. In this pass, any transpose node (not part of a QDQ group) if succeeds a DQ node, is moved above the DQ node. In QDQ models this helps to preserve the QDQ node group when a Transpose was pushed across a DQ into an existing QDQ node group. In all other scenarios this is beneficial as well because moving transpose above DQ node is more efficient as transpose node now handles less data. [Code](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/onnxruntime/core/optimizer/transpose_optimizer/transpose_optimizer.cc#L1956)

## Enabling layout transformer for a new Compile based EP
For details please refer to [PR 10371](https://github.com/microsoft/onnxruntime/pull/10371) which introduced layout transformer and enabled it for NNAPI.

Basic steps are as follows:
1. Implement [GetPreferredLayout](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/include/onnxruntime/core/framework/execution_provider.h#L285) method for the EP which overrides the base class method.
2. Remove any existing logic in the EP to convert layouts
3. Add a validation method similar to [IsOpInRequiredLayout](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/onnxruntime/core/providers/nnapi/nnapi_builtin/builders/op_builder.cc#L502) to validate that the layout sensitive op's domain matches "kMSInternalNHWCDomain". Layout Transformer updates the domain for layout sensitive ops to "kMSInternalNHWCDomain" after the conversion to NHWC format.
4. Add tests. The testing framework already includes [InternalTestingExecutionProvider](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/onnxruntime/test/providers/internal_testing/internal_testing_execution_provider.h#L11) which can be leveraged for such tests.

## Making Updates to Transformer and Testing
Apart from bug fixes, updates to layout sensitive op schema as well as addition of new layout sensitive ops will require changes in layout transformer as well as transpose optimizer.

These are some places which may need changes:
1. Updates to [op specific handlers](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/core/optimizer/transpose_optimizer/transpose_optimizer.cc#L1620) in transpose optimizer
2. Updates to layout sensitive op list - [GetLayoutSensitiveOps](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/onnxruntime/core/optimizer/transpose_optimizer/transpose_optimizer.cc#L2020) and [GetORTLayoutSensitiveOps](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/onnxruntime/core/optimizer/transpose_optimizer/optimizer_api_impl.cc#L804)
3. Updates in [TransformLayoutForEP](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/onnxruntime/core/optimizer/transpose_optimizer/optimizer_api_impl.cc#L815) method which relies on schema for deciding which inputs and outputs need to be wrapped with transpose nodes.
4. When upgrading to a new ONNX operator set version [kMaxSupportedOpset](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/onnxruntime/core/optimizer/transpose_optimizer/optimizer_api.h#L437) needs to be updated to enable the transformations for this new opset.

Testing framework provides [InternalTestingExecutionProvider](https://github.com/microsoft/onnxruntime/blob/1a4868e5c4c4a270ad91036e36f2a03410c4c278/onnxruntime/test/providers/internal_testing/internal_testing_execution_provider.h#L11). This can be leveraged to test the changes being introduced.
