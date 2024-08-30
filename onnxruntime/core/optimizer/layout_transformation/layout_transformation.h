// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

/* Layout Transformation Tools
 * These methods help change the channel ordering of layout sensitive ops (like Conv). ONNX currently only supports
 * channel first ordering for ops, so this requires changing the op type and domain to a contrib op supporting
 * the new ordering. The existence of a robust transpose optimizer means that we can freely add transpose ops during
 * conversion and then call Optimize to remove as many as possible. To change the channel ordering of some/all ops
 * in a model, a user of this tool should do the following:
 *
 * 1. Iterate over the graph nodes and identify nodes to convert. For each one:
 *    a. Change the op type and domain (and possibly attributes) to the op/contrib op with the desired ordering.
 *    b. The model is now invalid since the input tensors are in the original ordering (and all consumers
 *       expect the original ordering). Use WrapTransposesAroundNode helper to insert transposes around the
 *       inputs/outputs of the op to correct this.
 * 2. The model is now correct but has many unnecessary Transpose ops. Call Optimize on the graph.
 *
 * After step 1, the Transpose ops will wrap converted ops in a similar manner to q/dq ops in quantization.
 * The perm attributes essentially encode the information about which ops are being reordered.
 */

#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/framework/transform_layout_functions.h"
#include "core/optimizer/transpose_optimization/ort_transpose_optimization.h"

namespace onnxruntime {
class Graph;
class IExecutionProvider;

namespace layout_transformation {
/// <summary>
/// Transforms data layout to the EPs preferred layout and runs the transpose optimizer for nodes assigned to the EP.
/// When converting from NCHW to NHWC uses the kMSInternalNHWCDomain domain for updated nodes.
///
/// This can be used by a compiling EP such as NNAPI, where the synthetic domain is a signal that the node has been
/// updated to the EP's required layout, or an EP with statically registered kernels such as XNNPACK where a kernel
/// is registered for the NHWC version of an ONNX operator. The NHWC version of the ONNX operator uses the synthetic
/// domain and is defined by onnxruntime/core/graph/contrib_ops/internal_nhwc_onnx_opset.cc
///
/// Transforms are applied to layout sensitive nodes assigned to execution_provider provided by the caller,
/// and any other non-layout sensitive nodes in order to optimize the transposes as much as possible.
///
/// We call this for all EPs as transpose optimization for a Transpose -> Resize combination is EP specific so must
/// run after the node is assigned to an EP.
/// </summary>
/// <param name="graph">graph to transform</param>
/// <param name="modified">indicates whether the graph is modified during transformation</param>
/// <param name="execution_provider">execution provider for which the transformation needs to be performed</param>
/// <param name="cpu_allocator">a CPU allocator used in layout transformation.
/// <param name="debug_graph_fn">Optional functor to debug the graph produced during layout transformation.
/// This is called after layout transformation if new nodes are inserted, and again after those are optimized.
/// </param>
Status TransformLayoutForEP(onnxruntime::Graph& graph, bool& modified,
                            const onnxruntime::IExecutionProvider& execution_provider,
                            onnxruntime::AllocatorPtr cpu_allocator,
                            const onnxruntime::layout_transformation::DebugGraphFn& debug_graph_fn = {});

/// <summary>
/// Checks if the opset of the Graph is supported by the layout transformer.
/// </summary>
/// <param name="graph">Graph to check</param>
/// <returns></returns>
bool IsSupportedOpset(const Graph& graph);

/// <summary>
/// Gets a list of layout sensitive ops for ORT. This list contains ONNX standard defined
/// layout sensitive ops + contrib ops + ops which are not layout sensitive but are treated as
/// layout sensitive by ORT EPs (example Resize).
/// </summary>
/// <returns>unordered set of op_types which are layout sensitive</returns>
const std::unordered_set<std::string_view>& GetORTLayoutSensitiveOps();

/// <summary>
/// Inserts transposes around op inputs/outputs. Alternatively transposes initializers or uses existing Transpose
/// nodes if possible. Populates shape information on affected node inputs/outputs to reflect the change.
///
/// Ex:
///   * -> NhwcConv -> **
///   becomes
///   * -> Transpose -> NhwcConv -> Transpose -> **
///   Conv inputs/outputs have new shape. Shapes of * and ** are unchanged (carrying NCHW data).
///
/// input_perms/output_perms are matched with node inputs/outputs positionally. Their lengths must be at most equal to
/// the number of inputs/outputs, respectively. nullptr entries indicate an input or output should not be transposed.
/// </summary>
/// <param name="graph">Graph containing the node</param>
/// <param name="node">Node to modify</param>
/// <param name="input_perms">Input permutations. nullptr entries indicate to skip corresponding input.</param>
/// <param name="output_perms">Output permutations. nullptr entries indicate to skip corresponding output.</param>
void WrapTransposesAroundNode(onnx_transpose_optimization::api::GraphRef& graph,
                              onnx_transpose_optimization::api::NodeRef& node,
                              const std::vector<const std::vector<int64_t>*>& input_perms,
                              const std::vector<const std::vector<int64_t>*>& output_perms);
}  // namespace layout_transformation
}  // namespace onnxruntime
