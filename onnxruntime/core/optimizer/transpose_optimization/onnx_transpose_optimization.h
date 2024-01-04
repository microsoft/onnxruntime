// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <vector>

// implementation details of the transpose optimizer API defined in optimizer_api.h.
// This exposes some internals so they can be extended as needed.
#include "optimizer_api.h"

namespace onnx_transpose_optimization {

struct OptimizerCtx;

// Struct containing information passed to op handlers.
struct HandlerArgs {
  OptimizerCtx& ctx;
  api::NodeRef& transpose;  // Transpose node we are considering moving past `node`
  api::NodeRef& node;
  const std::vector<int64_t>& perm;      // perm attribute from Transpose
  const std::vector<int64_t>& perm_inv;  // inverse of perm.
  // Cached result from calling HandlerInfo.transposible_inputs_fn
  std::vector<size_t>& transposible_inputs;
};

// Each op handler points to a (potentially shared) function for determining which input indices are eligible for
// optimization. Handlers are only called if a transpose is on an eligible index, and if the optimization heuristics
// predict that pushing the transpose will be beneficial. Most of the time this function returns a static value, but
// for Sum/Concat/QLinearConcat it needs to be dynamic.
using TransposibleInputsFn = std::vector<size_t> (*)(OptimizerCtx& ctx, api::NodeRef& node);
using HandlerFunction = bool (*)(HandlerArgs& args);

struct HandlerInfo {
  TransposibleInputsFn transposible_inputs_fn;
  HandlerFunction handler_fn;
  // Does the handler have to transpose outputs? Used for cost estimation.
  bool transposes_outputs = true;
};

using NodeIdToInputIdxsMap = std::unordered_map<int64_t, std::vector<size_t>>;

struct OptimizerCtx {
  int64_t opset;
  api::GraphRef& graph;
  const std::string provider_type;

  CostCheckFn cost_check_fn;

  // Handlers for ops that are not in the ONNX opset, or for ONNX ops where special handling is required.
  // If a handler is not found in this map, the default handlers will be used.
  const HandlerMap& extended_handlers;
};

/// <summary>
/// TransposibleInputsFn that returns the first input index.
/// </summary>
/// <returns>{0}</returns>
inline std::vector<size_t> FirstInput(OptimizerCtx&, api::NodeRef&) { return {0}; }

std::vector<int64_t> InvertPerm(const std::vector<int64_t>& perm);

// Transpose all inputs and all outputs
bool HandleSimpleNode(HandlerArgs& args);

// Node with all inputs broadcastable
bool HandleSimpleNodeBroadcast(HandlerArgs& args);

// Transposes all inputs and all outputs. Updates axis attribute.
bool HandleSimpleNodeWithAxis(HandlerArgs& args, std::optional<int64_t> default_axis = std::nullopt);

// base handlers that are used by extended handlers. add from transpose_optimizer.cc as needed.
bool HandleReduceOps(HandlerArgs& args);
bool HandleResize([[maybe_unused]] HandlerArgs& args);

void TransposeInput(api::GraphRef& graph, api::NodeRef& node, size_t i,
                    const std::vector<int64_t>& perm,
                    const std::vector<int64_t>& perm_inv);

// Transposes specified inputs according to perm.
// NOTE: if a Transpose is expected to be above an input to this node, use the inverse of its permutation to cancel it.
void TransposeInputs(OptimizerCtx& ctx, api::NodeRef& node, const std::vector<int64_t>& perm,
                     const std::vector<size_t>& input_indices);

inline void TransposeFirstInput(OptimizerCtx& ctx, api::NodeRef& node, const std::vector<int64_t>& perm) {
  std::vector<size_t> indices{0};
  TransposeInputs(ctx, node, perm, indices);
}

// Inserts a Transpose op on the ith output of a node. Returns the new, transposed output.
// Updates shape information assuming that the output from the node will have a transposed shape (using perm_inv)
// but the overall (returned) output will match the initial shape.
std::string_view TransposeOutput(api::GraphRef& graph, api::NodeRef& node, size_t i,
                                 const std::vector<int64_t>& perm,
                                 const std::vector<int64_t>& perm_inv);

void TransposeOutputs(OptimizerCtx& ctx, api::NodeRef& node, const std::vector<int64_t>& perm);

/// <summary>
/// Computes the perm attribute needed to transpose a tensor from channel-first ordering (NCHW or NCD...D) to
/// channel-last ordering (NHWC or ND...DC). rank must be >= 2.
/// </summary>
/// <param name="rank">Rank of the tensor</param>
/// <returns>perm attribute to transpose from channel first to channel last. Ex: [0, 2, 3, 1]</returns>
std::vector<int64_t> ChannelFirstToLastPerm(size_t rank);

/// <summary>
/// Computes the perm attribute needed to transpose a tensor from channel-last ordering (NHWC or ND...DC) to
/// channel-last ordering (NCHW or NCD...D). rank must be >= 2.
/// </summary>
/// <param name="rank">Rank of the tensor</param>
/// <returns>perm attribute to transpose from channel last to channel first. Ex: [0, 3, 1, 2]</returns>
std::vector<int64_t> ChannelLastToFirstPerm(size_t rank);

/// <summary>
/// Updates the axis attribute of QuantizeLinear or DequantizeLinear operators according to the
/// provided transposition permutation. Only applies to per-axis (de)quantization.
/// </summary>
/// <param name="graph">The graph containing the node</param>
/// <param name="perm">The transpose permutation</param>
/// <param name="node">The QuantizeLinear or DequantizeLinear node</param>
/// <returns>True if the axis attribute remains valid</returns>
bool TransposeQuantizeDequantizeAxis(const api::GraphRef& graph, const std::vector<int64_t>& perm, api::NodeRef& node);
}  // namespace onnx_transpose_optimization
