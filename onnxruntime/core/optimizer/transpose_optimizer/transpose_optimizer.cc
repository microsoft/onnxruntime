// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "api.h"

#include <algorithm>
#include <gsl/gsl>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace onnx_layout_transformation {

struct OptimizerCtx {
  int64_t opset;
  api::GraphRef& graph;
  bool allow_extended_ops;
  bool skip_cost_check;
};

// Each op handler points to a (potentially shared) function for determining which input indices are eligible for
// optimization. Handlers are only called if a transpose is on an eligible index, and if the optimization heuristics
// predict that pushing the transpose will be beneficial. Most of the time this function returns a static value, but
// for Sum/Concat/QLinearConcat it needs to be dynamic.
using TransposibleInputsFn = std::vector<size_t> (*)(OptimizerCtx& ctx, api::NodeRef& node);

// Struct containing information passed to op handlers. Decreases binary size and allows perm_inv to be precomputed.
struct HandlerArgs {
  OptimizerCtx& ctx;
  api::NodeRef& transpose;
  api::NodeRef& node;
  const std::vector<int64_t>& perm;
  const std::vector<int64_t>& perm_inv;
  // Cached result from calling transposible_inputs_fn
  std::vector<size_t>& transposible_inputs;
};

using HandlerFunction = bool (*)(HandlerArgs& args);

struct HandlerInfo {
  TransposibleInputsFn transposible_inputs_fn;
  HandlerFunction handler_fn;
  // Does the handler have to transpose outputs? Used for cost estimation.
  bool transposes_outputs = true;
};


/////// <Helper Utils> ///////
/* Small utilities for editing nodes and manipulating axes/permutations */

static std::vector<int64_t> DataInt64(api::TensorRef& tensor) {
  std::vector<uint8_t> raw_data = tensor.Data();
  int64_t* data_int = reinterpret_cast<int64_t*>(raw_data.data());
  std::vector<int64_t> result(data_int, data_int + tensor.NumElements());
  return result;
}

static std::vector<int32_t> DataInt32(api::TensorRef& tensor) {
  std::vector<uint8_t> raw_data = tensor.Data();
  int32_t* data_int = reinterpret_cast<int32_t*>(raw_data.data());
  std::vector<int32_t> result(data_int, data_int + tensor.NumElements());
  return result;
}

static std::string_view AddInitializerInt64(api::GraphRef& graph, const std::vector<int64_t>& shape,
                                     const std::vector<int64_t>& values) {
  const uint8_t* raw_data = reinterpret_cast<const uint8_t*>(values.data());
  std::vector<uint8_t> data(raw_data, raw_data + values.size() * sizeof(int64_t));
  return graph.AddInitializer(api::DataType::INT64, shape, data);
}

static std::string_view AddInitializerInt32(api::GraphRef& graph, const std::vector<int64_t>& shape,
                                     const std::vector<int32_t>& values) {
  const uint8_t* raw_data = reinterpret_cast<const uint8_t*>(values.data());
  std::vector<uint8_t> data(raw_data, raw_data + values.size() * sizeof(int32_t));
  return graph.AddInitializer(api::DataType::INT32, shape, data);
}

// Replaces all node inputs referencing old_value with references to new_value. Values must be non-empty strings.
// This is an alternative to using MoveOutput for cases when the values aren't node outputs (if one is an initializer,
// for example).
static void ReplaceValueReferences(const std::vector<std::unique_ptr<api::NodeRef>>& nodes,
                                   std::string_view old_value, std::string_view new_value) {
  for (const std::unique_ptr<api::NodeRef>& node : nodes) {
    const std::vector<std::string_view>& inputs = node->Inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i] == old_value) {
        node->SetInput(i, new_value);
      }
    }
  }
}

// Create a node with a single attribute of type vector<int64_t>
static std::unique_ptr<api::NodeRef> MakeNode1Attr(api::GraphRef& graph, std::string_view op_type,
                                                   std::string_view input, std::string_view attr_name,
                                                   const std::vector<int64_t>& attr_val) {
  std::vector<std::string_view> inputs{input};
  std::unique_ptr<api::NodeRef> node = graph.AddNode(op_type, inputs, /*num_outputs*/ 1);
  node->SetAttributeInts(attr_name, attr_val);
  return node;
}

// Creates a Transpose node. Does not update output ValueInfo.
static std::unique_ptr<api::NodeRef> MakeTranspose(api::GraphRef& graph, std::string_view input,
                                                   const std::vector<int64_t>& perm) {
  return MakeNode1Attr(graph, "Transpose", input, "perm", perm);
}

// Creates a Squeeze/Unsqueeze node. Does not update output ValueInfo.
static std::unique_ptr<api::NodeRef> MakeSqueezeOrUnsqueeze(int64_t opset, api::GraphRef& graph, 
                                                            std::string_view op_type, std::string_view input,
                                                            const std::vector<int64_t>& axes) {
  if (opset < 13) {
    return MakeNode1Attr(graph, op_type, input, "axes", axes);
  }

  std::vector<int64_t> axes_shape{gsl::narrow_cast<int64_t>(axes.size())};
  std::string_view axes_initializer = AddInitializerInt64(graph, axes_shape, axes);

  std::vector<std::string_view> inputs{input, axes_initializer};

  return graph.AddNode(op_type, inputs, /*num_outputs*/ 1);
}

// Returns whether perm is a valid permutation (contains each value from 0 to perm.size() - 1 exactly once)
static bool IsValidPerm(const std::vector<int64_t>& perm) {
  size_t rank = perm.size();
  int64_t rank_int = gsl::narrow_cast<int64_t>(rank);
  std::vector<bool> used_dims(rank);
  for (size_t i = 0; i < rank; ++i) {
    int64_t x = perm[i];
    size_t x_size_t = gsl::narrow_cast<size_t>(x);
    if (x < 0 || x >= rank_int || used_dims[x_size_t]) {
      return false;
    }
    used_dims[x_size_t] = true;
  }
  return true;
}

static std::optional<std::vector<int64_t>> GetPermAttrIfValid(const api::NodeRef& node) {
  std::optional<std::vector<int64_t>> perm = node.GetAttributeInts("perm");
  if (perm != std::nullopt && !IsValidPerm(*perm)) {
    return std::nullopt;
  }
  return perm;
}

// Adds rank to negative axes and checks that axes are unique and within [0, rank). Returns false if invalid.
static bool NormalizeAndValidateAxes(std::vector<int64_t>& axes, size_t rank) {
  int64_t rank_int = gsl::narrow_cast<int64_t>(rank);
  std::vector<bool> used_dims(rank);
  for (size_t i = 0; i < axes.size(); ++i) {
    if (axes[i] < 0) {
      axes[i] += rank_int;
      size_t x_size_t = gsl::narrow_cast<size_t>(axes[i]);
      if (axes[i] < 0 || axes[i] >= rank_int || used_dims[x_size_t]) {
        return false;
      }
      used_dims[x_size_t] = true;
    }
  }
  return true;
}

static inline bool NormalizeAndValidateAxis(int64_t& axis, size_t rank) {
  int64_t rank_int = gsl::narrow_cast<int64_t>(rank);
  if (axis < 0) {
    axis += rank_int;
  }

  return axis >= 0 && axis < rank_int;
}

// Read int64 data from attribute or input, depending on whether model opset < provided opset
static std::optional<std::vector<int64_t>> ReadFromAttrOrInput(OptimizerCtx& ctx, api::NodeRef& node,
                                                               std::string_view attr_name, size_t inp_index,
                                                               int64_t opset) {
  if (ctx.opset < opset) {
    return node.GetAttributeInts(attr_name);
  } else {
    auto inputs = node.Inputs();
    if (inp_index >= inputs.size() || inputs[inp_index] == "") {
      return std::nullopt;
    }
    auto constant = ctx.graph.GetConstant(inputs[inp_index]);
    if (constant == nullptr) {
      return std::nullopt;
    }
    return DataInt64(*constant);
  }
}

// Computes inverse permutation. Unsafe if perm is not a valid permutation.
static std::vector<int64_t> InvertPerm(const std::vector<int64_t>& perm) {
  size_t rank = perm.size();
  std::vector<int64_t> perm_inv(rank);
  for (size_t i = 0; i < rank; ++i) {
    size_t j = gsl::narrow_cast<size_t>(perm[i]);
    perm_inv[j] = gsl::narrow_cast<int64_t>(i);
  }
  return perm_inv;
}

// Computes composition of perm1 and perm2. Unsafe if perm1 or perm2 are not valid permutations.
static std::vector<int64_t> ComposePerm(const std::vector<int64_t>& perm1, const std::vector<int64_t>& perm2) {
  std::vector<int64_t> perm;
  perm.reserve(perm2.size());
  for (int64_t p : perm2) {
    perm.push_back(perm1[gsl::narrow_cast<size_t>(p)]);
  }
  return perm;
}

// Returns true if perm[i] = i everywhere
static bool IsIdentityPerm(const std::vector<int64_t>& perm) {
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != gsl::narrow_cast<int64_t>(i)) {
      return false;
    }
  }
  return true;
}

// Computes permutation from channel last to channel first ordering of given rank. Nearly all handlers work for any
// permutation, but some are restricted. Also used for layout transformation. Rank must be >= 1.
std::vector<int64_t> ChannelLastToFirstPerm(size_t rank) {
  std::vector<int64_t> p(rank);
  p[0] = 0;
  p[1] = rank - 1;
  for (size_t i = 2; i < rank; ++i) {
    p[i] = i - 1;
  }
  return p;
}

std::vector<int64_t> ChannelFirstToLastPerm(size_t rank) {
  return InvertPerm(ChannelLastToFirstPerm(rank));
}

// Adds 1 dimensions to indices of shape corresponding to axes. Unsafe if axes has negative/duplicated entries.
static std::vector<int64_t> UnsqueezeShape(gsl::span<const int64_t> shape, const std::vector<int64_t>& axes) {
  size_t new_rank = shape.size() + axes.size();
  std::vector<int64_t> new_shape(new_rank);

  // Fill unsqueezed axes with 1s
  for (int64_t a : axes) {
    new_shape[gsl::narrow_cast<size_t>(a)] = 1;
  }

  // Fill remaining axes with existing shape. Skip entries prefilled with 1s.
  size_t j = 0;
  for (size_t i = 0; i < new_rank; i++) {
    if (new_shape[i] != 1) {
      new_shape[i] = shape[j++];
    }
  }
  return new_shape;
}

// Computes new perm for unsqueezed version of a tensor. Unsafe if axes/perm are invalid or have negative values.
// New perm reorders non-1 dimensions in the same way and leaves 1-dims from unsqueeze unchanged.
// Ex:
// perm = [2, 0, 1] means shape [A, B, C] -> [C, A, B]. If axes = [0, 3], map to
// result = [0, 4, 1, 3, 2] means shape [1, A, B, 1, C] -> [1, C, A, 1, B]
static std::vector<int64_t> UnsqueezePerm(const std::vector<int64_t>& axes, const std::vector<int64_t>& perm) {
  size_t old_rank = perm.size();
  size_t new_rank = old_rank + axes.size();

  // Determine added axes
  std::vector<bool> is_added_axis(new_rank);
  for (int64_t a : axes) {
    is_added_axis[gsl::narrow_cast<size_t>(a)] = true;
  }

  // map old axes to new (unsqueezed) axes
  std::vector<int64_t> axes_map;
  axes_map.reserve(axes.size());
  for (size_t i = 0; i < new_rank; ++i) {
    if (!is_added_axis[i]) {
      axes_map.push_back(i);
    }
  }

  std::vector<int64_t> new_perm;
  new_perm.reserve(new_rank);
  size_t j = 0;
  for (size_t i = 0; i < new_rank; ++i) {
    if (is_added_axis[i]) {
      // Leave 1s in the same place
      new_perm.push_back(i);
    } else {
      // Take next axis from perm
      size_t perm_axis = gsl::narrow_cast<size_t>(perm[j++]);
      new_perm.push_back(axes_map[perm_axis]);
    }
  }
  return new_perm;
}

// Computes new perm for squeezed version of a tensor. Unsafe if axes/perm are invalid or have negative values.
// Result has size perm.size() - axes.size() and reorders remaining axes according to perm.
static std::vector<int64_t> SqueezePerm(const std::vector<int64_t>& axes, const std::vector<int64_t>& perm) {
  // Determine removed axes
  std::vector<bool> is_removed_axis(perm.size());
  for (int64_t a : axes) {
    is_removed_axis[gsl::narrow_cast<size_t>(a)] = true;
  }

  // Map old axes to new axes. (leave removed axes unassigned)
  std::vector<int64_t> axes_map(perm.size());
  int64_t j = 0;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (!is_removed_axis[i]) {
      axes_map[i] = j++;
    }
  }

  // Add perm entries for retained axes.
  std::vector<int64_t> new_perm;
  new_perm.reserve(perm.size());
  for (int64_t p : perm) {
    size_t p_size_t = gsl::narrow_cast<size_t>(p);
    if (!is_removed_axis[p_size_t]) {
      new_perm.push_back(axes_map[p_size_t]);
    }
  }

  return new_perm;
}

// Computes a new axes attribute for an input that has been permuted using perm. Unsafe if axes/perm are invalid or 
// have negative values.
// 
// Ex: perm = [2, 0, 1], axes = [0, 1], new_axes = [2, 0]
static std::vector<int64_t> AxesForTransposedInput(const std::vector<int64_t>& axes,
                                                   const std::vector<int64_t>& perm) {
  std::vector<int64_t> new_axes;
  new_axes.reserve(axes.size());
  for (int64_t a : axes) {
    new_axes.push_back(perm[gsl::narrow_cast<size_t>(a)]);
  }
  return new_axes;
}

// Computes a new axes attribute for an input that has been permuted using perm and sorts the result. Axes attributes
// are commonly sorted (unless order matters like in Slice). Unsafe if axes/perm are invalid or have negative values.
// 
// Ex: perm = [2, 0, 1], axes = [0, 1], new_axes = [0, 2]
static std::vector<int64_t> SortedAxesForTransposedInput(const std::vector<int64_t>& axes,
                                                         const std::vector<int64_t>& perm) {
  size_t rank = perm.size();

  // Mark axes to include
  std::vector<bool> should_include_axis(perm.size());
  for (int64_t a : axes) {
    size_t a_size_t = gsl::narrow_cast<size_t>(a);
    size_t new_axis = gsl::narrow_cast<size_t>(perm[a_size_t]);
    should_include_axis[new_axis] = true;
  }

  // Create sorted result.
  std::vector<int64_t> new_axes;
  for (size_t a = 0; a < rank; a++) {
    if (should_include_axis[a]) {
      new_axes.push_back(gsl::narrow_cast<int64_t>(a));
    }
  }

  return new_axes;
}

/////// </Helper Utils> ///////

/////// <Core Helpers> ///////
/* These helpers hide the most gnarly parts of the transpose optimizer. */


static std::string_view HelpHandleUnsqueeze(HandlerArgs& args, const std::vector<int64_t>& axes);


// Replaces ith input to node with unsqueezed value. Might create a new Unsqueeze node, find an existing one,
// or reshape an initializer. Unsqueezing can be necessary before transposing inputs of a node that supports
// broadcasting.
static void UnsqueezeInput(OptimizerCtx& ctx, api::NodeRef& node, size_t i, const std::vector<int64_t>& axes) {
  std::string_view input = node.Inputs()[i];
  // Remove this node as a consumer
  node.SetInput(i, "");

  std::unique_ptr<api::TensorRef> constant = ctx.graph.GetLocalConstant(input);
  auto consumers = ctx.graph.GetValueConsumers(input);

  // Case 1: input is a constant with a known list of consumer nodes
  if (constant != nullptr && consumers->comprehensive) {
    // We will reshape the initializer. If there are existing consumers, still reshape it but add Squeeze nodes
    // to counteract its effect. If they later Unsqueeze the same input, the Squeeze nodes will simply be deleted
    // (see Case 2).
    if (consumers->nodes.size() > 0) {
      auto squeeze_ptr = MakeSqueezeOrUnsqueeze(ctx.opset, ctx.graph, "Squeeze", input, axes);
      api::NodeRef& squeeze = *squeeze_ptr;
      std::string_view sq_out = squeeze.Outputs()[0];
      ctx.graph.CopyValueInfo(input, sq_out);
      ReplaceValueReferences(consumers->nodes, input, sq_out);
    }
    auto new_shape = UnsqueezeShape(constant->Shape(), axes);
    ctx.graph.ReshapeInitializer(input, new_shape);
    node.SetInput(i, input);
    return;
  }

  // Case 2: input is a Squeeze node with matching axes
  std::unique_ptr<api::NodeRef> inp_node = ctx.graph.GetNodeProducingOutput(input);
  if (inp_node != nullptr && inp_node->IsOp("Squeeze")) {
    const std::vector<std::string_view>& inp_node_inputs = inp_node->Inputs();
    std::optional<std::vector<int64_t>> squeeze_axes = std::nullopt;
    squeeze_axes = ReadFromAttrOrInput(ctx, *inp_node, "axes", /*inp_index*/ 1, /*opset*/ 13);
    if (squeeze_axes != std::nullopt && *squeeze_axes == axes) {
      // Remove the Squeeze node if possible
      if (consumers->comprehensive && consumers->nodes.size() == 0) {
        ctx.graph.RemoveNode(*inp_node);
        if (ctx.opset >= 13 && !ctx.graph.HasValueConsumers(inp_node_inputs[1])) {
          ctx.graph.RemoveInitializer(inp_node_inputs[1]);
        }
      }
      node.SetInput(i, inp_node_inputs[0]);
      return;
    }

    // Axes don't match. Fall through to Case 3.
  }

  // Case 3: Add an Unsqueeze node.
  auto unsqueeze_ptr = MakeSqueezeOrUnsqueeze(ctx.opset, ctx.graph, "Unsqueeze", input, axes);
  api::NodeRef& unsqueeze = *unsqueeze_ptr;
  std::string_view unsq_out = unsqueeze.Outputs()[0];
  ctx.graph.CopyValueInfo(input, unsq_out);
  ctx.graph.GetValueInfo(unsq_out)->UnsqueezeDims(axes);

  // The transpose optimizer attempts to complete all optimization in a single pass. Adding Unsqueeze ops to inputs
  // is one of the few operations that violates the normal traversal order. If the input to the new Unsqueeze is
  // a Transpose, optimize it here.
  if (inp_node != nullptr && inp_node->IsOp("Transpose")) {
    auto perm = GetPermAttrIfValid(*inp_node);
    if (perm != std::nullopt) {
      auto perm_inv = InvertPerm(*perm);
      std::vector<size_t> indices = {0};
      HandlerArgs args{ctx, *inp_node, unsqueeze, *perm, perm_inv, indices};
      const auto new_input = HelpHandleUnsqueeze(args, axes);
      // Use output from optimization (likely from pushed transpose)
      node.SetInput(i, new_input);
      return;
    }
  }

  node.SetInput(i, unsq_out);
}

// Replaces ith input to node with transposed value. Might create a new Transpose node, find an existing one,
// or transpose an initializer.
static void TransposeInput(api::GraphRef& graph, api::NodeRef& node, size_t i,
                           const std::vector<int64_t>& perm, const std::vector<int64_t>& perm_inv) {
  std::string_view input = node.Inputs()[i];
  // Remove this node as a consumer
  node.SetInput(i, "");
  // Only local constants are editable
  std::unique_ptr<api::TensorRef> constant = graph.GetLocalConstant(input);
  auto consumers = graph.GetValueConsumers(input);

  // Case 1: input is a constant with a known list of consumer nodes
  if (constant != nullptr && consumers->comprehensive) {
    if (consumers->nodes.size() > 0) {
      // Transpose the initializer. If there are existing consumers, add Transpose nodes to them using perm_inv
      // to counteract the effect. These Transposes will hopefully be optimized out later.
      auto transpose_inv_ptr = MakeTranspose(graph, input, perm_inv);
      api::NodeRef& transpose_inv = *transpose_inv_ptr;
      std::string_view transpose_out = transpose_inv.Outputs()[0];
      graph.CopyValueInfo(input, transpose_out);
      ReplaceValueReferences(consumers->nodes, input, transpose_out);
    }
    graph.TransposeInitializer(input, perm);
    node.SetInput(i, input);
    return;
  }

  // Case 2: input is a Transpose node
  std::unique_ptr<api::NodeRef> inp_node = graph.GetNodeProducingOutput(input);
  if (inp_node != nullptr && inp_node->IsOp("Transpose")) {
    std::optional<std::vector<int64_t>> perm2 = GetPermAttrIfValid(*inp_node);
    if (perm2 != std::nullopt && perm2->size() == perm.size()) {
      // If they cancel, use pre_transpose_value and remove Transpose if possible.
      if (*perm2 == perm_inv) {
        std::string_view pre_transpose_value = inp_node->Inputs()[0];
        if (consumers->comprehensive && consumers->nodes.size() == 0) {
          graph.RemoveNode(*inp_node);
        }
        node.SetInput(i, pre_transpose_value);
        return;
      }

      // Otherwise, compose the perm and Transpose pre_transpose_value. Cost is the same and we may be able to remove
      // the other Transpose.
      const std::vector<int64_t>& perm_combined = ComposePerm(*perm2, perm);
      auto transpose_ptr = MakeTranspose(graph, inp_node->Inputs()[0], perm_combined);
      api::NodeRef& transpose = *transpose_ptr;
      std::string_view transpose_out = transpose.Outputs()[0];
      graph.CopyValueInfo(input, transpose_out);
      graph.GetValueInfo(transpose_out)->PermuteDims(perm);
      if (consumers->comprehensive && consumers->nodes.size() == 0) {
        graph.RemoveNode(*inp_node);
      }
      node.SetInput(i, transpose_out);
      return;
    }
  }
  
  // Case 3: A Transpose op might already exist
  for (size_t j = 0; j < consumers->nodes.size(); ++j) {
    api::NodeRef& consumer = *consumers->nodes[j];
    if (consumer.IsOp("Transpose") && GetPermAttrIfValid(consumer) == perm) {
      node.SetInput(i, consumer.Outputs()[0]);
      return;
    }
  }

  // Case 4: Add a new Transpose op
  auto transpose_ptr = MakeTranspose(graph, input, perm);
  api::NodeRef& transpose = *transpose_ptr;
  std::string_view transpose_out = transpose.Outputs()[0];
  graph.CopyValueInfo(input, transpose_out);
  graph.GetValueInfo(transpose_out)->PermuteDims(perm);
  node.SetInput(i, transpose_out);
}

// Unsqueezes inputs of node to have uniform rank. Returns false if input ranks are unknown or exceed the target rank.
static bool NormalizeInputRanks(OptimizerCtx ctx, api::NodeRef& node, size_t target_rank, 
                                const std::vector<size_t>& input_indices) {
  auto inputs = node.Inputs();

  // Get and validate input ranks
  std::vector<size_t> ranks;
  ranks.reserve(input_indices.size());
  for (size_t i : input_indices) {
    std::optional<std::vector<int64_t>> shape = ctx.graph.GetValueInfo(inputs[i])->Shape();
    if (shape == std::nullopt || shape->size() > target_rank) {
      return false;
    }
    ranks.push_back(shape->size());
  }

  // Normalize ranks
  for (size_t k = 0; k < ranks.size(); ++k) {
    size_t rank_diff = target_rank - ranks[k];
    if (rank_diff > 0) {
      std::vector<int64_t> axes(rank_diff);
      for (size_t j = 0; j < rank_diff; ++j) {
        axes[j] = j;
      }
      UnsqueezeInput(ctx, node, input_indices[k], axes);
    }
  }
  return true;
}

// Transposes specified inputs according to perm.
// NOTE: if a Transpose is expected to be above an input to this node, use the inverse of its permutation to cancel it. 
static void TransposeInputs(OptimizerCtx& ctx, api::NodeRef& node, const std::vector<int64_t>& perm,
                            const std::vector<size_t>& input_indices) {
  auto perm_inv = InvertPerm(perm);
  for (size_t j : input_indices) {
    TransposeInput(ctx.graph, node, j, perm, perm_inv);
  }
}

inline static void TransposeFirstInput(OptimizerCtx& ctx, api::NodeRef& node, const std::vector<int64_t>& perm) {
  std::vector<size_t> indices {0};
  TransposeInputs(ctx, node, perm, indices);
}

// Inserts a Transpose op on the ith output of a node. Returns the new, transposed output.
// Updates shape information assuming that the output from the node will have a transposed shape (using perm_inv)
// but the overall (returned) output will match the initial shape.
static std::string_view TransposeOutput(api::GraphRef& graph, api::NodeRef& node, size_t i,
                                        const std::vector<int64_t>& perm,
                                        const std::vector<int64_t>& perm_inv) {
  // Make transpose without input initially, then add it to avoid cyclic reference.

  // X -> Node -> Y,   Transpose
  auto transpose = MakeTranspose(graph, "", perm);

  // X -> Node -> *Y',   Transpose -> Y      *shape/dtype not set
  graph.MoveOutput(node, i, *transpose, 0);
  std::string_view new_output = node.Outputs()[i];

  // X -> Node -> *Y',   Y' -> Transpose -> Y      *shape/dtype not set
  transpose->SetInput(0, new_output);

  // Copy shape info from Y back to Y' and update it.
  std::string_view old_output = transpose->Outputs()[0];
  graph.CopyValueInfo(old_output, new_output);
  graph.GetValueInfo(new_output)->PermuteDims(perm_inv);
  return old_output;
}

// Inserts a Transpose op on all node outputs and updates the shapes of the node outputs. Skips if perm is identity.
// See TransposeOutput for details on shape updates.
static void TransposeOutputs(OptimizerCtx& ctx, api::NodeRef& node, const std::vector<int64_t>& perm) {
  if (IsIdentityPerm(perm)) {
    return;
  }
  auto perm_inv = InvertPerm(perm);
  for (size_t j = 0; j < node.Outputs().size(); ++j) {
    TransposeOutput(ctx.graph, node, j, perm, perm_inv);
  }
}

/////// </Core Helpers> ///////

/////// <Optimization Heuristics> ///////
// Tools to determine whether a transpose should be pushed
// When a node has multiple inputs, pushing a transpose from one can create more transposes on the other inputs.
// Generally, we push a transpose if the total number of transposes above the node will strictly decrease.
// To favor transposing smaller tensors, we actually try to minimize the total number of transposed dimensions = the
// total number of non-trivial (value != 1) dimensions involved in transposes.
//
// This rank estimation is used instead of the product of the dimensions for two reasons: Rank is almost always
// known statically, and the tensors involved in cost comparisons are almost always broadcastable to each other,
// meaning a count of non-trivial dimensions is sufficient for comparison.
//
// A rank of 5 is used if rank cannot be determined since 5 is the largest rank we expect from something like a Conv
// and an unknown rank likely corresponds to a data-carrying (non-weight) tensor, which will be large.

// Given a value, returns the rank of the value excluding dimensions of value 1. Returns 5 if the rank is unknown. 
static int EstimateValueRank(api::GraphRef& graph, std::string_view input) {
  auto value_info = graph.GetValueInfo(input);
  std::optional<std::vector<int64_t>> shape = value_info->Shape();
  if (shape == std::nullopt) {
    return 5;
  }
  int rank = 0;
  for (int64_t d : *shape) {
    if (d != 1) {
      ++rank;
    }
  }
  return rank;
}

static const HandlerInfo* GetHandler(api::NodeRef& node, bool allow_extended_ops);

// Returns true if the provided transpose node is only consumed by nodes we can likely push it through.
static bool CanLikelyRemoveTranspose(api::GraphRef& graph, api::NodeRef& transpose) {
  auto consumers = graph.GetValueConsumers(transpose.Outputs()[0]);
  if (!consumers->comprehensive) {
    return false;
  }
  for (auto& node : consumers->nodes) {
    if (GetHandler(*node, true) == nullptr) {
      return false;
    }
  }
  return true;
}

// Estimates the cost of transposing an input. Currently uses rank heuristic. Negative if transpose is removed.
// Feel free to improve as needed.
static int EstimateTransposeValueCost(api::GraphRef& graph, std::string_view input,
                                      const std::vector<int64_t>& perm_inv) {
  // Case 1: Transposing constants probably costs nothing.
  std::unique_ptr<api::TensorRef> constant = graph.GetConstant(input);
  if (constant != nullptr) {
    return 0;
  }

  // Case 2: Transposing a transpose either cancels it or composes the permutations.
  std::unique_ptr<api::NodeRef> node = graph.GetNodeProducingOutput(input);
  if (node != nullptr && node->IsOp("Transpose")) {
    std::optional<std::vector<int64_t>> perm2 = GetPermAttrIfValid(*node);
    if (perm2 != std::nullopt) {
      if (*perm2 == perm_inv && CanLikelyRemoveTranspose(graph, *node)) {
        return -EstimateValueRank(graph, input);
      } else {
        return 0;
      }
    }
  }

  // Case 3: We will likely need to add a transpose.
  return EstimateValueRank(graph, input);
}

// Estimates total cost of transposing a node's inputs. Negative if transposing is beneficial.
static int EstimateTransposeInputsCost(api::GraphRef& graph, api::NodeRef& node, const std::vector<int64_t>& perm_inv,
                                       const std::vector<size_t>& input_indices) {
  auto inputs = node.Inputs();
  int cost = 0;
  for (size_t j : input_indices) {
    cost += EstimateTransposeValueCost(graph, inputs[j], perm_inv);
  }
  return cost;
}

/////// </Optimization Heuristics> ///////

/////// <Handlers> ///////
// Op-specific optimization code. Handlers are called on nodes of a given optype with at least one Transpose as input.
// Handlers are responsible for determining if optimization should occur and performing it. They return a bool
// indicating whether the graph was modified.
//
// When making handlers, there are some things to be careful of:
//   - Ops can have multiple opsets. Check the model opset to determine the right spec. The opset is always within
//     the optimizers min/max opset range. The handler_ctx.opset is the model opset, not the op opset. Round down to
//     nearest supported opset to get op opset
//   - Read the full spec and watch out for optional inputs, attributes, etc.
//   - Shapes (ValueInfo) must be kept up-to-date on all values
//   - Add tests for the op (transpose_optimizer_test.cc)
//   - Return false if and only if no changes have been made to the graph. Do all checks up front before starting
//     modifications

// Common helper for making handlers.
static bool HandleSimpleNodeBase(HandlerArgs& args, bool broadcast_inputs) {
  size_t rank = args.perm.size();
  if (broadcast_inputs && !NormalizeInputRanks(args.ctx, args.node, rank, args.transposible_inputs)) {
    return false;
  }
  TransposeInputs(args.ctx, args.node, args.perm_inv, args.transposible_inputs);
  TransposeOutputs(args.ctx, args.node, args.perm);
  return true;
}

// Transposes all inputs and all outputs
static bool HandleSimpleNode(HandlerArgs& args) {
  return HandleSimpleNodeBase(args, /*broadcast_inputs*/ false);
}

std::vector<size_t> AllInputs(OptimizerCtx& ctx, api::NodeRef& node) {
  (void)ctx;
  size_t num_inputs = node.Inputs().size();
  std::vector<size_t> indices(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    indices[i] = i;
  }
  return indices;
}

constexpr HandlerInfo simple_node_handler = {&AllInputs, &HandleSimpleNode};

std::vector<size_t> FirstInput(OptimizerCtx& ctx, api::NodeRef& node) {
  (void)ctx;
  (void)node;
  return {0};
}

constexpr HandlerInfo node_1_inp_handler = {&FirstInput, &HandleSimpleNode};

// Node with all inputs broadcastable
static bool HandleSimpleNodeBroadcast(HandlerArgs& args) {
  return HandleSimpleNodeBase(args, /*broadcast_inputs*/ true);
}

std::vector<size_t> NonScalarInputs(OptimizerCtx& ctx, api::NodeRef& node) {
  auto inputs = node.Inputs();
  std::vector<size_t> indices;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto info = ctx.graph.GetValueInfo(inputs[i]);
    auto shape = info->Shape();
    if (shape == std::nullopt || shape->size() != 0) {
      indices.push_back(i);
    }
  }
  return indices;
}

constexpr HandlerInfo broadcast_node_handler = {&NonScalarInputs, &HandleSimpleNodeBroadcast};

// Transposes all inputs and all outputs. Updates axis attribute.
static bool HandleSimpleNodeWithAxis(HandlerArgs& args, std::optional<int64_t> default_axis=std::nullopt) {
  size_t rank = args.perm.size();
  std::optional<int64_t> axis = args.node.GetAttributeInt("axis");
  if (axis == std::nullopt) {
    if (default_axis != std::nullopt) {
      axis = *default_axis;
    } else {
      return false;
    }
  }

  if (!NormalizeAndValidateAxis(*axis, rank)) {
    return false;
  }

  if (!HandleSimpleNodeBase(args, /*broadcast_inputs*/ false)) {
    return false;
  }

  args.node.SetAttributeInt("axis", args.perm[gsl::narrow_cast<size_t>(*axis)]);
  return true;
}

static bool HandleSplit(HandlerArgs& args) {
  return HandleSimpleNodeWithAxis(args, /*default_axis*/ 0);
}

constexpr HandlerInfo split_handler = {&FirstInput, &HandleSplit};

static bool HandleConcat(HandlerArgs& args) {
  return HandleSimpleNodeWithAxis(args);
}

constexpr HandlerInfo concat_handler = {&AllInputs, &HandleConcat};

// Handles Softmax, Hardmax, and LogSoftmax
static bool HandleSoftHardMax(HandlerArgs& args) {
  if (args.ctx.opset >= 13) {
    return HandleSimpleNodeWithAxis(args, /*default_axis*/ -1);
  }

  // In opset < 13, the input is coerced into 2D then expanded back after.
  // The 'axis' attribute is the division point of the coercion.
  size_t rank = args.perm.size();
  int64_t axis = args.node.GetAttributeIntDefault("axis", 1);
  if (!NormalizeAndValidateAxis(axis, rank)) {
    return false;
  }

  // We can optimize only if the transpose does not move axes across the boundary. (normally this is the case)
  for (size_t i = 0; i < rank; ++i) {
    size_t axis_size_t = gsl::narrow_cast<size_t>(axis);
    bool to_lhs = i < axis_size_t;
    bool from_lhs = args.perm[i] < axis;
    if (to_lhs != from_lhs) {
      return false;
    }
  }

  // No need to update axis.
  return HandleSimpleNode(args);
}

constexpr HandlerInfo soft_hard_max_handler = {&FirstInput, &HandleSoftHardMax};

static bool HandleShape(HandlerArgs& args) {
  // Shape(Transpose(x, perm)) => Gather(Shape(x), perm)
  TransposeInputs(args.ctx, args.node, args.perm_inv, args.transposible_inputs);
  size_t rank = args.perm.size();
  int64_t rank_int = gsl::narrow_cast<int64_t>(rank);

  std::vector<int64_t> new_perm;
  // For opset 15, Shape(Transpose(x, perm))[starts:stops] = Gather(Shape(x), perm[starts:stops])
  if (args.ctx.opset >= 15) {

    // Assign new_perm = perm[starts:stops]
    int64_t start = args.node.GetAttributeIntDefault("start", 0);
    int64_t end = args.node.GetAttributeIntDefault("end", rank_int);
    if (start < 0) {
      start += rank;
    }
    if (end < 0) {
      end += rank;
    }
    size_t start_idx = gsl::narrow_cast<size_t>(std::clamp<int64_t>(start, 0, rank_int));
    size_t end_idx = gsl::narrow_cast<size_t>(std::clamp<int64_t>(end, 0, rank_int));
    for (size_t i = start_idx; i < end_idx; ++i) {
      new_perm.push_back(args.perm[i]);
    }
    args.node.ClearAttribute("start");
    args.node.ClearAttribute("end");
  } else {
    new_perm = args.perm;
  }

  // Make new_perm initializer
  std::vector<int64_t> perm_shape {gsl::narrow_cast<int64_t>(new_perm.size())};
  std::string_view perm_const = AddInitializerInt64(args.ctx.graph, perm_shape, new_perm);

  // X -> Shape -> Y,   Gather
  std::vector<std::string_view> gather_inputs{"", perm_const};
  auto gather_ptr = args.ctx.graph.AddNode("Gather", gather_inputs, /*num_outputs*/ 1);
  api::NodeRef& gather = *gather_ptr;
  gather.SetAttributeInt("axis", 0);

  // X -> Shape -> Y',   Gather -> Y
  args.ctx.graph.MoveOutput(args.node, 0, gather, 0);
  std::string_view new_output = args.node.Outputs()[0];

  // X -> Shape -> Y',   Y' -> Gather -> Y
  gather.SetInput(0, new_output);

  // Fix shapes
  args.ctx.graph.CopyValueInfo(gather.Outputs()[0], new_output);
  if (new_perm.size() != rank) {
    // Output Y' from Shape may be larger if we removed start/end
    auto info = args.ctx.graph.GetValueInfo(new_output);
    std::vector<int64_t> new_shape{rank_int};
    info->SetShape(&new_shape);
  }
  return true;
}

constexpr HandlerInfo shape_handler = {&FirstInput, &HandleShape, /*transposes_outputs*/ false};

// Permutes a 1D node input by creating a new initializer or inserting a Gather op
void PermuteInput(api::GraphRef& graph, api::NodeRef& node, size_t i, const std::vector<int64_t>& perm) {
  size_t rank = perm.size();
  int64_t rank_int = gsl::narrow_cast<int64_t>(rank);

  std::string_view input = node.Inputs()[i];
  auto constant = graph.GetConstant(input);
  if (constant != nullptr) {
    auto shape = constant->Shape();
    if (shape.size() == 1 && (shape[0] == rank_int || shape[0] == 0)) {
      // Create new transposed initializer
      std::vector<uint8_t> data = constant->Data();
      std::vector<uint8_t> new_data(data.size());
      size_t bytes_per_val = data.size() / rank;

      uint8_t* dst = new_data.data();
      for (size_t j = 0; j < rank; ++j) {
        uint8_t* src = data.data() + perm[j] * bytes_per_val;
        for (size_t k = 0; k < bytes_per_val; ++k) {
          *dst++ = *src++;
        }
      }

      std::string_view new_initializer = graph.AddInitializer(constant->DType(), shape, new_data);
      node.SetInput(i, new_initializer);
      if (!graph.HasValueConsumers(input)) {
        graph.RemoveInitializer(input);
      }

      return;
    }
  }

  std::string_view gather_indices_const = AddInitializerInt64(graph, /*shape*/ {rank_int}, perm);
  std::vector<std::string_view> gather_inputs{input, gather_indices_const};
  auto gather_ptr = graph.AddNode("Gather", gather_inputs, /*num_outputs*/ 1);
  api::NodeRef& gather = *gather_ptr;
  std::string_view gather_output = gather.Outputs()[0];
  graph.CopyValueInfo(input, gather_output);
  gather.SetAttributeInt("axis", 0);
  node.SetInput(i, gather_output);
}

static bool HandleResize(HandlerArgs& args) {
  auto inputs = args.node.Inputs();
  int64_t rank_int = gsl::narrow_cast<int64_t>(args.perm.size());

  if (args.ctx.opset < 11) {
    PermuteInput(args.ctx.graph, args.node, 1, args.perm_inv);
  } else {
    if (inputs[1] != "") {
      std::vector<int64_t> double_perm_inv = args.perm_inv;
      double_perm_inv.reserve(2 * args.perm_inv.size());
      for (int64_t p : args.perm_inv) {
        double_perm_inv.push_back(p + rank_int);
      }
      PermuteInput(args.ctx.graph, args.node, 1, double_perm_inv);
    }
    for (size_t i = 2; i < inputs.size(); ++i) {
      if (inputs[i] != "") {
        PermuteInput(args.ctx.graph, args.node, i, args.perm_inv);
      }
    }
  }

  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  TransposeOutputs(args.ctx, args.node, args.perm);

  return true;
}

constexpr HandlerInfo resize_handler = {&FirstInput, &HandleResize};

static bool HandlePad(HandlerArgs& args) {
  size_t rank = args.perm.size();
  int64_t opset = args.ctx.opset;

  // Pads length is twice perm length (all starts then all ends).
  std::vector<int64_t> pads_perm = args.perm_inv;
  pads_perm.reserve(rank * 2);
  for (int64_t p : args.perm_inv) {
    pads_perm.push_back(p + rank);
  }

  if (opset < 11) {
    // Permute pads attribute
    std::optional<std::vector<int64_t>> pads = args.node.GetAttributeInts("pads");
    if (pads == std::nullopt || pads->size() != rank * 2) {
      return false;
    }

    std::vector<int64_t> new_pads;
    new_pads.reserve(pads->size());
    for (int64_t i : pads_perm) {
      new_pads.push_back((*pads)[gsl::narrow_cast<size_t>(i)]);
    }

    args.node.SetAttributeInts("pads", new_pads);
  } else {
    // Permute pads input
    PermuteInput(args.ctx.graph, args.node, 1, pads_perm);
  }

  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  TransposeOutputs(args.ctx, args.node, args.perm);

  return true;
}

constexpr HandlerInfo pad_handler = {&FirstInput, &HandlePad};

static bool HandleReduceOp(HandlerArgs& args) {
  int64_t keepdims = args.node.GetAttributeIntDefault("keepdims", 1);

  std::optional<std::vector<int64_t>> axes = args.node.GetAttributeInts("axes");

  // Permutation for output transpose depends on which axes are removed
  std::vector<int64_t> out_perm;

  if (axes == std::nullopt) {
    // Default case is reduce over all dims
    if (keepdims == 0) {
      // Output rank is 0.
      out_perm = {};
    } else {
      out_perm = args.perm;
    }

  } else {

    if (!NormalizeAndValidateAxes(*axes, args.perm.size())) {
      return false;
    }

    std::vector<int64_t> new_axes = SortedAxesForTransposedInput(*axes, args.perm);
    args.node.SetAttributeInts("axes", new_axes);

    if (keepdims == 0) {
      out_perm = SqueezePerm(new_axes, args.perm);
    } else {
      out_perm = args.perm;
    }
  }

  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  TransposeOutputs(args.ctx, args.node, out_perm);

  return true;
}

constexpr HandlerInfo reduce_op_handler = {&FirstInput, &HandleReduceOp};

static bool HandleReduceSum(HandlerArgs& args) {
  if (args.ctx.opset < 13) {
    return HandleReduceOp(args);
  }

  bool keepdims = args.node.GetAttributeIntDefault("keepdims", 1) != 0;

  const std::vector<std::string_view>& inputs = args.node.Inputs();
  std::unique_ptr<api::TensorRef> axes_const = nullptr;
  bool empty_axes = false;

  if (inputs.size() < 2 || inputs[1] == "") {
    empty_axes = true;
  } else {
    axes_const = args.ctx.graph.GetConstant(inputs[1]);
    if (axes_const != nullptr && axes_const->NumElements() == 0) {
      empty_axes = true;
    }
  }

  // Case 1: Empty axes (either a no-op or reduce all axes)
  if (empty_axes) {
    bool noop_with_empty_axes = args.node.GetAttributeIntDefault("noop_with_empty_axes", 0) != 0;
    TransposeFirstInput(args.ctx, args.node, args.perm_inv);

    if (noop_with_empty_axes || keepdims) {
      // Original rank is maintained
      TransposeOutputs(args.ctx, args.node, args.perm);
    }

    return true;
  }

  // Case 2: Non-const axes (can't optimize)
  if (axes_const == nullptr) {
    // Technically we can handle this with Gather if keepdims is true, but this case is extremely rare.
    return false;
  }

  // Case 3: Const axes
  auto axes = DataInt64(*axes_const);
  if (!NormalizeAndValidateAxes(axes, args.perm.size())) {
    return false;
  }

  std::vector<int64_t> new_axes = SortedAxesForTransposedInput(axes, args.perm);
  std::vector<int64_t> axes_shape{gsl::narrow_cast<int64_t>(new_axes.size())};
  std::string_view new_axes_const = AddInitializerInt64(args.ctx.graph, axes_shape, new_axes);
  std::string_view axes_inp = inputs[1];
  args.node.SetInput(1, new_axes_const);

  if (!args.ctx.graph.HasValueConsumers(axes_inp)) {
    args.ctx.graph.RemoveInitializer(axes_inp);
  }

  TransposeFirstInput(args.ctx, args.node, args.perm_inv);

  if (keepdims) {
    TransposeOutputs(args.ctx, args.node, args.perm);
  } else {
    std::vector<int64_t> new_perm = SqueezePerm(new_axes, args.perm);
    TransposeOutputs(args.ctx, args.node, new_perm);
  }

  return true;
}

constexpr HandlerInfo reduce_sum_handler = {&FirstInput, &HandleReduceSum};

static bool HandleSqueeze(HandlerArgs& args) {
  std::vector<int64_t> new_axes;

  auto axes = ReadFromAttrOrInput(args.ctx, args.node, "axes", /*inp_index*/ 1, /*opset*/ 13);

  // If Squeeze axes are unset, output rank is unknown and must be skipped. Invalid axes are skipped too.
  if (axes == std::nullopt || !NormalizeAndValidateAxes(*axes, args.perm.size())) {
    return false;
  }

  new_axes = SortedAxesForTransposedInput(*axes, args.perm);

  // Update axes
  if (args.ctx.opset < 13) {
    args.node.SetAttributeInts("axes", new_axes);
  } else {
    std::string_view axes_inp = args.node.Inputs()[1];
    std::vector<int64_t> axes_shape{gsl::narrow_cast<int64_t>(new_axes.size())};
    std::string_view new_axes_const = AddInitializerInt64(args.ctx.graph, axes_shape, new_axes);
    args.node.SetInput(1, new_axes_const);
    if (!args.ctx.graph.HasValueConsumers(axes_inp)) {
      args.ctx.graph.RemoveInitializer(axes_inp);
    }

  }

  // Transpose inputs/outputs
  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  std::vector<int64_t> new_perm = SqueezePerm(new_axes, args.perm);
  TransposeOutputs(args.ctx, args.node, new_perm);

  return true;
}

constexpr HandlerInfo squeeze_handler = {&FirstInput, &HandleSqueeze};

// Pushes transpose through unsqueeze and returns final output. Helps UnsqueezeInput to push transposes.
// axes is the axes of the Unsqueeze node.
static std::string_view HelpHandleUnsqueeze(HandlerArgs& args, const std::vector<int64_t>& axes) {
  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  std::vector<int64_t> new_perm = UnsqueezePerm(axes, args.perm);
  return TransposeOutput(args.ctx.graph, args.node, 0, new_perm, InvertPerm(new_perm));
}

static bool HandleUnsqueeze(HandlerArgs& args) {
  auto axes = ReadFromAttrOrInput(args.ctx, args.node, "axes", /*inp_index*/ 1, /*opset*/ 13);

  if (axes == std::nullopt || !NormalizeAndValidateAxes(*axes, args.perm.size() + axes->size())) {
    return false;
  }

  // We will leave the axes unchanged and use them to determine how to transpose the output.
  HelpHandleUnsqueeze(args, *axes);
  return true;
}

constexpr HandlerInfo unsqueeze_handler = {&FirstInput, &HandleUnsqueeze};

static bool HandleQuantizeDequantizeLinear(HandlerArgs& args) {
  size_t rank = args.perm.size();

  if (args.ctx.opset >= 13) {
    // Update axis in Opset >= 13 if scale/zero_point are non-scalar
    auto inputs = args.node.Inputs();

    std::optional<std::vector<int64_t>> inp_shape = args.ctx.graph.GetValueInfo(inputs[1])->Shape();
    bool scalar_params = inp_shape != std::nullopt && inp_shape->size() == 0;

    if (!scalar_params) {
      int64_t axis = args.node.GetAttributeIntDefault("axis", 1);
      if (!NormalizeAndValidateAxis(axis, rank)) {
        return false;
      }

      args.node.SetAttributeInt("axis", args.perm[gsl::narrow_cast<size_t>(axis)]);
    }
  }

  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  TransposeOutputs(args.ctx, args.node, args.perm);

  return true;
}

constexpr HandlerInfo quantize_dequantize_linear_handler = {&FirstInput, &HandleQuantizeDequantizeLinear};

static bool HandleArgMinMax(HandlerArgs& args) {
  size_t rank = args.perm.size();

  int64_t keepdims = args.node.GetAttributeIntDefault("keepdims", 1);
  int64_t axis = args.node.GetAttributeIntDefault("axis", 0);
  if (!NormalizeAndValidateAxis(axis, rank)) {
    return false;
  }
  int64_t new_axis = args.perm[gsl::narrow_cast<size_t>(axis)];
  std::vector<int64_t> new_axes {new_axis};
  args.node.SetAttributeInt("axis", new_axis);

  TransposeInputs(args.ctx, args.node, args.perm_inv, args.transposible_inputs);
  if (keepdims != 0) {
    TransposeOutputs(args.ctx, args.node, args.perm);
  } else {
    TransposeOutputs(args.ctx, args.node, SqueezePerm(new_axes, args.perm));
  }
  return true;
}

constexpr HandlerInfo arg_min_max_handler = {&FirstInput, &HandleArgMinMax};

// Creates an int32 or int64 initializer and returns the name (Slice supports int64 or int32 axes)
static std::string_view AddIntInitializerMatchingDtype(api::GraphRef& graph, std::vector<int64_t> values,
                                                       api::DataType dtype) {
  std::vector<int64_t> shape{gsl::narrow_cast<int64_t>(values.size())};

  if (dtype == api::DataType::INT32) {
    std::vector<int32_t> values_int32;
    values_int32.reserve(values.size());
    for (int64_t v : values) {
      values_int32.push_back((int32_t)v);
    }

    return AddInitializerInt32(graph, shape, values_int32);
  }

  return AddInitializerInt64(graph, shape, values);
}

// Gets int data from an int32 or int64 tensor
static std::vector<int64_t> TensorIntData(api::TensorRef& tensor, api::DataType dtype) {
  if (dtype == api::DataType::INT32) {
    std::vector<int32_t> values_int32 = DataInt32(tensor);
    std::vector<int64_t> values;
    values.reserve(values_int32.size());
    for (int32_t v : values_int32) {
      values.push_back(gsl::narrow_cast<int64_t>(v));
    }

    return values;
  }

  return DataInt64(tensor);
}

static bool HandleSlice(HandlerArgs& args) {
  size_t rank = args.perm.size();

  if (args.ctx.opset < 10) {
    std::optional<std::vector<int64_t>> axes = args.node.GetAttributeInts("axes");

    if (axes == std::nullopt) {
      // When axes are not provided, [0, 1, ... len(starts)] is used
      std::optional<std::vector<int64_t>> starts = args.node.GetAttributeInts("starts");
      size_t num_starts = starts->size();
      axes = std::vector<int64_t>();
      axes->reserve(num_starts);
      for (size_t i = 0; i < num_starts; ++i) {
        axes->push_back(i);
      }
    }

    if (!NormalizeAndValidateAxes(*axes, rank)) {
      return false;
    }

    std::vector<int64_t> new_axes = AxesForTransposedInput(*axes, args.perm);
    args.node.SetAttributeInts("axes", new_axes);
    TransposeFirstInput(args.ctx, args.node, args.perm_inv);
    TransposeOutputs(args.ctx, args.node, args.perm);
    return true;
  }

  std::vector<std::string_view> inputs = args.node.Inputs();
  std::vector<int64_t> new_axes;

  // Inputs are: data, starts, ends, [axes, steps]. NOTE: axes can be int64 or int32
  if (inputs.size() < 4 || inputs[3] == "") {
    // Case 1: Axes is missing. Compute using length of starts.
    auto starts_value_info = args.ctx.graph.GetValueInfo(inputs[1]);
    const std::optional<std::vector<int64_t>> starts_shape = starts_value_info->Shape();
    api::DataType int_dtype = starts_value_info->DType();

    if (starts_shape == std::nullopt || starts_shape->size() != 1 || (*starts_shape)[0] < 0) {
      return false;
    }

    size_t ndims = gsl::narrow_cast<size_t>((*starts_shape)[0]);
    new_axes.reserve(ndims);
    for (size_t i = 0; i < ndims; ++i) {
      new_axes.push_back(args.perm[i]);
    }

    std::string_view new_axes_const = AddIntInitializerMatchingDtype(args.ctx.graph, new_axes, int_dtype);
    args.node.SetInput(3, new_axes_const);

  } else {
    // Case 2: Axes input provided. Update if constant.
    std::string_view axes_inp = inputs[3];
    std::unique_ptr<api::TensorRef> axes_const = args.ctx.graph.GetConstant(axes_inp);
    if (axes_const == nullptr) {
      return false;
    }

    api::DataType int_dtype = axes_const->DType();
    auto axes = TensorIntData(*axes_const, int_dtype);
    if (!NormalizeAndValidateAxes(axes, rank)) {
      return false;
    }

    // Update axes but leave the order unchanged (don't sort them). Need to line up with starts/ends/steps
    new_axes = AxesForTransposedInput(axes, args.perm);
    std::vector<int64_t> axes_shape{gsl::narrow_cast<int64_t>(new_axes.size())};
    std::string_view new_axes_const = AddIntInitializerMatchingDtype(args.ctx.graph, new_axes, int_dtype);
    args.node.SetInput(3, new_axes_const);
    if (!args.ctx.graph.HasValueConsumers(axes_inp)) {
      args.ctx.graph.RemoveInitializer(axes_inp);
    }
  }
  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  TransposeOutputs(args.ctx, args.node, args.perm);
  return true;
}

constexpr HandlerInfo slice_handler = {&FirstInput, &HandleSlice};

static bool HandleTile(HandlerArgs& args) {
  size_t rank = args.perm.size();
  std::vector<int64_t> perm_shape{gsl::narrow_cast<int64_t>(rank)};

  std::string_view repeats_inp = args.node.Inputs()[1];
  std::unique_ptr<api::TensorRef> repeats_const = args.ctx.graph.GetConstant(repeats_inp);
  if (repeats_const != nullptr) {
    // Case 1: Repeats is constant. Shuffle order.
    const std::vector<int64_t>& repeats = DataInt64(*repeats_const);
    std::vector<int64_t> new_repeats;
    new_repeats.reserve(rank);
    for (int64_t p : args.perm_inv) {
      new_repeats.push_back(repeats[gsl::narrow_cast<size_t>(p)]);
    }

    std::string_view new_repeats_const = AddInitializerInt64(args.ctx.graph, perm_shape, new_repeats);
    args.node.SetInput(1, new_repeats_const);
    if (!args.ctx.graph.HasValueConsumers(repeats_inp)) {
      args.ctx.graph.RemoveInitializer(repeats_inp);
    }

  } else {
    // Case 2: Repeats is computed. Insert Gather node.
    std::string_view perm_inv_const = AddInitializerInt64(args.ctx.graph, perm_shape, args.perm_inv);
    std::vector<std::string_view> gather_inputs {repeats_inp, perm_inv_const};
    auto gather_node_ptr = args.ctx.graph.AddNode("Gather", gather_inputs, /*num_outputs*/ 1);
    api::NodeRef& gather_node = *gather_node_ptr;
    std::string_view gather_output = gather_node.Outputs()[0];
    args.ctx.graph.CopyValueInfo(repeats_inp, gather_output);
    args.node.SetInput(1, gather_output);
  }

  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  TransposeOutputs(args.ctx, args.node, args.perm);
  return true;
}

constexpr HandlerInfo tile_handler = {&FirstInput, &HandleTile};

static bool HandleTranspose(HandlerArgs& args) {
  // In this handler a transpose leads to another transpose. "transpose" if the 1st and "node" is the 2nd.

  std::optional<std::vector<int64_t>> node_perm = GetPermAttrIfValid(args.node);
  if (node_perm == std::nullopt || node_perm->size() != args.perm.size()) {
    return false;
  }

  // Input to 1st transpose
  std::string_view transpose_input = args.transpose.Inputs()[0];
  // Output of 2nd transpose
  std::string_view node_output = args.node.Outputs()[0];

  if (args.perm_inv == *node_perm) {
    // Case 1: Permutations cancel.
    auto consumers = args.ctx.graph.GetValueConsumers(args.node.Outputs()[0]);
    if (consumers->comprehensive) {
      // If possible, replace references to output of 2nd transpose with input to 1st
      ReplaceValueReferences(consumers->nodes, node_output, transpose_input);
    } else {
      // Otherwise, (ex: 2nd transpose is a graph output, a reasonably common case) the output name of the 2nd
      // transpose must be maintained. Attempt to move the output directly to the 1st transpose's parent.
      auto transpose_inp_consumers = args.ctx.graph.GetValueConsumers(transpose_input);
      std::unique_ptr<api::NodeRef> transpose_inp_node = args.ctx.graph.GetNodeProducingOutput(transpose_input);

      if (transpose_inp_node != nullptr && transpose_inp_consumers->comprehensive) {
        // Will move output to parent. First replace parent references with name of 2nd transpose output.
        args.node.SetInput(0, "");
        ReplaceValueReferences(transpose_inp_consumers->nodes, transpose_input, node_output);
        const std::vector<std::string_view>& transpose_inp_outputs = transpose_inp_node->Outputs();

        // Find index of output from parent node
        size_t i;
        for (i = 0; i < transpose_inp_outputs.size(); ++i) {
          if (transpose_inp_outputs[i] == transpose_input) break;
        }

        // Move 2nd transpose output (possible graph output) over top of it.
        args.ctx.graph.MoveOutput(args.node, 0, *transpose_inp_node, i);
      } else {
        // Worst-case scenario: Both parent output and 2nd transpose output cannot be removed (both graph outputs)
        // despite computing the same value. Use an Identity op instead.
        std::vector<std::string_view> single_empty_input {""};
        auto identity_ptr = args.ctx.graph.AddNode("Identity", single_empty_input, /*num_outputs*/ 1);
        api::NodeRef& identity = *identity_ptr;
        args.ctx.graph.MoveOutput(args.node, 0, identity, 0);
        identity.SetInput(0, transpose_input);
      }
    }

    // In any case, the 2nd transpose can be removed.
    args.ctx.graph.RemoveNode(args.node);
  } else {
    // Case 2: Permutations don't cancel. Compose permutations.
    std::vector<int64_t> new_perm = ComposePerm(args.perm, *node_perm);
    args.node.SetAttributeInts("perm", new_perm);
    args.node.SetInput(0, transpose_input);
  }

  // 2nd transpose no longer references 1st. Remove 2nd if possible.
  if (!args.ctx.graph.HasValueConsumers(args.transpose.Outputs()[0])) {
    args.ctx.graph.RemoveNode(args.transpose);
  }

  return true;
}

constexpr HandlerInfo transpose_handler = {&FirstInput, &HandleTranspose, /*transposes_outputs*/ false};

static bool HandleQLinearConcat(HandlerArgs& args) {
  return HandleSimpleNodeWithAxis(args);
}

std::vector<size_t> QLinearConcatInputs(OptimizerCtx& ctx, api::NodeRef& node) {
  (void)ctx;
  std::vector<size_t> indices;
  size_t num_inputs = node.Inputs().size();
  for (size_t i = 2; i < num_inputs; i += 3) {
    indices.push_back(i);
  }
  return indices;
}

constexpr HandlerInfo q_linear_concat_handler = {&QLinearConcatInputs, &HandleQLinearConcat};

static bool HandleQLinearBinaryOp(HandlerArgs& args) {
  return HandleSimpleNodeBase(args, /*broadcast_inputs*/ true);
}

std::vector<size_t> QLinearBinaryOpInputs(OptimizerCtx& ctx, api::NodeRef& node) {
  (void)ctx;
  (void)node;
  // Inputs are: [A, A_scale, A_zero_point, B, B_scale, B_zero_point, C_scale, C_zero_point],
  // we want [A, B].
  return {0, 3};
}

constexpr HandlerInfo q_linear_binary_op_handler = {&QLinearBinaryOpInputs, &HandleQLinearBinaryOp};

static bool HandleQLinearPoolOp(HandlerArgs& args) {
  // Swap between channel first/last variants. Only works for applicable values of perm.
  int64_t channels_last = args.node.GetAttributeIntDefault("channels_last", 1);
  size_t rank = args.perm.size();
  if (rank < 2) return false;
  auto p = ChannelLastToFirstPerm(rank);
  if ((!channels_last && args.perm == p) || (channels_last && args.perm_inv == p)) {
    args.node.SetAttributeInt("channels_last", 1 - channels_last);
    TransposeFirstInput(args.ctx, args.node, args.perm_inv);
    TransposeOutputs(args.ctx, args.node, args.perm);
    return true;
  }
  return false;
}

constexpr HandlerInfo q_linear_pool_op_handler = {&FirstInput, &HandleQLinearPoolOp};

static bool HandleMaxPool(HandlerArgs& args) {
  // Replace with NhwcMaxPool if possible. Only int8 and uint8 dtypes are supported by NhwcMaxPool.
  auto outputs = args.node.Outputs();
  if (outputs.size() == 2 && outputs[1] != "") {
    // Can't optimize if optional "indices" output is provided
    return false;
  }

  auto info = args.ctx.graph.GetValueInfo(outputs[0]);
  api::DataType dtype = info->DType();
  if (dtype != api::DataType::UINT8 && dtype != api::DataType::INT8) {
    return false;
  }

  size_t rank = args.perm.size();
  if (args.perm != ChannelLastToFirstPerm(rank)) {
    return false;
  }

  auto new_node = SwapNodeOpTypeAndDomain(args.ctx.graph, args.node, "NhwcMaxPool", "com.microsoft");
  new_node->ClearAttribute("storage_order");  // Only relevant for indices output. Prohibited for NhwcMaxPool.

  TransposeFirstInput(args.ctx, *new_node, args.perm_inv);
  TransposeOutputs(args.ctx, *new_node, args.perm);
  return true;
}

constexpr HandlerInfo max_pool_op_handler = {&FirstInput, &HandleMaxPool};

// TODO: check binary size of this and replace it with constexpr if large
static const std::unordered_map<std::string_view, const HandlerInfo&> handler_map {

  {"Cast", simple_node_handler}, {"Exp", simple_node_handler}, {"Identity", simple_node_handler},
  {"LeakyRelu", simple_node_handler}, {"Log", simple_node_handler}, {"Reciprocal", simple_node_handler},
  {"Relu", simple_node_handler}, {"Sigmoid", simple_node_handler}, {"Sqrt", simple_node_handler},
  {"Tanh", simple_node_handler}, {"Abs", simple_node_handler}, {"Not", simple_node_handler},
  {"Ceil", simple_node_handler}, {"Floor", simple_node_handler}, {"Neg", simple_node_handler},
  {"Erf", simple_node_handler}, {"HardSigmoid", simple_node_handler}, {"Round", simple_node_handler},
  {"IsInf", simple_node_handler}, {"IsNaN", simple_node_handler},
  {"Selu", simple_node_handler}, {"Shrink", simple_node_handler}, {"Sign", simple_node_handler},
  {"Softplus", simple_node_handler}, {"Softsign", simple_node_handler}, {"ThresholdedRelu", simple_node_handler},
  {"Celu", simple_node_handler}, {"HardSwish", simple_node_handler},

  {"Sin", simple_node_handler}, {"Cos", simple_node_handler}, {"Tan", simple_node_handler},
  {"Sinh", simple_node_handler}, {"Cosh", simple_node_handler}, {"Tanh", simple_node_handler},
  {"Asin", simple_node_handler}, {"Acos", simple_node_handler}, {"Atan", simple_node_handler},
  {"Asinh", simple_node_handler}, {"Acosh", simple_node_handler}, {"Atanh", simple_node_handler},

  {"Add", broadcast_node_handler}, {"Max", broadcast_node_handler}, {"Min", broadcast_node_handler},
  {"Mul", broadcast_node_handler}, {"Sub", broadcast_node_handler}, {"Div", broadcast_node_handler},
  {"And", broadcast_node_handler}, {"Or", broadcast_node_handler}, {"Xor", broadcast_node_handler},
  {"Mod", broadcast_node_handler}, {"PRelu", broadcast_node_handler}, {"BitShift", broadcast_node_handler},
  {"Equal", broadcast_node_handler}, {"Greater", broadcast_node_handler}, {"Less", broadcast_node_handler},
  {"GreaterOrEqual", broadcast_node_handler}, {"LessOrEqual", broadcast_node_handler},
  {"Mean", broadcast_node_handler}, {"Sum", broadcast_node_handler},  {"Pow", broadcast_node_handler},
  {"Where", broadcast_node_handler},

  {"Clip", node_1_inp_handler}, {"CastLike", node_1_inp_handler},

  {"Transpose", transpose_handler},
  {"Concat", concat_handler},
  {"Split", split_handler},
  {"Shape", shape_handler},
  {"Pad", pad_handler},
  {"Resize", resize_handler},
  {"ReduceSum", reduce_sum_handler},

  {"ReduceLogSum", reduce_op_handler}, {"ReduceLogSumExp", reduce_op_handler}, {"ReduceMax", reduce_op_handler},
  {"ReduceMean", reduce_op_handler}, {"ReduceMin", reduce_op_handler}, {"ReduceProd", reduce_op_handler},
  {"ReduceSumSquare", reduce_op_handler}, {"ReduceL1", reduce_op_handler}, {"ReduceL2", reduce_op_handler},

  {"ArgMin", arg_min_max_handler}, {"ArgMax", arg_min_max_handler},

  {"Squeeze", squeeze_handler},
  {"Unsqueeze", unsqueeze_handler},
  {"Slice", slice_handler},
  {"Tile", tile_handler},

  {"Softmax", soft_hard_max_handler}, {"Hardmax", soft_hard_max_handler}, {"LogSoftmax", soft_hard_max_handler},

  {"QuantizeLinear", quantize_dequantize_linear_handler}, {"DequantizeLinear", quantize_dequantize_linear_handler},
};

static const std::unordered_map<std::string_view, const HandlerInfo&> extended_handler_map{
  {"com.microsoft.QLinearReduceMean", reduce_op_handler},
  {"com.microsoft.QLinearSigmoid", node_1_inp_handler},
  {"com.microsoft.QLinearLeakyRelu", node_1_inp_handler},
  {"com.microsoft.QLinearConcat", q_linear_concat_handler},
  {"com.microsoft.QLinearAdd", q_linear_binary_op_handler},
  {"com.microsoft.QLinearMul", q_linear_binary_op_handler},
  {"com.microsoft.QLinearAveragePool", q_linear_pool_op_handler},
  {"com.microsoft.QLinearGlobalAveragePool", q_linear_pool_op_handler},
  {"MaxPool", max_pool_op_handler},
};

static const HandlerInfo* GetHandler(api::NodeRef& node, bool allow_extended_ops) {
  std::string key;
  auto domain = node.Domain();
  auto op_type = node.OpType();
  if (domain == "" || domain == "ai.onnx") {
    key = std::string(op_type);
  } else if (domain == "com.microsoft") {
    key = "com.microsoft." + std::string(op_type);
  } else {
    return nullptr;
  }

  auto match = handler_map.find(key);
  if (match != handler_map.end()) {
    return &match->second;
  } else if (allow_extended_ops) {
    match = extended_handler_map.find(key);
    if (match != extended_handler_map.end()) {
      return &match->second;
    }
  }
  return nullptr;
}

// Finds a handler for the node and estimates the cost of pushing a transpose. Does so if deemed beneficial.
bool ProcessTranspose(OptimizerCtx& ctx, api::NodeRef& transpose, api::NodeRef& node,
                      const std::vector<int64_t>& perm, size_t transpose_input_index,
                      const std::unordered_set<std::string>& outputs_leading_to_transpose) {
  const HandlerInfo* info = GetHandler(node, ctx.allow_extended_ops);
  if (info == nullptr) {
    return false;
  }

  std::vector<size_t> input_indices = info->transposible_inputs_fn(ctx, node);
  if (std::find(input_indices.begin(), input_indices.end(), transpose_input_index) == input_indices.end()) {
    // Transpose is not on an eligible input
    return false;
  }

  // Transpose and MaxPool should be optimized any time there is a transpose as input and a handler is available.
  // Inclusion of MaxPool is a hack because it has higher perf in the NHWC variant when supported.
  if (!ctx.skip_cost_check && !node.IsOp("Transpose") && !node.IsOp("MaxPool")) {
    // We require the input cost (number of transposes before the op) and the total cost to strictly decrease.
    // Strict decrease of the input cost ensures the optimization is stable, since the total cost decrease is just an
    // estimate (the transpose after the op may or may not cancel with a subsequent transpose). We don't want
    // repeated runs of the optimizer to have a transpose toggle between two inputs of a binary op.
    int cost = EstimateTransposeInputsCost(ctx.graph, node, perm, input_indices);

    if (cost < 0 && info->transposes_outputs) {
      // If the output will be transposed and won't ultimately cancel, factor in that cost.
      bool has_output_leading_to_transpose = false;
      auto outputs = node.Outputs();
      int out_cost = 0;
      // Having multiple outputs is rare. When it happens (Split), the total size of the outputs isn't much larger
      // than the largest input. Cost is rank currently, so just use the largest cost (rank) over all outputs.
      for (auto out : outputs) {
        out_cost = std::max(out_cost, EstimateValueRank(ctx.graph, out));
        if (outputs_leading_to_transpose.find(std::string(out)) != outputs_leading_to_transpose.end()) {
          has_output_leading_to_transpose = true;
        }
      }

      if (!has_output_leading_to_transpose) {
        cost += out_cost;
      }
    }

    if (cost >= 0) {
      return false;
    }
  }

  std::vector<int64_t> perm_inv = InvertPerm(perm);
  HandlerArgs args = {ctx, transpose, node, perm, perm_inv, input_indices};
  return info->handler_fn(args);
}

// Returns nullopt if graph opset is unsupported.
std::optional<OptimizerCtx> MakeOptimizerContext(api::GraphRef& graph, bool allow_extended_ops) {
  auto opset = graph.Opset("");
  if (opset == std::nullopt) {
    opset = graph.Opset("ai.onnx");
  }
  if (opset == std::nullopt || *opset > kMaxSupportedOpset || *opset < kMinSupportedOpset) {
    return std::nullopt;
  }
  if (allow_extended_ops) {
    auto ms_opset = graph.Opset("com.microsoft");
    if (ms_opset == std::nullopt || *ms_opset != 1) {
      allow_extended_ops = false;
    }
  }
  OptimizerCtx ctx{*opset, graph, allow_extended_ops, /*skip_cost_check*/ false};
  return ctx;
}

// Performs optimization. General algorithm: iterate over nodes in topological order. If a node has a transpose
// as input, push it through if the transpose cost does not increase and is likely to decrease.
bool OptimizeImpl(OptimizerCtx& ctx) {
  
  const std::vector<std::unique_ptr<api::NodeRef>> nodes = ctx.graph.Nodes();

  std::unordered_set<std::string> outputs_leading_to_transpose;

  // First iterate over sorted nodes in reverse order to find which outputs have paths through supported ops to
  // transpose nodes. We pull push transposes towards these outputs.
  for (size_t i = 0; i < nodes.size(); ++i) {

    api::NodeRef& node = *nodes[nodes.size() - i - 1];
    if (node.IsOp("Transpose")) {
      outputs_leading_to_transpose.insert(std::string(node.Inputs()[0]));
      continue;
    }

    auto outputs = node.Outputs();
    for (auto out : outputs) {
      if (outputs_leading_to_transpose.find(std::string(out)) != outputs_leading_to_transpose.end()) {
        const HandlerInfo* info = GetHandler(node, ctx.allow_extended_ops);
        // Determine if node is supported and produces transposed outputs when pushed.
        if (info != nullptr && info->transposes_outputs) {
          auto input_indices = info->transposible_inputs_fn(ctx, node);
          auto inputs = node.Inputs();
          for (size_t j : input_indices) {
            outputs_leading_to_transpose.insert(std::string(inputs[j]));
          }
        }
      }
    }
  }

  bool changed = false;
  // Optimize graph. Nodes will be modified during iteration, but nodes are never deleted before we reach them.
  // New transpose nodes are inserted, but always as an input to an existing node.
  for (size_t i = 0; i < nodes.size(); ++i) {
    api::NodeRef& node = *nodes[i];
    std::vector<std::string_view> inputs = node.Inputs();
    for (size_t j = 0; j < inputs.size(); ++j) {
      std::string_view inp = inputs[j];
      if (inp == "") {
        continue;
      }
      std::unique_ptr<api::NodeRef> transpose = ctx.graph.GetNodeProducingOutput(inp);
      if (transpose != nullptr && transpose->IsOp("Transpose")) {
        std::optional<std::vector<int64_t>> perm = GetPermAttrIfValid(*transpose);
        if (perm != std::nullopt) {
          std::vector<int64_t> perm_inv = InvertPerm(*perm);
          if (ProcessTranspose(ctx, *transpose, node, *perm, j, outputs_leading_to_transpose)) {
            changed = true;
            // Subsequent inputs may have changed and node may have been removed.
            break;
          }
        }
      }
    }
  }
  return changed;
}

bool Optimize(api::GraphRef& graph, bool allow_extended_ops) {
  auto ctx = MakeOptimizerContext(graph, allow_extended_ops);
  if (ctx == std::nullopt) {
    return false;
  }
  return OptimizeImpl(*ctx);
}

void WrapTransposesAroundNode(api::GraphRef& graph, api::NodeRef& node,
                              const std::vector<const std::vector<int64_t>*>& input_perms,
                              const std::vector<const std::vector<int64_t>*>& output_perms) {
  for (size_t i = 0; i < input_perms.size(); ++i) {
    const std::vector<int64_t>* input_perm = input_perms[i];
    if (input_perm != nullptr) {
      TransposeInput(graph, node, i, *input_perm, InvertPerm(*input_perm));
    }
  }
  for (size_t i = 0; i < output_perms.size(); ++i) {
    const std::vector<int64_t>* output_perm = output_perms[i];
    if (output_perm != nullptr) {
      TransposeOutput(graph, node, i, *output_perm, InvertPerm(*output_perm));
    }
  }
}

std::unique_ptr<api::NodeRef> SwapNodeOpTypeAndDomain(api::GraphRef& graph, api::NodeRef& node,
                                                      std::string_view op_type, std::string_view domain) {
  auto inputs = node.Inputs();
  auto outputs = node.Outputs();
  auto new_node = graph.AddNode(op_type, inputs, outputs.size(), domain);
  for (size_t j = 0; j < outputs.size(); ++j) {
    if (outputs[j] != "") {
      graph.MoveOutput(node, j, *new_node, j);
    }
  }
  new_node->CopyAttributes(node);
  graph.RemoveNode(node);
  return new_node;
}

}  // namespace onnx_layout_transformation
