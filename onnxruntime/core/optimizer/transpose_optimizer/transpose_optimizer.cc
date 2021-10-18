// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <gsl/gsl>

#include "api.h"

namespace onnx_layout_transformation {

struct OptimizerCtx {
  int64_t opset;
  api::Graph& graph;
  bool allow_extended_ops;
  bool skip_cost_check;
};

// Struct containing information for a handler functions. Decreases binary size and allows perm_inv to be precomputed.
struct HandlerArgs {
  OptimizerCtx& ctx;
  api::Node& transpose;
  api::Node& node;
  const std::vector<int64_t>& perm;
  const std::vector<int64_t>& perm_inv;
  size_t transpose_input_index;
};

typedef bool HandlerFunction(HandlerArgs& args);


/////// <Helper Utils> ///////
/* Small utilities for editing nodes and manipulating axes/permutations */

// Replaces all node inputs referencing old_value with references to new_value. Values must be non-empty strings.
// This is an alternative to using MoveOutput for cases when the values aren't node outputs (if one is an initializer,
// for example).
static void ReplaceValueReferences(std::vector<std::unique_ptr<api::Node>>& nodes,
                                   const std::string_view old_value, const std::string_view new_value) {
  for (std::unique_ptr<api::Node>& node : nodes) {
    const std::vector<std::string_view>& inputs = node->Inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i] == old_value) {
        node->SetInput(i, new_value);
      }
    }
  }
}

// Create a node with a single attribute of type vector<int64_t>
static std::unique_ptr<api::Node> MakeNode1Attr(api::Graph& graph, const std::string_view op_type,
                                                const std::string_view input, const std::string_view attr_name,
                                                const std::vector<int64_t>& attr_val) {
  std::vector<std::string_view> inputs;
  inputs.push_back(input);
  std::unique_ptr<api::Node> node = graph.AddNode(op_type, inputs);
  node->SetAttributeInts(attr_name, attr_val);
  return node;
}

// Creates a Transpose node. Does not update output ValueInfo.
static std::unique_ptr<api::Node> MakeTranspose(api::Graph& graph, const std::string_view input,
                                                const std::vector<int64_t>& perm) {
  return MakeNode1Attr(graph, "Transpose", input, "perm", perm);
}

// Creates a Squeeze/Unsqueeze node. Does not update output ValueInfo.
static std::unique_ptr<api::Node> MakeSqueezeOrUnsqueeze(int64_t opset, api::Graph& graph, 
                                                         const std::string_view op_type, const std::string_view input,
                                                         const std::vector<int64_t>& axes) {
  if (opset < 13) {
    return MakeNode1Attr(graph, op_type, input, "axes", axes);
  }

  std::vector<int64_t> axes_shape;
  axes_shape.push_back(axes.size());
  const std::string_view axes_initializer = graph.AddInitializerInt64(axes_shape, axes);

  std::vector<std::string_view> inputs;
  inputs.push_back(input);
  inputs.push_back(axes_initializer);

  return graph.AddNode(op_type, inputs);
}

// Returns whether perm is a valid permutation (contains each value from 0 to perm.size() - 1 exactly once)
bool IsValidPerm(const std::vector<int64_t>& perm) {
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
  for (int64_t p : perm2) {
    perm.push_back(perm1[gsl::narrow_cast<size_t>(p)]);
  }
  return perm;
}

// Returns true if perm[i] = i everywhere
static bool IsIdentityPerm(const std::vector<int64_t>& perm) {
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != (int64_t)i) {
      return false;
    }
  }
  return true;
}

// Computes permutation from channel last to channel first ordering of given rank. Nearly all handlers work for any
// permutation, but some are restricted. Also used for layout transformation. Rank must be >= 1.
static std::vector<int64_t> ChannelLastToFirstPerm(size_t rank) {
  std::vector<int64_t> p(rank);
  p[0] = 0;
  p[1] = rank - 1;
  for (size_t i = 2; i < rank; ++i) {
    p[i] = i - 1;
  }
  return p;
}

// Adds 1 dimensions to indices of shape corresponding to axes. Unsafe if axes has negative/duplicated entries.
static std::vector<int64_t> UnsqueezeShape(const std::vector<int64_t>& shape, const std::vector<int64_t>& axes) {
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
  for (size_t i = 0; i < new_rank; ++i) {
    if (!is_added_axis[i]) {
      axes_map.push_back(i);
    }
  }

  std::vector<int64_t> new_perm;
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
  size_t j = 0;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (!is_removed_axis[i]) {
      axes_map[i] = (int64_t)j++;
    }
  }

  // Add perm entries for retained axes.
  std::vector<int64_t> new_perm;
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
      new_axes.push_back((int64_t)a);
    }
  }

  return new_axes;
}

/////// </Helper Utils> ///////

/////// <Core Helpers> ///////
/* These helpers hide the most gnarly parts of the transpose optimizer. */


static const std::string_view HelpHandleUnsqueeze(HandlerArgs& args, const std::vector<int64_t>& axes);


// Replaces ith input to node with unsqueezed value. Might create a new Unsqueeze node, find an existing one,
// or reshape an initializer. Unsqueezing can be necessary before transposing inputs of a node that supports
// broadcasting.
static void UnsqueezeInput(OptimizerCtx& ctx, api::Node& node, size_t i, const std::vector<int64_t>& axes) {
  std::string_view input = node.Inputs()[i];
  // Remove this node as a consumer
  node.SetInput(i, "");

  std::unique_ptr<api::Tensor> constant = ctx.graph.GetConstant(input);
  auto consumers = ctx.graph.GetValueConsumers(input);

  // Case 1: input is a constant with a known list of consumer nodes
  if (constant != nullptr && consumers->comprehensive) {
    // We will reshape the initializer. If there are existing consumers, still reshape it but add Squeeze nodes
    // to counteract its effect. If they later Unsqueeze the same input, the Squeeze nodes will simply be deleted
    // (see Case 2).
    if (consumers->nodes.size() > 0) {
      auto squeeze_ptr = MakeSqueezeOrUnsqueeze(ctx.opset, ctx.graph, "Squeeze", input, axes);
      api::Node& squeeze = *squeeze_ptr;
      const std::string_view sq_out = squeeze.Outputs()[0];
      ctx.graph.CopyValueInfo(input, sq_out);
      ReplaceValueReferences(consumers->nodes, input, sq_out);
    }
    auto new_shape = UnsqueezeShape(constant->Shape(), axes);
    ctx.graph.ReshapeInitializer(input, new_shape);
    node.SetInput(i, input);
    return;
  }

  // Case 2: input is a Squeeze node with matching axes
  std::unique_ptr<api::Node> inp_node = ctx.graph.GetNodeProducingOutput(input);
  if (inp_node != nullptr && inp_node->IsOp("Squeeze")) {
    const std::vector<std::string_view>& inp_node_inputs = inp_node->Inputs();
    std::optional<std::vector<int64_t>> squeeze_axes = std::nullopt;
    if (ctx.opset < 13) {
      squeeze_axes = inp_node->GetAttributeInts("axes");
    } else if (inp_node_inputs.size() == 2) {
      std::unique_ptr<api::Tensor> axes_const = ctx.graph.GetConstant(inp_node_inputs[1]);
      if (axes_const != nullptr) {
        squeeze_axes = axes_const->DataInt64();
      }
    }

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
  auto squeeze_ptr = MakeSqueezeOrUnsqueeze(ctx.opset, ctx.graph, "Unsqueeze", input, axes);
  api::Node& squeeze = *squeeze_ptr;
  const std::string_view sq_out = squeeze.Outputs()[0];
  ctx.graph.CopyValueInfo(input, sq_out);
  ctx.graph.GetValueInfo(sq_out)->UnsqueezeDims(axes);

  // The transpose optimizer attempts to complete all optimization in a single pass. Adding Unsqueeze ops to inputs
  // is one of the few operations that violates the normal traversal order. If the input to the new Unsqueeze is
  // a Transpose, optimize it here.
  if (inp_node != nullptr && inp_node->IsOp("Transpose")) {
    auto perm = inp_node->GetAttributeInts("perm");
    if (perm != std::nullopt) {
      auto perm_inv = InvertPerm(*perm);
      HandlerArgs args{ctx, *inp_node, squeeze, *perm, perm_inv, 0};
      const auto new_input = HelpHandleUnsqueeze(args, axes);
      // Use output from optimization (likely from pushed transpose)
      node.SetInput(i, new_input);
      return;
    }
  }

  node.SetInput(i, sq_out);
}

// Replaces ith input to node with transposed value. Might create a new Transpose node, find an existing one,
// or transpose an initializer.
static void TransposeInput(OptimizerCtx& ctx, api::Node& node, size_t i,
                           const std::vector<int64_t>& perm, const std::vector<int64_t>& perm_inv) {
  std::string_view input = node.Inputs()[i];
  // Remove this node as a consumer
  node.SetInput(i, "");
  std::unique_ptr<api::Tensor> constant = ctx.graph.GetConstant(input);
  auto consumers = ctx.graph.GetValueConsumers(input);

  // Case 1: input is a constant with a known list of consumer nodes
  if (constant != nullptr && consumers->comprehensive) {
    if (consumers->nodes.size() > 0) {
      // Transpose the initializer. If there are existing consumers, add Transpose nodes to them using perm_inv
      // to counteract the effect. These Transposes will hopefully be optimized out later.
      auto transpose_inv_ptr = MakeTranspose(ctx.graph, input, perm_inv);
      api::Node& transpose_inv = *transpose_inv_ptr;
      const std::string_view transpose_out = transpose_inv.Outputs()[0];
      ctx.graph.CopyValueInfo(input, transpose_out);
      ReplaceValueReferences(consumers->nodes, input, transpose_out);
    }
    ctx.graph.TransposeInitializer(input, perm);
    node.SetInput(i, input);
    return;
  }

  // Case 2: input is a Transpose node
  std::unique_ptr<api::Node> inp_node = ctx.graph.GetNodeProducingOutput(input);
  if (inp_node != nullptr && inp_node->IsOp("Transpose")) {
    std::optional<std::vector<int64_t>> perm2 = inp_node->GetAttributeInts("perm");
    if (perm2 != std::nullopt) {
      // If they cancel, use pre_transpose_value and remove Transpose if possible.
      if (*perm2 == perm_inv) {
        std::string_view pre_transpose_value = inp_node->Inputs()[0];
        if (consumers->comprehensive && consumers->nodes.size() == 0) {
          ctx.graph.RemoveNode(*inp_node);
        }
        node.SetInput(i, pre_transpose_value);
        return;
      }

      // Otherwise, compose the perm and Transpose pre_transpose_value. Cost is the same and we may be able to remove
      // the other Transpose.
      const std::vector<int64_t>& perm_combined = ComposePerm(*perm2, perm);
      auto transpose_ptr = MakeTranspose(ctx.graph, inp_node->Inputs()[0], perm_combined);
      api::Node& transpose = *transpose_ptr;
      const std::string_view transpose_out = transpose.Outputs()[0];
      ctx.graph.CopyValueInfo(input, transpose_out);
      ctx.graph.GetValueInfo(transpose_out)->PermuteDims(perm);
      if (consumers->comprehensive && consumers->nodes.size() == 0) {
        ctx.graph.RemoveNode(*inp_node);
      }
      node.SetInput(i, transpose_out);
      return;
    }
  }
  
  // Case 3: A Transpose op might already exist
  for (size_t j = 0; j < consumers->nodes.size(); ++j) {
    api::Node& consumer = *consumers->nodes[j];
    if (consumer.IsOp("Transpose") && consumer.GetAttributeInts("perm") == perm) {
      node.SetInput(i, consumer.Outputs()[0]);
      return;
    }
  }

  // Case 4: Add a new Transpose op
  auto transpose_ptr = MakeTranspose(ctx.graph, input, perm);
  api::Node& transpose = *transpose_ptr;
  const std::string_view transpose_out = transpose.Outputs()[0];
  ctx.graph.CopyValueInfo(input, transpose_out);
  ctx.graph.GetValueInfo(transpose_out)->PermuteDims(perm);
  node.SetInput(i, transpose_out);
}

static std::unique_ptr<std::vector<size_t>> SetInputIndicesIfNull(api::Node& node,
                                                                  std::vector<size_t>*& input_indices) {
  std::unique_ptr<std::vector<size_t>> indices_storage;
  if (input_indices == nullptr) {
    size_t num_inputs = node.Inputs().size();
    indices_storage = std::make_unique<std::vector<size_t>>(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
      (*indices_storage)[i] = i;
    }
    input_indices = &(*indices_storage);
  }
  return indices_storage;
}

// Unsqueezes inputs of node to have uniform rank. Returns false if input ranks are unknown or exceed the target rank.
static bool NormalizeInputRanks(OptimizerCtx ctx, api::Node& node, size_t target_rank, 
                                std::vector<size_t>* input_indices = nullptr) {
  auto inputs = node.Inputs();
  auto indices_storage = SetInputIndicesIfNull(node, input_indices);

  // Get and validate input ranks
  std::vector<size_t> ranks;
  for (size_t i : *input_indices) {
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
      std::vector<int64_t> axes;
      for (size_t j = 0; j < rank_diff; ++j) {
        axes.push_back(j);
      }
      UnsqueezeInput(ctx, node, (*input_indices)[k], axes);
    }
  }
  return true;
}

// Transposes specified inputs (or all by default) according to perm.
// NOTE: if a Transpose is expected to be above an input to this node, use the inverse of its permutation to cancel it. 
static void TransposeInputs(OptimizerCtx& ctx, api::Node& node, const std::vector<int64_t>& perm,
                            std::vector<size_t>* input_indices = nullptr) {
  auto indices_storage = SetInputIndicesIfNull(node, input_indices);
  auto perm_inv = InvertPerm(perm);
  for (size_t j : *input_indices) {
    TransposeInput(ctx, node, j, perm, perm_inv);
  }
  return;
}

inline static void TransposeFirstInput(OptimizerCtx& ctx, api::Node& node, const std::vector<int64_t>& perm) {
  std::vector<size_t> indices {0};
  TransposeInputs(ctx, node, perm, &indices);
}

// Inserts a Transpose op on the ith output of a node. Returns the new, transposed output.
// Updates shape information assuming that the output from the node will have a transposed shape (using perm_inv)
// but the overall (returned) output will match the initial shape.
static const std::string_view TransposeOutput(OptimizerCtx& ctx, api::Node& node, size_t i,
                                              const std::vector<int64_t>& perm,
                                              const std::vector<int64_t>& perm_inv) {
  // Make transpose without input, then add it to avoid cyclic reference.
  auto transpose_ptr = MakeTranspose(ctx.graph, "", perm);
  api::Node& transpose = *transpose_ptr;
  ctx.graph.MoveOutput(node, i, transpose, 0);
  const std::string_view new_output = node.Outputs()[i];
  transpose.SetInput(0, new_output);
  const std::string_view old_output = transpose.Outputs()[0];
  ctx.graph.CopyValueInfo(old_output, new_output);
  ctx.graph.GetValueInfo(new_output)->PermuteDims(perm_inv);
  return old_output;
}

// Inserts a Transpose op on all node outputs and updates the shapes of the node outputs.
// See TransposeOutput for details on shape updates.
static void TransposeOutputs(OptimizerCtx& ctx, api::Node& node, const std::vector<int64_t>& perm) {
  if (IsIdentityPerm(perm)) {
    return;
  }
  auto perm_inv = InvertPerm(perm);
  for (size_t j = 0; j < node.Outputs().size(); ++j) {
    TransposeOutput(ctx, node, j, perm, perm_inv);
  }
}

/////// </Core Helpers> ///////

/////// <Optimization Hueristics> ///////
// Tools to determine whether a transpose should be pushed
// When a node has multiple inputs, pushing a transpose from one can create more transposes on the other inputs.
// Generally, we push a transpose if the total number of transposes above the node will strictly decrease.
// To favor transposing smaller tensors, we actually try to minimize the total number of transposed dimensions = the
// total number of non-trivial (value != 1) dimensions involved in transposes.

// Given a value, returns the rank of the value excluding dimensions of value 1. Returns 5 if the rank is unknown. 
static int EstimateValueRank(api::Graph& graph, const std::string_view input) {
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

static HandlerFunction* GetHandler(api::Node& node, bool allow_extended_ops);

// Returns true if the provided transpose node is only consumed by nodes we can likely push it through.
static bool CanLikelyRemoveTranspose(api::Graph& graph, api::Node& transpose) {
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
static int EstimateTransposeValueCost(api::Graph& graph, const std::string_view input,
                                      const std::vector<int64_t>& perm_inv) {
  // Case 1: Transposing constants probably costs nothing.
  std::unique_ptr<api::Tensor> constant = graph.GetConstant(input);
  if (constant != nullptr) {
    return 0;
  }

  // Case 2: Transposing a transpose either cancels it or composes the permutations.
  std::unique_ptr<api::Node> node = graph.GetNodeProducingOutput(input);
  if (node != nullptr && node->IsOp("Transpose")) {
    std::optional<std::vector<int64_t>> perm2 = node->GetAttributeInts("perm");
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
static int EstimateTransposeInputsCost(api::Graph& graph, api::Node& node, const std::vector<int64_t>& perm_inv,
                                       std::vector<size_t>* input_indices = nullptr) {
  auto indices_storage = SetInputIndicesIfNull(node, input_indices);
  auto inputs = node.Inputs();
  int cost = 0;
  for (size_t j : *input_indices) {
    cost += EstimateTransposeValueCost(graph, inputs[j], perm_inv);
  }
  return cost;
}

/////// </Optimization Hueristics> ///////

/////// <Handlers> ///////
// Op-specific optimization code. Handlers are called on nodes of a given optype with at least one Transpose as input.
// Handlers are responsible for determining if optimization should occur and performing it. They return a bool
// indicating whether the graph was modified.
//
// When making handlers, there are some things to be careful of:
//   - Ops can have multiple opsets. Check the model opset to determine the right spec. The opset is always within
//     the optimizers min/max opset range.
//   - Read the full spec and watch out for optional inputs, attributes, etc.
//   - Shapes (ValueInfo) must be kept up-to-date on all values
//   - Add tests for the op (transpose_optimizer_test.cc)
//   - Return false if and only if no changes have been made to the graph. Do all checks up front before starting
//     modifications
//   - If the op has multiple inputs to transpose, make sure EstimateTransposeInputsCost is < 0.

// Common helper for making handlers.
static bool HandleSimpleNodeBase(HandlerArgs& args, bool broadcast_inputs,
                                 std::vector<size_t>* input_indices = nullptr) {
  size_t rank = args.perm.size();
  if (!args.ctx.skip_cost_check && (input_indices == nullptr || input_indices->size() > 1) 
      && EstimateTransposeInputsCost(args.ctx.graph, args.node, args.perm, input_indices) >= 0) {
    return false;
  }
  if (broadcast_inputs && !NormalizeInputRanks(args.ctx, args.node, rank, input_indices)) {
    return false;
  }
  TransposeInputs(args.ctx, args.node, args.perm_inv, input_indices);
  TransposeOutputs(args.ctx, args.node, args.perm);
  return true;
}

// Node with all inputs broadcastable
static bool HandleSimpleNodeBroadcast(HandlerArgs& args) {
  return HandleSimpleNodeBase(args, /*broadcast_inputs*/ true);
}

// Transposes all inputs and all outputs
static bool HandleSimpleNode(HandlerArgs& args) {
  return HandleSimpleNodeBase(args, /*broadcast_inputs*/ false);
}

// Transposes 1st input and all outputs
static bool HandleSimpleNode1Inp(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  std::vector<size_t> input_indices {0};
  return HandleSimpleNodeBase(args, /*broadcast_inputs*/ false, &input_indices);
}

// Transposes all inputs and all outputs. Updates axis attribute.
static bool HandleSimpleNodeWithAxis(HandlerArgs& args, bool has_default, int64_t default_axis=0) {
  size_t rank = args.perm.size();
  std::optional<int64_t> axis = args.node.GetAttributeInt("axis");
  if (axis == std::nullopt) {
    if (has_default) {
      axis = default_axis;
    } else {
      return false;
    }
  }
  if (*axis < 0) {
    *axis += rank;
  }
  if (*axis < 0 || (uint64_t)*axis >= args.perm.size()) return false;
  if (!HandleSimpleNodeBase(args, /*broadcast_inputs*/ false)) {
    return false;
  }
  args.node.SetAttributeInt("axis", args.perm[(size_t)*axis]);
  return true;
}

static bool HandleSplit(HandlerArgs& args) {
  return HandleSimpleNodeWithAxis(args, /*has_default*/ true, /*default_axis*/ 0);
}

static bool HandleConcat(HandlerArgs& args) {
  return HandleSimpleNodeWithAxis(args, /*has_default*/ false);
}

// Handles Softmax, Hardmax, and LogSoftmax
static bool HandleSoftHardMax(HandlerArgs& args) {
  if (args.ctx.opset >= 13) {
    return HandleSimpleNodeWithAxis(args, /*has_default*/ true, /*default_axis*/ -1);
  }

  // In opset < 13, the input is coerced into 2D then expanded back after.
  // The 'axis' attribute is the division point of the coercion.
  size_t rank = args.perm.size();
  int64_t axis = args.node.GetAttributeIntDefault("axis", 1);
  // TODO: consolidate this?
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || (uint64_t)axis >= rank) return false;

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

static bool HandleShape(HandlerArgs& args) {
  // Shape(Transpose(x, perm)) => Gather(Shape(x), perm)
  TransposeInputs(args.ctx, args.node, args.perm_inv);
  size_t rank = args.perm.size();

  std::vector<int64_t> new_perm;
  // For opset 15, Shape(Transpose(x, perm))[starts:stops] = Gather(Shape(x), perm[starts:stops])
  if (args.ctx.opset >= 15) {

    // Assign new_perm = perm[starts:stops]
    int64_t start = args.node.GetAttributeIntDefault("start", 0);
    int64_t end = args.node.GetAttributeIntDefault("end", (int64_t)rank);
    if (start < 0) {
      start += rank;
    }
    if (end < 0) {
      end += rank;
    }
    size_t start_idx = (size_t)std::clamp(start, (int64_t)0, (int64_t)rank);
    size_t end_idx = (size_t)std::clamp(end, (int64_t)0, (int64_t)rank);
    for (size_t i = start_idx; i < end_idx; ++i) {
      new_perm.push_back(args.perm[i]);
    }
    args.node.ClearAttribute("start");
    args.node.ClearAttribute("end");
  } else {
    new_perm = args.perm;
  }

  // Make new_perm initializer
  std::vector<int64_t> perm_shape {(int64_t)new_perm.size()};
  const std::string_view perm_const = args.ctx.graph.AddInitializerInt64(perm_shape, new_perm);

  // Add the Gather node
  std::vector<std::string_view> gather_inputs;
  gather_inputs.push_back(""); // Avoid cyclic reference
  gather_inputs.push_back(perm_const);
  auto gather_ptr = args.ctx.graph.AddNode("Gather", gather_inputs);
  api::Node& gather = *gather_ptr;
  gather.SetAttributeInt("axis", 0);
  args.ctx.graph.MoveOutput(args.node, 0, gather, 0);
  const std::string_view new_output = args.node.Outputs()[0];
  gather.SetInput(0, new_output); // Assign Gather input

  // Fix shapes
  args.ctx.graph.CopyValueInfo(gather.Outputs()[0], new_output);
  if (new_perm.size() != rank) {
    auto info = args.ctx.graph.GetValueInfo(new_output);
    std::vector<int64_t> new_shape {(int64_t)rank};
    info->SetShape(&new_shape);
  }
  return true;
}

// Reorder pads according to perm. Pads length is twice perm length (all starts then all ends).
static std::vector<int64_t> PermutePads(const std::vector<int64_t>& pads, const std::vector<int64_t>& perm) {
  size_t rank = perm.size();
  std::vector<int64_t> new_pads;
  for (int64_t i : perm) {
    new_pads.push_back(pads[(size_t)i]);
  }
  for (int64_t i : perm) {
    new_pads.push_back(pads[(size_t)i + rank]);
  }
  return new_pads;
}

static bool HandlePad(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  size_t rank = args.perm.size();
  int64_t opset = args.ctx.opset;

  if (opset < 11) {
    std::optional<std::vector<int64_t>> pads = args.node.GetAttributeInts("pads");
    if (pads == std::nullopt) {
      return false;
    }
    std::vector<int64_t> new_pads = PermutePads(*pads, args.perm_inv);
    args.node.SetAttributeInts("pads", new_pads);
  }

  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  TransposeOutputs(args.ctx, args.node, args.perm);

  if (opset < 11) {
    return true;
  }

  std::string_view pads_input = args.node.Inputs()[1];
  std::vector<int64_t> pads_shape { (int64_t)rank * 2 };
  std::shared_ptr<api::Tensor> pads_const = args.ctx.graph.GetConstant(pads_input);

  // Case 1: pads is a constant
  if (pads_const != nullptr) {
    auto pads = pads_const->DataInt64();
    std::vector<int64_t> new_pads = PermutePads(pads, args.perm_inv);
    std::string_view new_pads_const = args.ctx.graph.AddInitializerInt64(pads_shape, new_pads);
    args.node.SetInput(1, new_pads_const);
    if (!args.ctx.graph.HasValueConsumers(pads_input)) {
      args.ctx.graph.RemoveInitializer(pads_input);
    }
    return true;
  }

  // Case 2: pads is computed. Use Gather to reorder pads.

  // Form indices using perm_inv twice
  std::vector<int64_t> gather_indices = args.perm_inv;
  for (int64_t p : args.perm_inv) {
    gather_indices.push_back(p + rank);
  }
  std::string_view gather_indices_const = args.ctx.graph.AddInitializerInt64(pads_shape, gather_indices);

  std::vector<std::string_view> gather_inputs{pads_input, gather_indices_const};
  auto gather_ptr = args.ctx.graph.AddNode("Gather", gather_inputs);
  api::Node& gather = *gather_ptr;
  std::string_view gather_output = gather.Outputs()[0];
  args.ctx.graph.CopyValueInfo(pads_input, gather_output);
  gather.SetAttributeInt("axis", 0);
  args.node.SetInput(1, gather_output);

  return true;
}

static bool HandleReduceOp(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  // TODO: compress this impl

  int64_t keepdims = args.node.GetAttributeIntDefault("keepdims", 1);

  std::optional<std::vector<int64_t>> axes = args.node.GetAttributeInts("axes");
  // TODO: (compress impl) empty axes
  if (axes == std::nullopt) {
    if (keepdims != 0) {
      TransposeFirstInput(args.ctx, args.node, args.perm_inv);
      TransposeOutputs(args.ctx, args.node, args.perm);
    } else {
      TransposeFirstInput(args.ctx, args.node, args.perm_inv);
    }
    return true;
  }
  if (!NormalizeAndValidateAxes(*axes, args.perm.size())) {
    return false;
  }
  std::vector<int64_t> new_axes = SortedAxesForTransposedInput(*axes, args.perm);
  args.node.SetAttributeInts("axes", new_axes);

  if (keepdims != 0) {
    TransposeFirstInput(args.ctx, args.node, args.perm_inv);
    TransposeOutputs(args.ctx, args.node, args.perm);
    return true;
  }
  else {
    TransposeFirstInput(args.ctx, args.node, args.perm_inv);
    std::vector<int64_t> new_perm = SqueezePerm(new_axes, args.perm);
    TransposeOutputs(args.ctx, args.node, new_perm);
    return true;
  }
}

static bool HandleReduceSum(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  // TODO: compress this impl

  if (args.ctx.opset < 13) {
    return HandleReduceOp(args);
  }

  int64_t keepdims = args.node.GetAttributeIntDefault("keepdims", 1);

  const std::vector<std::string_view>& inputs = args.node.Inputs();
  std::unique_ptr<api::Tensor> axes_const = nullptr;
  bool empty_axes = false;
  if (inputs.size() < 2 || inputs[1] == "") {
    empty_axes = true;
  } else {
    axes_const = args.ctx.graph.GetConstant(inputs[1]);
    if (axes_const != nullptr && axes_const->DataInt64().size() == 0) {
      empty_axes = true;
    }
  }
  if (empty_axes) {
    int64_t noop_with_empty_axes = args.node.GetAttributeIntDefault("noop_with_empty_axes", 0);
    if (noop_with_empty_axes != 0 || keepdims != 0) {
      TransposeFirstInput(args.ctx, args.node, args.perm_inv);
      TransposeOutputs(args.ctx, args.node, args.perm);
    } else {
      TransposeFirstInput(args.ctx, args.node, args.perm_inv);
    }
    return true;
  }
  if (axes_const == nullptr) {
    // TODO: technically we can handle this with Gather if keepdims is true
    return false;
  }

  auto axes = axes_const->DataInt64();
  if (!NormalizeAndValidateAxes(axes, args.perm.size())) {
    return false;
  }
  std::vector<int64_t> new_axes = SortedAxesForTransposedInput(axes, args.perm);
  std::vector<int64_t> axes_shape { (int64_t)new_axes.size() };
  std::string_view new_axes_const = args.ctx.graph.AddInitializerInt64(axes_shape, new_axes);
  std::string_view axes_inp = inputs[1];
  args.node.SetInput(1, new_axes_const);
  if (!args.ctx.graph.HasValueConsumers(axes_inp)) {
    args.ctx.graph.RemoveInitializer(axes_inp);
  }

  if (keepdims != 0) {
    TransposeFirstInput(args.ctx, args.node, args.perm_inv);
    TransposeOutputs(args.ctx, args.node, args.perm);
    return true;
  }
  else {
    TransposeFirstInput(args.ctx, args.node, args.perm_inv);
    std::vector<int64_t> new_perm = SqueezePerm(new_axes, args.perm);
    TransposeOutputs(args.ctx, args.node, new_perm);
    return true;
  }

  return true;
}

static bool HandleSqueeze(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  const std::vector<size_t> indices { 0 };
  std::vector<int64_t> new_axes;
  if (args.ctx.opset < 13) {
    std::optional<std::vector<int64_t>> axes = args.node.GetAttributeInts("axes");
    // TODO: (compress impl) empty axes
    if (axes == std::nullopt || !NormalizeAndValidateAxes(*axes, args.perm.size())) {
      return false;
    }
    new_axes = SortedAxesForTransposedInput(*axes, args.perm);
    args.node.SetAttributeInts("axes", new_axes);
  } else {
    const std::vector<std::string_view>& inputs = args.node.Inputs();
    if (inputs.size() < 2) {
      return false;
    }
    std::string_view axes_inp = inputs[1];
    if (axes_inp == "") {
      return false;
    }
    std::unique_ptr<api::Tensor> axes_const = args.ctx.graph.GetConstant(axes_inp);
    if (axes_const == nullptr) {
      return false;
    }
    auto axes = axes_const->DataInt64();
    if (!NormalizeAndValidateAxes(axes, args.perm.size())) {
      return false;
    }
    new_axes = SortedAxesForTransposedInput(axes, args.perm);
    std::vector<int64_t> axes_shape { (int64_t)new_axes.size() };
    std::string_view new_axes_const = args.ctx.graph.AddInitializerInt64(axes_shape, new_axes);
    args.node.SetInput(1, new_axes_const);
    if (!args.ctx.graph.HasValueConsumers(axes_inp)) {
      args.ctx.graph.RemoveInitializer(axes_inp);
    }
  }
  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  std::vector<int64_t> new_perm = SqueezePerm(new_axes, args.perm);
  TransposeOutputs(args.ctx, args.node, new_perm);
  return true;
}


static const std::string_view HelpHandleUnsqueeze(HandlerArgs& args, const std::vector<int64_t>& axes) {
  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  std::vector<int64_t> new_perm = UnsqueezePerm(axes, args.perm);
  return TransposeOutput(args.ctx, args.node, 0, new_perm, InvertPerm(new_perm));
}

static bool HandleUnsqueeze(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  std::vector<int64_t> axes;
  if (args.ctx.opset < 13) {
    std::optional<std::vector<int64_t>> axes_attr = args.node.GetAttributeInts("axes");
    // TODO: (compress impl) empty axes
    if (axes_attr == std::nullopt) {
      return false;
    }
    axes = *axes_attr;
  } else {
    const std::vector<std::string_view>& inputs = args.node.Inputs();
    std::unique_ptr<api::Tensor> axes_const = args.ctx.graph.GetConstant(inputs[1]);
    if (axes_const == nullptr) {
      return false;
    }
    axes = axes_const->DataInt64();
  }
  if (!NormalizeAndValidateAxes(axes, args.perm.size() + axes.size())) {
    return false;
  }
  HelpHandleUnsqueeze(args, axes);
  return true;
}


static bool HandleQuantizeDequantizeLinear(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  size_t rank = args.perm.size();

  if (args.ctx.opset >= 13) {
    auto inputs = args.node.Inputs();
    bool all_scalars = true;
    for (size_t i = 1; i < 3; ++i) {
      std::optional<std::vector<int64_t>> inp_shape = args.ctx.graph.GetValueInfo(inputs[i])->Shape();
      if (inp_shape == std::nullopt || inp_shape->size() > 0) {
        all_scalars = false;
      }
    }
    if (!all_scalars) {
      int64_t axis = args.node.GetAttributeIntDefault("axis", 1);
      if (axis < 0) {
        axis += rank;
      }
      if (axis < 0 || (size_t)axis >= args.perm.size()) {
        return false;
      }
      args.node.SetAttributeInt("axis", args.perm[(size_t)axis]);
    }
  }

  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  TransposeOutputs(args.ctx, args.node, args.perm);

  return true;
}

static bool HandleArgMinMax(HandlerArgs& args) {
  size_t rank = args.perm.size();

  int64_t keepdims = args.node.GetAttributeIntDefault("keepdims", 1);
  int64_t axis = args.node.GetAttributeIntDefault("axis", 0);
  if (axis < 0) {
    axis += (int64_t)rank;
  }
  int64_t new_axis = args.perm[(size_t)axis];
  std::vector<int64_t> new_axes {new_axis};
  args.node.SetAttributeInt("axis", new_axis);

  TransposeInputs(args.ctx, args.node, args.perm_inv);
  if (keepdims != 0) {
    TransposeOutputs(args.ctx, args.node, args.perm);
  } else {
    TransposeOutputs(args.ctx, args.node, SqueezePerm(new_axes, args.perm));
  }
  return true;
}

bool HandleSlice(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  size_t rank = args.perm.size();

  if (args.ctx.opset < 10) {
    std::optional<std::vector<int64_t>> axes = args.node.GetAttributeInts("axes");
    if (axes == std::nullopt) {
      std::optional<std::vector<int64_t>> starts = args.node.GetAttributeInts("starts");
      if (starts == std::nullopt) {
        // Invalid model. TODO: raise exception
        return false;
      }
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
  if (inputs.size() < 3) {
    return false;
  }
  std::vector<int64_t> new_axes;
  if (inputs.size() < 4 || inputs[3] == "") {
    const std::optional<std::vector<int64_t>> starts_shape = args.ctx.graph.GetValueInfo(inputs[1])->Shape();
    if (starts_shape == std::nullopt || starts_shape->size() != 1 || (*starts_shape)[0] < 0) {
      return false;
    }
    size_t ndims = (size_t)(*starts_shape)[0];
    for (size_t i = 0; i < ndims; ++i) {
      new_axes.push_back(args.perm[i]);
    }
    std::vector<int64_t> axes_shape { (int64_t)new_axes.size() };
    std::string_view new_axes_const = args.ctx.graph.AddInitializerInt64(axes_shape, new_axes);
    if (inputs.size() == 3) {
      args.node.AddInput(new_axes_const);
    } else {
      args.node.SetInput(3, new_axes_const);
    }
  } else {
    std::string_view axes_inp = inputs[3];
    std::unique_ptr<api::Tensor> axes_const = args.ctx.graph.GetConstant(axes_inp);
    if (axes_const == nullptr) {
      return false;
    }
    auto axes = axes_const->DataInt64();
    if (!NormalizeAndValidateAxes(axes, rank)) {
      return false;
    }
    new_axes = AxesForTransposedInput(axes, args.perm);
    std::vector<int64_t> axes_shape { (int64_t)new_axes.size() };
    std::string_view new_axes_const = args.ctx.graph.AddInitializerInt64(axes_shape, new_axes);
    args.node.SetInput(3, new_axes_const);
    if (!args.ctx.graph.HasValueConsumers(axes_inp)) {
      args.ctx.graph.RemoveInitializer(axes_inp);
    }
  }
  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  TransposeOutputs(args.ctx, args.node, args.perm);
  return true;
}

static bool HandleTile(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  size_t rank = args.perm.size();
  std::vector<int64_t> perm_shape {(int64_t)rank};

  std::string_view repeats_inp = args.node.Inputs()[1];
  std::unique_ptr<api::Tensor> repeats_const = args.ctx.graph.GetConstant(repeats_inp);
  if (repeats_const != nullptr) {
    const std::vector<int64_t>& repeats = repeats_const->DataInt64();
    std::vector<int64_t> new_repeats;
    for (int64_t p : args.perm_inv) {
      new_repeats.push_back(repeats[(size_t)p]);
    }
    std::string_view new_repeats_const = args.ctx.graph.AddInitializerInt64(perm_shape, new_repeats);
    args.node.SetInput(1, new_repeats_const);
    if (!args.ctx.graph.HasValueConsumers(repeats_inp)) {
      args.ctx.graph.RemoveInitializer(repeats_inp);
    }
  } else {
    std::string_view perm_inv_const = args.ctx.graph.AddInitializerInt64(perm_shape, args.perm_inv);
    std::vector<std::string_view> gather_inputs {repeats_inp, perm_inv_const};
    auto gather_node_ptr = args.ctx.graph.AddNode("Gather", gather_inputs);
    api::Node& gather_node = *gather_node_ptr;
    std::string_view gather_output = gather_node.Outputs()[0];
    args.ctx.graph.CopyValueInfo(repeats_inp, gather_output);
    args.node.SetInput(1, gather_output);
  }
  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  TransposeOutputs(args.ctx, args.node, args.perm);
  return true;
}

static bool HandleTranspose(HandlerArgs& args) {
  // Two cases: 1 perm match, 2 they don't

  // TODO: assert perm is valid
  std::optional<std::vector<int64_t>> node_perm = args.node.GetAttributeInts("perm");
  if (node_perm == std::nullopt) {
    return false;
  }

  const std::string_view transpose_input = args.transpose.Inputs()[0];
  const std::string_view node_output = args.node.Outputs()[0];
  if (args.perm_inv == *node_perm) {
    auto consumers = args.ctx.graph.GetValueConsumers(args.node.Outputs()[0]);
    if (consumers->comprehensive) {
      ReplaceValueReferences(consumers->nodes, node_output, transpose_input);
    }
    else {
      auto transpose_inp_consumers = args.ctx.graph.GetValueConsumers(transpose_input);
      std::unique_ptr<api::Node> transpose_inp_node = args.ctx.graph.GetNodeProducingOutput(transpose_input);
      if (transpose_inp_node != nullptr && transpose_inp_consumers->comprehensive) {
        args.node.SetInput(0, "");
        ReplaceValueReferences(transpose_inp_consumers->nodes, transpose_input, node_output);
        const std::vector<std::string_view>& transpose_inp_outputs = transpose_inp_node->Outputs();
        size_t i;
        for (i = 0; i < transpose_inp_outputs.size(); ++i) {
          if (transpose_inp_outputs[i] == transpose_input) break;
        }
        args.ctx.graph.MoveOutput(args.node, 0, *transpose_inp_node, i);
      } else {
        std::vector<std::string_view> single_empty_input {""};
        auto identity_ptr = args.ctx.graph.AddNode("Identity", single_empty_input);
        api::Node& identity = *identity_ptr;
        args.ctx.graph.MoveOutput(args.node, 0, identity, 0);
        identity.SetInput(0, transpose_input);
      }
    }
    args.ctx.graph.RemoveNode(args.node);
  } else {
    std::vector<int64_t> new_perm = ComposePerm(args.perm, *node_perm);
    args.node.SetAttributeInts("perm", new_perm);
    args.node.SetInput(0, transpose_input);
  }
  if (!args.ctx.graph.HasValueConsumers(args.transpose.Outputs()[0])) {
    args.ctx.graph.RemoveNode(args.transpose);
  }
  return true;
}

static bool HandleQLinearConcat(HandlerArgs& args) {
  size_t rank = args.perm.size();

  std::vector<size_t> indices;
  size_t num_inputs = args.node.Inputs().size();
  for (size_t i = 2; i < num_inputs; i += 3) {
    indices.push_back(i);
  }
  if (!args.ctx.skip_cost_check && EstimateTransposeInputsCost(args.ctx.graph, args.node, args.perm, &indices) >= 0) {
    return false;
  }

  std::optional<int64_t> axis = args.node.GetAttributeInt("axis");
  if (axis == std::nullopt) {
    return false;
  }
  if (*axis < 0) {
    *axis += rank;
  }
  if (*axis < 0 || (size_t)*axis >= rank) {
    return false;
  }
  args.node.SetAttributeInt("axis", args.perm[(size_t)*axis]);
  TransposeInputs(args.ctx, args.node, args.perm_inv, &indices);
  TransposeOutputs(args.ctx, args.node, args.perm);
  return true;
}

static bool HandleQLinearBinaryOp(HandlerArgs& args) {
  std::vector<size_t> indices { 0, 3 };
  return HandleSimpleNodeBase(args, /*broadcast_inputs*/ true, &indices);
}

static bool HandleQLinearPoolOp(HandlerArgs& args) {
  if (args.transpose_input_index != 0) return false;
  int64_t channels_last = args.node.GetAttributeIntDefault("channels_last", 1);
  size_t rank = args.perm.size();
  if (rank < 2) return false;
  // Channels last to first perm
  auto p = ChannelLastToFirstPerm(rank);
  if ((!channels_last && args.perm == p) || (channels_last && args.perm_inv == p)) {
    args.node.SetAttributeInt("channels_last", 1 - channels_last);
    TransposeFirstInput(args.ctx, args.node, args.perm_inv);
    TransposeOutputs(args.ctx, args.node, args.perm);
    return true;
  }
  return false;
}

static bool HandleMaxPool(HandlerArgs& args) {
  auto outputs = args.node.Outputs();
  if (outputs.size() == 2 && outputs[1] != "") {
    return false;
  }
  size_t rank = args.perm.size();
  if (args.perm != ChannelLastToFirstPerm(rank)) {
    return false;
  }
  auto inputs = args.node.Inputs();
  std::shared_ptr<api::Node> new_node = args.ctx.graph.AddNode("NhwcMaxPool", inputs, /*num_outputs*/ 1, "com.microsoft");
  new_node->CopyAttributes(args.node);
  new_node->ClearAttribute("storage_order");
  args.ctx.graph.MoveOutput(args.node, 0, *new_node, 0);
  args.ctx.graph.RemoveNode(args.node);
  TransposeFirstInput(args.ctx, *new_node, args.perm_inv);
  TransposeOutputs(args.ctx, *new_node, args.perm);
  return true;
}

static const std::unordered_map<std::string_view, HandlerFunction*> handler_map {

  {"Cast", &HandleSimpleNode}, {"Exp", &HandleSimpleNode}, {"Identity", &HandleSimpleNode},
  {"LeakyRelu", &HandleSimpleNode}, {"Log", &HandleSimpleNode}, {"Reciprocal", &HandleSimpleNode},
  {"Relu", &HandleSimpleNode}, {"Sigmoid", &HandleSimpleNode}, {"Sqrt", &HandleSimpleNode},
  {"Tanh", &HandleSimpleNode}, {"Abs", &HandleSimpleNode}, {"Ceil", &HandleSimpleNode}, {"Floor", &HandleSimpleNode},
  {"Erf", &HandleSimpleNode}, {"HardSigmoid", &HandleSimpleNode}, {"Round", &HandleSimpleNode},
  {"IsInf", &HandleSimpleNode}, {"IsNaN", &HandleSimpleNode}, {"Neg", &HandleSimpleNode}, {"Not", &HandleSimpleNode},
  {"Selu", &HandleSimpleNode}, {"Shrink", &HandleSimpleNode}, {"Sign", &HandleSimpleNode},
  {"Softplus", &HandleSimpleNode}, {"Softsign", &HandleSimpleNode}, {"ThresholdedRelu", &HandleSimpleNode},
  {"Celu", &HandleSimpleNode}, {"HardSwish", &HandleSimpleNode},

  {"Sin", &HandleSimpleNode}, {"Cos", &HandleSimpleNode}, {"Tan", &HandleSimpleNode},
  {"Sinh", &HandleSimpleNode}, {"Cosh", &HandleSimpleNode}, {"Tanh", &HandleSimpleNode},
  {"Asin", &HandleSimpleNode}, {"Acos", &HandleSimpleNode}, {"Atan", &HandleSimpleNode},
  {"Asinh", &HandleSimpleNode}, {"Acosh", &HandleSimpleNode}, {"Atanh", &HandleSimpleNode},

  {"Add", &HandleSimpleNodeBroadcast}, {"Max", &HandleSimpleNodeBroadcast}, {"Min", &HandleSimpleNodeBroadcast},
  {"Mul", &HandleSimpleNodeBroadcast}, {"Sub", &HandleSimpleNodeBroadcast}, {"Div", &HandleSimpleNodeBroadcast},
  {"And", &HandleSimpleNodeBroadcast}, {"Or", &HandleSimpleNodeBroadcast}, {"Xor", &HandleSimpleNodeBroadcast},
  {"Mod", &HandleSimpleNodeBroadcast}, {"PRelu", &HandleSimpleNodeBroadcast}, {"BitShift", &HandleSimpleNodeBroadcast},
  {"Equal", &HandleSimpleNodeBroadcast}, {"Greater", &HandleSimpleNodeBroadcast}, {"Less", &HandleSimpleNodeBroadcast},
  {"GreaterOrEqual", &HandleSimpleNodeBroadcast}, {"LessOrEqual", &HandleSimpleNodeBroadcast},
  {"Mean", &HandleSimpleNodeBroadcast}, {"Sum", &HandleSimpleNodeBroadcast},  {"Pow", &HandleSimpleNodeBroadcast},
  {"Where", &HandleSimpleNodeBroadcast},

  {"Clip", &HandleSimpleNode1Inp}, {"CastLike", &HandleSimpleNode1Inp},

  {"Transpose", &HandleTranspose},
  {"Concat", &HandleConcat},
  {"Split", &HandleSplit},
  {"Shape", &HandleShape},
  {"Pad", &HandlePad},
  {"ReduceSum", &HandleReduceSum},

  {"ReduceLogSum", &HandleReduceOp}, {"ReduceLogSumExp", &HandleReduceOp}, {"ReduceMax", &HandleReduceOp},
  {"ReduceMean", &HandleReduceOp}, {"ReduceMin", &HandleReduceOp}, {"ReduceProd", &HandleReduceOp},
  {"ReduceSumSquare", &HandleReduceOp}, {"ReduceL1", &HandleReduceOp}, {"ReduceL2", &HandleReduceOp},

  {"ArgMin", &HandleArgMinMax}, {"ArgMax", &HandleArgMinMax},

  {"Squeeze", &HandleSqueeze},
  {"Unsqueeze", &HandleUnsqueeze},
  //{"Slice", &HandleSlice},
  {"Tile", &HandleTile},

  {"Softmax", &HandleSoftHardMax}, {"Hardmax", &HandleSoftHardMax}, {"LogSoftmax", &HandleSoftHardMax},

  {"QuantizeLinear", &HandleQuantizeDequantizeLinear}, {"DequantizeLinear", &HandleQuantizeDequantizeLinear},
};

static const std::unordered_map<std::string_view, HandlerFunction*> extended_handler_map {
  {"com.microsoft.QLinearReduceMean", &HandleReduceOp},
  {"com.microsoft.QLinearSigmoid", &HandleSimpleNode1Inp},
  {"com.microsoft.QLinearLeakyRelu", &HandleSimpleNode1Inp},
  {"com.microsoft.QLinearConcat", &HandleQLinearConcat},
  {"com.microsoft.QLinearAdd", &HandleQLinearBinaryOp},
  {"com.microsoft.QLinearMul", &HandleQLinearBinaryOp},
  {"com.microsoft.QLinearAveragePool", &HandleQLinearPoolOp},
  {"com.microsoft.QLinearGlobalAveragePool", &HandleQLinearPoolOp},
  {"MaxPool", &HandleMaxPool},
};

static HandlerFunction* GetHandler(api::Node& node, bool allow_extended_ops) {
  std::string key;
  auto domain = node.Domain();
  auto op_type = node.OpType();
  if (domain == "") {
    key = std::string(op_type);
  } else if (domain == "com.microsoft") {
    key = "com.microsoft." + std::string(op_type);
  } else {
    return nullptr;
  }

  auto match = handler_map.find(key);
  if (match != handler_map.end()) {
    HandlerFunction* fn = match->second;
    return fn;
  } else if (allow_extended_ops) {
    match = extended_handler_map.find(key);
    if (match != extended_handler_map.end()) {
      HandlerFunction* fn = match->second;
      return fn;
    }
  }
  return nullptr;
}

bool ProcessTranspose(HandlerArgs& args) {
  if (!args.ctx.skip_cost_check &&
      !CanLikelyRemoveTranspose(args.ctx.graph, args.transpose) &&
      !args.node.IsOp("Transpose")) {
    return false;
  }
  HandlerFunction* fn = GetHandler(args.node, args.ctx.allow_extended_ops);
  if (fn == nullptr) {
    return false;
  }
  return fn(args);
}

std::optional<OptimizerCtx> MakeOptimizerContext(api::Graph& graph, bool allow_extended_ops) {
  auto opset = graph.Opset();
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

bool OptimizeImpl(OptimizerCtx& ctx) {
  
  const std::vector<std::unique_ptr<api::Node>> nodes = ctx.graph.Nodes();
  bool changed = false;
  for (size_t i = 0; i < nodes.size(); ++i) {
    api::Node& node = *nodes[i];
    const std::vector<std::string_view> &inputs = node.Inputs();
    for (size_t j = 0; j < inputs.size(); ++j) {
      const std::string_view inp = inputs[j];
      if (inp == "") {
        continue;
      }
      std::unique_ptr<api::Node> transpose = ctx.graph.GetNodeProducingOutput(inp);
      if (transpose != nullptr && transpose->IsOp("Transpose")) {
        std::optional<std::vector<int64_t>> perm = transpose->GetAttributeInts("perm");
        if (perm != std::nullopt) {
          std::vector<int64_t> perm_inv = InvertPerm(*perm);
          HandlerArgs args = {ctx, *transpose, node, *perm, perm_inv, j};
          if (ProcessTranspose(args)) {
            changed = true;
            break;
          }
        }
      }
    }
  }
  return changed;
}

bool Optimize(api::Graph& graph, bool allow_extended_ops) {
  auto ctx = MakeOptimizerContext(graph, allow_extended_ops);
  if (ctx == std::nullopt) {
    return false;
  }
  return OptimizeImpl(*ctx);
}

static bool ChangeLayout(api::Graph& graph, std::unordered_map<std::string_view, LayoutHandler*>& layout_handler_map,
                         bool last_to_first, bool allow_extended_ops) {
  auto ctx = MakeOptimizerContext(graph, allow_extended_ops);
  if (ctx == std::nullopt) {
    return false;
  }
  const std::vector<std::unique_ptr<api::Node>> nodes = graph.Nodes();
  bool changed = false;
  for (size_t i = 0; i < nodes.size(); ++i) {
    api::Node* node = &(*nodes[i]);
    auto match = layout_handler_map.find(node->OpType());
    if (match != layout_handler_map.end()) {
      std::unique_ptr<api::Node> new_node;
      LayoutHandler* handler = match->second;
      LayoutHandlerResult result = handler(graph, *node);
      if (!result.should_transpose) {
        continue;
      }
      size_t rank = result.rank;
      if (result.new_op_type != std::nullopt || result.new_domain != std::nullopt) {
        std::string_view new_op_type;
        if (result.new_op_type != std::nullopt) {
          new_op_type = *result.new_op_type;
        } else {
          new_op_type = node->OpType();
        }
        std::string_view new_domain;
        if (result.new_domain != std::nullopt) {
          new_domain = *result.new_domain;
        } else {
          new_domain = node->Domain();
        }
        auto inputs = node->Inputs();
        auto outputs = node->Outputs();
        new_node = graph.AddNode(new_op_type, inputs, outputs.size(), new_domain);
        for (size_t j = 0; j < outputs.size(); ++j) {
          if (outputs[j] != "") {
            graph.MoveOutput(*node, j, *new_node, j);
          }
        }
        new_node->CopyAttributes(*node);
        graph.RemoveNode(*node);
        node = &(*new_node);
      }
      auto perm = ChannelLastToFirstPerm(rank);
      auto perm_inv = InvertPerm(perm);
      if (last_to_first) {
        std::swap(perm, perm_inv);
      }
      TransposeFirstInput(*ctx, *node, perm_inv);
      TransposeOutputs(*ctx, *node, perm);
      changed = true;
    }
  }
  if (changed) {
    Optimize(graph, allow_extended_ops);
  }
  return changed;
}

bool ChannelLastToChannelFirst(api::Graph& graph, std::unordered_map<std::string_view, LayoutHandler*>& layout_handler_map, bool allow_extended_ops) {
  return ChangeLayout(graph, layout_handler_map, /*last_to_first*/ true, allow_extended_ops);
}

bool ChannelFirstToChannelLast(api::Graph& graph, std::unordered_map<std::string_view, LayoutHandler*>& layout_handler_map, bool allow_extended_ops) {
  return ChangeLayout(graph, layout_handler_map, /*last_to_first*/ false, allow_extended_ops);
}

}  // namespace onnx_layout_transformation
