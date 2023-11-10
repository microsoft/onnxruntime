// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnx_transpose_optimization.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "core/common/gsl.h"
#include "core/common/make_string.h"
#include "core/graph/constants.h"

namespace onnx_transpose_optimization {

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
  if (perm.has_value() && !IsValidPerm(*perm)) {
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
std::vector<int64_t> InvertPerm(const std::vector<int64_t>& perm) {
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
// permutation, but some are restricted. Also used for layout transformation.
std::vector<int64_t> ChannelLastToFirstPerm(size_t rank) {
  if (rank < 2) {
    return {};
  }

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
    if (perm.has_value()) {
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

static void Permute1DConstant(api::GraphRef& graph, api::NodeRef& node, api::TensorRef& constant,
                              size_t i, std::string_view input_name, const std::vector<int64_t>& perm) {
  // Create new transposed initializer
  auto rank = perm.size();
  auto shape = constant.Shape();
  std::vector<uint8_t> data = constant.Data();
  std::vector<uint8_t> new_data(data.size());
  size_t bytes_per_val = data.size() / rank;

  uint8_t* dst = new_data.data();
  for (size_t j = 0; j < rank; ++j) {
    uint8_t* src = data.data() + perm[j] * bytes_per_val;
    std::memcpy(dst, src, bytes_per_val);
    dst += bytes_per_val;
  }

  std::string_view new_initializer = graph.AddInitializer(constant.DType(), shape, new_data);
  node.SetInput(i, new_initializer);
  if (!graph.HasValueConsumers(input_name)) {
    graph.RemoveInitializer(input_name);
  }
}

// Replaces ith input to node with transposed value. Might create a new Transpose node, find an existing one,
// or transpose an initializer.
void TransposeInput(api::GraphRef& graph, api::NodeRef& node, size_t i,
                    const std::vector<int64_t>& perm, const std::vector<int64_t>& perm_inv) {
  std::string_view input = node.Inputs()[i];
  // Remove this node as a consumer
  node.SetInput(i, "");
  // Only local constants are editable
  std::unique_ptr<api::TensorRef> constant = graph.GetLocalConstant(input);
  auto consumers = graph.GetValueConsumers(input);

  // Case 1: input is a constant with a known list of consumer nodes
  if (constant != nullptr && consumers->comprehensive) {
    // Input is scalar, return early.
    if (constant->Shape().size() == 1 && constant->Shape()[0] == 0) {
      return;
    }
    // This is a special case where the constant is 1D with length == perm.
    // TODO: TransposeInitializer should be updated to handle this case.
    // Permute1DConstant permutes the constant and adds a new initializer. The old initializer is removed only if
    // there are no other consumers.
    if (constant->Shape().size() == 1 && constant->Shape()[0] == gsl::narrow_cast<int64_t>(perm.size())) {
      Permute1DConstant(graph, node, *constant, i, input, perm);
      return;
    }
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
      } else if (*perm2 == perm) {
        // we are trying to add a duplicate transpose.
        // do nothing and return
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
void TransposeInputs(OptimizerCtx& ctx, api::NodeRef& node, const std::vector<int64_t>& perm,
                     const std::vector<size_t>& input_indices) {
  auto perm_inv = InvertPerm(perm);
  for (size_t j : input_indices) {
    TransposeInput(ctx.graph, node, j, perm, perm_inv);
  }
}

// Inserts a Transpose op on the ith output of a node. Returns the new, transposed output.
// Updates shape information assuming that the output from the node will have a transposed shape (using perm_inv)
// but the overall (returned) output will match the initial shape.
std::string_view TransposeOutput(api::GraphRef& graph, api::NodeRef& node, size_t i,
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
void TransposeOutputs(OptimizerCtx& ctx, api::NodeRef& node, const std::vector<int64_t>& perm) {
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
static int EstimateValueRank(const api::GraphRef& graph, std::string_view input) {
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

static const HandlerInfo* GetHandler(api::NodeRef& node, const HandlerMap& extended_handlers);

// Returns true if the provided transpose node is only consumed by nodes we can likely push it through.
static bool CanLikelyRemoveTranspose(const api::GraphRef& graph, api::NodeRef& transpose,
                                     const HandlerMap& extended_handlers) {
  auto consumers = graph.GetValueConsumers(transpose.Outputs()[0]);
  if (!consumers->comprehensive) {
    return false;
  }
  for (auto& node : consumers->nodes) {
    if (GetHandler(*node, extended_handlers) == nullptr) {
      return false;
    }
  }
  return true;
}

// Estimates the cost of transposing an input. Currently uses rank heuristic. Negative if transpose is removed.
// Feel free to improve as needed.
static int EstimateTransposeValueCost(const api::GraphRef& graph, std::string_view input,
                                      const std::vector<int64_t>& perm_inv,
                                      const HandlerMap& extended_handlers) {
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
      if (*perm2 == perm_inv && CanLikelyRemoveTranspose(graph, *node, extended_handlers)) {
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
static int EstimateTransposeInputsCost(const api::GraphRef& graph, const api::NodeRef& node,
                                       const std::vector<int64_t>& perm_inv,
                                       const std::vector<size_t>& input_indices,
                                       const HandlerMap& extended_handlers) {
  auto inputs = node.Inputs();
  int cost = 0;
  for (size_t j : input_indices) {
    cost += EstimateTransposeValueCost(graph, inputs[j], perm_inv, extended_handlers);
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
bool HandleSimpleNode(HandlerArgs& args) {
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
constexpr HandlerInfo node_1_inp_handler = {&FirstInput, &HandleSimpleNode};

// Node with all inputs broadcastable
bool HandleSimpleNodeBroadcast(HandlerArgs& args) {
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
bool HandleSimpleNodeWithAxis(HandlerArgs& args, std::optional<int64_t> default_axis /*std::nullopt*/) {
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
  std::vector<int64_t> perm_shape{gsl::narrow_cast<int64_t>(new_perm.size())};
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
static void PermuteInput(api::GraphRef& graph, api::NodeRef& node, size_t i, const std::vector<int64_t>& perm) {
  size_t rank = perm.size();
  int64_t rank_int = gsl::narrow_cast<int64_t>(rank);

  std::string_view input = node.Inputs()[i];
  auto constant = graph.GetConstant(input);
  if (constant != nullptr) {
    auto shape = constant->Shape();
    if (shape.size() == 1 && (shape[0] == rank_int || shape[0] == 0)) {
      Permute1DConstant(graph, node, *constant, i, input, perm);
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

static bool HandleResize([[maybe_unused]] HandlerArgs& args) {
#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_QNN)
  // The CUDA Resize kernel requires that the input is NCHW, so we can't push a Transpose through a Resize
  // in ORT builds with CUDA enabled.
  // The ROCm EP is generated from the CUDA EP kernel so the same applies to builds with ROCm enabled.
  // The QNN EP requires the input to be NHWC, so the Resize handler is also not enabled for QNN builds.
  //
  // TODO: Remove this special case once the CUDA Resize kernel is implemented "generically" (i.e.) aligning with the
  // generic nature of the ONNX spec.
  // See https://github.com/microsoft/onnxruntime/pull/10824 for a similar fix applied to the CPU Resize kernel.
  return false;
#else
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
#endif
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

static bool HandleReduceOpWithArg(HandlerArgs& args) {
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

bool HandleReduceOps(HandlerArgs& args) {
  if ((args.node.OpType() == "ReduceSum" && args.ctx.opset < 13) ||
      // or all other reduce operators since opset 18
      (args.node.OpType() != "ReduceSum" && args.ctx.opset < 18)) {
    return HandleReduceOpWithArg(args);
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

constexpr HandlerInfo reduce_op_handler = {&FirstInput, &HandleReduceOps};

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

static bool HandleQuantizeDequantizeScale(const api::GraphRef& graph, const std::vector<int64_t>& perm,
                                          api::NodeRef& node, int64_t opset) {
  if (opset >= 13) {
    size_t rank = perm.size();
    // Update axis in Opset >= 13 if scale/zero_point are non-scalar
    auto inputs = node.Inputs();

    auto inp_shape = graph.GetValueInfo(inputs[1])->Shape();
    bool scalar_params = inp_shape.has_value() && inp_shape->size() == 0;

    if (!scalar_params) {
      int64_t axis = node.GetAttributeIntDefault("axis", 1);
      if (!NormalizeAndValidateAxis(axis, rank)) {
        return false;
      }
      node.SetAttributeInt("axis", perm[gsl::narrow_cast<size_t>(axis)]);
    }
  }
  return true;
}

static bool HandleQuantizeDequantizeLinear(HandlerArgs& args) {
  if (!HandleQuantizeDequantizeScale(args.ctx.graph, args.perm, args.node, args.ctx.opset)) {
    return false;
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
  std::vector<int64_t> new_axes{new_axis};
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
    std::vector<std::string_view> gather_inputs{repeats_inp, perm_inv_const};
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

// Helper to remove cancelling Transpose -> Transpose or
// Transpose -> Reshape nodes.
static void RemoveCancelingTransposeNodes(HandlerArgs& args) {
  // Input to 1st transpose
  std::string_view transpose_input = args.transpose.Inputs()[0];
  // Output of 2nd transpose or reshape
  std::string_view node_output = args.node.Outputs()[0];

  auto consumers = args.ctx.graph.GetValueConsumers(node_output);
  if (consumers->comprehensive) {
    // If possible, replace references to output of 2nd transpose/reshape with input to 1st
    ReplaceValueReferences(consumers->nodes, node_output, transpose_input);
  } else {
    // Otherwise, (ex: 2nd transpose/reshape is a graph output, a reasonably common case) the output name of the 2nd
    // transpose/reshape must be maintained. Attempt to move the output directly to the 1st transpose's parent.
    auto transpose_inp_consumers = args.ctx.graph.GetValueConsumers(transpose_input);
    std::unique_ptr<api::NodeRef> transpose_inp_node = args.ctx.graph.GetNodeProducingOutput(transpose_input);

    if (transpose_inp_node != nullptr && transpose_inp_consumers->comprehensive) {
      // Will move output to parent. First replace parent references with name of 2nd transpose/reshape output.
      args.node.SetInput(0, "");
      ReplaceValueReferences(transpose_inp_consumers->nodes, transpose_input, node_output);
      const std::vector<std::string_view>& transpose_inp_outputs = transpose_inp_node->Outputs();

      // Find index of output from parent node
      size_t i;
      for (i = 0; i < transpose_inp_outputs.size(); ++i) {
        if (transpose_inp_outputs[i] == transpose_input) break;
      }

      // Move 2nd transpose/reshape output (possible graph output) over top of it.
      args.ctx.graph.MoveOutput(args.node, 0, *transpose_inp_node, i);
    } else {
      // Worst-case scenario: Both parent output and 2nd transpose/reshape output cannot be removed (both graph outputs)
      // despite computing the same value. Use an Identity op instead.
      std::vector<std::string_view> single_empty_input{""};
      auto identity_ptr = args.ctx.graph.AddNode("Identity", single_empty_input, /*num_outputs*/ 1);
      api::NodeRef& identity = *identity_ptr;
      args.ctx.graph.MoveOutput(args.node, 0, identity, 0);
      identity.SetInput(0, transpose_input);
    }
  }
  // Remove 2nd transpose/reshape node.
  args.ctx.graph.RemoveNode(args.node);

  // 2nd transpose/reshape no longer references 1st. Remove first if possible.
  if (!args.ctx.graph.HasValueConsumers(args.transpose.Outputs()[0])) {
    args.ctx.graph.RemoveNode(args.transpose);
  }
}

// helper to support scenario where second node is a Reshape that is logically equivalent to a Transpose.
// called from HandleTranspose and HandleReshape.
static bool HandleTransposeImpl(HandlerArgs& args, const std::vector<int64_t>& node_perm) {
  if (args.perm_inv == node_perm) {
    // Case 1: Permutations cancel.
    RemoveCancelingTransposeNodes(args);
  } else {
    // Case 2: Permutations don't cancel but can be merged. Update 2nd Transpose with merged perms and remove the 1st
    // Transpose. Keeping the 2nd Transpose is simpler as no updates are required to downstream nodes, and all we have
    // to do is use the input from the 1st Transpose in the updated 2nd Transpose.
    std::vector<int64_t> new_perm = ComposePerm(args.perm, node_perm);

    std::unique_ptr<api::NodeRef> new_node;
    if (args.node.OpType() == "Reshape") {
      // replace Reshape with Transpose to simplify the logic.
      // use the same input as the 1st Transpose, move the output from the Reshape to the new Transpose node,
      // and remove the Reshape node.
      new_node = args.ctx.graph.AddNode("Transpose", {args.transpose.Inputs()[0]}, 1);
      args.ctx.graph.MoveOutput(args.node, 0, *new_node, 0);
      args.ctx.graph.RemoveNode(args.node);
    } else {
      // use the input from the 1st Transpose to the 2nd.
      args.node.SetInput(0, args.transpose.Inputs()[0]);
    }

    // set the perm attribute to the merged version
    api::NodeRef& target_node = new_node ? *new_node : args.node;
    target_node.SetAttributeInts("perm", new_perm);

    // 2nd transpose no longer references 1st. Remove first if possible.
    if (!args.ctx.graph.HasValueConsumers(args.transpose.Outputs()[0])) {
      args.ctx.graph.RemoveNode(args.transpose);
    }
  }

  return true;
}

static bool HandleTranspose(HandlerArgs& args) {
  // In this handler a transpose leads to another transpose. "transpose" is the 1st and "node" is the 2nd.
  std::optional<std::vector<int64_t>> node_perm = GetPermAttrIfValid(args.node);
  if (node_perm == std::nullopt || node_perm->size() != args.perm.size()) {
    return false;
  }

  return HandleTransposeImpl(args, *node_perm);
}

constexpr HandlerInfo transpose_handler = {&FirstInput, &HandleTranspose, /*transposes_outputs*/ false};

static bool FinalizeReshapeShape(const std::vector<int64_t>& input_shape,      // Reshape input 0
                                 const std::vector<int64_t>& requested_shape,  // Reshape input 1
                                 bool allow_zero,
                                 std::vector<int64_t>& final_shape) {
  // we need a concrete input shape to handle Reshape here
  int64_t total_size = 1;
  for (auto dim : input_shape) {
    if (dim < 0) {
      return false;  // potentially valid symbolic dim denoted by -1 but we need a fixed value.
    }

    total_size *= dim;
  }

  auto input_rank = input_shape.size();
  auto rank = requested_shape.size();

  if (input_rank != rank) {
    return false;  // potentially valid but to treat a Reshape as a Transpose the rank must match
  }

  ptrdiff_t unknown_dim = -1;
  int64_t size = 1;

  final_shape = requested_shape;

  for (size_t i = 0; i < rank; ++i) {
    if (requested_shape[i] == -1) {
      if (unknown_dim != -1) {
        return false;  // invalid: only one -1 dim allowed
      }

      unknown_dim = i;
    } else {
      // if allow_zero is true we keep the 0 from the requested_shape as-is.
      // if allow_zero is false we copy the dim value from the input_shape.
      if (!allow_zero && requested_shape[i] == 0) {
        final_shape[i] = input_shape[i];
      }

      size *= final_shape[i];
    }
  }

  if (unknown_dim != -1) {
    // calculate unknown dimension
    if (size == 0 || (total_size % size) != 0) {
      return false;  // invalid: dims are mismatched
    }

    final_shape[unknown_dim] = total_size / size;
  } else {
    if (size != total_size) {
      return false;  // invalid: dims are mismatched
    }
  }

  return true;
}

static bool HandleReshape(HandlerArgs& args) {
  // A Reshape can be logically equivalent to a Transpose if all dims with a value > 1 remain in the same order
  // and do not change size. If so, we can use HandleTransposeImpl to merge them.
  //  e.g. Reshape(input {1, 512, 4, 1}, shape {1, 1, 512, 4}) is equivalent to Transpose with perms { 0, 3, 1, 2 }
  //       Reshape(input {1, 1, 512, 4}, shape {512, 4, 1, 1}) is equivalent to Transpose with perms { 2, 3, 0, 1 }
  //       Reshape(input {1, 512, 1, 4}, shape {1, 512, 4, 1}) is equivalent to Transpose with perms { 0, 1, 3, 2 }
  //

  // Get transpose input shape
  auto transpose_input_shape = args.ctx.graph.GetValueInfo(args.transpose.Inputs()[0])->Shape();
  if (!transpose_input_shape.has_value()) {
    return false;
  }

  auto transpose_output_shape = args.ctx.graph.GetValueInfo(args.transpose.Outputs()[0])->Shape();
  if (!transpose_output_shape.has_value()) {
    // given transpose_input_shape had a value the shape inferencing should have calculated this.
    // we could read the perms to calculate but that _really_ should not be necessary
    return false;
  }

  // `shape` input of reshape node is required to be constant
  auto requested_shape_data = args.ctx.graph.GetConstant(args.node.Inputs()[1]);
  if (requested_shape_data == nullptr || requested_shape_data->Data().size() == 0) {
    return false;
  }

  auto reshape_requested_shape = DataInt64(*requested_shape_data);

  // need rank to match for Reshape to be equivalent to a Transpose
  if (transpose_input_shape->size() != reshape_requested_shape.size()) {
    return false;
  }

  // process the requested shape to handle any -1 or 0 values.

  int64_t allow_zero = 0;  // default is to treat a 0 as copying the value from the input shape.
  if (args.node.SinceVersion() > 13) {
    auto allow_zero_attr = args.node.GetAttributeInt("allowzero");
    if (allow_zero_attr) {
      allow_zero = *allow_zero_attr;
    }
  }

  std::vector<int64_t> reshape_output_shape;
  if (!FinalizeReshapeShape(*transpose_output_shape, reshape_requested_shape, allow_zero, reshape_output_shape)) {
    return false;
  }

  // calculate the perms if the Reshape node is equivalent to a Transpose
  std::vector<int64_t> input_dims(*transpose_output_shape);
  std::vector<int64_t> perms(reshape_output_shape.size(), -1);

  auto reshape_out_cur = reshape_output_shape.begin();
  auto input_begin = input_dims.begin();
  auto input_end = input_dims.end();

  // for each output dim find the input dim that maps to it.
  // an input dim with value of '1' can be anywhere.
  // input dims with values != 1 must be in the same order in the output dims.
  // set the value of the input dim to -1 to mark it as used.
  for (size_t i = 0; i < perms.size(); ++i) {
    // start from the beginning each time looking for the first unused input dim that matches the current output dim
    auto cur = input_begin;
    auto target_dim = *reshape_out_cur++;
    while (*cur != target_dim) {
      if (*cur == -1) {
        // previously used
      } else if (*cur != 1 && target_dim != 1) {
        // failure. mis-match of ordering of dim with data
        return false;
      }

      if (++cur == input_end) {
        // failure: ran out of input and didn't find match for target_dim
        return false;
      }
    }

    // if we got here we found a valid match.
    // update perms with the input dim index and set the input_dim value to -1 to mark it as used.
    // narrow to int32 so we can use as an int64_t value and size_t index without warnings/multiple casts
    int32_t input_idx = gsl::narrow_cast<int32_t>(cur - input_begin);
    perms[i] = input_idx;
    input_dims[input_idx] = -1;
  }

  return HandleTransposeImpl(args, perms);
}

constexpr HandlerInfo reshape_handler = {&FirstInput, &HandleReshape, /*transposes_outputs*/ false};

// TODO: check binary size of this and replace it with constexpr if large
static const std::unordered_map<std::string_view, const HandlerInfo&> handler_map{
    {"Cast", simple_node_handler},
    {"Exp", simple_node_handler},
    {"Identity", simple_node_handler},
    {"LeakyRelu", simple_node_handler},
    {"Log", simple_node_handler},
    {"Reciprocal", simple_node_handler},
    {"Relu", simple_node_handler},
    {"Sigmoid", simple_node_handler},
    {"Sqrt", simple_node_handler},
    {"Tanh", simple_node_handler},
    {"Abs", simple_node_handler},
    {"Not", simple_node_handler},
    {"Ceil", simple_node_handler},
    {"Floor", simple_node_handler},
    {"Neg", simple_node_handler},
    {"Erf", simple_node_handler},
    {"HardSigmoid", simple_node_handler},
    {"Round", simple_node_handler},
    {"IsInf", simple_node_handler},
    {"IsNaN", simple_node_handler},
    {"Selu", simple_node_handler},
    {"Shrink", simple_node_handler},
    {"Sign", simple_node_handler},
    {"Softplus", simple_node_handler},
    {"Softsign", simple_node_handler},
    {"ThresholdedRelu", simple_node_handler},
    {"Celu", simple_node_handler},
    {"HardSwish", simple_node_handler},

    {"Sin", simple_node_handler},
    {"Cos", simple_node_handler},
    {"Tan", simple_node_handler},
    {"Sinh", simple_node_handler},
    {"Cosh", simple_node_handler},
    {"Tanh", simple_node_handler},
    {"Asin", simple_node_handler},
    {"Acos", simple_node_handler},
    {"Atan", simple_node_handler},
    {"Asinh", simple_node_handler},
    {"Acosh", simple_node_handler},
    {"Atanh", simple_node_handler},

    {"Add", broadcast_node_handler},
    {"Max", broadcast_node_handler},
    {"Min", broadcast_node_handler},
    {"Mul", broadcast_node_handler},
    {"Sub", broadcast_node_handler},
    {"Div", broadcast_node_handler},
    {"And", broadcast_node_handler},
    {"Or", broadcast_node_handler},
    {"Xor", broadcast_node_handler},
    {"Mod", broadcast_node_handler},
    {"PRelu", broadcast_node_handler},
    {"BitShift", broadcast_node_handler},
    {"Equal", broadcast_node_handler},
    {"Greater", broadcast_node_handler},
    {"Less", broadcast_node_handler},
    {"GreaterOrEqual", broadcast_node_handler},
    {"LessOrEqual", broadcast_node_handler},
    {"Mean", broadcast_node_handler},
    {"Sum", broadcast_node_handler},
    {"Pow", broadcast_node_handler},
    {"Where", broadcast_node_handler},

    {"Clip", node_1_inp_handler},
    {"CastLike", node_1_inp_handler},

    {"Transpose", transpose_handler},
    {"Concat", concat_handler},
    {"Split", split_handler},
    {"Shape", shape_handler},
    {"Pad", pad_handler},
    {"Resize", resize_handler},
    {"ReduceSum", reduce_op_handler},

    {"ReduceLogSum", reduce_op_handler},
    {"ReduceLogSumExp", reduce_op_handler},
    {"ReduceMax", reduce_op_handler},
    {"ReduceMean", reduce_op_handler},
    {"ReduceMin", reduce_op_handler},
    {"ReduceProd", reduce_op_handler},
    {"ReduceSumSquare", reduce_op_handler},
    {"ReduceL1", reduce_op_handler},
    {"ReduceL2", reduce_op_handler},

    {"ArgMin", arg_min_max_handler},
    {"ArgMax", arg_min_max_handler},

    {"Squeeze", squeeze_handler},
    {"Unsqueeze", unsqueeze_handler},
    {"Slice", slice_handler},
    {"Tile", tile_handler},

    {"Softmax", soft_hard_max_handler},
    {"Hardmax", soft_hard_max_handler},
    {"LogSoftmax", soft_hard_max_handler},

    {"QuantizeLinear", quantize_dequantize_linear_handler},
    {"DequantizeLinear", quantize_dequantize_linear_handler},
    {"Reshape", reshape_handler},
};

static const HandlerInfo* GetHandler(api::NodeRef& node, const HandlerMap& extended_handlers) {
  std::string key;
  auto domain = node.Domain();
  auto op_type = node.OpType();

  if (domain == onnxruntime::kOnnxDomain || domain == onnxruntime::kOnnxDomainAlias) {
    key = std::string(op_type);
  } else if (domain == onnxruntime::kMSDomain) {
    key = onnxruntime::MakeString(domain, ".", op_type);
  } else {
    return nullptr;
  }

  // extended map is higher priority
  auto match = extended_handlers.find(key);
  if (match != extended_handlers.end()) {
    return &match->second;
  }

  match = handler_map.find(key);
  if (match != handler_map.end()) {
    return &match->second;
  }

  return nullptr;
}

static int CalculateCost(const api::GraphRef& graph, const api::NodeRef& node,
                         const std::vector<int64_t>& perm,
                         const std::unordered_set<std::string>& outputs_leading_to_transpose,
                         const HandlerInfo& info,
                         const std::vector<size_t>& input_indices,
                         const HandlerMap& extended_handlers) {
  // We require the input cost (number of transposes before the op) and the total cost to strictly decrease.
  // Strict decrease of the input cost ensures the optimization is stable, since the total cost decrease is just an
  // estimate (the transpose after the op may or may not cancel with a subsequent transpose). We don't want
  // repeated runs of the optimizer to have a transpose toggle between two inputs of a binary op.
  int cost = EstimateTransposeInputsCost(graph, node, perm, input_indices, extended_handlers);

  if (cost < 0 && info.transposes_outputs) {
    // If the output will be transposed and won't ultimately cancel, factor in that cost.
    bool has_output_leading_to_transpose = false;
    auto outputs = node.Outputs();
    int out_cost = 0;
    // Having multiple outputs is rare. When it happens (Split), the total size of the outputs isn't much larger
    // than the largest input. Cost is rank currently, so just use the largest cost (rank) over all outputs.
    for (auto out : outputs) {
      out_cost = std::max(out_cost, EstimateValueRank(graph, out));
      if (outputs_leading_to_transpose.find(std::string(out)) != outputs_leading_to_transpose.end()) {
        has_output_leading_to_transpose = true;
      }
    }

    if (!has_output_leading_to_transpose) {
      cost += out_cost;
    }
  }

  return cost;
}

// Default cost check. Returns `true` if pushing the Transpose through the node is considered to be beneficial.
static bool ShouldPushTranspose(const api::GraphRef& graph, const api::NodeRef& node,
                                const std::vector<int64_t>& perm,
                                const std::unordered_set<std::string>& outputs_leading_to_transpose,
                                const HandlerInfo& info,
                                const std::vector<size_t> transposable_input_indices,
                                const HandlerMap& extended_handlers) {
  if (node.IsOp("Transpose")) {
    return true;
  }

  int cost = CalculateCost(graph, node, perm, outputs_leading_to_transpose, info, transposable_input_indices,
                           extended_handlers);
  return cost < 0;
}

// Finds a handler for the node and estimates the cost of pushing a transpose. Does so if deemed beneficial.
bool ProcessTranspose(OptimizerCtx& ctx, api::NodeRef& transpose, api::NodeRef& node,
                      const std::vector<int64_t>& perm, size_t transpose_input_index,
                      const std::unordered_set<std::string>& outputs_leading_to_transpose) {
  const HandlerInfo* info = GetHandler(node, ctx.extended_handlers);
  if (info == nullptr) {
    return false;
  }

  std::vector<size_t> input_indices = info->transposible_inputs_fn(ctx, node);
  if (std::find(input_indices.begin(), input_indices.end(), transpose_input_index) == input_indices.end()) {
    // Transpose is not on an eligible input
    return false;
  }

  CostCheckResult cost = CostCheckResult::kFallThrough;

  if (ctx.cost_check_fn) {
    cost = ctx.cost_check_fn(ctx.graph, node, perm, outputs_leading_to_transpose);
  }

  if (cost == CostCheckResult::kFallThrough) {
    cost = ShouldPushTranspose(ctx.graph, node, perm, outputs_leading_to_transpose, *info, input_indices,
                               ctx.extended_handlers)
               ? CostCheckResult::kPushTranspose
               : CostCheckResult::kStop;
  }

  if (cost == CostCheckResult::kStop) {
    return false;
  }

  std::vector<int64_t> perm_inv = InvertPerm(perm);
  HandlerArgs args = {ctx, transpose, node, perm, perm_inv, input_indices};
  return info->handler_fn(args);
}

// Returns nullopt if graph opset is unsupported.
std::optional<OptimizerCtx> MakeOptimizerContext(api::GraphRef& graph,
                                                 const std::string& provider_type,
                                                 CostCheckFn cost_check_fn,
                                                 const HandlerMap& extended_handlers,
                                                 std::string& error_msg) {
  auto opset = graph.Opset("");
  if (opset == std::nullopt) {
    opset = graph.Opset("ai.onnx");
  }

  if (opset == std::nullopt || *opset > kMaxSupportedOpset || *opset < kMinSupportedOpset) {
    // if the model doesn't have an ONNX opset that's fine as there are no ops we'd move around
    if (opset.has_value()) {
      error_msg = "Unsupported ONNX opset: " + std::to_string(*opset);
    }

    return std::nullopt;
  }

  OptimizerCtx ctx{*opset, graph, provider_type, cost_check_fn, extended_handlers};
  return ctx;
}

// Performs optimization. General algorithm: iterate over nodes in topological order. If a node has a transpose
// as input, push it through if the transpose cost does not increase and is likely to decrease.
OptimizeResult OptimizeImpl(OptimizerCtx& ctx) {
  OptimizeResult result{};
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
        const HandlerInfo* info = GetHandler(node, ctx.extended_handlers);
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
  bool have_dq = false;

  // 3 Scenarios:
  //
  // 1. Level 1 optimizer.
  //
  //    When level 1 optimizers are first run prior to graph partitioning no nodes are assigned.
  //    We can modify any existing nodes and add new nodes.
  //    ctx.provider_type is empty.
  //
  //    Level 1 optimizers may also run after layout transformation to do things like constant folding.
  //    In this case we can only modify unassigned nodes.
  //
  // 2. Layout transformation:
  //
  //    Existing nodes may be unassigned, assigned to the EP the layout is changing for, or assigned to a different EP.
  //    ctx.provider_type is set to the EP the layout is changing for.
  //
  //    We can modify unassigned nodes.
  //    We can not modify any nodes assigned to a different EP as the modification may render them incompatible with
  //    the EP.
  //    We can modify nodes assigned to the current EP and create new nodes, but do not assign any new nodes yet.
  //    Following the layout change GraphPartitioner will call GetCapability again for the current EP, which allows
  //    it to take the new nodes where possible.
  //    We also know that the CPU EP will take any remaining nodes as the last step of graph partitioning.
  //
  //    To do this, we check node assignment vs the current EP name to determine if the node can be modified.
  //    We leave onnxruntime::ApiGraph::new_node_ep_ empty so new nodes are not assigned here.
  //
  // 3. Level 3 NHWC Transformer:
  //
  //    Specific to CPU EP and runs post-partitioning, so all nodes are assigned at this point.
  //    ctx.provider_type is set to the CPU EP.
  //    Existing nodes assigned to the CPU EP can be modified.
  //    New nodes can be created and are directly assigned to the CPU EP by setting onnxruntime::ApiGraph::new_node_ep_
  //
  const auto can_modify_node = [&ctx](const api::NodeRef& node) {
    const auto& node_ep = node.GetExecutionProviderType();
    bool can_modify = false;

    if (node_ep.empty()) {
      // unassigned nodes can always be modified
      can_modify = true;
    } else if (node_ep == ctx.provider_type) {
      // we can also modify if the EP name in provider_type is not empty and the node is assigned to that EP.
      can_modify = true;
    }

    return can_modify;
  };

  // Optimize graph. Nodes will be modified during iteration, but nodes are never deleted before we reach them.
  // New transpose nodes are inserted, but always as an input to an existing node.
  for (size_t i = 0; i < nodes.size(); ++i) {
    api::NodeRef& node = *nodes[i];
    if (node.OpType() == "DequantizeLinear") {
      have_dq = true;
    }

    if (!can_modify_node(node)) {
      continue;
    }

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
          if (ProcessTranspose(ctx, *transpose, node, *perm, j, outputs_leading_to_transpose)) {
            changed = true;
            // Subsequent inputs may have changed and node may have been removed.
            break;
          }
        }
      }
    }
  }

  if (!have_dq) {
    result.graph_modified = changed;
    return result;
  }

  // Run second optimization pass.
  // If any transpose succeeds a DQ node, move it above the DQ node if it's not part of a QDQ node group.
  // In QDQ models this helps to preserve the QDQ node group when a Transpose was pushed across a DQ into
  // an existing QDQ node group.
  // In all other scenarios this is beneficial as well because moving transpose above DQ node is more efficient as
  // transpose node now handles less data.
  auto graph_nodes = ctx.graph.Nodes();
  for (size_t i = 1; i < graph_nodes.size(); i++) {
    const auto& node = *graph_nodes[i];

    if (!can_modify_node(node)) {
      continue;
    }

    if (node.OpType() == "Transpose") {
      auto& transpose_node = *graph_nodes[i];
      auto dq_node = ctx.graph.GetNodeProducingOutput(transpose_node.Inputs()[0]);
      if (!dq_node || dq_node->OpType() != "DequantizeLinear") {
        continue;
      }

      // Check if Transpose node is the only consumer of dq node
      auto consumers_of_dq_node = ctx.graph.GetValueConsumers(dq_node->Outputs()[0]);
      if (!consumers_of_dq_node->comprehensive || consumers_of_dq_node->nodes.size() > 1) {
        continue;
      }

      auto consumers_of_transpose_node = ctx.graph.GetValueConsumers(transpose_node.Outputs()[0]);
      bool is_part_of_qdq_group = std::find_if(consumers_of_transpose_node->nodes.cbegin(),
                                               consumers_of_transpose_node->nodes.cend(),
                                               [](const std::unique_ptr<api::NodeRef>& node) {
                                                 return node->OpType() == "QuantizeLinear";
                                               }) != consumers_of_transpose_node->nodes.cend();
      if (is_part_of_qdq_group) {
        continue;
      }

      // Update Dequantize Node and move the transpose above it
      auto perm = GetPermAttrIfValid(transpose_node);
      if (!perm.has_value()) {
        continue;
      }

      // we're moving the Transpose to before the DQ, so we need to use the inverse permutations to update the axis
      // attribute correctly when doing per-axis dequantization
      if (!HandleQuantizeDequantizeScale(ctx.graph, InvertPerm(*perm), *dq_node, ctx.opset)) {
        continue;
      }

      TransposeFirstInput(ctx, *dq_node, *perm);

      // remove existing transpose node
      transpose_node.SetInput(0, "");
      ctx.graph.MoveOutput(transpose_node, 0, *dq_node, 0);
      ctx.graph.RemoveNode(transpose_node);
      changed = true;
    }
  }

  result.graph_modified = changed;
  return result;
}

const std::unordered_set<std::string_view>& GetLayoutSensitiveOps() {
  // List of all layout sensitive ops defined in ONNX standard.
  static std::unordered_set<std::string_view> layout_sensitive_ops = {
      // normalization
      "BatchNormalization", "InstanceNormalization",

      // convolutions
      "Conv", "QLinearConv", "ConvTranspose",

      // pooling
      "AveragePool", "LpPool", "MaxPool", "MaxUnpool",
      "GlobalAveragePool", "GlobalLpPool", "GlobalMaxPool",

      // other
      "LRN",
      "GridSample",
      "DepthToSpace", "SpaceToDepth"};

  return layout_sensitive_ops;
}

OptimizeResult Optimize(api::GraphRef& graph, const std::string& provider_type, CostCheckFn cost_check_fn,
                        const HandlerMap& extended_handlers) {
  OptimizeResult result{};

  std::string error_msg;
  auto ctx = MakeOptimizerContext(graph, provider_type, cost_check_fn, extended_handlers, error_msg);

  if (ctx == std::nullopt) {
    if (!error_msg.empty()) {
      result.error_msg = error_msg;
    }

    return result;
  }

  return OptimizeImpl(*ctx);
}

}  // namespace onnx_transpose_optimization
