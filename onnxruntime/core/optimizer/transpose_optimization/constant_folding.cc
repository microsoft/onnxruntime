// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/transpose_optimization/constant_folding.h"

#include <cassert>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/common/gsl.h"
#include "core/optimizer/transpose_optimization/onnx_transpose_optimization.h"

namespace onnx_transpose_optimization {

/// <summary>
/// Represents a constant-folding operation. Currently only supports Transpose and Squeeze.
/// Used to look up operations that we've already done on shared initializers.
/// </summary>
struct FoldOp {
  enum class Type {
    kUnsupported,
    kTranspose,
    kSqueeze,
  };

  FoldOp() = default;

  static FoldOp::Type GetFoldOpType(std::string_view op_type) {
    FoldOp::Type type = Type::kUnsupported;
    if (op_type == "Transpose") {
      type = Type::kTranspose;
    } else if (op_type == "Squeeze") {
      type = Type::kSqueeze;
    }

    return type;
  }

  static bool Init(const OptimizerCtx& ctx, const api::TensorRef& const_input, const api::NodeRef& node,
                   /*out*/ FoldOp& fold_op) {
    FoldOp::Type type = GetFoldOpType(node.OpType());

    switch (type) {
      case FoldOp::Type::kTranspose: {
        std::optional<std::vector<int64_t>> perm = GetPermAttrIfValid(node);
        if (perm == std::nullopt) {
          // Invalid transpose perm attribute. Should not happen. Skip.
          return false;
        }

        fold_op.type = type;
        fold_op.op_data = *perm;
        break;
      }
      case FoldOp::Type::kSqueeze: {
        std::optional<std::vector<int64_t>> squeeze_axes = ReadFromAttrOrInput(ctx, node, "axes", /*inp_index*/ 1,
                                                                               /*opset*/ 13);
        if (squeeze_axes == std::nullopt) {
          // Invalid Squeeze axes value. Should not happen. Skip.
          return false;
        }

        fold_op.type = type;
        fold_op.op_data = SqueezeShape(const_input.Shape(), *squeeze_axes);
        break;
      }
      default:
        // Unsupported
        return false;
    }

    return true;
  }

  bool operator==(const FoldOp& other) {
    return type == other.type && op_data == other.op_data;
  }

  std::optional<std::string_view> Apply(OptimizerCtx& ctx, const api::TensorRef& const_input) const {
    std::string_view new_initializer_name;

    switch (type) {
      case FoldOp::Type::kTranspose: {
        new_initializer_name = ctx.graph.AddInitializer(const_input.DType(),
                                                        const_input.Shape(),
                                                        const_input.Data());
        ctx.graph.TransposeInitializer(new_initializer_name, op_data);
        break;
      }
      case FoldOp::Type::kSqueeze: {
        new_initializer_name = ctx.graph.AddInitializer(const_input.DType(),
                                                        const_input.Shape(),
                                                        const_input.Data());
        ctx.graph.ReshapeInitializer(new_initializer_name, op_data);
        break;
      }
      default:
        // Unsupported
        return std::nullopt;
    }

    return new_initializer_name;
  }

  Type type{Type::kUnsupported};
  std::vector<int64_t> op_data;  // perms (Transpose) or axes (Squeeze)
};

/// <summary>
/// POD type that stores an operation that has been done to an initializer and the name of the resulting initializer.
/// </summary>
struct FoldOpResult {
  FoldOp op;
  std::string_view result_initializer_name;
};

/// <summary>
/// Helper class used to run constant-folding on nodes. Keeps a cache of operations that have already been done
/// on initializers to enable re-use.
/// </summary>
class ConstantFoldingCtx {
 public:
  ConstantFoldingCtx(OptimizerCtx& ctx) : ctx_(ctx) {}

  bool TryConstantFoldNode(api::NodeRef& node);

 private:
  bool IsNodeSupported(const api::NodeRef& node) const {
    return FoldOp::GetFoldOpType(node.OpType()) != FoldOp::Type::kUnsupported;
  }

  std::optional<std::string_view> CreateFoldedInitializer(const api::TensorRef& const_input,
                                                          std::string_view const_input_name,
                                                          api::NodeRef& op_node) {
    FoldOp fold_op = {};
    if (!FoldOp::Init(ctx_, const_input, op_node, fold_op)) {
      return std::nullopt;
    }

    // Try to see if we've done this folding operation on this initializer before.
    // If so, just return the name of the previously computed initializer.
    auto it = folded_initializers_.find(const_input_name);
    if (it != folded_initializers_.end()) {
      for (const auto& op_result : it->second) {
        if (fold_op == op_result.op) {
          return op_result.result_initializer_name;
        }
      }
    }

    // Create new folded initializer.
    auto new_initializer_name = fold_op.Apply(ctx_, const_input);
    if (!new_initializer_name.has_value()) {
      return std::nullopt;
    }

    // Store result of this constant-folding operation in case it needs to be reused later.
    folded_initializers_[const_input_name].push_back({fold_op, *new_initializer_name});

    return new_initializer_name;
  }

  void TryRemoveInitializer(std::string_view name) {
    if (!ctx_.graph.HasValueConsumers(name)) {
      folded_initializers_.erase(name);
      ctx_.graph.RemoveInitializer(name);
    }
  }

  OptimizerCtx& ctx_;
  std::unordered_map<std::string_view, std::vector<FoldOpResult>> folded_initializers_;
};

/// <summary>
/// Try to constant fold Transpose or Squeeze nodes if their input is a constant.
/// Returns true if the graph was modified (i.e., at least one of the consumers received a constant-folded value).
/// </summary>
/// <param name="ctx">Optimization context state</param>
/// <param name="node">Squeeze or Transpose node to try to constant-fold</param>
/// <returns>True if graph was modified. The node may not have been removed in either case.</returns>
bool ConstantFoldingCtx::TryConstantFoldNode(api::NodeRef& node) {
  if (!IsNodeSupported(node)) {
    return false;
  }

  std::string_view node_input_name = node.Inputs()[0];
  auto const_input = ctx_.graph.GetLocalConstant(node_input_name);
  if (const_input == nullptr) {
    // Doesn't have a constant input. Skip.
    return false;
  }

  std::string_view node_output_name = node.Outputs()[0];
  auto consumers = ctx_.graph.GetValueConsumers(node_output_name);

  if (consumers->nodes.empty()) {
    // No consumers Skip.
    return false;
  }

  // Create new squeezed or transposed initializer.
  // Once we create this new initializer, we're committed to modifying the graph.
  std::optional<std::string_view> new_initializer_name = CreateFoldedInitializer(*const_input, node_input_name, node);
  if (!new_initializer_name.has_value()) {
    return false;
  }

  // Iterate through consumers and replace their input(s) with the new initializer.
  for (auto& consumer : consumers->nodes) {
    std::vector<std::string_view> inputs = consumer->Inputs();

    for (size_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
      if (inputs[input_idx] == node_output_name) {
        consumer->SetInput(input_idx, *new_initializer_name);
      }
    }
  }

  // Remove original node if its output is unused.
  if (!ctx_.graph.HasValueConsumers(node_output_name)) {
    ctx_.graph.RemoveNode(node);
  }

  // Remove old initializer if no longer used.
  // Will not happen if this initializer was unsqueezed/transposed in-place for another consumer.
  // Will happen if this initializer is a result of a previous constant-folding operation.
  //
  // Example: shared_const --+--> Transpose --> Squeeze --> Op0
  //                         |
  //                         +--> Op1
  //
  // The first call to TryConstantFoldNode(transpose) does not remove shared_const because it is used by 'Op1'.
  // However, the graph becomes:
  //   transposed_const --> Squeeze --> Op0 -->
  //   shared_const --> Op1 -->
  //
  // The subsequent call to TryConstantFoldNode(squeeze) removes transposed_const from the graph, and we end up with:
  //   transposed_squeezed_const --> Op0
  //   shared_const --> Op1
  TryRemoveInitializer(node_input_name);

  return true;
}

bool RunConstantFolding(OptimizerCtx& ctx, CanModifyNodeFn can_modify_node_fn) {
  bool changed = false;
  ConstantFoldingCtx const_fold_ctx(ctx);
  auto graph_nodes = ctx.graph.Nodes();
  for (size_t i = 0; i < graph_nodes.size(); i++) {
    auto& node = *graph_nodes[i];

    if (!can_modify_node_fn(ctx, node)) {
      continue;
    }

    if (const_fold_ctx.TryConstantFoldNode(node)) {
      changed = true;
    }
  }

  return changed;
}
}  // namespace onnx_transpose_optimization
