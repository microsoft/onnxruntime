// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/optimizer/selectors_actions/actions.h"
#include "core/platform/threadpool.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

class Graph;
class Node;

namespace QDQ {

// helper that sets optional zero point values before replacing a node
struct QDQReplaceWithNew : public ReplaceWithNewFixed {
  QDQReplaceWithNew(std::string domain,
                    std::string op_type,
                    std::vector<NodeAndMoveInfo>&& value_moves)
      : ReplaceWithNewFixed{std::move(domain), std::move(op_type), std::move(value_moves)} {}

  Status Run(Graph&, const NodesToOptimize& selected_nodes) const override;

#if !defined(ORT_MINIMAL_BUILD)
  Status RunForSave(Graph& graph, const NodesToOptimize& selected_nodes,
                    const SatRuntimeOptimizationSaveContext& save_context,
                    SavedState& saved_state, bool& graph_modified) const override;
#endif  // !defined(ORT_MINIMAL_BUILD)
};

// replace node with QLinear version
struct ReplaceWithQLinear : public QDQReplaceWithNew {
  // provide NodeLocation for source node, and ValueMoveInfo for the value to move to the replacement node
  ReplaceWithQLinear(std::string domain,
                     std::vector<NodeAndMoveInfo>&& value_moves)
      : QDQReplaceWithNew{std::move(domain), "generated at runtime", std::move(value_moves)} {}

 private:
  std::string OpType(const RuntimeState& state) const override {
    return "QLinear" + state.selected_nodes.Target().OpType();
  }
};

struct UnaryReplaceWithQLinear : ReplaceWithQLinear {
  UnaryReplaceWithQLinear(std::string domain);

 private:
  NodeAttributes ExtraAttributes(const RuntimeState& state) const override;
};

struct BinaryReplaceWithQLinear : ReplaceWithQLinear {
  BinaryReplaceWithQLinear(std::string domain);
};

struct VariadicReplaceWithQLinear : ReplaceWithQLinear {
  VariadicReplaceWithQLinear(std::string domain);
};

struct ConvReplaceWithQLinear : ReplaceWithQLinear {
  ConvReplaceWithQLinear();
};
struct WhereReplaceWithQLinear : ReplaceWithQLinear {
  WhereReplaceWithQLinear();
};
struct SplitReplaceWithQuant : public Action {
  Status Run(Graph&, const NodesToOptimize& selected_nodes) const override;
};

struct MatMulReplaceWithQLinear : public Action {
  MatMulReplaceWithQLinear();

  Status Run(Graph&, const NodesToOptimize& selected_nodes) const override;

 private:
  QDQReplaceWithNew matmul_int_to_float_replacer_;
  BinaryReplaceWithQLinear qlinear_matmul_replacer_;
};

// used together with DQMatMulNodeGroupSelector, which does the sanity check
struct DQMatMulToMatMulNBitsAction : public ReplaceWithNew {
  DQMatMulToMatMulNBitsAction(int64_t accuracy_level,
                              concurrency::ThreadPool* intra_op_thread_pool);

 private:
  std::string OpType(const RuntimeState&) const override { return op_type_; }

  std::string Domain(const RuntimeState&) const override { return domain_; }

  NodeAttributes ExtraAttributes(const RuntimeState&) const override;

  std::vector<NodeAndMoveInfo> ValueMoves(const RuntimeState&) const override { return value_moves_; }

  // transpose initializers, and add to the MatMulNBits inputs
  Status ProcessNewNode(Graph&, const NodesToOptimize&, Node&) const override;

  const int64_t accuracy_level_;
  const std::string domain_;
  const std::string op_type_;
  const std::vector<NodeAndMoveInfo> value_moves_;
  concurrency::ThreadPool* intra_op_thread_pool_;
};

struct GemmReplaceWithQuant : public Action {
  GemmReplaceWithQuant();

  Status Run(Graph&, const NodesToOptimize& selected_nodes) const override;

#if !defined(ORT_MINIMAL_BUILD)
  Status RunForSave(Graph& /*graph*/, const NodesToOptimize& /*selected_nodes*/,
                    const SatRuntimeOptimizationSaveContext& /*save_context*/,
                    SavedState& /*saved_state*/, bool& /*graph_modified*/) const override;
#endif  // !defined(ORT_MINIMAL_BUILD)

  static inline void RemoveAttrBeta(const NodesToOptimize& selected_nodes) {
    selected_nodes.Target().ClearAttribute("beta");
  }

 private:
  QDQReplaceWithNew qgemm_with_float_as_output_replacer_;
  QDQReplaceWithNew qgemm_with_8bits_as_output_replacer_;
};

}  // namespace QDQ
}  // namespace onnxruntime
