// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/selectors_actions/actions.h"

namespace onnxruntime {

class Graph;
class Node;

namespace QDQ {

// helper that sets optional zero point values before replacing a node
struct QDQReplaceWithNew : public ReplaceWithNew {
  QDQReplaceWithNew(const std::string& domain,
                    std::vector<NodeAndMoveInfo>&& value_moves,
                    const std::string& op_name)
      : ReplaceWithNew{domain, op_name, std::move(value_moves)} {}

  Status Run(Graph&, const NodesToOptimize& selected_nodes) const override;
};

// replace node with QLinear version
struct ReplaceWithQLinear : public QDQReplaceWithNew {
  // provide NodeLocation for source node, and ValueMoveInfo for the value to move to the replacement node
  ReplaceWithQLinear(const std::string& domain,
                     std::vector<NodeAndMoveInfo>&& value_moves)
      : QDQReplaceWithNew{domain, std::move(value_moves), "generated at runtime"} {}

 private:
  std::string OpType(const NodesToOptimize& selected_nodes) const override {
    return "QLinear" + selected_nodes.Target().OpType();
  }
};

struct UnaryReplaceWithQLinear : ReplaceWithQLinear {
  UnaryReplaceWithQLinear(const std::string& domain);
};

struct BinaryReplaceWithQLinear : ReplaceWithQLinear {
  BinaryReplaceWithQLinear(const std::string& domain);
};

struct VariadicReplaceWithQLinear : ReplaceWithQLinear {
  VariadicReplaceWithQLinear(const std::string& domain);
};

struct ConvReplaceWithQLinear : ReplaceWithQLinear {
  ConvReplaceWithQLinear();
};

struct MatMulReplaceWithQLinear : public Action {
  MatMulReplaceWithQLinear();

  Status Run(Graph&, const NodesToOptimize& selected_nodes) const override;

 private:
  QDQReplaceWithNew matmul_int_to_float_replacer_;
  BinaryReplaceWithQLinear qlinear_matmul_replacer_;
};

}  // namespace QDQ
}  // namespace onnxruntime
