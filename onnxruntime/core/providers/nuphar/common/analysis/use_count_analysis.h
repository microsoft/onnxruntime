// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/common.h"
#include "core/providers/nuphar/common/analysis/analysis.h"
#include "core/providers/nuphar/common/analysis/shape_expr.h"
#include "core/providers/nuphar/compiler/traverse_shape_infer.h"
#include "core/graph/graph.h"

#include <functional>
#include <unordered_map>

// TODO analysis move to nuphar

namespace onnxruntime {
namespace codegen {

class UseCountAnalysis : public OrtAnalysis {
 public:
  UseCountAnalysis(const std::shared_ptr<ShapeExprContext>& shape_inference);

  ~UseCountAnalysis() = default;

  void Evaluate(const onnxruntime::GraphViewer& graph) override;

  void IncrementCount(const onnxruntime::NodeArg* arg);

  int NodeUseCount(const onnxruntime::Node* node) const;

 private:
  std::unordered_map<NodeKey, int> node_use_counts_;
  std::function<const ShapeExpr*(const onnxruntime::NodeArg*)> shape_func_;

  // TODO: move these to source as make them local functions
  // local utility functions for analyze specific Node or NodeArg
  void CountGemmOp(const onnxruntime::Node& node,
                   const std::vector<const NodeArg*>& graph_inputs);

  void CountMatMulOp(const onnxruntime::Node& node,
                     const std::vector<const NodeArg*>& graph_inputs);

  void CountLSTMOp(const onnxruntime::Node& node,
                   const std::vector<const NodeArg*>& graph_inputs);

  void CountMatrixArgs(const onnxruntime::NodeArg* A,
                       const onnxruntime::NodeArg* B,
                       const onnxruntime::Node& node,
                       const std::vector<const NodeArg*>& graph_inputs);

  void CountNodeArg(const onnxruntime::NodeArg* input_def,
                    const onnxruntime::Node& node,
                    const std::vector<const NodeArg*>& graph_inputs,
                    int use_cnt);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(UseCountAnalysis);
};

}  // namespace codegen
}  // namespace onnxruntime
