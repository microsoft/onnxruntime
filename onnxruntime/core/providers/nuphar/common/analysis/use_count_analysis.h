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

// TODO change namespace from codegen to nuphar

namespace onnxruntime {
namespace nuphar {

class InternalUseCountAnalysis {
 public:
  InternalUseCountAnalysis(const std::shared_ptr<ShapeExprContext>& shape_inference);

  ~InternalUseCountAnalysis() = default;

  void Evaluate(const onnxruntime::GraphViewer& graph);

  void Evaluate(const NupharSubgraphUnit& graph);

  void IncrementCount(const onnxruntime::NodeArg* arg);

  int NodeUseCount(const onnxruntime::Node* node) const;

 private:
  void Traverse(gsl::span<const Node* const> nodes,
                gsl::span<const NodeArg* const> graph_inputs,
                gsl::span<const NodeArg* const> graph_outputs);

  std::unordered_map<NodeKey, int> node_use_counts_;
  std::function<const ShapeExpr*(const onnxruntime::NodeArg*)> shape_func_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(InternalUseCountAnalysis);
};

// TODO analysis move to namespace nuphar

class OrtUseCountAnalysis : public OrtAnalysis {
 public:
  OrtUseCountAnalysis(const std::shared_ptr<ShapeExprContext>& shape_inference);
  ~OrtUseCountAnalysis() = default;

  void Evaluate(const onnxruntime::GraphViewer& graph) override;

  void IncrementCount(const onnxruntime::NodeArg* arg);

  int NodeUseCount(const onnxruntime::Node* node) const;

 private:
  std::unique_ptr<InternalUseCountAnalysis> internal_analysis_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OrtUseCountAnalysis);
};

class NupharUseCountAnalysis : public NupharAnalysis {
 public:
  NupharUseCountAnalysis(const std::shared_ptr<ShapeExprContext>& shape_inference);

  ~NupharUseCountAnalysis() = default;

  void Evaluate(const onnxruntime::nuphar::NupharSubgraphUnit& graph) override;

  void IncrementCount(const onnxruntime::NodeArg* arg);

  int NodeUseCount(const onnxruntime::Node* node) const;

 private:
  std::unique_ptr<InternalUseCountAnalysis> internal_analysis_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NupharUseCountAnalysis);
};

}  // namespace nuphar
}  // namespace onnxruntime
