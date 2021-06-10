// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/common/analysis/use_count_analysis.h"

#include "core/codegen/common/common.h"
#include "core/graph/function.h"

namespace onnxruntime {
namespace nuphar {

constexpr int PRESET_USE_COUNT_FOR_UNKNOWN = 10;
constexpr int PRESET_USE_COUNT_FOR_SOFTMAX = 3;

static void CountGemmOp(const onnxruntime::Node& node,
                        const std::vector<const NodeArg*>& graph_inputs,
                        std::function<const ShapeExpr*(const onnxruntime::NodeArg*)> shape_func,
                        std::unordered_map<NodeKey, int>& node_use_counts);

static void CountMatMulOp(const onnxruntime::Node& node,
                          const std::vector<const NodeArg*>& graph_inputs,
                          std::function<const ShapeExpr*(const onnxruntime::NodeArg*)> shape_func,
                          std::unordered_map<NodeKey, int>& node_use_counts);

static void CountRecurrentOp(const onnxruntime::Node& node,
                             const std::vector<const NodeArg*>& graph_inputs,
                             std::function<const ShapeExpr*(const onnxruntime::NodeArg*)> shape_func,
                             std::unordered_map<NodeKey, int>& node_use_counts);

static void CountMatrixArgs(const onnxruntime::NodeArg* A,
                            const onnxruntime::NodeArg* B,
                            const onnxruntime::Node& node,
                            const std::vector<const NodeArg*>& graph_inputs,
                            std::function<const ShapeExpr*(const onnxruntime::NodeArg*)> shape_func,
                            std::unordered_map<NodeKey, int>& node_use_counts);

static void CountNodeArg(const onnxruntime::NodeArg* input_def,
                         const onnxruntime::Node& node,
                         const std::vector<const NodeArg*>& graph_inputs,
                         std::unordered_map<NodeKey, int>& node_use_counts,
                         int use_cnt);

static bool IsMatMulOp(const std::string& op) {
  return op == "MatMul" || op == "MatMulInteger" || op == "MatMulInteger16";
}

void CountGemmOp(const onnxruntime::Node& node,
                 const std::vector<const NodeArg*>& graph_inputs,
                 std::function<const ShapeExpr*(const onnxruntime::NodeArg*)> shape_func,
                 std::unordered_map<NodeKey, int>& node_use_counts) {
  ORT_ENFORCE(node.OpType() == "Gemm");

  auto inputs = node.InputDefs();
  CountMatrixArgs(inputs[0], inputs[1], node, graph_inputs, shape_func, node_use_counts);
  if (inputs.size() > 2) {
    // C's use cnt is fixed.
    CountNodeArg(inputs[2], node, graph_inputs, node_use_counts, 1);
  }
}

void CountMatMulOp(const onnxruntime::Node& node,
                   const std::vector<const NodeArg*>& graph_inputs,
                   std::function<const ShapeExpr*(const onnxruntime::NodeArg*)> shape_func,
                   std::unordered_map<NodeKey, int>& node_use_counts) {
  ORT_ENFORCE(IsMatMulOp(node.OpType()));
  auto inputs = node.InputDefs();
  CountMatrixArgs(inputs[0], inputs[1], node, graph_inputs, shape_func, node_use_counts);
}

void CountRecurrentOp(const onnxruntime::Node& node,
                      const std::vector<const NodeArg*>& graph_inputs,
                      std::function<const ShapeExpr*(const onnxruntime::NodeArg*)>,
                      std::unordered_map<NodeKey, int>& node_use_counts) {
  int use_count = PRESET_USE_COUNT_FOR_UNKNOWN;

  node.ForEachWithIndex(
      node.InputDefs(),
      [&node, &graph_inputs, &node_use_counts, &use_count](const NodeArg& def, size_t) {
        CountNodeArg(&def, node, graph_inputs, node_use_counts, use_count);
        return Status::OK();
      });
}

static bool IsSoftmaxOp(const std::string& op) {
  return op == "Softmax" || op == "LogSoftmax";
}

void CountSoftmaxOp(const onnxruntime::Node& node,
                    const std::vector<const NodeArg*>& graph_inputs,
                    std::function<const ShapeExpr*(const onnxruntime::NodeArg*)>,
                    std::unordered_map<NodeKey, int>& node_use_counts) {
  // Use preset use count for Softmax/LogSoftmax input
  int use_count = PRESET_USE_COUNT_FOR_SOFTMAX;

  node.ForEachWithIndex(
      node.InputDefs(),
      [&node, &graph_inputs, &node_use_counts, &use_count](const NodeArg& def, size_t) {
        CountNodeArg(&def, node, graph_inputs, node_use_counts, use_count);
        return Status::OK();
      });
}

void CountMatrixArgs(const onnxruntime::NodeArg* A,
                     const onnxruntime::NodeArg* B,
                     const onnxruntime::Node& node,
                     const std::vector<const NodeArg*>& graph_inputs,
                     std::function<const ShapeExpr*(const onnxruntime::NodeArg*)> shape_func,
                     std::unordered_map<NodeKey, int>& node_use_counts) {
  int use_cnt = PRESET_USE_COUNT_FOR_UNKNOWN;
  const ShapeExpr* a_shape = shape_func(A);
  if (nullptr != a_shape) {
    // B's use cnt is based on the rows of A
    // skip symbolic dimensions for Sequence and batch
    auto a_cols = (a_shape->Rank() > 0 && a_shape->at(a_shape->Rank() - 1).IsConst()) ? a_shape->at(a_shape->Rank() - 1).Value() : 1;
    use_cnt = a_shape->TotalTailedKnown() / a_cols;
  }
  CountNodeArg(B, node, graph_inputs, node_use_counts, use_cnt);

  // reset use_cnt
  use_cnt = PRESET_USE_COUNT_FOR_UNKNOWN;
  const ShapeExpr* b_shape = shape_func(B);
  if (nullptr != b_shape) {
    const DimExpr& dim = b_shape->Rank() > 1 ? b_shape->at(b_shape->Rank() - 1) : DimExpr(1);
    // A's use cnt is based on the cols of B. If B is 1-D, use cnt is 1
    if (dim.IsConst())
      use_cnt = dim.Value();
  }

  CountNodeArg(A, node, graph_inputs, node_use_counts, use_cnt);
}

void CountNodeArg(const onnxruntime::NodeArg* input_def,
                  const onnxruntime::Node& node,
                  const std::vector<const NodeArg*>& graph_inputs,
                  std::unordered_map<NodeKey, int>& node_use_counts,
                  int use_cnt) {
  // Skip graph's input args nodes
  if (std::find(graph_inputs.begin(), graph_inputs.end(), input_def) != graph_inputs.end())
    return;

  const Node* input_node = GetInputNode(node, input_def);

  if (nullptr != input_node) {
    node_use_counts[GetKey(input_node)] += use_cnt;
  }
}

InternalUseCountAnalysis::InternalUseCountAnalysis(const std::shared_ptr<ShapeExprContext>& shape_inference) {
  shape_func_ = [&shape_inference](const onnxruntime::NodeArg* X) {
    return shape_inference->Lookup(X);
  };
}

void InternalUseCountAnalysis::Traverse(
    const std::vector<const Node*>& nodes,
    const std::vector<const NodeArg*>& graph_inputs,
    const std::vector<const NodeArg*>& graph_outputs) {
  for (auto& node : nodes) {
    auto op_type = node->OpType();
    if (op_type == "Gemm") {
      CountGemmOp(*node, graph_inputs, shape_func_, node_use_counts_);
    } else if (IsMatMulOp(op_type)) {
      CountMatMulOp(*node, graph_inputs, shape_func_, node_use_counts_);
    } else if (op_type == "Scan") {
      auto subgraph = node->GetGraphAttribute("body");
      Evaluate(GraphViewer(*subgraph));
      int use_count = PRESET_USE_COUNT_FOR_UNKNOWN;
      node->ForEachWithIndex(
          node->InputDefs(),
          [this, &node, &graph_inputs, &use_count](const NodeArg& def, size_t) {
            CountNodeArg(&def, *node, graph_inputs, node_use_counts_, use_count);
            return Status::OK();
          });
    } else if (IsRecurrentNode(*node)) {
      CountRecurrentOp(*node, graph_inputs, shape_func_, node_use_counts_);
    } else if (node->NodeType() == Node::Type::Fused) {
      // note: when unboxing subgraph in fused node, use outermost graph input/output
      const auto& func_body = GraphViewer(node->GetFunctionBody()->Body());
      Traverse(ConvertGraphNodesToNodePtrs(func_body.Nodes()), graph_inputs, graph_outputs);
    } else if (IsSoftmaxOp(op_type)) {
      CountSoftmaxOp(*node, graph_inputs, shape_func_, node_use_counts_);
    } else if (op_type != "Shape") {  //don't count on Shape node input, because of no data dependency
      int use_count = 1;
      node->ForEachWithIndex(
          node->InputDefs(),
          [this, &node, &graph_inputs, &use_count](const NodeArg& def, size_t) {
            CountNodeArg(&def, *node, graph_inputs, node_use_counts_, use_count);
            return Status::OK();
          });
    }

    NodeKey key = GetKey(node);
    // For any output_def of the node that is part of graph's outputs but not from graph.Nodes(),
    // we need to increase the node's use cnt accordingly. Otherwise, we would lose those uses.
    node->ForEachWithIndex(
        node->OutputDefs(),
        [this, &graph_outputs, &key](const NodeArg& def, size_t) {
          if (std::find(graph_outputs.begin(), graph_outputs.end(), &def) != graph_outputs.end()) {
            node_use_counts_[key]++;
          }
          return Status::OK();
        });
  }
}

void InternalUseCountAnalysis::Evaluate(const onnxruntime::GraphViewer& graph) {
  const auto& graph_inputs = graph.GetInputs();
  const auto& graph_outputs = graph.GetOutputs();
  Traverse(ConvertGraphNodesToNodePtrs(graph.Nodes()), graph_inputs, graph_outputs);
}

void InternalUseCountAnalysis::Evaluate(const onnxruntime::nuphar::NupharSubgraphUnit& graph) {
  const auto& graph_inputs = graph.inputs;
  const auto& graph_outputs = graph.outputs;
  Traverse(graph.nodes, graph_inputs, graph_outputs);
}

void InternalUseCountAnalysis::IncrementCount(const onnxruntime::NodeArg* def) {
  node_use_counts_[GetKey(def)]++;
}

int InternalUseCountAnalysis::NodeUseCount(const onnxruntime::Node* node) const {
  auto node_iter = node_use_counts_.find(GetKey(node));
  if (node_iter != node_use_counts_.end()) {
    return node_iter->second;
  } else {
    return 0;
  }
}

OrtUseCountAnalysis::OrtUseCountAnalysis(const std::shared_ptr<ShapeExprContext>& shape_inference)
    : OrtAnalysis("OrtUseCountAnalysis") {
  internal_analysis_ = std::make_unique<InternalUseCountAnalysis>(shape_inference);
}

void OrtUseCountAnalysis::Evaluate(const onnxruntime::GraphViewer& graph) {
  internal_analysis_->Evaluate(graph);
}

void OrtUseCountAnalysis::IncrementCount(const onnxruntime::NodeArg* def) {
  internal_analysis_->IncrementCount(def);
}

int OrtUseCountAnalysis::NodeUseCount(const onnxruntime::Node* node) const {
  return internal_analysis_->NodeUseCount(node);
}

NupharUseCountAnalysis::NupharUseCountAnalysis(const std::shared_ptr<ShapeExprContext>& shape_inference)
    : NupharAnalysis("NupharUseCountAnalysis") {
  internal_analysis_ = std::make_unique<InternalUseCountAnalysis>(shape_inference);
}

void NupharUseCountAnalysis::Evaluate(const onnxruntime::nuphar::NupharSubgraphUnit& graph) {
  internal_analysis_->Evaluate(graph);
}

void NupharUseCountAnalysis::IncrementCount(const onnxruntime::NodeArg* def) {
  internal_analysis_->IncrementCount(def);
}

int NupharUseCountAnalysis::NodeUseCount(const onnxruntime::Node* node) const {
  return internal_analysis_->NodeUseCount(node);
}

}  // namespace nuphar
}  // namespace onnxruntime
