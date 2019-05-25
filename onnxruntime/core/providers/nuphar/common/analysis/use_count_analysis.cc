// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "use_count_analysis.h"
#include "core/graph/function.h"

// TODO analysis move to nuphar
namespace onnxruntime {
namespace codegen {

// explicit instantiation
//template class AnalysisBase<const onnxruntime::GraphViewer&>;

// This file contains old contents from old GraphStats.
// It will be removed after refactoring step 13
// So no need to do detailed review.

UseCountAnalysis::UseCountAnalysis(const std::shared_ptr<ShapeExprContext>& shape_inference)
    : OrtAnalysis("UseCountAnalysis") {
  shape_func_ = [&shape_inference](const onnxruntime::NodeArg* X) {
    return shape_inference->Lookup(X);
  };
}

void UseCountAnalysis::Evaluate(const onnxruntime::GraphViewer& graph) {
  const auto& graph_inputs = graph.GetInputs();
  const auto& graph_outputs = graph.GetOutputs();

  using TraverseFunc = std::function<void(const onnxruntime::GraphViewer&)>;
  TraverseFunc traverse = [&](const onnxruntime::GraphViewer& g) {
    for (auto& node : g.Nodes()) {
      auto op_type = node.OpType();
      if (op_type == "Gemm") {
        CountGemmOp(node, graph_inputs);
      } else if (op_type == "MatMul" || op_type == "MatMulInteger") {
        CountMatMulOp(node, graph_inputs);
      } else if (op_type == "Scan") {
        auto subgraph = node.GetGraphAttribute("body");
        Evaluate(GraphViewer(*subgraph));
        node.ForEachWithIndex(
            node.InputDefs(),
            [this, &node, &graph_inputs](const NodeArg& def, size_t) {
              CountNodeArg(&def, node, graph_inputs, /*use_cnt=*/2);  // change it 2
              return Status::OK();
            });
      } else if (IsRecurrentNode(node)) {
        node.ForEachWithIndex(
            node.InputDefs(),
            [this, &node, &graph_inputs](const NodeArg& def, size_t) {
              CountNodeArg(&def, node, graph_inputs, /*use_cnt=*/2);  // change it 2
              return Status::OK();
            });
      } else if (node.NodeType() == Node::Type::Fused) {
        // note: when unboxing subgraph in fused node, use outermost graph input/output
        traverse(GraphViewer(node.GetFunctionBody()->Body()));
      } else {
        node.ForEachWithIndex(
            node.InputDefs(),
            [this, &node, &graph_inputs](const NodeArg& def, size_t) {
              CountNodeArg(&def, node, graph_inputs, /*use_cnt=*/1);
              return Status::OK();
            });
      }

      NodeKey key = GetKey(&node);
      // For any output_def of the node that is part of graph's outputs but not from graph.Nodes(),
      // we need to increase the node's use cnt accordingly. Otherwise, we would lose those uses.
      node.ForEachWithIndex(
          node.OutputDefs(),
          [this, &graph_outputs, &key](const NodeArg& def, size_t) {
            if (std::find(graph_outputs.begin(), graph_outputs.end(), &def) != graph_outputs.end()) {
              node_use_counts_[key]++;
            }
            return Status::OK();
          });
    }
  };

  traverse(graph);
}

void UseCountAnalysis::IncrementCount(const onnxruntime::NodeArg* def) {
  node_use_counts_[GetKey(def)]++;
}

void UseCountAnalysis::CountGemmOp(const onnxruntime::Node& node,
                                   const std::vector<const NodeArg*>& graph_inputs) {
  ORT_ENFORCE(node.OpType() == "Gemm");

  auto inputs = node.InputDefs();
  CountMatrixArgs(inputs[0], inputs[1], node, graph_inputs);
  // C's use cnt is fixed.
  CountNodeArg(inputs[2], node, graph_inputs, 1);
}

void UseCountAnalysis::CountMatMulOp(const onnxruntime::Node& node,
                                     const std::vector<const NodeArg*>& graph_inputs) {
  ORT_ENFORCE(node.OpType() == "MatMul" || node.OpType() == "MatMulInteger");
  auto inputs = node.InputDefs();
  CountMatrixArgs(inputs[0], inputs[1], node, graph_inputs);
}

void UseCountAnalysis::CountLSTMOp(const onnxruntime::Node& node,
                                   const std::vector<const NodeArg*>& graph_inputs) {
  // TODO add Recurrent support
}

void UseCountAnalysis::CountMatrixArgs(const onnxruntime::NodeArg* A,
                                       const onnxruntime::NodeArg* B,
                                       const onnxruntime::Node& node,
                                       const std::vector<const NodeArg*>& graph_inputs) {
  const int default_use_cnt = 10;
  const ShapeExpr* a_shape = shape_func_(A);
  if (a_shape) {
    // B's use cnt is based on the rows of A
    // skip symbolic dimensions for Sequence and batch
    auto a_cols = (a_shape->Rank() > 0 && a_shape->at(a_shape->Rank() - 1).IsConst()) ? a_shape->at(a_shape->Rank() - 1).Value() : 1;
    CountNodeArg(B, node, graph_inputs, /*use_cnt=*/a_shape->TotalTailedKnown() / a_cols);
  } else {
    CountNodeArg(B, node, graph_inputs, default_use_cnt);
  }

  const ShapeExpr* b_shape = shape_func_(B);
  if (b_shape) {
    const DimExpr& dim = b_shape->Rank() > 1 ? b_shape->at(b_shape->Rank() - 1) : DimExpr(1);
    // A's use cnt is based on the cols of B. If B is 1-D, use cnt is 1
    CountNodeArg(A, node, graph_inputs, /*use_cnt=*/dim.Value());
  } else {
    CountNodeArg(A, node, graph_inputs, default_use_cnt);
  }
}

void UseCountAnalysis::CountNodeArg(const onnxruntime::NodeArg* input_def,
                                    const onnxruntime::Node& node,
                                    const std::vector<const NodeArg*>& graph_inputs,
                                    int use_cnt) {
  // Because we are not going to cut graph inputs nodes, just skip them
  if (std::find(graph_inputs.begin(), graph_inputs.end(), input_def) != graph_inputs.end())
    return;

  const Node* input_node = GetInputNode(node, input_def);

  if (nullptr != input_node) {
    node_use_counts_[GetKey(input_node)] += use_cnt;
  }
}

int UseCountAnalysis::NodeUseCount(const onnxruntime::Node* node) const {
  auto node_iter = node_use_counts_.find(GetKey(node));
  if (node_iter != node_use_counts_.end()) {
    return node_iter->second;
  } else {
    return 0;
  }
}

}  // namespace codegen
}  // namespace onnxruntime
