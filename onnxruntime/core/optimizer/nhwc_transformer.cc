// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nhwc_transformer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

class NhwcTransformerImpl {
 public:
  NhwcTransformerImpl(Graph& graph) noexcept : graph_(graph) {}

  void Transform(Node& node);
  void Finalize(bool& modified);

 private:
  struct NhwcArgument {
    Node& output_node_;
    NodeArg* nhwc_arg_;
    const size_t starting_original_uses_;
    size_t remaining_original_uses_;
    int rank_;

    NhwcArgument(Node& output_node, NodeArg* output_nhwc_arg, size_t original_uses, int rank)
        : output_node_(output_node),
          nhwc_arg_(output_nhwc_arg),
          starting_original_uses_(original_uses),
          remaining_original_uses_(original_uses),
          rank_(rank) {
    }
  };

  NhwcArgument* LookupNhwcArgument(NodeArg* arg) {
    auto it = nhwc_args_.find(arg);
    return (it != nhwc_args_.end()) ? it->second.get() : nullptr;
  }

  size_t RemoveOutputEdges(Node& node);
  void CreateNhwcArgument(Node& node, Node& nhwc_node, int rank);
  void InsertReorderInput(Node& node, int rank);

  void TransformQLinearConv(Node& node);
  void TransformQLinearBinary(Node& node);
  void TransformQLinearActivation(Node& node);

  Graph& graph_;

  // Stores a mapping from the original NodeArg outputs to the NHWC variants
  // created inside this graph transform.
  std::unordered_map<NodeArg*, std::unique_ptr<NhwcArgument>> nhwc_args_;

  // Stores a mapping of NodeArg inputs that have already been reordered, so
  // multiple nodes can share the NHWC input.
  std::unordered_map<NodeArg*, NodeArg*> reorder_inputs_;

  // Stores a queue of nodes to be removed after walking through the graph.
  std::deque<NodeIndex> removed_nodes_;
};

size_t NhwcTransformerImpl::RemoveOutputEdges(Node& node) {
  size_t output_edges_count = node.GetOutputEdgesCount();
  if (output_edges_count > 0) {
    graph_utils::RemoveNodeOutputEdges(graph_, node);
  }
  // Bias the edge count to handle the case of a node that produces a graph
  // output.
  if (!graph_.GetNodeOutputsInGraphOutputs(node).empty()) {
    output_edges_count++;
  }
  return output_edges_count;
}

void NhwcTransformerImpl::CreateNhwcArgument(Node& node, Node& nhwc_node, int rank) {
  size_t original_uses = RemoveOutputEdges(node);

  // Create a new NodeArg to track the output from the NHWC node.
  auto& output_defs = nhwc_node.MutableOutputDefs();
  auto* output_original_arg = output_defs[0];
  std::string output_reorder_def_name = graph_.GenerateNodeArgName("reorder");
  auto* output_nhwc_arg = &graph_.GetOrCreateNodeArg(output_reorder_def_name, nullptr);
  nhwc_args_[output_original_arg] =
      onnxruntime::make_unique<NhwcArgument>(nhwc_node, output_nhwc_arg, original_uses, rank);
  output_defs[0] = output_nhwc_arg;
}

void NhwcTransformerImpl::InsertReorderInput(Node& node, int rank) {
  auto& input_defs = node.MutableInputDefs();
  auto* input_original_arg = input_defs[0];

  auto it = reorder_inputs_.find(input_original_arg);
  if (it == reorder_inputs_.end()) {
    std::string input_reorder_def_name = graph_.GenerateNodeArgName("reorder");
    auto* input_nhwc_arg = &graph_.GetOrCreateNodeArg(input_reorder_def_name, nullptr);
    reorder_inputs_[input_original_arg] = input_nhwc_arg;
    Node& reorder_input_node = graph_.AddNode(graph_.GenerateNodeName("ReorderInput"),
                                              "Transpose",
                                              "ReorderInput",
                                              {input_original_arg},
                                              {input_nhwc_arg},
                                              nullptr);
    reorder_input_node.SetExecutionProviderType(kCpuExecutionProvider);

    // Build the permute vector, example: {0, 2, 3, 1}
    std::vector<int64_t> perm(static_cast<size_t>(rank));
    perm[rank - 1] = 1;
    for (auto r = 2; r < rank; r++) {
      perm[r - 1] = r;
    }
    reorder_input_node.AddAttribute("perm", perm);

    input_defs[0] = input_nhwc_arg;
  } else {
    input_defs[0] = it->second;
  }
}

void NhwcTransformerImpl::TransformQLinearConv(Node& node) {
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  // Require that the weights tensor have a shape so that the necessary
  // Transpose nodes can be inserted into the graph.
  auto* weights_shape = input_defs[3]->Shape();
  if (weights_shape == nullptr) {
    return;
  }

  // If the output is immediately dequantized, then skip wrapping QLinearConv
  // with Transpose nodes and use the NCHW variant that does this internally.
  auto* nhwc_input = LookupNhwcArgument(input_defs[0]);
  if (nhwc_input == nullptr) {
    if (optimizer_utils::CheckOutputEdges(graph_, node, 1)) {
      const auto& next_node = *node.OutputNodesBegin();
      if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "DequantizeLinear", {10, 13})) {
        return;
      }
    }
  }

  // Create the replacement node.
  std::string nhwc_node_name = graph_.GenerateNodeName(output_defs[0]->Name() + "_nhwc");
  Node& nhwc_node = graph_.AddNode(nhwc_node_name,
                                   "QLinearConv",
                                   nhwc_node_name,
                                   input_defs,
                                   output_defs,
                                   &node.GetAttributes(),
                                   kMSDomain);
  nhwc_node.SetExecutionProviderType(kCpuExecutionProvider);
  nhwc_node.AddAttribute("channels_last", static_cast<int64_t>(1));

  if (nhwc_input == nullptr) {
    InsertReorderInput(nhwc_node, weights_shape->dim_size());
  } else {
    nhwc_node.MutableInputDefs()[0] = nhwc_input->nhwc_arg_;
    nhwc_input->remaining_original_uses_--;
  }

  CreateNhwcArgument(node, nhwc_node, weights_shape->dim_size());
  removed_nodes_.push_front(node.Index());
}

void NhwcTransformerImpl::TransformQLinearBinary(Node& node) {
  auto& input_defs = node.MutableInputDefs();

  auto* input_def_a = input_defs[0];
  auto* input_def_b = input_defs[3];

  // For simplicity, require that both inputs have the same tensor rank.
  auto* input_shape_a = input_def_a->Shape();
  auto* input_shape_b = input_def_b->Shape();
  if (input_shape_a == nullptr || input_shape_b == nullptr) {
    return;
  }
  if (input_shape_a->dim_size() != input_shape_b->dim_size()) {
    return;
  }

  auto* nhwc_input_a = LookupNhwcArgument(input_def_a);
  auto* nhwc_input_b = LookupNhwcArgument(input_def_b);
  if (nhwc_input_a == nullptr || nhwc_input_b == nullptr) {
    return;
  }

  // Update the node to directly use the NHWC inputs and decrement the original
  // use counts of the NHWC inputs.
  input_defs[0] = nhwc_input_a->nhwc_arg_;
  nhwc_input_a->remaining_original_uses_--;
  input_defs[3] = nhwc_input_b->nhwc_arg_;
  nhwc_input_b->remaining_original_uses_--;

  CreateNhwcArgument(node, node, nhwc_input_a->rank_);
}

void NhwcTransformerImpl::TransformQLinearActivation(Node& node) {
  auto& input_defs = node.MutableInputDefs();

  auto* nhwc_input = LookupNhwcArgument(input_defs[0]);
  if (nhwc_input == nullptr) {
    return;
  }

  // Update the node to directly use the NHWC inputs and decrement the original
  // use counts of the NHWC inputs.
  input_defs[0] = nhwc_input->nhwc_arg_;
  nhwc_input->remaining_original_uses_--;

  CreateNhwcArgument(node, node, nhwc_input->rank_);
}

void NhwcTransformerImpl::Transform(Node& node) {
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "QLinearConv", {10})) {
    TransformQLinearConv(node);
  } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "QLinearAdd", {1}, kMSDomain) ||
             graph_utils::IsSupportedOptypeVersionAndDomain(node, "QLinearMul", {1}, kMSDomain)) {
    TransformQLinearBinary(node);
  } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "QLinearLeakyRelu", {1}, kMSDomain) ||
             graph_utils::IsSupportedOptypeVersionAndDomain(node, "QLinearSigmoid", {1}, kMSDomain)) {
    TransformQLinearActivation(node);
  }
}

void NhwcTransformerImpl::Finalize(bool& modified) {
  // Create ReorderOutput nodes for any NHWC outputs that still have uses with
  // the original tensor format.
  for (auto& nhwc_output : nhwc_args_) {
    if (nhwc_output.second->remaining_original_uses_ > 0) {
      auto* output_original_arg = nhwc_output.first;
      auto* output_nhwc_arg = nhwc_output.second->nhwc_arg_;
      int rank = nhwc_output.second->rank_;
      Node& reorder_output_node = graph_.AddNode(graph_.GenerateNodeName("ReorderOutput"),
                                                 "Transpose",
                                                 "ReorderOutput",
                                                 {output_nhwc_arg},
                                                 {output_original_arg},
                                                 nullptr);
      reorder_output_node.SetExecutionProviderType(kCpuExecutionProvider);

      // Build the permute vector, example: {0, 3, 1, 2}
      std::vector<int64_t> perm(static_cast<size_t>(rank));
      perm[1] = rank - 1;
      for (auto r = 2; r < rank; r++) {
        perm[r] = r - 1;
      }
      reorder_output_node.AddAttribute("perm", perm);
    }
  }

  for (auto index : removed_nodes_) {
    graph_.RemoveNode(index);
  }

  if (!removed_nodes_.empty()) {
    modified = true;
  }
}

Status NhwcTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  NhwcTransformerImpl impl(graph);
  GraphViewer graph_viewer(graph);

  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
    if (node.GetExecutionProviderType() == kCpuExecutionProvider) {
      impl.Transform(node);
    }
  }
  impl.Finalize(modified);
  return Status::OK();
}

}  // namespace onnxruntime
