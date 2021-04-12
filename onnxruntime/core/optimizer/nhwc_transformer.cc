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

  size_t RemoveOutputEdge(Node& node, size_t output_index);
  void CreateNhwcArgument(Node& node, Node& nhwc_node, int rank, size_t output_index);
  void CreateNhwcArgument(Node& node, Node& nhwc_node, int rank);
  void InsertReorderInput(Node& node, int rank);

  void TransformQLinearConv(Node& node);
  void TransformQLinearBinary(Node& node);
  void TransformQLinearActivation(Node& node);
  void TransformQLinearGlobalAveragePool(Node& node);
  void TransformMaxPool(Node& node);
  void TransformSplit(Node& node);
  void TransformPad(Node& node);

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

// Remove node's output edge starting from specified index, return number of edges removed.
// If output at specified index for the node is graph output, inc the count returned.
size_t NhwcTransformerImpl::RemoveOutputEdge(Node& node, size_t output_index) {
  size_t output_edges_count = graph_utils::RemoveNodeOutputEdges(graph_, node, static_cast<int>(output_index));

  // Bias the edge count to if the node produces a graph output at output_index.
  auto node_outputs_for_graph = graph_.GetNodeOutputsInGraphOutputs(node);
  for (auto idx : node_outputs_for_graph) {
    if (idx == static_cast<int>(output_index)) {
      output_edges_count++;
      break;
    }
  }
  return output_edges_count;
}

void NhwcTransformerImpl::CreateNhwcArgument(Node& node, Node& nhwc_node, int rank, size_t output_index) {
  size_t original_uses = RemoveOutputEdge(node, output_index);

  // Create a new NodeArg to track the output from the NHWC node.
  auto& output_defs = nhwc_node.MutableOutputDefs();
  auto* output_original_arg = output_defs[output_index];
  std::string output_reorder_def_name = graph_.GenerateNodeArgName("reorder");
  auto* output_nhwc_arg = &graph_.GetOrCreateNodeArg(output_reorder_def_name, nullptr);
  nhwc_args_[output_original_arg] =
      onnxruntime::make_unique<NhwcArgument>(nhwc_node, output_nhwc_arg, original_uses, rank);
  output_defs[output_index] = output_nhwc_arg;
}

void NhwcTransformerImpl::CreateNhwcArgument(Node& node, Node& nhwc_node, int rank) {
  size_t output_count = node.OutputDefs().size();
  for (size_t output_index = 0; output_index < output_count; ++output_index) {
    CreateNhwcArgument(node, nhwc_node, rank, output_index);
  }
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

void NhwcTransformerImpl::TransformQLinearGlobalAveragePool(Node& node) {
  auto& input_defs = node.MutableInputDefs();

  auto* nhwc_input = LookupNhwcArgument(input_defs[0]);
  if (nhwc_input == nullptr) {
    return;
  }

  // Verify that the node is using NCHW tensors.
  const auto* channels_last_attr = graph_utils::GetNodeAttribute(node, "channels_last");
  if (channels_last_attr != nullptr && utils::HasInt(*channels_last_attr) && channels_last_attr->i() != 0) {
    return;
  }

  // Update the node to directly use the NHWC inputs and decrement the original
  // use counts of the NHWC inputs.
  input_defs[0] = nhwc_input->nhwc_arg_;
  nhwc_input->remaining_original_uses_--;
  node.AddAttribute("channels_last", static_cast<int64_t>(1));

  CreateNhwcArgument(node, node, nhwc_input->rank_);
}

void NhwcTransformerImpl::TransformMaxPool(Node& node) {
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  // Bail out if MaxPool has the optional index tensor specified.
  if (output_defs.size() > 1) {
    return;
  }

  auto* nhwc_input = LookupNhwcArgument(input_defs[0]);
  if (nhwc_input == nullptr) {
    return;
  }

  // Create the replacement node.
  std::string nhwc_node_name = graph_.GenerateNodeName(output_defs[0]->Name() + "_nhwc");
  Node& nhwc_node = graph_.AddNode(nhwc_node_name,
                                   "NhwcMaxPool",
                                   nhwc_node_name,
                                   input_defs,
                                   output_defs,
                                   &node.GetAttributes(),
                                   kMSDomain);
  nhwc_node.SetExecutionProviderType(kCpuExecutionProvider);

  // Remove the storage_order attribute, used for the unsupported index output tensor.
  nhwc_node.ClearAttribute("storage_order");

  // Update the node to directly use the NHWC inputs and decrement the original
  // use counts of the NHWC inputs.
  nhwc_node.MutableInputDefs()[0] = nhwc_input->nhwc_arg_;
  nhwc_input->remaining_original_uses_--;

  CreateNhwcArgument(node, nhwc_node, nhwc_input->rank_);
  removed_nodes_.push_front(node.Index());
}

void NhwcTransformerImpl::TransformSplit(Node& node) {
  auto& input_defs = node.MutableInputDefs();

  auto* nhwc_input = LookupNhwcArgument(input_defs[0]);
  if (nhwc_input == nullptr) {
    return;
  }

  // Change the axis attribute accordingly for NCHW to NHWC model.
  const auto* axis_attr = graph_utils::GetNodeAttribute(node, "axis");
  if (axis_attr != nullptr && utils::HasInt(*axis_attr)) {
    int64_t axis = axis_attr->i();
    if (axis < -nhwc_input->rank_ || axis >= nhwc_input->rank_) {
      // direct return on invalid axis
      return;
    }
    if (axis < 0) {
      axis = axis + nhwc_input->rank_;
    }
    if (axis == 1) {
      axis = nhwc_input->rank_ - 1;
    } else if (axis > 1) {
      axis = axis - 1;
    }
    node.AddAttribute("axis", axis);
  }

  // Update the node to directly use the NHWC inputs and decrement the original
  // use counts of the NHWC inputs.
  input_defs[0] = nhwc_input->nhwc_arg_;
  nhwc_input->remaining_original_uses_--;

  CreateNhwcArgument(node, node, nhwc_input->rank_);
}

void NhwcTransformerImpl::TransformPad(Node& node) {
  auto& input_defs = node.MutableInputDefs();

  auto* nhwc_input = LookupNhwcArgument(input_defs[0]);
  if (nhwc_input == nullptr) {
    return;
  }

  const ONNX_NAMESPACE::TensorProto* pads_tensor_proto = nullptr;
  if (!graph_utils::NodeArgIsConstant(graph_, *input_defs[1]) ||
      !graph_.GetInitializedTensor(input_defs[1]->Name(), pads_tensor_proto) ||
      (pads_tensor_proto->dims_size() != 1) ||
      (pads_tensor_proto->dims(0) != nhwc_input->rank_ * 2) ||
      (nhwc_input->rank_ <= 2)) {  // nc only, no any hw axises
    return;
  }

  // perm nchw to nhwc on pad tensor
  Initializer pads_initializer{*pads_tensor_proto, graph_.ModelPath()};
  const int64_t* nchw_pads_data = pads_initializer.data<int64_t>();
  size_t n_dim = static_cast<size_t>(pads_tensor_proto->dims(0)) / 2;
  std::vector<int64_t> nhwc_pads(nchw_pads_data, nchw_pads_data + pads_tensor_proto->dims(0));
  std::copy_n(nchw_pads_data + 2, n_dim - 2, nhwc_pads.data() + 1);
  std::copy_n(nchw_pads_data + 2 + n_dim, n_dim - 2, nhwc_pads.data() + 1 + n_dim);
  nhwc_pads[n_dim - 1] = nchw_pads_data[1];
  nhwc_pads[2 * n_dim - 1] = nchw_pads_data[n_dim + 1];

  ONNX_NAMESPACE::TensorProto nhwc_pads_tensor_proto;
  nhwc_pads_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  nhwc_pads_tensor_proto.set_name(graph_.GenerateNodeArgName("nhwc_permutated_pads"));
  nhwc_pads_tensor_proto.set_raw_data(nhwc_pads.data(), n_dim * 2 * sizeof(int64_t));
  nhwc_pads_tensor_proto.add_dims(n_dim * 2);
  NodeArg* nhwc_pads_arg = &graph_utils::AddInitializer(graph_, nhwc_pads_tensor_proto);

  // Update the node to directly use the NHWC inputs and decrement the original
  // use counts of the NHWC inputs.
  input_defs[1] = nhwc_pads_arg;
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
  } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "QLinearGlobalAveragePool", {1}, kMSDomain)) {
    TransformQLinearGlobalAveragePool(node);
  } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", {12})) {
    TransformMaxPool(node);
  } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Split", {2, 11, 13})) {
    TransformSplit(node);
  } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Pad", {11, 13})) {
    TransformPad(node);
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
