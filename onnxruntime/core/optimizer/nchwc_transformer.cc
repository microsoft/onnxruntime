// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nchwc_transformer.h"
#include "core/mlas/inc/mlas.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

class NchwcTransformerImpl {
 public:
  NchwcTransformerImpl(Graph& graph) noexcept : graph_(graph) {}

  void Transform(Node& node);
  void Finalize(bool& modified);

  static constexpr size_t kNchwcDims = 4;

 private:
  // Associate the following state with each created NCHWc output keyed off the
  // original NodeArg.
  struct OutputState {
    struct Shape {
      const NodeArg* dims_[kNchwcDims];
    };

    // Stores the node that generated the NCHWc output.
    Node& output_node_;

    // Stores the NodeArg that represents the NCHWc output. The shape of this
    // tensor is padded to the NCHWc block size.
    NodeArg* output_nchwc_arg_;

    // Stores the proto
    OutputState::Shape output_shape_;

    // Stores the original number of uses for the original NodeArg. Edges are
    // removed from the graph as nodes are converted to NCHWc form.
    const size_t starting_original_use_count_;

    // Stores the remaining number of uses for the original NodeArg. The count
    // is decremented as uses are converted to NCHWc node. Nodes are inserted
    // to reorder the output if this count is non-zero.
    size_t remaining_original_use_count_;

    OutputState(Node& output_node, NodeArg* output_nchwc_arg, size_t original_use_count, const OutputState::Shape& output_shape)
        : output_node_(output_node),
          output_nchwc_arg_(output_nchwc_arg),
          remaining_original_use_count_(original_use_count),
          starting_original_use_count_(original_use_count),
          output_shape_(output_shape) {
    }
  };

  const ONNX_NAMESPACE::AttributeProto* GetAttribute(const Node& node, const char* attribute_name);
  const ONNX_NAMESPACE::AttributeProto* GetIntsAttribute(const Node& node, const char* attribute_name, int expected_size);
  size_t RemoveOutputEdges(Node& node);
  void ConvertOutputDefToNchwc(Node& original_node,
                               Node& nchwc_node,
                               const OutputState::Shape& output_shape);
  void FuseOutputDefToNchwc(Node& original_node,
                            Node& nchwc_node,
                            const OutputState::Shape& output_shape);
  void InsertReorderInput(Node& node);

  void TransformConv(Node& node);
  void TransformPool(Node& node);
  void TransformAdd(Node& node);
  void TransformActivation(Node& node);
  void TransformElementwise(Node& node);

  void ConvPoolShapeInference(const Node& node,
                              const OutputState::Shape& input_shape,
                              OutputState::Shape& output_shape,
                              const ONNX_NAMESPACE::TensorProto* filter_shape);

  Graph& graph_;

  // Stores a queue of nodes to be removed after walking through the graph.
  std::deque<NodeIndex> removed_nodes_;

  // Stores a mapping from the original NodeArg outputs to the NCHWc variants
  // created inside this graph transform.
  std::unordered_map<NodeArg*, std::unique_ptr<OutputState>> nchwc_outputs_;

  // Stores a mapping of NodeArg inputs that have already been reordered, so
  // multiple nodes can share the NCHWc input.
  std::unordered_map<NodeArg*, NodeArg*> reorder_inputs_;
};

const ONNX_NAMESPACE::AttributeProto* NchwcTransformerImpl::GetAttribute(const Node& node, const char* attribute_name) {
  auto& node_attributes = node.GetAttributes();
  auto it = node_attributes.find(attribute_name);
  if (it != node_attributes.end()) {
    return &(it->second);
  } else {
    return nullptr;
  }
}

const ONNX_NAMESPACE::AttributeProto* NchwcTransformerImpl::GetIntsAttribute(const Node& node,
                                                                             const char* attribute_name,
                                                                             int expected_size) {
  auto* attr = GetAttribute(node, attribute_name);
  if (attr != nullptr && attr->ints_size() == expected_size) {
    return attr;
  } else {
    return nullptr;
  }
}

size_t NchwcTransformerImpl::RemoveOutputEdges(Node& node) {
  size_t output_edges_count = node.GetOutputEdgesCount();
  if (output_edges_count > 0) {
    std::vector<Node::EdgeEnd> output_edges;
    output_edges.reserve(output_edges_count);
    for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
      ORT_ENFORCE(it->GetSrcArgIndex() == 0);
      output_edges.push_back(*it);
    }
    for (auto& edge : output_edges) {
      graph_.RemoveEdge(node.Index(), edge.GetNode().Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
    }
  } else {
    output_edges_count = 1;
  }
  return output_edges_count;
}

void NchwcTransformerImpl::ConvertOutputDefToNchwc(Node& original_node,
                                                   Node& nchwc_node,
                                                   const OutputState::Shape& output_shape) {
  size_t original_use_count = RemoveOutputEdges(original_node);

  // Create a new NodeArg to track the output from the NCHWc node.
  auto& output_defs = nchwc_node.MutableOutputDefs();
  auto* output_original_arg = output_defs[0];
  std::string output_reorder_def_name = graph_.GenerateNodeArgName("reorder");
  auto* output_nchwc_arg = &graph_.GetOrCreateNodeArg(output_reorder_def_name, nullptr);
  nchwc_outputs_[output_original_arg] =
      std::make_unique<OutputState>(nchwc_node, output_nchwc_arg, original_use_count, output_shape);
  output_defs[0] = output_nchwc_arg;
}

void NchwcTransformerImpl::FuseOutputDefToNchwc(Node& original_node,
                                                Node& nchwc_node,
                                                const OutputState::Shape& output_shape) {
  size_t original_use_count = RemoveOutputEdges(original_node);

  // Associate the existing NCHWc NodeArg with the output from this node.
  auto* output_original_arg = original_node.MutableOutputDefs()[0];
  auto* output_nchwc_arg = nchwc_node.MutableOutputDefs()[0];
  nchwc_outputs_[output_original_arg] =
      std::make_unique<OutputState>(nchwc_node, output_nchwc_arg, original_use_count, output_shape);
}

void NchwcTransformerImpl::InsertReorderInput(Node& node) {
  auto& input_defs = node.MutableInputDefs();
  auto* input_original_arg = input_defs[0];

  auto it = reorder_inputs_.find(input_original_arg);
  if (it == reorder_inputs_.end()) {
    std::string input_reorder_def_name = graph_.GenerateNodeArgName("reorder");
    auto* input_nchwc_arg = &graph_.GetOrCreateNodeArg(input_reorder_def_name, nullptr);
    reorder_inputs_[input_original_arg] = input_nchwc_arg;
    Node& reorder_input_node = graph_.AddNode(graph_.GenerateNodeName("ReorderInput"),
                                              "ReorderInput",
                                              "ReorderInput",
                                              {input_original_arg},
                                              {input_nchwc_arg},
                                              nullptr,
                                              kMSDomain);
    reorder_input_node.SetExecutionProviderType(node.GetExecutionProviderType());
    input_defs[0] = input_nchwc_arg;
  } else {
    input_defs[0] = it->second;
  }
}

void NchwcTransformerImpl::ConvPoolShapeInference(const Node& node,
                                                  const OutputState::Shape& input_shape,
                                                  OutputState::Shape& output_shape,
                                                  const ONNX_NAMESPACE::TensorProto* filter_shape) {
  auto output_original_arg = node.OutputDefs()[0];

  // Default the shape to preserve the batch count from the input and override
  // the other dimensions. The channel count is also preserved for pooling
  // operations.
  output_shape.dims_[0] = input_shape.dims_[0];
  output_shape.dims_[1] = (filter_shape == nullptr) ? input_shape.dims_[1] : output_original_arg;
  output_shape.dims_[2] = output_original_arg;
  output_shape.dims_[3] = output_original_arg;

  const int kernel_size = kNchwcDims - 2;

  const ONNX_NAMESPACE::AttributeProto* pads_attr = GetIntsAttribute(node, "pads", kernel_size * 2);
  const ONNX_NAMESPACE::AttributeProto* strides_attr = GetIntsAttribute(node, "strides", kernel_size);
  const ONNX_NAMESPACE::AttributeProto* dilations_attr = GetIntsAttribute(node, "dilations", kernel_size);

  auto* auto_pad_attr = GetAttribute(node, "auto_pad");
  if (auto_pad_attr != nullptr && auto_pad_attr->has_s()) {
    auto& auto_pad = auto_pad_attr->s();
    if (auto_pad != "NOTSET") {
      if (auto_pad != "VALID") {
        return;
      }
      pads_attr = nullptr;
    }
  }

  // Require the kernel_shape attribute for pooling operators. Convolution
  // uses the weight tensor shape to derive the kernel shape.
  const ONNX_NAMESPACE::AttributeProto* kernel_shape_attr = nullptr;
  if (filter_shape == nullptr) {
    kernel_shape_attr = GetIntsAttribute(node, "kernel_shape", kernel_size);
    if (kernel_shape_attr == nullptr) {
      return;
    }
  }

  for (int i = 0; i < kernel_size; i++) {
    if ((strides_attr != nullptr && strides_attr->ints(i) != 1) ||
        (dilations_attr != nullptr && dilations_attr->ints(i) != 1)) {
      continue;
    }

    int64_t padding = 0;
    if (pads_attr != nullptr) {
      padding = pads_attr->ints(i) + pads_attr->ints(i + kernel_size);
    }

    int64_t kernel;
    if (kernel_shape_attr != nullptr) {
      kernel = kernel_shape_attr->ints(i);
    } else {
      kernel = filter_shape->dims(2 + i);
    }

    if (padding + 1 == kernel) {
      output_shape.dims_[2 + i] = input_shape.dims_[2 + i];
    }
  }
}

void NchwcTransformerImpl::TransformConv(Node& node) {
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  // Require that the weights tensor be static.
  const ONNX_NAMESPACE::TensorProto* conv_W_tensor_proto = nullptr;
  if (!graph_.GetInitializedTensor(input_defs[1]->Name(), conv_W_tensor_proto) ||
      (conv_W_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) ||
      (conv_W_tensor_proto->dims_size() != 4)) {
    return;
  }

  const int64_t output_channels = conv_W_tensor_proto->dims(0);
  const int64_t input_channels = conv_W_tensor_proto->dims(1);

  int64_t group_count;
  auto* group_attr = GetAttribute(node, "group");
  if (group_attr != nullptr &&
      group_attr->has_i()) {
    group_count = group_attr->i();
  } else {
    group_count = 1;
  }

  const size_t nchwc_block_size = MlasNchwcGetBlockSize();

  bool do_reorder_input = true;
  bool reorder_filter_OIHWBo = false;

  if (group_count > 1) {
    if (input_channels == 1 && output_channels == group_count) {
      // Depthwise convolution.
      reorder_filter_OIHWBo = true;
    } else if (((input_channels % nchwc_block_size) != 0) ||
               ((output_channels % group_count) != 0) ||
               (((output_channels / group_count) % nchwc_block_size) != 0)) {
      return;
    }
  } else {
    if (static_cast<size_t>(input_channels) < nchwc_block_size) {
      // Use NCHW input buffer directly.
      reorder_filter_OIHWBo = true;
      do_reorder_input = false;
    } else if ((input_channels % nchwc_block_size) != 0) {
      return;
    }
  }

  if ((output_channels % nchwc_block_size) != 0) {
    return;
  }

  ONNX_NAMESPACE::TensorProto new_conv_W_tensor_proto(*conv_W_tensor_proto);

  auto conv_W = std::make_unique<Initializer>(conv_W_tensor_proto);
  std::vector<float> reordered_filter(conv_W->size());

  // Reorder the weights tensor statically.
  if (reorder_filter_OIHWBo) {
    MlasReorderFilterOIHWBo(conv_W->dims().data(), conv_W->data<float>(), reordered_filter.data());
  } else {
    MlasReorderFilterOIHWBiBo(conv_W->dims().data(), conv_W->data<float>(), reordered_filter.data());
  }

  new_conv_W_tensor_proto.set_raw_data(reordered_filter.data(), reordered_filter.size() * sizeof(float));

  graph_.RemoveInitializedTensor(input_defs[1]->Name());
  graph_.AddInitializedTensor(new_conv_W_tensor_proto);

  // Create the replacement node.
  std::string nchwc_node_name = graph_.GenerateNodeName("NchwcConv");
  Node& nchwc_node = graph_.AddNode(output_defs[0]->Name() + "_nchwc",
                                    "NchwcConv",
                                    nchwc_node_name,
                                    input_defs,
                                    output_defs,
                                    &node.GetAttributes(),
                                    kMSDomain);
  nchwc_node.SetExecutionProviderType(node.GetExecutionProviderType());

  OutputState::Shape output_shape;
  std::fill_n(output_shape.dims_, kNchwcDims, output_defs[0]);

  if (do_reorder_input) {
    auto nchwc_output = nchwc_outputs_.find(input_defs[0]);
    if (nchwc_output == nchwc_outputs_.end()) {
      InsertReorderInput(nchwc_node);
    } else {
      ConvPoolShapeInference(node, nchwc_output->second->output_shape_, output_shape, &new_conv_W_tensor_proto);
      nchwc_node.MutableInputDefs()[0] = nchwc_output->second->output_nchwc_arg_;
      nchwc_output->second->remaining_original_use_count_--;
    }
  }

  ConvertOutputDefToNchwc(node, nchwc_node, output_shape);
  removed_nodes_.push_front(node.Index());
}

void NchwcTransformerImpl::TransformPool(Node& node) {
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  // Bail out if MaxPool has the optional index tensor specified.
  if (output_defs.size() > 1) {
    return;
  }

  const size_t nchwc_block_size = MlasNchwcGetBlockSize();

  auto* input_shape = input_defs[0]->Shape();
  if (input_shape->dim_size() != 4) {
    return;
  }
  auto& channels_dim = input_shape->dim(1);
  if (!channels_dim.has_dim_value() ||
      ((channels_dim.dim_value() % nchwc_block_size) != 0)) {
    return;
  }

  OutputState::Shape output_shape;
  std::fill_n(output_shape.dims_, kNchwcDims, output_defs[0]);

  // Create the replacement node.
  std::string nchwc_node_name = graph_.GenerateNodeName("NchwcMaxPool");
  Node& nchwc_node = graph_.AddNode(output_defs[0]->Name() + "_nchwc",
                                    "NchwcMaxPool",
                                    nchwc_node_name,
                                    input_defs,
                                    output_defs,
                                    &node.GetAttributes(),
                                    kMSDomain);
  nchwc_node.SetExecutionProviderType(node.GetExecutionProviderType());

  auto nchwc_output = nchwc_outputs_.find(input_defs[0]);
  if (nchwc_output == nchwc_outputs_.end()) {
    InsertReorderInput(nchwc_node);
  } else {
    nchwc_node.MutableInputDefs()[0] = nchwc_output->second->output_nchwc_arg_;
    nchwc_output->second->remaining_original_use_count_--;
  }

  ConvertOutputDefToNchwc(node, nchwc_node, output_shape);
  removed_nodes_.push_front(node.Index());
}

// The existing Add/Sum operator implementations can be used with tensors
// in NCHWc format if the tensor shapes are exactly the same (elementwise
// add).
void NchwcTransformerImpl::TransformAdd(Node& node) {
  auto& input_defs = node.MutableInputDefs();

  // Verify that all of the inputs to this operator are from NCHWc outputs.
  std::vector<OutputState*> nchwc_inputs;
  nchwc_inputs.reserve(input_defs.size());
  for (size_t i = 0; i < input_defs.size(); i++) {
    auto it = nchwc_outputs_.find(input_defs[i]);
    if (it == nchwc_outputs_.end()) {
      return;
    }
    nchwc_inputs.push_back(it->second.get());
  }

  // Test if all of the NCHWc inputs have a compatible shape.
  auto* nchwc_input_0 = nchwc_inputs[0];
  auto* nchwc_input_0_shape = input_defs[0]->Shape();
  for (size_t n = 1; n < input_defs.size(); n++) {
    auto* nchwc_input_n = nchwc_inputs[n];
    for (int i = 0; i < kNchwcDims; i++) {
      // Test if this dimension is derived from the same NodeArg.
      if (nchwc_input_0->output_shape_.dims_[i] != nchwc_input_n->output_shape_.dims_[i]) {
        // Check if ONNX shape inferencing has computed a precise dimension value.
        // Avoid dimension values (<= 1) that could indicate broadcasting.
        auto* nchwc_input_n_shape = input_defs[n]->Shape();
        if ((nchwc_input_0_shape == nullptr) || (nchwc_input_n_shape == nullptr)) {
          return;
        }
        auto& nchwc_input_0_dim = nchwc_input_0_shape->dim(i);
        auto& nchwc_input_n_dim = nchwc_input_n_shape->dim(i);
        if (!nchwc_input_0_dim.has_dim_value() ||
            !nchwc_input_n_dim.has_dim_value() ||
            (nchwc_input_0_dim.dim_value() <= 1) ||
            (nchwc_input_0_dim.dim_value() != nchwc_input_n_dim.dim_value())) {
          return;
        }
      }
    }
  }

  // Update the inputs to the Add/Sum node to directly use the NCHWc inputs
  // and decrement the original use counts of all NCHWc inputs.
  for (size_t n = 0; n < input_defs.size(); n++) {
    input_defs[n] = nchwc_inputs[n]->output_nchwc_arg_;
    nchwc_inputs[n]->remaining_original_use_count_--;
  }

  // If one of the inputs to the Add/Sum node is a NCHWc convolution, then
  // attempt to fuse the addition into the convolution itself.
  if (input_defs.size() == 2) {
    for (size_t n = 0; n < 2; n++) {
      auto* nchwc_input_n = nchwc_inputs[n];
      auto& nchwc_node = nchwc_input_n->output_node_;
      auto& nchwc_input_defs = nchwc_node.MutableInputDefs();
      auto& nchwc_input_args_count = nchwc_node.MutableInputArgsCount();
      // Check if this is a NCHWc convolution. Note that nchwc_node can only
      // be a node that was created by this transformer, so there is no need
      // to also check operator domain and version. The Add/Sum can only be
      // fused if the convolution doesn't have an activation also fused.
      if ((nchwc_node.OpType() == "NchwcConv") &&
          (nchwc_input_defs.size() < 4) && (nchwc_input_args_count.size() < 4) &&
          (nchwc_input_n->starting_original_use_count_ == 1) &&
          (GetAttribute(nchwc_node, "activation") == nullptr)) {
        // Feed the output of the other NCHWc node into the selected convolution
        // node.
        nchwc_input_defs.resize(4);
        nchwc_input_defs[3] = nchwc_inputs[n ^ 1]->output_node_.MutableOutputDefs()[0];
        nchwc_input_args_count.resize(4);
        nchwc_input_args_count[3] = 1;

        FuseOutputDefToNchwc(node, nchwc_node, nchwc_input_n->output_shape_);
        removed_nodes_.push_front(node.Index());
        return;
      }
    }
  }

  ConvertOutputDefToNchwc(node, node, nchwc_input_0->output_shape_);
}

void NchwcTransformerImpl::TransformActivation(Node& node) {
  auto& input_defs = node.MutableInputDefs();

  auto it = nchwc_outputs_.find(input_defs[0]);
  if (it != nchwc_outputs_.end()) {
    auto& nchwc_input = it->second;
    input_defs[0] = nchwc_input->output_nchwc_arg_;
    nchwc_input->remaining_original_use_count_--;

    auto& nchwc_node = nchwc_input->output_node_;
    if (nchwc_node.OpType() == "NchwcConv" &&
        (nchwc_input->starting_original_use_count_ == 1) &&
        (GetAttribute(nchwc_node, "activation") == nullptr)) {
      nchwc_node.AddAttribute("activation", node.OpType());
      FuseOutputDefToNchwc(node, nchwc_node, nchwc_input->output_shape_);
      removed_nodes_.push_front(node.Index());
    } else {
      ConvertOutputDefToNchwc(node, node, nchwc_input->output_shape_);
    }
  }
}

void NchwcTransformerImpl::TransformElementwise(Node& node) {
  auto& input_defs = node.MutableInputDefs();

  auto it = nchwc_outputs_.find(input_defs[0]);
  if (it != nchwc_outputs_.end()) {
    auto& nchwc_output = it->second;
    input_defs[0] = nchwc_output->output_nchwc_arg_;
    nchwc_output->remaining_original_use_count_--;
    ConvertOutputDefToNchwc(node, node, nchwc_output->output_shape_);
  }
}

void NchwcTransformerImpl::Transform(Node& node) {
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Conv", {1}) ||
      graph_utils::IsSupportedOptypeVersionAndDomain(node, "FusedConv", {1}, kMSDomain)) {
    TransformConv(node);
  } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", {1, 8, 10})) {
    TransformPool(node);
  } else if (node.GetInputEdgesCount() == 0) {
    // The following transforms only run when the input edge count has already
    // been decremented to zero by earlier transforms. This is a quick check
    // that all inputs are NCHWc candidates. Also, these transforms do not need
    // to remove any input edges themselves.
    if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Add", {6}) ||
        graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sum", {8})) {
      TransformAdd(node);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Relu", {6})) {
      TransformActivation(node);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Clip", {6})) {
      TransformElementwise(node);
    }
  }
}

void NchwcTransformerImpl::Finalize(bool& modified) {
  // Create ReorderOutput nodes for any NCHWc outputs that still have uses with
  // the original tensor format.
  for (auto& nchwc_output : nchwc_outputs_) {
    if (nchwc_output.second->remaining_original_use_count_ > 0) {
      auto* output_original_arg = nchwc_output.first;
      auto* output_nchwc_arg = nchwc_output.second->output_nchwc_arg_;
      Node& reorder_output_node = graph_.AddNode(graph_.GenerateNodeName("ReorderOutput"),
                                                 "ReorderOutput",
                                                 "ReorderOutput",
                                                 {output_nchwc_arg},
                                                 {output_original_arg},
                                                 nullptr,
                                                 kMSDomain);
      reorder_output_node.SetExecutionProviderType(kCpuExecutionProvider);
    }
  }

  for (auto index : removed_nodes_) {
    auto& node = *graph_.GetNode(index);
    std::vector<Node::EdgeEnd> input_edges;
    input_edges.reserve(node.GetInputEdgesCount());
    for (auto it = node.InputEdgesBegin(); it != node.InputEdgesEnd(); ++it) {
      input_edges.push_back(*it);
    }
    for (auto& edge : input_edges) {
      graph_.RemoveEdge(edge.GetNode().Index(), node.Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
    }

    graph_.RemoveNode(index);
  }

  if (!removed_nodes_.empty()) {
    modified = true;
  }
}

Status NchwcTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  NchwcTransformerImpl impl(graph);
  GraphViewer graph_viewer(graph);
  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));
    if (node.GetExecutionProviderType() == kCpuExecutionProvider) {
      impl.Transform(node);
    }
  }
  impl.Finalize(modified);
  return Status::OK();
}

}  // namespace onnxruntime
