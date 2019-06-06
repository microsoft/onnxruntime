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
  struct NchwcArgument {
    struct Shape {
      const NodeArg* dims_[kNchwcDims];
    };

    // Stores the node that generated the NCHWc output.
    Node& output_node_;

    // Stores the NodeArg that represents the NCHWc output.
    NodeArg* nchwc_arg_;

    // Stores the logical number of channels for this NCHWc output. The NCHWc
    // NodeArg is zero padded to the NCHWc block size. If the output needs to
    // be reordered back to a standard tensor format, this channel count is
    // used to generate the expected number of channels.
    const int64_t channels_;

    // Stores the proto shape for the NCHWc output.
    NchwcArgument::Shape shape_;

    // Stores the original number of uses for the original NodeArg. Edges are
    // removed from the graph as nodes are converted to NCHWc form.
    const size_t starting_original_uses_;

    // Stores the remaining number of uses for the original NodeArg. The count
    // is decremented as uses are converted to NCHWc format. Nodes are inserted
    // to reorder the output if this count is non-zero.
    size_t remaining_original_uses_;

    NchwcArgument(Node& output_node, NodeArg* output_nchwc_arg, size_t original_uses, size_t channels, const NchwcArgument::Shape& shape)
        : output_node_(output_node),
          nchwc_arg_(output_nchwc_arg),
          remaining_original_uses_(original_uses),
          starting_original_uses_(original_uses),
          channels_(channels),
          shape_(shape) {
    }
  };

  const ONNX_NAMESPACE::AttributeProto* GetAttribute(const Node& node, const char* attribute_name);
  const ONNX_NAMESPACE::AttributeProto* GetIntsAttribute(const Node& node, const char* attribute_name, int expected_size);
  size_t RemoveOutputEdges(Node& node);
  void CreateNchwcArgument(Node& node, Node& nchwc_node, int64_t channels, const NchwcArgument::Shape& shape);
  void FuseNchwcArgument(Node& node, const NchwcArgument& nchwc_arg);
  void InsertReorderInput(Node& node);

  void ConvPoolShapeInference(const Node& node,
                              const NchwcArgument::Shape& input_shape,
                              NchwcArgument::Shape& output_shape,
                              const ONNX_NAMESPACE::TensorProto* filter_shape);

  void TransformConv(Node& node);
  void TransformPool(Node& node);
  void TransformAdd(Node& node);
  void TransformConcat(Node& node);
  void TransformActivation(Node& node);
  void TransformElementwise(Node& node);

  Graph& graph_;

  // Stores a queue of nodes to be removed after walking through the graph.
  std::deque<NodeIndex> removed_nodes_;

  // Stores a mapping from the original NodeArg outputs to the NCHWc variants
  // created inside this graph transform.
  std::unordered_map<NodeArg*, std::unique_ptr<NchwcArgument>> nchwc_args_;

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
    // Bias the edge count to handle the case of a node that produces a graph
    // output.
    output_edges_count = 1;
  }
  return output_edges_count;
}

void NchwcTransformerImpl::CreateNchwcArgument(Node& node,
                                               Node& nchwc_node,
                                               int64_t channels,
                                               const NchwcArgument::Shape& shape) {
  size_t original_uses = RemoveOutputEdges(node);

  // Create a new NodeArg to track the output from the NCHWc node.
  auto& output_defs = nchwc_node.MutableOutputDefs();
  auto* output_original_arg = output_defs[0];
  std::string output_reorder_def_name = graph_.GenerateNodeArgName("reorder");
  auto* output_nchwc_arg = &graph_.GetOrCreateNodeArg(output_reorder_def_name, nullptr);
  nchwc_args_[output_original_arg] =
      std::make_unique<NchwcArgument>(nchwc_node, output_nchwc_arg, original_uses, channels, shape);
  output_defs[0] = output_nchwc_arg;
}

void NchwcTransformerImpl::FuseNchwcArgument(Node& node, const NchwcArgument& nchwc_arg) {
  size_t original_uses = RemoveOutputEdges(node);

  // Associate the existing NCHWc NodeArg with the output from this node.
  auto* output_original_arg = node.MutableOutputDefs()[0];
  auto& nchwc_node = nchwc_arg.output_node_;
  auto* output_nchwc_arg = nchwc_node.MutableOutputDefs()[0];
  nchwc_args_[output_original_arg] =
      std::make_unique<NchwcArgument>(nchwc_node, output_nchwc_arg, original_uses, nchwc_arg.channels_, nchwc_arg.shape_);
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
                                              kMSNchwcDomain);
    reorder_input_node.SetExecutionProviderType(node.GetExecutionProviderType());
    input_defs[0] = input_nchwc_arg;
  } else {
    input_defs[0] = it->second;
  }
}

void NchwcTransformerImpl::ConvPoolShapeInference(const Node& node,
                                                  const NchwcArgument::Shape& input_shape,
                                                  NchwcArgument::Shape& output_shape,
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
  if (group_attr != nullptr && group_attr->has_i()) {
    group_count = group_attr->i();
  } else {
    group_count = 1;
  }

  const size_t nchwc_block_size = MlasNchwcGetBlockSize();

  const int64_t nchwc_output_channels = (output_channels + nchwc_block_size - 1) & ~(nchwc_block_size - 1);

  bool do_reorder_input = true;
  bool reorder_filter_OIHWBo = false;

  if (group_count > 1) {
    if ((output_channels % nchwc_block_size) != 0) {
      return;
    }
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

  auto conv_W = std::make_unique<Initializer>(conv_W_tensor_proto);

  std::vector<float> reordered_filter;
  reordered_filter.resize(conv_W->size() / output_channels * nchwc_output_channels);

  // Reorder the weights tensor statically.
  if (reorder_filter_OIHWBo) {
    MlasReorderFilterOIHWBo(conv_W->dims().data(), conv_W->data<float>(), reordered_filter.data());
  } else {
    MlasReorderFilterOIHWBiBo(conv_W->dims().data(), conv_W->data<float>(), reordered_filter.data());
  }

  ONNX_NAMESPACE::TensorProto new_conv_W_tensor_proto(*conv_W_tensor_proto);

  if (output_channels != nchwc_output_channels) {
    new_conv_W_tensor_proto.set_dims(0, nchwc_output_channels);
    new_conv_W_tensor_proto.set_name(graph_.GenerateNodeArgName("reorder"));
  }

  new_conv_W_tensor_proto.set_raw_data(reordered_filter.data(), reordered_filter.size() * sizeof(float));

  graph_.RemoveInitializedTensor(input_defs[1]->Name());
  graph_.AddInitializedTensor(new_conv_W_tensor_proto);

  // Create the replacement node.
  std::string nchwc_node_name = graph_.GenerateNodeName("Nchwc");
  Node& nchwc_node = graph_.AddNode(output_defs[0]->Name() + "_nchwc",
                                    "Conv",
                                    nchwc_node_name,
                                    input_defs,
                                    output_defs,
                                    &node.GetAttributes(),
                                    kMSNchwcDomain);
  nchwc_node.SetExecutionProviderType(node.GetExecutionProviderType());

  if (output_channels != nchwc_output_channels) {
    nchwc_node.MutableInputDefs()[1] = &graph_.GetOrCreateNodeArg(new_conv_W_tensor_proto.name(), nullptr);
  }

  NchwcArgument::Shape output_shape;
  std::fill_n(output_shape.dims_, kNchwcDims, output_defs[0]);

  if (do_reorder_input) {
    auto nchwc_input = nchwc_args_.find(input_defs[0]);
    if (nchwc_input == nchwc_args_.end()) {
      InsertReorderInput(nchwc_node);
    } else {
      ConvPoolShapeInference(node, nchwc_input->second->shape_, output_shape, &new_conv_W_tensor_proto);
      nchwc_node.MutableInputDefs()[0] = nchwc_input->second->nchwc_arg_;
      nchwc_input->second->remaining_original_uses_--;
    }
  }

  CreateNchwcArgument(node, nchwc_node, output_channels, output_shape);
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
  if ((input_shape == nullptr) || (input_shape->dim_size() != 4)) {
    return;
  }
  auto& channels_dim = input_shape->dim(1);
  if (!channels_dim.has_dim_value()) {
    return;
  }
  const int64_t channels = channels_dim.dim_value();
  if ((channels % nchwc_block_size) != 0) {
    return;
  }

  // Create the replacement node.
  std::string nchwc_node_name = graph_.GenerateNodeName("Nchwc");
  Node& nchwc_node = graph_.AddNode(output_defs[0]->Name() + "_nchwc",
                                    node.OpType(),
                                    nchwc_node_name,
                                    input_defs,
                                    output_defs,
                                    &node.GetAttributes(),
                                    kMSNchwcDomain);
  nchwc_node.SetExecutionProviderType(node.GetExecutionProviderType());

  NchwcArgument::Shape output_shape;
  std::fill_n(output_shape.dims_, kNchwcDims, output_defs[0]);

  auto nchwc_input = nchwc_args_.find(input_defs[0]);
  if (nchwc_input == nchwc_args_.end()) {
    InsertReorderInput(nchwc_node);
  } else {
    ConvPoolShapeInference(node, nchwc_input->second->shape_, output_shape, nullptr);
    nchwc_node.MutableInputDefs()[0] = nchwc_input->second->nchwc_arg_;
    nchwc_input->second->remaining_original_uses_--;
  }

  CreateNchwcArgument(node, nchwc_node, channels, output_shape);
  removed_nodes_.push_front(node.Index());
}

// The existing Add/Sum operator implementations can be used with tensors
// in NCHWc format if the tensor shapes are exactly the same (elementwise
// add).
void NchwcTransformerImpl::TransformAdd(Node& node) {
  auto& input_defs = node.MutableInputDefs();

  // Verify that all of the inputs to this operator are from NCHWc outputs.
  std::vector<NchwcArgument*> nchwc_inputs;
  size_t input_defs_count = input_defs.size();
  nchwc_inputs.reserve(input_defs_count);
  for (size_t i = 0; i < input_defs_count; i++) {
    auto it = nchwc_args_.find(input_defs[i]);
    if (it == nchwc_args_.end()) {
      return;
    }
    nchwc_inputs.push_back(it->second.get());
  }

  // Test if all of the NCHWc inputs have a compatible shape.
  auto* nchwc_input_0 = nchwc_inputs[0];
  auto* nchwc_input_0_shape = input_defs[0]->Shape();
  for (size_t n = 1; n < input_defs_count; n++) {
    auto* nchwc_input_n = nchwc_inputs[n];
    for (int i = 0; i < kNchwcDims; i++) {
      // Test if this dimension is derived from the same NodeArg.
      if (nchwc_input_0->shape_.dims_[i] != nchwc_input_n->shape_.dims_[i]) {
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

  // Update the node to directly use the NCHWc inputs directly and decrement
  // the original use counts of the NCHWc inputs.
  for (size_t n = 0; n < input_defs_count; n++) {
    input_defs[n] = nchwc_inputs[n]->nchwc_arg_;
    nchwc_inputs[n]->remaining_original_uses_--;
  }

  // If one of the inputs to the Add/Sum node is a NCHWc convolution, then
  // attempt to fuse the addition into the convolution itself.
  if (input_defs_count == 2) {
    for (size_t n = 0; n < 2; n++) {
      auto* nchwc_input_n = nchwc_inputs[n];
      auto& nchwc_node = nchwc_input_n->output_node_;
      auto& nchwc_input_defs = nchwc_node.MutableInputDefs();
      auto& nchwc_input_args_count = nchwc_node.MutableInputArgsCount();
      // Check if this is a single use NCHWc convolution that hasn't already
      // been fused with another Add/Sum node. The Add/Sum can also only be
      // fused if the convolution isn't itself fused with an activation.
      if ((nchwc_node.OpType() == "Conv") && (nchwc_node.Domain() == kMSNchwcDomain) &&
          (nchwc_input_defs.size() < 4) && (nchwc_input_args_count.size() < 4) &&
          (nchwc_input_n->starting_original_uses_ == 1) &&
          (GetAttribute(nchwc_node, "activation") == nullptr)) {
        // Feed the output of the other NCHWc node into the selected convolution
        // node.
        nchwc_input_defs.resize(4);
        nchwc_input_defs[3] = nchwc_inputs[n ^ 1]->output_node_.MutableOutputDefs()[0];
        nchwc_input_args_count.resize(4);
        nchwc_input_args_count[3] = 1;

        FuseNchwcArgument(node, *nchwc_input_n);
        removed_nodes_.push_front(node.Index());
        return;
      }
    }
  }

  CreateNchwcArgument(node, node, nchwc_input_0->channels_, nchwc_input_0->shape_);
}

void NchwcTransformerImpl::TransformConcat(Node& node) {
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  // Verify that this is a concatenation along the channel axis.
  auto* axis_attr = GetAttribute(node, "axis");
  if (axis_attr == nullptr || !axis_attr->has_i() || axis_attr->i() != 1) {
    return;
  }

  const size_t nchwc_block_size = MlasNchwcGetBlockSize();

  // Verify that all of the inputs to this operator are from NCHWc outputs.
  std::vector<NchwcArgument*> nchwc_inputs;
  size_t input_defs_count = input_defs.size();
  nchwc_inputs.reserve(input_defs_count);
  int64_t total_channels = 0;
  for (size_t i = 0; i < input_defs_count; i++) {
    auto it = nchwc_args_.find(input_defs[i]);
    if (it == nchwc_args_.end()) {
      return;
    }
    // Verify that the logical number of channels is block aligned.
    int64_t input_channels = it->second->channels_;
    if ((input_channels % nchwc_block_size) != 0) {
      return;
    }
    total_channels += input_channels;
    nchwc_inputs.push_back(it->second.get());
  }

  // Update the node to directly use the NCHWc inputs directly and decrement
  // the original use counts of the NCHWc inputs.
  for (size_t n = 0; n < input_defs_count; n++) {
    input_defs[n] = nchwc_inputs[n]->nchwc_arg_;
    nchwc_inputs[n]->remaining_original_uses_--;
  }

  // Copy the shape from any of the NCHWc inputs, but use the current node for
  // the channel dimension.
  NchwcArgument::Shape output_shape = nchwc_inputs[0]->shape_;
  output_shape.dims_[1] = output_defs[0];

  CreateNchwcArgument(node, node, total_channels, output_shape);
}

void NchwcTransformerImpl::TransformActivation(Node& node) {
  auto& input_defs = node.MutableInputDefs();

  auto it = nchwc_args_.find(input_defs[0]);
  if (it != nchwc_args_.end()) {
    auto& nchwc_input = it->second;
    input_defs[0] = nchwc_input->nchwc_arg_;
    nchwc_input->remaining_original_uses_--;

    // Check if this is a single use NCHWc convolution that hasn't already
    // been fused with another activation.
    auto& nchwc_node = nchwc_input->output_node_;
    if ((nchwc_node.OpType() == "Conv") && (nchwc_node.Domain() == kMSNchwcDomain) &&
        (nchwc_input->starting_original_uses_ == 1) &&
        (GetAttribute(nchwc_node, "activation") == nullptr)) {
      nchwc_node.AddAttribute("activation", node.OpType());
      FuseNchwcArgument(node, *nchwc_input);
      removed_nodes_.push_front(node.Index());
    } else {
      CreateNchwcArgument(node, node, nchwc_input->channels_, nchwc_input->shape_);
    }
  }
}

void NchwcTransformerImpl::TransformElementwise(Node& node) {
  auto& input_defs = node.MutableInputDefs();

  auto it = nchwc_args_.find(input_defs[0]);
  if (it != nchwc_args_.end()) {
    auto& nchwc_input = it->second;
    input_defs[0] = nchwc_input->nchwc_arg_;
    nchwc_input->remaining_original_uses_--;
    CreateNchwcArgument(node, node, nchwc_input->channels_, nchwc_input->shape_);
  }
}

void NchwcTransformerImpl::Transform(Node& node) {
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Conv", {1}) ||
      graph_utils::IsSupportedOptypeVersionAndDomain(node, "FusedConv", {1}, kMSDomain)) {
    TransformConv(node);
  } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", {1, 8, 10}) ||
             graph_utils::IsSupportedOptypeVersionAndDomain(node, "AveragePool", {1, 7, 10})) {
    TransformPool(node);
  } else if (node.GetInputEdgesCount() == 0 && node.InputDefs().size() != 0) {
    // The following transforms only run when the input edge count has already
    // been decremented to zero by earlier transforms. This is a quick check
    // that all inputs are NCHWc candidates. Also, these transforms do not need
    // to remove any input edges themselves.
    if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Add", {6}) ||
        graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sum", {8})) {
      TransformAdd(node);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Concat", {4})) {
      TransformConcat(node);
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
  for (auto& nchwc_output : nchwc_args_) {
    if (nchwc_output.second->remaining_original_uses_ > 0) {
      auto* output_original_arg = nchwc_output.first;
      auto* output_nchwc_arg = nchwc_output.second->nchwc_arg_;
      Node& reorder_output_node = graph_.AddNode(graph_.GenerateNodeName("ReorderOutput"),
                                                 "ReorderOutput",
                                                 "ReorderOutput",
                                                 {output_nchwc_arg},
                                                 {output_original_arg},
                                                 nullptr,
                                                 kMSNchwcDomain);
      reorder_output_node.AddAttribute("channels", nchwc_output.second->channels_);
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
