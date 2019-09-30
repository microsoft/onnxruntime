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

  static constexpr int kNchwcBatchChannelDims = 2;
  static constexpr int kNchwcSpatialDims = 2;
  static constexpr int kNchwcDims = kNchwcBatchChannelDims + kNchwcSpatialDims;

 private:
  // Associate the following state with each created NCHWc output keyed off the
  // original NodeArg.
  struct NchwcArgument {
    // Symbolic shape information for this NCHWc output. Each dimension stores
    // the original NodeArg* that sourced the value. Spatial dimensions also
    // track the number of times the original value has been shifted down due
    // to a stride count of 2.
    //
    // For example, the first Conv node that takes NCHW input will create a
    // NchwcArgument with the shape referencing itself. Other NCHWc nodes that
    // use this first Conv node then do a limited shape inference. The shape
    // inference carries forward any of the first Conv node's dimensions that
    // are unchanged or resets to the NodeArg* of the updated output node.
    //
    // The benefit of doing this is for models where the model inputs are not
    // fixed. For example, The YoloV3 model has the image height and width as
    // parameters. The model has branches that are candidates for Conv/Add
    // fusion that can be detected using this additional shape hint.
    struct Shape {
      const NodeArg* dims_[kNchwcDims];
      size_t shifts_[kNchwcSpatialDims];

      Shape(const NodeArg* initial_dim) {
        std::fill_n(dims_, kNchwcDims, initial_dim);
        std::fill_n(shifts_, kNchwcSpatialDims, 0);
      }

      bool IsDimEqual(const Shape& other, int dim) const {
        bool is_dim_equal = false;
        // Test if this dimension is derived from the same NodeArg.
        if (dims_[dim] == other.dims_[dim]) {
          if (dim >= kNchwcBatchChannelDims) {
            // Test if the NodeArg has been shifted down the same number of
            // times due to striding.
            int spatial_dim = dim - kNchwcBatchChannelDims;
            if (shifts_[spatial_dim] == other.shifts_[spatial_dim]) {
              is_dim_equal = true;
            }
          } else {
            is_dim_equal = true;
          }
        }
        return is_dim_equal;
      }
    };

    // Stores the node that generated the NCHWc output.
    Node& output_node_;

    // Stores the NodeArg that represents the NCHWc output.
    NodeArg* nchwc_arg_;

    // Stores the original number of uses for the original NodeArg. Edges are
    // removed from the graph as nodes are converted to NCHWc form.
    const size_t starting_original_uses_;

    // Stores the remaining number of uses for the original NodeArg. The count
    // is decremented as uses are converted to NCHWc format. Nodes are inserted
    // to reorder the output if this count is non-zero.
    size_t remaining_original_uses_;

    // Stores the logical number of channels for this NCHWc output. The NCHWc
    // NodeArg is zero padded to the NCHWc block size. If the output needs to
    // be reordered back to a standard tensor format, this channel count is
    // used to generate the expected number of channels.
    const int64_t channels_;

    // Stores the proto shape for the NCHWc output.
    NchwcArgument::Shape shape_;

    NchwcArgument(Node& output_node, NodeArg* output_nchwc_arg, size_t original_uses, size_t channels, const NchwcArgument::Shape& shape)
        : output_node_(output_node),
          nchwc_arg_(output_nchwc_arg),
          starting_original_uses_(original_uses),
          remaining_original_uses_(original_uses),
          channels_(channels),
          shape_(shape) {
    }
  };

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

  Graph& graph_;

  // Stores a queue of nodes to be removed after walking through the graph.
  std::deque<NodeIndex> removed_nodes_;

  // Stores a mapping from the original NodeArg outputs to the NCHWc variants
  // created inside this graph transform.
  std::unordered_map<NodeArg*, std::unique_ptr<NchwcArgument>> nchwc_args_;

  // Stores a mapping of NodeArg inputs that have already been reordered, so
  // multiple nodes can share the NCHWc input.
  std::unordered_map<NodeArg*, NodeArg*> reorder_inputs_;

  // Stores a mapping of NodeArg filters that have already been reordered, so
  // multiple nodes can share the NCHWc filter.
  std::unordered_map<NodeArg*, NodeArg*> filters_OIHWBo_;
  std::unordered_map<NodeArg*, NodeArg*> filters_OIHWBiBo_;

  // Stores a mapping of NodeArg biases that have already been aligned to the
  // NCHWc block size, so multiple nodes can share the NCHWc biases.
  std::unordered_map<NodeArg*, NodeArg*> aligned_biases_;
};

size_t NchwcTransformerImpl::RemoveOutputEdges(Node& node) {
  size_t output_edges_count = node.GetOutputEdgesCount();
  if (output_edges_count > 0) {
    graph_utils::RemoveNodeOutputEdges(graph_, node);
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
  // Skip the leading batch and channel counts.
  const int kernel_size = kNchwcSpatialDims;

  // Maintain the batch count dimension from the NCHWc input.
  output_shape.dims_[0] = input_shape.dims_[0];

  const ONNX_NAMESPACE::AttributeProto* pads_attr = graph_utils::GetNodeAttribute(node, "pads");
  const ONNX_NAMESPACE::AttributeProto* strides_attr = graph_utils::GetNodeAttribute(node, "strides");
  const ONNX_NAMESPACE::AttributeProto* dilations_attr = graph_utils::GetNodeAttribute(node, "dilations");

  if ((pads_attr != nullptr && pads_attr->ints_size() != kernel_size * 2) ||
      (strides_attr != nullptr && strides_attr->ints_size() != kernel_size) ||
      (dilations_attr != nullptr && dilations_attr->ints_size() != kernel_size)) {
    return;
  }

  // Require the kernel_shape attribute for pooling operators. Convolution
  // uses the weight tensor shape to derive the kernel shape.
  const ONNX_NAMESPACE::AttributeProto* kernel_shape_attr = nullptr;
  if (filter_shape == nullptr) {
    kernel_shape_attr = graph_utils::GetNodeAttribute(node, "kernel_shape");
    if (kernel_shape_attr == nullptr || kernel_shape_attr->ints_size() != kernel_size) {
      return;
    }
  }

  auto* auto_pad_attr = graph_utils::GetNodeAttribute(node, "auto_pad");
  bool auto_pad_same_shape = false;
  if (auto_pad_attr != nullptr && utils::HasString(*auto_pad_attr)) {
    auto& auto_pad = auto_pad_attr->s();
    if (auto_pad != "NOTSET") {
      if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        auto_pad_same_shape = true;
      } else if (auto_pad != "VALID") {
        return;
      }
      pads_attr = nullptr;
    }
  }

  for (int i = 0; i < kernel_size; i++) {
    if (dilations_attr != nullptr && dilations_attr->ints(i) != 1) {
      continue;
    }

    int64_t stride = 1;
    if (strides_attr != nullptr) {
      stride = strides_attr->ints(i);
      if (stride != 1 && stride != 2) {
        continue;
      }
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

    // Maintain the spatial dimension from the NCHWc input if the implicit or
    // explicit padding results in the same symbolic dimension before applying
    // the stride. When the stride is 2, then the actual output dimensions is
    // half the original value. Track the number of times the symbolic dimension
    // has been halved in the shifts field.
    if (padding + 1 == kernel || auto_pad_same_shape) {
      output_shape.dims_[kNchwcBatchChannelDims + i] = input_shape.dims_[kNchwcBatchChannelDims + i];
      output_shape.shifts_[i] = input_shape.shifts_[i] + static_cast<size_t>(stride) - 1;
    }
  }
}

void NchwcTransformerImpl::TransformConv(Node& node) {
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  // Require that the weights tensor be static.
  const ONNX_NAMESPACE::TensorProto* conv_W_tensor_proto = nullptr;
  if (!graph_utils::NodeArgIsConstant(graph_, *input_defs[1]) ||
      !graph_.GetInitializedTensor(input_defs[1]->Name(), conv_W_tensor_proto) ||
      (conv_W_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) ||
      (conv_W_tensor_proto->dims_size() != 4)) {
    return;
  }

  const int64_t output_channels = conv_W_tensor_proto->dims(0);
  const int64_t input_channels = conv_W_tensor_proto->dims(1);

  int64_t group_count;
  auto* group_attr = graph_utils::GetNodeAttribute(node, "group");
  if (group_attr != nullptr && utils::HasInt(*group_attr)) {
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

  // Also require that the optional bias tensor be static.
  const ONNX_NAMESPACE::TensorProto* conv_B_tensor_proto = nullptr;
  if (input_defs.size() >= 3) {
    if (!graph_utils::NodeArgIsConstant(graph_, *input_defs[2]) ||
        !graph_.GetInitializedTensor(input_defs[2]->Name(), conv_B_tensor_proto) ||
        (conv_B_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) ||
        (conv_B_tensor_proto->dims_size() != 1) ||
        (conv_B_tensor_proto->dims(0) != output_channels)) {
      return;
    }
  }

  // Check if the filter has already been converted to the target format.
  std::unordered_map<NodeArg*, NodeArg*>* filters_map;
  if (reorder_filter_OIHWBo) {
    filters_map = &filters_OIHWBo_;
  } else {
    filters_map = &filters_OIHWBiBo_;
  }

  NodeArg* nchwc_conv_W_arg;
  auto filters_it = filters_map->find(input_defs[1]);
  if (filters_it != filters_map->end()) {
    // Reuse the existing NodeArg.
    nchwc_conv_W_arg = filters_it->second;
  } else {
    auto conv_W = std::make_unique<Initializer>(conv_W_tensor_proto);

    std::vector<float> reordered_filter(conv_W->size() / output_channels * nchwc_output_channels);

    // Reorder the weights tensor statically.
    if (reorder_filter_OIHWBo) {
      MlasReorderFilterOIHWBo(conv_W->dims().data(), conv_W->data<float>(), reordered_filter.data());
    } else {
      MlasReorderFilterOIHWBiBo(conv_W->dims().data(), conv_W->data<float>(), reordered_filter.data());
    }

    ONNX_NAMESPACE::TensorProto nchwc_conv_W_tensor_proto;

    nchwc_conv_W_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    nchwc_conv_W_tensor_proto.set_name(graph_.GenerateNodeArgName("reorder"));
    nchwc_conv_W_tensor_proto.set_raw_data(reordered_filter.data(), reordered_filter.size() * sizeof(float));

    nchwc_conv_W_tensor_proto.add_dims(nchwc_output_channels);
    for (size_t i = 1; i < 4; i++) {
      nchwc_conv_W_tensor_proto.add_dims(conv_W->dims()[i]);
    }

    graph_.AddInitializedTensor(nchwc_conv_W_tensor_proto);

    nchwc_conv_W_arg = &graph_.GetOrCreateNodeArg(nchwc_conv_W_tensor_proto.name(), nullptr);
    filters_map->emplace(input_defs[1], nchwc_conv_W_arg);
  }

  // Align the optional bias tensor up to the number of NCHWc output channels.
  NodeArg* nchwc_conv_B_arg = nullptr;
  if ((conv_B_tensor_proto != nullptr) && (output_channels != nchwc_output_channels)) {
    auto biases_it = aligned_biases_.find(input_defs[2]);
    if (biases_it != aligned_biases_.end()) {
      // Reuse the existing NodeArg.
      nchwc_conv_B_arg = biases_it->second;
    } else {
      auto conv_B = std::make_unique<Initializer>(conv_B_tensor_proto);

      std::vector<float> aligned_bias(nchwc_output_channels);
      std::copy_n(conv_B->data<float>(), output_channels, aligned_bias.data());

      ONNX_NAMESPACE::TensorProto nchwc_conv_B_tensor_proto;

      nchwc_conv_B_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      nchwc_conv_B_tensor_proto.set_name(graph_.GenerateNodeArgName("reorder"));
      nchwc_conv_B_tensor_proto.set_raw_data(aligned_bias.data(), aligned_bias.size() * sizeof(float));

      nchwc_conv_B_tensor_proto.add_dims(nchwc_output_channels);

      graph_.AddInitializedTensor(nchwc_conv_B_tensor_proto);

      nchwc_conv_B_arg = &graph_.GetOrCreateNodeArg(nchwc_conv_B_tensor_proto.name(), nullptr);
      aligned_biases_.emplace(input_defs[2], nchwc_conv_B_arg);
    }
  }

  // Create the replacement node.
  std::string nchwc_node_name = graph_.GenerateNodeName(output_defs[0]->Name() + "_nchwc");
  Node& nchwc_node = graph_.AddNode(nchwc_node_name,
                                    "Conv",
                                    nchwc_node_name,
                                    input_defs,
                                    output_defs,
                                    &node.GetAttributes(),
                                    kMSNchwcDomain);
  nchwc_node.SetExecutionProviderType(node.GetExecutionProviderType());

  nchwc_node.MutableInputDefs()[1] = nchwc_conv_W_arg;

  if (nchwc_conv_B_arg != nullptr) {
    nchwc_node.MutableInputDefs()[2] = nchwc_conv_B_arg;
  }

  NchwcArgument::Shape output_shape(output_defs[0]);

  if (do_reorder_input) {
    auto it = nchwc_args_.find(input_defs[0]);
    if (it == nchwc_args_.end()) {
      InsertReorderInput(nchwc_node);
    } else {
      auto* nchwc_input = it->second.get();
      nchwc_node.MutableInputDefs()[0] = nchwc_input->nchwc_arg_;
      nchwc_input->remaining_original_uses_--;
      ConvPoolShapeInference(node, nchwc_input->shape_, output_shape, conv_W_tensor_proto);
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
  if (!utils::HasDimValue(channels_dim)) {
    return;
  }
  const int64_t channels = channels_dim.dim_value();
  if ((channels % nchwc_block_size) != 0) {
    return;
  }

  // Create the replacement node.
  std::string nchwc_node_name = graph_.GenerateNodeName(output_defs[0]->Name() + "_nchwc");
  Node& nchwc_node = graph_.AddNode(nchwc_node_name,
                                    node.OpType(),
                                    nchwc_node_name,
                                    input_defs,
                                    output_defs,
                                    &node.GetAttributes(),
                                    kMSNchwcDomain);
  nchwc_node.SetExecutionProviderType(node.GetExecutionProviderType());

  NchwcArgument::Shape output_shape(output_defs[0]);

  auto it = nchwc_args_.find(input_defs[0]);
  if (it == nchwc_args_.end()) {
    InsertReorderInput(nchwc_node);
  } else {
    auto* nchwc_input = it->second.get();
    nchwc_node.MutableInputDefs()[0] = nchwc_input->nchwc_arg_;
    nchwc_input->remaining_original_uses_--;
    ConvPoolShapeInference(node, nchwc_input->shape_, output_shape, nullptr);
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
      if (!nchwc_input_0->shape_.IsDimEqual(nchwc_input_n->shape_, i)) {
        // Check if ONNX shape inferencing has computed a precise dimension value.
        auto* nchwc_input_n_shape = input_defs[n]->Shape();
        if ((nchwc_input_0_shape == nullptr) || (nchwc_input_n_shape == nullptr)) {
          return;
        }
        auto& nchwc_input_0_dim = nchwc_input_0_shape->dim(i);
        auto& nchwc_input_n_dim = nchwc_input_n_shape->dim(i);
        if (!utils::HasDimValue(nchwc_input_0_dim) ||
            !utils::HasDimValue(nchwc_input_n_dim) ||
            (nchwc_input_0_dim.dim_value() <= 0) ||
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
          (graph_utils::GetNodeAttribute(nchwc_node, "activation") == nullptr)) {
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
  auto* axis_attr = graph_utils::GetNodeAttribute(node, "axis");
  if (axis_attr == nullptr || !utils::HasInt(*axis_attr) || axis_attr->i() != 1) {
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

// After doing a Conv/Add fusion, there may be an activation node that could now
// be fused into the Conv node as well.
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
        (graph_utils::GetNodeAttribute(nchwc_node, "activation") == nullptr)) {
      nchwc_node.AddAttribute("activation", node.OpType());
      FuseNchwcArgument(node, *nchwc_input);
      removed_nodes_.push_front(node.Index());
    } else {
      CreateNchwcArgument(node, node, nchwc_input->channels_, nchwc_input->shape_);
    }
  }
}

void NchwcTransformerImpl::Transform(Node& node) {
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Conv", {1, 11}) ||
      graph_utils::IsSupportedOptypeVersionAndDomain(node, "FusedConv", {1}, kMSDomain)) {
    TransformConv(node);
  } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", {1, 8, 10, 11}) ||
             graph_utils::IsSupportedOptypeVersionAndDomain(node, "AveragePool", {1, 7, 10, 11}) ||
             graph_utils::IsSupportedOptypeVersionAndDomain(node, "GlobalMaxPool", {1}) ||
             graph_utils::IsSupportedOptypeVersionAndDomain(node, "GlobalAveragePool", {1})) {
    TransformPool(node);
  } else if (node.GetInputEdgesCount() == 0 && node.InputDefs().size() != 0) {
    // The following transforms only run when the input edge count has already
    // been decremented to zero by earlier transforms. This is a hint that the
    // node may already have all inputs converted to NCHWc format and is not
    // needed for correct operation. This avoids doing extra string checks for
    // nodes unrelated to this transformer.
    if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Add", {7}) ||
        graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sum", {6, 8})) {
      TransformAdd(node);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Concat", {4})) {
      TransformConcat(node);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Relu", {6})) {
      TransformActivation(node);
    }
  }

  // The node may not match any of the checks above or may not have been
  // transformed for other reasons such as unsupported attributes or alignment.
  // However, the node may still use an input that has been produced by a NCHWc
  // node. Finalize() walks through the list of NCHWc outputs and inserts the
  // needed reorder operations to ensure that these inputs remain in NCHW
  // format.
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
