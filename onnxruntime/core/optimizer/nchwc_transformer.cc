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
    const NchwcArgument::Shape shape_;

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
  Node& InsertReshape(NodeArg* input_arg, NodeArg* output_arg, int64_t channels, bool split_channels);

  void TransformConv(Node& node);
  void TransformPool(Node& node);
  void TransformBinary(Node& node, bool add_node);
  void TransformConcat(Node& node);
  void TransformActivation(Node& node);
  void TransformBatchNormalization(Node& node);
  void TransformTranspose(Node& node);
  void TransformResize(Node& node);

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

  // Stores a mapping of shape initializers for use by Reshape when splitting
  // or unsplitting the channels dimension of a tensor.
  std::unordered_map<int64_t, NodeArg*> reshape_split_;
  std::unordered_map<int64_t, NodeArg*> reshape_unsplit_;
};

size_t NchwcTransformerImpl::RemoveOutputEdges(Node& node) {
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
      onnxruntime::make_unique<NchwcArgument>(nchwc_node, output_nchwc_arg, original_uses, channels, shape);
  output_defs[0] = output_nchwc_arg;
}

void NchwcTransformerImpl::FuseNchwcArgument(Node& node, const NchwcArgument& nchwc_arg) {
  size_t original_uses = RemoveOutputEdges(node);

  // Associate the existing NCHWc NodeArg with the output from this node.
  auto* output_original_arg = node.MutableOutputDefs()[0];
  auto& nchwc_node = nchwc_arg.output_node_;
  auto* output_nchwc_arg = nchwc_node.MutableOutputDefs()[0];
  nchwc_args_[output_original_arg] =
      onnxruntime::make_unique<NchwcArgument>(nchwc_node, output_nchwc_arg, original_uses, nchwc_arg.channels_, nchwc_arg.shape_);
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
    reorder_input_node.SetExecutionProviderType(kCpuExecutionProvider);
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

  const auto* pads_attr = graph_utils::GetNodeAttribute(node, "pads");
  const auto* strides_attr = graph_utils::GetNodeAttribute(node, "strides");
  const auto* dilations_attr = graph_utils::GetNodeAttribute(node, "dilations");

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

  const auto* auto_pad_attr = graph_utils::GetNodeAttribute(node, "auto_pad");
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
  const auto* group_attr = graph_utils::GetNodeAttribute(node, "group");
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
    Initializer conv_W{*conv_W_tensor_proto, graph_.ModelPath()};

    int64_t reordered_filter_vec_size = conv_W.size() / output_channels * nchwc_output_channels;
    std::vector<float> reordered_filter(gsl::narrow<size_t>(reordered_filter_vec_size));

    // Reorder the weights tensor statically.
    if (reorder_filter_OIHWBo) {
      MlasReorderFilterOIHWBo(conv_W.dims().data(), conv_W.data<float>(), reordered_filter.data());
    } else {
      MlasReorderFilterOIHWBiBo(conv_W.dims().data(), conv_W.data<float>(), reordered_filter.data());
    }

    ONNX_NAMESPACE::TensorProto nchwc_conv_W_tensor_proto;

    nchwc_conv_W_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    nchwc_conv_W_tensor_proto.set_name(graph_.GenerateNodeArgName("reorder"));
    nchwc_conv_W_tensor_proto.set_raw_data(reordered_filter.data(), reordered_filter.size() * sizeof(float));

    nchwc_conv_W_tensor_proto.add_dims(nchwc_output_channels);
    for (size_t i = 1; i < 4; i++) {
      nchwc_conv_W_tensor_proto.add_dims(conv_W.dims()[i]);
    }

    nchwc_conv_W_arg = &graph_utils::AddInitializer(graph_, nchwc_conv_W_tensor_proto);
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
      Initializer conv_B{*conv_B_tensor_proto, graph_.ModelPath()};

      std::vector<float> aligned_bias(gsl::narrow<size_t>(nchwc_output_channels));
      std::copy_n(conv_B.data<float>(), output_channels, aligned_bias.data());

      ONNX_NAMESPACE::TensorProto nchwc_conv_B_tensor_proto;

      nchwc_conv_B_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      nchwc_conv_B_tensor_proto.set_name(graph_.GenerateNodeArgName("reorder"));
      nchwc_conv_B_tensor_proto.set_raw_data(aligned_bias.data(), gsl::narrow<size_t>(nchwc_output_channels) * sizeof(float));

      nchwc_conv_B_tensor_proto.add_dims(nchwc_output_channels);

      nchwc_conv_B_arg = &graph_utils::AddInitializer(graph_, nchwc_conv_B_tensor_proto);
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
  nchwc_node.SetExecutionProviderType(kCpuExecutionProvider);

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

  const auto* input_type = input_defs[0]->TypeAsProto();
  if ((input_type == nullptr) || (input_type->tensor_type().elem_type() != TensorProto_DataType_FLOAT)) {
    return;
  }
  const auto* input_shape = input_defs[0]->Shape();
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
  nchwc_node.SetExecutionProviderType(kCpuExecutionProvider);

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

Node& NchwcTransformerImpl::InsertReshape(NodeArg* input_arg,
                                          NodeArg* output_arg,
                                          int64_t channels,
                                          bool split_channels) {
  const int64_t nchwc_block_size = static_cast<int64_t>(MlasNchwcGetBlockSize());
  const int64_t nchwc_channels = (channels + nchwc_block_size - 1) & ~(nchwc_block_size - 1);

  // Reuse the shape initializer across reshapes for the same channel configuration.
  auto& shape_arg_map = split_channels ? reshape_split_ : reshape_unsplit_;
  NodeArg* shape_arg = shape_arg_map[nchwc_channels];
  if (shape_arg == nullptr) {
    ONNX_NAMESPACE::TensorProto shape_tensor_proto;
    shape_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    shape_tensor_proto.set_name(graph_.GenerateNodeArgName("Reshape"));
    // Passthrough the batch dimension.
    shape_tensor_proto.add_int64_data(0);
    if (split_channels) {
      shape_tensor_proto.add_int64_data(nchwc_channels / nchwc_block_size);
    } else {
      shape_tensor_proto.add_int64_data(nchwc_channels);
    }
    // Passthrough the spatial dimensions.
    for (int i = 0; i < kNchwcSpatialDims; i++) {
      shape_tensor_proto.add_int64_data(0);
    }
    if (split_channels) {
      shape_tensor_proto.add_int64_data(nchwc_block_size);
      shape_tensor_proto.add_dims(kNchwcDims + 1);
    } else {
      shape_tensor_proto.add_dims(kNchwcDims);
    }

    shape_arg = &graph_utils::AddInitializer(graph_, shape_tensor_proto);
    shape_arg_map[nchwc_channels] = shape_arg;
  }

  Node& reshape_node = graph_.AddNode(graph_.GenerateNodeName("Reshape"),
                                      "Reshape",
                                      "Reshape",
                                      {input_arg, shape_arg},
                                      {output_arg});
  reshape_node.SetExecutionProviderType(kCpuExecutionProvider);

  return reshape_node;
}

void NchwcTransformerImpl::TransformBinary(Node& node, bool add_node) {
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

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

  auto* nchwc_input_0 = nchwc_inputs[0];
  const int64_t channels = nchwc_inputs[0]->channels_;

  // Test if all of the NCHWc inputs have an equal shape.
  bool all_shapes_match = true;
  auto* input_0_shape = input_defs[0]->Shape();
  for (size_t n = 1; n < input_defs_count; n++) {
    auto* nchwc_input_n = nchwc_inputs[n];
    // Require that all inputs have the same logical number of channels.
    if (nchwc_input_n->channels_ != channels) {
      return;
    }
    for (int i = 0; i < kNchwcDims; i++) {
      // Test if this dimension is derived from the same NodeArg.
      if (!nchwc_input_0->shape_.IsDimEqual(nchwc_input_n->shape_, i)) {
        // Check if ONNX shape inferencing has computed a precise dimension value.
        auto* input_n_shape = input_defs[n]->Shape();
        if ((input_0_shape == nullptr) || (input_n_shape == nullptr)) {
          all_shapes_match = false;
        } else {
          auto& input_0_dim = input_0_shape->dim(i);
          auto& input_n_dim = input_n_shape->dim(i);
          if (!utils::HasDimValue(input_0_dim) ||
              !utils::HasDimValue(input_n_dim) ||
              (input_0_dim.dim_value() <= 0) ||
              (input_0_dim.dim_value() != input_n_dim.dim_value())) {
            if (!utils::HasDimParam(input_0_dim) ||
                !utils::HasDimParam(input_n_dim) ||
                (input_0_dim.dim_param() != input_n_dim.dim_param())) {
              all_shapes_match = false;
            }
          }
        }
      }
    }
  }

  if (all_shapes_match) {
    // Update the node to directly use the NCHWc inputs directly and decrement
    // the original use counts of the NCHWc inputs.
    for (size_t n = 0; n < input_defs_count; n++) {
      input_defs[n] = nchwc_inputs[n]->nchwc_arg_;
      nchwc_inputs[n]->remaining_original_uses_--;
    }

    // If one of the inputs to the Add/Sum node is a NCHWc convolution, then
    // attempt to fuse the addition into the convolution itself.
    if (add_node && input_defs_count == 2) {
      for (size_t n = 0; n < 2; n++) {
        auto* nchwc_input_n = nchwc_inputs[n];
        auto& nchwc_node = nchwc_input_n->output_node_;
        auto& nchwc_input_defs = nchwc_node.MutableInputDefs();
        auto& nchwc_input_args_count = nchwc_node.MutableInputArgsCount();
        size_t nchwc_input_defs_count = nchwc_input_defs.size();
        // Check if this is a single use NCHWc convolution that hasn't already
        // been fused with another Add/Sum node. The Add/Sum can also only be
        // fused if the convolution isn't itself fused with an activation.
        if ((nchwc_node.OpType() == "Conv") && (nchwc_node.Domain() == kMSNchwcDomain) &&
            (nchwc_input_defs_count < 4) && (nchwc_input_args_count.size() < 4) &&
            (nchwc_input_n->starting_original_uses_ == 1) &&
            (graph_utils::GetNodeAttribute(nchwc_node, "activation") == nullptr)) {
          // Feed the output of the other NCHWc node into the selected convolution
          // node.
          nchwc_input_defs.resize(4);
          nchwc_input_args_count.resize(4);
          if (nchwc_input_defs_count < 3) {
            // The optional bias parameter is empty so set to an empty string.
            nchwc_input_defs[2] = &graph_.GetOrCreateNodeArg("", nullptr);
            nchwc_input_args_count[2] = 1;
          }
          nchwc_input_defs[3] = nchwc_inputs[n ^ 1]->output_node_.MutableOutputDefs()[0];
          nchwc_input_args_count[3] = 1;

          FuseNchwcArgument(node, *nchwc_input_n);
          removed_nodes_.push_front(node.Index());
          return;
        }
      }
    }

    CreateNchwcArgument(node, node, nchwc_input_0->channels_, nchwc_input_0->shape_);
    return;
  }

  if (add_node) {
    // The input shapes cannot be shown to be identical, but the channel dimension
    // is the same. Reshape the tensors to explicitly use the true NCHWc shape in
    // order to perform the binary operation.
    //
    // Typically, both tensors are of the same shape at inferencing time, but this
    // could not be proven using symbolic dimensions. This reshaping avoids the
    // alternative of reordering the tensors back to NCHW.
    //
    // This optimization is restricted to Add/Sum nodes. Mul nodes would also work
    // using this code, however the common case here is multiplying a NxCxHxW
    // matrix by a NxCx1x1 vector. The implementation of Mul does not currently
    // vectorize well for the case of broadcasting a NCHWc sized channel block.
    // This case is better served by a
    for (size_t n = 0; n < input_defs_count; n++) {
      std::string reshape_input_def_name = graph_.GenerateNodeArgName("reshape");
      auto* reshape_input_arg = &graph_.GetOrCreateNodeArg(reshape_input_def_name, nullptr);
      InsertReshape(nchwc_inputs[n]->nchwc_arg_, reshape_input_arg, channels, true);

      input_defs[n] = reshape_input_arg;
      nchwc_inputs[n]->remaining_original_uses_--;
    }

    std::string output_reshaped_def_name = graph_.GenerateNodeArgName("reshape");
    auto* output_reshaped_arg = &graph_.GetOrCreateNodeArg(output_reshaped_def_name, nullptr);
    Node& nchwc_node = InsertReshape(output_reshaped_arg, output_defs[0], channels, false);

    NchwcArgument::Shape output_shape(output_defs[0]);

    CreateNchwcArgument(node, nchwc_node, channels, output_shape);
    output_defs[0] = output_reshaped_arg;
    return;
  }
}

void NchwcTransformerImpl::TransformConcat(Node& node) {
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  // Verify that this is a concatenation along the channel axis.
  const auto* axis_attr = graph_utils::GetNodeAttribute(node, "axis");
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
// be fused into the Conv node as well. Otherwise, this is an elementwise
// operation that can directly use the NCHWc input.
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

// Transform BatchNormalization to a depthwise separable 1x1 convolution. This
// enables reuse of the existing NCHWc convolution operator and other fusions
// such as BatchNormalization+Relu using Conv+Relu.
void NchwcTransformerImpl::TransformBatchNormalization(Node& node) {
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  // Bail out if the node has the optional training outputs specified.
  if (output_defs.size() > 1) {
    return;
  }

  // Don't transform the node if the input is not already in NCHWc format.
  auto it = nchwc_args_.find(input_defs[0]);
  if (it == nchwc_args_.end()) {
    return;
  }
  auto* nchwc_input = it->second.get();

  // Require that BatchNormalization-7 uses spatial normalization.
  const auto* spatial_attr = graph_utils::GetNodeAttribute(node, "spatial");
  if (spatial_attr != nullptr && utils::HasInt(*spatial_attr) && spatial_attr->i() != 1) {
    return;
  }

  const auto* epsilon_attr = graph_utils::GetNodeAttribute(node, "epsilon");
  if (epsilon_attr == nullptr || !utils::HasFloat(*epsilon_attr)) {
    return;
  }
  float epsilon = static_cast<float>(epsilon_attr->f());

  const int64_t channels = nchwc_input->channels_;

  auto get_bn_tensor_proto = [this, channels](const std::string& input_name) {
    const auto* tensor_proto = graph_utils::GetConstantInitializer(graph_, input_name);
    if (tensor_proto != nullptr) {
      if ((tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) ||
          (tensor_proto->dims_size() != 1) ||
          (tensor_proto->dims(0) != channels)) {
        tensor_proto = nullptr;
      }
    }
    return tensor_proto;
  };

  const auto* bn_scale_tensor_proto = get_bn_tensor_proto(input_defs[1]->Name());
  if (bn_scale_tensor_proto == nullptr) {
    return;
  }
  const auto* bn_B_tensor_proto = get_bn_tensor_proto(input_defs[2]->Name());
  if (bn_B_tensor_proto == nullptr) {
    return;
  }
  const auto* bn_mean_tensor_proto = get_bn_tensor_proto(input_defs[3]->Name());
  if (bn_mean_tensor_proto == nullptr) {
    return;
  }
  const auto* bn_var_tensor_proto = get_bn_tensor_proto(input_defs[4]->Name());
  if (bn_var_tensor_proto == nullptr) {
    return;
  }

  Initializer bn_scale{*bn_scale_tensor_proto, graph_.ModelPath()};
  Initializer bn_B{*bn_B_tensor_proto, graph_.ModelPath()};
  Initializer bn_mean{*bn_mean_tensor_proto, graph_.ModelPath()};
  Initializer bn_var{*bn_var_tensor_proto, graph_.ModelPath()};

  // Calculate the scale and bias for the replacement convolution.
  bn_var.add(epsilon);
  bn_var.sqrt();
  bn_scale.div(bn_var);
  bn_mean.mul(bn_scale);
  bn_B.sub(bn_mean);

  const size_t nchwc_block_size = MlasNchwcGetBlockSize();
  const int64_t nchwc_channels = (channels + nchwc_block_size - 1) & ~(nchwc_block_size - 1);

  std::vector<float> padded_buffer(gsl::narrow<size_t>(nchwc_channels));

  std::copy_n(bn_scale.data<float>(), channels, padded_buffer.data());

  ONNX_NAMESPACE::TensorProto nchwc_conv_W_tensor_proto;
  nchwc_conv_W_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  nchwc_conv_W_tensor_proto.set_name(graph_.GenerateNodeArgName("bn_scale"));
  nchwc_conv_W_tensor_proto.set_raw_data(padded_buffer.data(), gsl::narrow<size_t>(nchwc_channels) * sizeof(float));
  nchwc_conv_W_tensor_proto.add_dims(nchwc_channels);
  nchwc_conv_W_tensor_proto.add_dims(1);
  nchwc_conv_W_tensor_proto.add_dims(1);
  nchwc_conv_W_tensor_proto.add_dims(1);

  auto* nchwc_conv_W_arg = &graph_utils::AddInitializer(graph_, nchwc_conv_W_tensor_proto);

  std::copy_n(bn_B.data<float>(), channels, padded_buffer.data());

  ONNX_NAMESPACE::TensorProto nchwc_conv_B_tensor_proto;
  nchwc_conv_B_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  nchwc_conv_B_tensor_proto.set_name(graph_.GenerateNodeArgName("bn_B"));
  nchwc_conv_B_tensor_proto.set_raw_data(padded_buffer.data(), gsl::narrow<size_t>(nchwc_channels) * sizeof(float));
  nchwc_conv_B_tensor_proto.add_dims(nchwc_channels);

  auto* nchwc_conv_B_arg = &graph_utils::AddInitializer(graph_, nchwc_conv_B_tensor_proto);

  // Create the replacement node.
  std::string nchwc_node_name = graph_.GenerateNodeName(output_defs[0]->Name() + "_bn_nchwc");
  Node& nchwc_node = graph_.AddNode(nchwc_node_name,
                                    "Conv",
                                    nchwc_node_name,
                                    {nchwc_input->nchwc_arg_, nchwc_conv_W_arg, nchwc_conv_B_arg},
                                    output_defs,
                                    nullptr,
                                    kMSNchwcDomain);
  nchwc_node.SetExecutionProviderType(kCpuExecutionProvider);
  nchwc_node.AddAttribute("group", nchwc_channels);

  nchwc_input->remaining_original_uses_--;

  CreateNchwcArgument(node, nchwc_node, channels, nchwc_input->shape_);
  removed_nodes_.push_front(node.Index());
}

void NchwcTransformerImpl::TransformTranspose(Node& node) {
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  // Don't transform the node if the input is not already in NCHWc format.
  auto it = nchwc_args_.find(input_defs[0]);
  if (it == nchwc_args_.end()) {
    return;
  }
  auto* nchwc_input = it->second.get();

  const auto* perm_attr = graph_utils::GetNodeAttribute(node, "perm");
  if (perm_attr == nullptr || perm_attr->ints_size() != 4) {
    return;
  }

  // Test if this transposes from NCHW to NHWC layout order.
  const int64_t* perm_data = perm_attr->ints().data();
  if (perm_data[0] != 0 || perm_data[1] != 2 || perm_data[2] != 3 || perm_data[3] != 1) {
    return;
  }

  // Create the replacement node.
  Node& reorder_output_node = graph_.AddNode(graph_.GenerateNodeName("ReorderOutput"),
                                             "ReorderOutput",
                                             "ReorderOutput",
                                             {nchwc_input->nchwc_arg_},
                                             output_defs,
                                             nullptr,
                                             kMSNchwcDomain);
  reorder_output_node.SetExecutionProviderType(kCpuExecutionProvider);
  reorder_output_node.AddAttribute("channels", nchwc_input->channels_);
  reorder_output_node.AddAttribute("channels_last", static_cast<int64_t>(1));

  nchwc_input->remaining_original_uses_--;

  graph_utils::RemoveNodeOutputEdges(graph_, node);

  removed_nodes_.push_front(node.Index());
}

void NchwcTransformerImpl::TransformResize(Node& node) {
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  // Don't transform the node if the input is not already in NCHWc format.
  auto it = nchwc_args_.find(input_defs[0]);
  if (it == nchwc_args_.end()) {
    return;
  }
  auto* nchwc_input = it->second.get();

  // Only support the nearest interpolation mode (the default value).
  const auto* mode_attr = graph_utils::GetNodeAttribute(node, "mode");
  if (mode_attr != nullptr && utils::HasString(*mode_attr)) {
    if (mode_attr->s() != "nearest") {
      return;
    }
  }

  NodeArg* scales_arg;
  if (node.SinceVersion() >= 11) {
    // Bail out if Resize has the optional "sizes" tensor.
    if (input_defs.size() == 3) {
      scales_arg = input_defs[2];
    } else {
      return;
    }

    // Only support the asymmetric coordinate transformation mode.
    const auto* transform_mode_attr = graph_utils::GetNodeAttribute(node, "coordinate_transformation_mode");
    if ((transform_mode_attr == nullptr) ||
        !utils::HasString(*transform_mode_attr) ||
        (transform_mode_attr->s() != "asymmetric")) {
      return;
    }

    // Only support the floor rounding mode.
    const auto* nearest_mode_attr = graph_utils::GetNodeAttribute(node, "nearest_mode");
    if ((nearest_mode_attr == nullptr) ||
        !utils::HasString(*nearest_mode_attr) ||
        (nearest_mode_attr->s() != "floor")) {
      return;
    }
  } else {
    scales_arg = input_defs[1];
  }

  // Require that the scales tensor be static.
  const ONNX_NAMESPACE::TensorProto* scales_tensor_proto = nullptr;
  if (!graph_utils::NodeArgIsConstant(graph_, *scales_arg) ||
      !graph_.GetInitializedTensor(scales_arg->Name(), scales_tensor_proto) ||
      (scales_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) ||
      (scales_tensor_proto->dims_size() != 1) ||
      (scales_tensor_proto->dims(0) != 4)) {
    return;
  }

  Initializer scales{*scales_tensor_proto, graph_.ModelPath()};
  auto* scales_data = scales.template data<float>();

  // Cast the scales to integers and verify that the scales are positive and
  // round trip back to floating point.
  std::vector<int64_t> scales_attr(4);
  for (size_t n = 0; n < 4; n++) {
    int64_t scale_value = static_cast<int64_t>(scales_data[n]);
    if (scale_value <= 0 || static_cast<float>(scale_value) != scales_data[n]) {
      return;
    }
    scales_attr[n] = scale_value;
  }

  // Only support spatial scaling at this time (batch and channel are unscaled).
  if (scales_attr[0] != 1 || scales_attr[1] != 1) {
    return;
  }

  std::string nchwc_node_name = graph_.GenerateNodeName(output_defs[0]->Name() + "_nchwc");
  Node& nchwc_node = graph_.AddNode(nchwc_node_name,
                                    "Upsample",
                                    nchwc_node_name,
                                    {nchwc_input->nchwc_arg_},
                                    output_defs,
                                    nullptr,
                                    kMSNchwcDomain);
  nchwc_node.SetExecutionProviderType(kCpuExecutionProvider);
  nchwc_node.AddAttribute("scales", scales_attr);

  nchwc_input->remaining_original_uses_--;

  NchwcArgument::Shape output_shape(output_defs[0]);

  CreateNchwcArgument(node, nchwc_node, nchwc_input->channels_, output_shape);
  removed_nodes_.push_front(node.Index());
}

void NchwcTransformerImpl::Transform(Node& node) {
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Conv", {1, 11}) ||
      graph_utils::IsSupportedOptypeVersionAndDomain(node, "FusedConv", {1}, kMSDomain)) {
    TransformConv(node);
  } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", {1, 8, 10, 11, 12}) ||
             graph_utils::IsSupportedOptypeVersionAndDomain(node, "AveragePool", {1, 7, 10, 11})) {
    TransformPool(node);
  } else if (node.GetInputEdgesCount() == 0 && node.InputDefs().size() != 0) {
    // The following transforms only run when the input edge count has already
    // been decremented to zero by earlier transforms. This is a hint that the
    // node may already have all inputs converted to NCHWc format and is not
    // needed for correct operation. This avoids doing extra string checks for
    // nodes unrelated to this transformer.
    if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Add", {7, 13}) ||
        graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sum", {6, 8, 13})) {
      TransformBinary(node, true);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Mul", {7, 13})) {
      TransformBinary(node, false);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Concat", {4, 11, 13})) {
      TransformConcat(node);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Relu", {6, 13}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sigmoid", {6, 13}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(node, "Tanh", {6, 13})) {
      TransformActivation(node);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "BatchNormalization", {7, 9})) {
      TransformBatchNormalization(node);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Transpose", {1, 13})) {
      TransformTranspose(node);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Upsample", {9, 13}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(node, "Resize", {10, 11, 13})) {
      TransformResize(node);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "GlobalMaxPool", {1}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(node, "GlobalAveragePool", {1})) {
      // Convert these pooling types only if the input is already in NCHWc format.
      TransformPool(node);
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
      reorder_output_node.SetExecutionProviderType(kCpuExecutionProvider);
      reorder_output_node.AddAttribute("channels", nchwc_output.second->channels_);
    }
  }

  for (auto index : removed_nodes_) {
    graph_.RemoveNode(index);
  }

  if (!removed_nodes_.empty()) {
    modified = true;
  }
}

Status NchwcTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  NchwcTransformerImpl impl(graph);
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
