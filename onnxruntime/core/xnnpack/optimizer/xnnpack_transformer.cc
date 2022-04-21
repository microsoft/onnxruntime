// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/xnnpack/optimizer/xnnpack_transformer.h"

#include <deque>

#include "core/framework/op_node_proto_helper.h"
#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/graph/schema_registry.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nhwc_transformer.h"
#include "core/optimizer/selectors_actions/helpers.h"
#include "core/optimizer/transpose_optimizer/optimizer_utils.h"
#include "core/optimizer/utils.h"
#include "core/providers/cpu/nn/pool_attributes.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_layout_transformation;

#define XNNPACK_RETURN_IF_ERROR(expr)                                                                                 \
  do {                                                                                                                \
    auto _status = (expr);                                                                                            \
    if ((!_status.IsOK())) {                                                                                          \
      std::ostringstream oss;                                                                                         \
      oss << __FILE__ << ":" << __LINE__ << " Convert the model to XNNPack graph failed: " << _status.ErrorMessage(); \
      return Status(_status.Category(), _status.Code(), oss.str());                                                   \
    }                                                                                                                 \
  } while (0)

namespace onnxruntime {
static bool IsPaddingTypeSupported(AutoPadType auto_pad) {
  return auto_pad == AutoPadType::NOTSET || auto_pad == AutoPadType::VALID || auto_pad == AutoPadType::SAME_UPPER;
}

bool IsMaxPoolSupportedByXNNPack(const Node& nodeRef, bool input_is_nchw) {
  if (nodeRef.OpType() != "MaxPool") return false;
  auto input_defs = nodeRef.InputDefs();
  if (input_defs.size() != 1) return false;
  if (!input_defs[0]->HasTensorOrScalarShape()) return false;
  auto X_input = input_defs[0]->TypeAsProto();
  assert(X_input != nullptr);
  int32_t etype = X_input->tensor_type().elem_type();
  // Currently we only implemented a f32 kernel
  // if (etype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && etype != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 && etype !=
  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) return false;
  if (etype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
  ProtoHelperNodeContext nc(nodeRef);
  OpNodeProtoHelper info(&nc);
  PoolAttributes pool_attrs(info, "MaxPool", nodeRef.SinceVersion());
  if (!IsPaddingTypeSupported(pool_attrs.auto_pad)) return false;
  if (pool_attrs.kernel_shape.size() != 2) return false;
  if (pool_attrs.kernel_shape[0] <= 0 || pool_attrs.kernel_shape[1] <= 0) return false;
  if (pool_attrs.kernel_shape[0] == 1 && pool_attrs.kernel_shape[1] == 1) {
    // XNNPack doesn't like to support 1x1 maxpool.
    return false;
  }
  auto& input_shape = X_input->tensor_type().shape();
  if (input_shape.dim_size() != 4) return false;
  auto& channel_dim = input_is_nchw ? input_shape.dim(1) : input_shape.dim(3);
  if (!channel_dim.has_dim_value()) return false;
  return true;
}

Status IsConvSupportedByXNNPack(const Node& nodeRef, std::unordered_set<const NodeArg*>& graph_const_values,
                                bool input_is_nchw) {
  if (nodeRef.OpType() != "Conv") return Status(ONNXRUNTIME, FAIL);
  // Conv has either 2 or 3 inputs.
  auto input_defs = nodeRef.InputDefs();
  if (input_defs.size() != 2 && input_defs.size() != 3) return Status(ONNXRUNTIME, FAIL);
  if (graph_const_values.find(input_defs[1]) == graph_const_values.end()) {
    // Weight is not const, we can't run it.
    return Status(ONNXRUNTIME, FAIL);
  }
  // The two or three inputs are: X, W, B
  const NodeArg* weight_node_arg = input_defs[1];
  if (weight_node_arg == nullptr) return Status(ONNXRUNTIME, FAIL);
  // Weight must be a const and all dims are known
  bool is_weight_shape_known = optimizer_utils::IsShapeKnownOnAllDims(*weight_node_arg, 4);
  if (!is_weight_shape_known) return Status(ONNXRUNTIME, FAIL);

  ProtoHelperNodeContext nc(nodeRef);
  OpNodeProtoHelper info(&nc);
  auto X_input = input_defs[0]->TypeAsProto();
  if (X_input == nullptr || !X_input->has_tensor_type() || !X_input->tensor_type().has_shape() ||
      X_input->tensor_type().elem_type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return Status(ONNXRUNTIME, FAIL);
  std::string auto_pad_str;
  XNNPACK_RETURN_IF_ERROR(info.GetAttr<std::string>("auto_pad", &auto_pad_str));
  AutoPadType padding_type = StringToAutoPadType(auto_pad_str);
  if (!IsPaddingTypeSupported(padding_type)) return Status(ONNXRUNTIME, FAIL);
  // The "auto_pad_str" string must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID
  // TF2ONNX converter doesn't use SAME_LOWER.
  // SAME_UPPER maps to TF SAME padding
  if (padding_type == AutoPadType::SAME_UPPER) {
    std::vector<int64_t> dilations;
    Status st1 = info.GetAttrs<int64_t>("dilations", dilations);
    if (dilations.size() != 2) return Status(ONNXRUNTIME, FAIL);
    // Don't know how to handle dilation!=1 cases yet. TF doesn't have it.
    if (dilations[0] != 1 || dilations[1] != 1) return Status(ONNXRUNTIME, FAIL);
  }

  auto& input_shape = X_input->tensor_type().shape();
  if (input_shape.dim_size() != 4) return Status(ONNXRUNTIME, FAIL);
  auto& channel_dim = input_is_nchw ? input_shape.dim(1) : input_shape.dim(3);
  if (!channel_dim.has_dim_value()) return Status(ONNXRUNTIME, FAIL);

  auto weight_input = weight_node_arg->TypeAsProto();
  TensorShape weight_shape = utils::GetTensorShapeFromTensorShapeProto(weight_input->tensor_type().shape());
  int64_t group = 1;
  XNNPACK_RETURN_IF_ERROR(info.GetAttr<int64_t>("group", &group));
  int64_t input_channels = input_is_nchw ? input_shape.dim(1).dim_value() : input_shape.dim(3).dim_value();
  if (group != 1 && group != input_channels) return Status(ONNXRUNTIME, FAIL);

  std::vector<int64_t> pads;
  Status st = info.GetAttrs<int64_t>("pads", pads);
  if (st.IsOK()) {
    if (pads.size() != 4) return Status(ONNXRUNTIME, FAIL);
  }
  return Status::OK();
}

Status CreateTransposeNode(Graph& main_graph, const std::vector<int64_t>& input_perm, bool create_input,
                           Node** new_node, NodeArg** newarg) {
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  if (create_input) {
    std::string output_name = main_graph.GenerateNodeArgName("trans");
    NodeArg& transpose_output = main_graph.GetOrCreateNodeArg(output_name, nullptr);
    *newarg = &transpose_output;
    inputs.push_back(&transpose_output);
  } else {
    std::string output_name = main_graph.GenerateNodeArgName("trans");
    NodeArg& transpose_output = main_graph.GetOrCreateNodeArg(output_name, nullptr);
    *newarg = &transpose_output;
    outputs.push_back(&transpose_output);
  }
  Node& transpose_node = main_graph.AddNode("", "Transpose", "", inputs, outputs, nullptr, kOnnxDomain);
  transpose_node.AddAttribute("perm", input_perm);
  *new_node = &transpose_node;
  return Status::OK();
}

static Status TransposeInput(Graph& main_graph, const std::vector<int64_t>& input_perm, int input_index, Node& node) {
  InOutDefSlot src_slot{ArgType::kInput, input_index};
  Node* transpose_node = nullptr;
  NodeArg* new_arg = nullptr;
  XNNPACK_RETURN_IF_ERROR(CreateTransposeNode(main_graph, input_perm, false, &transpose_node, &new_arg));
  // Append a single slot to transpose_node. As the dest is empty, it will be the first one.
  XNNPACK_RETURN_IF_ERROR(MoveInputOutput(main_graph, node, *transpose_node,
                                          ValueMoveInfo(src_slot, ArgType::kInput, false, false), false));
  XNNPACK_RETURN_IF_ERROR(main_graph.UpdateShapeInference(*transpose_node));
  main_graph.AddEdge(transpose_node->Index(), node.Index(), 0, input_index);
  return Status::OK();
}

static Status TransposeOutput(Graph& main_graph, const std::vector<int64_t>& output_perm, int output_index, Node& node,
                              Node** new_node) {
  InOutDefSlot src_slot{ArgType::kOutput, output_index};
  Node* transpose_node = nullptr;
  NodeArg* transpose_input = nullptr;
  XNNPACK_RETURN_IF_ERROR(CreateTransposeNode(main_graph, output_perm, true, &transpose_node, &transpose_input));
  node.MutableOutputDefs()[output_index]->ClearShape();
  // Append a single slot to dest. As the dest is empty, it will be the first one.
  XNNPACK_RETURN_IF_ERROR(MoveInputOutput(main_graph, node, *transpose_node,
                                          ValueMoveInfo(src_slot, ArgType::kOutput, false, false), false));
  node.MutableOutputDefs()[output_index] = transpose_input;
  main_graph.AddEdge(node.Index(), transpose_node->Index(), output_index, 0);
  XNNPACK_RETURN_IF_ERROR(main_graph.UpdateShapeInference(node));
  XNNPACK_RETURN_IF_ERROR(main_graph.UpdateShapeInference(*transpose_node));

  if (new_node) {
    *new_node = transpose_node;
  }
  return Status::OK();
}

static Status TranposeNCHWToNHWC(Graph& main_graph, int rank, Node& nodeRef, Node** new_node = nullptr) {
  std::vector<int64_t> input_perm = onnx_layout_transformation::ChannelFirstToLastPerm(rank);
  std::vector<int64_t> output_perm = onnx_layout_transformation::ChannelLastToFirstPerm(rank);
  XNNPACK_RETURN_IF_ERROR(TransposeInput(main_graph, input_perm, 0, nodeRef));
  XNNPACK_RETURN_IF_ERROR(main_graph.UpdateShapeInference(nodeRef));
  XNNPACK_RETURN_IF_ERROR(TransposeOutput(main_graph, output_perm, 0, nodeRef, new_node));
  return Status::OK();
}

Status AddBiasInitializer(Graph& main_graph, int64_t bias_size, const std::string& bias_tensor_name, NodeArg** out) {
  if (bias_size < 0 || static_cast<uint64_t>(bias_size) >= std::numeric_limits<size_t>::max()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "size is too large");
  }
  // Create a bias tensor and set all elements to zero
  ::ONNX_NAMESPACE::TensorProto bias_tensor;
  std::vector<float> bias_data(bias_size, 0.0f);
  bias_tensor.mutable_float_data()->Add(bias_data.begin(), bias_data.end());
  bias_tensor.mutable_dims()->Add(bias_size);
  bias_tensor.set_name(bias_tensor_name);
  bias_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  *out = &graph_utils::AddInitializer(main_graph, bias_tensor);
  return Status::OK();
}

// It assumes the new node has the same number of inputs/outputs as the old node. And types of each node arg do not
// change. For each node arg, only the shape may change.
static Status ReplaceNode(Graph& main_graph, Node& old_node, const std::string& op_type, const std::string& description,
                          const NodeAttributes* attributes, const std::string& domain, Node** out) {
  Node& new_node = main_graph.AddNode(old_node.Name(), op_type, description, {}, {}, attributes, domain);

  // Move all the inputs to the new node
  ValueMoveInfo move_info(ArgType::kInput, ArgType::kInput);
  XNNPACK_RETURN_IF_ERROR(MoveInputOutput(main_graph, old_node, new_node, move_info, false));
  // Move all the outputs to the new node
  ValueMoveInfo move_info2(ArgType::kOutput, ArgType::kOutput);
  XNNPACK_RETURN_IF_ERROR(MoveInputOutput(main_graph, old_node, new_node, move_info2, false));
  // Clear output shapes.
  for (NodeArg* p : new_node.MutableOutputDefs()) {
    if (p) p->ClearShape();
  }
  if (!main_graph.RemoveNode(old_node.Index())) {
    return Status(ONNXRUNTIME, FAIL, "remove node failed");
  }
  *out = &new_node;
  return Status::OK();
}

template <typename T>
static void AddAttribute(std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto>& attrs,
                         const std::string& name, const T& value) {
  attrs[name] = utils::MakeAttribute(name, value);
}

static Status ReplaceMaxPool(Graph& main_graph, Node& nodeRef, bool& modified) {
  ProtoHelperNodeContext nc(nodeRef);
  OpNodeProtoHelper info(&nc);
  PoolAttributes pool_attrs(info, "MaxPool", nodeRef.SinceVersion());

  auto inputdefs = nodeRef.InputDefs();
  if (inputdefs.size() != 1) return Status::OK();

  // Skip if unknown rank
  auto shape = inputdefs[0]->Shape();
  if (shape == nullptr || shape->dim_size() != 4) {
    return Status::OK();
  }

  modified = true;

  std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto> attrs;
  AddAttribute(attrs, "input_padding_top", pool_attrs.pads[0]);
  AddAttribute(attrs, "input_padding_right", pool_attrs.pads[3]);
  AddAttribute(attrs, "input_padding_bottom", pool_attrs.pads[2]);
  AddAttribute(attrs, "input_padding_left", pool_attrs.pads[1]);

  AddAttribute(attrs, "pooling_height", pool_attrs.kernel_shape[0]);
  AddAttribute(attrs, "pooling_width", pool_attrs.kernel_shape[1]);

  AddAttribute(attrs, "stride_height", pool_attrs.strides[0]);
  AddAttribute(attrs, "stride_width", pool_attrs.strides[1]);

  AddAttribute(attrs, "dilation_height", pool_attrs.dilations[0]);
  AddAttribute(attrs, "dilation_width", pool_attrs.dilations[1]);
  if (pool_attrs.auto_pad == AutoPadType::SAME_UPPER) {
    AddAttribute(attrs, "padding_mode", static_cast<int64_t>(1));
  } else {
    AddAttribute(attrs, "padding_mode", static_cast<int64_t>(0));
  }

  Node* new_node = nullptr;
  XNNPACK_RETURN_IF_ERROR(
      ReplaceNode(main_graph, nodeRef, "XnnPackMaxPooling2d", "", &attrs, "com.microsoft", &new_node));
  XNNPACK_RETURN_IF_ERROR(TranposeNCHWToNHWC(main_graph, 4, *new_node));
  return Status::OK();
}

static Status ReplaceConv(Graph& main_graph, Node& nodeRef, bool& modified) {
  // The next node after ONNX Conv
  const Node* const next_node_to_fuse =
      optimizer_utils::CheckOutputEdges(main_graph, nodeRef, 1) ? &*nodeRef.OutputNodesBegin() : nullptr;

  ProtoHelperNodeContext nc(nodeRef);
  OpNodeProtoHelper info(&nc);
  int64_t group = 1;
  XNNPACK_RETURN_IF_ERROR(info.GetAttr<int64_t>("group", &group));
  auto X_input = info.GetInputType(0);
  auto weight_input = info.GetInputType(1);
  TensorShape weight_shape = utils::GetTensorShapeFromTensorShapeProto(weight_input->tensor_type().shape());
  TensorShape X_shape = utils::GetTensorShapeFromTensorShapeProto(X_input->tensor_type().shape());
  if (weight_shape.NumDimensions() != 4 || X_shape.NumDimensions() != 4) return Status::OK();

  // Now the input shape is still NCHW
  int64_t input_channels = X_shape[1];

  if (group != 1 && group != input_channels) return Status::OK();
  for (size_t i = 0; i != weight_shape.NumDimensions(); ++i) {
    if (weight_shape[i] <= 0) return Status::OK();
  }
  modified = true;

  // const_cast
  NodeArg* bias_node_arg = nullptr;
  const bool has_bias = nodeRef.InputDefs().size() >= 3;
  if (!has_bias) {
    int64_t bias_size = weight_shape[0];
    std::string bias_tensor_name = main_graph.GenerateNodeArgName(nodeRef.Name() + "_bias");
    XNNPACK_RETURN_IF_ERROR(AddBiasInitializer(main_graph, bias_size, bias_tensor_name, &bias_node_arg));
  }
  std::string auto_pad_str;
  XNNPACK_RETURN_IF_ERROR(info.GetAttr<std::string>("auto_pad", &auto_pad_str));
  // group == 1 || group  == input / output channel count
  // For now we assume input channel count isn't 1, so that group count != input/output channel count
  bool is_depthwise = input_channels != 1 && group == input_channels;

  if (nodeRef.InputDefs().size() < 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Expect at least 2 inputs, got ", nodeRef.InputDefs().size());
  }

  std::vector<int64_t> weight_perm = is_depthwise ? std::vector<int64_t>{1, 2, 3, 0} : std::vector<int64_t>{0, 2, 3, 1};

  std::vector<int64_t> strides, dilations, pads;
  Status st = info.GetAttrs<int64_t>("strides", strides);
  if (!st.IsOK()) {
    // ONNX spec says: "If not present, the stride defaults is 1 along each spatial axis."
    strides.assign(4, 1);
  }
  st = info.GetAttrs<int64_t>("dilations", dilations);
  if (!st.IsOK()) {
    // ONNX spec says: "If not present, the dilation defaults is 1 along each spatial axis."
    dilations.assign(4, 1);
  }
  st = info.GetAttrs<int64_t>("pads", pads);
  if (!st.IsOK()) {
    // ONNX spec says: "If not present, the padding defaults to 0 along start and end of each spatial axis."
    pads.resize(4);
  }

  std::string node_name = nodeRef.Name();
  std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto> attrs;

  AddAttribute(attrs, "input_padding_top", pads[0]);
  AddAttribute(attrs, "input_padding_right", pads[3]);
  AddAttribute(attrs, "input_padding_bottom", pads[2]);
  AddAttribute(attrs, "input_padding_left", pads[1]);

  AddAttribute(attrs, "subsampling_height", strides[0]);
  AddAttribute(attrs, "subsampling_width", strides[1]);

  AddAttribute(attrs, "dilation_height", dilations[0]);
  AddAttribute(attrs, "dilation_width", dilations[1]);

  if (!is_depthwise) AddAttribute(attrs, "groups", group);

  if (auto_pad_str == "NOTSET") {
    AddAttribute(attrs, "padding_mode", static_cast<int64_t>(0));
  } else if (auto_pad_str == "VALID") {
    AddAttribute(attrs, "padding_mode", static_cast<int64_t>(0));
  } else if (auto_pad_str == "SAME_UPPER") {
    AddAttribute(attrs, "padding_mode", static_cast<int64_t>(1));
  } else {
    // This line of code should not be reached because in IsConvSupportedByXNNPack function we already checked
    // auto_pad_str
    return Status(ONNXRUNTIME, NOT_IMPLEMENTED);
  }
  Node* xnnpack_conv_node = nullptr;
  XNNPACK_RETURN_IF_ERROR(ReplaceNode(main_graph, nodeRef,
                                      is_depthwise ? "XnnPackDepthwiseConvolution2d" : "XnnPackConvolution2d", "",
                                      &attrs, "com.microsoft", &xnnpack_conv_node));

  if (bias_node_arg != nullptr) {
    xnnpack_conv_node->MutableInputDefs().push_back(bias_node_arg);
    xnnpack_conv_node->MutableInputArgsCount().push_back(1);
  }
  Node* the_second_transpose_node = nullptr;
  {
    int rank = 4;
    std::vector<int64_t> input_perm = onnx_layout_transformation::ChannelFirstToLastPerm(rank);
    std::vector<int64_t> output_perm = onnx_layout_transformation::ChannelLastToFirstPerm(rank);
    XNNPACK_RETURN_IF_ERROR(TransposeInput(main_graph, input_perm, 0, *xnnpack_conv_node));
    XNNPACK_RETURN_IF_ERROR(TransposeInput(main_graph, weight_perm, 1, *xnnpack_conv_node));
    // TODO: is the input shape right??
    XNNPACK_RETURN_IF_ERROR(main_graph.UpdateShapeInference(*xnnpack_conv_node));
    XNNPACK_RETURN_IF_ERROR(
        TransposeOutput(main_graph, output_perm, 0, *xnnpack_conv_node, &the_second_transpose_node));
  }

  // Conv-clip fusion
  if (next_node_to_fuse) {
    float output_min;
    float output_max;

    bool has_clip =
        optimizer_utils::GetClipConstantMinMax(main_graph, *next_node_to_fuse, output_min, output_max).IsOK();
    if (has_clip) {
      xnnpack_conv_node->AddAttribute("output_min", output_min);
      xnnpack_conv_node->AddAttribute("output_max", output_max);

      ValueMoveInfo value_move_info3(InOutDefSlot{ArgType::kOutput, 0}, InOutDefSlot{ArgType::kOutput, 0});
      // const_cast
      Node* clip_node = main_graph.GetNode(next_node_to_fuse->Index());

      XNNPACK_RETURN_IF_ERROR(
          MoveInputOutput(main_graph, *clip_node, *the_second_transpose_node, value_move_info3, false));
      if (!main_graph.RemoveNode(next_node_to_fuse->Index())) {
        return Status(ONNXRUNTIME, FAIL, "remove node failed");
      }
    }
  }
  return Status::OK();
}
Status XNNPackTransformer::ApplyImpl(Graph& main_graph, bool& modified, int graph_level,
                                     const logging::Logger& logger) const {
  IOnnxRuntimeOpSchemaCollectionPtr ptr = main_graph.GetSchemaRegistry();
  if (ptr == nullptr) {
    return Status::OK();
  }
  const ONNX_NAMESPACE::OpSchema* xnnPackMaxPooling2dSchema = ptr->GetSchema("XnnPackMaxPooling2d", 1, "com.microsoft");
  if (xnnPackMaxPooling2dSchema == nullptr) {
    return Status::OK();
  }
  GraphViewer gv(main_graph);
  // Run constant propagation for XNNPack EP. XNNPack expects that weights are constant.
  // Here we expect a constant folding optimizer will be invoked at least once after this NhwcTransformer and
  // XNNPackTransformer. So I can't register XNNPack Optimizer before the constant folding optimizer.
  std::unordered_set<const NodeArg*> graph_const_values;

  for (auto index : gv.GetNodesInTopologicalOrder()) {
    auto& node = *main_graph.GetNode(index);
    if (!node.ContainsSubgraph() && node.OpType() != "DequantizeLinear" && node.OpType() != "QuantizeLinear" &&
        optimizer_utils::IsOperationDeterministic(node.Domain(), node.OpType())) {
      bool is_all_const = true;
      for (const NodeArg* in : node.InputDefs()) {
        if (!in->Exists()) continue;
        if (graph_const_values.find(in) != graph_const_values.end()) continue;
        if (main_graph.GetConstantInitializer(in->Name(), false) != nullptr) {
          graph_const_values.insert(in);
          continue;
        }
        // This input is not const
        is_all_const = false;
      }
      if (is_all_const) {
        for (const NodeArg* out : node.OutputDefs()) {
          graph_const_values.insert(out);
        }
      }
    }
    XNNPACK_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
  }

  std::vector<NodeIndex> conv_nodes;
  std::vector<NodeIndex> maxpool_nodes;
  // Iterate all the nodes first to figure out what can be run by XNNPack. Then we will update the selected nodes one by
  // one. However, there could be chance that in the first pass we thought a node is supported by XNNPack, then we did
  // some updates on the graph which break the assumption. For example, if there is a Maxpool followd by a Conv. At
  // first, the input channel of Conv is known, then we replaced ONNX Maxpool with XNNPack Maxpool and run shape
  // inference again. Assume the XNNPack maxpool shape inference didn't do a great job and lost of information of the
  // output channel dim of the Maxpool, then this transformer would failed to update ONNX Conv to XNNPack Conv because
  // the later one expects the input channel should be known. So the shape inference functions of XNNPack schemas play a
  // key role here.
  for (auto& nodeRef : gv.Nodes()) {
    auto inputs = nodeRef.InputDefs();
    auto iter_end = nodeRef.InputEdgesEnd();
    if (nodeRef.OpType() == "DequantizeLinear") {
      return Status::OK();
    }
    Status st = IsConvSupportedByXNNPack(nodeRef, graph_const_values, true);
    if (st.IsOK()) {
      conv_nodes.push_back(nodeRef.Index());
    } else if (IsMaxPoolSupportedByXNNPack(nodeRef, true)) {
      maxpool_nodes.push_back(nodeRef.Index());
    }
  }
  for (NodeIndex ni : maxpool_nodes) {
    Node* node_p = main_graph.GetNode(ni);
    if (node_p == nullptr) continue;
    bool node_modified = false;
    XNNPACK_RETURN_IF_ERROR(ReplaceMaxPool(main_graph, *node_p, node_modified));
    modified |= node_modified;
  }

  for (NodeIndex ni : conv_nodes) {
    Node* node_p = main_graph.GetNode(ni);
    if (node_p == nullptr) continue;
    bool node_modified = false;
    XNNPACK_RETURN_IF_ERROR(ReplaceConv(main_graph, *node_p, node_modified));
    modified |= node_modified;
  }
  if (modified) {
    XNNPACK_RETURN_IF_ERROR(main_graph.Resolve());
    auto api_graph = MakeApiGraph(main_graph, cpu_allocator_, kCpuExecutionProvider);
    // Ignore the return value.
    onnx_layout_transformation::Optimize(*api_graph, /*allow_extended_ops*/ true);
  }

  return Status::OK();
}

}  // namespace onnxruntime
