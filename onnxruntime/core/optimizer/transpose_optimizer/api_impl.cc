// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/transpose_optimizer/ort_transpose_optimizer.h"
#include "core/optimizer/transpose_optimizer/api_impl.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/providers/cpu/tensor/transpose.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_layout_transformation;

namespace onnxruntime {

const std::string_view OrtValueInfo::Name() const {
  return name_;
}
std::optional<std::vector<int64_t>> OrtValueInfo::Shape() const {
  auto* node_arg = graph_.GetNodeArg(name_);
  if (node_arg == nullptr) {
    return std::nullopt;
  }
  auto* type = node_arg->TypeAsProto();
  if (type == nullptr || !utils::HasShape(*type)) {
    return std::nullopt;
  }
  auto& shape_proto = utils::GetShape(*type);
  std::vector<int64_t> shape;
  shape.reserve(shape_proto.dim_size());
  for (auto dim : shape_proto.dim()) {
    if (dim.has_dim_value()) {
      shape.push_back(dim.dim_value());
    } else {
      shape.push_back(-1);
    }
  }
  return shape;
}
void OrtValueInfo::SetShape(const std::vector<int64_t>* shape) const {
  auto& node_arg = graph_.GetOrCreateNodeArg(name_, nullptr);
  if (shape == nullptr) {
    node_arg.ClearShape();
    return;
  }
  TensorShapeProto new_shape;
  for (int64_t d : *shape) {
    new_shape.add_dim()->set_dim_value(d);
  }
  node_arg.SetShape(new_shape);
}
void OrtValueInfo::PermuteDims(const std::vector<int64_t>& perm) const {
  auto& node_arg = graph_.GetOrCreateNodeArg(name_, nullptr);
  auto* type = node_arg.TypeAsProto();
  if (type == nullptr || !utils::HasShape(*type)) {
    return;
  }
  auto& shape_proto = utils::GetShape(*type);
  // TODO: Assert size matches
  TensorShapeProto new_shape;
  for (int64_t p : perm) {
    auto& dim = *new_shape.add_dim();
    auto& src_dim = shape_proto.dim((int)p);
    if (src_dim.has_dim_value()) {
      dim.set_dim_value(src_dim.dim_value());
    }
    if (src_dim.has_dim_param()) {
      dim.set_dim_param(src_dim.dim_param());
    }
  }
  node_arg.SetShape(new_shape);
}
void OrtValueInfo::UnsqueezeDims(const std::vector<int64_t>& axes) const {
  auto& node_arg = graph_.GetOrCreateNodeArg(name_, nullptr);
  auto* type = node_arg.TypeAsProto();
  if (type == nullptr || !utils::HasShape(*type)) {
    return;
  }
  auto& shape_proto = utils::GetShape(*type);
  size_t rank = shape_proto.dim_size();
  // TODO: Assert axes in range
  TensorShapeProto new_shape;
  size_t j = 0;
  size_t i = 0;
  while (true) {
    if (std::find(axes.begin(), axes.end(), (int64_t)i) != axes.end()) {
      new_shape.add_dim()->set_dim_value(1);
    } else if (j < rank) {
      auto& dim = *new_shape.add_dim();
      auto& src_dim = shape_proto.dim((int)j);
      if (src_dim.has_dim_value()) {
        dim.set_dim_value(src_dim.dim_value());
      }
      if (src_dim.has_dim_param()) {
        dim.set_dim_param(src_dim.dim_param());
      }
      ++j;
    } else {
      break;
    }
    ++i;
  }
  node_arg.SetShape(new_shape);
}


std::vector<int64_t> OrtTensor::Shape() const {
  std::vector<int64_t> shape;
  for (int64_t d : tensor_proto_.dims()) {
    shape.push_back(d);
  }
  return shape;
}
std::vector<int64_t> OrtTensor::DataInt64() const {
  // TODO: assert data type is int64
  const DataTypeImpl* const tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto_.data_type())->GetElementType();
  auto tensor_shape_dims = utils::GetTensorShapeFromTensorProto(tensor_proto_);
  TensorShape tensor_shape{tensor_shape_dims};
  auto tensor = onnxruntime::Tensor::Create(tensor_dtype, tensor_shape, cpu_allocator_);
  auto status = utils::TensorProtoToTensor(Env::Default(), graph_.ModelPath().ToPathString().c_str(), tensor_proto_, *tensor);
  const int64_t* data = tensor->Data<int64_t>();
  size_t num_elements = tensor->SizeInBytes() / sizeof(int64_t);
  std::vector<int64_t> int_data(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    int_data[i] = *data++;
  }
  return int_data;
}


std::vector<std::string_view> NodeArgsToStrings(ConstPointerContainer<std::vector<NodeArg*>> node_args) {
  std::vector<std::string_view> result;
  for (auto* arg : node_args) {
    // TODO: can this be null?
    if (arg->Exists()) {
      result.push_back(arg->Name());
    } else {
      result.push_back("");
    }
  }
  return result;
}


const std::string_view OrtNode::Name() const {
  return node_.Name();
}
const std::string_view OrtNode::OpType() const {
  return node_.OpType();
}
const std::string_view OrtNode::Domain() const {
  return node_.Domain();
}
std::vector<std::string_view> OrtNode::Inputs() const {
  return NodeArgsToStrings(node_.InputDefs());
}
std::vector<std::string_view> OrtNode::Outputs() const {
  return NodeArgsToStrings(node_.OutputDefs());
}
std::optional<int64_t> OrtNode::GetAttributeInt(const std::string_view name) const {
  const onnx::AttributeProto* attr = graph_utils::GetNodeAttribute(node_, std::string(name));
  if (attr == nullptr || attr->type() != onnx::AttributeProto_AttributeType_INT) {
    return std::nullopt;
  }
  return attr->i();
}
std::optional<std::vector<int64_t>> OrtNode::GetAttributeInts(const std::string_view name) const {
  const onnx::AttributeProto* attr = graph_utils::GetNodeAttribute(node_, std::string(name));
  if (attr == nullptr || attr->type() != onnx::AttributeProto_AttributeType_INTS) {
    return std::nullopt;
  }
  std::vector<int64_t> value;
  for (int64_t x : attr->ints()) {
    value.push_back(x);
  }
  return value;
}
void OrtNode::SetAttributeInt(const std::string_view name, int64_t value) {
  node_.AddAttribute(std::string(name), value);
}
void OrtNode::SetAttributeInts(const std::string_view name, const std::vector<int64_t>& value) {
  node_.AddAttribute(std::string(name), value);
}
void OrtNode::CopyAttributes(const api::Node& node) {
  const OrtNode& ort_node = static_cast<const OrtNode&>(node);
  const NodeAttributes& attributes = ort_node.node_.GetAttributes();
  for (auto pair : attributes) {
    node_.AddAttribute(pair.first, pair.second);
  }
}
void OrtNode::ClearAttribute(const std::string_view name) {
  node_.ClearAttribute(std::string(name));
}
void OrtNode::AddInput(const std::string_view name) {
  // TODO: optional input
  auto name_str = std::string(name);
  NodeArg* node_arg = &graph_.GetOrCreateNodeArg(name_str, nullptr);
  //graph_utils::AddNodeInput(node_)
  std::vector<int32_t>& args_count = node_.MutableInputArgsCount();
  // TODO: how to know which case?
  size_t i = 0;
  while (i < args_count.size() && args_count[i] > 0) {
    ++i;
  }
  if (i < args_count.size()) {
    ++args_count[i];
  } else {
    args_count.push_back(1);
  }
  auto& muable_input_defs = node_.MutableInputDefs();
  int inp_index = (int)muable_input_defs.size();
  muable_input_defs.push_back(node_arg);
  graph_.AddConsumerNode(name_str, &node_);
  auto* inp_node = graph_.GetProducerNode(name_str);
  // TODO: is this from same graph?
  if (inp_node != nullptr) {
    int inp_node_out_index = graph_utils::GetNodeOutputIndexFromOutputName(*inp_node, name_str);
    graph_.AddEdge(inp_node->Index(), node_.Index(), inp_node_out_index, inp_index);
  }
}
void OrtNode::SetInput(size_t i, const std::string_view name) {
  // TODO: optional input
  auto name_str = std::string(name);
  NodeArg* new_node_arg = &graph_.GetOrCreateNodeArg(name_str, nullptr);
  auto& mutable_input_defs = node_.MutableInputDefs();
  NodeArg* old_node_arg = mutable_input_defs[i];
  if (old_node_arg->Exists()) {
    size_t usages = 0;
    for (auto* node_arg : mutable_input_defs) {
      if (node_arg == old_node_arg) {
        ++usages;
      }
    }
    // TODO: assert usages >= 1?
    if (usages == 1) {
      graph_.RemoveConsumerNode(old_node_arg->Name(), &node_);
    }
    auto* old_node = graph_.GetProducerNode(old_node_arg->Name());
    if (old_node != nullptr) {
      int inp_node_out_index = graph_utils::GetNodeOutputIndexFromOutputName(*old_node, old_node_arg->Name());
      graph_.RemoveEdge(old_node->Index(), node_.Index(), inp_node_out_index, (int)i);
    }
  }
  mutable_input_defs[i] = new_node_arg;
  if (new_node_arg->Exists()) {
    graph_.AddConsumerNode(name_str, &node_);
    auto* inp_node = graph_.GetProducerNode(name_str);
    if (inp_node != nullptr) {
      int inp_node_out_index = graph_utils::GetNodeOutputIndexFromOutputName(*inp_node, name_str);
      graph_.AddEdge(inp_node->Index(), node_.Index(), inp_node_out_index, (int)i);
    }
  }
}


OrtGraph::OrtGraph(onnxruntime::Graph& graph, AllocatorPtr cpu_allocator, const logging::Logger& logger) : graph_(graph), cpu_allocator_(cpu_allocator), logger_(logger) {
  const std::vector<const NodeArg*>& graph_inputs = graph.GetInputsIncludingInitializers();
  inputs_.reserve(graph_inputs.size());
  for (const NodeArg* input : graph_inputs) {
    inputs_.push_back(input->Name());
  }
  const std::vector<const NodeArg*>& graph_outputs = graph.GetOutputs();
  outputs_.reserve(graph_inputs.size());
  for (const NodeArg* input : graph_outputs) {
    outputs_.push_back(input->Name());
  }
};

std::optional<int64_t> OrtGraph::Opset(const std::string_view domain) const {
  auto version_map = graph_.DomainToVersionMap();
  auto match = version_map.find(std::string(domain));
  if (match == version_map.end()) {
    return std::nullopt;
  }
  return match->second;
}

std::vector<std::unique_ptr<api::Node>> OrtGraph::Nodes() const {
  GraphViewer graph_viewer(graph_);
  auto nodes = std::vector<std::unique_ptr<api::Node>>();
  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph_.GetNode(index);
    nodes.push_back(std::unique_ptr<api::Node>(new OrtNode(node, graph_)));
  }
  return nodes;
}

std::vector<std::string_view> OrtGraph::Inputs() const {
  return inputs_;
}
std::vector<std::string_view> OrtGraph::Outputs() const {
  return outputs_;
}
std::unique_ptr<api::Tensor> OrtGraph::GetConstant(const std::string_view name) const {
  auto* tensor = graph_.GetConstantInitializer(std::string(name), false);
  if (tensor == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<api::Tensor>(new OrtTensor(*tensor, graph_, cpu_allocator_));
}
std::unique_ptr<api::ValueInfo> OrtGraph::GetValueInfo(const std::string_view name) const {
  return std::unique_ptr<api::ValueInfo>(new OrtValueInfo(graph_, std::string(name)));
}
std::unique_ptr<api::ValueConsumers> OrtGraph::GetValueConsumers(const std::string_view name) const {
  auto consumers = std::make_unique<api::ValueConsumers>();
  consumers->comprehensive = true;
  auto nodes = graph_.GetMutableConsumerNodes(std::string(name));
  for (auto node : nodes) {
    for (auto* input : node->ImplicitInputDefs()) {
      if (input->Exists() && input->Name() == name) {
        consumers->comprehensive = false;
        break;
      }
    }
    for (auto* input : node->InputDefs()) {
      if (input->Exists() && input->Name() == name) {
        consumers->nodes.push_back(std::unique_ptr<api::Node>(new OrtNode(*node, graph_)));
        break;
      }
    }
  }

  auto& graph_outputs = graph_.GetOutputs();
  for (auto* output : graph_outputs) {
    if (output->Name() == name) {
      consumers->comprehensive = false;
    }
  }

  return consumers;
}
std::unique_ptr<api::Node> OrtGraph::GetNodeByOutput(const std::string_view name) const {
  auto* node = graph_.GetMutableProducerNode(std::string(name));
  if (node == nullptr) {
    return std::unique_ptr<api::Node>(nullptr);
  }
  return std::unique_ptr<api::Node>(new OrtNode(*node, graph_));
}
void OrtGraph::TransposeInitializer(const std::string_view name, const std::vector<int64_t> perm) {
  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  auto name_str = std::string(name);
  bool success = graph_.GetInitializedTensor(name_str, tensor_proto);
  ORT_IGNORE_RETURN_VALUE(success);
  //TODO: assert success
  const DataTypeImpl* const tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto->data_type())->GetElementType();
  auto tensor_shape_dims = utils::GetTensorShapeFromTensorProto(*tensor_proto);
  TensorShape tensor_shape{tensor_shape_dims};
  std::unique_ptr<Tensor> in_tensor = Tensor::Create(tensor_dtype, tensor_shape, cpu_allocator_);

  std::vector<int64_t> new_tensor_shape_dims;
  std::vector<size_t> permutations;
  for (int64_t p : perm) {
    permutations.push_back((size_t)p);
    new_tensor_shape_dims.push_back(tensor_shape_dims[p]);
  }
  auto new_tensor_shape = TensorShape(new_tensor_shape_dims);

  std::unique_ptr<Tensor> out_tensor = Tensor::Create(tensor_dtype, new_tensor_shape, cpu_allocator_);

  auto status = utils::TensorProtoToTensor(Env::Default(), graph_.ModelPath().ToPathString().c_str(), *tensor_proto, *in_tensor);
  ORT_IGNORE_RETURN_VALUE(status);
  //TODO: assert status

  Transpose::DoTranspose(permutations, *in_tensor, *out_tensor);

  auto* node_arg = graph_.GetNodeArg(name_str);
  if (node_arg != nullptr) {
    TensorShapeProto new_shape;
    for (int64_t d : new_tensor_shape_dims) {
      new_shape.add_dim()->set_dim_value(d);
    }
    node_arg->SetShape(new_shape);
  }

  ONNX_NAMESPACE::TensorProto new_tensor_proto = utils::TensorToTensorProto(*out_tensor, name_str);
  graph_.RemoveInitializedTensor(name_str);
  graph_.AddInitializedTensor(new_tensor_proto);
  //auto status2 = graph_.ReplaceInitializedTensor(new_tensor_proto);
  //TODO: assert status
}
void OrtGraph::ReshapeInitializer(const std::string_view name, const std::vector<int64_t>& shape) {
  auto name_str = std::string(name);
  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  bool success = graph_.GetInitializedTensor(name_str, tensor_proto);
  ORT_ENFORCE(success, "Failed to find initializer to reshape with name ", name);
  int64_t new_num_elts = 1;
  for (int64_t d : shape) {
    new_num_elts *= d;
  }
  int64_t old_num_elts = 1;
  for (int64_t d : tensor_proto->dims()) {
    old_num_elts *= d;
  }
  ORT_ENFORCE(new_num_elts == old_num_elts, "Cannot reshape initializer ", name, " to have different number of elements");

  auto new_tensor_proto = ONNX_NAMESPACE::TensorProto(*tensor_proto);
  new_tensor_proto.clear_dims();
  for (int64_t d : shape) {
    new_tensor_proto.add_dims(d);
  }
  graph_.RemoveInitializedTensor(name_str);
  graph_.AddInitializedTensor(new_tensor_proto);

  auto* node_arg = graph_.GetNodeArg(name_str);
  if (node_arg != nullptr) {
    TensorShapeProto new_shape;
    for (int64_t d : shape) {
      new_shape.add_dim()->set_dim_value(d);
    }
    node_arg->SetShape(new_shape);
  }
}
std::unique_ptr<api::Node> OrtGraph::AddNode(const std::string_view op_type, const std::vector<std::string_view>& inputs, size_t num_outputs, const std::string_view domain) {
  std::string op_type_str = std::string(op_type);
  std::string name = graph_.GenerateNodeName(op_type_str);
  std::vector<NodeArg*> input_args;
  std::vector<NodeArg*> output_args;
  for (auto input : inputs) {
    // TODO: How to deal with optional inputs?
    // TODO: this null type may be a problem for constants
    NodeArg* arg = &graph_.GetOrCreateNodeArg(std::string(input), nullptr);
    input_args.push_back(arg);
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    std::string output = graph_.GenerateNodeArgName(name + "_out" + std::to_string(i));
    NodeArg* arg = &graph_.GetOrCreateNodeArg(output, nullptr);
    output_args.push_back(arg);
  }
  std::vector<NodeArg*> outputs;
  Node& node = graph_.AddNode(name, op_type_str, "Added in transpose optimizer", input_args, output_args, nullptr, std::string(domain));
  node.SetExecutionProviderType(kCpuExecutionProvider);

  for (size_t i = 0; i < input_args.size(); ++i) {
    NodeArg* arg = input_args[i];
    if (arg->Exists()) {
      auto name_str = arg->Name();
      graph_.AddConsumerNode(name_str, &node);
      auto* inp_node = graph_.GetProducerNode(name_str);
      if (inp_node != nullptr) {
        int inp_node_out_index = graph_utils::GetNodeOutputIndexFromOutputName(*inp_node, name_str);
        graph_.AddEdge(inp_node->Index(), node.Index(), inp_node_out_index, (int)i);
      }
    }
  }
  for (NodeArg* arg : output_args) {
    graph_.UpdateProducerNode(arg->Name(), node.Index());
  }
  return std::unique_ptr<api::Node>(new OrtNode(node, graph_));
}
void OrtGraph::RemoveNode(api::Node& node) {
  Node& ort_node = static_cast<OrtNode&>(node).Node();
  for (auto* node_arg : ort_node.InputDefs()) {
    std::string input_name = node_arg->Name();
    if (node_arg->Exists()) {
      // TODO make sure this is right for optional args
      graph_.RemoveConsumerNode(node_arg->Name(), &ort_node);
    }
  }
  graph_.RemoveNode(ort_node.Index());
}
void OrtGraph::RemoveInitializer(const std::string_view name) {
  graph_.RemoveInitializedTensor(std::string(name));
}
const std::string_view OrtGraph::AddInitializerInt64(const std::vector<int64_t>& shape, const std::vector<int64_t>& values) {
  std::string name = graph_.GenerateNodeArgName("const_transpose_optimizer");
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  tensor_proto.set_name(name);
  tensor_proto.set_raw_data(values.data(), values.size() * sizeof(int64_t));
  for (int64_t dim : shape) {
    tensor_proto.add_dims(dim);
  }
  auto& node_arg = graph_utils::AddInitializer(graph_, tensor_proto);
  return node_arg.Name();
}
void OrtGraph::MoveOutput(api::Node& src_node, size_t src_idx, api::Node& dst_node, size_t dst_idx) {
  // TODO: double check node args and edges
  Node& src_ort_node = static_cast<OrtNode&>(src_node).Node();
  Node& dst_ort_node = static_cast<OrtNode&>(dst_node).Node();

  std::vector<NodeArg*>& src_output_defs = src_ort_node.MutableOutputDefs();
  std::vector<NodeArg*>& dst_output_defs = dst_ort_node.MutableOutputDefs();
  const NodeArg* node_arg = src_output_defs[src_idx];
  std::string node_arg_name = node_arg->Name();
  dst_output_defs[dst_idx] = src_output_defs[src_idx];
  auto dst_node_idx = dst_ort_node.Index();
  auto src_node_idx = src_ort_node.Index();
  graph_.UpdateProducerNode(node_arg_name, dst_node_idx);

  auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(src_ort_node, src_idx);
  for (auto cur = output_edges.cbegin(), end = output_edges.cend(); cur != end; ++cur) {
    graph_.AddEdge(dst_node_idx, cur->dst_node, (int)dst_idx, (int)cur->dst_arg_index);
  }
  graph_utils::GraphEdge::RemoveGraphEdges(graph_, output_edges);

  std::string new_name = graph_.GenerateNodeArgName(src_ort_node.Name());
  src_output_defs[src_idx] = &graph_.GetOrCreateNodeArg(new_name, nullptr);
  graph_.UpdateProducerNode(new_name, src_node_idx);
}
void OrtGraph::CopyValueInfo(const std::string_view src_name, const std::string_view dst_name) {
  NodeArg* src_arg = graph_.GetNodeArg(std::string(src_name));
  if (src_arg != nullptr) {
    NodeArg& dst_arg = graph_.GetOrCreateNodeArg(std::string(dst_name), src_arg->TypeAsProto());
    const TensorShapeProto* shape = src_arg->Shape();
    if (shape == nullptr) {
      dst_arg.ClearShape();
    } else {
      dst_arg.SetShape(*shape);
    }
    // TODO: is this dereference safe?
    dst_arg.UpdateTypeAndShape(*src_arg, /*strict*/ false, /*override_types*/ true, logger_);
  }
}

}  // namespace onnxruntime