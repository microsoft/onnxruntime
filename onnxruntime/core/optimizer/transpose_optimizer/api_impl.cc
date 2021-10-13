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

const onnx::TensorShapeProto* GetNodeArgShape(const NodeArg* node_arg) {
  if (node_arg == nullptr) {
    return nullptr;
  }
  const auto* type = node_arg->TypeAsProto();
  if (type == nullptr || !utils::HasShape(*type)) {
    return nullptr;
  }
  return &utils::GetShape(*type);
}

const std::string_view OrtValueInfo::Name() const {
  return name_;
}

std::optional<std::vector<int64_t>> OrtValueInfo::Shape() const {
  const auto* node_arg = graph_.GetNodeArg(name_);
  const auto* shape_proto = GetNodeArgShape(node_arg);
  if (shape_proto == nullptr) {
    return std::nullopt;
  }
  TensorShape shape = utils::GetTensorShapeFromTensorShapeProto(*shape_proto);
  return shape.GetDims();
}

void OrtValueInfo::SetShape(const std::vector<int64_t>* shape) {
  auto& node_arg = graph_.GetOrCreateNodeArg(name_, nullptr);
  if (shape == nullptr) {
    node_arg.ClearShape();
    return;
  }
  TensorShapeProto new_shape;
  for (int64_t d : *shape) {
    const auto dim = new_shape.add_dim();
    if (d > 0) {
      dim->set_dim_value(d);
    }
  }
  node_arg.SetShape(new_shape);
}

void CopyDim(const onnx::TensorShapeProto_Dimension& src, onnx::TensorShapeProto_Dimension& dst) {
  if (utils::HasDimValue(src)) {
    dst.set_dim_value(src.dim_value());
  }
  if (utils::HasDimValue(src)) {
    dst.set_dim_param(src.dim_param());
  }
}

void OrtValueInfo::PermuteDims(const std::vector<int64_t>& perm) {
  auto* node_arg = graph_.GetNodeArg(name_);
  const auto* shape_proto = GetNodeArgShape(node_arg);
  if (shape_proto == nullptr) {
    return;
  }
  //ORT_ENFORCE(perm.size() == (size_t)shape_proto->dim_size(), "Permutation ", perm, " length does not match shape ",
  //            shape_proto->dim_size());
  TensorShapeProto new_shape;
  for (int64_t p : perm) {
    //ORT_ENFORCE(0 <= p && (int)p < shape_proto->dim_size(), "Permutation ", perm, " out of bounds for shape ",
    //            shape_proto->dim_size());
    auto& dim = *new_shape.add_dim();
    const auto& src_dim = shape_proto->dim((int)p);
    CopyDim(src_dim, dim);
  }
  node_arg->SetShape(new_shape);
}

void OrtValueInfo::UnsqueezeDims(const std::vector<int64_t>& axes) {
  auto* node_arg = graph_.GetNodeArg(name_);
  const auto* shape_proto = GetNodeArgShape(node_arg);
  if (shape_proto == nullptr) {
    return;
  }
  size_t rank = shape_proto->dim_size();
  TensorShapeProto new_shape;
  int j = 0;
  int64_t i = 0;
  while (true) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
      new_shape.add_dim()->set_dim_value(1);
    } else if (gsl::narrow_cast<size_t>(j) < rank) {
      auto& dim = *new_shape.add_dim();
      const auto& src_dim = shape_proto->dim(j);
      CopyDim(src_dim, dim);
      ++j;
    } else {
      break;
    }
    ++i;
  }
  node_arg->SetShape(new_shape);
}

std::vector<int64_t> OrtTensor::Shape() const {
  std::vector<int64_t> shape;
  for (int64_t d : tensor_proto_.dims()) {
    shape.push_back(d);
  }
  return shape;
}

std::vector<int64_t> OrtTensor::DataInt64() const {
  const DataTypeImpl* tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto_.data_type())->GetElementType();
  auto tensor_shape_dims = utils::GetTensorShapeFromTensorProto(tensor_proto_);
  TensorShape tensor_shape{std::move(tensor_shape_dims)};
  const auto tensor = onnxruntime::Tensor::Create(tensor_dtype, tensor_shape, cpu_allocator_);
  const auto status = utils::TensorProtoToTensor(Env::Default(), graph_.ModelPath().ToPathString().c_str(),
                                                 tensor_proto_, *tensor);
  const int64_t* data = tensor->Data<int64_t>();
  size_t num_elements = gsl::narrow_cast<size_t>(tensor->Shape().Size());
  std::vector<int64_t> int_data(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    int_data[i] = *data++;
  }
  return int_data;
}


template<class T>
std::vector<std::string_view> NodeArgsToStrings(T node_args) {
  std::vector<std::string_view> result;
  for (const auto* arg : node_args) {
    result.push_back(arg->Name());
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
  for (const auto& pair : attributes) {
    node_.AddAttribute(pair.first, pair.second);
  }
}

void OrtNode::ClearAttribute(const std::string_view name) {
  node_.ClearAttribute(std::string(name));
}

void OrtNode::AddInput(const std::string_view name) {
  const std::string name_str(name);
  NodeArg& node_arg = graph_.GetOrCreateNodeArg(name_str, nullptr);
  // Append a 1 to ArgsCount or increment the first entry with value 0.
  std::vector<int32_t>& args_count = node_.MutableInputArgsCount();
  size_t i = 0;
  while (i < args_count.size() && args_count[i] > 0) {
    ++i;
  }
  args_count.push_back(1);
  auto& mutable_input_defs = node_.MutableInputDefs();
  int inp_index = gsl::narrow_cast<int>(mutable_input_defs.size());
  mutable_input_defs.push_back(&node_arg);
  if (node_arg.Exists()) {
    graph_.AddConsumerNode(name_str, &node_);
    const auto* inp_node = graph_.GetProducerNode(name_str);
    if (inp_node != nullptr) {
      int inp_node_out_index = graph_utils::GetNodeOutputIndexFromOutputName(*inp_node, name_str);
      graph_.AddEdge(inp_node->Index(), node_.Index(), inp_node_out_index, inp_index);
    }
  }
}

void OrtNode::SetInput(size_t i, const std::string_view name) {
  const std::string name_str(name);
  NodeArg* new_node_arg = &graph_.GetOrCreateNodeArg(name_str, nullptr);
  auto& mutable_input_defs = node_.MutableInputDefs();
  NodeArg* old_node_arg = mutable_input_defs[i];
  if (old_node_arg->Exists()) {
    // Input may be referenced multiple times. Only remove from consumers if all references are gone.
    size_t usages = 0;
    for (const auto* node_arg : mutable_input_defs) {
      if (node_arg == old_node_arg) {
        ++usages;
      }
    }
    if (usages == 1) {
      graph_.RemoveConsumerNode(old_node_arg->Name(), &node_);
    }
    const auto* old_node = graph_.GetProducerNode(old_node_arg->Name());
    if (old_node != nullptr) {
      int inp_node_out_index = graph_utils::GetNodeOutputIndexFromOutputName(*old_node, old_node_arg->Name());
      graph_.RemoveEdge(old_node->Index(), node_.Index(), inp_node_out_index, (int)i);
    }
  }
  mutable_input_defs[i] = new_node_arg;
  if (new_node_arg->Exists()) {
    graph_.AddConsumerNode(name_str, &node_);
    const auto* inp_node = graph_.GetProducerNode(name_str);
    if (inp_node != nullptr) {
      int inp_node_out_index = graph_utils::GetNodeOutputIndexFromOutputName(*inp_node, name_str);
      graph_.AddEdge(inp_node->Index(), node_.Index(), inp_node_out_index, (int)i);
    }
  }
}


OrtGraph::OrtGraph(onnxruntime::Graph& graph, AllocatorPtr cpu_allocator, const logging::Logger& logger,
                   const char* new_node_ep) : graph_(graph), cpu_allocator_(cpu_allocator), logger_(logger),
                   new_node_ep_(new_node_ep) { };
std::optional<int64_t> OrtGraph::Opset(const std::string_view domain) const {
  const auto& version_map = graph_.DomainToVersionMap();
  const auto match = version_map.find(std::string(domain));
  if (match == version_map.end()) {
    return std::nullopt;
  }
  return match->second;
}

std::vector<std::unique_ptr<api::Node>> OrtGraph::Nodes() const {
  GraphViewer graph_viewer(graph_);
  std::vector<std::unique_ptr<api::Node>> nodes;
  const auto& sorted_nodes = graph_viewer.GetNodesInTopologicalOrder();
  nodes.reserve(sorted_nodes.size());
  for (const auto index : sorted_nodes) {
    auto& node = *graph_.GetNode(index);
    nodes.push_back(std::unique_ptr<api::Node>(new OrtNode(node, graph_)));
  }
  return nodes;
}

std::vector<std::string_view> OrtGraph::Inputs() const {
  return NodeArgsToStrings(graph_.GetInputsIncludingInitializers());
}

std::vector<std::string_view> OrtGraph::Outputs() const {
  return NodeArgsToStrings(graph_.GetOutputs());
}

std::unique_ptr<api::Tensor> OrtGraph::GetConstant(const std::string_view name) const {
  const auto* tensor = graph_.GetConstantInitializer(std::string(name), false);
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
  const auto nodes = graph_.GetConsumerNodes(std::string(name));
  for (const auto node : nodes) {
    for (const auto* input : node->ImplicitInputDefs()) {
      if (input->Exists() && input->Name() == name) {
        consumers->comprehensive = false;
        break;
      }
    }
    for (const auto* input : node->InputDefs()) {
      if (input->Exists() && input->Name() == name) {
        consumers->nodes.push_back(std::unique_ptr<api::Node>(new OrtNode(*graph_.GetNode(node->Index()), graph_)));
        break;
      }
    }
  }

  const auto& graph_outputs = graph_.GetOutputs();
  for (const auto* output : graph_outputs) {
    if (output->Name() == name) {
      consumers->comprehensive = false;
    }
  }

  return consumers;
}

std::unique_ptr<api::Node> OrtGraph::GetNodeProducingOutput(const std::string_view name) const {
  auto* node = graph_.GetMutableProducerNode(std::string(name));
  if (node == nullptr) {
    return std::unique_ptr<api::Node>(nullptr);
  }
  return std::unique_ptr<api::Node>(new OrtNode(*node, graph_));
}

void OrtGraph::TransposeInitializer(const std::string_view name, const std::vector<int64_t>& perm) {
  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  const std::string name_str(name);
  bool success = graph_.GetInitializedTensor(name_str, tensor_proto);
  ORT_ENFORCE(success, "Failed to find initializer for name: ", name_str);
  const DataTypeImpl* tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto->data_type())->GetElementType();
  const auto tensor_shape_dims = utils::GetTensorShapeFromTensorProto(*tensor_proto);
  TensorShape tensor_shape{tensor_shape_dims};
  std::unique_ptr<Tensor> in_tensor = Tensor::Create(tensor_dtype, tensor_shape, cpu_allocator_);

  std::vector<int64_t> new_tensor_shape_dims;
  std::vector<size_t> permutations;
  for (int64_t p : perm) {
    permutations.push_back((size_t)p);
    new_tensor_shape_dims.push_back(tensor_shape_dims[(size_t)p]);
  }
  TensorShape new_tensor_shape(new_tensor_shape_dims);

  std::unique_ptr<Tensor> out_tensor = Tensor::Create(tensor_dtype, new_tensor_shape, cpu_allocator_);

  const auto status = utils::TensorProtoToTensor(Env::Default(), graph_.ModelPath().ToPathString().c_str(),
                                                 *tensor_proto, *in_tensor);
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());

  const auto status2 = Transpose::DoTranspose(permutations, *in_tensor, *out_tensor);
  ORT_ENFORCE(status2.IsOK(), status2.ErrorMessage());

  auto* node_arg = graph_.GetNodeArg(name_str);
  TensorShapeProto new_shape;
  for (int64_t d : new_tensor_shape_dims) {
    new_shape.add_dim()->set_dim_value(d);
  }
  node_arg->SetShape(new_shape);

  ONNX_NAMESPACE::TensorProto new_tensor_proto = utils::TensorToTensorProto(*out_tensor, name_str);
  graph_.RemoveInitializedTensor(name_str);
  graph_.AddInitializedTensor(new_tensor_proto);
}

void OrtGraph::ReshapeInitializer(const std::string_view name, const std::vector<int64_t>& shape) {
  const std::string name_str(name);
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
  ORT_ENFORCE(new_num_elts == old_num_elts, "Cannot reshape initializer ", name,
              " to have different number of elements");

  auto new_tensor_proto = ONNX_NAMESPACE::TensorProto(*tensor_proto);
  new_tensor_proto.clear_dims();
  for (int64_t d : shape) {
    new_tensor_proto.add_dims(d);
  }
  graph_.RemoveInitializedTensor(name_str);
  graph_.AddInitializedTensor(new_tensor_proto);

  auto* node_arg = graph_.GetNodeArg(name_str);
  TensorShapeProto new_shape;
  for (int64_t d : shape) {
    new_shape.add_dim()->set_dim_value(d);
  }
  node_arg->SetShape(new_shape);
}

std::unique_ptr<api::Node> OrtGraph::AddNode(const std::string_view op_type,
                                             const std::vector<std::string_view>& inputs, size_t num_outputs, 
                                             const std::string_view domain) {
  const std::string op_type_str(op_type);
  std::string name = graph_.GenerateNodeName(op_type_str);
  std::vector<NodeArg*> input_args;
  std::vector<NodeArg*> output_args;

  for (const auto& input : inputs) {
    NodeArg* arg = &graph_.GetOrCreateNodeArg(std::string(input), nullptr);
    input_args.push_back(arg);
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    std::string output = graph_.GenerateNodeArgName(name + "_out" + std::to_string(i));
    NodeArg* arg = &graph_.GetOrCreateNodeArg(output, nullptr);
    output_args.push_back(arg);
  }
  std::vector<NodeArg*> outputs;
  Node& node = graph_.AddNode(name, op_type_str, "Added in transpose optimizer", input_args, output_args, nullptr,
                              std::string(domain));

  if (new_node_ep_ != nullptr) {
    node.SetExecutionProviderType(new_node_ep_);
  }

  for (size_t i = 0; i < input_args.size(); ++i) {
    NodeArg* arg = input_args[i];
    if (arg->Exists()) {
      const std::string& name_str = arg->Name();
      graph_.AddConsumerNode(name_str, &node);
      const auto* inp_node = graph_.GetProducerNode(name_str);
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
  for (const auto* node_arg : ort_node.InputDefs()) {
    if (node_arg->Exists()) {
      graph_.RemoveConsumerNode(node_arg->Name(), &ort_node);
    }
  }
  graph_.RemoveNode(ort_node.Index());
}

void OrtGraph::RemoveInitializer(const std::string_view name) {
  graph_.RemoveInitializedTensor(std::string(name));
}

const std::string_view OrtGraph::AddInitializerInt64(const std::vector<int64_t>& shape,
                                                     const std::vector<int64_t>& values) {
  std::string name = graph_.GenerateNodeArgName("const_transpose_optimizer");
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  tensor_proto.set_name(name);
  tensor_proto.set_raw_data(values.data(), values.size() * sizeof(int64_t));
  for (int64_t dim : shape) {
    tensor_proto.add_dims(dim);
  }
  const auto& node_arg = graph_utils::AddInitializer(graph_, tensor_proto);
  return node_arg.Name();
}

void OrtGraph::MoveOutput(api::Node& src_node, size_t src_idx, api::Node& dst_node, size_t dst_idx) {
  Node& src_ort_node = static_cast<OrtNode&>(src_node).Node();
  Node& dst_ort_node = static_cast<OrtNode&>(dst_node).Node();

  std::vector<NodeArg*>& src_output_defs = src_ort_node.MutableOutputDefs();
  std::vector<NodeArg*>& dst_output_defs = dst_ort_node.MutableOutputDefs();
  const NodeArg* node_arg = src_output_defs[src_idx];
  const std::string& node_arg_name = node_arg->Name();
  dst_output_defs[dst_idx] = src_output_defs[src_idx];
  const auto dst_node_idx = dst_ort_node.Index();
  const auto src_node_idx = src_ort_node.Index();
  graph_.UpdateProducerNode(node_arg_name, dst_node_idx);

  const auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(src_ort_node, src_idx);
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
    const auto status = dst_arg.UpdateTypeAndShape(*src_arg, /*strict*/ false, /*override_types*/ true, logger_);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
  }
}

}  // namespace onnxruntime
