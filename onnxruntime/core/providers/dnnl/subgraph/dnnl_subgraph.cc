// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_subgraph.h"
#include <queue>

namespace onnxruntime {
namespace ort_dnnl {

DnnlTensor DnnlNode::empty_tensor_ = DnnlTensor("");

DnnlTensor::DnnlTensor(const NodeArg* arg) {
  if (!arg || !arg->Exists()) {
    tensor_name_ = "";
  } else {
    tensor_name_ = arg->Name();
  }
  // because the passed in ort graph will be released after compile
  // need to save the type/shape in dnnl IR
  arg_type_ = arg->Type();
  arg_type_proto_ = ONNX_NAMESPACE::TypeProto::Create();
  arg_type_proto_->copy_from(arg->TypeAsProto());
}

DnnlTensor::DnnlTensor(std::string name) {
  tensor_name_ = name;
  arg_type_ = nullptr;
  arg_type_proto_ = nullptr;
}

std::string DnnlTensor::Name() const {
  return tensor_name_;
}

const ONNX_NAMESPACE::TensorShapeProto* DnnlTensor::GetShape() const{
  if (arg_type_proto_ == nullptr || arg_type_ == nullptr) {
    return nullptr;
  }

  if (arg_type_proto_->value_case() != ONNX_NAMESPACE::TypeProto::ValueCase::kTensorType) {
    return nullptr;
  }
  auto& tensor_type = arg_type_proto_->tensor_type();
  if (tensor_type.has_shape()) {
    return &tensor_type.shape();
  }
  return nullptr;
}

dnnl::memory::dims DnnlTensor::Dim() const {
  if (arg_type_proto_ == nullptr || arg_type_ == nullptr) {
    return dnnl::memory::dims();
  }
  auto* shape_proto = GetShape();
  // a shape without any information
  if (shape_proto == nullptr) {
    LOGS_DEFAULT(INFO) << "nullptr shape for " << arg_type_ << ": " << tensor_name_;
    return dnnl::memory::dims();
  }
  std::vector<int64_t> shape;
  const auto& dims = shape_proto->dim();
  for (const auto& dim : dims) {
    bool has_dim_value = dim.value_case() == dim.kDimValue;
    if (!has_dim_value) {
      LOGS_DEFAULT(INFO) << "Dynamic shape for " << arg_type_ << ": " << tensor_name_;
      shape.push_back(DNNL_RUNTIME_DIM_VAL);
    } else {
      shape.push_back(dim.dim_value());
    }
  }
  //make scaler as having dimension of 1
  if (shape.size() == 0) {
    shape.push_back(1);
  }
  auto dnnl_dims = dnnl::memory::dims(shape);
  return dnnl_dims;
}

dnnl::memory::data_type DnnlTensor::Type() const {
  if (arg_type_proto_ == nullptr) {
    ORT_THROW("Invoke DnnlTensor's arg_type_proto_ not initialized yet.");
  }
  auto data_type = arg_type_proto_->tensor_type().elem_type();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
      return dnnl::memory::data_type::undef;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return dnnl::memory::data_type::f16;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return dnnl::memory::data_type::bf16;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return dnnl::memory::data_type::f32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      // OneDNN does not have support for tensors of int64_t so we just say
      // the tensor is int32_t and then use casting in the actual operator
      // to convert the dnnl::memory::data_handle to an int64_t*.  Care
      // must be taken that an int64_t tensor does not make it pass the
      // node capability check unless the operator is explicitly expecting
      // the int64_t
      return dnnl::memory::data_type::s32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return dnnl::memory::data_type::s32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return dnnl::memory::data_type::s8;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return dnnl::memory::data_type::u8;
      // Same here, we use u8 as the handler for bool
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return dnnl::memory::data_type::u8;
    default:
      ORT_THROW("Unsupported data type: ", data_type);
  }
}

bool DnnlTensor::IsDynamic() {
  if (Dim().size() == 0) {
    return true;
  }
  for (auto dim : Dim()) {
    if (dim == DNNL_RUNTIME_DIM_VAL) {
      return true;
    }
  }
  return false;
}

bool DnnlTensor::Exists() {
  return !(tensor_name_ == "");
}

dnnl::memory::format_tag DnnlTensor::Format() {
  return dnnl::memory::format_tag::any;
}

void DnnlTensor::SetProducer(const DnnlNodeArg& arg) {
  producer_ = arg;
}

void DnnlTensor::ResetProducer() {
  producer_ = DnnlNodeArg();
}

void DnnlTensor::AddConsumer(const DnnlNodeArg& arg) {
  consumers_.push_back(arg);
}

void DnnlTensor::RemoveConsumer(const DnnlNodeArg& arg) {
  consumers_.erase(std::remove(consumers_.begin(), consumers_.end(), arg), consumers_.end());
}

DnnlNode::DnnlNode(const Node* node) {
  since_version_ = node->SinceVersion();
  name_ = node->Name();
  op_type_ = node->OpType();
  attr_->insert(node->GetAttributes());
}

std::string& DnnlNode::Name() {
  return name_;
}

std::string& DnnlNode::OpType() {
  return op_type_;
}

std::vector<DnnlTensor*>& DnnlNode::Inputs() {
  return inputs_;
}

std::vector<DnnlTensor*>& DnnlNode::Outputs() {
  return outputs_;
}

size_t& DnnlNode::Index() {
  return index_;
}

DnnlTensor& DnnlNode::Input(int index) {
  if (inputs_.size() <= (size_t)index) {
    return empty_tensor_;
  }
  if (inputs_[index] && inputs_[index]->Exists()) {
    return *inputs_[index];
  }
  return empty_tensor_;
}

size_t DnnlNode::InputCount() {
  return inputs_.size();
}

DnnlTensor& DnnlNode::Output(int index) {
  return *outputs_[index];
}

size_t DnnlNode::OutputCount() {
  return outputs_.size();
}

NodeAttributes& DnnlNode::Attributes() {
  return *attr_;
}

int DnnlNode::SinceVersion() {
  return since_version_;
}

void DnnlNode::AppendPostOp(std::string op) {
  postops_.push_back(op);
}

const std::vector<std::string>& DnnlNode::GetPostOps() {
  return postops_;
}

DnnlSubgraph::DnnlSubgraph(const GraphViewer& graph_viewer) {
  Build(graph_viewer);
  is_dynamic_ = false;
  for (auto input : GetDnnlInputs()) {
    if (input->IsDynamic()) {
      is_dynamic_ = true;
      break;
    }
  }
}

bool DnnlSubgraph::IsDynamic() {
  return is_dynamic_;
}

void DnnlSubgraph::TopoSort() {
  nodes_in_topological_order_.clear();

  std::unordered_map<size_t, int> indegrees;
  for (auto& node : dnnl_nodes_) {
    if (node.get()) {
      indegrees[node->Index()] = 0;
    }
  }

  for (auto& e : dnnl_tensors_) {
    auto tensor = e.second.get();
    if (tensor->Exists() && tensor->GetProducer().GetNode()) {
      for (auto edge : tensor->GetConsumers()) {
        if (edge.GetNode()) {
          indegrees[edge.GetNode()->Index()]++;
        }
      }
    }
  }

  std::queue<DnnlNode*> queue;
  for (auto e : indegrees) {
    if (e.second == 0) {
      queue.push(dnnl_nodes_[e.first].get());
    }
  }

  //need to make sure all indegrees are computed before doing bfs
  while (!queue.empty()) {
    auto cur = queue.front();
    queue.pop();
    nodes_in_topological_order_.push_back(cur->Index());
    for (auto output : cur->Outputs()) {
      if (output && output->Exists()) {
        for (auto edge : output->GetConsumers()) {
          indegrees[edge.GetNode()->Index()] -= 1;
          if (indegrees[edge.GetNode()->Index()] == 0) {
            queue.push(edge.GetNode());
          }
        }
      }
    }
  }
  assert(indegrees.size() == nodes_in_topological_order_.size());
}

std::vector<size_t> DnnlSubgraph::GetDnnlNodesInTopologicalOrder() {
  TopoSort();
  return nodes_in_topological_order_;
}

DnnlNode* DnnlSubgraph::GetDnnlNode(size_t node_index) {
  return dnnl_nodes_[node_index].get();
}

DnnlTensor* DnnlSubgraph::GetDnnlTensor(const std::string& tensor_name){
  if(dnnl_tensors_.count(tensor_name)){
    return dnnl_tensors_[tensor_name].get();
  }else{
    return nullptr;
  }
}


size_t DnnlSubgraph::GetMaxNodeIndex() {
  return dnnl_nodes_.size();
}

std::vector<DnnlNode*> DnnlSubgraph::GetDnnlNodes() {
  std::vector<DnnlNode*> result;
  for (auto& node : dnnl_nodes_) {
    if (node.get()) {
      result.push_back(node.get());
    }
  }
  return result;
}

std::vector<DnnlTensor*> DnnlSubgraph::GetDnnlInputs() {
  return inputs_;
}

std::vector<DnnlTensor*> DnnlSubgraph::GetDnnlOutputs() {
  return outputs_;
}

std::vector<DnnlTensor*> DnnlSubgraph::GetDnnlInitializers() {
  return initializers_;
}

void DnnlSubgraph::RemoveNode(size_t node_index) {
  dnnl_nodes_[node_index].reset(nullptr);
}

void DnnlSubgraph::RemoveTensor(const std::string& tensor_name) {
  inputs_.erase(std::remove_if(inputs_.begin(), inputs_.end(), [=](DnnlTensor* t){ return t->Name() == tensor_name;}), inputs_.end());
  initializers_.erase(std::remove_if(initializers_.begin(), initializers_.end(), [=](DnnlTensor* t){ return t->Name() == tensor_name;}), initializers_.end());
  dnnl_tensors_.erase(tensor_name);
}

void DnnlSubgraph::AddTensor(std::unique_ptr<DnnlTensor> new_tensor) {
  if (!dnnl_tensors_.count(new_tensor->Name())) {
    dnnl_tensors_.emplace(new_tensor->Name(), std::move(new_tensor));
  } else {
    ORT_THROW("tensor exists, modify or delete first before inseting");
  }
}

void DnnlSubgraph::AddNode(std::unique_ptr<DnnlNode> new_node) {
  auto index = dnnl_nodes_.size();
  dnnl_nodes_.emplace_back(std::move(new_node));
  dnnl_nodes_.back()->Index() = index;
}

void DnnlSubgraph::Build(const GraphViewer& graph_viewer) {
  //establish nodes, tensors and nodeargs
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer.GetNode(node_indices[i]));
    AddNode(std::make_unique<DnnlNode>(node));
    auto dnnl_node = dnnl_nodes_.back().get();
    std::vector<DnnlTensor*> inputs;
    size_t index = 0;
    for (auto input : node->InputDefs()) {
      if (input && input->Exists() && input->Name() != "") {
        if (!dnnl_tensors_.count(input->Name())) {
          dnnl_tensors_[input->Name()] = std::make_unique<DnnlTensor>(input);
        }
        dnnl_tensors_[input->Name()]->AddConsumer(DnnlNodeArg(dnnl_node, index, false));
        inputs.push_back(dnnl_tensors_[input->Name()].get());
      } else {
        inputs.push_back(nullptr);
      }
      index++;
    }
    std::vector<DnnlTensor*> outputs;
    index = 0;
    for (auto output : node->OutputDefs()) {
      if (output && output->Exists() && output->Name() != "") {
        if (!dnnl_tensors_.count(output->Name())) {
          dnnl_tensors_[output->Name()] = std::make_unique<DnnlTensor>(output);
        }
        dnnl_tensors_[output->Name()]->SetProducer(DnnlNodeArg(dnnl_node, index, true));
        outputs.push_back(dnnl_tensors_[output->Name()].get());
      } else {
        outputs.push_back(nullptr);
      }
      index++;
    }
    dnnl_node->Inputs() = inputs;
    dnnl_node->Outputs() = outputs;
  }

  //all tensors should have been established in graph
  //establish inputs, outputs and initializers
  //graph inputs including initializers and outputs can be deleted by graph transformation (eg, gelu fusion)
  //delete unneeded inputs don't affect onnxruntime passing them as input data handle
  //delete unneeded outputs will cause ep to output to fewer data handles then expected
  for (const auto* node_arg : graph_viewer.GetInputsIncludingInitializers()) {
    inputs_.push_back(dnnl_tensors_[node_arg->Name()].get());
  }

  for (const auto* node_arg : graph_viewer.GetOutputs()) {
    outputs_.push_back(dnnl_tensors_[node_arg->Name()].get());
  }

  for (auto& initializer : graph_viewer.GetAllInitializedTensors()) {
    auto& name = initializer.first;
    initializers_.push_back(dnnl_tensors_[name].get());
  }
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
