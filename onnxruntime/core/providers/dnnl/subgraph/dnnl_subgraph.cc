// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_subgraph.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlTensor::DnnlTensor(const NodeArg* arg) {
  arg_ = arg;
  tensor_name_ = arg->Name();
}

DnnlTensor::DnnlTensor(std::string name) {
  tensor_name_ = name;
}

std::string DnnlTensor::Name() {
  return tensor_name_;
}

dnnl::memory::dims DnnlTensor::Dim() {
  auto shape_proto = arg_->Shape();
  // a shape without any information
  if (shape_proto == nullptr) {
    LOGS_DEFAULT(INFO) << "nullptr shape for " << arg_->Type() << ": " << arg_->Name();
    return dnnl::memory::dims();
  }
  std::vector<int64_t> shape;
  const auto& dims = shape_proto->dim();
  for (const auto& dim : dims) {
    bool has_dim_value = dim.value_case() == dim.kDimValue;
    if (!has_dim_value) {
      LOGS_DEFAULT(INFO) << "Dynamic shape for " << arg_->Type() << ": " << arg_->Name();
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

dnnl::memory::data_type DnnlTensor::Type() {
  auto data_type = arg_->TypeAsProto()->tensor_type().elem_type();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
      return dnnl::memory::data_type::undef;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return dnnl::memory::data_type::f16;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return dnnl::memory::data_type::bf16;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return dnnl::memory::data_type::f32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return dnnl::memory::data_type::s32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return dnnl::memory::data_type::s8;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
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

DnnlNode::DnnlNode(const Node* node) {
  onnx_node_ = node;
}

std::string DnnlNode::Name() {
  return onnx_node_->Name();
}

std::string DnnlNode::OpType() {
  return onnx_node_->OpType();
}

DnnlTensor DnnlNode::Input(int index) {
  if (onnx_node_->InputDefs().size() <= (size_t)index) {
    return DnnlTensor("");
  }
  if (!onnx_node_->InputDefs()[index]) {
    return DnnlTensor("");
  }
  if (onnx_node_->InputDefs()[index]->Exists()) {
    auto def = onnx_node_->InputDefs()[index];
    return DnnlTensor(def);
  }
  return DnnlTensor("");
}

size_t DnnlNode::InputCount() {
  return onnx_node_->InputDefs().size();
}

DnnlTensor DnnlNode::Output(int index) {
  auto def = onnx_node_->OutputDefs()[index];
  return DnnlTensor(def);
}

size_t DnnlNode::OutputCount() {
  return onnx_node_->OutputDefs().size();
}

const NodeAttributes& DnnlNode::Attributes() {
  return onnx_node_->GetAttributes();
}

DnnlSubgraph::DnnlSubgraph(const GraphViewer& graph_viewer) : graph_viewer_(graph_viewer) {
  Build();
  is_dynamic_ = false;
  for (auto& input : GetDnnlInputs()) {
    if (input.IsDynamic()) {
      is_dynamic_ = true;
      break;
    }
  }
}

bool DnnlSubgraph::IsDynamic() {
  return is_dynamic_;
}

std::vector<DnnlNode> DnnlSubgraph::GetDnnlNodes() {
  return dnnl_nodes_;
}

std::vector<DnnlTensor> DnnlSubgraph::GetDnnlInputs() {
  return inputs_;
}

std::vector<DnnlTensor> DnnlSubgraph::GetDnnlOutputs() {
  return outputs_;
}

std::vector<DnnlTensor> DnnlSubgraph::GetDnnlInitializers() {
  return initializers_;
}

void DnnlSubgraph::Build() {
  for (const auto* node_arg : graph_viewer_.GetInputsIncludingInitializers()) {
    inputs_.push_back(DnnlTensor(node_arg));
  }
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_.GetNode(node_indices[i]));
    dnnl_nodes_.push_back(DnnlNode(node));
  }
  for (const auto* node_arg : graph_viewer_.GetOutputs()) {
    outputs_.push_back(DnnlTensor(node_arg));
  }
  for (auto& initializer : graph_viewer_.GetAllInitializedTensors()) {
    auto& name = initializer.first;
    initializers_.push_back(DnnlTensor(name));
  }
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
