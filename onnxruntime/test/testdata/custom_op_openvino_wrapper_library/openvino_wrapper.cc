// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "openvino_wrapper.h"

#include <iostream>
#include <cassert>

static ov::element::Type ConvertONNXToOVType(ONNXTensorElementDataType onnx_type) {
  switch (onnx_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return ov::element::f32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return ov::element::u8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return ov::element::i8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return ov::element::u16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return ov::element::i16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return ov::element::i32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return ov::element::i64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return ov::element::boolean;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return ov::element::f16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return ov::element::f64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return ov::element::u32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return ov::element::u64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return ov::element::bf16;
    default:
      return ov::element::undefined;
  }
}

static bool AreShapesEqual(const std::vector<int64_t>& ort_shape, const ov::Shape& ov_shape) {
  if (ort_shape.size() != ov_shape.size()) {
    return false;
  }

  const size_t num_dims = ort_shape.size();

  for (size_t i = 0; i < num_dims; ++i) {
    if (static_cast<decltype(ov_shape[i])>(ort_shape[i]) != ov_shape[i]) {
      return false;
    }
  }

  return true;
}

static bool AreIOTypesEqual(Ort::ConstTypeInfo type_info, const ov::Output<ov::Node>& ov_node) {
  Ort::ConstTensorTypeAndShapeInfo type_shape_info = type_info.GetTensorTypeAndShapeInfo();

  // Check element type.
  const ov::element::Type ort_elem_type = ConvertONNXToOVType(type_shape_info.GetElementType());
  const ov::element::Type ov_elem_type = ov_node.get_element_type();
  if (ort_elem_type != ov_elem_type) {
    return false;
  }

  // Check shape.
  const std::vector<int64_t> ort_shape = type_shape_info.GetShape();
  const ov::Shape& ov_shape = ov_node.get_shape();
  if (!AreShapesEqual(ort_shape, ov_shape)) {
    return false;
  }

  return true;
}

static bool ValidateInputsAndOutputs(const Ort::ConstKernelInfo& kinfo, const ov::OutputVector& ov_inputs,
                                     const ov::OutputVector& ov_outputs) {
  const size_t num_inputs = kinfo.GetInputCount();
  const size_t num_outputs = kinfo.GetOutputCount();

  // Number of inputs and outputs must match.
  if (ov_inputs.size() != num_inputs || ov_outputs.size() != num_outputs) {
    return false;
  }

  // Check input names, shapes, and element types.
  for (size_t i = 0; i < num_inputs; ++i) {
    Ort::TypeInfo ort_type = kinfo.GetInputTypeInfo(i);
    const auto& ov_input = ov_inputs[i];

    if (kinfo.GetInputName(i) != ov_input.get_any_name()) {
      return false;
    }

    if (!AreIOTypesEqual(ort_type.GetConst(), ov_input)) {
      return false;
    }
  }

  // Check output names, shapes, and element types.
  for (size_t i = 0; i < num_outputs; ++i) {
    Ort::TypeInfo ort_type = kinfo.GetOutputTypeInfo(i);
    const auto& ov_output = ov_outputs[i];

    if (kinfo.GetOutputName(i) != ov_output.get_any_name()) {
      return false;
    }

    if (!AreIOTypesEqual(ort_type.GetConst(), ov_output)) {
      return false;
    }
  }

  return true;
}

/// <summary>
/// Converts an ORT tensor (as a Ort::Value) into an OpenVINO tensor that uses
/// the same underlying tensor data.
/// </summary>
static ov::Tensor OrtToOpenVINOTensor(Ort::UnownedValue ort_tensor) {
  Ort::TensorTypeAndShapeInfo type_shape_info = ort_tensor.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> ort_shape = type_shape_info.GetShape();

  ov::Shape ov_shape(ort_shape.begin(), ort_shape.end());  // Copy shape because ORT uses int64_t, not size_t.
  void* raw_data = ort_tensor.GetTensorMutableData<void>();
  auto elem_type = ConvertONNXToOVType(type_shape_info.GetElementType());

  return ov::Tensor(elem_type, ov_shape, raw_data);
}

KernelOpenVINO::KernelOpenVINO(const OrtApi& /* api*/, const OrtKernelInfo* info,
                               const std::unordered_map<std::string, std::string>& session_configs)
    : weights_(nullptr) {
  Ort::ConstKernelInfo kinfo(info);
  Ort::AllocatorWithDefaultOptions allocator;

  // Extract OpenVINO .bin and .xml contents from node attributes.
  this->weights_ = kinfo.GetTensorAttribute("BIN", allocator);  // Must keep the weights memory alive for inference.
  std::string xml_contents = kinfo.GetAttribute<std::string>("XML");

  // Create OpenVINO model.
  ov::Core core;
  const ov::Tensor weights_tensor = OrtToOpenVINOTensor(this->weights_.GetUnowned());
  std::shared_ptr<ov::Model> model = core.read_model(xml_contents, weights_tensor);

  // Validate input/output shapes and types.
  this->ov_inputs_ = model->inputs();
  this->ov_outputs_ = model->outputs();

  if (!ValidateInputsAndOutputs(kinfo, this->ov_inputs_, this->ov_outputs_)) {
    // A more detailed error message would be better.
    ORT_CXX_API_THROW("I/O names, shapes, or element types do not match OpenVINO model.", ORT_INVALID_GRAPH);
  }

  // Get OpenVINO device type from provider options.
  auto device_type_it = session_configs.find("device_type");

  if ((device_type_it == session_configs.end()) || device_type_it->second.empty()) {
    this->device_type_ = "CPU";
  } else {
    this->device_type_ = device_type_it->second;
  }

  // Compile OpenVINO model.
  this->compiled_model_ = core.compile_model(model, this->device_type_);
}

void KernelOpenVINO::Compute(OrtKernelContext* context) {
  Ort::KernelContext kcontext(context);

  // Create inference request.
  ov::InferRequest infer_req = this->compiled_model_.create_infer_request();

  const size_t num_inputs = kcontext.GetInputCount();
  assert(num_inputs == this->ov_inputs_.size());

  // Set input tensors.
  for (size_t i = 0; i < num_inputs; ++i) {
    Ort::ConstValue ort_val = kcontext.GetInput(i);
    const auto& input_info = this->ov_inputs_[i];

    const void* p_input_data = ort_val.GetTensorData<void>();

    // OpenVINO does not always observe const-correctness.
    ov::Tensor input_tensor(input_info.get_element_type(), input_info.get_shape(), const_cast<void*>(p_input_data));

    infer_req.set_input_tensor(i, input_tensor);
  }

  const size_t num_outputs = kcontext.GetOutputCount();
  assert(num_outputs == this->ov_outputs_.size());

  // Set output tensors that are backed by ORT memory.
  for (size_t i = 0; i < num_outputs; ++i) {
    const auto& output_info = this->ov_outputs_[i];
    const ov::Shape& ov_shape = output_info.get_shape();
    const ov::element::Type ov_elem_type = output_info.get_element_type();

    std::vector<int64_t> ort_shape(ov_shape.begin(), ov_shape.end());
    Ort::UnownedValue ort_val = kcontext.GetOutput(i, ort_shape);
    void* ort_memory = ort_val.GetTensorMutableData<void>();

    infer_req.set_output_tensor(i, ov::Tensor(ov_elem_type, ov_shape, ort_memory));
  }

  // Run inference.
  infer_req.infer();
}

//
// CustomOpOpenVINO
//
CustomOpOpenVINO::CustomOpOpenVINO(Ort::ConstSessionOptions session_options) {
  GetSessionConfigs(this->session_configs_, session_options);
}

void* CustomOpOpenVINO::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
  return new KernelOpenVINO(api, info, this->session_configs_);
}

std::vector<std::string> CustomOpOpenVINO::GetSessionConfigKeys() const { return {"device_type"}; }
