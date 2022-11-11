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

static bool AreIONodesEqual(Ort::ConstKernelIOInfo ort_node, const ov::Output<ov::Node>& ov_node) {
  // Check name
  const std::string ort_name = ort_node.GetName();
  const std::string ov_name = ov_node.get_any_name();
  if (ort_name != ov_name) {
    return false;
  }

  Ort::ConstTypeInfo type_info = ort_node.GetTypeInfo();
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
    const Ort::KernelIOInfo ort_input = kinfo.GetInputInfo(i);
    const auto& ov_input = ov_inputs[i];

    if (!AreIONodesEqual(ort_input.GetConst(), ov_input)) {
      return false;
    }
  }

  // Check output names, shapes, and element types.
  for (size_t i = 0; i < num_outputs; ++i) {
    const Ort::KernelIOInfo ort_output = kinfo.GetOutputInfo(i);
    const auto& ov_output = ov_outputs[i];

    if (!AreIONodesEqual(ort_output.GetConst(), ov_output)) {
      return false;
    }
  }

  return true;
}

KernelOpenVINO::KernelOpenVINO(const OrtApi& api, const OrtKernelInfo* info,
                               const std::unordered_map<std::string, std::string>& session_configs) : ort_(api) {
  Ort::ConstKernelInfo kinfo(info);

  // Extract OpenVINO .bin and .xml contents from node attributes.
  this->weights_ = kinfo.GetAttribute<std::string>("BIN");
  std::string xml_contents = kinfo.GetAttribute<std::string>("XML");

  // Create OpenVINO model.
  ov::Core core;
  const ov::Shape shape{this->weights_.size()};
  const ov::Tensor weights_tensor(ov::element::u8, shape, weights_.data());
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
  this->device_type_ = device_type_it != session_configs.end() ? device_type_it->second : "CPU";

  // Compile OpenVINO model.
  this->compiled_model_ = core.compile_model(model, this->device_type_);
}

void KernelOpenVINO::Compute(OrtKernelContext* context) {
  Ort::KernelContext kcontext(context);

  const size_t num_inputs = kcontext.GetInputCount();
  assert(num_inputs == this->ov_inputs_.size());

  ov::TensorVector ov_inputs(num_inputs);

  // Gather OpenVINO model inputs.
  for (size_t i = 0; i < num_inputs; ++i) {
    Ort::ConstValue ort_val = kcontext.GetInput(i);
    const auto& input_info = this->ov_inputs_[i];

    const void* p_input_data = ort_val.GetTensorData<void>();
    ov_inputs[i] = ov::Tensor(input_info.get_element_type(), input_info.get_shape(), const_cast<void*>(p_input_data));
  }

  // Inference.
  ov::InferRequest infer_req = this->compiled_model_.create_infer_request();

  infer_req.set_input_tensors(ov_inputs);
  infer_req.infer();

  const size_t num_outputs = kcontext.GetOutputCount();
  assert(num_outputs == this->ov_outputs_.size());

  // Copy inference results to ORT memory.
  for (size_t i = 0; i < num_outputs; ++i) {
    const auto& output_info = this->ov_outputs_[i];

    // Get pointer to output data (src) from OpenVINO inference.
    ov::element::Type elem_type = output_info.get_element_type();
    const void* src = infer_req.get_output_tensor(i).data(elem_type);

    // Get dst to which to copy result.
    const ov::Shape& ov_shape = output_info.get_shape();
    std::vector<int64_t> shape(ov_shape.begin(), ov_shape.end());
    Ort::UnownedValue ort_val = kcontext.GetOutput(i, shape);
    void* dst = ort_val.GetTensorMutableData<void>();

    // Copy data.
    size_t copy_size = elem_type.size() * ov::shape_size(ov_shape);
    std::memcpy(dst, src, copy_size);
  }
}

//
// CustomOpOpenVINO
//

CustomOpOpenVINO::CustomOpOpenVINO(Ort::ConstSessionOptions session_options) : session_options_(session_options) {
}

void* CustomOpOpenVINO::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
  std::unordered_map<std::string, std::string> session_configs;
  GetSessionConfigs(session_configs, this->session_options_);
  return new KernelOpenVINO(api, info, session_configs);
}

const char* CustomOpOpenVINO::GetName() const { return "OpenVINO_Wrapper"; }

size_t CustomOpOpenVINO::GetInputTypeCount() const { return 1; }

ONNXTensorElementDataType CustomOpOpenVINO::GetInputType(size_t /* index */) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

OrtCustomOpInputOutputCharacteristic CustomOpOpenVINO::GetInputCharacteristic(size_t /* index */) const {
  return INPUT_OUTPUT_VARIADIC;
}

size_t CustomOpOpenVINO::GetOutputTypeCount() const { return 1; }

ONNXTensorElementDataType CustomOpOpenVINO::GetOutputType(size_t /* index */) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

OrtCustomOpInputOutputCharacteristic CustomOpOpenVINO::GetOutputCharacteristic(size_t /* index */) const {
  return INPUT_OUTPUT_VARIADIC;
}

const char* CustomOpOpenVINO::GetExecutionProviderType() const { return nullptr; }

std::vector<std::string> CustomOpOpenVINO::GetSessionConfigKeys() const { return {"device_type"}; }