// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "include/model_info.hpp"
#include "onnx/onnx_pb.h"

OnnxModelInfo::OnnxModelInfo(const OrtApi* g_ort, const OrtSession* session, OrtAllocator* allocator) {
  g_ort->SessionGetInputCount(session, &num_in_tensors);
  g_ort->SessionGetOutputCount(session, &num_out_tensors);

  in_tensor_names.resize(num_in_tensors);
  in_tensor_dims.resize(num_in_tensors);
  in_tensor_element_types.resize(num_in_tensors);
  in_tensor_element_nums.resize(num_in_tensors);
  in_tensors.resize(num_in_tensors);
  for (size_t i = 0; i < num_in_tensors; i++) {
    // Get tensor name, tensor info
    g_ort->SessionGetInputName(session, i, allocator, &in_tensor_names[i]);

    OrtTypeInfo* type_info;
    const OrtTensorTypeAndShapeInfo* tensor_info;
    g_ort->SessionGetInputTypeInfo(session, i, &type_info);
    g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    g_ort->GetTensorElementType(tensor_info, &in_tensor_element_types[i]);

    // Get tensor shapes/dims/bytes
    size_t num_dims;
    g_ort->GetDimensionsCount(tensor_info, &num_dims);
    in_tensor_dims[i].resize(num_dims);
    g_ort->GetDimensions(tensor_info, in_tensor_dims[i].data(), num_dims);
    in_tensor_element_nums[i] = 1;
    for (size_t j = 0; j < in_tensor_dims[i].size(); ++j) {
      // If onnx model has dynamic dimension on tensors, e.g. [N, 3, 224, 224]
      // g_ort->GetDimensions yields [-1, 3, 224, 224]. We need to handle it with abs().
      in_tensor_dims[i][j] = abs(in_tensor_dims[i][j]);
      in_tensor_element_nums[i] *= in_tensor_dims[i][j];
    }

    if (type_info) g_ort->ReleaseTypeInfo(type_info);
  }
  out_tensor_names.resize(num_out_tensors);
  out_tensor_dims.resize(num_out_tensors);
  out_tensor_element_types.resize(num_out_tensors);
  out_tensor_element_nums.resize(num_out_tensors);
  out_tensors.resize(num_out_tensors);
  for (size_t i = 0; i < num_out_tensors; i++) {
    g_ort->SessionGetOutputName(session, i, allocator, &out_tensor_names[i]);
    OrtTypeInfo* type_info;
    const OrtTensorTypeAndShapeInfo* tensor_info;
    g_ort->SessionGetOutputTypeInfo(session, i, &type_info);
    g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    g_ort->GetTensorElementType(tensor_info, &out_tensor_element_types[i]);

    // Get tensor shapes/dims/bytes
    size_t num_dims;
    g_ort->GetDimensionsCount(tensor_info, &num_dims);
    out_tensor_dims[i].resize(num_dims);
    g_ort->GetDimensions(tensor_info, out_tensor_dims[i].data(), num_dims);
    out_tensor_element_nums[i] = 1;
    for (size_t j = 0; j < out_tensor_dims[i].size(); ++j) {
      // If onnx model has dynamic dimension on tensors, e.g. [N, 3, 224, 224]
      // g_ort->GetDimensions yields [-1, 3, 224, 224]. We need to handle it with abs().
      out_tensor_dims[i][j] = abs(out_tensor_dims[i][j]);
      out_tensor_element_nums[i] *= out_tensor_dims[i][j];
    }

    if (type_info) g_ort->ReleaseTypeInfo(type_info);
  }
}
size_t OnnxModelInfo::get_num_in_tensors() { return num_in_tensors; }
std::vector<char*> OnnxModelInfo::get_in_tensor_names() { return in_tensor_names; }
std::vector<std::vector<int64_t>> OnnxModelInfo::get_in_tensor_dims() { return in_tensor_dims; }
std::vector<ONNXTensorElementDataType> OnnxModelInfo::get_in_tensor_element_types() { return in_tensor_element_types; }
std::vector<int64_t> OnnxModelInfo::get_in_tensor_element_nums() { return in_tensor_element_nums; }
std::vector<OrtValue*>& OnnxModelInfo::get_in_tensors() { return in_tensors; }

size_t OnnxModelInfo::get_num_out_tensors() { return num_out_tensors; }
std::vector<char*> OnnxModelInfo::get_out_tensor_names() { return out_tensor_names; }
std::vector<std::vector<int64_t>> OnnxModelInfo::get_out_tensor_dims() { return out_tensor_dims; }
std::vector<ONNXTensorElementDataType> OnnxModelInfo::get_out_tensor_element_types() { return out_tensor_element_types; }
std::vector<int64_t> OnnxModelInfo::get_out_tensor_element_nums() { return out_tensor_element_nums; }
std::vector<OrtValue*>& OnnxModelInfo::get_out_tensors() { return out_tensors; }

void OnnxModelInfo::release_ort_values(const OrtApi* g_ort) {
  for (size_t i = 0; i < num_in_tensors; i++) {
    if (in_tensors[i]) {
      g_ort->ReleaseValue(in_tensors[i]);
      in_tensors[i] = nullptr;
    }
  }
  for (size_t i = 0; i < num_out_tensors; i++) {
    if (out_tensors[i]) {
      g_ort->ReleaseValue(out_tensors[i]);
      out_tensors[i] = nullptr;
    }
  }
}

void OnnxModelInfo::PrintOnnxModelInfo() {
  std::cout << "num_in_tensors: " << num_in_tensors << std::endl;
  std::cout << "num_out_tensors: " << num_out_tensors << std::endl;
  for (size_t i = 0; i < num_in_tensors; i++) {
    std::cout << "in_tensor_dims " << i << ": [";
    for (size_t j = 0; j < in_tensor_dims[i].size(); ++j) {
      std::cout << ' ' << in_tensor_dims[i][j];
    }
    std::cout << " ]" << std::endl;
    std::cout << "InTensorElementType " << in_tensor_element_types[i] << std::endl;
    std::cout << "InElementSize " << GetONNXTypeSize(in_tensor_element_types[i]) << std::endl;
    std::cout << "InTensorElementNums " << in_tensor_element_nums[i] << std::endl;
  }
  for (size_t i = 0; i < num_out_tensors; i++) {
    std::cout << "out_tensor_dims " << i << ": [";
    for (size_t j = 0; j < out_tensor_dims[i].size(); ++j) {
      std::cout << ' ' << out_tensor_dims[i][j];
    }
    std::cout << " ]" << std::endl;
    std::cout << "OutTensorElementType " << out_tensor_element_types[i] << std::endl;
    std::cout << "OutElementSize " << GetONNXTypeSize(out_tensor_element_types[i]) << std::endl;
    std::cout << "OutTensorElementNums " << out_tensor_element_nums[i] << std::endl;
  }
}

size_t GetONNXTypeSize(ONNXTensorElementDataType dtype) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return sizeof(double);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return sizeof(uint8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return sizeof(uint16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return sizeof(uint32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return sizeof(uint64_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return sizeof(int8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return sizeof(int16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return sizeof(int32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return sizeof(int64_t);
    default:
      throw std::runtime_error("Unsupported ONNX data type");
  }
}

int onnx_element_type_to_tensorproto_dtype(ONNXTensorElementDataType dtype) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BFLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FNUZ;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2FNUZ;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_STRING;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL;
    default:
      throw std::runtime_error("Unsupported ONNX data type");
  }
}