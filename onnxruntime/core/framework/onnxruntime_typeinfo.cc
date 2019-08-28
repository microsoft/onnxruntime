// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//this file contains implementations of the C API

#include <cassert>
#include "onnxruntime_typeinfo.h"
#include "core/framework/tensor.h"
#include "core/framework/sparse_tensor.h"
#include "core/graph/onnx_protobuf.h"

using onnxruntime::BFloat16;
using onnxruntime::DataTypeImpl;
using onnxruntime::MLFloat16;
using onnxruntime::Tensor;
using onnxruntime::SparseTensor;
using onnxruntime::TensorShape;

OrtTypeInfo::OrtTypeInfo(ONNXType type1, OrtTensorTypeAndShapeInfo* data1) noexcept : type(type1), data(data1) {
}

OrtTypeInfo::~OrtTypeInfo() {
  OrtReleaseTensorTypeAndShapeInfo(data);
}

ORT_API_STATUS_IMPL(OrtGetOnnxTypeFromTypeInfo, _In_ const struct OrtTypeInfo* input, ONNXType* out) {
  *out = input->type;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtCastTypeInfoToTensorInfo, _In_ const struct OrtTypeInfo* input, const struct OrtTensorTypeAndShapeInfo** out) {
  *out = input->type == ONNX_TYPE_TENSOR ? input->data : nullptr;
  return nullptr;
}

ORT_API(void, OrtReleaseTypeInfo, _Frees_ptr_opt_ OrtTypeInfo* ptr) {
  delete ptr;
}

OrtStatus* GetTensorShapeAndType(const TensorShape* shape, const onnxruntime::DataTypeImpl* tensor_data_type, OrtTensorTypeAndShapeInfo** out);
OrtStatus* GetTensorShapeAndType(const TensorShape* shape, const ONNX_NAMESPACE::TypeProto* type_proto, OrtTensorTypeAndShapeInfo** out);

OrtStatus* OrtTypeInfo::FromDataTypeImpl(const onnxruntime::DataTypeImpl* input, const TensorShape* shape, const onnxruntime::DataTypeImpl* tensor_data_type, OrtTypeInfo** out) {
  if (input == nullptr) {
    *out = new OrtTypeInfo(ONNX_TYPE_UNKNOWN, nullptr);
    return nullptr;
  }
  // GetType<Tensor> and GetType<SparseTensor> do not have TypeProto populated because they return a static
  // TensorBase/SparseTensorBase instances, but other types are real MLDataTypes and they do have real protos
  // unless they are primitive data types, in which case we as before return them not implemented
  // however, this way we can support Opaque and we can avoid excessive calls to GetType()
  if (input == DataTypeImpl::GetType<Tensor>()) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (tensor_data_type != nullptr) {
      OrtStatus* st = GetTensorShapeAndType(shape, tensor_data_type, &info);
      if (st != nullptr) return st;
    }
    *out = new OrtTypeInfo(ONNX_TYPE_TENSOR, info);
    return nullptr;
  }
  if (input == DataTypeImpl::GetType<SparseTensor>()) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (tensor_data_type != nullptr) {
      OrtStatus* st = GetTensorShapeAndType(shape, tensor_data_type, &info);
      if (st != nullptr) return st;
    }
    *out = new OrtTypeInfo(ONNX_TYPE_SPARSETENSOR, info);
    return nullptr;
  }
  const auto* type_proto = input->GetTypeProto();
  if (type_proto != nullptr) {
    // Place Opaque first as tensors will be
    // mostly handled above and maps and sequences
    // are not common
    if (type_proto->has_opaque_type()) {
      *out = new OrtTypeInfo(ONNX_TYPE_OPAQUE, nullptr);
      return nullptr;
    }
    if (type_proto->has_map_type()) {
      *out = new OrtTypeInfo(ONNX_TYPE_MAP, nullptr);
      return nullptr;
    }
    if (type_proto->has_sequence_type()) {
      *out = new OrtTypeInfo(ONNX_TYPE_SEQUENCE, nullptr);
      return nullptr;
    }
    // Add support for real Tensor Types at the end
    if (type_proto->has_tensor_type() || type_proto->has_sparse_tensor_type()) {
      OrtTensorTypeAndShapeInfo* info = nullptr;
      OrtStatus* st = GetTensorShapeAndType(shape, type_proto, &info);
      if (st != nullptr) return st;
      if (type_proto->has_tensor_type()) {
        *out = new OrtTypeInfo(ONNX_TYPE_TENSOR, info);
      } else {
        *out = new OrtTypeInfo(ONNX_TYPE_SPARSETENSOR, nullptr);
      }
      return nullptr;
    }
  }
  return OrtCreateStatus(ORT_NOT_IMPLEMENTED, "not implemented");
}

const DataTypeImpl* ElementTypeFromProto(int type) {
  switch (type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return DataTypeImpl::GetType<float>();
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return DataTypeImpl::GetType<bool>();
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return DataTypeImpl::GetType<int>();
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return DataTypeImpl::GetType<double>();
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      return DataTypeImpl::GetType<std::string>();
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return DataTypeImpl::GetType<int8_t>();
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return DataTypeImpl::GetType<uint8_t>();
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      return DataTypeImpl::GetType<uint16_t>();
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return DataTypeImpl::GetType<int16_t>();
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return DataTypeImpl::GetType<int64_t>();
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      return DataTypeImpl::GetType<uint32_t>();
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      return DataTypeImpl::GetType<uint64_t>();
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return DataTypeImpl::GetType<MLFloat16>();
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return DataTypeImpl::GetType<BFloat16>();

    default:
      ORT_NOT_IMPLEMENTED(__FUNCTION__, ":tensor type ", type, " is not supported");
  }
}

OrtStatus* OrtTypeInfo::FromDataTypeImpl(const ONNX_NAMESPACE::TypeProto* input, OrtTypeInfo** out) {
  if (input->has_tensor_type() || input->has_sparse_tensor_type()) {

    const ONNX_NAMESPACE::TypeProto_Tensor* tensor_type = nullptr;
    const ONNX_NAMESPACE::TypeProto_SparseTensor* sparse_type = nullptr;
    if (input->has_tensor_type()) { 
      tensor_type = &input->tensor_type(); 
    } else if (input->has_sparse_tensor_type()) {
      sparse_type = &input->sparse_tensor_type();
    }
    assert(tensor_type != nullptr || sparse_type != nullptr);

    OrtStatus* st;
    OrtTensorTypeAndShapeInfo* info = nullptr;

    const ::ONNX_NAMESPACE::TensorShapeProto* sp = nullptr;
    if (tensor_type != nullptr && tensor_type->has_shape()) {
      sp = &tensor_type->shape();
    } else if (sparse_type != nullptr && sparse_type->has_shape()) {
      sp = &sparse_type->shape();
    }

    if (sp != nullptr) {
      const ::ONNX_NAMESPACE::TensorShapeProto& s = *sp;
      std::vector<int64_t> dims(s.dim_size());
      TensorShape shape_data(std::move(dims));
      for (int i = 0; i != s.dim_size(); ++i) {
        auto& t = s.dim(i);
        shape_data[i] = t.has_dim_value() ? t.dim_value() : -1;
      }
      st = GetTensorShapeAndType(&shape_data, input, &info);
    } else {
      st = GetTensorShapeAndType(nullptr, input, &info);
    }

    if (st != nullptr) return st;
    if (tensor_type != nullptr) {
      *out = new OrtTypeInfo(ONNX_TYPE_TENSOR, info);
    } else {
      *out = new OrtTypeInfo(ONNX_TYPE_SPARSETENSOR, info);
    }
    return nullptr;
  }
  if (input->has_sequence_type()) {
    *out = new OrtTypeInfo(ONNX_TYPE_SEQUENCE, nullptr);
    return nullptr;
  }
  if (input->has_map_type()) {
    *out = new OrtTypeInfo(ONNX_TYPE_MAP, nullptr);
    return nullptr;
  }
  if (input->has_opaque_type()) {
    *out = new OrtTypeInfo(ONNX_TYPE_OPAQUE, nullptr);
    return nullptr;
  }
  return OrtCreateStatus(ORT_NOT_IMPLEMENTED, "not implemented");
}
