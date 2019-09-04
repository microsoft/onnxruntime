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

namespace on = ONNX_NAMESPACE;

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
  if (input->IsTensorType()) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (tensor_data_type != nullptr) {
      OrtStatus* st = GetTensorShapeAndType(shape, tensor_data_type, &info);
      if (st != nullptr) return st;
    }
    *out = new OrtTypeInfo(ONNX_TYPE_TENSOR, info);
    return nullptr;
  }
  if (input->IsSparseTensorType()) {
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
    switch (type_proto->value_case ()) {
      case on::TypeProto::kOpaqueType: {
        *out = new OrtTypeInfo(ONNX_TYPE_OPAQUE, nullptr);
        return nullptr;
      } break;
      case on::TypeProto::kMapType: {
        *out = new OrtTypeInfo(ONNX_TYPE_MAP, nullptr);
        return nullptr;
      } break;
      case on::TypeProto::kSequenceType: {
        *out = new OrtTypeInfo(ONNX_TYPE_SEQUENCE, nullptr);
        return nullptr;
      } break;
      // Real Tensor support
      case on::TypeProto::kTensorType:
      case on::TypeProto::kSparseTensorType: {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        OrtStatus* st = GetTensorShapeAndType(shape, type_proto, &info);
        if (st != nullptr) return st;
        if (type_proto->value_case() == on::TypeProto::kTensorType) {
          *out = new OrtTypeInfo(ONNX_TYPE_TENSOR, info);
        } else {
          *out = new OrtTypeInfo(ONNX_TYPE_SPARSETENSOR, nullptr);
        }
        return nullptr;
      } break;
      default:
        // NOT_IMPLEMENTED
        break;
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
  auto value_case = input->value_case();
  switch (value_case) {
    case on::TypeProto::kTensorType:
    case on::TypeProto::kSparseTensorType: {
      ONNXType ten_type = ONNX_TYPE_UNKNOWN;
      const on::TypeProto_Tensor* tensor_type = nullptr;
      const on::TypeProto_SparseTensor* sparse_type = nullptr;
      const on::TensorShapeProto* sp = nullptr;
      if (value_case == on::TypeProto::kTensorType) {
        tensor_type = &input->tensor_type();
        ten_type = ONNX_TYPE_TENSOR;
        if (&tensor_type->shape() != &on::TensorShapeProto::default_instance()) {
          sp = &tensor_type->shape();
        }
      } else if (value_case == on::TypeProto::kSparseTensorType) {
        sparse_type = &input->sparse_tensor_type();
        ten_type = ONNX_TYPE_SPARSETENSOR;
        if (&sparse_type->shape() != &on::TensorShapeProto::default_instance()) {
          sp = &sparse_type->shape();
        }
      }
      OrtStatus* st = nullptr;
      OrtTensorTypeAndShapeInfo* info = nullptr;
      if (sp != nullptr) {
        const on::TensorShapeProto& s = *sp;
        std::vector<int64_t> dims(s.dim_size());
        TensorShape shape_data(std::move(dims));
        for (int i = 0; i < s.dim_size(); ++i) {
          auto& t = s.dim(i);
          switch (t.value_case ()) {
            case on::TensorShapeProto::Dimension::kDimValue:
              shape_data[i] = t.dim_value();
              break;
            case on::TensorShapeProto::Dimension::kDimParam:
              shape_data[i] = -1;
              break;
            default:
              assert(false);
          }
        }
        st = GetTensorShapeAndType(&shape_data, input, &info);
      } else {
        st = GetTensorShapeAndType(nullptr, input, &info);
      }
      if (st != nullptr) return st;
      *out = new OrtTypeInfo(ten_type, info);
      return nullptr;
    } break;
    case on::TypeProto::kSequenceType: {
      *out = new OrtTypeInfo(ONNX_TYPE_SEQUENCE, nullptr);
      return nullptr;
    } break;
    case on::TypeProto::kMapType: {
      *out = new OrtTypeInfo(ONNX_TYPE_MAP, nullptr);
      return nullptr;
    } break;
    case on::TypeProto::kOpaqueType: {
      *out = new OrtTypeInfo(ONNX_TYPE_OPAQUE, nullptr);
      return nullptr;
    } break;
    default:
      assert(false);
      break;
  }
  return OrtCreateStatus(ORT_NOT_IMPLEMENTED, "not implemented");
}
