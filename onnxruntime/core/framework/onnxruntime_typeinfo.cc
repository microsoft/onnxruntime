// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// this file contains implementations of the C API

#include <cassert>
#include "onnxruntime_typeinfo.h"
#include "core/framework/tensor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/sparse_tensor.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"

#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/onnxruntime_map_type_info.h"
#include "core/framework/onnxruntime_sequence_type_info.h"
#include "core/framework/onnxruntime_optional_type_info.h"
#include "core/framework/TensorSeq.h"

using onnxruntime::BFloat16;
using onnxruntime::DataTypeImpl;
using onnxruntime::MLFloat16;
#if !defined(DISABLE_SPARSE_TENSORS)
using onnxruntime::SparseTensor;
#endif
using onnxruntime::Tensor;
using onnxruntime::TensorShape;

namespace on = ONNX_NAMESPACE;

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26409)
#endif

OrtTypeInfo::OrtTypeInfo(ONNXType type1) noexcept : type(type1) {
}

OrtTypeInfo::OrtTypeInfo(std::unique_ptr<OrtMapTypeInfo> map_type_info1) noexcept
  : type(ONNX_TYPE_MAP), map_type_info(std::move(map_type_info1)) {}

OrtTypeInfo::OrtTypeInfo(std::unique_ptr<OrtSequenceTypeInfo> sequence_type_info1) noexcept
  : type(ONNX_TYPE_SEQUENCE), sequence_type_info(std::move(sequence_type_info1)) {}

OrtTypeInfo::OrtTypeInfo(std::unique_ptr<OrtOptionalTypeInfo> optional_type_info1) noexcept
  : type(ONNX_TYPE_OPTIONAL), optional_type_info(std::move(optional_type_info1)) {}

OrtTypeInfo::OrtTypeInfo(ONNXType type1, std::unique_ptr<OrtTensorTypeAndShapeInfo> data1) noexcept
  : type(type1), data(std::move(data1)) {
}

OrtTypeInfo::~OrtTypeInfo() = default;

ORT_API_STATUS_IMPL(OrtApis::GetOnnxTypeFromTypeInfo, _In_ const struct OrtTypeInfo* input, _Out_ ONNXType* out) {
  API_IMPL_BEGIN
  *out = input->type;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CastTypeInfoToTensorInfo, _In_ const struct OrtTypeInfo* input,
  _Outptr_result_maybenull_ const struct OrtTensorTypeAndShapeInfo** out) {
  API_IMPL_BEGIN
  *out = (input->type == ONNX_TYPE_TENSOR || input->type == ONNX_TYPE_SPARSETENSOR) ? input->data.get() : nullptr;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CastTypeInfoToMapTypeInfo, _In_ const OrtTypeInfo* type_info,
  _Outptr_result_maybenull_ const OrtMapTypeInfo** out) {
  API_IMPL_BEGIN
  *out = type_info->type == ONNX_TYPE_MAP ? type_info->map_type_info.get() : nullptr;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CastTypeInfoToSequenceTypeInfo, _In_ const OrtTypeInfo* type_info,
  _Outptr_result_maybenull_ const OrtSequenceTypeInfo** out) {
  API_IMPL_BEGIN
  *out = type_info->type == ONNX_TYPE_SEQUENCE ? type_info->sequence_type_info.get() : nullptr;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CastTypeInfoToOptionalTypeInfo, _In_ const OrtTypeInfo* type_info,
  _Outptr_result_maybenull_ const OrtOptionalTypeInfo** out) {
  API_IMPL_BEGIN
  *out = (type_info->type != ONNX_TYPE_OPTIONAL) ? type_info->optional_type_info.get() : nullptr;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetDenotationFromTypeInfo, _In_ const OrtTypeInfo* type_info, _Out_ const char** const out,
                    _Out_ size_t* len) {
  API_IMPL_BEGIN
  *out = type_info->denotation.c_str();
  *len = type_info->denotation.size();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseTypeInfo, _Frees_ptr_opt_ OrtTypeInfo* ptr) {
  std::unique_ptr<OrtTypeInfo> p(ptr);
}

OrtTypeInfo::Ptr OrtTypeInfo::FromOrtValue(const OrtValue& value) {
  onnxruntime::MLDataType type = value.Type();
  if (type == nullptr) {
    return MakePtr(ONNX_TYPE_UNKNOWN);
  }

  // GetType<Tensor> and GetType<SparseTensor> do not have TypeProto populated because they return a static
  // TensorBase/SparseTensorBase instances, but other types are real MLDataTypes and they do have real protos
  // unless they are primitive data types, in which case we as before return them not implemented
  // however, this way we can support Opaque and we can avoid excessive calls to GetType()
  if (type->IsTensorType()) {
    const Tensor& tensor = value.Get<onnxruntime::Tensor>();
    const auto* tensor_data_type = tensor.DataType();
    if (tensor_data_type != nullptr) {
      auto type_shape = OrtTensorTypeAndShapeInfo::GetTensorShapeAndType(tensor.Shape(), *tensor_data_type);
      return MakePtr(ONNX_TYPE_TENSOR, std::move(type_shape));
    }
    return MakePtr(ONNX_TYPE_TENSOR);
  }

  if (type->IsSparseTensorType()) {
#if !defined(DISABLE_SPARSE_TENSORS)
    const SparseTensor& tensor = value.Get<onnxruntime::SparseTensor>();
    const auto* tensor_data_type = tensor.DataType();
    if (tensor_data_type != nullptr) {
      auto type_shape = OrtTensorTypeAndShapeInfo::GetTensorShapeAndType(tensor.DenseShape(), *tensor_data_type);
      return MakePtr(ONNX_TYPE_SPARSETENSOR, std::move(type_shape));
    }
    return MakePtr(ONNX_TYPE_SPARSETENSOR);
#else
    ORT_NOT_IMPLEMENTED("SparseTensor is not supported in this build.");
#endif
  }

  if (type->IsTensorSequenceType()) {
    const auto* tensor_data_type = value.Get<onnxruntime::TensorSeq>().DataType();
    if (tensor_data_type != nullptr) {
      TensorShape void_shape = {};
      auto type_shape = OrtTensorTypeAndShapeInfo::GetTensorShapeAndType(void_shape, *tensor_data_type);
      auto type_info = MakePtr(ONNX_TYPE_TENSOR, std::move(type_shape));
      auto sequence_type_info = std::make_unique<OrtSequenceTypeInfo>(std::move(type_info));
      return MakePtr(std::move(sequence_type_info));
    } else {
      ORT_THROW("OrtValue is TensorSequence type but has no element Tensor DataType.");
    }
  }

  const auto* type_proto = type->GetTypeProto();
  if (type_proto != nullptr) {
    // Place Opaque first as tensors will be mostly handled above and maps and sequences are not common
    switch (type_proto->value_case()) {
      case on::TypeProto::kOpaqueType: {
        return MakePtr(ONNX_TYPE_OPAQUE);
      } break;
      case on::TypeProto::kMapType: {
#if !defined(DISABLE_ML_OPS)
        auto map_type_info = OrtMapTypeInfo::FromTypeProto(*type_proto);
        return MakePtr(std::move(map_type_info));
      } break;
#else
        ORT_NOT_IMPLEMENTED("Map types are not supported in this build");
#endif
      case on::TypeProto::kSequenceType: {
        auto seq_info = OrtSequenceTypeInfo::FromTypeProto(*type_proto);
        return MakePtr(std::move(seq_info));
      } break;
      // Real Tensor support
#if !defined(DISABLE_SPARSE_TENSORS)
      case on::TypeProto::kSparseTensorType:
        [[fallthrough]];
#else
        ORT_NOT_IMPLEMENTED("SparseTensor types are not supported in this build");
#endif
      case on::TypeProto::kTensorType: {
        ORT_THROW("Tensor types should have been handled already");
      } break;
      default:
        // NOT_IMPLEMENTED
        break;
    }
  }
  ORT_NOT_IMPLEMENTED("This OrtValue is neither Tensor, SparseTensor, Map or Sequence type");
}

const DataTypeImpl* OrtTypeInfo::ElementTypeFromProto(int type) {
  auto tensor_type = DataTypeImpl::TensorTypeFromONNXEnum(type);
  return tensor_type->GetElementType();
}

OrtTypeInfo::Ptr OrtTypeInfo::FromTypeProto(const ONNX_NAMESPACE::TypeProto& input) {
  auto value_case = input.value_case();
  switch (value_case) {
    case on::TypeProto::kSparseTensorType:
#if !defined(DISABLE_SPARSE_TENSORS)
      [[fallthrough]];
#else
      ORT_NOT_IMPLEMENTED("SparseTensor types are not supported in this build");
#endif
    case on::TypeProto::kTensorType: {
      ONNXType ten_type = ONNX_TYPE_UNKNOWN;
      const on::TypeProto_Tensor* tensor_type = nullptr;
#if !defined(DISABLE_SPARSE_TENSORS)
      const on::TypeProto_SparseTensor* sparse_type = nullptr;
#endif
      const on::TensorShapeProto* sp = nullptr;
      if (value_case == on::TypeProto::kTensorType) {
        tensor_type = &input.tensor_type();
        ten_type = ONNX_TYPE_TENSOR;
        if (onnxruntime::utils::HasShape(*tensor_type)) {
          sp = &tensor_type->shape();
        }
      } else if (value_case == on::TypeProto::kSparseTensorType) {
#if !defined(DISABLE_SPARSE_TENSORS)
        sparse_type = &input.sparse_tensor_type();
        ten_type = ONNX_TYPE_SPARSETENSOR;
        if (onnxruntime::utils::HasShape(*sparse_type)) {
          sp = &sparse_type->shape();
        }
#endif
      }

      OrtTensorTypeAndShapeInfo::Ptr type_shape;
      if (sp != nullptr) {
        const on::TensorShapeProto& s = *sp;
        std::vector<int64_t> dims(s.dim_size());
        std::vector<std::string> dim_params(s.dim_size());
        TensorShape shape_data(std::move(dims));
        for (int i = 0, dim_size = s.dim_size(); i < dim_size; ++i) {
          auto& t = s.dim(i);
          switch (t.value_case()) {
            case on::TensorShapeProto::Dimension::kDimValue:
              shape_data[i] = t.dim_value();
              break;
            case on::TensorShapeProto::Dimension::kDimParam:
              dim_params[i] = t.dim_param();
              [[fallthrough]];
            case on::TensorShapeProto::Dimension::VALUE_NOT_SET:
              shape_data[i] = -1;
              break;
            default:
              assert(false);
          }
        }
        type_shape = OrtTensorTypeAndShapeInfo::GetTensorShapeAndType(std::move(shape_data), &dim_params, input);
      } else {
        type_shape = OrtTensorTypeAndShapeInfo::GetTensorShapeAndType(TensorShape(), nullptr, input);
      }

      auto type_info = MakePtr(ten_type, std::move(type_shape));
      type_info->denotation = input.denotation();
      return type_info;
    } break;
    case on::TypeProto::kSequenceType: {
      auto sequence_type_info = OrtSequenceTypeInfo::FromTypeProto(input);
      auto type_info = MakePtr(std::move(sequence_type_info));
      type_info->denotation = input.denotation();
      return type_info;
    } break;
#if !defined(DISABLE_ML_OPS)
    case on::TypeProto::kMapType: {
      auto map_type_info = OrtMapTypeInfo::FromTypeProto(input);
      auto type_info = MakePtr(std::move(map_type_info));
      type_info->denotation = input.denotation();
      return type_info;
    } break;
#endif
    case on::TypeProto::kOptionalType: {
      auto optional_type_info = OrtOptionalTypeInfo::FromTypeProto(input);
      auto type_info = MakePtr(std::move(optional_type_info));
      type_info->denotation = input.denotation();
      return type_info;
    } break;
    case on::TypeProto::kOpaqueType: {
      auto type_info = MakePtr(ONNX_TYPE_OPAQUE);
      type_info->denotation = input.denotation();
      return type_info;
    } break;
    case on::TypeProto::VALUE_NOT_SET:
      ORT_THROW("This TypeProto does not have ValueCase set");
      break;
    default:
      // Not implemented
      break;
  }
  ORT_NOT_IMPLEMENTED("The type is not tensor, sparse tensor, sequence, map or optional type");
}

OrtTypeInfo::Ptr OrtTypeInfo::Clone() const {
  switch (type) {
    case ONNX_TYPE_SPARSETENSOR:
#if !defined(DISABLE_SPARSE_TENSORS)
      [[fallthrough]];
#else
      ORT_NOT_IMPLEMENTED("SparseTensor is not supported in this build.");
#endif
    case ONNX_TYPE_TENSOR: {
      OrtTensorTypeAndShapeInfo::Ptr info;
      if (data) {
        info = data->Clone();
      }
      auto type_info = MakePtr(type, std::move(info));
      type_info->denotation = denotation;
      return type_info;
    }

    case ONNX_TYPE_SEQUENCE: {
      auto seq_clone = sequence_type_info->Clone();
      auto type_info = MakePtr(std::move(seq_clone));
      type_info->denotation = denotation;
      return type_info;
    }
    case ONNX_TYPE_MAP: {
      auto map_clone = map_type_info->Clone();
      auto type_info = MakePtr(std::move(map_clone));
      type_info->denotation = denotation;
      return type_info;
    }
    case ONNX_TYPE_OPTIONAL: {
      auto opt_clone = optional_type_info->Clone();
      auto type_info = MakePtr(std::move(opt_clone));
      type_info->denotation = denotation;
      return type_info;
    }
    case ONNX_TYPE_OPAQUE: {
      auto type_info = MakePtr(type);
      type_info->denotation = denotation;
      return type_info;
    }
    default:
      // Not implemented
      break;
  }
  ORT_NOT_IMPLEMENTED("The type is not tensor, sparse tensor, sequence, map or optional type");
}
