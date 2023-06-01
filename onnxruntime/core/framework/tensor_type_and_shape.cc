// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor_type_and_shape.h"

#include <assert.h>
#include <atomic>
#include <stdexcept>

#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/ort_value.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/tensor_shape.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"

using onnxruntime::DataTypeImpl;
#if !defined(DISABLE_SPARSE_TENSORS)
using onnxruntime::SparseTensor;
#endif
using onnxruntime::narrow;
using onnxruntime::Tensor;

OrtTensorTypeAndShapeInfo::OrtTensorTypeAndShapeInfo() = default;
OrtTensorTypeAndShapeInfo::~OrtTensorTypeAndShapeInfo() = default;
OrtTensorTypeAndShapeInfo::OrtTensorTypeAndShapeInfo(const OrtTensorTypeAndShapeInfo& other) = default;
OrtTensorTypeAndShapeInfo& OrtTensorTypeAndShapeInfo::operator=(const OrtTensorTypeAndShapeInfo& other) = default;

ORT_API_STATUS_IMPL(OrtApis::CreateTensorTypeAndShapeInfo, _Outptr_ OrtTensorTypeAndShapeInfo** out) {
  API_IMPL_BEGIN
  *out = std::make_unique<OrtTensorTypeAndShapeInfo>().release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseTensorTypeAndShapeInfo, _Frees_ptr_opt_ OrtTensorTypeAndShapeInfo* ptr) {
  std::unique_ptr<OrtTensorTypeAndShapeInfo> p(ptr);
}

ORT_API_STATUS_IMPL(OrtApis::SetTensorElementType, _Inout_ OrtTensorTypeAndShapeInfo* this_ptr,
                    enum ONNXTensorElementDataType type) {
  API_IMPL_BEGIN
  this_ptr->type = type;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SetDimensions, OrtTensorTypeAndShapeInfo* this_ptr,
                    _In_ const int64_t* dim_values, size_t dim_count) {
  API_IMPL_BEGIN
  this_ptr->shape = onnxruntime::TensorShape(dim_values, dim_count);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorElementType, _In_ const struct OrtTensorTypeAndShapeInfo* info,
                    _Out_ ONNXTensorElementDataType* out) {
  *out = info->type;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetDimensionsCount, _In_ const struct OrtTensorTypeAndShapeInfo* info,
                    _Out_ size_t* out) {
  *out = info->shape.NumDimensions();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetDimensions, _In_ const struct OrtTensorTypeAndShapeInfo* info,
                    _Out_ int64_t* dim_values, size_t dim_values_length) {
  info->shape.CopyDims(dim_values, dim_values_length);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetSymbolicDimensions,
                    _In_ const struct OrtTensorTypeAndShapeInfo* info,
                    _Out_writes_all_(dim_params_length) const char** names, size_t dim_params_length) {
  for (size_t idx = 0, end = std::min(info->dim_params.size(), dim_params_length); idx < end; ++idx) {
    names[idx] = info->dim_params[idx].c_str();
  }

  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorShapeElementCount,
                    _In_ const OrtTensorTypeAndShapeInfo* this_ptr, _Out_ size_t* out) {
  API_IMPL_BEGIN
  *out = SafeInt<size_t>{this_ptr->shape.Size()};
  return nullptr;
  API_IMPL_END
}

struct OrtValue;

constexpr ONNXTensorElementDataType TensorDataTypeToOnnxRuntimeTensorElementDataType(
    int32_t dtype) {
  namespace o = ONNX_NAMESPACE;
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  switch (dtype) {
    case o::TensorProto_DataType_FLOAT:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      break;
    case o::TensorProto_DataType_DOUBLE:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
      break;
    case o::TensorProto_DataType_FLOAT8E4M3FN:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN;
      break;
    case o::TensorProto_DataType_FLOAT8E4M3FNUZ:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ;
      break;
    case o::TensorProto_DataType_FLOAT8E5M2:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2;
      break;
    case o::TensorProto_DataType_FLOAT8E5M2FNUZ:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ;
      break;
    case o::TensorProto_DataType_FLOAT16:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
      break;
    case o::TensorProto_DataType_BFLOAT16:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
      break;
    case o::TensorProto_DataType_INT8:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
      break;
    case o::TensorProto_DataType_UINT8:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      break;
    case o::TensorProto_DataType_INT16:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
      break;
    case o::TensorProto_DataType_UINT16:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
      break;
    case o::TensorProto_DataType_INT32:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      break;
    case o::TensorProto_DataType_UINT32:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
      break;
    case o::TensorProto_DataType_INT64:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      break;
    case o::TensorProto_DataType_UINT64:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
      break;
    case o::TensorProto_DataType_STRING:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
      break;
    case o::TensorProto_DataType_BOOL:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
      break;
    default:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      break;
  }
  return type;
}

ONNXTensorElementDataType MLDataTypeToOnnxRuntimeTensorElementDataType(
    const onnxruntime::DataTypeImpl* cpp_type) {
  auto prim_type = cpp_type->AsPrimitiveDataType();
  if (prim_type == nullptr) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  return TensorDataTypeToOnnxRuntimeTensorElementDataType(prim_type->GetDataType());
}

std::unique_ptr<OrtTensorTypeAndShapeInfo> OrtTensorTypeAndShapeInfo::GetTensorShapeAndTypeHelper(
    ONNXTensorElementDataType type,
    onnxruntime::TensorShape shape,
    const std::vector<std::string>* dim_params) {
  auto type_and_shape = std::make_unique<OrtTensorTypeAndShapeInfo>();
  type_and_shape->type = type;
  type_and_shape->shape = std::move(shape);

  if (dim_params != nullptr) {
    type_and_shape->dim_params = *dim_params;
  } else {
    // we expect to be being called with a concrete shape so validate that
    assert(type_and_shape->shape.Size() >= 0);
    type_and_shape->dim_params.resize(type_and_shape->shape.NumDimensions(), "");
  }

  return type_and_shape;
}

std::unique_ptr<OrtTensorTypeAndShapeInfo> OrtTensorTypeAndShapeInfo::GetTensorShapeAndType(
    onnxruntime::TensorShape shape,
    const onnxruntime::DataTypeImpl& tensor_data_type) {
  ONNXTensorElementDataType type = MLDataTypeToOnnxRuntimeTensorElementDataType(&tensor_data_type);
  if (ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED == type) {
    ORT_NOT_IMPLEMENTED("Tensor type is undefined");
  }
  return GetTensorShapeAndTypeHelper(type, std::move(shape), nullptr);
}

std::unique_ptr<OrtTensorTypeAndShapeInfo> OrtTensorTypeAndShapeInfo::GetTensorShapeAndType(
    onnxruntime::TensorShape shape,
    const std::vector<std::string>* dim_params,
    const ONNX_NAMESPACE::TypeProto& type_proto) {
  auto value_case = type_proto.value_case();
  assert(value_case == ONNX_NAMESPACE::TypeProto::kTensorType ||
         value_case == ONNX_NAMESPACE::TypeProto::kSparseTensorType);

  auto dtype = (value_case == ONNX_NAMESPACE::TypeProto::kTensorType) ? type_proto.tensor_type().elem_type()
                                                                      : type_proto.sparse_tensor_type().elem_type();
  ONNXTensorElementDataType type = TensorDataTypeToOnnxRuntimeTensorElementDataType(dtype);
  if (ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED == type) {
    ORT_NOT_IMPLEMENTED("Tensor type is undefined");
  }
  return GetTensorShapeAndTypeHelper(type, std::move(shape), dim_params);
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorTypeAndShape,
                    _In_ const OrtValue* v, _Outptr_ OrtTensorTypeAndShapeInfo** out) {
  API_IMPL_BEGIN
  if (!v->IsAllocated()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "the ort_value must contain a constructed tensor or sparse tensor");
  }
  if (v->IsTensor() || v->IsSparseTensor()) {
    const onnxruntime::TensorShape* shape = nullptr;
    onnxruntime::MLDataType data_type = nullptr;
    if (v->IsTensor()) {
      const Tensor& tensor = v->Get<onnxruntime::Tensor>();
      shape = &tensor.Shape();
      data_type = tensor.DataType();
      auto ptr = OrtTensorTypeAndShapeInfo::GetTensorShapeAndType(*shape, *data_type);
      *out = ptr.release();
    } else {
#if !defined(DISABLE_SPARSE_TENSORS)
      const SparseTensor& tensor = v->Get<onnxruntime::SparseTensor>();
      shape = &tensor.DenseShape();
      data_type = tensor.DataType();
      auto ptr = OrtTensorTypeAndShapeInfo::GetTensorShapeAndType(*shape, *data_type);
      *out = ptr.release();
#else
      ORT_NOT_IMPLEMENTED("SparseTensor is not supported in this build.");
#endif
    }
  } else {
    ORT_THROW("Argument is not a tensor");
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetSparseTensorValuesTypeAndShape, _In_ const OrtValue* v,
                    _Outptr_ OrtTensorTypeAndShapeInfo** out) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  const auto& sparse_tensor = SparseTensor::GetSparseTensorFromOrtValue(*v);
  const auto& values = sparse_tensor.Values();
  auto ptr = OrtTensorTypeAndShapeInfo::GetTensorShapeAndType(values.Shape(), *values.DataType());
  *out = ptr.release();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(v);
  ORT_UNUSED_PARAMETER(out);
  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

#if !defined(DISABLE_SPARSE_TENSORS)
namespace {
const Tensor& GetIndicesTensor(const OrtValue& v, OrtSparseIndicesFormat indices_format) {
  const auto& sparse_tensor = SparseTensor::GetSparseTensorFromOrtValue(v);
  const Tensor* indices_tensor = nullptr;
  switch (indices_format) {
    case OrtSparseIndicesFormat::ORT_SPARSE_COO_INDICES:
      indices_tensor = &sparse_tensor.AsCoo().Indices();
      break;
    case OrtSparseIndicesFormat::ORT_SPARSE_CSR_INNER_INDICES:
      indices_tensor = &sparse_tensor.AsCsr().Inner();
      break;
    case OrtSparseIndicesFormat::ORT_SPARSE_CSR_OUTER_INDICES:
      indices_tensor = &sparse_tensor.AsCsr().Outer();
      break;
    case OrtSparseIndicesFormat::ORT_SPARSE_BLOCK_SPARSE_INDICES:
      indices_tensor = &sparse_tensor.AsBlockSparse().Indices();
      break;
    default:
      ORT_THROW(ORT_INVALID_ARGUMENT, "Unsupported indices_format passed");
  }
  return *indices_tensor;
}
}  // namespace
#endif  // !defined(DISABLE_SPARSE_TENSORS)

ORT_API_STATUS_IMPL(OrtApis::GetSparseTensorIndicesTypeShape, _In_ const OrtValue* v,
                    OrtSparseIndicesFormat indices_format, _Outptr_ OrtTensorTypeAndShapeInfo** out) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  const Tensor& indices_tensor = GetIndicesTensor(*v, indices_format);
  auto ptr = OrtTensorTypeAndShapeInfo::GetTensorShapeAndType(indices_tensor.Shape(), *indices_tensor.DataType());
  *out = ptr.release();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(v);
  ORT_UNUSED_PARAMETER(indices_format);
  ORT_UNUSED_PARAMETER(out);
  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetSparseTensorIndices, _In_ const OrtValue* v,
                    enum OrtSparseIndicesFormat indices_format, _Out_ size_t* num_indices, _Outptr_ const void** indices) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  const Tensor& indices_tensor = GetIndicesTensor(*v, indices_format);
  *num_indices = narrow<size_t>(indices_tensor.Shape().Size());
  *indices = indices_tensor.DataRaw();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(v);
  ORT_UNUSED_PARAMETER(indices_format);
  ORT_UNUSED_PARAMETER(num_indices);
  ORT_UNUSED_PARAMETER(indices);
  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetValueType, _In_ const OrtValue* v, _Out_ ONNXType* out) {
  API_IMPL_BEGIN
  auto type_info = OrtTypeInfo::FromOrtValue(*v);
  *out = type_info->type;
  return nullptr;
  API_IMPL_END
}

/**
 * Get the type information of an OrtValue
 * \param value
 * \return The returned value should be freed by OrtReleaseTypeInfo after use
 */
ORT_API_STATUS_IMPL(OrtApis::GetTypeInfo,
                    _In_ const OrtValue* v, _Outptr_result_maybenull_ struct OrtTypeInfo** out) {
  API_IMPL_BEGIN
  // TODO: This is consistent with the previous implementation but inconsistent with GetValueType which returns
  // ONNX_TYPE_UNKNOWN if v->Type() is null. Should we instead just call OrtTypeInfo::FromOrtValue and
  // return an OrtTypeInfo value in 'out' with type set to ONNX_TYPE_UNKNOWN? Or is the inconsistency fine?
  if (v->Type() == nullptr) {
    *out = nullptr;
    return nullptr;
  }
  auto ptr = OrtTypeInfo::FromOrtValue(*v);
  *out = ptr.release();
  return nullptr;
  API_IMPL_END
}