// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor_type_and_shape_c_api.h"
#include "core/framework/onnx_object.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/ml_value.h"
#include "core/framework/onnxruntime_typeinfo.h"

#include <assert.h>
#include <stdexcept>
#include <atomic>

using onnxruntime::DataTypeImpl;
using onnxruntime::MLFloat16;
using onnxruntime::Tensor;

struct ONNXRuntimeTensorTypeAndShapeInfo : public onnxruntime::ObjectBase<ONNXRuntimeTensorTypeAndShapeInfo> {
 public:
  friend class onnxruntime::ObjectBase<ONNXRuntimeTensorTypeAndShapeInfo>;

  OnnxRuntimeTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  onnxruntime::TensorShape shape;

  static ONNXRuntimeTensorTypeAndShapeInfo* Create() {
    return new ONNXRuntimeTensorTypeAndShapeInfo();
  }

  ONNXRuntimeTensorTypeAndShapeInfo(const ONNXRuntimeTensorTypeAndShapeInfo& other) = delete;
  ONNXRuntimeTensorTypeAndShapeInfo& operator=(const ONNXRuntimeTensorTypeAndShapeInfo& other) = delete;

 private:
  ONNXRuntimeTensorTypeAndShapeInfo() = default;
  ~ONNXRuntimeTensorTypeAndShapeInfo() {
    assert(ref_count == 0);
  }
};

#define API_IMPL_BEGIN try {
#define API_IMPL_END                                                   \
  }                                                                    \
  catch (std::exception & ex) {                                        \
    return CreateONNXStatus(ONNXRUNTIME_RUNTIME_EXCEPTION, ex.what()); \
  }

ONNXRUNTIME_API(ONNXRuntimeTensorTypeAndShapeInfo*, ONNXRuntimeCreateTensorTypeAndShapeInfo) {
  return ONNXRuntimeTensorTypeAndShapeInfo::Create();
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeSetTensorElementType, _In_ ONNXRuntimeTensorTypeAndShapeInfo* this_ptr, enum OnnxRuntimeTensorElementDataType type) {
  API_IMPL_BEGIN
  this_ptr->type = type;
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeSetDims, _In_ ONNXRuntimeTensorTypeAndShapeInfo* this_ptr, _In_ const int64_t* dim_values, size_t dim_count) {
  API_IMPL_BEGIN
  this_ptr->shape = onnxruntime::TensorShape(dim_values, dim_count);
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API(enum OnnxRuntimeTensorElementDataType, ONNXRuntimeGetTensorElementType, _In_ const struct ONNXRuntimeTensorTypeAndShapeInfo* info) {
  return info->type;
}

ONNXRUNTIME_API(size_t, ONNXRuntimeGetNumOfDimensions, _In_ const struct ONNXRuntimeTensorTypeAndShapeInfo* info) {
  return info->shape.NumDimensions();
}

ONNXRUNTIME_API(void, ONNXRuntimeGetDimensions, _In_ const struct ONNXRuntimeTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length) {
  info->shape.CopyDims(dim_values, dim_values_length);
}

ONNXRUNTIME_API(int64_t, ONNXRuntimeGetTensorShapeElementCount, _In_ const ONNXRuntimeTensorTypeAndShapeInfo* this_ptr) {
  return this_ptr->shape.Size();
}

struct ONNXValue;

namespace {
inline OnnxRuntimeTensorElementDataType MLDataTypeToOnnxRuntimeTensorElementDataType(
    const onnxruntime::DataTypeImpl* cpp_type) {
  OnnxRuntimeTensorElementDataType type;
  if (cpp_type == onnxruntime::DataTypeImpl::GetType<float>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint8_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int8_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint16_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int16_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int32_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int64_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<std::string>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<bool>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<MLFloat16>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<double>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint32_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint64_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
  } else {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  return type;
}
}  // namespace

ONNXStatusPtr GetTensorShapeAndType(const onnxruntime::TensorShape* shape, const onnxruntime::DataTypeImpl* tensor_data_type, ONNXRuntimeTensorTypeAndShapeInfo** out) {
  OnnxRuntimeTensorElementDataType type = MLDataTypeToOnnxRuntimeTensorElementDataType(tensor_data_type);
  if (ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED == type) {
    return CreateONNXStatus(ONNXRUNTIME_FAIL, "Not implemented");
  }
  ONNXRuntimeTensorTypeAndShapeInfo* ret = ONNXRuntimeCreateTensorTypeAndShapeInfo();
  auto status = ONNXRuntimeSetTensorElementType(ret, type);
  if (status != nullptr) {
    ONNXRuntimeReleaseObject(ret);
    return status;
  }
  if (shape != nullptr) {
    status = ONNXRuntimeSetDims(ret, shape->GetDims().data(), shape->GetDims().size());
    if (status != nullptr) {
      ONNXRuntimeReleaseObject(ret);
      return status;
    }
  }
  *out = ret;
  return nullptr;
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeGetTensorShapeAndType, _In_ const ONNXValue* value,
                            _Out_ ONNXRuntimeTensorTypeAndShapeInfo** out) {
  API_IMPL_BEGIN
  auto v = reinterpret_cast<const ::onnxruntime::MLValue*>(value);
  const onnxruntime::Tensor& tensor = v->Get<onnxruntime::Tensor>();
  return GetTensorShapeAndType(&tensor.Shape(), tensor.DataType(), out);
  API_IMPL_END
}

ONNXRUNTIME_API(enum ONNXRuntimeType, ONNXRuntimeGetValueType, _In_ const ONNXValue* value) {
  try {
    auto v = reinterpret_cast<const ::onnxruntime::MLValue*>(value);
    onnxruntime::MLDataType type = v->Type();
    ONNXRuntimeTypeInfo* out;
    ONNXStatusPtr ptr = ONNXRuntimeTypeInfo::FromDataTypeImpl(type, nullptr, nullptr, &out);
    if (ptr != nullptr) {
      ReleaseONNXStatus(ptr);
      return ONNXRUNTIME_TYPE_UNKNOWN;
    }
    ONNXRuntimeType ret = out->type;
    ONNXRuntimeReleaseObject(out);
    return ret;
  } catch (std::exception&) {
    return ONNXRUNTIME_TYPE_UNKNOWN;
  }
}

/**
 * Get the type information of an ONNXValue
 * \param value
 * \return The returned value should be freed by ONNXRuntimeReleaseObject after use
 */
ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeGetTypeInfo, _In_ const ONNXValue* value, struct ONNXRuntimeTypeInfo** out) {
  auto v = reinterpret_cast<const ::onnxruntime::MLValue*>(value);
  onnxruntime::MLDataType type = v->Type();
  if (type == nullptr) {
    *out = nullptr;
    return nullptr;
  }
  if (type == DataTypeImpl::GetType<Tensor>()) {
    const onnxruntime::Tensor& tensor = v->Get<onnxruntime::Tensor>();
    const onnxruntime::TensorShape& shape = tensor.Shape();
    return ONNXRuntimeTypeInfo::FromDataTypeImpl(type, &shape, tensor.DataType(), out);
  }
  return ONNXRuntimeTypeInfo::FromDataTypeImpl(type, nullptr, nullptr, out);
}