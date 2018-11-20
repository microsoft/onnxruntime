// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/tensor_type_and_shape_c_api.h"
#include "core/framework/onnx_object.h"
#include "core/framework/tensor_shape.h"
#include <assert.h>
#include <stdexcept>
#include <atomic>

struct ONNXRuntimeTensorTypeAndShapeInfo {
 public:
  const ONNXObject* const cls;
  std::atomic_int ref_count;
  OnnxRuntimeTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  onnxruntime::TensorShape shape;

  static ONNXRuntimeTensorTypeAndShapeInfo* Create() {
    return new ONNXRuntimeTensorTypeAndShapeInfo();
  }
  static uint32_t ONNXRUNTIME_API_STATUSCALL ReleaseImpl(void* this_) {
    ONNXRuntimeTensorTypeAndShapeInfo* this_ptr = static_cast<ONNXRuntimeTensorTypeAndShapeInfo*>(this_);
    if (--this_ptr->ref_count == 0)
      delete this_ptr;
    return 0;
  }

  static uint32_t ONNXRUNTIME_API_STATUSCALL AddRefImpl(void* this_) {
    ONNXRuntimeTensorTypeAndShapeInfo* this_ptr = static_cast<ONNXRuntimeTensorTypeAndShapeInfo*>(this_);
    ++this_ptr->ref_count;
    return 0;
  }

 private:
  ONNXRuntimeTensorTypeAndShapeInfo();
  ~ONNXRuntimeTensorTypeAndShapeInfo() {
    assert(ref_count == 0);
  }
  ONNXRuntimeTensorTypeAndShapeInfo(const ONNXRuntimeTensorTypeAndShapeInfo& other) = delete;
  ONNXRuntimeTensorTypeAndShapeInfo& operator=(const ONNXRuntimeTensorTypeAndShapeInfo& other) = delete;
};

constexpr ONNXObject shape_cls = {
    ONNXRuntimeTensorTypeAndShapeInfo::AddRefImpl,
    ONNXRuntimeTensorTypeAndShapeInfo::ReleaseImpl,
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

ONNXRuntimeTensorTypeAndShapeInfo::ONNXRuntimeTensorTypeAndShapeInfo() : cls(&shape_cls), ref_count(1) {
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
