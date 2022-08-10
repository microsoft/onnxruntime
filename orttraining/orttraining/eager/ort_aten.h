// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ort_eager_common.h"
#include <core/framework/ort_value.h>

#include "ort_util.h"
#include "ort_ops.h"
#include "ort_log.h"
#include "ort_tensor.h"

#define CHECK_STATUS(status) if (!status.IsOK()) { \
  std::stringstream err; \
  err << "ORT return failure (line " << __LINE__ << "): " << status.ErrorMessage(); \
  throw std::runtime_error(err.str()); }

namespace torch_ort {
namespace eager {

at::Tensor aten_tensor_from_ort(
  OrtValue&& ot,
  const at::TensorOptions& options);

const std::vector<at::Tensor> aten_tensor_from_ort(
  std::vector<OrtValue>& ortvalues,
  const at::TensorOptions& options);

onnxruntime::MLDataType ort_scalar_type_from_aten(
  at::ScalarType dtype);

OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  const at::Scalar& scalar);

OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  const at::Scalar& scalar,
  at::ScalarType type);

OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  const at::Tensor& tensor);

std::vector<OrtValue> create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  at::TensorList tensors);

OrtValue create_ort_value(const at::Tensor& tensor);

// Create 1-dimensional ORT tensor from a given value
template <typename T>
OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  const T val) {
  OrtValue ort_val;
  onnxruntime::Tensor::InitOrtValue(onnxruntime::DataTypeImpl::GetType<T>(), onnxruntime::TensorShape({1}),
                                    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), ort_val);
  auto* ort_tensor = ort_val.GetMutable<onnxruntime::Tensor>();
  CopyVectorToTensor<T>(invoker, &val, 1, *ort_tensor);
  return ort_val;
}

template <typename T>
OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  c10::optional<T> val) {
  return create_ort_value(invoker, val.value());
}

template<typename T>
OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  const std::vector<T> values) {
  OrtValue ort_value;
  onnxruntime::Tensor::InitOrtValue(
      onnxruntime::DataTypeImpl::GetType<T>(), onnxruntime::TensorShape({(int64_t)values.size()}),
      invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), ort_value);
  CopyVectorToTensor<T>(
    invoker,
    values.data(),
    values.size(),
    *ort_value.GetMutable<onnxruntime::Tensor>());
  return ort_value;
}

template<typename T>
OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  const at::ArrayRef<T> values) {
  std::vector<T> values_vector;
  values_vector.assign(values.begin(), values.end());
  return create_ort_value(invoker, values_vector);
}

onnx::AttributeProto create_ort_attribute(
  const char* name,
  at::Scalar value,
  const bool isTensor=false);

onnx::AttributeProto create_ort_attribute(
  const char* name,
  at::Scalar value,
  at::ScalarType type);

onnx::AttributeProto create_ort_attribute(
  const char* name,
  const char* value);

bool IsSupportedType(at::Scalar scalar, const std::vector<at::ScalarType>& valid_types);

bool IsSupportedType(at::Tensor tensor, const std::vector<at::ScalarType>& valid_types);

bool IsSupportedType(at::IntArrayRef arrary, const std::vector<at::ScalarType>& valid_types);

bool IsSupportedType(int64_t val, const std::vector<at::ScalarType>& valid_types);

bool IsSupportedType(c10::optional<int64_t> val, const std::vector<at::ScalarType>& valid_types);

bool IsSupportedType(at::TensorList tensors, const std::vector<at::ScalarType>& valid_types);

c10::optional<at::ScalarType> PromoteScalarTypesWithCategory(
    const std::vector<at::ScalarType>& typesFromTensors,
    const std::vector<at::ScalarType>& typesFromScalars);

ONNX_NAMESPACE::TensorProto_DataType GetONNXTensorProtoDataType(at::ScalarType dtype);

OrtValue CastToType(onnxruntime::ORTInvoker& invoker, const OrtValue& input, at::ScalarType type);
void CastToType_out(onnxruntime::ORTInvoker& invoker, const OrtValue& input, OrtValue& output, at::ScalarType type);

void resize_output(
  onnxruntime::ORTInvoker& invoker,
  ORTTensorImpl* output,
  at::IntArrayRef shape);

void resize_impl_ort_(
  onnxruntime::ORTInvoker& invoker,
  ORTTensorImpl* self,
  at::IntArrayRef size);

namespace aten {

// aten::nonzero(Tensor self) -> Tensor
at::Tensor nonzero(
  const at::Tensor& self);

} // namespace aten
} // namespace eager
} // namespace torch_ort
