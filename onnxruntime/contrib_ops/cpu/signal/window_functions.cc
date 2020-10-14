// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/Map.h"
#include "window_functions.h"
#include <functional>

#include "core/platform/threadpool.h"

#include <complex>

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    HannWindow,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)
        .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>()),
    HannWindow);

ONNX_OPERATOR_KERNEL_EX(
    HammingWindow,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)
        .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>()),
    HammingWindow);

ONNX_OPERATOR_KERNEL_EX(
    BlackmanWindow,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)
        .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>()),
    BlackmanWindow);


ONNX_OPERATOR_KERNEL_EX(
    MelWeightMatrix,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)
        .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>())
        .TypeConstraint("T3", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>()),
    MelWeightMatrix);


template <typename T>
static Status cosine_sum_window(Tensor* Y, size_t size, float a0, float a1, float a2) {
  auto* Y_data = reinterpret_cast<T*>(Y->MutableDataRaw());

  static const double pi = 3.14159265;
  static const double tau = 2 * pi;
  const double angular_increment = tau / size;

  for (size_t i = 0; i < size; i++) {
    T& value = *(Y_data + i);
    if (a2 == 0) {
      value = static_cast<T>(a0 - (a1 * cos(angular_increment * i)));
    } else {
      value = static_cast<T>(a0 - (a1 * cos(angular_increment * i)) + (a2 * cos(2 * angular_increment * i)));
    }
  }

  return Status::OK();
}

static Status create_cosine_sum_window(
    OpKernelContext* ctx,
    onnx::TensorProto_DataType output_datatype,
    float a0, float a1, float a2) {

  const auto* size_tensor = ctx->Input<Tensor>(0);
  ORT_ENFORCE(size_tensor->Shape().Size() == 1, "ratio input should have a single value.");

  auto input_data_type = size_tensor->DataType()->AsPrimitiveDataType()->GetDataType();

  size_t size;
  switch (input_data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      size = *reinterpret_cast<const int8_t*>(size_tensor->DataRaw());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      size = *reinterpret_cast<const int16_t*>(size_tensor->DataRaw());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      size = *reinterpret_cast<const int32_t*>(size_tensor->DataRaw());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      size = *reinterpret_cast<const int64_t*>(size_tensor->DataRaw());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      size = *reinterpret_cast<const uint8_t*>(size_tensor->DataRaw());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      size = *reinterpret_cast<const uint16_t*>(size_tensor->DataRaw());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      size = *reinterpret_cast<const uint32_t*>(size_tensor->DataRaw());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      size = *reinterpret_cast<const uint64_t*>(size_tensor->DataRaw());
      break;
    default:
      ORT_THROW("Unsupported input data type of ", input_data_type);
  }

  Status status;

  onnxruntime::TensorShape Y_shape({static_cast<int64_t>(size)});
  auto* Y = ctx->Output(0, Y_shape);

  switch (output_datatype) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      status = cosine_sum_window<float>(Y, size, a0, a1, a2);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      status = cosine_sum_window<double>(Y, size, a0, a1, a2);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      status = cosine_sum_window<int16_t>(Y, size, a0, a1, a2);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      status = cosine_sum_window<int32_t>(Y, size, a0, a1, a2);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      status = cosine_sum_window<int64_t>(Y, size, a0, a1, a2);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      status = cosine_sum_window<uint8_t>(Y, size, a0, a1, a2);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      status = cosine_sum_window<uint16_t>(Y, size, a0, a1, a2);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      status = cosine_sum_window<uint32_t>(Y, size, a0, a1, a2);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      status = cosine_sum_window<uint64_t>(Y, size, a0, a1, a2);
      break;
    default:
      ORT_THROW("Unsupported input data type of ", output_datatype);
  }

  return status;
}

Status HannWindow::Compute(OpKernelContext* ctx) const {
  float a0 = .5f;
  float a1 = a0;
  float a2 = 0;
  return create_cosine_sum_window(ctx, data_type_, a0, a1, a2);
}

Status HammingWindow::Compute(OpKernelContext* ctx) const {
  float a0 = 25.f / 46.f;
  float a1 = 1 - a0;
  float a2 = 0;
  return create_cosine_sum_window(ctx, data_type_, a0, a1, a2);
}

Status BlackmanWindow::Compute(OpKernelContext* ctx) const {
  float alpha = .16f;
  float a2 = alpha / 2.f;
  float a0 = .5f - a2;
  float a1 = .5f;
  return create_cosine_sum_window(ctx, data_type_, a0, a1, a2);
}

Status MelWeightMatrix::Compute(OpKernelContext* ctx) const {
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
