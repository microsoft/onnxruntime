// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef BUILD_MS_EXPERIMENTAL_OPS

#include "core/providers/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/signal/utils.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/Map.h"
#include "window_functions.h"
#include <functional>

#include "core/platform/threadpool.h"

#include <complex>
#include <cmath>

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    HannWindow,
    kMSExperimentalDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)  //
        .TypeConstraint("T1", BuildKernelDefConstraints<int32_t, int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>()),
    HannWindow);

ONNX_OPERATOR_KERNEL_EX(
    HammingWindow,
    kMSExperimentalDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)  //
        .TypeConstraint("T1", BuildKernelDefConstraints<int32_t, int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>()),
    HammingWindow);

ONNX_OPERATOR_KERNEL_EX(
    BlackmanWindow,
    kMSExperimentalDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)  //
        .TypeConstraint("T1", BuildKernelDefConstraints<int32_t, int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>()),
    BlackmanWindow);

ONNX_OPERATOR_KERNEL_EX(
    MelWeightMatrix,
    kMSExperimentalDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)  //
        .TypeConstraint("T1", BuildKernelDefConstraints<int32_t, int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float>())
        .TypeConstraint("T3", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>()),
    MelWeightMatrix);

template <typename T>
static Status cosine_sum_window(Tensor* Y, size_t size, float a0, float a1, float a2) {
  auto* Y_data = reinterpret_cast<T*>(Y->MutableDataRaw());

  // Calculate the radians to increment per sample
  constexpr double pi = 3.14159265;
  constexpr double tau = 2 * pi;
  const double angular_increment = tau / size;

  for (size_t i = 0; i < size; i++) {
    auto a2_component = a2 == 0 ? 0 : (a2 * cos(2 * angular_increment * i));

    T& value = *(Y_data + i);
    value = static_cast<T>(a0 - (a1 * cos(angular_increment * i)) + a2_component);
  }

  return Status::OK();
}

static Status create_cosine_sum_window(
    OpKernelContext* ctx,
    onnx::TensorProto_DataType output_datatype,
    float a0, float a1, float a2) {
  // Get the size of the window
  auto size = ::onnxruntime::signal::get_scalar_value_from_tensor<int64_t>(ctx->Input<Tensor>(0));

  // Get the output tensor
  auto Y_shape = onnxruntime::TensorShape({size});
  auto Y = ctx->Output(0, Y_shape);

  switch (output_datatype) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      ORT_RETURN_IF_ERROR((cosine_sum_window<float>(Y, size, a0, a1, a2)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
      ORT_RETURN_IF_ERROR((cosine_sum_window<double>(Y, size, a0, a1, a2)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      ORT_RETURN_IF_ERROR((cosine_sum_window<int8_t>(Y, size, a0, a1, a2)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
      ORT_RETURN_IF_ERROR((cosine_sum_window<int16_t>(Y, size, a0, a1, a2)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      ORT_RETURN_IF_ERROR((cosine_sum_window<int32_t>(Y, size, a0, a1, a2)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      ORT_RETURN_IF_ERROR((cosine_sum_window<int64_t>(Y, size, a0, a1, a2)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      ORT_RETURN_IF_ERROR((cosine_sum_window<uint8_t>(Y, size, a0, a1, a2)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
      ORT_RETURN_IF_ERROR((cosine_sum_window<uint16_t>(Y, size, a0, a1, a2)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32: {
      ORT_RETURN_IF_ERROR((cosine_sum_window<uint32_t>(Y, size, a0, a1, a2)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
      ORT_RETURN_IF_ERROR((cosine_sum_window<uint64_t>(Y, size, a0, a1, a2)));
      break;
    }
    default:
      ORT_THROW("Unsupported input data type of ", output_datatype);
  }

  return Status::OK();
}

Status HannWindow::Compute(OpKernelContext* ctx) const {
  // HannWindows are a special case of Cosine-Sum Windows which take the following form:
  // w[n] = SUM_k=0_K( (-1)^k * a_k * cos(2*pi*k*n/N) ) with values the following values for a_k:
  float a0 = .5f;
  float a1 = a0;
  float a2 = 0;
  return create_cosine_sum_window(ctx, data_type_, a0, a1, a2);
}

Status HammingWindow::Compute(OpKernelContext* ctx) const {
  // HammingWindows are a special case of Cosine-Sum Windows which take the following form:
  // w[n] = SUM_k=0_K( (-1)^k * a_k * cos(2*pi*k*n/N) ) with values the following values for a_k:
  float a0 = 25.f / 46.f;
  float a1 = 1 - a0;
  float a2 = 0;
  return create_cosine_sum_window(ctx, data_type_, a0, a1, a2);
}

Status BlackmanWindow::Compute(OpKernelContext* ctx) const {
  // BlackmanWindows are a special case of Cosine-Sum Windows which take the following form:
  // w[n] = SUM_k=0_K( (-1)^k * a_k * cos(2*pi*k*n/N) ) with values the following values for a_k:
  float alpha = .16f;
  float a2 = alpha / 2.f;
  float a0 = .5f - a2;
  float a1 = .5f;
  return create_cosine_sum_window(ctx, data_type_, a0, a1, a2);
}

}  // namespace contrib
}  // namespace onnxruntime

#endif
