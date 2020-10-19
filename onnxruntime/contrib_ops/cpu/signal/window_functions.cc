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
        .TypeConstraint("T1", BuildKernelDefConstraints<int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>()),
    HannWindow);

ONNX_OPERATOR_KERNEL_EX(
    HammingWindow,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)
        .TypeConstraint("T1", BuildKernelDefConstraints<int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>()),
    HammingWindow);

ONNX_OPERATOR_KERNEL_EX(
    BlackmanWindow,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)
        .TypeConstraint("T1", BuildKernelDefConstraints<int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>()),
    BlackmanWindow);


ONNX_OPERATOR_KERNEL_EX(
    MelWeightMatrix,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)
        .TypeConstraint("T1", BuildKernelDefConstraints<int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float>())
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

template <typename T>
static T get_scalar_value_from_tensor(const Tensor* t) {
  ORT_ENFORCE(t->Shape().Size() == 1, "ratio input should have a single value.");

  T value;

  auto data_type = t->DataType()->AsPrimitiveDataType()->GetDataType();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      value = static_cast<T>(*reinterpret_cast<const float*>(t->DataRaw()));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      value = static_cast<T>(*reinterpret_cast<const double*>(t->DataRaw()));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      value = static_cast<T>(*reinterpret_cast<const int32_t*>(t->DataRaw()));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      value = static_cast<T>(*reinterpret_cast<const int64_t*>(t->DataRaw()));
      break;
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }

  return value;
}

static Status create_cosine_sum_window(
    OpKernelContext* ctx,
    onnx::TensorProto_DataType output_datatype,
    float a0, float a1, float a2) {
  
  auto size = get_scalar_value_from_tensor<int64_t>(ctx->Input<Tensor>(0));
  onnxruntime::TensorShape Y_shape({size});
  auto* Y = ctx->Output(0, Y_shape);

  switch (output_datatype) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return cosine_sum_window<float>(Y, size, a0, a1, a2);
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return cosine_sum_window<double>(Y, size, a0, a1, a2);
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return cosine_sum_window<int16_t>(Y, size, a0, a1, a2);
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return cosine_sum_window<int32_t>(Y, size, a0, a1, a2);
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return cosine_sum_window<int64_t>(Y, size, a0, a1, a2);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return cosine_sum_window<uint8_t>(Y, size, a0, a1, a2);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      return cosine_sum_window<uint16_t>(Y, size, a0, a1, a2);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      return cosine_sum_window<uint32_t>(Y, size, a0, a1, a2);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      return cosine_sum_window<uint64_t>(Y, size, a0, a1, a2);
    default:
      ORT_THROW("Unsupported input data type of ", output_datatype);
  }
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

template <typename T>
Status create_mel_weight_matrix(Tensor* Y, int64_t num_mel_bins, int64_t num_spectrogram_bins, int64_t sample_rate, float lower_edge_hertz, float upper_edge_hertz) {
  auto* Y_data = reinterpret_cast<T*>(Y->MutableDataRaw());
  
  std::vector<T> frequency_bins(num_mel_bins + 2);

  auto low_frequency_mel = 2595 * std::log10l(1 + lower_edge_hertz / 700);
  auto high_frequency_mel = 2595 * std::log10l(1 + upper_edge_hertz / 700);
  auto mel_step = (high_frequency_mel - low_frequency_mel) / frequency_bins.size();

  for (size_t index = 0; index < frequency_bins.size(); index++) {
    auto mel_value = low_frequency_mel + mel_step * index;
    auto point = (700 * (10 * (mel_value / 2595) - 1));
    frequency_bins[index] = std::floor(((num_spectrogram_bins + 1) * point) / sample_rate);
  }

  for (int j = 0; j < num_spectrogram_bins; j++) {
    for (int i = 0; i < num_mel_bins; i++) {
      auto& current_element = *(Y_data + (j * num_mel_bins) + i);
      current_element = static_cast<T>(100);
    }
  }

  return Status::OK();
}

static Status create_mel_weight_matrix(OpKernelContext* ctx, onnx::TensorProto_DataType output_datatype,
  int64_t num_mel_bins, int64_t num_spectrogram_bins, int64_t sample_rate, float lower_edge_hertz, float upper_edge_hertz) {
  onnxruntime::TensorShape output_shape({static_cast<int64_t>(num_spectrogram_bins), static_cast<int64_t>(num_mel_bins)});
  auto* Y = ctx->Output(0, output_shape);

  switch (output_datatype) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return create_mel_weight_matrix<float>(Y, num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz);
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return create_mel_weight_matrix<double>(Y, num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz);
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return create_mel_weight_matrix<int16_t>(Y, num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz);
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return create_mel_weight_matrix<int32_t>(Y, num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz);
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return create_mel_weight_matrix<int64_t>(Y, num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return create_mel_weight_matrix<uint8_t>(Y, num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      return create_mel_weight_matrix<uint16_t>(Y, num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      return create_mel_weight_matrix<uint32_t>(Y, num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      return create_mel_weight_matrix<uint64_t>(Y, num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz);
    default:
      ORT_THROW("Unsupported input data type of ", output_datatype);
  }
}

Status MelWeightMatrix::Compute(OpKernelContext* ctx) const {
  const auto num_mel_bins = get_scalar_value_from_tensor<int64_t>(ctx->Input<Tensor>(0));
  const auto num_spectrogram_bins = get_scalar_value_from_tensor<int64_t>(ctx->Input<Tensor>(1));
  const auto sample_rate = get_scalar_value_from_tensor<int64_t>(ctx->Input<Tensor>(2));
  const auto lower_edge_hertz = get_scalar_value_from_tensor<float>(ctx->Input<Tensor>(3));
  const auto upper_edge_hertz = get_scalar_value_from_tensor<float>(ctx->Input<Tensor>(4));

  return create_mel_weight_matrix(ctx, data_type_, num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz);
}

}  // namespace contrib
}  // namespace onnxruntime
