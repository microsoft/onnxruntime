// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef BUILD_MS_EXPERIMENTAL_OPS

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
    kMSExperimentalDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)
        .TypeConstraint("T1", BuildKernelDefConstraints<int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>()),
    HannWindow);

ONNX_OPERATOR_KERNEL_EX(
    HammingWindow,
    kMSExperimentalDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)
        .TypeConstraint("T1", BuildKernelDefConstraints<int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>()),
    HammingWindow);

ONNX_OPERATOR_KERNEL_EX(
    BlackmanWindow,
    kMSExperimentalDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)
        .TypeConstraint("T1", BuildKernelDefConstraints<int64_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>()),
    BlackmanWindow);


ONNX_OPERATOR_KERNEL_EX(
    MelWeightMatrix,
    kMSExperimentalDomain,
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

template <typename T>
static T get_scalar_value_from_tensor(const Tensor* tensor) {
  ORT_ENFORCE(tensor->Shape().Size() == 1, "Tensor input should have a single value.");
  auto data_type = tensor->DataType()->AsPrimitiveDataType()->GetDataType();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return static_cast<T>(*reinterpret_cast<const float*>(tensor->DataRaw()));
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return static_cast<T>(*reinterpret_cast<const double*>(tensor->DataRaw()));
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return static_cast<T>(*reinterpret_cast<const int32_t*>(tensor->DataRaw()));
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return static_cast<T>(*reinterpret_cast<const int64_t*>(tensor->DataRaw()));
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
}

static Status create_cosine_sum_window(
    OpKernelContext* ctx,
    onnx::TensorProto_DataType output_datatype,
    float a0, float a1, float a2) {
  
  // Get the size of the window
  auto size = get_scalar_value_from_tensor<int64_t>(ctx->Input<Tensor>(0));

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

static inline double hz_to_mel_scale(double hz) {
  return 2595 * std::log10(1 + hz / 700);
}

static inline double mel_scale_to_hz(double mels) {
  return 700 * (pow(10, (mels / 2595)) - 1);
}

template <typename T>
Status create_mel_weight_matrix(OpKernelContext* ctx, int64_t num_mel_bins, int64_t dft_length, int64_t sample_rate, float lower_edge_hertz, float upper_edge_hertz) {
  // Determine the width of the spectrogram.
  // This is determined as half the size of the fft size. The first element of the spectrum is always retained,
  // and the remaining are halved. The second half can be discarded due to the conjugate symmetry of the output with real valued ffts.
  // Taken together the formula for the size of the output will be std::floor(dft_length / 2) + 1.
  int64_t num_spectrogram_bins = static_cast<int64_t>(std::floor(dft_length / 2 + 1));

  // Checks
  auto lowest_index = std::floor(((dft_length + 1) * lower_edge_hertz) / sample_rate);
  auto highest_index = std::floor(((dft_length + 1) * upper_edge_hertz) / sample_rate);
  ORT_ENFORCE(lowest_index >= 0 && lowest_index < num_spectrogram_bins, "lower_edge_hertz produces a mel triangle filter bank that is out of range given the dft_length and the sample_rate.");
  ORT_ENFORCE(highest_index >= 0 && highest_index < num_spectrogram_bins, "upper_edge_hertz produces a mel triangle filter bank that is out of range given the dft_length and the sample_rate.");

  // Create the output shape
  onnxruntime::TensorShape output_shape(
      {
          static_cast<int64_t>(num_spectrogram_bins),
          num_mel_bins
      });
  auto* Y = ctx->Output(0, output_shape);

  // Get the raw output data
  auto* Y_data = reinterpret_cast<T*>(Y->MutableDataRaw());

  // Set the weight matrix to 0
  memset(Y_data, 0, num_spectrogram_bins * num_mel_bins * sizeof(T));

  // The mel filterbank is a triangular shaped peak with a height of 1 and a base equal to the size of the MEL range divided by
  // the number of bins needed times 2. This triagle is then slid across the mel domain linearly, with a constant step size that
  // is equal to half of the base of the triange. To accomodate N bins, N+2 data points will be needed to determine the
  // start, center and end points of each mel triange filter.
  //
  // low_frequency where the mel triangle filter banks begin, and they end on the high_frequency_mel
  // The range is divided evenly to create the needed points corresponding to the begin, center, end points of each triangle filterbank
  std::vector<size_t> frequency_bins(num_mel_bins + 2);
  auto low_frequency_mel = hz_to_mel_scale(lower_edge_hertz);
  auto high_frequency_mel = hz_to_mel_scale(upper_edge_hertz);
  auto mel_step = (high_frequency_mel - low_frequency_mel) / static_cast<float>(frequency_bins.size());

  // Convert each point from mel scale back to hertz, and then compute the corresponding index in the fft
  for (size_t i = 0; i < frequency_bins.size(); i++) {
    auto hz = mel_scale_to_hz(low_frequency_mel + mel_step * i);
    frequency_bins[i] = static_cast<size_t>(std::floor(((dft_length + 1) * hz) / sample_rate));
  }

  for (size_t i = 0; i < static_cast<size_t>(num_mel_bins); i++) {
    auto lower_frequency_value = frequency_bins[i];     //left
    auto center_frequency_point = frequency_bins[i+1];  //center
    auto higher_frequency_point = frequency_bins[i+2];  //right

    auto low_to_center = center_frequency_point - lower_frequency_value;
    if (low_to_center == 0) {
      auto& current_element = *(Y_data + (center_frequency_point * num_mel_bins) + i);
      current_element = static_cast<T>(1);
    } else {
      for (size_t j = lower_frequency_value; j <= center_frequency_point; j++) {
        auto& current_element = *(Y_data + (j * num_mel_bins) + i);
        current_element = static_cast<T>((j - lower_frequency_value) / static_cast<T>(low_to_center));
      }
    }

    auto center_to_high = higher_frequency_point - center_frequency_point;
    if (center_to_high > 0) {
      for (size_t j = center_frequency_point; j < higher_frequency_point; j++) {
        auto& current_element = *(Y_data + (j * num_mel_bins) + i);
        current_element = static_cast<T>((higher_frequency_point - j) / static_cast<T>(center_to_high));
      }
    }
  }

  return Status::OK();
}

static Status create_mel_weight_matrix(OpKernelContext* ctx, onnx::TensorProto_DataType output_datatype,
  int64_t num_mel_bins, int64_t dft_length, int64_t sample_rate, float lower_edge_hertz, float upper_edge_hertz) {
  switch (output_datatype) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      ORT_RETURN_IF_ERROR((create_mel_weight_matrix<float>(ctx, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
      ORT_RETURN_IF_ERROR((create_mel_weight_matrix<double>(ctx, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      ORT_RETURN_IF_ERROR((create_mel_weight_matrix<int8_t>(ctx, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
      ORT_RETURN_IF_ERROR((create_mel_weight_matrix<int16_t>(ctx, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      ORT_RETURN_IF_ERROR((create_mel_weight_matrix<int32_t>(ctx, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      ORT_RETURN_IF_ERROR((create_mel_weight_matrix<int64_t>(ctx, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      ORT_RETURN_IF_ERROR((create_mel_weight_matrix<uint8_t>(ctx, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
      ORT_RETURN_IF_ERROR((create_mel_weight_matrix<uint16_t>(ctx, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32: {
      ORT_RETURN_IF_ERROR((create_mel_weight_matrix<uint32_t>(ctx, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
      ORT_RETURN_IF_ERROR((create_mel_weight_matrix<uint64_t>(ctx, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)));
      break;
    }
    default:
      ORT_THROW("Unsupported input data type of ", output_datatype);
  }
  return Status::OK();
}

Status MelWeightMatrix::Compute(OpKernelContext* ctx) const {
  const auto num_mel_bins = get_scalar_value_from_tensor<int64_t>(ctx->Input<Tensor>(0));
  const auto dft_length = get_scalar_value_from_tensor<int64_t>(ctx->Input<Tensor>(1));
  const auto sample_rate = get_scalar_value_from_tensor<int64_t>(ctx->Input<Tensor>(2));
  const auto lower_edge_hertz = get_scalar_value_from_tensor<float>(ctx->Input<Tensor>(3));
  const auto upper_edge_hertz = get_scalar_value_from_tensor<float>(ctx->Input<Tensor>(4));

  ORT_RETURN_IF_ERROR(create_mel_weight_matrix(ctx, data_type_, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz));
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif