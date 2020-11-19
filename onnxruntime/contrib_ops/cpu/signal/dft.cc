// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/Map.h"
#include "dft.h"
#include <functional>

#include "core/platform/threadpool.h"

#include <complex>

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    DFT,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", BuildKernelDefConstraints<float, double>()),
    DFT);

ONNX_OPERATOR_KERNEL_EX(
    IDFT,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", BuildKernelDefConstraints<float, double>()),
    IDFT);

ONNX_OPERATOR_KERNEL_EX(
    STFT,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", BuildKernelDefConstraints<float, double>()),
    STFT);

ONNX_OPERATOR_KERNEL_EX(
    ISTFT,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", BuildKernelDefConstraints<float, double>()),
    ISTFT);

bool is_power_of_2(size_t size) {
  unsigned n_bits = 0;
  while (size != 0) {
    n_bits += size & 1;
    size = size >> 1;
  }
  return n_bits == 1;
}

size_t bit_reverse(size_t num, unsigned significant_bits) {
  unsigned output = 0;
  for (unsigned i = 0; i < significant_bits; i++) {
    output += ((num >> i) & 1) << (significant_bits - 1 - i);
  }
  return output;
}

template <typename T>
static T compute_angular_velocity(size_t number_of_samples, bool inverse) {
  // Calculate fundamental angular velocity
  static const T pi = static_cast<T>(3.14159265);
  static const T tau = 2 * pi;
  T inverse_switch = inverse ? 1.f : -1.f;
  T angular_velocity = inverse_switch * tau / number_of_samples;
  return angular_velocity;
}

template <typename T, typename U>
static Status fft_radix2(OpKernelContext* ctx, size_t batch_idx, const Tensor* X, Tensor* Y, bool is_onesided, bool inverse) {
  // Get shape and significant bits
  const auto& X_shape = X->Shape();
  size_t number_of_samples = static_cast<size_t>(X_shape[1]);
  unsigned significant_bits = static_cast<unsigned>(log2(number_of_samples));

  // Get data
  auto* X_data = const_cast<U*>(reinterpret_cast<const U*>(X->DataRaw())) + (batch_idx * number_of_samples);

  std::unique_ptr<std::complex<T>[]> temp_output = nullptr;
  std::complex<T>* Y_data;
  if (is_onesided) {
    temp_output = std::unique_ptr<std::complex<T>[]>(new std::complex<T>[number_of_samples]);
    Y_data = temp_output.get();
  } else {
    Y_data = reinterpret_cast<std::complex<T>*>(Y->MutableDataRaw()) + (batch_idx * number_of_samples);
  }

  auto angular_velocity = compute_angular_velocity<T>(number_of_samples, inverse);

  // Create vandermonde matrix V ordered with the bit-reversed permutation
  auto V = std::vector<std::complex<T>>(number_of_samples);  // e^(i *2*pi / N * k)
  for (size_t i = 0; i < number_of_samples; i++) {
    size_t bit_reversed_index = bit_reverse(i, significant_bits);
    V[bit_reversed_index] = std::complex<T>(cos(i * angular_velocity), sin(i * angular_velocity));

    auto x = *(X_data + bit_reversed_index);
    *(Y_data + i) = std::complex<T>(1, 0) * x;
  }

  // Run fft_radix2
  unsigned current_significant_bits = 0;
  for (size_t i = 2; i <= number_of_samples; i *= 2) {
    size_t midpoint = i >> 1;
    current_significant_bits++;

    if (current_significant_bits < significant_bits) {
      onnxruntime::concurrency::ThreadPool::TryBatchParallelFor(
        ctx->GetOperatorThreadPool(),
        static_cast<int32_t>(number_of_samples/i),
        [=, &V](ptrdiff_t task_idx) {
          size_t j = task_idx * i;
          for (size_t k = 0; k < midpoint; k++) {
            std::complex<T>* even = (Y_data + j) + k;
            std::complex<T>* odd = (Y_data + j) + (midpoint + k);
            std::complex<T> first = *even + (V[bit_reverse(k, current_significant_bits)] * *odd);
            std::complex<T> second = *even + (V[bit_reverse(midpoint + k, current_significant_bits)] * *odd);
            *even = first;
            *odd = second;
          }
        }, 0);
    } else {
      for (size_t j = 0; j < number_of_samples; j += i) {
        for (size_t k = 0; k < midpoint; k++) {
          std::complex<T>* even = (Y_data + j) + k;
          std::complex<T>* odd = (Y_data + j) + (midpoint + k);
          std::complex<T> first = *even + (V[bit_reverse(k, current_significant_bits)] * *odd);
          std::complex<T> second = *even + (V[bit_reverse(midpoint + k, current_significant_bits)] * *odd);
          *even = first;
          *odd = second;
        }
      }
    }
  }

  // Scale the output if inverse
  if (inverse) {
    for (int i = 0; i < number_of_samples; i++) {
      std::complex<T>& val = *(Y_data + i);
      val /= static_cast<T>(number_of_samples);
    }
  }

  if (is_onesided) {
    const auto& Y_shape = Y->Shape();
    size_t n_fft = static_cast<size_t>(Y_shape[1]);
    auto destination = reinterpret_cast<std::complex<T>*>(Y->MutableDataRaw()) + (batch_idx * n_fft);
    memcpy(destination, Y_data, sizeof(std::complex<T>) * n_fft);
  }

  return Status::OK();
}

template <typename T, typename U>
static Status dft_naive(size_t batch_idx, const Tensor* X, Tensor* Y, bool inverse) {
  // Get shape and significant bits
  const auto& X_shape = X->Shape();
  size_t number_of_samples = static_cast<size_t>(X_shape[1]);
  const auto& Y_shape = Y->Shape();
  size_t n_fft = static_cast<size_t>(Y_shape[1]);

  // Get data
  auto* X_data = const_cast<U*>(reinterpret_cast<const U*>(X->DataRaw())) + (batch_idx * number_of_samples);
  auto* Y_data = reinterpret_cast<std::complex<T>*>(Y->MutableDataRaw()) + (batch_idx * n_fft);

  auto angular_velocity = compute_angular_velocity<T>(number_of_samples, inverse);

  for (int i = 0; i < n_fft; i++) {
    std::complex<T>& out = *(Y_data + i);
    out.real(0);
    out.imag(0);

    for (int j = 0; j < number_of_samples; j++) {  // vectorize over this loop
      auto exponential = std::complex<T>(cos(i * j * angular_velocity), sin(i * j * angular_velocity));
      auto element = *(X_data + j);
      out += exponential * element;
    }

    if (inverse) {
      out /= static_cast<T>(number_of_samples);
    }
  }

  return Status::OK();
}

template <typename T, typename U>
static Status dft(OpKernelContext* ctx, const Tensor* X, Tensor* Y, bool is_onesided, bool inverse) {
  // Get shape and significant bits
  const auto& X_shape = X->Shape();
  size_t number_of_batches = static_cast<size_t>(X_shape[0]);
  size_t number_of_samples = static_cast<size_t>(X_shape[1]);
   
  Status status = Status::OK();

  // radix 2 fft
  for (size_t i = 0; i < number_of_batches; i++) {
    if (is_power_of_2(number_of_samples)) {
      status = fft_radix2<T, U>(ctx, i, X, Y, is_onesided, inverse);
    } else {
      status = dft_naive<T, U>(i, X, Y, inverse);
    }
      
    if (!status.IsOK()) {
      return status;
    }
  }
  return status;
}

static Status dft(OpKernelContext* ctx, bool is_onesided, bool inverse) {
  Status status;
  const auto* X = ctx->Input<Tensor>(0);
  const auto& X_shape = X->Shape();

  onnxruntime::TensorShape Y_shape;
  int64_t n_fft = is_onesided ? static_cast<int64_t>(std::floor(X_shape[1]/2.f) + 1) : X_shape[1];
  Y_shape = onnxruntime::TensorShape({X_shape[0], n_fft, 2});
  auto* Y = ctx->Output(0, Y_shape);
  
  MLDataType data_type = X->DataType();
  const auto element_size = data_type->Size();
  switch (element_size) {
    case sizeof(float):
      if (X_shape.NumDimensions() == 2) {
        status = dft<float, float>(ctx, X, Y, is_onesided, inverse);
      } else if (X_shape.NumDimensions() == 3 && X_shape[2] == 2) {
        status = dft<float, std::complex<float>>(ctx, X, Y, is_onesided, inverse);
      }
      break;
    case sizeof(double):
      if (X_shape.NumDimensions() == 2) {
        status = dft<double, double>(ctx, X, Y, is_onesided, inverse);
      } else if (X_shape.NumDimensions() == 3 && X_shape[2] == 2) {
        status = dft<double, std::complex<double>>(ctx, X, Y, is_onesided, inverse);
      }
      break;
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
  return status;
}

Status DFT::Compute(OpKernelContext* ctx) const {
  return dft(ctx, is_onesided_, false);
}

Status IDFT::Compute(OpKernelContext* ctx) const {
  return dft(ctx, false, true);
}

// dedupe with the other one in window_functions.cc
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


template <typename T, typename U>
static Status stft(OpKernelContext* ctx, bool is_onesided, bool /*inverse*/) {
  Status status = Status::OK();

  
  // Get signal
  const auto* signal = ctx->Input<Tensor>(0);
  const auto& signal_shape = signal->Shape();

  const auto* window = ctx->Input<Tensor>(2);
  const auto& window_shape = window->Shape();
  const auto window_size = window_shape[0];

  const auto batch_size = signal_shape[0];
  const auto signal_size = signal_shape[1];
  const auto dft_size = get_scalar_value_from_tensor<int64_t>(ctx->Input<Tensor>(1));
  const auto hop_length = get_scalar_value_from_tensor<int64_t>(ctx->Input<Tensor>(3));
  const auto dft_output_size = is_onesided ? static_cast<int64_t>(std::floor(window_size / 2.f) + 1) : window_size;
  ORT_ENFORCE(window_size < signal_size, "Ensure that the dft size is smaller than the signal.");

  const auto number_of_dfts = static_cast<int64_t>(std::ceil((signal_size - window_size) / static_cast<float>(hop_length)));
  onnxruntime::TensorShape spectra_shape({batch_size, number_of_dfts, dft_output_size, 2});
  auto* Y = ctx->Output(0, spectra_shape);
  auto* Y_data = reinterpret_cast<T*>(Y->MutableDataRaw());
  memset(Y_data, 1, sizeof(T) * spectra_shape.Size());

  return status;

}



Status STFT::Compute(OpKernelContext* ctx) const {
  Status status;
  const auto* signal = ctx->Input<Tensor>(0);
  const auto& signal_shape = signal->Shape();
  MLDataType data_type = signal->DataType();
  const auto element_size = data_type->Size();
  switch (element_size) {
    case sizeof(float):
      if (signal_shape.NumDimensions() == 2) {
        // real
        status = stft<float, float>(ctx, is_onesided_, false);
      } else if (signal_shape.NumDimensions() == 3 && signal_shape[2] == 2) {
        // complex
        status = stft<float, std::complex<float>>(ctx, is_onesided_, false);
      }
      break;
    case sizeof(double):
      if (signal_shape.NumDimensions() == 2) {
        status = stft<double, double>(ctx, is_onesided_, false);
      } else if (signal_shape.NumDimensions() == 3 && signal_shape[2] == 2) {
        status = stft<double, std::complex<double>>(ctx, is_onesided_, false);
      }
      break;
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }

  return status;
}

Status ISTFT::Compute(OpKernelContext* /*ctx*/) const {
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
