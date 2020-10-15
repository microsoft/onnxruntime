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
static Status fft_radix2(OpKernelContext* ctx, size_t batch_idx, const Tensor* X, Tensor* Y, bool inverse) {
  // Get shape and significant bits
  const auto& X_shape = X->Shape();
  size_t number_of_samples = static_cast<size_t>(X_shape[1]);
  unsigned significant_bits = static_cast<unsigned>(log2(number_of_samples));

  // Get data
  auto* X_data = const_cast<U*>(reinterpret_cast<const U*>(X->DataRaw())) + (batch_idx * number_of_samples);
  auto* Y_data = reinterpret_cast<std::complex<T>*>(Y->MutableDataRaw()) + (batch_idx * number_of_samples);

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

  return Status::OK();
}

template <typename T, typename U>
static Status dft_naive(size_t batch_idx, const Tensor* X, Tensor* Y, bool inverse) {
  // Get shape and significant bits
  const auto& X_shape = X->Shape();
  size_t number_of_samples = static_cast<size_t>(X_shape[1]);

  // Get data
  auto* X_data = const_cast<U*>(reinterpret_cast<const U*>(X->DataRaw())) + (batch_idx * number_of_samples);
  auto* Y_data = reinterpret_cast<std::complex<T>*>(Y->MutableDataRaw()) + (batch_idx * number_of_samples);

  auto angular_velocity = compute_angular_velocity<T>(number_of_samples, inverse);

  for (int i = 0; i < number_of_samples; i++) {
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
static Status dft(OpKernelContext* ctx, const Tensor* X, Tensor* Y, bool inverse) {
  // Get shape and significant bits
  const auto& X_shape = X->Shape();
  size_t number_of_batches = static_cast<size_t>(X_shape[0]);
  size_t number_of_samples = static_cast<size_t>(X_shape[1]);
   
  Status status = Status::OK();

  // radix 2 fft
  for (size_t i = 0; i < number_of_batches; i++) {
    if (is_power_of_2(number_of_samples)) {
      status = fft_radix2<T, U>(ctx, i, X, Y, inverse);
    } else {
      status = dft_naive<T, U>(i, X, Y, inverse);
    }
      
    if (!status.IsOK()) {
      return status;
    }
  }
  return status;
}

static Status dft(OpKernelContext* ctx, int64_t signal_ndim, bool inverse) {
  Status status;
  const auto* X = ctx->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  int64_t X_num_dims = static_cast<int64_t>(X_shape.NumDimensions());

  onnxruntime::TensorShape Y_shape;
  if (signal_ndim == 1) {
    ORT_ENFORCE(X_num_dims == 3 || X_num_dims == 2, "FFT dimension is 1D (signal_ndim=1), and so the input tensor dimension must be either [BatchIdx][NumberOfSamples][2] for complex inputs, or [BatchIdx][NumberOfSamples] for real inputs");
    Y_shape = onnxruntime::TensorShape({X_shape[0], X_shape[1], 2});
  } else {
    ORT_ENFORCE(false, "Only 1D DFT is supported. signal_ndim must be 1.");
  }

  auto* Y = ctx->Output(0, Y_shape);
  
  if (signal_ndim == 1) {
    int64_t Y_num_dims = static_cast<int64_t>(Y_shape.NumDimensions());
    ORT_ENFORCE(Y_num_dims == 3, "FFT dimension is 1D (signal_ndim=1), and so the output tensor dimension must be [BatchIdx][NumberOfSamples][2].");
  }

  MLDataType data_type = X->DataType();
  const auto element_size = data_type->Size();
  switch (element_size) {
    case sizeof(float):
      if (X_shape.NumDimensions() == 2) {
        status = dft<float, float>(ctx, X, Y, inverse);
      } else if (X_shape.NumDimensions() == 3 && X_shape[2] == 2) {
        status = dft<float, std::complex<float>>(ctx, X, Y, inverse);
      }
      break;
    case sizeof(double):
      if (X_shape.NumDimensions() == 2) {
        status = dft<double, double>(ctx, X, Y, inverse);
      } else if (X_shape.NumDimensions() == 3 && X_shape[2] == 2) {
        status = dft<double, std::complex<double>>(ctx, X, Y, inverse);
      }
      break;
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
  return status;
}

Status DFT::Compute(OpKernelContext* ctx) const {
  return dft(ctx, signal_ndim_, false);
}

Status IDFT::Compute(OpKernelContext* ctx) const {
  return dft(ctx, signal_ndim_, true);
}

Status STFT::Compute(OpKernelContext* /*ctx*/) const {
  return Status::OK();
}

Status ISTFT::Compute(OpKernelContext* /*ctx*/) const {
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
