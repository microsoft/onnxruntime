// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/Map.h"
#include "fft.h"
#include <functional>

#include "core/platform/threadpool.h"

#include <complex>

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    Fft,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", BuildKernelDefConstraints<float, double>()),
    Fft);

ONNX_OPERATOR_KERNEL_EX(
    Ifft,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", BuildKernelDefConstraints<float, double>()),
    Ifft);


size_t bit_reverse(size_t num, unsigned significant_bits) {
  unsigned output = 0;
  for (unsigned i = 0; i < significant_bits; i++) {
    output += ((num >> i) & 1) << (significant_bits - 1 - i);
  }
  return output;
}

template <typename T, typename U>
static Status fft(OpKernelContext* ctx, const Tensor* X, Tensor* Y, bool inverse) {
  // Get shape and significant bits
  const auto& X_shape = X->Shape();
  size_t number_of_samples = static_cast<size_t>(X_shape[0]);
  unsigned significant_bits = static_cast<unsigned>(log2(number_of_samples));

  // Get data
  auto* X_data = const_cast<U*>(reinterpret_cast<const U*>(X->DataRaw()));
  auto* Y_data = reinterpret_cast<std::complex<T>*>(Y->MutableDataRaw());

  // Calculate fundamental angular velocity
  static const T pi = static_cast<T>(3.14159265);
  static const T tau = 2 * pi;
  T inverse_switch = inverse ? 1.f : -1.f;
  T angular_velocity = inverse_switch * tau / number_of_samples;

  // Create vandermonde matrix V ordered with the bit-reversed permutation
  auto V = std::vector<std::complex<T>>(number_of_samples);  // e^(i *2*pi / N * k)
  for (size_t i = 0; i < number_of_samples; i++) {
    size_t bit_reversed_index = bit_reverse(i, significant_bits);
    V[bit_reversed_index] = std::complex<T>(cos(i * angular_velocity), sin(i * angular_velocity));

    auto x = *(X_data + bit_reversed_index);
    *(Y_data + i) = std::complex<T>(1, 0) * x;
  }

  // Run fft
  unsigned current_significant_bits = 0;
  for (size_t i = 2; i <= number_of_samples; i *= 2) {
    size_t midpoint = i >> 1;
    current_significant_bits++;

    if (current_significant_bits < significant_bits) {
      ctx->GetOperatorThreadPool()->SimpleParallelFor(
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
        });
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

  // Scale the output if inverse fft
  if (inverse) {
    for (int i = 0; i < number_of_samples; i++) {
      std::complex<T>& val = *(Y_data + i);
      val /= static_cast<T>(number_of_samples);
    }
  }

  return Status::OK();
}

Status Fft::Compute(OpKernelContext* ctx) const {
  Status status;
  const auto* X = ctx->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  onnxruntime::TensorShape Y_shape({X_shape[0], 2});
  auto* Y = ctx->Output(0, Y_shape);

  int64_t X_num_dims = static_cast<int64_t>(X_shape.NumDimensions());
  ORT_ENFORCE(X_num_dims >= signal_ndim_, "signal_ndim cannot be greater than the dimension of Input: ", signal_ndim_, " > ", X_num_dims);

  int64_t Y_num_dims = static_cast<int64_t>(Y_shape.NumDimensions());
  ORT_ENFORCE(Y_num_dims == 2, "complex output of fft is returned in a 2D array. But output shape is not 2D.");

  MLDataType data_type = X->DataType();
  const auto element_size = data_type->Size();
  switch (element_size) {
    case sizeof(float):
      if (X_shape.NumDimensions() == 1) {
        status = fft<float, float>(ctx, X, Y, false);
      } else if (X_shape.NumDimensions() == 2 && X_shape[1] == 2) {
        status = fft<float, std::complex<float>>(ctx, X, Y, false);
      }
      break;
    case sizeof(double):
      if (X_shape.NumDimensions() == 1) {
        status = fft<double, double>(ctx, X, Y, false);
      } else if (X_shape.NumDimensions() == 2 && X_shape[1] == 2) {
        status = fft<double, std::complex<double>>(ctx, X, Y, false);
      }
      break;
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
  return status;
}


Status Ifft::Compute(OpKernelContext* ctx) const {
  Status status;
  const auto* X = ctx->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  onnxruntime::TensorShape Y_shape({X_shape[0], 2});
  auto* Y = ctx->Output(0, Y_shape);

  int64_t X_num_dims = static_cast<int64_t>(X_shape.NumDimensions());
  ORT_ENFORCE(X_num_dims >= signal_ndim_, "signal_ndim cannot be greater than the dimension of Input: ", signal_ndim_, " > ", X_num_dims);

  int64_t Y_num_dims = static_cast<int64_t>(Y_shape.NumDimensions());
  ORT_ENFORCE(Y_num_dims == 2, "complex output of fft is returned in a 2D array. But output shape is not 2D.");

  MLDataType data_type = X->DataType();
  const auto element_size = data_type->Size();
  switch (element_size) {
    case sizeof(float):
      if (X_shape.NumDimensions() == 1) {
        status = fft<float, float>(ctx, X, Y, true);
      } else if (X_shape.NumDimensions() == 2 && X_shape[1] == 2) {
        status = fft<float, std::complex<float>>(ctx, X, Y, true);
      }
      break;
    case sizeof(double):
      if (X_shape.NumDimensions() == 1) {
        status = fft<double, double>(ctx, X, Y, true);
      } else if (X_shape.NumDimensions() == 2 && X_shape[1] == 2) {
        status = fft<double, std::complex<double>>(ctx, X, Y, true);
      }
      break;
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
  return status;
}

}  // namespace contrib
}  // namespace onnxruntime
