// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/signal/dft.h"

#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <vector>
#include <core/common/safeint.h>

#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/providers/cpu/signal/utils.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/Map.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    DFT,
    17, 19,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraints<float, double>())
        .TypeConstraint("T2", BuildKernelDefConstraints<int32_t, int64_t>()),
    DFT);

ONNX_CPU_OPERATOR_KERNEL(DFT, 20,
                         KernelDefBuilder()
                             .TypeConstraint("T1", BuildKernelDefConstraints<float, double>())
                             .TypeConstraint("T2", BuildKernelDefConstraints<int32_t, int64_t>()),
                         DFT);

ONNX_CPU_OPERATOR_KERNEL(STFT, 17,
                         KernelDefBuilder()
                             .MayInplace(0, 0)
                             .TypeConstraint("T1", BuildKernelDefConstraints<float, double>())
                             .TypeConstraint("T2", BuildKernelDefConstraints<int32_t, int64_t>()),
                         STFT);

static bool is_real_valued_signal(const onnxruntime::TensorShape& shape) {
  return shape.NumDimensions() == 2 || shape[shape.NumDimensions() - 1] == 1;
}

static bool is_complex_valued_signal(const onnxruntime::TensorShape& shape) {
  return shape.NumDimensions() > 2 && shape[shape.NumDimensions() - 1] == 2;
}

constexpr static bool is_power_of_2(size_t size) {
  unsigned n_bits = 0;
  while (size != 0) {
    n_bits += size & 1;
    size = size >> 1;
  }
  return n_bits == 1;
}

static const unsigned char BitReverseTable256[] = {
    0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0, 0x08, 0x88, 0x48,
    0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8, 0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4,
    0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4, 0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C,
    0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC, 0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2,
    0x32, 0xB2, 0x72, 0xF2, 0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A,
    0xFA, 0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6, 0x0E, 0x8E,
    0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE, 0x01, 0x81, 0x41, 0xC1, 0x21,
    0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1, 0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9,
    0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9, 0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55,
    0xD5, 0x35, 0xB5, 0x75, 0xF5, 0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD,
    0x7D, 0xFD, 0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3, 0x0B,
    0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB, 0x07, 0x87, 0x47, 0xC7,
    0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7, 0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F,
    0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF};

template <typename T>
static inline T bit_reverse(T num, unsigned significant_bits) {
  if (significant_bits > 32) {
    ORT_THROW("Unsupported bit size.");
  }
  uint32_t num_32 = static_cast<uint32_t>(num);
  uint32_t rev = (BitReverseTable256[num_32 & 0xff] << 24) | (BitReverseTable256[(num_32 >> 8) & 0xff] << 16) |
                 (BitReverseTable256[(num_32 >> 16) & 0xff] << 8) | (BitReverseTable256[(num_32 >> 24) & 0xff]);
  return static_cast<T>(((uint64_t)rev) >> (32 - significant_bits));
}

template <typename T>
static T compute_angular_velocity(size_t number_of_samples, bool inverse) {
  // Calculate fundamental angular velocity
  static constexpr T pi = static_cast<T>(M_PI);
  static constexpr T tau = 2 * pi;
  T inverse_switch = inverse ? 1.f : -1.f;
  T angular_velocity = inverse_switch * tau / number_of_samples;
  return angular_velocity;
}

template <typename T>
static std::complex<T> compute_exponential(size_t index, const T angular_velocity) {
  const T angle = static_cast<T>(index) * angular_velocity;
  return std::complex<T>(cos(angle), sin(angle));
}

template <typename T, typename U>
static Status fft_radix2(OpKernelContext* /*ctx*/, const Tensor* X, Tensor* Y, size_t X_offset, size_t X_stride,
                         size_t Y_offset, size_t Y_stride, int64_t axis, size_t dft_length, const Tensor* window,
                         bool is_onesided, bool inverse, InlinedVector<std::complex<T>>& V,
                         InlinedVector<std::complex<T>>& temp_output) {
  // Get shape and significant bits
  const auto& X_shape = X->Shape();
  size_t number_of_samples = static_cast<size_t>(X_shape[onnxruntime::narrow<size_t>(axis)]);
  unsigned significant_bits = static_cast<unsigned>(log2(dft_length));

  // Get data
  auto* X_data = const_cast<U*>(reinterpret_cast<const U*>(X->DataRaw())) + X_offset;
  // Get window
  U* window_data = nullptr;
  if (window) {
    window_data = const_cast<U*>(reinterpret_cast<const U*>(window->DataRaw()));
  }

  size_t Y_data_stride = 1;
  std::complex<T>* Y_data;
  if (is_onesided) {
    if (temp_output.size() != dft_length) {
      temp_output.resize(dft_length);
    }
    Y_data = temp_output.data();
  } else {
    Y_data = reinterpret_cast<std::complex<T>*>(Y->MutableDataRaw()) + Y_offset;
    Y_data_stride = Y_stride;
  }

  auto angular_velocity = compute_angular_velocity<T>(dft_length, inverse);

  // Create vandermonde matrix V ordered with the bit-reversed permutation
  if (V.size() != dft_length) {
    V.resize(dft_length);
    for (size_t i = 0; i < dft_length; i++) {
      size_t bit_reversed_index = bit_reverse(i, significant_bits);
      V[bit_reversed_index] = compute_exponential(i, angular_velocity);
    }
  }

  for (size_t i = 0; i < dft_length; i++) {
    size_t bit_reversed_index = bit_reverse(i, significant_bits);
    auto x = (bit_reversed_index < number_of_samples) ? *(X_data + bit_reversed_index * X_stride) : 0;
    auto window_element = window_data ? *(window_data + bit_reversed_index) : 1;
    *(Y_data + i * Y_data_stride) = std::complex<T>(1, 0) * x * window_element;
  }

  // Run fft_radix2
  unsigned current_significant_bits = 0;
  for (size_t i = 2; i <= dft_length; i <<= 1) {
    size_t midpoint = i >> 1;
    current_significant_bits++;

    for (size_t k = 0; k < midpoint; k++) {
      auto first_idx = bit_reverse(k, current_significant_bits);
      auto second_idx = bit_reverse(midpoint + k, current_significant_bits);
      for (size_t j = 0; j < dft_length; j += i) {
        auto even_index = k + j;
        auto odd_index = k + j + midpoint;
        std::complex<T>* even = (Y_data + even_index * Y_data_stride);
        std::complex<T>* odd = (Y_data + odd_index * Y_data_stride);
        std::complex<T> first = *even + (V[first_idx] * *odd);
        std::complex<T> second = *even + (V[second_idx] * *odd);
        *even = first;
        *odd = second;
      }
    }
  }

  // Scale the output if inverse
  if (inverse) {
    for (size_t i = 0; i < dft_length; i++) {
      std::complex<T>& val = *(Y_data + i * Y_data_stride);
      val /= static_cast<T>(dft_length);
    }
  }

  if (is_onesided) {
    const size_t output_size = (dft_length >> 1) + 1;
    auto destination = reinterpret_cast<std::complex<T>*>(Y->MutableDataRaw()) + Y_offset;
    for (size_t i = 0; i < output_size; i++) {
      *(destination + Y_stride * i) = *(Y_data + i * Y_data_stride);
    }
  }

  return Status::OK();
}

template <typename T>
T next_power_of_2(T in) {
  in--;
  T out = 1;
  while (out <= in) {
    out <<= 1;
  }
  return out;
}

template <typename T, typename U>
static Status dft_bluestein_z_chirp(
    OpKernelContext* ctx, const Tensor* X, Tensor* Y, Tensor& b_fft, Tensor& chirp, size_t X_offset, size_t X_stride, size_t Y_offset, size_t Y_stride,
    int64_t axis, size_t dft_length, const Tensor* window, bool inverse, InlinedVector<std::complex<T>>& V,
    InlinedVector<std::complex<T>>& temp_output) {
  static constexpr T pi = static_cast<T>(M_PI);

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

  size_t N = static_cast<size_t>(dft_length);
  size_t M = next_power_of_2(2 * N - 1);
  auto dft_input_shape = onnxruntime::TensorShape({1, (int64_t)M, 2});
  T scale = inverse ? 1.f / N : 1.f;
  T direction = inverse ? 1.f : -1.f;

  bool should_recreate_b_fft = b_fft.Shape().Size() != dft_input_shape.Size();
  bool should_recreate_chirp = chirp.Shape().Size() != dft_input_shape.Size();
  bool should_recreate = should_recreate_b_fft || should_recreate_chirp;
  if (should_recreate) {
    auto b = onnxruntime::Tensor(X->DataType(), dft_input_shape, alloc);
    b_fft = onnxruntime::Tensor(Y->DataType(), dft_input_shape, alloc);
    chirp = onnxruntime::Tensor(X->DataType(), dft_input_shape, alloc);

    std::complex<T>* b_data = reinterpret_cast<std::complex<T>*>(b.MutableDataRaw());
    std::complex<T>* b_fft_data = reinterpret_cast<std::complex<T>*>(b_fft.MutableDataRaw());
    std::complex<T>* chirp_data = reinterpret_cast<std::complex<T>*>(chirp.MutableDataRaw());
    memset(reinterpret_cast<void*>(b_data), 0, b.SizeInBytes());
    memset(reinterpret_cast<void*>(b_fft_data), 0, b_fft.SizeInBytes());
    memset(reinterpret_cast<void*>(chirp_data), 0, chirp.SizeInBytes());

    for (size_t n = 0; n < N; n++) {
      std::complex<T>& chirp_n = *(chirp_data + n);
      // chirp
      auto exponent = direction * pi * n * n / N;
      chirp_n = std::complex<T>(cos(exponent), sin(exponent));

      // b
      std::complex<T>& b_n = *(b_data + n);
      b_n = std::conj(chirp_n);
    }

    for (size_t n = M - N + 1; n < M; n++) {
      std::complex<T>& b_n = *(b_data + n);
      std::complex<T>& b_m_minus_n = *(b_data + M - n);
      b_n = b_m_minus_n;
    }

    // Forward FFT radix2 for the "b" signal
    // This will be cached and reused!
    ORT_RETURN_IF_ERROR((fft_radix2<T, std::complex<T>>(ctx, &b, &b_fft, 0, 1, 0, 1, 1, M, nullptr,
                                                        false, false, V, temp_output)));
  }

  // Get data
  auto* X_data = const_cast<U*>(reinterpret_cast<const U*>(X->DataRaw())) + X_offset;
  auto* Y_data = reinterpret_cast<std::complex<T>*>(Y->MutableDataRaw()) + Y_offset;
  U* window_data = nullptr;
  if (window) {
    window_data = const_cast<U*>(reinterpret_cast<const U*>(window->DataRaw()));
  }

  auto a = onnxruntime::Tensor(X->DataType(), dft_input_shape, alloc);
  auto a_fft = onnxruntime::Tensor(Y->DataType(), dft_input_shape, alloc);
  std::complex<T>* a_data = reinterpret_cast<std::complex<T>*>(a.MutableDataRaw());
  std::complex<T>* a_fft_data = reinterpret_cast<std::complex<T>*>(a_fft.MutableDataRaw());
  std::complex<T>* b_fft_data = reinterpret_cast<std::complex<T>*>(b_fft.MutableDataRaw());
  std::complex<T>* chirp_data = reinterpret_cast<std::complex<T>*>(chirp.MutableDataRaw());
  memset(reinterpret_cast<void*>(a_data), 0, a.SizeInBytes());

  const auto& X_shape = X->Shape();
  size_t number_of_samples = static_cast<size_t>(X_shape[onnxruntime::narrow<size_t>(axis)]);

  // Prepare "a" signal
  for (size_t n = 0; n < number_of_samples; n++) {
    std::complex<T>& a_n = *(a_data + n);
    std::complex<T>& chirp_n = *(chirp_data + n);
    auto window_n = window_data ? *(window_data + n) : 1;
    a_n = *(X_data + n * X_stride);  // input
    a_n *= window_n;
    a_n *= chirp_n;
  }

  // Forward FFT radix2 for the "a" signal
  ORT_RETURN_IF_ERROR((fft_radix2<T, std::complex<T>>(ctx, &a, &a_fft, 0, 1, 0, 1, 1, M, nullptr,
                                                      false, false, V, temp_output)));

  for (size_t i = 0; i < M; i++) {
    std::complex<T>& a_i = *(a_fft_data + i);
    std::complex<T>& b_i = *(b_fft_data + i);
    a_i *= b_i;
  }

  // Inverse FFT radix2 for the "a" signal
  ORT_RETURN_IF_ERROR((fft_radix2<T, std::complex<T>>(ctx, &a_fft, &a, 0, 1, 0, 1, 1, M, nullptr,
                                                      false, true, V, temp_output)));
  const auto& Y_shape = Y->Shape();
  size_t dft_output_size = static_cast<size_t>(Y_shape[onnxruntime::narrow<size_t>(axis)]);

  for (size_t i = 0; i < dft_output_size; i++) {
    std::complex<T>& chirp_i = *(chirp_data + i);
    std::complex<T>& out = *(Y_data + i * Y_stride);
    std::complex<T>& c_i = *(a_data + i);
    if (i > 0) {
      // The inverse fft is computed using the same cached vandermonde matrix (V) created by the
      // forward fft. This reversal causes the output to be reversed as well.
      // Therefore we undo the reversal when writing the output back out.
      c_i = *(a_data + M - i);
    }
    out = c_i * chirp_i * scale;
  }
  return Status::OK();
}

template <typename T, typename U>
static Status discrete_fourier_transform(OpKernelContext* ctx, const Tensor* X, Tensor* Y, Tensor& b_fft, Tensor& chirp,
                                         int64_t axis, int64_t dft_length, const Tensor* window, bool is_onesided, bool inverse,
                                         InlinedVector<std::complex<T>>& V,
                                         InlinedVector<std::complex<T>>& temp_output) {
  // Get shape
  const auto& X_shape = X->Shape();
  const auto& Y_shape = Y->Shape();

  auto batch_and_signal_rank = X->Shape().NumDimensions();
  auto total_dfts = static_cast<size_t>(X->Shape().Size() / X->Shape()[onnxruntime::narrow<size_t>(axis)]);

  auto is_input_real = X->Shape().NumDimensions() == 2 || X->Shape()[X->Shape().NumDimensions() - 1] == 1;
  auto complex_input_factor = is_input_real ? 1 : 2;
  if (X->Shape().NumDimensions() > 2) {
    total_dfts /= onnxruntime::narrow<size_t>(X->Shape()[X->Shape().NumDimensions() - 1]);
    batch_and_signal_rank -= 1;
  }

  // Calculate x/y offsets/strides
  for (size_t i = 0; i < total_dfts; i++) {
    size_t X_offset = 0;
    size_t X_stride = onnxruntime::narrow<size_t>(X_shape.SizeFromDimension(SafeInt<size_t>(axis) + 1) / complex_input_factor);
    size_t cumulative_packed_stride = total_dfts;
    size_t temp = i;
    for (size_t r = 0; r < batch_and_signal_rank; r++) {
      if (r == static_cast<size_t>(axis)) {
        continue;
      }
      cumulative_packed_stride /= onnxruntime::narrow<size_t>(X_shape[r]);
      auto index = temp / cumulative_packed_stride;
      temp -= (index * cumulative_packed_stride);
      X_offset += index * SafeInt<size_t>(X_shape.SizeFromDimension(r + 1)) / complex_input_factor;
    }

    size_t Y_offset = 0;
    size_t Y_stride = onnxruntime::narrow<size_t>(Y_shape.SizeFromDimension(SafeInt<size_t>(axis) + 1) / 2);
    cumulative_packed_stride = total_dfts;
    temp = i;
    for (size_t r = 0; r < batch_and_signal_rank; r++) {
      if (r == static_cast<size_t>(axis)) {
        continue;
      }
      cumulative_packed_stride /= onnxruntime::narrow<size_t>(X_shape[r]);
      auto index = temp / cumulative_packed_stride;
      temp -= (index * cumulative_packed_stride);
      Y_offset += index * SafeInt<size_t>(Y_shape.SizeFromDimension(r + 1)) / 2;
    }

    if (is_power_of_2(onnxruntime::narrow<size_t>(dft_length))) {
      ORT_RETURN_IF_ERROR((fft_radix2<T, U>(ctx, X, Y, X_offset, X_stride, Y_offset, Y_stride, axis, onnxruntime::narrow<size_t>(dft_length), window,
                                            is_onesided, inverse, V, temp_output)));
    } else {
      ORT_RETURN_IF_ERROR(
          (dft_bluestein_z_chirp<T, U>(ctx, X, Y, b_fft, chirp, X_offset, X_stride, Y_offset, Y_stride, axis, onnxruntime::narrow<size_t>(dft_length), window, inverse, V, temp_output)));
    }
  }

  return Status::OK();
}

static Status discrete_fourier_transform(OpKernelContext* ctx, int64_t axis, bool is_onesided, bool inverse) {
  // Get input shape
  const auto* X = ctx->Input<Tensor>(0);
  const auto* dft_length = ctx->Input<Tensor>(1);
  const auto& X_shape = X->Shape();
  const auto is_real_valued = is_real_valued_signal(X_shape);
  const auto is_complex_valued = is_complex_valued_signal(X_shape);
  axis = HandleNegativeAxis(axis, X_shape.NumDimensions());

  int64_t number_of_samples = static_cast<int64_t>(X_shape[onnxruntime::narrow<size_t>(axis)]);
  if (dft_length) {
    const auto& dft_length_shape = dft_length->Shape();
    ORT_RETURN_IF(!dft_length_shape.IsScalar(), "dft_length must be a scalar value.");
    number_of_samples = static_cast<int>(signal::get_scalar_value_from_tensor<int64_t>(dft_length));
    ORT_RETURN_IF(number_of_samples <= 0, "dft_length must be greater than zero.");
  }

  // Get the DFT output size. Onesided will return only the unique values!
  // note: x >> 1 === std::floor(x / 2.f)
  auto dft_output_size = is_onesided ? ((number_of_samples >> 1) + 1) : number_of_samples;

  // Get output shape
  auto Y_shape = onnxruntime::TensorShape(X_shape);
  if (X_shape.NumDimensions() == 2) {
    Y_shape = onnxruntime::TensorShape({X_shape[0], dft_output_size, 2});
  } else {
    Y_shape[Y_shape.NumDimensions() - 1] = 2;
  }
  Y_shape[onnxruntime::narrow<size_t>(axis)] = dft_output_size;
  auto Y = ctx->Output(0, Y_shape);

  // Get data type
  auto data_type = X->DataType();

  Tensor b_fft, chirp;
  auto element_size = data_type->Size();
  if (element_size == sizeof(float)) {
    InlinedVector<std::complex<float>> V;
    InlinedVector<std::complex<float>> temp_output;
    if (is_real_valued) {
      ORT_RETURN_IF_ERROR((discrete_fourier_transform<float, float>(ctx, X, Y, b_fft, chirp, axis, number_of_samples, nullptr,
                                                                    is_onesided, inverse, V, temp_output)));
    } else if (is_complex_valued) {
      ORT_RETURN_IF_ERROR((discrete_fourier_transform<float, std::complex<float>>(
          ctx, X, Y, b_fft, chirp, axis, number_of_samples, nullptr, is_onesided, inverse, V, temp_output)));
    } else {
      ORT_THROW(
          "Unsupported input signal shape. The signal's first dimension must be the batch dimension and its second "
          "dimension must be the signal length dimension. It may optionally include a 3rd dimension of size 2 for "
          "complex inputs.",
          data_type);
    }
  } else if (element_size == sizeof(double)) {
    InlinedVector<std::complex<double>> V;
    InlinedVector<std::complex<double>> temp_output;
    if (is_real_valued) {
      ORT_RETURN_IF_ERROR((discrete_fourier_transform<double, double>(ctx, X, Y, b_fft, chirp, axis, number_of_samples, nullptr,
                                                                      is_onesided, inverse, V, temp_output)));
    } else if (is_complex_valued) {
      ORT_RETURN_IF_ERROR((discrete_fourier_transform<double, std::complex<double>>(
          ctx, X, Y, b_fft, chirp, axis, number_of_samples, nullptr, is_onesided, inverse, V, temp_output)));
    } else {
      ORT_THROW(
          "Unsupported input signal shape. The signal's first dimension must be the batch dimension and its second "
          "dimension must be the signal length dimension. It may optionally include a 3rd dimension of size 2 for "
          "complex inputs.",
          data_type);
    }
  } else {
    ORT_THROW("Unsupported input data type of ", data_type);
  }

  return Status::OK();
}

Status DFT::Compute(OpKernelContext* ctx) const {
  int64_t axis = axis_;
  if (opset_ >= 20 && ctx->InputCount() >= 3) {
    const Tensor* axes_tensor = ctx->Input<Tensor>(2);
    axis = axes_tensor->Data<int64_t>()[0];
  }

  ORT_RETURN_IF_ERROR(discrete_fourier_transform(ctx, axis, is_onesided_, is_inverse_));
  return Status::OK();
}

template <typename T, typename U>
static Status short_time_fourier_transform(OpKernelContext* ctx, bool is_onesided, bool /*inverse*/) {
  // Attr("onesided"): default = 1
  // Input(0, "signal") type = T1
  // Input(1, "frame_length") type = T2
  // Input(2, "window") type = T1, optional
  // Input(3, "frame_step") type = T2
  // Output(0, "output") type = T1

  // Get signal
  const auto* signal = ctx->Input<Tensor>(0);
  const auto frame_step = signal::get_scalar_value_from_tensor<int64_t>(ctx->Input<Tensor>(1));
  const auto* window = ctx->Input<Tensor>(2);
  const auto* frame_length_tensor = ctx->Input<Tensor>(3);

  // Get input signal shape
  const auto& signal_shape = signal->Shape();
  const auto batch_size = signal_shape[0];
  const auto signal_size = signal_shape[1];
  const auto signal_components = signal_shape.NumDimensions() == 2   ? 1
                                 : signal_shape.NumDimensions() == 3 ? signal_shape[2]
                                                                     : 0;  // error
  ORT_ENFORCE(signal_components == 1 || signal_components == 2,
              "signal shape must end in 1 (real) or 2 (real, imaginary).");

  // Get the frame length
  int64_t frame_length = std::numeric_limits<int64_t>::min();
  if (frame_length_tensor) {
    frame_length = signal::get_scalar_value_from_tensor<int64_t>(frame_length_tensor);
  }

  // Get window length
  int64_t window_length = std::numeric_limits<int64_t>::min();
  if (window) {
    window_length = window->Shape()[0];
  }

  // The frame_length and window inputs are generally used interchangeably, and should match!
  if (frame_length != std::numeric_limits<int64_t>::min() && window_length != std::numeric_limits<int64_t>::min()) {
    ORT_ENFORCE(
        frame_length == window_length,
        "If both frame_length and window are set, then the size of the window must be equal to the frame_length.");
  }

  // Calculate the window size with preference to the window input.
  const auto window_size = window ? window->Shape()[0] : frame_length;
  ORT_ENFORCE(window_size <= signal_size, "Ensure that the dft size is smaller than the signal.");

  // Calculate the number of dfts to run
  const auto n_dfts =
      static_cast<int64_t>(std::floor((signal_size - window_size) / static_cast<float>(frame_step))) + 1;

  // Calculate the output spectra length (onesided will return only the unique values)
  // note: x >> 1 === std::floor(x / 2.f)
  const auto dft_output_size = is_onesided ? (window_size >> 1) + 1 : window_size;

  // Get/create the output mutable data
  auto output_spectra_shape = onnxruntime::TensorShape({batch_size, n_dfts, dft_output_size, 2});
  auto Y = ctx->Output(0, output_spectra_shape);
  auto Y_data = reinterpret_cast<T*>(Y->MutableDataRaw());

  // Get/create the signal mutable data
  auto* signal_data = const_cast<U*>(reinterpret_cast<const U*>(signal->DataRaw()));

  // Define tensor shapes for each dft run
  constexpr int64_t output_components = 2;
  auto dft_input_shape = onnxruntime::TensorShape({1, window_size, signal_components});
  auto dft_output_shape = onnxruntime::TensorShape({1, dft_output_size, output_components});

  Tensor b_fft, chirp;
  InlinedVector<std::complex<T>> V;
  InlinedVector<std::complex<T>> temp_output;

  // Run each dft of each batch as if it was a real-valued batch size 1 dft operation
  for (int64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    for (int64_t i = 0; i < n_dfts; i++) {
      auto input_frame_begin =
          signal_data + (batch_idx * signal_size * signal_components) + (i * frame_step * signal_components);

      auto output_frame_begin = Y_data + (batch_idx * n_dfts * dft_output_size * output_components) +
                                (i * dft_output_size * output_components);

      // Tensors do not own the backing memory, so no worries on destruction
      auto input = onnxruntime::Tensor(signal->DataType(), dft_input_shape, input_frame_begin, signal->Location(), 0);

      auto output = onnxruntime::Tensor(Y->DataType(), dft_output_shape, output_frame_begin, Y->Location(), 0);

      // Run individual dft
      ORT_RETURN_IF_ERROR((discrete_fourier_transform<T, U>(ctx, &input, &output, b_fft, chirp, 1, window_size, window, is_onesided,
                                                            false, V, temp_output)));
    }
  }

  return Status::OK();
}

Status STFT::Compute(OpKernelContext* ctx) const {
  // Attr("onesided"): default = 1
  // Input(0, "signal") type = T1
  // Input(1, "frame_length") type = T2
  // Input(2, "window") type = T1, optional
  // Input(3, "frame_step") type = T2
  // Output(0, "output") type = T1

  // Get signal shape
  const auto* signal = ctx->Input<Tensor>(0);
  const auto& signal_shape = signal->Shape();
  const auto is_real_valued = is_real_valued_signal(signal_shape);
  const auto is_complex_valued = is_complex_valued_signal(signal_shape);

  // Get data type
  auto data_type = signal->DataType();

  const auto element_size = data_type->Size();
  if (element_size == sizeof(float)) {
    if (is_real_valued) {
      ORT_RETURN_IF_ERROR((short_time_fourier_transform<float, float>(ctx, is_onesided_, false)));
    } else if (is_complex_valued) {
      ORT_RETURN_IF_ERROR((short_time_fourier_transform<float, std::complex<float>>(ctx, is_onesided_, false)));
    } else {
      ORT_THROW(
          "Unsupported input signal shape. The signal's first dimenstion must be the batch dimension and its second "
          "dimension must be the signal length dimension. It may optionally include a 3rd dimension of size 2 for "
          "complex inputs.",
          data_type);
    }
  } else if (element_size == sizeof(double)) {
    if (is_real_valued) {
      ORT_RETURN_IF_ERROR((short_time_fourier_transform<double, double>(ctx, is_onesided_, false)));
    } else if (is_complex_valued) {
      ORT_RETURN_IF_ERROR((short_time_fourier_transform<double, std::complex<double>>(ctx, is_onesided_, false)));
    } else {
      ORT_THROW(
          "Unsupported input signal shape. The signal's first dimenstion must be the batch dimension and its second "
          "dimension must be the signal length dimension. It may optionally include a 3rd dimension of size 2 for "
          "complex inputs.",
          data_type);
    }
  } else {
    ORT_THROW("Unsupported input data type of ", data_type);
  }

  return Status::OK();
}

}  // namespace onnxruntime
