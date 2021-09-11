// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef BUILD_MS_EXPERIMENTAL_OPS

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
    kMSExperimentalDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", BuildKernelDefConstraints<float, double>()),
    DFT);

ONNX_OPERATOR_KERNEL_EX(
    IDFT,
    kMSExperimentalDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", BuildKernelDefConstraints<float, double>()),
    IDFT);

ONNX_OPERATOR_KERNEL_EX(
    STFT,
    kMSExperimentalDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", BuildKernelDefConstraints<float, double>()),
    STFT);

static bool is_real_valued_signal(const onnxruntime::TensorShape & shape) {
  // The first dimention is the batch size
  // The second dimention is the signal value
  return shape.NumDimensions() == 2;
}

static bool is_complex_valued_signal(const onnxruntime::TensorShape& shape) {
  // The first dimention is the batch size
  // The second dimention is the signal length
  // The third dimention is set to 2 and represents the real and imaginary parts of the complex sample
  return shape.NumDimensions() == 3 && shape[2] == 2;
}

static bool is_power_of_2(size_t size) {
  unsigned n_bits = 0;
  while (size != 0) {
    n_bits += size & 1;
    size = size >> 1;
  }
  return n_bits == 1;
}

static const unsigned char BitReverseTable256[] =
{
    0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0,
    0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,
    0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4,
    0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,
    0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,
    0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
    0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,
    0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
    0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
    0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
    0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
    0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
    0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
    0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
    0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,
    0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF};

template <unsigned TSignificantBits>
uint32_t bit_reverse(uint32_t num) {
  uint32_t rev = (BitReverseTable256[num & 0xff] << 24) |
         (BitReverseTable256[(num >> 8) & 0xff] << 16) |
         (BitReverseTable256[(num >> 16) & 0xff] << 8) |
         (BitReverseTable256[(num >> 24) & 0xff]);
  return static_cast<uint32_t>(((uint64_t)rev) >> (32 - TSignificantBits));
}

template <typename T>
static inline T bit_reverse(T num, unsigned significant_bits) {
  switch (significant_bits) {
    case 0: return static_cast<T>(bit_reverse<0>(static_cast<uint32_t>(num)));
    case 1: return static_cast<T>(bit_reverse<1>(static_cast<uint32_t>(num)));
    case 2: return static_cast<T>(bit_reverse<2>(static_cast<uint32_t>(num)));
    case 3: return static_cast<T>(bit_reverse<3>(static_cast<uint32_t>(num)));
    case 4: return static_cast<T>(bit_reverse<4>(static_cast<uint32_t>(num)));
    case 5: return static_cast<T>(bit_reverse<5>(static_cast<uint32_t>(num)));
    case 6: return static_cast<T>(bit_reverse<6>(static_cast<uint32_t>(num)));
    case 7: return static_cast<T>(bit_reverse<7>(static_cast<uint32_t>(num)));
    case 8: return static_cast<T>(bit_reverse<8>(static_cast<uint32_t>(num)));
    case 9: return static_cast<T>(bit_reverse<9>(static_cast<uint32_t>(num)));
    case 10: return static_cast<T>(bit_reverse<10>(static_cast<uint32_t>(num)));
    case 11: return static_cast<T>(bit_reverse<11>(static_cast<uint32_t>(num)));
    case 12: return static_cast<T>(bit_reverse<12>(static_cast<uint32_t>(num)));
    case 13: return static_cast<T>(bit_reverse<13>(static_cast<uint32_t>(num)));
    case 14: return static_cast<T>(bit_reverse<14>(static_cast<uint32_t>(num)));
    case 15: return static_cast<T>(bit_reverse<15>(static_cast<uint32_t>(num)));
    case 16: return static_cast<T>(bit_reverse<16>(static_cast<uint32_t>(num)));
    case 17: return static_cast<T>(bit_reverse<17>(static_cast<uint32_t>(num)));
    case 18: return static_cast<T>(bit_reverse<18>(static_cast<uint32_t>(num)));
    case 19: return static_cast<T>(bit_reverse<19>(static_cast<uint32_t>(num)));
    case 20: return static_cast<T>(bit_reverse<20>(static_cast<uint32_t>(num)));
    case 21: return static_cast<T>(bit_reverse<21>(static_cast<uint32_t>(num)));
    case 22: return static_cast<T>(bit_reverse<22>(static_cast<uint32_t>(num)));
    case 23: return static_cast<T>(bit_reverse<23>(static_cast<uint32_t>(num)));
    case 24: return static_cast<T>(bit_reverse<24>(static_cast<uint32_t>(num)));
    case 25: return static_cast<T>(bit_reverse<25>(static_cast<uint32_t>(num)));
    case 26: return static_cast<T>(bit_reverse<26>(static_cast<uint32_t>(num)));
    case 27: return static_cast<T>(bit_reverse<27>(static_cast<uint32_t>(num)));
    case 28: return static_cast<T>(bit_reverse<28>(static_cast<uint32_t>(num)));
    case 29: return static_cast<T>(bit_reverse<29>(static_cast<uint32_t>(num)));
    case 30: return static_cast<T>(bit_reverse<30>(static_cast<uint32_t>(num)));
    case 31: return static_cast<T>(bit_reverse<31>(static_cast<uint32_t>(num)));
    case 32: return static_cast<T>(bit_reverse<32>(static_cast<uint32_t>(num)));
    default: ORT_THROW("Unsupported bit size.");
  }
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
static Status fft_radix2(OpKernelContext* /*ctx*/, size_t batch_idx,
    const Tensor* X, Tensor* Y, const Tensor* window, bool is_onesided, bool inverse,
    std::vector<std::complex<T>>& V,
    std::vector<std::complex<T>>& temp_output) {

  // Get shape and significant bits
  const auto& X_shape = X->Shape();
  size_t number_of_samples = static_cast<size_t>(X_shape[1]);
  unsigned significant_bits = static_cast<unsigned>(log2(number_of_samples));

  // Get data
  auto* X_data = const_cast<U*>(reinterpret_cast<const U*>(X->DataRaw())) + (batch_idx * number_of_samples);
  // Get window
  U* window_data = nullptr;
  if (window) {
    window_data = const_cast<U*>(reinterpret_cast<const U*>(window->DataRaw()));
  }

  std::complex<T>* Y_data;
  if (is_onesided) {
    if (temp_output.size() != number_of_samples) {
      temp_output = std::vector<std::complex<T>>(number_of_samples);
    }
    Y_data = temp_output.data();
  } else {
    Y_data = reinterpret_cast<std::complex<T>*>(Y->MutableDataRaw()) + (batch_idx * number_of_samples);
  }

  auto angular_velocity = compute_angular_velocity<T>(number_of_samples, inverse);

  // Create vandermonde matrix V ordered with the bit-reversed permutation
  if (V.size() != number_of_samples) {
    V = std::vector<std::complex<T>>(number_of_samples);  // e^(i *2*pi / N * k)
    for (size_t i = 0; i < number_of_samples; i++) {
      size_t bit_reversed_index = bit_reverse(i, significant_bits);
      V[bit_reversed_index] = std::complex<T>(cos(i * angular_velocity), sin(i * angular_velocity));
    }
  }

  for (size_t i = 0; i < number_of_samples; i++) {
    size_t bit_reversed_index = bit_reverse(i, significant_bits);
    auto x = *(X_data + bit_reversed_index);
    auto window_element = window_data ? *(window_data + bit_reversed_index) : 1; 
    *(Y_data + i) = std::complex<T>(1, 0) * x * window_element;
  }

  // Run fft_radix2
  unsigned current_significant_bits = 0;
  for (size_t i = 2; i <= number_of_samples; i <<= 1) {
    size_t midpoint = i >> 1;
    current_significant_bits++;

    for (size_t k = 0; k < midpoint; k++) {
      auto first_idx = bit_reverse(k, current_significant_bits);
      auto second_idx = bit_reverse(midpoint + k, current_significant_bits);
      for (size_t j = 0; j < number_of_samples; j += i) {
        std::complex<T>* even = (Y_data + j) + k;
        std::complex<T>* odd = (Y_data + j) + (midpoint + k);
        std::complex<T> first = *even + (V[first_idx] * *odd);
        std::complex<T> second = *even + (V[second_idx] * *odd);
        *even = first;
        *odd = second;
      }
    }
  }

  // Scale the output if inverse
  if (inverse) {
    for (size_t i = 0; i < number_of_samples; i++) {
      std::complex<T>& val = *(Y_data + i);
      val /= static_cast<T>(number_of_samples);
    }
  }

  if (is_onesided) {
    const auto& Y_shape = Y->Shape();
    size_t fft_output_size = static_cast<size_t>(Y_shape[1]);
    auto destination = reinterpret_cast<std::complex<T>*>(Y->MutableDataRaw()) + (batch_idx * fft_output_size);
    memcpy(destination, Y_data, sizeof(std::complex<T>) * fft_output_size);
  }

  return Status::OK();
}

template <typename T, typename U>
static Status dft_naive(size_t batch_idx, const Tensor* X, Tensor* Y, const Tensor* window, bool inverse) {
  // Get shape and significant bits
  const auto& X_shape = X->Shape();
  size_t number_of_samples = static_cast<size_t>(X_shape[1]);
  const auto& Y_shape = Y->Shape();
  size_t dft_output_size = static_cast<size_t>(Y_shape[1]);

  // Get data
  auto* X_data = const_cast<U*>(reinterpret_cast<const U*>(X->DataRaw())) + (batch_idx * number_of_samples);
  auto* Y_data = reinterpret_cast<std::complex<T>*>(Y->MutableDataRaw()) + (batch_idx * dft_output_size);
  
  U* window_data = nullptr;
  if (window) {
    window_data = const_cast<U*>(reinterpret_cast<const U*>(window->DataRaw()));
  }

  auto angular_velocity = compute_angular_velocity<T>(number_of_samples, inverse);

  for (size_t i = 0; i < dft_output_size; i++) {
    std::complex<T>& out = *(Y_data + i);
    out.real(0);
    out.imag(0);

    for (size_t j = 0; j < number_of_samples; j++) {  // vectorize over this loop
      auto exponential = std::complex<T>(cos(i * j * angular_velocity), sin(i * j * angular_velocity));
      auto window_element = window_data ? * (window_data + j) : 1;
      auto element = *(X_data + j) * window_element;
      out += exponential * element;
    }

    if (inverse) {
      out /= static_cast<T>(number_of_samples);
    }
  }

  return Status::OK();
}

template <typename T, typename U>
static Status discrete_fourier_transform(OpKernelContext* ctx, const Tensor* X, Tensor* Y, const Tensor* window, bool is_onesided, bool inverse,
                                         std::vector<std::complex<T>>& V, std::vector<std::complex<T>>& temp_output) {
  // Get shape
  const auto& X_shape = X->Shape();
  size_t number_of_batches = static_cast<size_t>(X_shape[0]);
  size_t number_of_samples = static_cast<size_t>(X_shape[1]);
   
  // radix 2 fft
  for (size_t i = 0; i < number_of_batches; i++) {
    if (is_power_of_2(number_of_samples)) {
      ORT_RETURN_IF_ERROR((fft_radix2<T, U>(ctx, i, X, Y, window, is_onesided, inverse, V, temp_output)));
    } else {
      ORT_RETURN_IF_ERROR((dft_naive<T, U>(i, X, Y, window, inverse)));
    }
  }

  return Status::OK();
}

static Status discrete_fourier_transform(OpKernelContext* ctx, bool is_onesided, bool inverse) {
  // Get input shape
  const auto* X = ctx->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  const auto is_real_valued = is_real_valued_signal(X_shape);
  const auto is_complex_valued = is_complex_valued_signal(X_shape);

  // Get the DFT output size. Onesided will return only the unique values!
  // note: x >> 1 === std::floor(x / 2.f)
  int64_t number_of_samples = static_cast<int64_t>(X_shape[1]);
  auto dft_output_size = is_onesided ?
      ((number_of_samples >> 1) + 1) :
      number_of_samples;

  // Get output shape
  auto Y_shape = onnxruntime::TensorShape({X_shape[0], dft_output_size, 2});
  auto Y = ctx->Output(0, Y_shape);

  // Get data type
  auto data_type = X->DataType();

  auto element_size = data_type->Size();
  if (element_size == sizeof(float)) {
    std::vector<std::complex<float>> V;
    std::vector<std::complex<float>> temp_output;
    if (is_real_valued) {
      ORT_RETURN_IF_ERROR((discrete_fourier_transform<float, float>(ctx, X, Y, nullptr, is_onesided, inverse, V, temp_output)));
    } else if (is_complex_valued) {
      ORT_RETURN_IF_ERROR((discrete_fourier_transform<float, std::complex<float>>(ctx, X, Y, nullptr, is_onesided, inverse, V, temp_output)));
    } else {
        ORT_THROW("Unsupported input signal shape. The signal's first dimenstion must be the batch dimension and its second dimension must be the signal length dimension. It may optionally include a 3rd dimension of size 2 for complex inputs.", data_type);
    }
  } else if (element_size == sizeof(double)) {
    std::vector<std::complex<double>> V;
    std::vector<std::complex<double>> temp_output;
    if (is_real_valued) {
      ORT_RETURN_IF_ERROR((discrete_fourier_transform<double, double>(ctx, X, Y, nullptr, is_onesided, inverse, V, temp_output)));
    } else if (is_complex_valued) {
      ORT_RETURN_IF_ERROR((discrete_fourier_transform<double, std::complex<double>>(ctx, X, Y, nullptr, is_onesided, inverse, V, temp_output)));
    } else {
      ORT_THROW("Unsupported input signal shape. The signal's first dimenstion must be the batch dimension and its second dimension must be the signal length dimension. It may optionally include a 3rd dimension of size 2 for complex inputs.", data_type);
    }
  } else {
    ORT_THROW("Unsupported input data type of ", data_type);
  }

  return Status::OK();
}

Status DFT::Compute(OpKernelContext* ctx) const {
  ORT_RETURN_IF_ERROR(discrete_fourier_transform(ctx, is_onesided_, false));
  return Status::OK();
}

Status IDFT::Compute(OpKernelContext* ctx) const {
  ORT_RETURN_IF_ERROR(discrete_fourier_transform(ctx, false, true));
  return Status::OK();
}

// dedupe with the other one in window_functions.cc
template <typename T>
static T get_scalar_value_from_tensor(const Tensor* tensor) {
  ORT_ENFORCE(tensor->Shape().Size() == 1, "ratio input should have a single value.");

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
  const auto* window = ctx->Input<Tensor>(1);
  const auto* frame_length_tensor = ctx->Input<Tensor>(2);
  const auto frame_step = get_scalar_value_from_tensor<int64_t>(ctx->Input<Tensor>(3));

  // Get input signal shape
  const auto& signal_shape = signal->Shape();
  const auto batch_size = signal_shape[0];
  const auto signal_size = signal_shape[1];
  const auto signal_components =
      signal_shape.NumDimensions() == 2 ? 1 : signal_shape.NumDimensions() == 3 ? signal_shape[2] : 0;  // error
  ORT_ENFORCE(signal_components == 1 || signal_components == 2, "Ensure that the signal has either 1 or 2 components.");

  // Get the frame length
  int64_t frame_length = std::numeric_limits<int64_t>::min();  
  if (frame_length_tensor) 
  {
    frame_length = get_scalar_value_from_tensor<int64_t>(frame_length_tensor);
  }

  // Get window length
  int64_t window_length = std::numeric_limits<int64_t>::min();
   if (window) {
    window_length = window->Shape()[0];
  }

  // The frame_length and window inputs are generally used interchangably, and should match!
  if (frame_length != std::numeric_limits<int64_t>::min() &&
      window_length != std::numeric_limits<int64_t>::min()) {
    ORT_ENFORCE(frame_length == window_length, "If both frame_length and window are set, then the size of the window must be equal to the frame_length.");
  }

  // Calculate the window size with preference to the window input.
  const auto window_size = window ? window->Shape()[0] : frame_length;
  ORT_ENFORCE(window_size < signal_size, "Ensure that the dft size is smaller than the signal.");

  // Calculate the number of dfts to run
  const auto n_dfts = static_cast<int64_t>(std::floor((signal_size - window_size) / static_cast<float>(frame_step)) + 1);

  // Calculate the output spectra length (onesided will return only the unique values)
  // note: x >> 1 === std::floor(x / 2.f)
  const auto dft_output_size =
      is_onesided ?
        (window_size >> 1) + 1 :
        window_size;

  // Get/create the output mutable data
  auto output_spectra_shape = onnxruntime::TensorShape({batch_size, n_dfts, dft_output_size, 2});
  auto Y = ctx->Output(0, output_spectra_shape);
  auto Y_data = reinterpret_cast<T*>(Y->MutableDataRaw());

  // Get/create the signal mutable data
  auto* signal_data = const_cast<U*>(reinterpret_cast<const U*>(signal->DataRaw()));

  // Define tensor shapes for each dft run
  const int64_t output_components = 2;
  auto dft_input_shape = onnxruntime::TensorShape({1, window_size, signal_components});
  auto dft_output_shape = onnxruntime::TensorShape({1, dft_output_size, output_components});

  std::vector<std::complex<T>> V;
  std::vector<std::complex<T>> temp_output;

  // Run each dft of each batch as if it was a real-valued batch size 1 dft operation
  for (int64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    for (int64_t i = 0; i < n_dfts; i++) {
      auto input_frame_begin =
        signal_data +
        (batch_idx * signal_size * signal_components) +
        (i * frame_step * signal_components);

      auto output_frame_begin =
        Y_data +
        (batch_idx * n_dfts * dft_output_size * output_components) +
        (i * dft_output_size * output_components);

      // Tensors do not own the backing memory, so no worries on destruction
      auto input =
          onnxruntime::Tensor(
              signal->DataType(),
              dft_input_shape,
              input_frame_begin,
              signal->Location(),
              0);

      auto output =
          onnxruntime::Tensor(
              Y->DataType(),
              dft_output_shape,
              output_frame_begin,
              Y->Location(),
              0);

      // Run individual dft
      ORT_RETURN_IF_ERROR((discrete_fourier_transform<T, U>(ctx, &input, &output, window, is_onesided, false, V, temp_output)));
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
      ORT_THROW("Unsupported input signal shape. The signal's first dimenstion must be the batch dimension and its second dimension must be the signal length dimension. It may optionally include a 3rd dimension of size 2 for complex inputs.", data_type);
    }
  } else if (element_size == sizeof(double)) {
    if (is_real_valued) {
      ORT_RETURN_IF_ERROR((short_time_fourier_transform<double, double>(ctx, is_onesided_, false)));
    } else if (is_complex_valued) {
      ORT_RETURN_IF_ERROR((short_time_fourier_transform<double, std::complex<double>>(ctx, is_onesided_, false)));
    } else {
      ORT_THROW("Unsupported input signal shape. The signal's first dimenstion must be the batch dimension and its second dimension must be the signal length dimension. It may optionally include a 3rd dimension of size 2 for complex inputs.", data_type);
    }
  } else {
    ORT_THROW("Unsupported input data type of ", data_type);
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif