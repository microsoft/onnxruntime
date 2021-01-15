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

template <unsigned TSignificantBits>
static inline size_t bit_reverse(size_t num) {
  size_t output = 0;
  size_t i = 0;
  do {
    output <<= 1;
    output |= (num & 1);
    num >>= 1;
    i++;
  } while (i < TSignificantBits);
  return output;
}

static inline size_t bit_reverse(size_t num, unsigned significant_bits) {
  size_t output = 0;
  switch (significant_bits) {
    case 0: return bit_reverse<0>(num);
    case 1: return bit_reverse<1>(num);
    case 2: return bit_reverse<2>(num);
    case 3: return bit_reverse<3>(num);
    case 4: return bit_reverse<4>(num);
    case 5: return bit_reverse<5>(num);
    case 6: return bit_reverse<6>(num);
    case 7: return bit_reverse<7>(num);
    case 8: return bit_reverse<8>(num);
    case 9: return bit_reverse<9>(num);
    case 10: return bit_reverse<10>(num);
    case 11: return bit_reverse<11>(num);
    case 12: return bit_reverse<12>(num);
    case 13: return bit_reverse<13>(num);
    case 14: return bit_reverse<14>(num);
    case 15: return bit_reverse<15>(num);
    case 16: return bit_reverse<16>(num);
    case 17: return bit_reverse<17>(num);
    case 18: return bit_reverse<18>(num);
    case 19: return bit_reverse<19>(num);
    case 20: return bit_reverse<20>(num);
    case 21: return bit_reverse<21>(num);
    case 22: return bit_reverse<22>(num);
    case 23: return bit_reverse<23>(num);
    case 24: return bit_reverse<24>(num);
    case 25: return bit_reverse<25>(num);
    case 26: return bit_reverse<26>(num);
    case 27: return bit_reverse<27>(num);
    case 28: return bit_reverse<28>(num);
    case 29: return bit_reverse<29>(num);
    case 30: return bit_reverse<30>(num);
    case 31: return bit_reverse<31>(num);
    case 32: return bit_reverse<32>(num);
    default: ORT_THROW("Unsupported bit size.");
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
    for (int i = 0; i < number_of_samples; i++) {
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

  for (int i = 0; i < dft_output_size; i++) {
    std::complex<T>& out = *(Y_data + i);
    out.real(0);
    out.imag(0);

    for (int j = 0; j < number_of_samples; j++) {  // vectorize over this loop
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
  auto start = std::chrono::high_resolution_clock::now();
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
  const auto n_dfts = static_cast<int64_t>(std::ceil((signal_size - window_size) / static_cast<float>(frame_step)));

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

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> evaluate_duration_in_microseconds = end - start;
  printf("\nSpectrogram evaluate took: %f\n", evaluate_duration_in_microseconds.count());
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
