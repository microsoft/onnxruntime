// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/Map.h"
#include "fft.h"
#include <functional>

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

auto next_power_of_2(uint32_t input)
{
    input--;
    input |= input >> 1;
    input |= input >> 2;
    input |= input >> 4;
    input |= input >> 8;
    input |= input >> 16;
    return input + 1;
}

struct samples_container {
    uint8_t* data_;
    size_t length_;
    size_t max_length_;
    size_t element_length_;

    template <typename T>
    samples_container(T* data, size_t length, size_t max_length) :
        data_(reinterpret_cast<uint8_t*>(data)),
        length_(length),
        max_length_(max_length),
        element_length_(sizeof(T))
    {}
    
    template <typename T>
    samples_container(T* data, size_t length, size_t max_length, size_t element_length) :
        data_(reinterpret_cast<uint8_t*>(data)),
        length_(length),
        max_length_(max_length),
        element_length_(element_length)
    {}

    template <typename T>
    const T at(size_t index) const
    {
        if (index < length_)
        {
            return *reinterpret_cast<T*>(data_ + index * element_length_);
        }
        return T{ 0 };
    }

    auto size() const { return max_length_; }

    auto evens() const {
        return samples_container(data_, static_cast<size_t>(ceil(length_ / 2.f)), static_cast<size_t>(ceil(max_length_ / 2.f)), element_length_ * 2);
    }

    auto odds() const {
        if (max_length_ > 1)
        {
            return samples_container(data_ + element_length_, static_cast<size_t>(ceil((length_ - 1) / 2.f)), static_cast<size_t>(ceil((max_length_ - 1) / 2.f)), element_length_ * 2);
        }
        else
        {
            return samples_container(data_ + element_length_, 0, 0, element_length_ * 2);
        }
    }

    auto begin() { return iterator(*this); }
    auto end() { return iterator(*this).end(); }

    struct iterator {
        samples_container& container_;
        size_t read_ = 0;

        iterator& end() {
            read_ = container_.size();
            return *this;
        }

        iterator(samples_container& container) :
            container_(container) {}

        iterator operator++() {
            read_++;
            return *this;
        }

        iterator operator+=(int step) {
            read_+=step;
            return *this;
        }

        iterator operator+(int step) {
            return operator+=(step);
        }

        bool operator==(iterator other) {
            return this->read_ == other.read_;
        }

        bool operator!=(iterator other) {
            return !(this->operator==(other));
        }

        template <typename T>
        const T get() {
            return container_.at<T>(read_);
        }
    };
};

template <typename T>
auto make_fft_samples_container(T* data, size_t length) {
    auto power_of_2 = next_power_of_2(length);
    return samples_container(data, length, power_of_2);
}

template <typename T = float, typename U = float>
auto fft_internal(
    const samples_container& samples,
    std::vector<std::complex<T>>& x,
    std::complex<T>* output,
    size_t size) {

    if (samples.size() == 1)
    {
      *(output) = x[0] * samples.at<U>(0);
      return;
    }

    auto evens = samples.evens();
    auto odds = samples.odds();
    
    int midpoint = static_cast<size_t>((size) / 2);
    fft_internal<T, U>(evens, x, output, midpoint);
    fft_internal<T, U>(odds, x, output + midpoint, midpoint);

    unsigned significant_bits = static_cast<unsigned>(log2(size));

    for (size_t i = 0; i < midpoint; i++) {
      std::complex<T>* even = output + i;
      std::complex<T>* odd = output + (midpoint + i);
      std::complex<T> first = *even + (x[bit_reverse(i, significant_bits)] * *odd);
      std::complex<T> second = *even + (x[bit_reverse(midpoint + i, significant_bits)] * *odd);
      *even = first;
      *odd = second;
    }
}

unsigned bit_reverse(unsigned num, unsigned significant_bits) {
  unsigned output = 0;
  for (unsigned i = 0; i < significant_bits; i++) {

      output += ((num >> i) & 1) << (significant_bits - 1 - i);
  }
  return output;
}

template <typename T, typename U>
void fft(const samples_container& samples, std::complex<T>* output, bool inverse) {
  static const double pi = 3.14159265;
  static const double tau = 2 * pi;

  size_t number_of_samples = samples.size();
  auto x = std::vector<std::complex<T>>(number_of_samples); // e^(i *2*pi / N * k)

  T inverse_switch = inverse ? 1.f : -1.f;
  T increment_in_radians = inverse_switch * tau / number_of_samples;
  unsigned significant_bits = static_cast<unsigned>(log2(number_of_samples));
  for (unsigned i = 0; i < number_of_samples; i++) {
    unsigned bit_reversed_index = bit_reverse(i, significant_bits);
    x[bit_reversed_index] = std::complex<T>(cos(i * increment_in_radians), sin(i * increment_in_radians));
  }

  fft_internal<T, U>(samples, x, output, number_of_samples);
}

template <typename T, typename U>
static Status FftImpl(const Tensor* X, Tensor* Y, bool inverse) {
  const auto& X_shape = X->Shape();
  auto* X_data = const_cast<U*>(reinterpret_cast<const U*>(X->DataRaw()));
  auto* Y_data = reinterpret_cast<std::complex<T>*>(Y->MutableDataRaw());


  const auto samples = make_fft_samples_container(X_data, static_cast<uint32_t>(X_shape[0]));
  fft<T, U>(samples, Y_data, inverse);

  if (inverse) {
    for (int i = 0; i < samples.size(); i++) {
      std::complex<T>& val = *(Y_data + i);
      val /= static_cast<T>(samples.size());
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
        status = FftImpl<float, float>(X, Y, false);
      } else if (X_shape.NumDimensions() == 2 && X_shape[1] == 2) {
        status = FftImpl<float, std::complex<float>>(X, Y, false);
      }
      break;
    case sizeof(double):
      if (X_shape.NumDimensions() == 1) {
        status = FftImpl<double, double>(X, Y, false);
      } else if (X_shape.NumDimensions() == 2 && X_shape[1] == 2) {
        status = FftImpl<double, std::complex<double>>(X, Y, false);
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
        status = FftImpl<float, float>(X, Y, true);
      } else if (X_shape.NumDimensions() == 2 && X_shape[1] == 2) {
        status = FftImpl<float, std::complex<float>>(X, Y, true);
      }
      break;
    case sizeof(double):
      if (X_shape.NumDimensions() == 1) {
        status = FftImpl<double, double>(X, Y, true);
      } else if (X_shape.NumDimensions() == 2 && X_shape[1] == 2) {
        status = FftImpl<double, std::complex<double>>(X, Y, true);
      }
      break;
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
  return status;
}

}  // namespace contrib
}  // namespace onnxruntime
