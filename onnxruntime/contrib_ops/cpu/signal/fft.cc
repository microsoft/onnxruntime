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

template <typename T>
struct output_container {
  std::complex<T>* data_;
  std::vector<unsigned> mapping_;

  output_container(size_t size, std::complex<T>* data) : mapping_(size)
  {
    data_ = data;
  }

  std::complex<T>& operator[](size_t i) {
    std::complex<T>* value = (data_ + mapping_[i]);
    return *value;
  }
};

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
    std::complex<T>* x_begin,
    output_container<T>& optimized_container,
    size_t begin,
    size_t end) {

    if (samples.size() == 1)
    {
      optimized_container[begin] = x_begin[0] * samples.at<U>(0);
      return;
    }

    auto next_number_of_samples = samples.size() / 2;
    auto evens = samples.evens();
    auto odds = samples.odds();
    
    size_t midpoint = static_cast<size_t>((end + begin) / 2);
    fft_internal<T, U>(evens, x_begin, optimized_container, begin, midpoint);
    fft_internal<T, U>(odds, x_begin, optimized_container, midpoint, end);

    for (size_t index = begin; index < midpoint; index++) {
      auto i = index - begin;

      auto even = optimized_container[begin + i];
      auto odd = optimized_container[midpoint + i];

      auto first = begin + (2 * i);
      auto second = begin + (2 * i) + 1;
      
      auto first_val = even + (x_begin[(2*i)] * odd);
      optimized_container[first]  = even + (x_begin[(2*i)] * odd);

      auto second_val = even + (x_begin[(2*i)+1] * odd);
      optimized_container[second] = even + (x_begin[(2*i)+1] * odd);
    }
}

template <typename T, typename U>
void fft(const samples_container& samples, std::complex<T>* output) {
    size_t number_of_samples = samples.size();

    float increment = 2.f / number_of_samples;  // in pi radians
    
    output_container<T> optimized_output(number_of_samples, output);
    optimized_output.mapping_[0] = 0;
    optimized_output.mapping_[1] = static_cast<unsigned>(1 / increment);

    auto x = std::vector<std::complex<T>>(number_of_samples); // e^(i *2*pi / N * k)
    x[0].real(1);     x[0].imag(0);
    x[1].real(-1);    x[1].imag(0);

    auto angles = std::vector<float>(number_of_samples);
    angles[0] = 0;  // 0pi
    angles[1] = 1;  // 1pi

    float pi = -3.14159265;
    for (unsigned i = 1; ((i * 2) + 1) <= number_of_samples; i++) {
      unsigned half_angle_index = i * 2;
      unsigned half_angle_plus_pi_index = half_angle_index + 1;

      auto& angle_in_radians = angles[i];
      auto half_angle = angle_in_radians / 2;
      auto reflected_half_angle = half_angle + 1;

      angles[half_angle_index] = half_angle;
      angles[half_angle_plus_pi_index] = reflected_half_angle;

      x[half_angle_index].real(static_cast<T>(cos(half_angle * pi)));
      x[half_angle_index].imag(static_cast<T>(sin(half_angle * pi)));
      optimized_output.mapping_[half_angle_index] = static_cast<unsigned>(half_angle / increment);

      x[half_angle_plus_pi_index].real(static_cast<T>(cos(reflected_half_angle * pi)));
      x[half_angle_plus_pi_index].imag(static_cast<T>(sin(reflected_half_angle * pi)));
      optimized_output.mapping_[half_angle_plus_pi_index] = static_cast<unsigned>(reflected_half_angle / increment);
    }

    fft_internal<T, U>(samples, x.data(), optimized_output, 0, number_of_samples);
}

template <typename T, typename U>
static Status FftImpl(const Tensor* X, Tensor* Y) {
  const auto& X_shape = X->Shape();
  auto* X_data = const_cast<U*>(reinterpret_cast<const U*>(X->DataRaw()));
  auto* Y_data = reinterpret_cast<std::complex<T>*>(Y->MutableDataRaw());


  const auto samples = make_fft_samples_container(X_data, static_cast<uint32_t>(X_shape[0]));
  fft<T, U>(samples, Y_data);

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
        status = FftImpl<float, float>(X, Y);
      } else if (X_shape.NumDimensions() == 2 && X_shape[1] == 2) {
        status = FftImpl<float, std::complex<float>>(X, Y);
      }
      break;
    case sizeof(double):
      if (X_shape.NumDimensions() == 1) {
        status = FftImpl<double, double>(X, Y);
      } else if (X_shape.NumDimensions() == 2 && X_shape[1] == 2) {
        status = FftImpl<double, std::complex<double>>(X, Y);
      }
      break;
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
  return status;
}

}  // namespace contrib
}  // namespace onnxruntime
