// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/Map.h"
#include "fft.h"
#include <functional>

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

template <typename T>
struct complex {
    T real;
    T imaginary;
};

template <typename T = float>
auto fft_internal(
    const samples_container& samples,
    complex<T>* x_begin) {

    std::vector<complex<T>> output(samples.size());

    if (samples.size() == 1)
    {
        output[0].real = samples.at<T>(0) * x_begin[0].real;
        output[0].imaginary = samples.at<T>(0) * x_begin[0].imaginary;
        return output;
    }

    auto next_number_of_samples = samples.size() / 2;
    auto evens = samples.evens();
    auto odds = samples.odds();
    
    auto evens_output = fft_internal<T>(evens, x_begin);
    auto odds_output = fft_internal<T>(odds, x_begin);

    auto output_index = 0;
    auto odds_it = odds_output.begin();
    auto evens_it = evens_output.begin();
    while (evens_it != evens_output.end())
    {
        output[output_index].real = evens_it->real + (x_begin[output_index].real * odds_it->real - x_begin[output_index].imaginary * odds_it->imaginary);
        output[output_index].imaginary = evens_it->imaginary + (x_begin[output_index].real * odds_it->imaginary + x_begin[output_index].imaginary * odds_it->real);
        output_index++;

        output[output_index].real = evens_it->real + (x_begin[output_index].real * odds_it->real - x_begin[output_index].imaginary * odds_it->imaginary);
        output[output_index].imaginary = evens_it->imaginary + (x_begin[output_index].real * odds_it->imaginary + x_begin[output_index].imaginary * odds_it->real);
        output_index++;

        evens_it++;
        odds_it++;
    }

    return output;
}

template <typename T>
auto fft(const samples_container& samples) {

    auto number_of_samples = samples.size();
    auto x = std::vector<complex<T>>(number_of_samples); // e^(i *2*pi / N * k)
    x[0].real = 1;
    x[0].imaginary = 0;

    double nth_root_level = log2(number_of_samples);

    size_t index = 1;
    for (size_t level = 1; level < nth_root_level + 1; level++)
    {
        auto num_in_level = pow(2, level);
        for (size_t k = 0; k < num_in_level; k++)
        {
            auto tau_over_N = 3.14159265 * 2 * k / num_in_level;
            if (k % 2 != 0)
            {
                x[index].real = static_cast<T>(cos(tau_over_N));
                x[index].imaginary = static_cast<T>(-sin(tau_over_N));
                index++;
            }
        }
    }

    return fft_internal<T>(samples, x.data());
}

template <typename T>
static Status FftImpl(const Tensor* X, Tensor* Y) {
  const auto& X_shape = X->Shape();
  auto* X_data = const_cast<T*>(reinterpret_cast<const T*>(X->DataRaw()));
  auto* Y_data = reinterpret_cast<T*>(Y->MutableDataRaw());

  const auto samples = make_fft_samples_container(X_data, static_cast<uint32_t>(X_shape[0]));
  auto output = fft<T>(samples);
  memcpy(Y_data, output.data(), output.size() * 2 * sizeof(T));

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
      status = FftImpl<float>(X, Y);
      break;
    case sizeof(double):
      status = FftImpl<double>(X, Y);
      break;
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
  return status;
}

}  // namespace contrib
}  // namespace onnxruntime
