// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include "core/util/math.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include <random>

namespace onnxruntime {
namespace test {

template <typename T>
class NhwcMaxPoolOpTester {
 private:
  std::default_random_engine generator_{1234};
  std::vector<T> X_data_;
  std::vector<int64_t> X_shape_;
  std::vector<int64_t> kernel_shape_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> dilations_;

  static size_t ShapeSize(const std::vector<int64_t>& shape) {
    return static_cast<size_t>(std::accumulate(shape.cbegin(), shape.cend(), 1LL, std::multiplies<int64_t>()));
  }

  static bool NextPosition(int64_t N, const int64_t* shape, int64_t* dims) {
    // Loop over spatial axes in reverse order to choose an index, like counting.
    bool incremented = false;
    for (int64_t d_i = N - 1; d_i >= 0; --d_i) {
      int64_t d_max = shape[d_i];
      ORT_ENFORCE(dims[d_i] < d_max);
      if (dims[d_i] == d_max - 1) {
        dims[d_i] = 0;
      } else {  // dims[d_i] < d_max - 1
        ++dims[d_i];
        incremented = true;
        break;
      }
    }
    return incremented;
  }

  void ComputeExpectedOutput(std::vector<T>& Y_data, std::vector<int64_t>& Y_shape) {
    ORT_ENFORCE(X_shape_.size() >= 2 && X_shape_.size() == kernel_shape_.size() + 2);

    const size_t kernel_rank = kernel_shape_.size();

    const int64_t batch_count = X_shape_[0];
    const int64_t channels = X_shape_[X_shape_.size() - 1];

    std::vector<int64_t> pads(pads_);
    if (pads.empty()) {
      pads.resize(kernel_rank * 2, 0);
    }
    std::vector<int64_t> dilations(dilations_);
    if (dilations.empty()) {
      dilations.resize(kernel_rank, 1);
    }
    std::vector<int64_t> strides(strides_);
    if (strides.empty()) {
      strides.resize(kernel_rank, 1);
    }

    const int64_t* input_shape = X_shape_.data() + 1;

    // Compute the expected shape of the output.
    Y_shape.reserve(kernel_rank + 2);
    Y_shape.push_back(batch_count);
    for (size_t n = 0; n < kernel_rank; n++) {
      Y_shape.push_back(((input_shape[n] + pads[n] + pads[kernel_rank + n]) -
                         (dilations[n] * (kernel_shape_[n] - 1) + 1)) / strides[n] + 1);
    }
    Y_shape.push_back(channels);
    Y_data.resize(ShapeSize(Y_shape));

    const int64_t* output_shape = Y_shape.data() + 1;

    const int64_t input_image_size = std::accumulate(
        input_shape, input_shape + kernel_rank, 1LL, std::multiplies<int64_t>());

    const T* Xdata = X_data_.data();
    T* Ydata = Y_data.data();

    for (int64_t batch = 0; batch < batch_count; batch++) {
      std::vector<int64_t> d_output(kernel_rank, 0);
      std::vector<int64_t> d_kernel(kernel_rank, 0);
      do {
        std::fill_n(Ydata, channels, static_cast<T>(0));
        do {
          int64_t input_offset = 0;
          bool is_padding = false;
          for (size_t axis = 0; axis < kernel_rank; ++axis) {
            int64_t input_dim = d_kernel[axis] * dilations[axis] + d_output[axis] * strides[axis] - pads[axis];
            is_padding |= !math::is_a_ge_zero_and_a_lt_b(input_dim, input_shape[axis]);
            input_offset *= input_shape[axis];
            input_offset += input_dim;
          }
          if (!is_padding) {
            const T* data_ptr = Xdata + input_offset * channels;
            for (int64_t c = 0; c < channels; c++) {
              Ydata[c] = std::max(Ydata[c], data_ptr[c]);
            }
          }
        } while (NextPosition(kernel_rank, kernel_shape_.data(), d_kernel.data()));
        Ydata += channels;
      } while (NextPosition(kernel_rank, output_shape, d_output.data()));
      Xdata += channels * input_image_size;
    }
  }

 public:
  NhwcMaxPoolOpTester() {
  }

  void GenerateRandomInput(const std::vector<int64_t>& shape) {
    std::uniform_int_distribution<int32_t> distribution(0, 255);
    size_t shape_size = ShapeSize(shape);
    X_data_.resize(shape_size);
    for (size_t n = 0; n < shape_size; n++) {
      X_data_[n] = static_cast<T>(distribution(generator_));
    }
    X_shape_ = shape;
  }

  void SetKernelShape(const std::vector<int64_t>& kernel_shape) {
    kernel_shape_ = kernel_shape;
  }

  void SetPads(const std::vector<int64_t>& pads) {
    pads_ = pads;
  }

  void SetStrides(const std::vector<int64_t>& strides) {
    strides_ = strides;
  }

  void SetDilations(const std::vector<int64_t>& dilations) {
    dilations_ = dilations;
  }

  void Run() {
    std::vector<T> Y_data;
    std::vector<int64_t> Y_shape;
    ComputeExpectedOutput(Y_data, Y_shape);

    OpTester test("NhwcMaxPool", 1, onnxruntime::kMSDomain);
    test.AddInput<T>("x", X_shape_, X_data_);
    test.AddOutput<T>("y", Y_shape, Y_data);
    test.AddAttribute("kernel_shape", kernel_shape_);
    if (!pads_.empty()) {
      test.AddAttribute("pads", pads_);
    }
    if (!strides_.empty()) {
      test.AddAttribute("strides", strides_);
    }
    if (!dilations_.empty()) {
      test.AddAttribute("dilations", dilations_);
    }
    test.Run(OpTester::ExpectResult::kExpectSuccess, "");
  }
};

TEST(NhwcMaxPoolContribOpTest, MaxPool1D) {
  for (int64_t channels = 1; channels < 64; channels++) {
    NhwcMaxPoolOpTester<uint8_t> test;
    test.GenerateRandomInput({1, 23, channels});
    test.SetKernelShape({5});
    test.SetPads({2, 2});
    test.Run();
  }
}

TEST(NhwcMaxPoolContribOpTest, MaxPool2D) {
  for (int64_t channels = 1; channels < 64; channels++) {
    NhwcMaxPoolOpTester<uint8_t> test;
    test.GenerateRandomInput({1, 15, 19, channels});
    test.SetKernelShape({3, 5});
    test.SetPads({1, 1, 1, 1});
    test.Run();
  }
}

TEST(NhwcMaxPoolContribOpTest, MaxPool3D) {
  for (int64_t channels = 1; channels < 64; channels++) {
    NhwcMaxPoolOpTester<uint8_t> test;
    test.GenerateRandomInput({1, 9, 13, 15, channels});
    test.SetKernelShape({2, 4, 6});
    test.SetPads({0, 0, 0, 1, 1, 1});
    test.Run();
  }
}

TEST(NhwcMaxPoolContribOpTest, MaxPoolStrides) {
  NhwcMaxPoolOpTester<uint8_t> test;
  test.GenerateRandomInput({4, 23, 19, 32});
  test.SetKernelShape({3, 3});
  test.SetStrides({2, 2});
  test.Run();
}

TEST(NhwcMaxPoolContribOpTest, MaxPoolDilations) {
  NhwcMaxPoolOpTester<uint8_t> test;
  test.GenerateRandomInput({4, 23, 19, 32});
  test.SetKernelShape({3, 3});
  test.SetDilations({2, 2});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
