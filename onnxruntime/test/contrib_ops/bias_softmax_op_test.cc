// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/providers/provider_test_utils.h"

#include <algorithm>
#include <numeric>

namespace onnxruntime {
namespace test {

#if USE_ROCM
constexpr const char* kGpuExecutionProvider = kRocmExecutionProvider;
#else
constexpr const char* kGpuExecutionProvider = kCudaExecutionProvider;
#endif

// followed example of fastgelu_op_test.cc
// in retrospect would have been better to compare BiasSoftmax to Add + Softmax graph

class BiasSoftmaxTester {
  std::vector<int64_t> in_shape_;
  int64_t broadcast_axis_;
  int64_t softmax_axis_;
  bool use_float16_;

  std::vector<float> in_data_;
  std::vector<float> bias_data_;
  std::vector<int64_t> bias_shape_;
  std::vector<float> out_data_;
  std::vector<int64_t> out_shape_;

  int64_t nelements_;
  int64_t nbatches_;
  int64_t broadcast_size_;

 public:
  BiasSoftmaxTester(
      std::vector<int64_t> in_shape,
      std::vector<int64_t> bias_shape,
      int64_t broadcast_axis,
      int64_t softmax_axis,
      bool use_float16) : in_shape_(in_shape),
                          broadcast_axis_(broadcast_axis),
                          softmax_axis_(softmax_axis),
                          use_float16_(use_float16),
                          bias_shape_(bias_shape) {
    // auto-set bias shape if not provided
    if (bias_shape.size() == 0) {
      bias_shape_ = in_shape_;
      for (int64_t i = broadcast_axis; i < softmax_axis; i++)
        bias_shape_[i] = 1;
    }

    // softmax element count
    nelements_ = std::accumulate(
        in_shape_.cbegin() + softmax_axis_,
        in_shape_.cend(),
        1LL, std::multiplies<int64_t>());

    // input batches
    nbatches_ = std::accumulate(
        in_shape_.cbegin(),
        in_shape_.cbegin() + softmax_axis,
        1LL, std::multiplies<int64_t>());

    // bias broadcast repeat count
    // broadcast is along dimensions [broadcast_axis, softmax_axis)
    broadcast_size_ = std::accumulate(
        in_shape_.cbegin() + broadcast_axis_,
        in_shape_.cbegin() + softmax_axis_,
        1LL, std::multiplies<int64_t>());

    FillInputs();
    ComputeInternal();
  }

  void FillInputs() {
    srand(10);

    // Need to keep enough output values above OpTester threshold of 0.005
    // (Recall softmax normalizes outputs to sum to 1.000)
    auto allow_fill = [n = nelements_](int64_t i) {
      return i < 50 || i >= n - 50;
    };

    size_t len = nelements_ * nbatches_;
    in_data_.resize(len);
    for (int64_t b = 0; b < nbatches_; b++)
      for (int64_t i = 0; i < nelements_; i++)
        in_data_[i] = allow_fill(i) ? -5.0f + 10.0f * ((float)rand() / float(RAND_MAX)) : -10000.0f;

    len = nelements_ * nbatches_ / broadcast_size_;
    bias_data_.resize(len);
    for (int64_t b = 0; b < nbatches_ / broadcast_size_; b++)
      for (int64_t i = 0; i < nelements_; i++)
        bias_data_[i] = allow_fill(i) ? -5.0f + 10.0f * ((float)rand() / float(RAND_MAX)) : 0.0f;
  }

  void ComputeInternal() {
    out_data_.resize(in_data_.size());
    out_shape_ = in_shape_;

    // for every batch in input
    int64_t B = nbatches_, N = nelements_;
    for (int64_t batch = 0; batch < B; batch++) {
      // offset to batch in input and bias
      int64_t b = batch * N;
      int64_t c = batch / broadcast_size_ * N;

      // add bias to input
      for (int64_t i = 0; i < N; i++) {
        out_data_[b + i] = in_data_[b + i] + bias_data_[c + i];
      }

      // view into this batch
      auto out_s = out_data_.begin() + b;
      auto out_e = out_data_.begin() + b + N;

      // pick out maximum element in batch
      float max = *std::max_element(out_s, out_e);

      // do sum for normalization factor
      double sum = std::accumulate(out_s, out_e, 0.0,
                                   [max](double sum, float x) { return sum + exp(double(x) - max); });

      // do softmax
      std::transform(out_s, out_e, out_s,
                     [max, sum](float x) { return exp(x - max) / float(sum); });
    }
  }

  void RunComparison() {
    // BiasSoftmax only implemented for cuda architecture
    int min_cuda_architecture = use_float16_ ? 530 : 0;
    if (HasCudaEnvironment(min_cuda_architecture) ||
        kGpuExecutionProvider == kRocmExecutionProvider) {
      OpTester tester("BiasSoftmax", 1, onnxruntime::kMSDomain);
      tester.AddAttribute<int64_t>("softmax_axis", softmax_axis_);
      tester.AddAttribute<int64_t>("broadcast_axis", broadcast_axis_);

      if (use_float16_) {
        tester.AddInput<MLFloat16>("data", in_shape_, ToFloat16(in_data_));
        tester.AddInput<MLFloat16>("bias", bias_shape_, ToFloat16(bias_data_));
        tester.AddOutput<MLFloat16>("output", out_shape_, ToFloat16(out_data_));
      } else {
        tester.AddInput<float>("data", in_shape_, in_data_);
        tester.AddInput<float>("bias", bias_shape_, bias_data_);
        tester.AddOutput<float>("output", out_shape_, out_data_);
      }

      std::vector<std::unique_ptr<IExecutionProvider>> ep;
      #ifdef USE_CUDA
        ep.push_back(DefaultCudaExecutionProvider());
      #elif USE_ROCM
        ep.push_back(DefaultRocmExecutionProvider());
      #endif
      
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &ep);
    }
  }
};

// broadcast is along dimensions [broadcast_axis, softmax_axis)
TEST(BiasSoftmaxTest, BiasSoftmaxExtendedShapeFloat32) {
  BiasSoftmaxTester test({8, 4, 4, 2, 2, 8}, {}, 2, 4, false);
  test.RunComparison();
}

TEST(BiasSoftmaxTest, BiasSoftmaxMismatchedShapeFloat32) {
  BiasSoftmaxTester test({8, 4, 4, 2, 2, 8}, {1, 2, 8}, 0, 4, false);
  test.RunComparison();
}

// medium softmax batch tests kernel that computes on single SM
TEST(BiasSoftmaxTest, BiasSoftmaxMediumBatchFloat32) {
  BiasSoftmaxTester test({48, 16, 128, 32}, {}, 1, 3, false);
  test.RunComparison();
}

TEST(BiasSoftmaxTest, BiasSoftmaxMediumBatchFloat16) {
  BiasSoftmaxTester test({48, 16, 128, 32}, {}, 1, 3, true);
  test.RunComparison();
}

// large softmax batch tests falls back to cuda DNN library
TEST(BiasSoftmaxTest, BiasSoftmaxLargeBatchFloat32) {
  BiasSoftmaxTester test({4, 2, 4096}, {}, 1, 2, false);
  test.RunComparison();
}

TEST(BiasSoftmaxTest, BiasSoftmaxLargeBatchFloat16) {
  BiasSoftmaxTester test({4, 2, 4096}, {}, 1, 2, true);
  test.RunComparison();
}

}  // namespace test
}  // namespace onnxruntime
