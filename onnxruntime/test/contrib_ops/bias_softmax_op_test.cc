// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <numeric>

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/providers/provider_test_utils.h"

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
  std::vector<int64_t> bias_shape_;
  int64_t axis_;
  bool is_inner_broadcast_;
  bool use_float16_;

  std::vector<float> in_data_;
  std::vector<float> bias_data_;
  std::vector<float> out_data_;

  int64_t num_elements_;
  int64_t num_batches_;
  int64_t num_bias_batches_;

 public:
  BiasSoftmaxTester(std::vector<int64_t> in_shape, std::vector<int64_t> bias_shape, int64_t axis,
                    bool is_inner_broadcast, bool use_float16)
      : in_shape_(in_shape),
        bias_shape_(bias_shape),
        axis_(axis),
        is_inner_broadcast_(is_inner_broadcast),
        use_float16_(use_float16) {
    int64_t new_axis = axis_ < 0 ? axis_ + static_cast<int64_t>(in_shape.size()) : axis_;
    // softmax element count
    num_elements_ = std::accumulate(in_shape_.cbegin() + new_axis, in_shape_.cend(), 1LL, std::multiplies<int64_t>());

    // input batches
    num_batches_ = std::accumulate(in_shape_.cbegin(), in_shape_.cbegin() + new_axis, 1LL, std::multiplies<int64_t>());

    // bias batches
    num_bias_batches_ =
        std::accumulate(bias_shape_.cbegin(), bias_shape_.cend(), 1LL, std::multiplies<int64_t>()) / num_elements_;

    FillInputs();
    ComputeInternal();
  }

  void FillInputs() {
    srand(10);

    // Need to keep enough output values above OpTester threshold of 0.005
    // (Recall softmax normalizes outputs to sum to 1.000)
    auto allow_fill = [n = num_elements_](int64_t i) { return i < 50 || i >= n - 50; };

    in_data_.resize(static_cast<size_t>(num_batches_ * num_elements_));
    for (int64_t b = 0; b < num_batches_; b++) {
      for (int64_t i = 0; i < num_elements_; i++) {
        in_data_[b * num_elements_ + i] = allow_fill(i) ? -5.0f + 10.0f * ((float)rand() / float(RAND_MAX)) : -10000.0f;
      }
    }

    bias_data_.resize(static_cast<size_t>(num_bias_batches_ * num_elements_));
    for (int64_t b = 0; b < num_bias_batches_; b++) {
      for (int64_t i = 0; i < num_elements_; i++) {
        bias_data_[b * num_elements_ + i] = allow_fill(i) ? -5.0f + 10.0f * ((float)rand() / float(RAND_MAX)) : 0.0f;
      }
    }
  }

  void ComputeInternal() {
    out_data_.resize(in_data_.size());
    size_t rank = in_shape_.size();
    size_t offset = in_shape_.size() - bias_shape_.size();
    std::vector<int64_t> in_strides(rank);
    std::vector<int64_t> bias_strides(rank);
    in_strides[rank - 1] = bias_strides[rank - 1] = 1;
    if (rank > 1) {
      int64_t bias_stride = bias_shape_[rank - 1 - offset];
      for (size_t i = rank - 2;; --i) {
        in_strides[i] = in_shape_[i + 1] * in_strides[i + 1];
        bias_strides[i] = i < offset || in_shape_[i] != bias_shape_[i - offset] ? 0 : bias_stride;
        if (i >= offset) bias_stride *= bias_shape_[i - offset];
        if (i == 0) break;
      }
    }

    for (int64_t i = 0; i < static_cast<int64_t>(out_data_.size()); ++i) {
      int64_t bias_offset = 0;
      int64_t remain = i;
      for (size_t j = 0; j < rank; ++j) {
        int64_t q = remain / in_strides[j];
        bias_offset += q * bias_strides[j];
        remain = remain % in_strides[j];
      }
      out_data_[i] = in_data_[i] + bias_data_[bias_offset];
    }

    // for every batch in input
    for (int64_t batch = 0; batch < num_batches_; batch++) {
      // offset to batch in input and bias
      int64_t b = batch * num_elements_;

      // view into this batch
      auto out_s = out_data_.begin() + b;
      auto out_e = out_data_.begin() + b + num_elements_;

      // pick out maximum element in batch
      float max = *std::max_element(out_s, out_e);

      // do sum for normalization factor
      double sum =
          std::accumulate(out_s, out_e, 0.0, [max](double sum, float x) { return sum + exp(double(x) - max); });

      // do softmax
      std::transform(out_s, out_e, out_s, [max, sum](float x) { return exp(x - max) / float(sum); });
    }
  }

  void RunComparison() {
    // BiasSoftmax only implemented for cuda architecture
    int min_cuda_architecture = use_float16_ ? 530 : 0;
    if (HasCudaEnvironment(min_cuda_architecture) || kGpuExecutionProvider == kRocmExecutionProvider) {
      OpTester tester("BiasSoftmax", 1, onnxruntime::kMSDomain);
      tester.AddAttribute<int64_t>("axis", axis_);
      tester.AddAttribute<int64_t>("is_inner_broadcast", is_inner_broadcast_);

      if (use_float16_) {
        tester.AddInput<MLFloat16>("data", in_shape_, ToFloat16(in_data_));
        tester.AddInput<MLFloat16>("bias", bias_shape_, ToFloat16(bias_data_));
        tester.AddOutput<MLFloat16>("output", in_shape_, ToFloat16(out_data_));
      } else {
        tester.AddInput<float>("data", in_shape_, in_data_);
        tester.AddInput<float>("bias", bias_shape_, bias_data_);
        tester.AddOutput<float>("output", in_shape_, out_data_);
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

void RunBiasSoftmaxTests(std::vector<int64_t> in_shape, std::vector<int64_t> bias_shape, int64_t axis,
                         bool is_inner_broadcast) {
  BiasSoftmaxTester test_fp32(in_shape, bias_shape, axis, is_inner_broadcast, false);
  test_fp32.RunComparison();
  BiasSoftmaxTester test_fp16(in_shape, bias_shape, axis, is_inner_broadcast, true);
  test_fp16.RunComparison();
}

TEST(BiasSoftmaxTest, InnerBroadcast) { RunBiasSoftmaxTests({8, 4, 4, 2, 2, 8}, {8, 4, 1, 1, 2, 8}, 4, true); }

TEST(BiasSoftmaxTest, InnerBroadcastNegativeAxis) {
  RunBiasSoftmaxTests({8, 4, 4, 2, 2, 8}, {8, 1, 1, 2, 2, 8}, -3, true);
}

TEST(BiasSoftmaxTest, InnerBroadcastEmptyBiasBatch) { RunBiasSoftmaxTests({8, 4, 4, 2, 2, 8}, {1, 2, 8}, 4, true); }

TEST(BiasSoftmaxTest, InnerBroadcastFullBiasBatch) {
  RunBiasSoftmaxTests({8, 4, 4, 2, 2, 8}, {8, 4, 4, 2, 2, 8}, 4, true);
}

TEST(BiasSoftmaxTest, OuterBroadcast) { RunBiasSoftmaxTests({8, 4, 4, 2, 2, 8}, {1, 4, 2, 2, 8}, 4, false); }

TEST(BiasSoftmaxTest, OuterBroadcastNegativeAxis) {
  RunBiasSoftmaxTests({8, 4, 4, 2, 2, 8}, {4, 4, 2, 2, 8}, -3, false);
}

TEST(BiasSoftmaxTest, OuterBroadcastEmptyBiasBatch) { RunBiasSoftmaxTests({8, 4, 4, 2, 2, 8}, {1, 1, 2, 8}, 4, false); }

TEST(BiasSoftmaxTest, OuterBroadcastFullBiasBatch) {
  RunBiasSoftmaxTests({8, 4, 4, 2, 2, 8}, {8, 4, 4, 2, 2, 8}, 4, false);
}

// medium softmax batch tests kernel that computes on single SM
TEST(BiasSoftmaxTest, InnerBroadcastMediumBatch) { RunBiasSoftmaxTests({48, 16, 128, 32}, {48, 1, 1, 32}, 3, true); }

TEST(BiasSoftmaxTest, OuterBroadcastMediumBatch) { RunBiasSoftmaxTests({48, 16, 128, 32}, {16, 128, 32}, -1, false); }

// large softmax batch tests falls back to cuda DNN library
TEST(BiasSoftmaxTest, InnerBroadcastLargeBatch) { RunBiasSoftmaxTests({4, 2, 4096}, {4, 1, 4096}, 2, true); }

TEST(BiasSoftmaxTest, OuterBroadcastLargeBatch) { RunBiasSoftmaxTests({4, 2, 4096}, {2, 4096}, 2, false); }

}  // namespace test
}  // namespace onnxruntime
