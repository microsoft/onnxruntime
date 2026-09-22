// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/*++

Module Name:

    test_nchwc_depthwise_sliding.cpp

Abstract:

    Tests that the sliding window AVX-512 NCHWc depthwise kernel
    (MLAS_BACKEND_KERNEL_SELECTOR_CONFIG::nchwc_depthwise_sliding_kernel) is
    bitwise identical to the assembly kernel, through MlasNchwcConv, across
    kernel sizes, padding, spatial shapes, strides/dilations (forwarded to the
    assembly kernel), Sum accumulation, bias, activations and non-finite
    operands.

--*/

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

#include "test_util.h"

namespace {

void FillHostile(std::vector<float>& v, size_t count, std::mt19937& rng, bool non_finite) {
  std::uniform_int_distribution<int> kind(0, 19);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  v.resize(count);
  for (size_t i = 0; i < count; i++) {
    float value;
    switch (kind(rng)) {
      case 0:
        value = 0.0f;
        break;
      case 1:
        value = -0.0f;
        break;
      case 2:
        value = normal(rng) * 1.0e3f;
        break;
      case 3:
        value = normal(rng) * 1.0e-5f;
        break;
      case 4:
        value = non_finite ? std::numeric_limits<float>::infinity() : 1.0f;
        break;
      default:
        value = normal(rng);
        break;
    }
    v[i] = ((i & 1) != 0) ? -value : value;
  }
}

struct DepthwiseCase {
  size_t batch;
  size_t channels;  // multiple of the block size
  size_t height;
  size_t width;
  size_t kh;
  size_t kw;
  size_t pad_top, pad_left, pad_bottom, pad_right;
  size_t stride;
  size_t dilation;
  bool bias;
  MLAS_ACTIVATION_KIND activation;
  bool sum;         // accumulate into the existing output (ZeroMode = false)
  bool non_finite;  // Inf in filters (skipped padding taps must not become NaN)
};

std::string Describe(const DepthwiseCase& c, size_t threads) {
  std::ostringstream os;
  os << "N=" << c.batch << " C=" << c.channels << " H=" << c.height << " W=" << c.width << " K=" << c.kh << "x" << c.kw
     << " pads=" << c.pad_top << "," << c.pad_left << "," << c.pad_bottom << "," << c.pad_right
     << " stride=" << c.stride << " dilation=" << c.dilation << " bias=" << c.bias
     << " act=" << int(c.activation) << " sum=" << c.sum << " nonfinite=" << c.non_finite << " threads=" << threads;
  return os.str();
}

std::vector<float> RunConv(const DepthwiseCase& c, const std::vector<float>& input, const std::vector<float>& filter,
                       const std::vector<float>& bias, const std::vector<float>& initial_output,
                       MLAS_THREADPOOL* tp, bool sliding) {
  const int64_t oh = (int64_t(c.height + c.pad_top + c.pad_bottom) - int64_t(c.dilation * (c.kh - 1) + 1)) /
                         int64_t(c.stride) + 1;
  const int64_t ow = (int64_t(c.width + c.pad_left + c.pad_right) - int64_t(c.dilation * (c.kw - 1) + 1)) /
                         int64_t(c.stride) + 1;
  const int64_t input_shape[] = {int64_t(c.batch), int64_t(c.channels), int64_t(c.height), int64_t(c.width)};
  const int64_t output_shape[] = {int64_t(c.batch), int64_t(c.channels), oh, ow};
  const int64_t kernel[] = {int64_t(c.kh), int64_t(c.kw)};
  const int64_t dilation[] = {int64_t(c.dilation), int64_t(c.dilation)};
  const int64_t pads[] = {int64_t(c.pad_top), int64_t(c.pad_left), int64_t(c.pad_bottom), int64_t(c.pad_right)};
  const int64_t stride[] = {int64_t(c.stride), int64_t(c.stride)};

  MLAS_ACTIVATION activation;
  activation.ActivationKind = c.activation;
  if (c.activation == MlasLeakyReluActivation) {
    activation.Parameters.LeakyRelu.alpha = 0.125f;
  }

  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG cfg;
  cfg.nchwc_depthwise_sliding_kernel = sliding;

  std::vector<float> output = initial_output;
  output.resize(size_t(c.batch * c.channels * oh * ow));
  MlasNchwcConv(input_shape, kernel, dilation, pads, stride, output_shape, c.channels, input.data(), filter.data(),
                c.bias ? bias.data() : nullptr, output.data(), &activation, !c.sum, tp, &cfg, false);
  return output;
}

void ExpectBitwiseEqual(const std::vector<float>& expected, const std::vector<float>& actual,
                        const std::string& context) {
  ASSERT_EQ(expected.size(), actual.size()) << context;
  if (std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0) {
    return;
  }
  size_t mismatches = 0;
  size_t first = expected.size();
  for (size_t i = 0; i < expected.size(); i++) {
    if (std::memcmp(&expected[i], &actual[i], sizeof(float)) != 0) {
      if (first == expected.size()) {
        first = i;
      }
      mismatches++;
    }
  }
  FAIL() << context << ": " << mismatches << " of " << expected.size() << " elements differ; first at " << first
         << " expected " << expected[first] << " actual " << actual[first];
}

class NchwcDepthwiseSlidingTest : public testing::Test {
 protected:
  void SetUp() override {
    block_size_ = MlasNchwcGetBlockSize();
    if (block_size_ <= 1) {
      GTEST_SKIP() << "NCHWc is not supported on this platform";
    }
  }

  void Check(const DepthwiseCase& c, uint32_t seed, size_t threads = 1) {
    std::mt19937 rng(seed);
    std::vector<float> input, filter, bias, initial;
    FillHostile(input, c.batch * c.channels * c.height * c.width, rng, false);
    FillHostile(filter, c.channels * c.kh * c.kw, rng, c.non_finite);
    FillHostile(bias, c.channels, rng, false);
    // Large enough for any output shape; Run trims it.
    FillHostile(initial, c.batch * c.channels * (c.height + 16) * (c.width + 16), rng, false);

#if !defined(BUILD_MLAS_NO_ONNXRUNTIME)
    std::unique_ptr<onnxruntime::concurrency::ThreadPool> pool;
    if (threads > 1) {
      pool = std::make_unique<onnxruntime::concurrency::ThreadPool>(
          &onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), nullptr, int(threads), true);
    }
    MLAS_THREADPOOL* tp = pool.get();
#else
    MLAS_THREADPOOL* tp = nullptr;
#endif

    const std::vector<float> expected = RunConv(c, input, filter, bias, initial, tp, false);
    const std::vector<float> actual = RunConv(c, input, filter, bias, initial, tp, true);
    ExpectBitwiseEqual(expected, actual, Describe(c, threads));
  }

  size_t block_size_ = 1;
};

TEST_F(NchwcDepthwiseSlidingTest, SamePaddingShapes) {
  const size_t b = block_size_;
  uint32_t seed = 1;
  const size_t sides[][2] = {{1, 1}, {1, 2}, {2, 1}, {3, 3}, {5, 4}, {7, 7}, {8, 8}, {9, 13},
                             {13, 9}, {16, 16}, {17, 23}, {32, 32}, {64, 64}, {3, 70}};
  for (size_t k : {3, 5, 7}) {
    for (const auto& hw : sides) {
      for (bool bias : {false, true}) {
        Check({1, 2 * b, hw[0], hw[1], k, k, k / 2, k / 2, k / 2, k / 2, 1, 1, bias, MlasIdentityActivation,
               false, false},
              seed++);
        if (HasFailure()) return;
      }
    }
  }
}

TEST_F(NchwcDepthwiseSlidingTest, AsymmetricAndNoPadding) {
  const size_t b = block_size_;
  uint32_t seed = 100;
  const size_t pads[][4] = {{0, 0, 0, 0}, {1, 0, 0, 1}, {0, 2, 1, 0}, {3, 1, 0, 2}, {2, 3, 2, 3}, {0, 6, 0, 6}};
  for (size_t k : {3, 5, 7}) {
    for (const auto& p : pads) {
      for (size_t w : {7, 8, 11, 20, 31}) {
        Check({1, b, 9, w, k, k, p[0], p[1], p[2], p[3], 1, 1, true, MlasIdentityActivation, false, false}, seed++);
        if (HasFailure()) return;
      }
    }
  }
}

TEST_F(NchwcDepthwiseSlidingTest, NonSquareAndForwardedKernels) {
  const size_t b = block_size_;
  uint32_t seed = 200;
  // Widths other than 3/5/7, strides and dilations are forwarded to the assembly kernel.
  const size_t kernels[][2] = {{7, 3}, {3, 7}, {1, 5}, {5, 1}, {1, 1}, {2, 2}, {4, 4}, {9, 9}, {1, 3}};
  for (const auto& k : kernels) {
    for (size_t stride : {1, 2}) {
      for (size_t dilation : {1, 2}) {
        Check({2, 3 * b, 12, 19, k[0], k[1], k[0] / 2, k[1] / 2, k[0] / 2, k[1] / 2, stride, dilation, true,
               MlasIdentityActivation, false, false},
              seed++);
        if (HasFailure()) return;
      }
    }
  }
}

TEST_F(NchwcDepthwiseSlidingTest, SumBiasActivationsNonFinite) {
  const size_t b = block_size_;
  uint32_t seed = 300;
  for (size_t k : {3, 7}) {
    for (MLAS_ACTIVATION_KIND act : {MlasIdentityActivation, MlasReluActivation, MlasLeakyReluActivation}) {
      for (bool sum : {false, true}) {
        for (bool non_finite : {false, true}) {
          Check({1, 2 * b, 10, 21, k, k, k / 2, k / 2, k / 2, k / 2, 1, 1, true, act, sum, non_finite}, seed++);
          if (HasFailure()) return;
        }
      }
    }
  }
}

TEST_F(NchwcDepthwiseSlidingTest, Threaded) {
  const size_t b = block_size_;
  Check({2, 4 * b, 32, 32, 7, 7, 3, 3, 3, 3, 1, 1, true, MlasIdentityActivation, false, false}, 400, 4);
  Check({1, 4 * b, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, true, MlasReluActivation, true, false}, 401, 3);
}

}  // namespace
