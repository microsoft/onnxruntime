// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef USE_WEBGPU

namespace onnxruntime {
namespace test {

namespace {

void RunOnWebGpu(OpTester& test) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  test.ConfigEp(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

constexpr int kNChw4cBlock = 4;
constexpr int kABcd16aBlock = 16;
constexpr int kABcd4bBlock = 4;

int ComputePaddedSize(int value, int block) {
  return ((value + block - 1) / block) * block;
}

// Helper to reshape plain NCHW data into nChw4c layout for expectations.
std::vector<float> PlainToBlockedNChw4c(const std::vector<float>& plain,
                                        int N,
                                        int C,
                                        int H,
                                        int W,
                                        int padded_C_override = -1) {
  const int padded_C = padded_C_override > 0 ? padded_C_override : ComputePaddedSize(C, kNChw4cBlock);
  const int channel_blocks = padded_C / kNChw4cBlock;
  std::vector<float> blocked(static_cast<size_t>(N) * padded_C * H * W, 0.0f);

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          const int plain_idx = ((n * C + c) * H + h) * W + w;
          const int block_idx = c / kNChw4cBlock;
          const int c_in_block = c % kNChw4cBlock;
          const int blocked_idx = ((((n * channel_blocks) + block_idx) * H + h) * W + w) * kNChw4cBlock + c_in_block;
          blocked[blocked_idx] = plain[plain_idx];
        }
      }
    }
  }

  return blocked;
}

std::vector<float> PlainToBlockedABcd16a4b(const std::vector<float>& plain,
                                           int N,
                                           int C,
                                           int H,
                                           int W,
                                           int padded_N_override = -1,
                                           int padded_C_override = -1) {
  assert(static_cast<int>(plain.size()) == N * C * H * W);
  const int padded_N = padded_N_override > 0 ? padded_N_override : ComputePaddedSize(N, kABcd16aBlock);
  const int padded_C = padded_C_override > 0 ? padded_C_override : ComputePaddedSize(C, kABcd4bBlock);
  const int c_blocks = padded_C / kABcd4bBlock;
  std::vector<float> blocked(static_cast<size_t>(padded_N) * padded_C * H * W, 0.0f);

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          const int plain_idx = ((n * C + c) * H + h) * W + w;
          const int a_block_idx = n / kABcd16aBlock;
          const int n_in_block = n % kABcd16aBlock;
          const int b_block_idx = c / kABcd4bBlock;
          const int c_in_block = c % kABcd4bBlock;
          const int blocked_idx = (((((a_block_idx * c_blocks) + b_block_idx) * H + h) * W + w) * kABcd16aBlock + n_in_block) * kABcd4bBlock + c_in_block;
          blocked[blocked_idx] = plain[plain_idx];
        }
      }
    }
  }

  return blocked;
}

}  // namespace

// Test Plain (NCHW) to Blocked (nChw4c) format transformation
TEST(FormatTransformTest, PlainToBlocked_nChw4c) {
  OpTester test("FormatTransform", 1, kMSInternalNHWCDomain);

  // Attributes
  test.AddAttribute<std::string>("src_format", "Plain");
  test.AddAttribute<std::string>("dst_format", "nChw4c");

  // Input: [1, 16, 4, 4] in NCHW format
  std::vector<float> input_data(1 * 16 * 4 * 4);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i);
  }
  test.AddInput<float>("X", {1, 16, 4, 4}, input_data);

  // Output: Still [1, 16, 4, 4] shape but with blocked layout
  const auto output_data = PlainToBlockedNChw4c(input_data, 1, 16, 4, 4);
  test.AddOutput<float>("Y", {1, 16, 4, 4}, output_data);

  RunOnWebGpu(test);
}

// Test Blocked (nChw4c) to Plain (NCHW) format transformation
TEST(FormatTransformTest, BlockedToPlain_nChw4c) {
  OpTester test("FormatTransform", 1, kMSInternalNHWCDomain);

  // Attributes
  test.AddAttribute<std::string>("src_format", "nChw4c");
  test.AddAttribute<std::string>("dst_format", "Plain");

  // Input: [1, 16, 4, 4] in blocked nChw4c format
  std::vector<float> plain_reference(1 * 16 * 4 * 4);
  for (size_t i = 0; i < plain_reference.size(); ++i) {
    plain_reference[i] = static_cast<float>(100 + i);
  }
  const auto input_data = PlainToBlockedNChw4c(plain_reference, 1, 16, 4, 4);

  test.AddInput<float>("X", {1, 16, 4, 4}, input_data);

  // Output: [1, 16, 4, 4] in NCHW format
  test.AddOutput<float>("Y", {1, 16, 4, 4}, plain_reference);

  RunOnWebGpu(test);
}

// Test Plain to Blocked with padding (non-divisible channels)
TEST(FormatTransformTest, PlainToBlocked_WithPadding) {
  OpTester test("FormatTransform", 1, kMSInternalNHWCDomain);

  // Attributes
  test.AddAttribute<std::string>("src_format", "Plain");
  test.AddAttribute<std::string>("dst_format", "nChw4c");

  // Input: [1, 10, 2, 2] - 10 channels, not divisible by 4
  std::vector<float> input_data(1 * 10 * 2 * 2);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i % 100);
  }
  test.AddInput<float>("X", {1, 10, 2, 2}, input_data);

  // Output: [1, 12, 2, 2] - padded to 12 channels (3 blocks of 4)
  // The last 2 channels (10-11) will be filled with zeros (padding)
  const int output_C = 12;  // Padded from 10 to 12
  const auto output_data = PlainToBlockedNChw4c(input_data, 1, 10, 2, 2, output_C);
  test.AddOutput<float>("Y", {1, 12, 2, 2}, output_data);

  RunOnWebGpu(test);
}

// Test round-trip transformation: Plain -> Blocked -> Plain
TEST(FormatTransformTest, RoundTrip) {
  // Original data in plain format
  std::vector<float> original_data(1 * 16 * 4 * 4);
  for (size_t i = 0; i < original_data.size(); ++i) {
    original_data[i] = static_cast<float>(i);
  }

  // First transformation: Plain to nChw4c
  OpTester test1("FormatTransform", 1, kMSInternalNHWCDomain);
  test1.AddAttribute<std::string>("src_format", "Plain");
  test1.AddAttribute<std::string>("dst_format", "nChw4c");

  test1.AddInput<float>("X", {1, 16, 4, 4}, original_data);

  // Calculate blocked output
  const auto blocked_data = PlainToBlockedNChw4c(original_data, 1, 16, 4, 4);
  test1.AddOutput<float>("Y", {1, 16, 4, 4}, blocked_data);
  RunOnWebGpu(test1);

  // Second transformation: nChw4c back to Plain
  OpTester test2("FormatTransform", 1, kMSInternalNHWCDomain);
  test2.AddAttribute<std::string>("src_format", "nChw4c");
  test2.AddAttribute<std::string>("dst_format", "Plain");

  test2.AddInput<float>("X", {1, 16, 4, 4}, blocked_data);
  test2.AddOutput<float>("Y", {1, 16, 4, 4}, original_data);

  RunOnWebGpu(test2);
}

TEST(FormatTransformTest, PlainToBlocked_ABcd16a4b) {
  OpTester test("FormatTransform", 1, kMSInternalNHWCDomain);

  test.AddAttribute<std::string>("src_format", "Plain");
  test.AddAttribute<std::string>("dst_format", "ABcd16a4b");

  // Input: [32, 8, 4, 4] in plain NCHW format
  // This will be transformed to ABcd16a4b format: [32/16, 8/4, 4, 4, 16, 4]
  // N=32 (2 blocks of 16), C=8 (2 blocks of 4)
  std::vector<float> input_data;
  for (int n = 0; n < 32; n++) {
    for (int c = 0; c < 8; c++) {
      for (int h = 0; h < 4; h++) {
        for (int w = 0; w < 4; w++) {
          input_data.push_back(static_cast<float>(n * 1000 + c * 100 + h * 10 + w));
        }
      }
    }
  }

  // Expected output in ABcd16a4b format
  const auto expected_data = PlainToBlockedABcd16a4b(input_data, 32, 8, 4, 4);

  test.AddInput<float>("X", {32, 8, 4, 4}, input_data);
  test.AddOutput<float>("Y", {32, 8, 4, 4}, expected_data);

  RunOnWebGpu(test);
}

TEST(FormatTransformTest, PlainToBlocked_ABcd16a4b_WithPadding) {
  OpTester test("FormatTransform", 1, kMSInternalNHWCDomain);

  test.AddAttribute<std::string>("src_format", "Plain");
  test.AddAttribute<std::string>("dst_format", "ABcd16a4b");

  const int N = 18;
  const int C = 6;
  const int H = 2;
  const int W = 2;
  std::vector<float> input_data(N * C * H * W);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i);
  }

  test.AddInput<float>("X", {N, C, H, W}, input_data);

  const int padded_N = ComputePaddedSize(N, kABcd16aBlock);
  const int padded_C = ComputePaddedSize(C, kABcd4bBlock);
  const auto expected_data = PlainToBlockedABcd16a4b(input_data, N, C, H, W, padded_N, padded_C);

  test.AddOutput<float>("Y", {padded_N, padded_C, H, W}, expected_data);

  RunOnWebGpu(test);
}

TEST(FormatTransformTest, BlockedToPlain_ABcd16a4b) {
  OpTester test("FormatTransform", 1, kMSInternalNHWCDomain);

  test.AddAttribute<std::string>("src_format", "ABcd16a4b");
  test.AddAttribute<std::string>("dst_format", "Plain");

  // Input and expected tensors in plain format
  std::vector<float> plain_data;
  for (int n = 0; n < 32; n++) {
    for (int c = 0; c < 8; c++) {
      for (int h = 0; h < 4; h++) {
        for (int w = 0; w < 4; w++) {
          plain_data.push_back(static_cast<float>(n * 1000 + c * 100 + h * 10 + w));
        }
      }
    }
  }
  const auto input_data = PlainToBlockedABcd16a4b(plain_data, 32, 8, 4, 4);

  test.AddInput<float>("X", {32, 8, 4, 4}, input_data);
  test.AddOutput<float>("Y", {32, 8, 4, 4}, plain_data);

  RunOnWebGpu(test);
}

TEST(FormatTransformTest, BlockedToPlain_ABcd16a4b_WithPadding) {
  OpTester test("FormatTransform", 1, kMSInternalNHWCDomain);

  test.AddAttribute<std::string>("src_format", "ABcd16a4b");
  test.AddAttribute<std::string>("dst_format", "Plain");

  const int N = 18;
  const int C = 6;
  const int H = 2;
  const int W = 2;
  const int padded_N = ComputePaddedSize(N, kABcd16aBlock);
  const int padded_C = ComputePaddedSize(C, kABcd4bBlock);

  std::vector<float> plain_data(N * C * H * W);
  for (size_t i = 0; i < plain_data.size(); ++i) {
    plain_data[i] = static_cast<float>(1000 + i);
  }

  const auto input_data = PlainToBlockedABcd16a4b(plain_data, N, C, H, W, padded_N, padded_C);

  std::vector<float> expected_data(static_cast<size_t>(padded_N) * padded_C * H * W, 0.0f);
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          const int padded_idx = ((n * padded_C + c) * H + h) * W + w;
          const int compact_idx = ((n * C + c) * H + h) * W + w;
          expected_data[padded_idx] = plain_data[compact_idx];
        }
      }
    }
  }

  test.AddInput<float>("X", {padded_N, padded_C, H, W}, input_data);
  test.AddOutput<float>("Y", {padded_N, padded_C, H, W}, expected_data);

  RunOnWebGpu(test);
}

TEST(FormatTransformTest, RoundTrip_ABcd16a4b) {
  OpTester test1("FormatTransform", 1, kMSInternalNHWCDomain);
  test1.AddAttribute<std::string>("src_format", "Plain");
  test1.AddAttribute<std::string>("dst_format", "ABcd16a4b");

  std::vector<float> input_data;
  for (int n = 0; n < 32; n++) {
    for (int c = 0; c < 8; c++) {
      for (int h = 0; h < 2; h++) {
        for (int w = 0; w < 2; w++) {
          input_data.push_back(static_cast<float>(n * 100 + c * 10 + h * 2 + w));
        }
      }
    }
  }

  // First transform to ABcd16a4b
  const auto intermediate_data = PlainToBlockedABcd16a4b(input_data, 32, 8, 2, 2);

  test1.AddInput<float>("X", {32, 8, 2, 2}, input_data);
  test1.AddOutput<float>("Y", {32, 8, 2, 2}, intermediate_data);

  RunOnWebGpu(test1);

  // Then transform back to Plain
  OpTester test2("FormatTransform", 1, kMSInternalNHWCDomain);
  test2.AddAttribute<std::string>("src_format", "ABcd16a4b");
  test2.AddAttribute<std::string>("dst_format", "Plain");

  test2.AddInput<float>("X", {32, 8, 2, 2}, intermediate_data);
  test2.AddOutput<float>("Y", {32, 8, 2, 2}, input_data);

  RunOnWebGpu(test2);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_WEBGPU
