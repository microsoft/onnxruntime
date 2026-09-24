// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

TEST(PoolTest, AveragePool_19_ceil_count_include_pad_1d_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("AveragePool", 19);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{3});
  test.AddAttribute("pads", std::vector<int64_t>{3, 3});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{7});
  test.AddAttribute("ceil_mode", (int64_t)1);
  test.AddAttribute("count_include_pad", (int64_t)1);

  std::vector<float> x_vals = {2.0903f, 4.6493f, 1.6320f, -3.2051f, 4.6975f, 4.7296f, 3.3653f, -1.5815f, -2.3832f, 0.9628f, -1.5899f, -2.6820f, 5.7529f, 7.7346f, -0.8910f, -2.0151f, 0.1313f, -0.5374f};
  std::vector<int64_t> x_dims = {1, 2, 9};
  std::vector<int64_t> expected_dims = {1, 2, 4};
  std::vector<float> expected_vals = {0.73807144f, 2.5655572f, 0.8032287f, -0.09990001f, 0.34911433f, 1.0389f, 1.4536142f, -0.40353334f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// AveragePool registered on WebGPU for opsets 11-18, 19-21 and 22 (a single kernel). The opset-19
// test above exercises the 19-21 registration (and pins the count_include_pad divisor fix); the two
// tests below exercise the other two registration ranges so none is left unverified. All three use
// the same shared kernel, so these use a trivially hand-verifiable case (kernel 2, stride 2, no pad)
// and just confirm the registration for that opset resolves and runs on WebGPU.
// Window means of {1,2,3,4}: (1+2)/2=1.5, (3+4)/2=3.5.
TEST(PoolTest, AveragePool_11_1d_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("AveragePool", 11);  // exercises the 11-18 registration
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddInput<float>("X", {1, 1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddOutput<float>("Y", {1, 1, 2}, {1.5f, 3.5f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(PoolTest, AveragePool_22_1d_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("AveragePool", 22);  // exercises the opset-22 registration
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddInput<float>("X", {1, 1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddOutput<float>("Y", {1, 1, 2}, {1.5f, 3.5f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// The WebGPU EP runs a pool as a workgroup-cooperative reduction when the serial path would not
// fill the device and the kernel window is large. Global pooling only reaches that path with a
// single output element per channel, which leaves two things unexercised: an output index taken
// from workgroup_idx, and a divisor that has to be reduced across the workgroup.
static void RunWebGpuLargeKernelPoolTest(const char* op_type, int opset, bool is_max_pool) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  // A 12x12 window with pad 2 over a 24x24 input: kernel_size 144, output_size 578.
  constexpr int64_t kChannels = 2, kSpatial = 24, kKernel = 12, kPad = 2;
  constexpr int64_t kOutSpatial = kSpatial + 2 * kPad - kKernel + 1;

  std::vector<float> x_vals(kChannels * kSpatial * kSpatial);
  for (size_t i = 0; i < x_vals.size(); ++i) {
    x_vals[i] = static_cast<float>(i) * 0.01f;
  }

  std::vector<float> expected_vals(kChannels * kOutSpatial * kOutSpatial);
  for (int64_t c = 0; c < kChannels; ++c) {
    for (int64_t oh = 0; oh < kOutSpatial; ++oh) {
      for (int64_t ow = 0; ow < kOutSpatial; ++ow) {
        float acc = is_max_pool ? std::numeric_limits<float>::lowest() : 0.0f;
        int64_t count = 0;
        for (int64_t kh = 0; kh < kKernel; ++kh) {
          for (int64_t kw = 0; kw < kKernel; ++kw) {
            const int64_t ih = oh + kh - kPad;
            const int64_t iw = ow + kw - kPad;
            if (ih < 0 || ih >= kSpatial || iw < 0 || iw >= kSpatial) {
              continue;
            }
            const float v = x_vals[static_cast<size_t>((c * kSpatial + ih) * kSpatial + iw)];
            acc = is_max_pool ? (v > acc ? v : acc) : acc + v;
            ++count;
          }
        }
        expected_vals[static_cast<size_t>((c * kOutSpatial + oh) * kOutSpatial + ow)] =
            is_max_pool ? acc : acc / static_cast<float>(count);
      }
    }
  }

  OpTester test(op_type, opset);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{kKernel, kKernel});
  test.AddAttribute("pads", std::vector<int64_t>{kPad, kPad, kPad, kPad});
  test.AddInput<float>("X", {1, kChannels, kSpatial, kSpatial}, x_vals);
  test.AddOutput<float>("Y", {1, kChannels, kOutSpatial, kOutSpatial}, expected_vals,
                        /*sort_output=*/false, /*rel_error=*/1e-3f, /*abs_error=*/1e-2f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(PoolTest, AveragePool_LargeKernelMultiElementOutput_WebGpu) {
  RunWebGpuLargeKernelPoolTest("AveragePool", 11, /*is_max_pool=*/false);
}

TEST(PoolTest, MaxPool_LargeKernelMultiElementOutput_WebGpu) {
  RunWebGpuLargeKernelPoolTest("MaxPool", 12, /*is_max_pool=*/true);
}

}  // namespace test
}  // namespace onnxruntime
