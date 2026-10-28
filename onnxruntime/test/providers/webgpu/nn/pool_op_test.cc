// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/conversion.h"
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

namespace {

struct LpPool2dTestCase {
  const char* op_type = "LpPool";
  int opset = 18;
  int64_t p = 2;
  std::vector<int64_t> x_dims;  // NCHW
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> strides{1, 1};
  std::vector<int64_t> pads{0, 0, 0, 0};
  std::vector<int64_t> dilations{1, 1};
  float value_scale = 0.25f;
};

// y = (sum |x|^p)^(1/p) over each window, with padding contributing 0.
std::vector<float> ReferenceLpPool2d(const LpPool2dTestCase& tc, const std::vector<float>& x,
                                     const std::vector<int64_t>& kernel_shape, std::vector<int64_t>& y_dims) {
  const int64_t n = tc.x_dims[0], c = tc.x_dims[1], h = tc.x_dims[2], w = tc.x_dims[3];
  const int64_t out_h =
      (h + tc.pads[0] + tc.pads[2] - ((kernel_shape[0] - 1) * tc.dilations[0] + 1)) / tc.strides[0] + 1;
  const int64_t out_w =
      (w + tc.pads[1] + tc.pads[3] - ((kernel_shape[1] - 1) * tc.dilations[1] + 1)) / tc.strides[1] + 1;
  y_dims = {n, c, out_h, out_w};

  std::vector<float> y;
  y.reserve(static_cast<size_t>(n * c * out_h * out_w));
  for (int64_t nc = 0; nc < n * c; ++nc) {
    for (int64_t oh = 0; oh < out_h; ++oh) {
      for (int64_t ow = 0; ow < out_w; ++ow) {
        double acc = 0.0;
        for (int64_t kh = 0; kh < kernel_shape[0]; ++kh) {
          for (int64_t kw = 0; kw < kernel_shape[1]; ++kw) {
            const int64_t ih = oh * tc.strides[0] + kh * tc.dilations[0] - tc.pads[0];
            const int64_t iw = ow * tc.strides[1] + kw * tc.dilations[1] - tc.pads[1];
            if (ih < 0 || ih >= h || iw < 0 || iw >= w) {
              continue;
            }
            acc += std::pow(std::abs(static_cast<double>(x[static_cast<size_t>((nc * h + ih) * w + iw)])),
                            static_cast<double>(tc.p));
          }
        }
        y.push_back(static_cast<float>(std::pow(acc, 1.0 / static_cast<double>(tc.p))));
      }
    }
  }
  return y;
}

// The model is NCHW; with is_nhwc the layout transformer moves LpPool into the internal NHWC domain.
template <typename T>
void RunWebGpuLpPool2dTest(const LpPool2dTestCase& tc, bool is_nhwc) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider(is_nhwc);
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  constexpr bool is_fp16 = std::is_same_v<T, MLFloat16>;
  const bool is_global = std::string_view(tc.op_type) == "GlobalLpPool";
  const std::vector<int64_t> kernel_shape = is_global ? std::vector<int64_t>{tc.x_dims[2], tc.x_dims[3]}
                                                      : tc.kernel_shape;

  std::vector<float> x(static_cast<size_t>(tc.x_dims[0] * tc.x_dims[1] * tc.x_dims[2] * tc.x_dims[3]));
  for (size_t i = 0; i < x.size(); ++i) {
    // Mixed signs and some zeros so |x| matters and an all-zero window is possible with padding.
    x[i] = static_cast<float>(static_cast<int64_t>((i * 7) % 23) - 11) * tc.value_scale;
    if constexpr (is_fp16) {
      x[i] = MLFloat16(x[i]).ToFloat();
    }
  }
  std::vector<int64_t> y_dims;
  const std::vector<float> y = ReferenceLpPool2d(tc, x, kernel_shape, y_dims);

  OpTester test(tc.op_type, tc.opset);
  test.AddAttribute("p", tc.p);
  if (!is_global) {
    test.AddAttribute("kernel_shape", tc.kernel_shape);
    test.AddAttribute("strides", tc.strides);
    test.AddAttribute("pads", tc.pads);
    if (tc.opset >= 18) {
      test.AddAttribute("dilations", tc.dilations);
    }
  }
  if constexpr (is_fp16) {
    test.AddInput<MLFloat16>("X", tc.x_dims, FloatsToMLFloat16s(x));
    test.AddOutput<MLFloat16>("Y", y_dims, FloatsToMLFloat16s(y), /*sort_output=*/false,
                              /*rel_error=*/4e-3f, /*abs_error=*/1e-2f);
  } else {
    test.AddInput<float>("X", tc.x_dims, x);
    test.AddOutput<float>("Y", y_dims, y, /*sort_output=*/false, /*rel_error=*/1e-4f, /*abs_error=*/1e-4f);
  }

  // Disable CPU fallback so a missing WebGPU kernel fails instead of silently running on CPU.
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

// Runs a case on both layouts and both float types. Four channels lets the NHWC path vectorize.
void RunWebGpuLpPool2dAllVariants(const LpPool2dTestCase& tc) {
  for (bool is_nhwc : {false, true}) {
    SCOPED_TRACE(is_nhwc ? "NHWC" : "NCHW");
    RunWebGpuLpPool2dTest<float>(tc, is_nhwc);
    RunWebGpuLpPool2dTest<MLFloat16>(tc, is_nhwc);
  }
}

}  // namespace

TEST(PoolTest, LpPool_P2_WebGpu) {
  LpPool2dTestCase tc;
  tc.x_dims = {1, 4, 5, 6};
  tc.kernel_shape = {3, 3};
  RunWebGpuLpPool2dAllVariants(tc);
}

TEST(PoolTest, LpPool_P1_StridesPads_WebGpu) {
  LpPool2dTestCase tc;
  tc.p = 1;
  tc.x_dims = {2, 4, 5, 5};
  tc.kernel_shape = {2, 3};
  tc.strides = {2, 1};
  tc.pads = {1, 0, 1, 2};
  RunWebGpuLpPool2dAllVariants(tc);
}

TEST(PoolTest, LpPool_P3_Dilations_WebGpu) {
  LpPool2dTestCase tc;
  tc.p = 3;
  tc.x_dims = {1, 4, 7, 7};
  tc.kernel_shape = {2, 2};
  tc.strides = {2, 2};
  tc.dilations = {2, 2};
  RunWebGpuLpPool2dAllVariants(tc);
}

TEST(PoolTest, LpPool_Opset11_WebGpu) {
  LpPool2dTestCase tc;
  tc.opset = 11;  // exercises the 11-17 registrations
  tc.x_dims = {1, 4, 4, 4};
  tc.kernel_shape = {2, 2};
  tc.pads = {1, 1, 1, 1};
  RunWebGpuLpPool2dAllVariants(tc);
}

TEST(PoolTest, LpPool_Opset22_WebGpu) {
  LpPool2dTestCase tc;
  tc.opset = 22;
  tc.x_dims = {1, 4, 4, 4};
  tc.kernel_shape = {2, 2};
  RunWebGpuLpPool2dAllVariants(tc);
}

TEST(PoolTest, GlobalLpPool_WebGpu) {
  LpPool2dTestCase tc;
  tc.op_type = "GlobalLpPool";
  tc.p = 3;
  tc.x_dims = {1, 4, 3, 5};
  // Opset 7 resolves to GlobalLpPool-2 (the WebGPU EP needs a model opset of at least 7); 22 to GlobalLpPool-22.
  for (int opset : {7, 22}) {
    SCOPED_TRACE(opset);
    tc.opset = opset;
    RunWebGpuLpPool2dAllVariants(tc);
  }
}

// A 12x12 window takes the workgroup-cooperative reduction path.
TEST(PoolTest, LpPool_LargeKernel_WebGpu) {
  LpPool2dTestCase tc;
  tc.x_dims = {1, 2, 24, 24};
  tc.kernel_shape = {12, 12};
  tc.pads = {2, 2, 2, 2};
  RunWebGpuLpPool2dAllVariants(tc);
}

// sum(x^2) over the window exceeds the float16 range although the result does not, so the
// accumulation must not happen in float16.
TEST(PoolTest, LpPool_Float16NoOverflow_WebGpu) {
  LpPool2dTestCase tc;
  tc.x_dims = {1, 4, 4, 4};
  tc.kernel_shape = {3, 3};
  tc.value_scale = 16.0f;  // |x| up to 176, so a window's sum of squares reaches ~1e5.
  for (bool is_nhwc : {false, true}) {
    SCOPED_TRACE(is_nhwc ? "NHWC" : "NCHW");
    RunWebGpuLpPool2dTest<MLFloat16>(tc, is_nhwc);
  }
}

}  // namespace test
}  // namespace onnxruntime
