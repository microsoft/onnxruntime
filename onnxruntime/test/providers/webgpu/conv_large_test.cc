// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

// Compute Conv2D output height/width given input, kernel, pad, stride, dilation.
static int64_t ConvOutputSize(int64_t input_size, int64_t kernel_size, int64_t pad_begin, int64_t pad_end,
                              int64_t stride, int64_t dilation) {
  int64_t effective_kernel = (kernel_size - 1) * dilation + 1;
  return (input_size + pad_begin + pad_end - effective_kernel) / stride + 1;
}

// Reference implementation for Conv2D (NCHW layout) with group, stride, pad, dilation, and optional bias.
static void ComputeExpectedConv2D(const std::vector<float>& x_vals, const std::vector<float>& w_vals,
                                  const std::vector<float>& b_vals, std::vector<float>& out_vals,
                                  int64_t N, int64_t C, int64_t H, int64_t W,
                                  int64_t M, int64_t kH, int64_t kW,
                                  int64_t oH, int64_t oW,
                                  int64_t pad_h_begin, int64_t pad_w_begin,
                                  int64_t stride_h, int64_t stride_w,
                                  int64_t dilation_h, int64_t dilation_w,
                                  int64_t group) {
  const int64_t C_per_group = C / group;
  const int64_t M_per_group = M / group;

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t m = 0; m < M; ++m) {
      const int64_t g = m / M_per_group;
      const int64_t m_in_group = m % M_per_group;
      for (int64_t oh = 0; oh < oH; ++oh) {
        for (int64_t ow = 0; ow < oW; ++ow) {
          float sum = b_vals.empty() ? 0.0f : b_vals[m];
          for (int64_t c = 0; c < C_per_group; ++c) {
            for (int64_t kh = 0; kh < kH; ++kh) {
              const int64_t ih = oh * stride_h + kh * dilation_h - pad_h_begin;
              if (ih < 0 || ih >= H) continue;
              for (int64_t kw = 0; kw < kW; ++kw) {
                const int64_t iw = ow * stride_w + kw * dilation_w - pad_w_begin;
                if (iw < 0 || iw >= W) continue;
                const size_t x_idx = static_cast<size_t>(((n * C + g * C_per_group + c) * H + ih) * W + iw);
                const size_t w_idx = static_cast<size_t>(((m_in_group + g * M_per_group) * C_per_group + c) * kH * kW +
                                                         kh * kW + kw);
                sum += x_vals[x_idx] * w_vals[w_idx];
              }
            }
          }
          const size_t o_idx = static_cast<size_t>(((n * M + m) * oH + oh) * oW + ow);
          out_vals[o_idx] = sum;
        }
      }
    }
  }
}

struct ConvTestParams {
  // Input: [N, C, H, W]
  int64_t N, C, H, W;
  // Weight: [M, C/group, kH, kW]
  int64_t M, kH, kW;
  // Conv attributes
  int64_t pad_h, pad_w;
  int64_t stride_h, stride_w;
  int64_t dilation_h, dilation_w;
  int64_t group;
  bool has_bias;
};

template <typename T, int version = 11>
void RunConvTest(const ConvTestParams& p) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, MLFloat16>, "unexpected type for T");

  const int64_t C_per_group = p.C / p.group;
  const int64_t oH = ConvOutputSize(p.H, p.kH, p.pad_h, p.pad_h, p.stride_h, p.dilation_h);
  const int64_t oW = ConvOutputSize(p.W, p.kW, p.pad_w, p.pad_w, p.stride_w, p.dilation_w);

  std::vector<int64_t> x_dims = {p.N, p.C, p.H, p.W};
  std::vector<int64_t> w_dims = {p.M, C_per_group, p.kH, p.kW};
  std::vector<int64_t> b_dims = p.has_bias ? std::vector<int64_t>{p.M} : std::vector<int64_t>{};
  std::vector<int64_t> y_dims = {p.N, p.M, oH, oW};

  RandomValueGenerator random{1234};
  std::vector<float> x_vals(random.Gaussian<float>(x_dims, 0.0f, 0.25f));
  std::vector<float> w_vals(random.Gaussian<float>(w_dims, 0.0f, 0.25f));
  std::vector<float> b_vals;
  if (p.has_bias) {
    b_vals = random.Gaussian<float>(b_dims, 0.0f, 0.1f);
  }

  std::vector<float> expected_vals(static_cast<size_t>(p.N * p.M * oH * oW), 0.0f);
  ComputeExpectedConv2D(x_vals, w_vals, b_vals, expected_vals,
                        p.N, p.C, p.H, p.W, p.M, p.kH, p.kW, oH, oW,
                        p.pad_h, p.pad_w, p.stride_h, p.stride_w,
                        p.dilation_h, p.dilation_w, p.group);

  OpTester test("Conv", version);
  test.AddAttribute("group", p.group);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{p.kH, p.kW});
  test.AddAttribute("pads", std::vector<int64_t>{p.pad_h, p.pad_w, p.pad_h, p.pad_w});
  test.AddAttribute("strides", std::vector<int64_t>{p.stride_h, p.stride_w});
  test.AddAttribute("dilations", std::vector<int64_t>{p.dilation_h, p.dilation_w});

  if constexpr (std::is_same_v<T, float>) {
    test.AddInput<T>("X", x_dims, x_vals);
    test.AddInput<T>("W", w_dims, w_vals, true /*is_initializer*/);
    if (p.has_bias)
      test.AddInput<T>("B", b_dims, b_vals);
    test.AddOutput<T>("Y", y_dims, expected_vals);
  } else {
    test.AddInput<T>("X", x_dims, FloatsToMLFloat16s(x_vals));
    test.AddInput<T>("W", w_dims, FloatsToMLFloat16s(w_vals), true /*is_initializer*/);
    if (p.has_bias)
      test.AddInput<T>("B", b_dims, FloatsToMLFloat16s(b_vals));
    test.AddOutput<T>("Y", y_dims, FloatsToMLFloat16s(expected_vals));
    test.SetOutputAbsErr("Y", 0.06f);
    test.SetOutputRelErr("Y", 0.02f);
  }

  // Disable TensorRT because weight as input is not supported.
  // QNN SDK 2.10.0 has a bug that breaks support for dynamic bias inputs.
  std::unordered_set<std::string> excluded_providers = {kTensorrtExecutionProvider, kQnnExecutionProvider};
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
}

template <int version = 11>
void RunConvBothTypes(const ConvTestParams& p) {
  RunConvTest<float, version>(p);
  RunConvTest<MLFloat16, version>(p);
}

TEST(Conv_Large, DISABLED_Conv1x1) {
  RunConvBothTypes({1, 256, 56, 56, 512, 1, 1, 0, 0, 1, 1, 1, 1, 1, true});
  RunConvBothTypes({1, 256, 56, 56, 511, 1, 1, 0, 0, 1, 1, 1, 1, 1, true});
}

}  // namespace test
}  // namespace onnxruntime
