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
  // Conv attributes. pad_h/pad_w are the leading (begin) pads.
  int64_t pad_h, pad_w;
  int64_t stride_h, stride_w;
  int64_t dilation_h, dilation_w;
  int64_t group;
  bool has_bias;
  // Trailing (end) pads. Negative means "same as the leading pad", i.e. symmetric
  // padding; set them explicitly to build an asymmetrically padded Conv.
  int64_t pad_h_end = -1;
  int64_t pad_w_end = -1;
};

template <typename T, int version = 11>
void RunConvTest(const ConvTestParams& p) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, MLFloat16>, "unexpected type for T");

  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  const int64_t C_per_group = p.C / p.group;
  const int64_t pad_h_end = p.pad_h_end < 0 ? p.pad_h : p.pad_h_end;
  const int64_t pad_w_end = p.pad_w_end < 0 ? p.pad_w : p.pad_w_end;
  const int64_t oH = ConvOutputSize(p.H, p.kH, p.pad_h, pad_h_end, p.stride_h, p.dilation_h);
  const int64_t oW = ConvOutputSize(p.W, p.kW, p.pad_w, pad_w_end, p.stride_w, p.dilation_w);

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
  test.AddAttribute("pads", std::vector<int64_t>{p.pad_h, p.pad_w, pad_h_end, pad_w_end});
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

  // WebGPU only: the reference above is computed on the host, and an MLFloat16 Conv has
  // no CPU kernel on most platforms, so running the other EPs would only add noise.
  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

// Small enabled coverage for the Conv MatMul routes, including the subgroup-matrix
// 1x1 Conv path. It only engages when the device reports the f16 8x16x16
// subgroup-matrix config; elsewhere these shapes take the generic path and the same
// assertions still apply. f16 with a constant weight, which is what the subgroup path
// requires (a float weight or a non-constant weight falls back).
//
// Input channels are the matmul K and output channels are the matmul N, so:
//  - C=32:  K is a multiple of the subgroup-matrix K (16), the path is eligible.
//  - M=16:  even N, the plain path.
//  - M=17:  odd N, which the f16 subgroup-matrix load cannot stride over. The constant
//           weight is padded to an even stride once and cached, and the output is still
//           written at the real, odd N.
//  - C=24:  K is not a multiple of 16 and the kernel has no K tail handling, so the
//           tiling selector must decline and let the generic path compute it.
TEST(Conv_Large, Conv1x1_MatMulRoute_F16) {
  RunConvTest<MLFloat16>({1, 32, 8, 8, 16, 1, 1, 0, 0, 1, 1, 1, 1, 1, true});
  RunConvTest<MLFloat16>({1, 32, 8, 8, 17, 1, 1, 0, 0, 1, 1, 1, 1, 1, true});
  RunConvTest<MLFloat16>({1, 24, 8, 8, 16, 1, 1, 0, 0, 1, 1, 1, 1, 1, true});
  // Batch > 1: the 1x1 reshape folds N,H,W into independent matmul rows.
  RunConvTest<MLFloat16>({2, 32, 8, 8, 17, 1, 1, 0, 0, 1, 1, 1, 1, 1, true});
}

// A kernel that covers the whole input (the "same size" reshape): the window folds into
// a single matmul row per batch element, so the output is 1x1 spatially and the matmul K
// is H*W*C (256 here, a multiple of 16). Odd output channels again exercise the padded
// constant weight, and the 2-row matmul exercises the kernel's partial-M stores.
TEST(Conv_Large, ConvSameSize_MatMulRoute_F16) {
  RunConvTest<MLFloat16>({2, 16, 4, 4, 16, 4, 4, 0, 0, 1, 1, 1, 1, 1, true});
  RunConvTest<MLFloat16>({2, 16, 4, 4, 17, 4, 4, 0, 0, 1, 1, 1, 1, 1, true});
}

// Trailing-only padding. Neither reshape is equivalent to the Conv here: the output grid
// is larger than what the matmul computes (1x1: (H+1) x (W+1) instead of H x W;
// same-size: 2x2 instead of 1x1), so both routes must decline and fall back to the
// general Conv path. A gate that only looks at the leading pads would accept these and
// leave part of the output unwritten.
TEST(Conv_Large, ConvMatMulRoute_TrailingPadDeclined_F16) {
  ConvTestParams conv_1x1{1, 32, 8, 8, 16, 1, 1, 0, 0, 1, 1, 1, 1, 1, true};
  conv_1x1.pad_h_end = 1;
  conv_1x1.pad_w_end = 1;
  RunConvTest<MLFloat16>(conv_1x1);

  ConvTestParams conv_same_size{1, 16, 4, 4, 16, 4, 4, 0, 0, 1, 1, 1, 1, 1, true};
  conv_same_size.pad_h_end = 1;
  conv_same_size.pad_w_end = 1;
  RunConvTest<MLFloat16>(conv_same_size);
}

}  // namespace test
}  // namespace onnxruntime
