// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

// Reference NCHW 2D convolution (group == 1). Mirrors the ONNX Conv semantics the
// WebGPU EP exercises after its internal NCHW->NHWC layout transform, so the
// subgroup-matrix implicit-GEMM conv can be validated against it.
static void ComputeExpectedConv(const std::vector<float>& x, const std::vector<float>& w,
                                const std::vector<float>& b, std::vector<float>& y,
                                int64_t N, int64_t Cin, int64_t H, int64_t W,
                                int64_t Cout, int64_t kh, int64_t kw,
                                int64_t stride_h, int64_t stride_w,
                                int64_t pad_h, int64_t pad_w,
                                int64_t dil_h, int64_t dil_w,
                                int64_t out_h, int64_t out_w) {
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t co = 0; co < Cout; ++co) {
      for (int64_t oh = 0; oh < out_h; ++oh) {
        for (int64_t ow = 0; ow < out_w; ++ow) {
          float sum = b.empty() ? 0.0f : b[co];
          for (int64_t ci = 0; ci < Cin; ++ci) {
            for (int64_t r = 0; r < kh; ++r) {
              for (int64_t s = 0; s < kw; ++s) {
                const int64_t ih = oh * stride_h - pad_h + r * dil_h;
                const int64_t iw = ow * stride_w - pad_w + s * dil_w;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                  continue;
                }
                const float xv = x[((n * Cin + ci) * H + ih) * W + iw];
                const float wv = w[((co * Cin + ci) * kh + r) * kw + s];
                sum += xv * wv;
              }
            }
          }
          y[((n * Cout + co) * out_h + oh) * out_w + ow] = sum;
        }
      }
    }
  }
}

// Runs a Conv over the WebGPU EP and compares against the CPU reference above.
// Inputs/outputs are standard ONNX NCHW; the WebGPU EP transforms to its
// channels-last kernels internally, which is where the subgroup-matrix conv runs.
static void RunConvTest(int64_t N, int64_t Cin, int64_t H, int64_t W,
                        int64_t Cout, int64_t kh, int64_t kw,
                        int64_t stride = 1, int64_t pad = 0, int64_t dilation = 1,
                        bool has_bias = false) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  const int64_t out_h = (H + 2 * pad - dilation * (kh - 1) - 1) / stride + 1;
  const int64_t out_w = (W + 2 * pad - dilation * (kw - 1) - 1) / stride + 1;
  ASSERT_GT(out_h, 0);
  ASSERT_GT(out_w, 0);

  const std::vector<int64_t> x_dims{N, Cin, H, W};
  const std::vector<int64_t> w_dims{Cout, Cin, kh, kw};
  const std::vector<int64_t> b_dims{Cout};
  const std::vector<int64_t> y_dims{N, Cout, out_h, out_w};

  RandomValueGenerator random{1234};
  std::vector<float> x_vals(random.Gaussian<float>(AsSpan(x_dims), 0.0f, 0.25f));
  std::vector<float> w_vals(random.Gaussian<float>(AsSpan(w_dims), 0.0f, 0.25f));
  std::vector<float> b_vals = has_bias ? random.Gaussian<float>(AsSpan(b_dims), 0.0f, 0.25f)
                                       : std::vector<float>{};

  std::vector<float> expected(static_cast<size_t>(N * Cout * out_h * out_w));
  ComputeExpectedConv(x_vals, w_vals, b_vals, expected, N, Cin, H, W, Cout, kh, kw,
                      stride, stride, pad, pad, dilation, dilation, out_h, out_w);

  OpTester test("Conv", 11);
  test.AddAttribute("group", static_cast<int64_t>(1));
  test.AddAttribute("kernel_shape", std::vector<int64_t>{kh, kw});
  test.AddAttribute("pads", std::vector<int64_t>{pad, pad, pad, pad});
  test.AddAttribute("strides", std::vector<int64_t>{stride, stride});
  test.AddAttribute("dilations", std::vector<int64_t>{dilation, dilation});

  test.AddInput<MLFloat16>("X", x_dims, FloatsToMLFloat16s(x_vals));
  test.AddInput<MLFloat16>("W", w_dims, FloatsToMLFloat16s(w_vals));
  if (has_bias) {
    test.AddInput<MLFloat16>("B", b_dims, FloatsToMLFloat16s(b_vals));
  }
  test.AddOutput<MLFloat16>("Y", y_dims, FloatsToMLFloat16s(expected));
  test.SetOutputAbsErr("Y", 0.06f);
  test.SetOutputRelErr("Y", 0.02f);

  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

// K = kh*kw*Cin is a multiple of 16 and N = Cout is even: the aligned fast path.
// out_h*out_w and Cout are tile multiples here (no partial tiles).
TEST(SubgroupMatrixConv, DISABLED_Aligned) {
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/16, /*W=*/16, /*Cout=*/64, /*kh=*/3, /*kw=*/3);
  RunConvTest(/*N=*/1, /*Cin=*/32, /*H=*/8, /*W=*/8, /*Cout=*/128, /*kh=*/3, /*kw=*/3);
}

// Partial output tiles: out_h*out_w (M) and/or Cout (N) are not tile multiples,
// exercising the kernel's bounds-checked M/N stores. N stays even.
TEST(SubgroupMatrixConv, DISABLED_PartialTiles) {
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/15, /*W=*/13, /*Cout=*/66, /*kh=*/3, /*kw=*/3);
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/17, /*W=*/9, /*Cout=*/130, /*kh=*/3, /*kw=*/3);
}

// Padding and strides: the im2col gather must return zero outside the input.
TEST(SubgroupMatrixConv, DISABLED_PadStride) {
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/16, /*W=*/16, /*Cout=*/64, /*kh=*/3, /*kw=*/3,
              /*stride=*/2, /*pad=*/1);
  RunConvTest(/*N=*/1, /*Cin=*/32, /*H=*/14, /*W=*/14, /*Cout=*/64, /*kh=*/5, /*kw=*/5,
              /*stride=*/1, /*pad=*/2);
}

// Dilated convolution.
TEST(SubgroupMatrixConv, DISABLED_Dilation) {
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/20, /*W=*/20, /*Cout=*/64, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/2, /*dilation=*/2);
}

// Optional bias add.
TEST(SubgroupMatrixConv, DISABLED_Bias) {
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/16, /*W=*/16, /*Cout=*/64, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/0, /*dilation=*/1, /*has_bias=*/true);
}

// Batched conv: each batch slice is dispatched on z. Small per-slice M x N grids
// with a larger batch also stress the selector's split-K clamping.
TEST(SubgroupMatrixConv, DISABLED_Batched) {
  RunConvTest(/*N=*/4, /*Cin=*/16, /*H=*/16, /*W=*/16, /*Cout=*/64, /*kh=*/3, /*kw=*/3);
  RunConvTest(/*N=*/8, /*Cin=*/32, /*H=*/8, /*W=*/8, /*Cout=*/64, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/1);
}

}  // namespace test
}  // namespace onnxruntime
