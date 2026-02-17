// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for DeformConv (CPU), aligned with PyTorch Vision deform_conv2d tests.
// Reference: https://github.com/pytorch/vision/blob/main/test/test_ops.py (TestDeformConv)

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

// Parameters similar to PyTorch TestDeformConv::get_fn_args (smaller for speed).
struct DeformConvTestParams {
  int64_t batch_sz;
  int64_t n_in_channels;
  int64_t n_out_channels;
  int64_t n_weight_grps;
  int64_t n_offset_grps;
  std::vector<int64_t> kernel_shape;  // {kH, kW}
  std::vector<int64_t> stride;
  std::vector<int64_t> pad;
  std::vector<int64_t> dilation;
  int64_t in_h;
  int64_t in_w;
};

void RunDeformConvTest(const DeformConvTestParams& params,
                       const std::vector<float>& X,
                       const std::vector<float>& W,
                       const std::vector<float>& offset,
                       const std::vector<float>& B,
                       const std::vector<float>* mask,
                       const std::vector<float>& expected_Y,
                       int opset = 19,
                       float rtol = 1e-5f,
                       float atol = 1e-5f) {
  const int64_t kH = params.kernel_shape[0];
  const int64_t kW = params.kernel_shape[1];
  const int64_t out_h = (params.in_h + params.pad[0] + params.pad[2] -
                         params.dilation[0] * (kH - 1) - 1) / params.stride[0] + 1;
  const int64_t out_w = (params.in_w + params.pad[1] + params.pad[3] -
                         params.dilation[1] * (kW - 1) - 1) / params.stride[1] + 1;

  OpTester test("DeformConv", opset);
  test.AddAttribute("kernel_shape", params.kernel_shape);
  test.AddAttribute("strides", params.stride);
  test.AddAttribute("pads", params.pad);
  test.AddAttribute("dilations", params.dilation);
  test.AddAttribute("group", params.n_weight_grps);
  test.AddAttribute("offset_group", params.n_offset_grps);

  const std::vector<int64_t> X_shape = {params.batch_sz, params.n_in_channels, params.in_h, params.in_w};
  const std::vector<int64_t> W_shape = {params.n_out_channels, params.n_in_channels / params.n_weight_grps, kH, kW};
  const std::vector<int64_t> offset_shape = {params.batch_sz, params.n_offset_grps * 2 * kH * kW, out_h, out_w};
  const std::vector<int64_t> Y_shape = {params.batch_sz, params.n_out_channels, out_h, out_w};

  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("W", W_shape, W);
  test.AddInput<float>("offset", offset_shape, offset);
  test.AddInput<float>("B", {params.n_out_channels}, B);
  if (mask != nullptr) {
    const std::vector<int64_t> mask_shape = {params.batch_sz, params.n_offset_grps * kH * kW, out_h, out_w};
    test.AddInput<float>("mask", mask_shape, *mask);
  } else {
    test.AddOptionalInputEdge<float>();
  }

  test.AddOutput<float>("Y", Y_shape, expected_Y, false, rtol, atol);

  std::unordered_set<std::string> excluded = {kTensorrtExecutionProvider, kCudaExecutionProvider,
                                               kOpenVINOExecutionProvider, kQnnExecutionProvider};
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded);
}

}  // namespace

// Minimal case: 1x1 kernel, 2x2 input, one output position with fractional offset (bilinear).
// At (0,0) offset (0.5, 0.5) samples center of [1,2;3,4] -> 2.5.
TEST(DeformConvTest, MinimalBilinear) {
  DeformConvTestParams p = {};
  p.batch_sz = 1;
  p.n_in_channels = 1;
  p.n_out_channels = 1;
  p.n_weight_grps = 1;
  p.n_offset_grps = 1;
  p.kernel_shape = {1, 1};
  p.stride = {1, 1};
  p.pad = {0, 0, 0, 0};
  p.dilation = {1, 1};
  p.in_h = 2;
  p.in_w = 2;

  std::vector<float> X = {1.f, 2.f, 3.f, 4.f};  // NCHW
  std::vector<float> W = {1.f};
  // offset (1, 2, 2, 2): ch0=offset_h, ch1=offset_w per output position. (0,0):(0.5,0)->2.5, (0,1):(0.5,-1)->1
  std::vector<float> offset = {
    0.5f, 0.f, 0.f, 0.f,
    0.5f, -1.0f, 0.f, 0.f
  };
  std::vector<float> B = {0.f};
  std::vector<float> mask = {1.f, 1.f, 1.f, 1.f};  // (1,1,2,2)
  std::vector<float> expected_Y = {2.5f, 1.f, 3.f, 4.f};

  RunDeformConvTest(p, X, W, offset, B, &mask, expected_Y);
}

// Forward with mask and bias: 2 batches, 2 groups, zero offset -> behaves like grouped conv.
// With offset=0 and mask=1, Y = Conv(X,W) + B. Use small inputs and compute expected.
TEST(DeformConvTest, ForwardWithMaskAndBias) {
  DeformConvTestParams p = {};
  p.batch_sz = 2;
  p.n_in_channels = 4;
  p.n_out_channels = 2;
  p.n_weight_grps = 2;
  p.n_offset_grps = 2;
  p.kernel_shape = {2, 2};
  p.stride = {1, 1};
  p.pad = {0, 0, 0, 0};
  p.dilation = {1, 1};
  p.in_h = 3;
  p.in_w = 3;
  const int64_t out_h = 2;
  const int64_t out_w = 2;

  const size_t x_size = static_cast<size_t>(p.batch_sz * p.n_in_channels * p.in_h * p.in_w);
  const size_t w_size = static_cast<size_t>(p.n_out_channels * (p.n_in_channels / p.n_weight_grps) * 2 * 2);
  const size_t offset_size = static_cast<size_t>(p.batch_sz * p.n_offset_grps * 2 * 2 * 2 * out_h * out_w);
  const size_t mask_size = static_cast<size_t>(p.batch_sz * p.n_offset_grps * 2 * 2 * out_h * out_w);

  std::vector<float> X(x_size, 0.1f);
  std::vector<float> W(w_size, 0.1f);
  std::vector<float> offset(offset_size, 0.f);  // zero offset -> regular grid sampling
  std::vector<float> mask(mask_size, 1.f);
  std::vector<float> B = {0.5f, -0.5f};

  // With offset=0, mask=1: deform_conv equals grouped conv. Per ONNX, group 0 -> output ch 0, group 1 -> ch 1.
  // Uniform X=0.1, W=0.1, 2x2 kernel -> 0.08 + B per channel; Y[:,0,:,:]=0.58, Y[:,1,:,:]=-0.42.
  const size_t y_size = static_cast<size_t>(p.batch_sz * p.n_out_channels * out_h * out_w);
  std::vector<float> expected_Y(y_size);
  for (int64_t b = 0; b < p.batch_sz; ++b) {
    for (int64_t c = 0; c < p.n_out_channels; ++c) {
      float val = (c % 2 == 0) ? 0.58f : -0.42f;
      for (int64_t i = 0; i < out_h * out_w; ++i) {
        expected_Y[b * p.n_out_channels * out_h * out_w + c * out_h * out_w + i] = val;
      }
    }
  }

  RunDeformConvTest(p, X, W, offset, B, &mask, expected_Y, 19, 1e-4f, 1e-4f);
}

// No mask (optional): same as above but mask omitted; compare to run with ones mask via tolerance.
TEST(DeformConvTest, ForwardNoMask) {
  DeformConvTestParams p = {};
  p.batch_sz = 1;
  p.n_in_channels = 2;
  p.n_out_channels = 2;
  p.n_weight_grps = 1;
  p.n_offset_grps = 1;
  p.kernel_shape = {2, 2};
  p.stride = {1, 1};
  p.pad = {0, 0, 0, 0};
  p.dilation = {1, 1};
  p.in_h = 3;
  p.in_w = 3;
  const int64_t out_h = 2;
  const int64_t out_w = 2;

  const size_t x_size = 1 * 2 * 3 * 3;
  const size_t w_size = 2 * 2 * 2 * 2;
  const size_t offset_size = 1 * 2 * 2 * 2 * out_h * out_w;
  const size_t y_size = 1 * 2 * out_h * out_w;

  std::vector<float> X(x_size, 0.1f);
  std::vector<float> W(w_size, 0.1f);
  std::vector<float> offset(offset_size, 0.f);
  std::vector<float> B(2, 0.f);
  // No mask => mask=1. Zero offset => same as conv. Y = 4*2*0.1*0.1 = 0.08 per position.
  std::vector<float> expected_Y(y_size, 0.08f);

  OpTester test("DeformConv", 19);
  test.AddAttribute("kernel_shape", p.kernel_shape);
  test.AddAttribute("strides", p.stride);
  test.AddAttribute("pads", p.pad);
  test.AddAttribute("dilations", p.dilation);
  test.AddAttribute("group", p.n_weight_grps);
  test.AddAttribute("offset_group", p.n_offset_grps);
  const std::vector<int64_t> X_shape = {p.batch_sz, p.n_in_channels, p.in_h, p.in_w};
  const std::vector<int64_t> W_shape = {p.n_out_channels, p.n_in_channels / p.n_weight_grps, 2, 2};
  const std::vector<int64_t> offset_shape = {p.batch_sz, p.n_offset_grps * 2 * 2 * 2, out_h, out_w};
  const std::vector<int64_t> Y_shape = {p.batch_sz, p.n_out_channels, out_h, out_w};

  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("W", W_shape, W);
  test.AddInput<float>("offset", offset_shape, offset);
  test.AddInput<float>("B", {p.n_out_channels}, B);
  test.AddOptionalInputEdge<float>();  // no mask
  test.AddOutput<float>("Y", Y_shape, expected_Y, false, 1e-4f, 1e-4f);
  std::unordered_set<std::string> excluded = {kTensorrtExecutionProvider, kCudaExecutionProvider,
                                               kOpenVINOExecutionProvider, kQnnExecutionProvider};
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded);
}

// Empty batch (like PyTorch batch_sz=0). (like PyTorch batch_sz=0).
TEST(DeformConvTest, EmptyBatch) {
  DeformConvTestParams p = {};
  p.batch_sz = 0;
  p.n_in_channels = 2;
  p.n_out_channels = 2;
  p.n_weight_grps = 1;
  p.n_offset_grps = 1;
  p.kernel_shape = {2, 2};
  p.stride = {1, 1};
  p.pad = {0, 0, 0, 0};
  p.dilation = {1, 1};
  p.in_h = 3;
  p.in_w = 3;
  const int64_t out_h = 2;
  const int64_t out_w = 2;

  std::vector<float> X;
  std::vector<float> W = std::vector<float>(2 * 2 * 2 * 2, 0.1f);
  std::vector<float> offset;
  std::vector<float> B(2, 0.f);
  std::vector<float> expected_Y;

  OpTester test("DeformConv", 19);
  test.AddAttribute("kernel_shape", p.kernel_shape);
  test.AddAttribute("strides", p.stride);
  test.AddAttribute("pads", p.pad);
  test.AddAttribute("dilations", p.dilation);
  test.AddAttribute("group", p.n_weight_grps);
  test.AddAttribute("offset_group", p.n_offset_grps);
  const std::vector<int64_t> X_shape = {0, p.n_in_channels, p.in_h, p.in_w};
  const std::vector<int64_t> W_shape = {p.n_out_channels, p.n_in_channels / p.n_weight_grps, 2, 2};
  const std::vector<int64_t> offset_shape = {0, p.n_offset_grps * 2 * 2 * 2, out_h, out_w};
  const std::vector<int64_t> Y_shape = {0, p.n_out_channels, out_h, out_w};

  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("W", W_shape, W);
  test.AddInput<float>("offset", offset_shape, offset);
  test.AddInput<float>("B", {p.n_out_channels}, B);
  test.AddOptionalInputEdge<float>();
  test.AddOutput<float>("Y", Y_shape, expected_Y);
  std::unordered_set<std::string> excluded = {kTensorrtExecutionProvider, kCudaExecutionProvider,
                                               kOpenVINOExecutionProvider, kQnnExecutionProvider};
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded);
}

// Wrong offset channel count -> expect failure (like PyTorch test_wrong_sizes). -> expect failure (like PyTorch test_wrong_sizes).
TEST(DeformConvTest, WrongOffsetShape) {
  DeformConvTestParams p = {};
  p.batch_sz = 1;
  p.n_in_channels = 2;
  p.n_out_channels = 2;
  p.n_weight_grps = 1;
  p.n_offset_grps = 1;
  p.kernel_shape = {2, 2};
  p.stride = {1, 1};
  p.pad = {0, 0, 0, 0};
  p.dilation = {1, 1};
  p.in_h = 3;
  p.in_w = 3;
  const int64_t out_h = 2;
  const int64_t out_w = 2;

  std::vector<float> X(1 * 2 * 3 * 3, 0.1f);
  std::vector<float> W(2 * 2 * 2 * 2, 0.1f);
  std::vector<float> wrong_offset(1 * 2 * out_h * out_w);  // wrong: only 2 channels instead of 8
  std::vector<float> B(2, 0.f);
  std::vector<float> expected_Y(1 * 2 * out_h * out_w, 0.f);

  const std::vector<int64_t> offset_shape_wrong = {1, 2, out_h, out_w};
  const std::vector<int64_t> Y_shape_wrong = {1, 2, out_h, out_w};

  OpTester test("DeformConv", 19);
  test.AddAttribute("kernel_shape", p.kernel_shape);
  test.AddAttribute("strides", p.stride);
  test.AddAttribute("pads", p.pad);
  test.AddAttribute("dilations", p.dilation);
  test.AddAttribute("group", p.n_weight_grps);
  test.AddAttribute("offset_group", p.n_offset_grps);
  test.AddInput<float>("X", {1, 2, 3, 3}, X);
  test.AddInput<float>("W", {2, 2, 2, 2}, W);
  test.AddInput<float>("offset", offset_shape_wrong, wrong_offset);  // invalid channels
  test.AddInput<float>("B", {2}, B);
  test.AddOptionalInputEdge<float>();
  test.AddOutput<float>("Y", Y_shape_wrong, expected_Y);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Offset channel count must be offset_group * 2 * kH * kW");
}

// Wrong mask channel count -> expect failure.
TEST(DeformConvTest, WrongMaskShape) {
  DeformConvTestParams p = {};
  p.batch_sz = 1;
  p.n_in_channels = 2;
  p.n_out_channels = 2;
  p.n_weight_grps = 1;
  p.n_offset_grps = 1;
  p.kernel_shape = {2, 2};
  p.stride = {1, 1};
  p.pad = {0, 0, 0, 0};
  p.dilation = {1, 1};
  p.in_h = 3;
  p.in_w = 3;
  const int64_t out_h = 2;
  const int64_t out_w = 2;

  std::vector<float> X(1 * 2 * 3 * 3, 0.1f);
  std::vector<float> W(2 * 2 * 2 * 2, 0.1f);
  const size_t offset_size = static_cast<size_t>(
      p.batch_sz * p.n_offset_grps * 2 * p.kernel_shape[0] * p.kernel_shape[1] * out_h * out_w);
  std::vector<float> offset(offset_size, 0.f);
  std::vector<float> B(2, 0.f);
  std::vector<float> wrong_mask(1 * 2 * out_h * out_w);  // wrong: 2 instead of 4
  std::vector<float> expected_Y(1 * 2 * out_h * out_w, 0.f);

  const std::vector<int64_t> mask_shape_wrong = {1, 2, out_h, out_w};
  const std::vector<int64_t> Y_shape_mask = {1, 2, out_h, out_w};

  OpTester test("DeformConv", 19);
  test.AddAttribute("kernel_shape", p.kernel_shape);
  test.AddAttribute("strides", p.stride);
  test.AddAttribute("pads", p.pad);
  test.AddAttribute("dilations", p.dilation);
  test.AddAttribute("group", p.n_weight_grps);
  test.AddAttribute("offset_group", p.n_offset_grps);
  test.AddInput<float>("X", {1, 2, 3, 3}, X);
  test.AddInput<float>("W", {2, 2, 2, 2}, W);
  test.AddInput<float>("offset", {1, 8, out_h, out_w}, offset);
  test.AddInput<float>("B", {2}, B);
  test.AddInput<float>("mask", mask_shape_wrong, wrong_mask);
  test.AddOutput<float>("Y", Y_shape_mask, expected_Y);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Mask channel count");
}

// Opset 22 (same behavior, different opset).
TEST(DeformConvTest, Opset22) {
  DeformConvTestParams p = {};
  p.batch_sz = 1;
  p.n_in_channels = 1;
  p.n_out_channels = 1;
  p.n_weight_grps = 1;
  p.n_offset_grps = 1;
  p.kernel_shape = {1, 1};
  p.stride = {1, 1};
  p.pad = {0, 0, 0, 0};
  p.dilation = {1, 1};
  p.in_h = 2;
  p.in_w = 2;

  std::vector<float> X = {1.f, 2.f, 3.f, 4.f};
  std::vector<float> W = {1.f};
  std::vector<float> offset = {0.5f, 0.f, 0.f, 0.f, 0.5f, 0.f, 0.f, 0.f};
  std::vector<float> B = {0.f};
  std::vector<float> mask = {1.f, 1.f, 1.f, 1.f};
  std::vector<float> expected_Y = {2.5f, 2.f, 3.f, 4.f};

  RunDeformConvTest(p, X, W, offset, B, &mask, expected_Y, 22);
}

}  // namespace test
}  // namespace onnxruntime
