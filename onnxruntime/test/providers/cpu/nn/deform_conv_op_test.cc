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

// Non-square kernel (kH != kW): 2x3 kernel, zero offset -> same as standard conv.
TEST(DeformConvTest, NonSquareKernel) {
  DeformConvTestParams p = {};
  p.batch_sz = 1;
  p.n_in_channels = 1;
  p.n_out_channels = 1;
  p.n_weight_grps = 1;
  p.n_offset_grps = 1;
  p.kernel_shape = {2, 3};
  p.stride = {1, 1};
  p.pad = {0, 0, 0, 0};
  p.dilation = {1, 1};
  p.in_h = 4;
  p.in_w = 5;
  // ONNX output size: out_h = (4 - 1*(2-1) - 1)/1 + 1 = 3, out_w = (5 - 1*(3-1) - 1)/1 + 1 = 3
  const int64_t out_h = 3;
  const int64_t out_w = 3;

  const size_t x_size = static_cast<size_t>(1 * 1 * 4 * 5);
  const size_t w_size = static_cast<size_t>(1 * 1 * 2 * 3);
  const size_t offset_size = static_cast<size_t>(1 * 1 * 2 * 2 * 3 * out_h * out_w);  // n_offset_grps * 2 * kH * kW * out_h * out_w
  const size_t mask_size = static_cast<size_t>(1 * 1 * 2 * 3 * out_h * out_w);          // n_offset_grps * kH * kW * out_h * out_w

  std::vector<float> X(x_size, 0.1f);
  std::vector<float> W(w_size, 0.1f);
  std::vector<float> offset(offset_size, 0.f);
  std::vector<float> mask(mask_size, 1.f);
  std::vector<float> B = {0.f};
  // With offset=0, mask=1: each output = 6 * 0.1 * 0.1 = 0.06 (9 positions)
  std::vector<float> expected_Y(static_cast<size_t>(out_h * out_w), 0.06f);

  RunDeformConvTest(p, X, W, offset, B, &mask, expected_Y);
}

// Asymmetric stride (stride_h != stride_w): stride=(2,1), zero offset.
TEST(DeformConvTest, AsymmetricStride) {
  DeformConvTestParams p = {};
  p.batch_sz = 1;
  p.n_in_channels = 1;
  p.n_out_channels = 1;
  p.n_weight_grps = 1;
  p.n_offset_grps = 1;
  p.kernel_shape = {2, 2};
  p.stride = {2, 1};
  p.pad = {0, 0, 0, 0};
  p.dilation = {1, 1};
  p.in_h = 5;
  p.in_w = 4;
  // out_h = (5 - 1*(2-1) - 1) / 2 + 1 = 2, out_w = (4 - 1*(2-1) - 1) / 1 + 1 = 3
  const int64_t out_h = 2;
  const int64_t out_w = 3;

  const size_t x_size = static_cast<size_t>(1 * 1 * 5 * 4);
  const size_t w_size = static_cast<size_t>(1 * 1 * 2 * 2);
  const size_t offset_size = static_cast<size_t>(1 * 1 * 2 * 2 * 2 * out_h * out_w);
  const size_t mask_size = static_cast<size_t>(1 * 1 * 2 * 2 * out_h * out_w);

  std::vector<float> X(x_size, 0.1f);
  std::vector<float> W(w_size, 0.1f);
  std::vector<float> offset(offset_size, 0.f);
  std::vector<float> mask(mask_size, 1.f);
  std::vector<float> B = {0.f};
  std::vector<float> expected_Y(static_cast<size_t>(out_h * out_w), 0.04f);

  RunDeformConvTest(p, X, W, offset, B, &mask, expected_Y);
}

// groups > 0 and non-zero offset; expected from deform_conv_expected_gen.py (seed=123).
TEST(DeformConvTest, GroupsWithNonZeroOffset) {
  DeformConvTestParams p = {};
  p.batch_sz = 1;
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

  std::vector<float> X = {0.296112f, 0.516562f, 0.251671f, 0.688557f, 0.073972f, 0.866522f, 0.136580f, 0.102479f, 0.184056f, 0.726447f, 0.315254f, 0.687107f, 0.075635f, 0.196638f, 0.316412f, 0.401740f, 0.118568f, 0.827395f, 0.382084f, 0.660494f, 0.853572f, 0.593153f, 0.636725f, 0.982629f, 0.274495f, 0.658376f, 0.277542f, 0.857325f, 0.899328f, 0.039014f, 0.926823f, 0.738757f, 0.717884f, 0.705837f, 0.915650f, 0.433980f};
  std::vector<float> W = {-1.182045f, -0.287745f, -0.604301f, 0.600237f, -1.420473f, -0.223828f, 0.430555f, -0.898857f, -0.017858f, 0.426403f, -0.765741f, -0.054514f, -0.732053f, 1.234742f, 1.186221f, -0.220099f};
  std::vector<float> offset = {-0.388483f, -0.934346f, -0.499144f, -1.086653f, 0.962421f, 0.249208f, -0.484502f, -2.092915f, 0.098284f, -0.093507f, 0.266215f, -0.585035f, -0.343038f, -0.682148f, -0.988689f, -1.701830f, -1.220290f, 1.313853f, 1.053300f, 0.138805f, -0.204445f, -2.268529f, -0.913328f, -0.420363f, -0.659559f, -0.797928f, 0.183831f, 0.229347f, 0.617743f, -0.287578f, 0.821824f, 0.151178f, -0.044382f, 1.623557f, -2.322871f, 1.087831f, -0.063545f, -0.448641f, -1.278470f, -1.144004f, -0.152640f, 0.116741f, 0.440260f, -1.446546f, -0.558082f, -0.051696f, -0.908273f, 0.350683f, -0.394809f, 0.489227f, -0.216815f, -1.747165f, 1.722842f, 0.773806f, 0.404630f, -1.646126f, -0.595084f, -0.711218f, 0.622965f, -1.372881f, -0.128065f, -1.283835f, -0.290120f, 1.276741f};
  std::vector<float> B = {0.983955f, 0.204512f};
  std::vector<float> mask = {-0.031861f, -0.478956f, 0.766809f, 0.027468f, 0.047470f, -0.923866f, -1.060737f, -2.324446f, -2.062818f, 0.006375f, -0.989555f, 0.701609f, -0.982238f, 0.277031f, 0.645495f, -0.895681f, 0.492753f, -0.014078f, -0.274663f, -0.764091f, -0.587157f, 1.195165f, -1.209575f, -0.556008f, -0.077105f, 1.277377f, -1.459629f, -2.159528f, -0.706709f, -0.922245f, 3.895372f, -0.602697f};
  std::vector<float> expected_Y = {0.971546f, 1.139858f, 0.452817f, 1.863882f, -0.565266f, 1.423187f, -2.462833f, -0.104923f};

  RunDeformConvTest(p, X, W, offset, B, &mask, expected_Y, 19, 1e-4f, 1e-4f);
}

// Sampling out of bounds: offset pushes sampling to (-5,-5), BilinearInterpolate returns 0.
TEST(DeformConvTest, OutOfBoundsSampling) {
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
  // out_h=out_w=2 (2x2 output), offset shape [1, 2, 2, 2] = 8 values. All (-5,-5) -> OOB -> 0
  std::vector<float> offset = {-5.f, -5.f, -5.f, -5.f, -5.f, -5.f, -5.f, -5.f};
  std::vector<float> B = {0.f};
  std::vector<float> mask = {1.f, 1.f, 1.f, 1.f};
  std::vector<float> expected_Y = {0.f, 0.f, 0.f, 0.f};

  RunDeformConvTest(p, X, W, offset, B, &mask, expected_Y);
}

// Dilation > 1: 2x2 kernel with dilation (2,2), zero offset -> 4 sample points with stride 2.
TEST(DeformConvTest, DilationGt1) {
  DeformConvTestParams p = {};
  p.batch_sz = 1;
  p.n_in_channels = 1;
  p.n_out_channels = 1;
  p.n_weight_grps = 1;
  p.n_offset_grps = 1;
  p.kernel_shape = {2, 2};
  p.stride = {1, 1};
  p.pad = {0, 0, 0, 0};
  p.dilation = {2, 2};
  p.in_h = 5;
  p.in_w = 5;
  // out_h = (5 - 2*(2-1) - 1)/1 + 1 = 3, out_w = 3
  const int64_t out_h = 3;
  const int64_t out_w = 3;

  const size_t x_size = 25;
  const size_t w_size = 4;
  const size_t offset_size = static_cast<size_t>(1 * 1 * 2 * 2 * 2 * out_h * out_w);
  const size_t mask_size = static_cast<size_t>(1 * 1 * 2 * 2 * out_h * out_w);

  std::vector<float> X(x_size, 0.1f);
  std::vector<float> W(w_size, 0.1f);
  std::vector<float> offset(offset_size, 0.f);
  std::vector<float> mask(mask_size, 1.f);
  std::vector<float> B = {0.f};
  // Each output: 4 samples at (0,0),(0,2),(2,0),(2,2) -> 4 * 0.1 * 0.1 = 0.04
  std::vector<float> expected_Y(static_cast<size_t>(out_h * out_w), 0.04f);

  RunDeformConvTest(p, X, W, offset, B, &mask, expected_Y);
}

// Decoupled groups: group=2, offset_group=1 (one offset map shared by all input channels).
TEST(DeformConvTest, DecoupledGroups) {
  DeformConvTestParams p = {};
  p.batch_sz = 1;
  p.n_in_channels = 4;
  p.n_out_channels = 2;
  p.n_weight_grps = 2;
  p.n_offset_grps = 1;
  p.kernel_shape = {2, 2};
  p.stride = {1, 1};
  p.pad = {0, 0, 0, 0};
  p.dilation = {1, 1};
  p.in_h = 3;
  p.in_w = 3;
  const int64_t out_h = 2;
  const int64_t out_w = 2;

  const size_t x_size = static_cast<size_t>(1 * 4 * 3 * 3);
  const size_t w_size = static_cast<size_t>(2 * 2 * 2 * 2);
  const size_t offset_size = static_cast<size_t>(1 * 1 * 2 * 2 * 2 * out_h * out_w);
  const size_t mask_size = static_cast<size_t>(1 * 1 * 2 * 2 * out_h * out_w);

  std::vector<float> X(x_size, 0.1f);
  std::vector<float> W(w_size, 0.1f);
  std::vector<float> offset(offset_size, 0.f);
  std::vector<float> mask(mask_size, 1.f);
  std::vector<float> B = {0.f, 0.f};
  // Zero offset -> grouped conv. Per output ch: 2 in_ch * 4 kernel * 0.01 = 0.08
  std::vector<float> expected_Y(static_cast<size_t>(1 * 2 * out_h * out_w), 0.08f);

  RunDeformConvTest(p, X, W, offset, B, &mask, expected_Y);
}

// Asymmetric padding: pads [top=1, left=0, bottom=0, right=1]; output 3x3, some positions have OOB samples.
TEST(DeformConvTest, AsymmetricPadding) {
  DeformConvTestParams p = {};
  p.batch_sz = 1;
  p.n_in_channels = 1;
  p.n_out_channels = 1;
  p.n_weight_grps = 1;
  p.n_offset_grps = 1;
  p.kernel_shape = {2, 2};
  p.stride = {1, 1};
  p.pad = {1, 0, 0, 1};
  p.dilation = {1, 1};
  p.in_h = 3;
  p.in_w = 3;
  // out_h = (3+1+0-1*(2-1)-1)/1+1 = 3, out_w = (3+0+1-1-1)/1+1 = 3
  const int64_t out_h = 3;
  const int64_t out_w = 3;

  const size_t offset_size = static_cast<size_t>(1 * 1 * 2 * 2 * 2 * out_h * out_w);
  const size_t mask_size = static_cast<size_t>(1 * 1 * 2 * 2 * out_h * out_w);

  std::vector<float> X(1 * 1 * 3 * 3, 0.1f);
  std::vector<float> W(1 * 1 * 2 * 2, 0.1f);
  std::vector<float> offset(offset_size, 0.f);
  std::vector<float> mask(mask_size, 1.f);
  std::vector<float> B = {0.f};
  // Row 0: (0,0),(0,1) 2 valid -> 0.02; (0,2) only (0,2) in, (0,3) OOB -> 1 valid -> 0.01. Row 1/2: as before.
  std::vector<float> expected_Y = {0.02f, 0.02f, 0.01f, 0.04f, 0.04f, 0.02f, 0.04f, 0.04f, 0.02f};

  RunDeformConvTest(p, X, W, offset, B, &mask, expected_Y);
}

// Tiny offset (near zero): offset (1e-6, 1e-6), sample ~(0,0) -> bilinear ≈ X[0,0]. Use 1x1 input for 1 output.
TEST(DeformConvTest, TinyOffset) {
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
  p.in_h = 1;
  p.in_w = 1;

  std::vector<float> X = {1.f};
  std::vector<float> W = {1.f};
  std::vector<float> offset = {1e-6f, 1e-6f};
  std::vector<float> B = {0.f};
  std::vector<float> mask = {1.f};
  std::vector<float> expected_Y = {1.f};  // bilinear at (1e-6, 1e-6) ≈ 1

  RunDeformConvTest(p, X, W, offset, B, &mask, expected_Y, 19, 1e-4f, 1e-4f);
}

// Offset (0.5, 0.5) at each kernel point: sampling at (i+0.5, j+0.5) -> (0.5,0.5),(0.5,1.5),(1.5,0.5),(1.5,1.5).
// Only (0.5,0.5) is fully in-bounds for 2x2 input; others hit boundary (OOB gives 0). Result = 1.6875.
TEST(DeformConvTest, OffsetAtPixelCenters) {
  DeformConvTestParams p = {};
  p.batch_sz = 1;
  p.n_in_channels = 1;
  p.n_out_channels = 1;
  p.n_weight_grps = 1;
  p.n_offset_grps = 1;
  p.kernel_shape = {2, 2};
  p.stride = {1, 1};
  p.pad = {0, 0, 0, 0};
  p.dilation = {1, 1};
  p.in_h = 2;
  p.in_w = 2;

  std::vector<float> X = {1.f, 2.f, 3.f, 4.f};
  std::vector<float> W = {0.25f, 0.25f, 0.25f, 0.25f};
  std::vector<float> offset = {
      0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f
  };
  std::vector<float> B = {0.f};
  std::vector<float> mask = {1.f, 1.f, 1.f, 1.f};
  std::vector<float> expected_Y = {1.6875f};  // op output: one center sample 2.5 + boundary samples

  RunDeformConvTest(p, X, W, offset, B, &mask, expected_Y);
}

}  // namespace test
}  // namespace onnxruntime
