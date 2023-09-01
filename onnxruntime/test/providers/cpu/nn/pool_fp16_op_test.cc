// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/mlas/inc/mlas.h"

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

#include "core/providers/cpu/nn/pool.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

namespace onnxruntime {
namespace test {

// Disable TensorRT on some of the tests because "pads" attribute is not supported

TEST(PoolFp16Test, MaxPool) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{8, 8});

  std::vector<MLFloat16> x_vals = {
      MLFloat16(0x1.884p-3f), MLFloat16(0x1.3e8p-1f), MLFloat16(0x1.c04p-2f), MLFloat16(0x1.92p-1f),
      MLFloat16(0x1.8f4p-1f), MLFloat16(0x1.174p-2f), MLFloat16(0x1.1bp-2f), MLFloat16(0x1.9a8p-1f),
      MLFloat16(0x1.ea8p-1f), MLFloat16(0x1.c08p-1f), MLFloat16(0x1.6e8p-2f), MLFloat16(0x1.008p-1f),
      MLFloat16(0x1.5ep-1f), MLFloat16(0x1.6dp-1f), MLFloat16(0x1.7b4p-2f), MLFloat16(0x1.1f4p-1f),
      MLFloat16(0x1.018p-1f), MLFloat16(0x1.c34p-7f), MLFloat16(0x1.8bcp-1f), MLFloat16(0x1.c4p-1f),
      MLFloat16(0x1.75cp-2f), MLFloat16(0x1.3bp-1f), MLFloat16(0x1.34cp-4f), MLFloat16(0x1.79cp-2f),
      MLFloat16(0x1.ddcp-1f), MLFloat16(0x1.4d8p-1f), MLFloat16(0x1.96cp-2f), MLFloat16(0x1.93cp-1f),
      MLFloat16(0x1.448p-2f), MLFloat16(0x1.22cp-1f), MLFloat16(0x1.bdp-1f), MLFloat16(0x1.becp-2f),
      MLFloat16(0x1.9acp-1f), MLFloat16(0x1.268p-3f), MLFloat16(0x1.688p-1f), MLFloat16(0x1.68cp-1f),
      MLFloat16(0x1.cp-3f), MLFloat16(0x1.d98p-1f), MLFloat16(0x1.c4cp-2f), MLFloat16(0x1.d18p-1f),
      MLFloat16(0x1.eap-5f), MLFloat16(0x1.798p-3f), MLFloat16(0x1.84p-5f), MLFloat16(0x1.598p-1f),
      MLFloat16(0x1.308p-1f), MLFloat16(0x1.11p-1f), MLFloat16(0x1.63p-5f), MLFloat16(0x1.1f8p-1f),
      MLFloat16(0x1.518p-2f), MLFloat16(0x1.018p-1f), MLFloat16(0x1.ca4p-4f), MLFloat16(0x1.37p-1f),
      MLFloat16(0x1.21cp-1f), MLFloat16(0x1.bb4p-8f), MLFloat16(0x1.3c4p-1f), MLFloat16(0x1.d3p-1f),
      MLFloat16(0x1.94cp-1f), MLFloat16(0x1.fcp-1f), MLFloat16(0x1.ebp-1f), MLFloat16(0x1.958p-1f),
      MLFloat16(0x1.24p-2f), MLFloat16(0x1.4p-1f), MLFloat16(0x1.e98p-2f), MLFloat16(0x1.90cp-3f),

      MLFloat16(0x1.878p-2f), MLFloat16(0x1.b94p-5f), MLFloat16(0x1.ce8p-2f), MLFloat16(0x1.f6cp-1f),
      MLFloat16(0x1.fbcp-4f), MLFloat16(0x1.e9p-4f), MLFloat16(0x1.7ap-1f), MLFloat16(0x1.2ccp-1f),
      MLFloat16(0x1.e3p-2f), MLFloat16(0x1.b6cp-4f), MLFloat16(0x1.d58p-3f), MLFloat16(0x1.cccp-1f),
      MLFloat16(0x1.aacp-2f), MLFloat16(0x1.124p-1f), MLFloat16(0x1.97p-8f), MLFloat16(0x1.33cp-2f),
      MLFloat16(0x1.bf8p-2f), MLFloat16(0x1.398p-1f), MLFloat16(0x1.d6p-1f), MLFloat16(0x1.408p-1f),
      MLFloat16(0x1.698p-1f), MLFloat16(0x1.32cp-3f), MLFloat16(0x1.7ep-1f), MLFloat16(0x1.a98p-1f),
      MLFloat16(0x1.448p-1f), MLFloat16(0x1.c0cp-2f), MLFloat16(0x1.388p-3f), MLFloat16(0x1.23p-1f),
      MLFloat16(0x1.0e8p-1f), MLFloat16(0x1.e74p-1f), MLFloat16(0x1.ecp-2f), MLFloat16(0x1.014p-1f),
      MLFloat16(0x1.13p-1f), MLFloat16(0x1.a38p-1f), MLFloat16(0x1.d4p-5f), MLFloat16(0x1.56cp-1f),
      MLFloat16(0x1.88cp-1f), MLFloat16(0x1.6a8p-1f), MLFloat16(0x1.98p-1f), MLFloat16(0x1.1d8p-1f),
      MLFloat16(0x1.ee8p-1f), MLFloat16(0x1.2d8p-3f), MLFloat16(0x1.e5cp-6f), MLFloat16(0x1.3p-1f),
      MLFloat16(0x1.d34p-4f), MLFloat16(0x1.e6cp-1f), MLFloat16(0x1.4d8p-2f), MLFloat16(0x1.8c8p-3f),
      MLFloat16(0x1.d4cp-2f), MLFloat16(0x1.d74p-1f), MLFloat16(0x1.c2p-1f), MLFloat16(0x1.02cp-2f),
      MLFloat16(0x1.644p-2f), MLFloat16(0x1.76p-3f), MLFloat16(0x1.cdcp-1f), MLFloat16(0x1.69cp-1f),
      MLFloat16(0x1.74p-1f), MLFloat16(0x1.cccp-1f), MLFloat16(0x1.8fp-1f), MLFloat16(0x1.32cp-1f),
      MLFloat16(0x1.2ap-2f), MLFloat16(0x1.36p-3f), MLFloat16(0x1.574p-2f), MLFloat16(0x1.50cp-1f),

      MLFloat16(0x1.2c8p-4f), MLFloat16(0x1.c28p-5f), MLFloat16(0x1.4bp-2f), MLFloat16(0x1.2e4p-1f),
      MLFloat16(0x1.b54p-1f), MLFloat16(0x1.26p-2f), MLFloat16(0x1.628p-3f), MLFloat16(0x1.128p-3f),
      MLFloat16(0x1.fd4p-1f), MLFloat16(0x1.6f8p-3f), MLFloat16(0x1.454p-2f), MLFloat16(0x1.23p-1f),
      MLFloat16(0x1.324p-7f), MLFloat16(0x1.cd4p-1f), MLFloat16(0x1.f44p-1f), MLFloat16(0x1.1d4p-1f),
      MLFloat16(0x1.5b4p-4f), MLFloat16(0x1.55p-2f), MLFloat16(0x1.75p-1f), MLFloat16(0x1.23cp-3f),
      MLFloat16(0x1.1acp-1f), MLFloat16(0x1.178p-2f), MLFloat16(0x1.f3p-1f), MLFloat16(0x1.56p-1f),
      MLFloat16(0x1.05cp-2f), MLFloat16(0x1.bbcp-4f), MLFloat16(0x1.8d8p-1f), MLFloat16(0x1.90cp-1f),
      MLFloat16(0x1.86p-1f), MLFloat16(0x1.d44p-1f), MLFloat16(0x1.514p-1f), MLFloat16(0x1.23p-1f),
      MLFloat16(0x1.9d4p-3f), MLFloat16(0x1.658p-1f), MLFloat16(0x1.e78p-1f), MLFloat16(0x1.c7cp-1f),
      MLFloat16(0x1.fccp-1f), MLFloat16(0x1.a34p-1f), MLFloat16(0x1.17p-1f), MLFloat16(0x1.cep-2f),
      MLFloat16(0x1.c8p-1f), MLFloat16(0x1.f24p-1f), MLFloat16(0x1.2fcp-1f), MLFloat16(0x1.76cp-2f),
      MLFloat16(0x1.4acp-2f), MLFloat16(0x1.be4p-1f), MLFloat16(0x1.b98p-3f), MLFloat16(0x1.784p-1f),
      MLFloat16(0x1.768p-2f), MLFloat16(0x1.9a8p-1f), MLFloat16(0x1.90cp-1f), MLFloat16(0x1.67p-1f),
      MLFloat16(0x1.3ecp-1f), MLFloat16(0x1.f98p-2f), MLFloat16(0x1.ae4p-1f), MLFloat16(0x1.6c8p-1f),
      MLFloat16(0x1.c68p-2f), MLFloat16(0x1.fc8p-6f), MLFloat16(0x1.74p-2f), MLFloat16(0x1.764p-1f),
      MLFloat16(0x1.e7p-2f), MLFloat16(0x1.60cp-2f), MLFloat16(0x1.484p-1f), MLFloat16(0x1.028p-3f)};
  std::vector<int64_t> x_dims = {1, 3, 8, 8};
  std::vector<int64_t> expected_dims = {1, 3, 1, 1};
  std::vector<MLFloat16> expected_vals = {MLFloat16(0x1.fcp-1f), MLFloat16(0x1.f6cp-1f), MLFloat16(0x1.fd4p-1f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: result differs
}

TEST(PoolFp16Test, MaxPool_10_Dilation_1d) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3});
  test.AddAttribute("dilations", std::vector<int64_t>{3});

  std::vector<MLFloat16> x_vals = {
      MLFloat16(1.f), MLFloat16(3.f), MLFloat16(2.f), MLFloat16(4.f),
      MLFloat16(-1.f), MLFloat16(-3.f), MLFloat16(-2.f), MLFloat16(-4.f),
      MLFloat16(-6.f), MLFloat16(-5.f), MLFloat16(-4.f), MLFloat16(-2.f)};
  std::vector<int64_t> x_dims = {1, 1, 12};
  std::vector<int64_t> expected_dims = {1, 1, 6};
  std::vector<MLFloat16> expected_vals = {
      MLFloat16(4.f), MLFloat16(3.f), MLFloat16(2.f),
      MLFloat16(4.f), MLFloat16(-1.f), MLFloat16(-2.f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolFp16Test, MaxPool_DefaultDilations) {
  OpTester test("MaxPool", 12);

  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});

  std::vector<int64_t> x_dims = {1, 3, 3};
  std::vector<MLFloat16> x_vals = {
      MLFloat16(0.f), MLFloat16(1.f), MLFloat16(2.f),
      MLFloat16(3.f), MLFloat16(4.f), MLFloat16(5.f),
      MLFloat16(6.f), MLFloat16(7.f), MLFloat16(8.f)};

  std::vector<int64_t> expected_dims = {1, 3, 2};
  std::vector<MLFloat16> expected_vals = {
      MLFloat16(1.f), MLFloat16(2.f),
      MLFloat16(4.f), MLFloat16(5.f),
      MLFloat16(7.f), MLFloat16(8.f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolFp16Test, MaxPool_DilationPadding_1d) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1});
  test.AddAttribute("pads", std::vector<int64_t>{1, 1});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3});
  test.AddAttribute("dilations", std::vector<int64_t>{3});

  std::vector<MLFloat16> x_vals = {
      MLFloat16(1.f), MLFloat16(3.f), MLFloat16(2.f), MLFloat16(4.f),
      MLFloat16(-1.f), MLFloat16(-3.f), MLFloat16(-2.f), MLFloat16(-4.f),
      MLFloat16(-6.f), MLFloat16(-5.f), MLFloat16(-4.f), MLFloat16(-2.f)};
  std::vector<int64_t> x_dims = {1, 1, 12};
  std::vector<int64_t> expected_dims = {1, 1, 8};
  std::vector<MLFloat16> expected_vals = {
      MLFloat16(2.f), MLFloat16(4.f), MLFloat16(3.f), MLFloat16(2.f),
      MLFloat16(4.f), MLFloat16(-1.f), MLFloat16(-2.f), MLFloat16(-2.f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(PoolFp16Test, MaxPool_Dilation_2d) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});

  std::vector<MLFloat16> x_vals = {
      MLFloat16(1.f), MLFloat16(3.f), MLFloat16(2.f), MLFloat16(4.f), MLFloat16(-1.f),
      MLFloat16(5.f), MLFloat16(7.f), MLFloat16(6.f), MLFloat16(8.f), MLFloat16(-2.f),
      MLFloat16(9.f), MLFloat16(11.f), MLFloat16(10.f), MLFloat16(12.f), MLFloat16(-3.f),
      MLFloat16(13.f), MLFloat16(15.f), MLFloat16(14.f), MLFloat16(16.f), MLFloat16(-4.f)};
  std::vector<int64_t> x_dims = {1, 1, 4, 5};
  std::vector<int64_t> expected_dims = {1, 1, 2, 3};
  std::vector<MLFloat16> expected_vals = {
      MLFloat16(10.f), MLFloat16(12.f), MLFloat16(10.f), MLFloat16(14.f), MLFloat16(16.f), MLFloat16(14.f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolFp16Test, MaxPool_DilationPadding_2d) {
  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});

  std::vector<MLFloat16> x_vals = {
      MLFloat16(1.f), MLFloat16(3.f), MLFloat16(2.f), MLFloat16(4.f), MLFloat16(-1.f),
      MLFloat16(5.f), MLFloat16(7.f), MLFloat16(6.f), MLFloat16(8.f), MLFloat16(-2.f),
      MLFloat16(9.f), MLFloat16(11.f), MLFloat16(10.f), MLFloat16(12.f), MLFloat16(-3.f),
      MLFloat16(13.f), MLFloat16(15.f), MLFloat16(14.f), MLFloat16(16.f), MLFloat16(-4.f)};
  std::vector<int64_t> x_dims = {1, 1, 4, 5};
  std::vector<int64_t> expected_dims = {1, 1, 4, 5};
  std::vector<MLFloat16> expected_vals = {
      MLFloat16(7.f), MLFloat16(6.f), MLFloat16(8.f), MLFloat16(6.f), MLFloat16(8.f),
      MLFloat16(11.f), MLFloat16(10.f), MLFloat16(12.f), MLFloat16(10.f), MLFloat16(12.f),
      MLFloat16(15.f), MLFloat16(14.f), MLFloat16(16.f), MLFloat16(14.f), MLFloat16(16.f),
      MLFloat16(11.f), MLFloat16(10.f), MLFloat16(12.f), MLFloat16(10.f), MLFloat16(12.f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(PoolFp16Test, MaxPool_Dilation_Ceil0_2d) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});

  std::vector<MLFloat16> x_vals = {
      MLFloat16(1.f), MLFloat16(3.f), MLFloat16(2.f), MLFloat16(4.f), MLFloat16(-1.f),
      MLFloat16(5.f), MLFloat16(7.f), MLFloat16(6.f), MLFloat16(8.f), MLFloat16(-2.f),
      MLFloat16(9.f), MLFloat16(11.f), MLFloat16(10.f), MLFloat16(12.f), MLFloat16(-3.f),
      MLFloat16(13.f), MLFloat16(15.f), MLFloat16(14.f), MLFloat16(16.f), MLFloat16(-4.f)};
  std::vector<int64_t> x_dims = {1, 1, 4, 5};
  std::vector<int64_t> expected_dims = {1, 1, 1, 3};
  std::vector<MLFloat16> expected_vals = {MLFloat16(10.f), MLFloat16(12.f), MLFloat16(10.f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kAclExecutionProvider});
}

TEST(PoolFp16Test, MaxPool_Dilation_Ceil1_2d) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});
  test.AddAttribute("ceil_mode", (int64_t)1);

  std::vector<MLFloat16> x_vals = {
      MLFloat16(1.f), MLFloat16(3.f), MLFloat16(2.f), MLFloat16(4.f), MLFloat16(-1.f),
      MLFloat16(5.f), MLFloat16(7.f), MLFloat16(6.f), MLFloat16(8.f), MLFloat16(-2.f),
      MLFloat16(9.f), MLFloat16(11.f), MLFloat16(10.f), MLFloat16(12.f), MLFloat16(-3.f),
      MLFloat16(13.f), MLFloat16(15.f), MLFloat16(14.f), MLFloat16(16.f), MLFloat16(-4.f)};
  std::vector<int64_t> x_dims = {1, 1, 4, 5};
  std::vector<int64_t> expected_dims = {1, 1, 2, 3};
  std::vector<MLFloat16> expected_vals = {MLFloat16(10.f), MLFloat16(12.f), MLFloat16(10.f), MLFloat16(10.f), MLFloat16(12.f), MLFloat16(10.f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kAclExecutionProvider});
}

TEST(PoolTest, MaxPool_DilationPadding_3d) {
  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1, 1, 1});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2, 2});

  std::vector<MLFloat16> x_vals = {
      MLFloat16(1.f), MLFloat16(3.f), MLFloat16(2.f), MLFloat16(4.f), MLFloat16(-1.f),
      MLFloat16(5.f), MLFloat16(7.f), MLFloat16(6.f), MLFloat16(8.f), MLFloat16(-2.f),
      MLFloat16(9.f), MLFloat16(11.f), MLFloat16(10.f), MLFloat16(12.f), MLFloat16(-3.f),
      MLFloat16(13.f), MLFloat16(15.f), MLFloat16(14.f), MLFloat16(16.f), MLFloat16(-4.f),
      MLFloat16(1.f), MLFloat16(3.f), MLFloat16(2.f), MLFloat16(4.f), MLFloat16(-1.f),
      MLFloat16(5.f), MLFloat16(7.f), MLFloat16(6.f), MLFloat16(8.f), MLFloat16(-2.f),
      MLFloat16(9.f), MLFloat16(11.f), MLFloat16(10.f), MLFloat16(12.f), MLFloat16(-3.f),
      MLFloat16(13.f), MLFloat16(15.f), MLFloat16(14.f), MLFloat16(16.f), MLFloat16(-4.f)};
  std::vector<int64_t> x_dims = {1, 1, 2, 4, 5};
  std::vector<int64_t> expected_dims = {1, 1, 2, 4, 5};
  std::vector<MLFloat16> expected_vals = {
      MLFloat16(7.f), MLFloat16(6.f), MLFloat16(8.f), MLFloat16(6.f), MLFloat16(8.f),
      MLFloat16(11.f), MLFloat16(10.f), MLFloat16(12.f), MLFloat16(10.f), MLFloat16(12.f),
      MLFloat16(15.f), MLFloat16(14.f), MLFloat16(16.f), MLFloat16(14.f), MLFloat16(16.f),
      MLFloat16(11.f), MLFloat16(10.f), MLFloat16(12.f), MLFloat16(10.f), MLFloat16(12.f),
      MLFloat16(7.f), MLFloat16(6.f), MLFloat16(8.f), MLFloat16(6.f), MLFloat16(8.f),
      MLFloat16(11.f), MLFloat16(10.f), MLFloat16(12.f), MLFloat16(10.f), MLFloat16(12.f),
      MLFloat16(15.f), MLFloat16(14.f), MLFloat16(16.f), MLFloat16(14.f), MLFloat16(16.f),
      MLFloat16(11.f), MLFloat16(10.f), MLFloat16(12.f), MLFloat16(10.f), MLFloat16(12.f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(PoolFp16Test, AveragePool) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("AveragePool", 11);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{8, 8});

  std::vector<MLFloat16> x_vals = {
      MLFloat16(0x1.55cp-2f), MLFloat16(0x1.c24p-1f), MLFloat16(0x1.598p-2f),
      MLFloat16(0x1.554p-1f), MLFloat16(0x1.c54p-2f), MLFloat16(0x1.4b8p-1f),
      MLFloat16(0x1.89p-1f), MLFloat16(0x1.c3cp-1f), MLFloat16(0x1.c54p-1f),
      MLFloat16(0x1.7dcp-1f), MLFloat16(0x1.208p-2f), MLFloat16(0x1.bdcp-1f),
      MLFloat16(0x1.14cp-1f), MLFloat16(0x1.4a4p-6f), MLFloat16(0x1.9cp-1f),
      MLFloat16(0x1.35p-1f), MLFloat16(0x1.3f8p-1f), MLFloat16(0x1.418p-3f),
      MLFloat16(0x1.91p-3f), MLFloat16(0x1.6ccp-1f), MLFloat16(0x1.1bp-4f),
      MLFloat16(0x1.0cp-1f), MLFloat16(0x1.d18p-1f), MLFloat16(0x1.ba4p-2f),
      MLFloat16(0x1.9ecp-1f), MLFloat16(0x1.a8cp-4f), MLFloat16(0x1.8e4p-2f),
      MLFloat16(0x1.19cp-2f), MLFloat16(0x1.118p-2f), MLFloat16(0x1.60cp-5f),
      MLFloat16(0x1.7ap-2f), MLFloat16(0x1.bccp-1f), MLFloat16(0x1.43p-1f),
      MLFloat16(0x1.6c4p-1f), MLFloat16(0x1.03p-2f), MLFloat16(0x1.06cp-1f),
      MLFloat16(0x1.c34p-4f), MLFloat16(0x1.9ccp-3f), MLFloat16(0x1.c04p-2f),
      MLFloat16(0x1.838p-1f), MLFloat16(0x1.a08p-4f), MLFloat16(0x1.72cp-1f),
      MLFloat16(0x1.fcp-2f), MLFloat16(0x1.d5cp-1f), MLFloat16(0x1.36p-1f),
      MLFloat16(0x1.008p-2f), MLFloat16(0x1.e7p-2f), MLFloat16(0x1.cfp-1f),
      MLFloat16(0x1.e3cp-2f), MLFloat16(0x1.b38p-1f), MLFloat16(0x1.1d8p-3f),
      MLFloat16(0x1.f84p-1f), MLFloat16(0x1.57cp-1f), MLFloat16(0x1.cap-1f),
      MLFloat16(0x1.9a4p-2f), MLFloat16(0x1.68cp-4f), MLFloat16(0x1.c98p-1f),
      MLFloat16(0x1.61cp-2f), MLFloat16(0x1.9e4p-1f), MLFloat16(0x1.8acp-3f),
      MLFloat16(0x1.43cp-5f), MLFloat16(0x1.bep-6f), MLFloat16(0x1.9f8p-1f),
      MLFloat16(0x1.8acp-1f), MLFloat16(0x1.b8p-1f), MLFloat16(0x1.a1p-3f),
      MLFloat16(0x1.918p-1f), MLFloat16(0x1.2bp-2f), MLFloat16(0x1.034p-1f),
      MLFloat16(0x1.7bcp-1f), MLFloat16(0x1.b6p-4f), MLFloat16(0x1.2dcp-1f),
      MLFloat16(0x1.b7p-3f), MLFloat16(0x1.404p-3f), MLFloat16(0x1.59p-3f),
      MLFloat16(0x1.818p-1f), MLFloat16(0x1.2b4p-1f), MLFloat16(0x1.d38p-1f),
      MLFloat16(0x1.5b4p-1f), MLFloat16(0x1.b1p-3f), MLFloat16(0x1.d8cp-5f),
      MLFloat16(0x1.p-1f), MLFloat16(0x1.d9p-3f), MLFloat16(0x1.a8p-5f),
      MLFloat16(0x1.64cp-1f), MLFloat16(0x1.e3cp-2f), MLFloat16(0x1.cf4p-4f),
      MLFloat16(0x1.3ccp-1f), MLFloat16(0x1.cb4p-1f), MLFloat16(0x1.374p-1f),
      MLFloat16(0x1.3acp-2f), MLFloat16(0x1.43cp-4f), MLFloat16(0x1.908p-5f),
      MLFloat16(0x1.fc8p-3f), MLFloat16(0x1.f8p-1f), MLFloat16(0x1.cfp-2f),
      MLFloat16(0x1.128p-2f), MLFloat16(0x1.84cp-1f), MLFloat16(0x1.834p-2f),
      MLFloat16(0x1.3dp-2f), MLFloat16(0x1.c48p-1f), MLFloat16(0x1.7ecp-4f),
      MLFloat16(0x1.84cp-2f), MLFloat16(0x1.93p-4f), MLFloat16(0x1.334p-1f),
      MLFloat16(0x1.97p-1f), MLFloat16(0x1.d68p-2f), MLFloat16(0x1.1b8p-1f),
      MLFloat16(0x1.8ep-2f), MLFloat16(0x1.a14p-2f), MLFloat16(0x1.8b8p-2f),
      MLFloat16(0x1.c88p-1f), MLFloat16(0x1.bdp-3f), MLFloat16(0x1.57cp-1f),
      MLFloat16(0x1.278p-1f), MLFloat16(0x1.f9p-1f), MLFloat16(0x1.3acp-5f),
      MLFloat16(0x1.424p-3f), MLFloat16(0x1.7e8p-4f), MLFloat16(0x1.db8p-1f),
      MLFloat16(0x1.49p-3f), MLFloat16(0x1.a64p-1f), MLFloat16(0x1.b1p-2f),
      MLFloat16(0x1.f98p-1f), MLFloat16(0x1.e54p-1f), MLFloat16(0x1.d94p-1f),
      MLFloat16(0x1.ff4p-1f), MLFloat16(0x1.50cp-2f), MLFloat16(0x1.85p-7f),
      MLFloat16(0x1.f7cp-1f), MLFloat16(0x1.7f8p-4f), MLFloat16(0x1.56cp-2f),
      MLFloat16(0x1.47p-1f), MLFloat16(0x1.f8cp-1f), MLFloat16(0x1.de8p-2f),
      MLFloat16(0x1.e8p-1f), MLFloat16(0x1.458p-3f), MLFloat16(0x1.6f4p-1f),
      MLFloat16(0x1.91cp-6f), MLFloat16(0x1.a4cp-1f), MLFloat16(0x1.274p-3f),
      MLFloat16(0x1.cfp-2f), MLFloat16(0x1.c58p-2f), MLFloat16(0x1.fc8p-1f),
      MLFloat16(0x1.b38p-1f), MLFloat16(0x1.0b4p-3f), MLFloat16(0x1.4p-4f),
      MLFloat16(0x1.e3p-1f), MLFloat16(0x1.fb8p-6f), MLFloat16(0x1.bb4p-3f),
      MLFloat16(0x1.e6p-1f), MLFloat16(0x1.258p-1f), MLFloat16(0x1.2f8p-1f),
      MLFloat16(0x1.88p-1f), MLFloat16(0x1.2p-1f), MLFloat16(0x1.68cp-7f),
      MLFloat16(0x1.75cp-1f), MLFloat16(0x1.8f4p-2f), MLFloat16(0x1.5d8p-4f),
      MLFloat16(0x1.bbp-2f), MLFloat16(0x1.afcp-1f), MLFloat16(0x1.0f8p-1f),
      MLFloat16(0x1.4a4p-1f), MLFloat16(0x1.518p-3f), MLFloat16(0x1.6fcp-2f),
      MLFloat16(0x1.2d4p-5f), MLFloat16(0x1.23cp-1f), MLFloat16(0x1.2b4p-1f),
      MLFloat16(0x1.ee4p-1f), MLFloat16(0x1.cf8p-1f), MLFloat16(0x1.2c4p-2f),
      MLFloat16(0x1.0bcp-2f), MLFloat16(0x1.ee8p-2f), MLFloat16(0x1.21p-1f),
      MLFloat16(0x1.ad4p-3f), MLFloat16(0x1.7f4p-2f), MLFloat16(0x1.f8p-2f),
      MLFloat16(0x1.90cp-1f), MLFloat16(0x1.24p-2f), MLFloat16(0x1.dd8p-2f),
      MLFloat16(0x1.974p-3f), MLFloat16(0x1.9dcp-3f), MLFloat16(0x1.46p-2f),
      MLFloat16(0x1.cfcp-2f), MLFloat16(0x1.204p-2f), MLFloat16(0x1.a4p-1f),
      MLFloat16(0x1.fc4p-2f), MLFloat16(0x1.dep-2f), MLFloat16(0x1.7b4p-1f),
      MLFloat16(0x1.9b8p-2f), MLFloat16(0x1.b3p-3f), MLFloat16(0x1.e08p-2f)};
  std::vector<int64_t> x_dims = {1, 3, 8, 8};
  std::vector<int64_t> expected_dims = {1, 3, 1, 1};
  std::vector<MLFloat16> expected_vals = {MLFloat16(0.514681101f), MLFloat16(0.485104561f), MLFloat16(0.475683808f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolFp16Test, AveragePool_IncludePadPixel) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("AveragePool", 11);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("count_include_pad", (int64_t)1);
  std::vector<MLFloat16> x_vals = {
      MLFloat16(0x1.55cp-2f), MLFloat16(0x1.c24p-1f), MLFloat16(0x1.598p-2f),
      MLFloat16(0x1.554p-1f), MLFloat16(0x1.c54p-2f), MLFloat16(0x1.4b8p-1f),
      MLFloat16(0x1.89p-1f), MLFloat16(0x1.c3cp-1f), MLFloat16(0x1.c54p-1f)};

  std::vector<int64_t> x_dims = {1, 1, 3, 3};
  std::vector<int64_t> expected_dims = {1, 1, 4, 4};
  std::vector<MLFloat16> expected_vals = {
      MLFloat16(0x1.55cp-4f), MLFloat16(0x1.369p-2f), MLFloat16(0x1.378p-2f), MLFloat16(0x1.598p-4f),
      MLFloat16(0x1.001p-2f), MLFloat16(0x1.294p-1f), MLFloat16(0x1.2748p-1f), MLFloat16(0x1.f84p-3f),
      MLFloat16(0x1.6f2p-2f), MLFloat16(0x1.6128p-1f), MLFloat16(0x1.6dc8p-1f), MLFloat16(0x1.886p-2f),
      MLFloat16(0x1.89p-3f), MLFloat16(0x1.a66p-2f), MLFloat16(0x1.c48p-2f), MLFloat16(0x1.c54p-3f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// test 'strides' attribute not specified
TEST(PoolFp16Test, AveragePool_DefaultStrides) {
  OpTester test("AveragePool", 11);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});
  std::vector<MLFloat16> x_vals = {
      MLFloat16(0.f), MLFloat16(1.f), MLFloat16(2.f),
      MLFloat16(3.f), MLFloat16(4.f), MLFloat16(5.f),
      MLFloat16(6.f), MLFloat16(7.f), MLFloat16(8.f)};

  std::vector<int64_t> x_dims = {1, 3, 3};
  std::vector<int64_t> expected_dims = {1, 3, 2};
  std::vector<MLFloat16> expected_vals = {
      MLFloat16(0.5f), MLFloat16(1.5f),
      MLFloat16(3.5f), MLFloat16(4.5f),
      MLFloat16(6.5f), MLFloat16(7.5f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolFp16Test, AveragePool_10_ceil1_2d) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("AveragePool", 11);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{3, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("ceil_mode", (int64_t)1);

  std::vector<MLFloat16> x_vals = {
      MLFloat16(1.f), MLFloat16(3.f), MLFloat16(2.f), MLFloat16(4.f),
      MLFloat16(5.f), MLFloat16(7.f), MLFloat16(6.f), MLFloat16(8.f),
      MLFloat16(9.f), MLFloat16(11.f), MLFloat16(10.f), MLFloat16(12.f),
      MLFloat16(13.f), MLFloat16(15.f), MLFloat16(14.f), MLFloat16(16.f)};
  std::vector<int64_t> x_dims = {1, 1, 4, 4};
  std::vector<int64_t> expected_dims = {1, 1, 2, 3};
  std::vector<MLFloat16> expected_vals = {
      MLFloat16(4.0f), MLFloat16(4.5f), MLFloat16(5.0f), MLFloat16(14.0f), MLFloat16(14.5f), MLFloat16(15.0f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kAclExecutionProvider});
}

TEST(PoolFp16Test, GlobalAveragePool) {
  OpTester test("GlobalAveragePool");

  std::vector<MLFloat16> x_vals = {
      MLFloat16(0x1.55cp-2f), MLFloat16(0x1.c24p-1f), MLFloat16(0x1.598p-2f),
      MLFloat16(0x1.554p-1f), MLFloat16(0x1.c54p-2f), MLFloat16(0x1.4b8p-1f),
      MLFloat16(0x1.890p-1f), MLFloat16(0x1.c3cp-1f), MLFloat16(0x1.c54p-1f),
      MLFloat16(0x1.7dcp-1f), MLFloat16(0x1.208p-2f), MLFloat16(0x1.bdcp-1f),
      MLFloat16(0x1.14cp-1f), MLFloat16(0x1.4a4p-6f), MLFloat16(0x1.9c0p-1f),
      MLFloat16(0x1.350p-1f), MLFloat16(0x1.3f8p-1f), MLFloat16(0x1.418p-3f),
      MLFloat16(0x1.910p-3f), MLFloat16(0x1.6ccp-1f), MLFloat16(0x1.1b0p-4f),
      MLFloat16(0x1.0c0p-1f), MLFloat16(0x1.d18p-1f), MLFloat16(0x1.ba4p-2f),
      MLFloat16(0x1.9ecp-1f), MLFloat16(0x1.a8cp-4f), MLFloat16(0x1.8e4p-2f),
      MLFloat16(0x1.19cp-2f), MLFloat16(0x1.118p-2f), MLFloat16(0x1.60cp-5f),
      MLFloat16(0x1.7a0p-2f), MLFloat16(0x1.bccp-1f), MLFloat16(0x1.430p-1f),
      MLFloat16(0x1.6c4p-1f), MLFloat16(0x1.030p-2f), MLFloat16(0x1.06cp-1f),
      MLFloat16(0x1.c34p-4f), MLFloat16(0x1.9ccp-3f), MLFloat16(0x1.c04p-2f),
      MLFloat16(0x1.838p-1f), MLFloat16(0x1.a08p-4f), MLFloat16(0x1.72cp-1f),
      MLFloat16(0x1.fc0p-2f), MLFloat16(0x1.d5cp-1f), MLFloat16(0x1.360p-1f),
      MLFloat16(0x1.008p-2f), MLFloat16(0x1.e70p-2f), MLFloat16(0x1.cf0p-1f),
      MLFloat16(0x1.e3cp-2f), MLFloat16(0x1.b38p-1f), MLFloat16(0x1.1d8p-3f),
      MLFloat16(0x1.f84p-1f), MLFloat16(0x1.57cp-1f), MLFloat16(0x1.ca0p-1f),
      MLFloat16(0x1.9a4p-2f), MLFloat16(0x1.68cp-4f), MLFloat16(0x1.c98p-1f),
      MLFloat16(0x1.61cp-2f), MLFloat16(0x1.9e4p-1f), MLFloat16(0x1.8acp-3f),
      MLFloat16(0x1.43cp-5f), MLFloat16(0x1.be0p-6f), MLFloat16(0x1.9f8p-1f),
      MLFloat16(0x1.8acp-1f), MLFloat16(0x1.b80p-1f), MLFloat16(0x1.a10p-3f),
      MLFloat16(0x1.918p-1f), MLFloat16(0x1.2b0p-2f), MLFloat16(0x1.034p-1f),
      MLFloat16(0x1.7bcp-1f), MLFloat16(0x1.b60p-4f), MLFloat16(0x1.2dcp-1f),
      MLFloat16(0x1.b70p-3f), MLFloat16(0x1.404p-3f), MLFloat16(0x1.590p-3f),
      MLFloat16(0x1.818p-1f), MLFloat16(0x1.2b4p-1f), MLFloat16(0x1.d38p-1f),
      MLFloat16(0x1.5b4p-1f), MLFloat16(0x1.b10p-3f), MLFloat16(0x1.d8cp-5f),
      MLFloat16(0x1.000p-1f), MLFloat16(0x1.d90p-3f), MLFloat16(0x1.a80p-5f),
      MLFloat16(0x1.64cp-1f), MLFloat16(0x1.e3cp-2f), MLFloat16(0x1.cf4p-4f),
      MLFloat16(0x1.3ccp-1f), MLFloat16(0x1.cb4p-1f), MLFloat16(0x1.374p-1f),
      MLFloat16(0x1.3acp-2f), MLFloat16(0x1.43cp-4f), MLFloat16(0x1.908p-5f),
      MLFloat16(0x1.fc8p-3f), MLFloat16(0x1.f80p-1f), MLFloat16(0x1.cf0p-2f),
      MLFloat16(0x1.128p-2f), MLFloat16(0x1.84cp-1f), MLFloat16(0x1.834p-2f),
      MLFloat16(0x1.3d0p-2f), MLFloat16(0x1.c48p-1f), MLFloat16(0x1.7ecp-4f),
      MLFloat16(0x1.84cp-2f), MLFloat16(0x1.930p-4f), MLFloat16(0x1.334p-1f),
      MLFloat16(0x1.970p-1f), MLFloat16(0x1.d68p-2f), MLFloat16(0x1.1b8p-1f),
      MLFloat16(0x1.8e0p-2f), MLFloat16(0x1.a14p-2f), MLFloat16(0x1.8b8p-2f),
      MLFloat16(0x1.c88p-1f), MLFloat16(0x1.bd0p-3f), MLFloat16(0x1.57cp-1f),
      MLFloat16(0x1.278p-1f), MLFloat16(0x1.f90p-1f), MLFloat16(0x1.3acp-5f),
      MLFloat16(0x1.424p-3f), MLFloat16(0x1.7e8p-4f), MLFloat16(0x1.db8p-1f),
      MLFloat16(0x1.490p-3f), MLFloat16(0x1.a64p-1f), MLFloat16(0x1.b10p-2f),
      MLFloat16(0x1.f98p-1f), MLFloat16(0x1.e54p-1f), MLFloat16(0x1.d94p-1f),
      MLFloat16(0x1.ff4p-1f), MLFloat16(0x1.50cp-2f), MLFloat16(0x1.850p-7f),
      MLFloat16(0x1.f7cp-1f), MLFloat16(0x1.7f8p-4f), MLFloat16(0x1.56cp-2f),
      MLFloat16(0x1.470p-1f), MLFloat16(0x1.f8cp-1f), MLFloat16(0x1.de8p-2f),
      MLFloat16(0x1.e80p-1f), MLFloat16(0x1.458p-3f), MLFloat16(0x1.6f4p-1f),
      MLFloat16(0x1.91cp-6f), MLFloat16(0x1.a4cp-1f), MLFloat16(0x1.274p-3f),
      MLFloat16(0x1.cf0p-2f), MLFloat16(0x1.c58p-2f), MLFloat16(0x1.fc8p-1f),
      MLFloat16(0x1.b38p-1f), MLFloat16(0x1.0b4p-3f), MLFloat16(0x1.400p-4f),
      MLFloat16(0x1.e30p-1f), MLFloat16(0x1.fb8p-6f), MLFloat16(0x1.bb4p-3f),
      MLFloat16(0x1.e60p-1f), MLFloat16(0x1.258p-1f), MLFloat16(0x1.2f8p-1f),
      MLFloat16(0x1.880p-1f), MLFloat16(0x1.200p-1f), MLFloat16(0x1.68cp-7f),
      MLFloat16(0x1.75cp-1f), MLFloat16(0x1.8f4p-2f), MLFloat16(0x1.5d8p-4f),
      MLFloat16(0x1.bb0p-2f), MLFloat16(0x1.afcp-1f), MLFloat16(0x1.0f8p-1f),
      MLFloat16(0x1.4a4p-1f), MLFloat16(0x1.518p-3f), MLFloat16(0x1.6fcp-2f),
      MLFloat16(0x1.2d4p-5f), MLFloat16(0x1.23cp-1f), MLFloat16(0x1.2b4p-1f),
      MLFloat16(0x1.ee4p-1f), MLFloat16(0x1.cf8p-1f), MLFloat16(0x1.2c4p-2f),
      MLFloat16(0x1.0bcp-2f), MLFloat16(0x1.ee8p-2f), MLFloat16(0x1.210p-1f),
      MLFloat16(0x1.ad4p-3f), MLFloat16(0x1.7f4p-2f), MLFloat16(0x1.f80p-2f),
      MLFloat16(0x1.90cp-1f), MLFloat16(0x1.240p-2f), MLFloat16(0x1.dd8p-2f),
      MLFloat16(0x1.974p-3f), MLFloat16(0x1.9dcp-3f), MLFloat16(0x1.460p-2f),
      MLFloat16(0x1.cfcp-2f), MLFloat16(0x1.204p-2f), MLFloat16(0x1.a40p-1f),
      MLFloat16(0x1.fc4p-2f), MLFloat16(0x1.de0p-2f), MLFloat16(0x1.7b4p-1f),
      MLFloat16(0x1.9b8p-2f), MLFloat16(0x1.b30p-3f), MLFloat16(0x1.e08p-2f)};
  std::vector<int64_t> x_dims = {1, 3, 8, 8};
  std::vector<int64_t> expected_dims = {1, 3, 1, 1};
  std::vector<MLFloat16> expected_vals = {MLFloat16(0x1.078448p-1f), MLFloat16(0x1.f0bf4p-2f), MLFloat16(0x1.e719a8p-2f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

#endif