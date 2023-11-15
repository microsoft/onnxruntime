// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORT_MINIMAL_BUILD

#include "core/common/span_utils.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/util/qmath.h"

#include <chrono>
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace onnxruntime {
namespace test {

void MlasJblasQ4Test(int64_t M, int64_t N, int64_t K, int block_size, bool is_asym, MLAS_COMPUTE_TYPE acc_lvl, float err = 0.1f) {
  // (M x K) X (K x N)

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("accuracy_level", int64_t(acc_lvl));
  test.AddAttribute<int64_t>("block_size", int64_t(block_size));
  test.AddAttribute<int64_t>("bits", 4);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("K", K);

  std::vector<float> input0_vals(M * K);
  float fv = -135.f;
  for (auto& f : input0_vals) {
    f = fv / 127;
    fv++;
    if (fv > 135.f) {
      fv = -135.f;
    }
  }

  size_t kblks = K / block_size;
  std::vector<uint8_t> input1_vals(N * K / 2);
  for (size_t i = 0; i < N * K / 2; i++) {
    input1_vals[i] = uint8_t(i);
  }
  std::vector<float> input2_vals(N * kblks, 0.002f);
  for (size_t i = 0; i < N * kblks; i++) {
    input2_vals[i] += (i % 100) * 0.00003f;
  }
  std::vector<uint8_t> input3_vals(N * kblks / 2, (uint8_t)0x88);

  std::vector<float> input1_f_vals(N * K);
  if (is_asym) {
    for (size_t i = 0; i < N * kblks; i += 2) {
      input3_vals[i / 2] = uint8_t(i + 1);
    }
    for (size_t i = 0; i < K; i += 2) {
      for (size_t j = 0; j < N; j++) {
        auto srcv = input1_vals[j * K / 2 + i / 2];
        auto koff = i % (block_size * 2);
        auto zpv = input3_vals[j * kblks / 2 + i / block_size / 2];
        auto zp0 = koff < block_size ? (zpv & 0xf) - 8 : ((zpv & 0xf0) >> 4) - 8;
        auto src0 = (srcv & 0xf) - 8;
        auto src1 = ((srcv & 0xf0) >> 4) - 8;
        auto scale0 = input2_vals[j * kblks + i / block_size];
        auto scale1 = input2_vals[j * kblks + (i + 1) / block_size];
        input1_f_vals[i * N + j] = (float(src0) - zp0) * scale0;
        input1_f_vals[(i + 1) * N + j] = (float(src1) - zp0) * scale1;
      }
    }
  } else {
    for (size_t i = 0; i < K; i += 2) {
      for (size_t j = 0; j < N; j++) {
        auto srcv = input1_vals[j * K / 2 + i / 2];
        auto src0 = (srcv & 0xf) - 8;
        auto src1 = ((srcv & 0xf0) >> 4) - 8;
        auto scale0 = input2_vals[j * kblks + i / block_size];
        auto scale1 = input2_vals[j * kblks + (i + 1) / block_size];
        input1_f_vals[i * N + j] = float(src0) * scale0;
        input1_f_vals[(i + 1) * N + j] = float(src1) * scale1;
      }
    }
  }

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += input0_vals[m * K + k] * input1_f_vals[k * N + n];
      }
      expected_vals[m * N + n] = sum;
    }
  }

  test.AddInput<float>("A", {M, K}, input0_vals, false);

  test.AddInput<uint8_t>("B", {N, (int64_t)kblks, (int64_t)(block_size / 2)}, input1_vals, true);
  test.AddInput<float>("scales", {N, (int64_t)kblks}, input2_vals, true);
  if (is_asym) {
    test.AddInput<uint8_t>("zero_points", {N, (int64_t)(kblks / 2)}, input3_vals, true);
  }
  test.AddOutput<float>("Y", {M, N}, expected_vals, false, 0.f, err);

  OrtValue b, scale, zp;
  Tensor::InitOrtValue(DataTypeImpl::GetType<uint8_t>(), TensorShape({N, (int64_t)kblks, (int64_t)(block_size / 2)}),
                       input1_vals.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), b);

  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape({N, (int64_t)kblks}),
                       input2_vals.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), scale);
  if (is_asym) {
    Tensor::InitOrtValue(DataTypeImpl::GetType<uint8_t>(), TensorShape({N, (int64_t)(kblks / 2)}),
                         input3_vals.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), zp);
  }
  SessionOptions so;
  // Set up B as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("B", &b), Status::OK());
  ASSERT_EQ(so.AddInitializer("scales", &scale), Status::OK());
  if (is_asym) {
    ASSERT_EQ(so.AddInitializer("zero_points", &zp), Status::OK());
  }

  // We want all sessions running using this OpTester to be able to share pre-packed weights if applicable
  test.EnableSharingOfPrePackedWeightsAcrossSessions();

  // Pre-packing is limited just to the CPU EP for now and we will only test the CPU EP
  // and we want to ensure that it is available in this build
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  test.Config(so)
      .ConfigEps(cpu_ep())
      .RunWithConfig();
}

#ifdef MLAS_JBLAS
TEST(MatMulNBits, MlasJblasQ4Fp32G128Sym) {
  MlasJblasQ4Test(2, 4096, 4096, 128, false, CompFp32);
}

TEST(MatMulNBits, MlasJblasQ4Fp32G32Sym) {
  MlasJblasQ4Test(2, 4096, 4096, 32, false, CompFp32);
}

TEST(MatMulNBits, MlasJblasQ4Fp32G32Asym) {
  MlasJblasQ4Test(2, 4096, 4096, 32, true, CompFp32);
}

TEST(MatMulNBits, MlasJblasQ4Int8G128Sym) {
  MlasJblasQ4Test(2, 4096, 4096, 128, false, CompInt8);
}

TEST(MatMulNBits, MlasJblasQ4Int8G1024) {
  MlasJblasQ4Test(2, 4096, 4096, 1024, false, CompInt8);
}

TEST(MatMulNBits, MlasJblasQ4Int8GPerN) {
  MlasJblasQ4Test(2, 4096, 4096, 4096, false, CompInt8);
}
#endif

}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
