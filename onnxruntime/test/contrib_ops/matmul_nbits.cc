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

TEST(MatMulNBits, MlasJblasQ4G32Sym) {
  // (128 x 1024) X (1024 x 1024)
  constexpr int64_t M = 2;
  constexpr int64_t N = 4096;
  constexpr int64_t K = 4096;
  constexpr int BlkSize = 32;
  //constexpr bool IsAsym = false;
  //constexpr MLAS_COMPUTE_TYPE CompType = CompFp32;

  OpTester test("MatMulNBits", 1, kMSDomain);
  //test.AddAttribute<int64_t>("compute_type", int64_t(CompType));
  test.AddAttribute<int64_t>("block_size", BlkSize);
  //test.AddAttribute<int64_t>("is_asym", IsAsym ? 1 : 0);
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

  size_t kblks = K / BlkSize;
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
  for (size_t i = 0; i < K; i += 2) {
    for (size_t j = 0; j < N; j++) {
      auto srcv = input1_vals[j * K / 2 + i / 2];
      auto src0 = (srcv & 0xf) - 8;
      auto src1 = ((srcv & 0xf0) >> 4) - 8;
      auto scale0 = input2_vals[j * kblks + i / BlkSize];
      auto scale1 = input2_vals[j * kblks + (i + 1) / BlkSize];
      input1_f_vals[i * N + j] = float(src0) * scale0;
      input1_f_vals[(i + 1) * N + j] = float(src1) * scale1;
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
  test.AddInput<uint8_t>("B_Q", {N, (int64_t)kblks, (int64_t)(BlkSize / 2)}, input1_vals, true);
  test.AddInput<float>("B_Scale", {N, (int64_t)kblks}, input2_vals, true);
  test.AddInput<uint8_t>("B_Zp", {N, (int64_t)(kblks / 2)}, input3_vals, true);

  test.AddOutput<float>("Y", {M, N}, expected_vals, false, 0.1f, 0.1f);
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
