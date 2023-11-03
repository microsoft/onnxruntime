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
  constexpr MLAS_COMPUTE_TYPE CompType = CompFp32;

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("compute_type", int64_t(CompType));
  test.AddAttribute<int64_t>("block_size", BlkSize);
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

  test.AddInput<uint8_t>("B", {N, (int64_t)kblks, (int64_t)(BlkSize / 2)}, input1_vals, true);
  test.AddInput<float>("scales", {N, (int64_t)kblks}, input2_vals, true);
  test.AddOutput<float>("Y", {M, N}, expected_vals, false, 0.1f, 0.1f);

  OrtValue b, scale;
  Tensor::InitOrtValue(DataTypeImpl::GetType<uint8_t>(), TensorShape({N, (int64_t)kblks, (int64_t)(BlkSize / 2)}),
                       input1_vals.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), b);

  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape({N, (int64_t)kblks}),
                       input2_vals.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), scale);
  SessionOptions so;
  // Set up B as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("B", &b), Status::OK());
  ASSERT_EQ(so.AddInitializer("scales", &scale), Status::OK());

  // We want all sessions running using this OpTester to be able to share pre-packed weights if applicable
  test.EnableSharingOfPrePackedWeightsAcrossSessions();

  // Pre-packing is limited just to the CPU EP for now and we will only test the CPU EP
  // and we want to ensure that it is available in this build
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  size_t number_of_pre_packed_weights_counter_session_1 = 0;
  size_t number_of_shared_pre_packed_weights_counter = 0;

  // Session 1
  {
    test.Config(so)
        .ConfigEps(cpu_ep())
        .RunWithConfig(&number_of_pre_packed_weights_counter_session_1, &number_of_shared_pre_packed_weights_counter);
    // Assert that no pre-packed weights have been shared thus far
    // ASSERT_EQ(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
  }

  // auto number_of_elements_in_shared_prepacked_buffers_container =
  //     test.GetNumPrePackedWeightsShared();
  //  Assert that the number of elements in the shared container
  //  is the same as the number of weights that have been pre-packed
  //  ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_elements_in_shared_prepacked_buffers_container);

  // On some platforms/architectures MLAS may choose to not do any pre-packing and the number of elements
  // that have been pre-packed will be zero in which case we do not continue with the testing
  // of "sharing" of pre-packed weights as there are no pre-packed weights to be shared at all.
  if (number_of_pre_packed_weights_counter_session_1 == 0)
    return;

  // Session 2
  {
    size_t number_of_pre_packed_weights_counter_session_2 = 0;
    test.Config(so)
        .ConfigEps(cpu_ep())
        .RunWithConfig(&number_of_pre_packed_weights_counter_session_2, &number_of_shared_pre_packed_weights_counter);

    // Assert that the same number of weights were pre-packed in both sessions
    // ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_pre_packed_weights_counter_session_2);

    // Assert that the number of pre-packed weights that were shared equals
    // the number of pre-packed weights in the second session
    /*ASSERT_EQ(number_of_pre_packed_weights_counter_session_2,
              static_cast<size_t>(number_of_shared_pre_packed_weights_counter));*/
  }
}

TEST(MatMulNBits, MlasJblasQ4G32Asym) {
  // (128 x 1024) X (1024 x 1024)
  constexpr int64_t M = 2;
  constexpr int64_t N = 4096;
  constexpr int64_t K = 4096;
  constexpr int BlkSize = 32;
  constexpr MLAS_COMPUTE_TYPE CompType = CompFp32;

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("compute_type", int64_t(CompType));
  test.AddAttribute<int64_t>("block_size", BlkSize);
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
  for (size_t i = 0; i < N * kblks; i += 2) {
    input3_vals[i / 2] = uint8_t(i + 1);
  }
  std::vector<float> input1_f_vals(N * K);
  for (size_t i = 0; i < K; i += 2) {
    for (size_t j = 0; j < N; j++) {
      auto srcv = input1_vals[j * K / 2 + i / 2];
      auto koff = i % BlkSize;
      auto zpv = input3_vals[j * kblks / 2 + i / BlkSize / 2];
      auto zp0 = koff < BlkSize / 2 ? (zpv & 0xf) - 8 : ((zpv & 0xf0) >> 4) - 8;
      auto src0 = (srcv & 0xf) - 8;
      auto src1 = ((srcv & 0xf0) >> 4) - 8;
      auto scale0 = input2_vals[j * kblks + i / BlkSize];
      auto scale1 = input2_vals[j * kblks + (i + 1) / BlkSize];
      input1_f_vals[i * N + j] = (float(src0) - zp0) * scale0;
      input1_f_vals[(i + 1) * N + j] = (float(src1) - zp0) * scale1;
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

  test.AddInput<uint8_t>("B", {N, (int64_t)kblks, (int64_t)(BlkSize / 2)}, input1_vals, true);
  test.AddInput<float>("scales", {N, (int64_t)kblks}, input2_vals, true);
  test.AddInput<uint8_t>("zero_points", {N, (int64_t)(kblks / 2)}, input3_vals, true);
  test.AddOutput<float>("Y", {M, N}, expected_vals, false, 0.1f, 0.1f);

  OrtValue b, scale, zp;
  Tensor::InitOrtValue(DataTypeImpl::GetType<uint8_t>(), TensorShape({N, (int64_t)kblks, (int64_t)(BlkSize / 2)}),
                       input1_vals.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), b);

  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape({N, (int64_t)kblks}),
                       input2_vals.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), scale);

  Tensor::InitOrtValue(DataTypeImpl::GetType<uint8_t>(), TensorShape({N, (int64_t)(kblks / 2)}),
                       input3_vals.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), zp);
  SessionOptions so;
  // Set up B as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("B", &b), Status::OK());
  ASSERT_EQ(so.AddInitializer("scales", &scale), Status::OK());
  ASSERT_EQ(so.AddInitializer("zero_points", &zp), Status::OK());

  // We want all sessions running using this OpTester to be able to share pre-packed weights if applicable
  test.EnableSharingOfPrePackedWeightsAcrossSessions();

  // Pre-packing is limited just to the CPU EP for now and we will only test the CPU EP
  // and we want to ensure that it is available in this build
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  size_t number_of_pre_packed_weights_counter_session_1 = 0;
  size_t number_of_shared_pre_packed_weights_counter = 0;

  // Session 1
  {
    test.Config(so)
        .ConfigEps(cpu_ep())
        .RunWithConfig(&number_of_pre_packed_weights_counter_session_1, &number_of_shared_pre_packed_weights_counter);
    // Assert that no pre-packed weights have been shared thus far
    // ASSERT_EQ(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
  }

  // auto number_of_elements_in_shared_prepacked_buffers_container =
  //     test.GetNumPrePackedWeightsShared();
  //  Assert that the number of elements in the shared container
  //  is the same as the number of weights that have been pre-packed
  //  ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_elements_in_shared_prepacked_buffers_container);

  // On some platforms/architectures MLAS may choose to not do any pre-packing and the number of elements
  // that have been pre-packed will be zero in which case we do not continue with the testing
  // of "sharing" of pre-packed weights as there are no pre-packed weights to be shared at all.
  if (number_of_pre_packed_weights_counter_session_1 == 0)
    return;

  // Session 2
  {
    size_t number_of_pre_packed_weights_counter_session_2 = 0;
    test.Config(so)
        .ConfigEps(cpu_ep())
        .RunWithConfig(&number_of_pre_packed_weights_counter_session_2, &number_of_shared_pre_packed_weights_counter);

    // Assert that the same number of weights were pre-packed in both sessions
    // ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_pre_packed_weights_counter_session_2);

    // Assert that the number of pre-packed weights that were shared equals
    // the number of pre-packed weights in the second session
    /*ASSERT_EQ(number_of_pre_packed_weights_counter_session_2,
              static_cast<size_t>(number_of_shared_pre_packed_weights_counter));*/
  }
}
}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
