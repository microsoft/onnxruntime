// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_util_kernels.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"

#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.h"
#endif

namespace onnxruntime {
namespace test {
namespace {

namespace qmoe = contrib::cuda;

struct CudaBuffer {
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaBuffer);

  explicit CudaBuffer(size_t size) : bytes(size) {
    CUDA_CALL_THROW(cudaMalloc(&data, bytes));
  }

  ~CudaBuffer() noexcept {
    const cudaError_t status = cudaFree(data);
    if (status != cudaSuccess) {
      std::fprintf(stderr, "cudaFree failed during QMoE test cleanup: %s\n", cudaGetErrorString(status));
    }
  }

  template <typename T>
  T* As() { return static_cast<T*>(data); }

  void Upload(const void* source) {
    CUDA_CALL_THROW(cudaMemcpy(data, source, bytes, cudaMemcpyHostToDevice));
  }

  void Download(void* destination) {
    CUDA_CALL_THROW(cudaMemcpy(destination, data, bytes, cudaMemcpyDeviceToHost));
  }

  void* data = nullptr;
  size_t bytes;
};

template <typename T>
void TestDynamicFp8Quantization() {
  std::array<T, 384> input{};
  input[128] = T(448.0f);
  input[129] = T(-448.0f);
  input[130] = T(1.0625f);
  input[131] = T(1.1875f);
  input[256] = T(0.65625f);
  input[257] = T(57.0f / 32768.0f);
  CudaBuffer source(sizeof(input));
  CudaBuffer quantized(input.size());
  CudaBuffer scales(3 * sizeof(float));
  source.Upload(input.data());
  qmoe::LaunchQMoEDynamicFp8Quantize(source.As<T>(), quantized.As<uint8_t>(), scales.As<float>(),
                                     1, 384, 0, nullptr);
  std::array<uint8_t, 384> codes{};
  std::array<float, 3> actual_scales{};
  quantized.Download(codes.data());
  scales.Download(actual_scales.data());
  EXPECT_FLOAT_EQ(actual_scales[0], 1.0e-10f / 448.0f);
  EXPECT_FLOAT_EQ(actual_scales[1], 1.0f);
  for (int col = 0; col < 128; ++col) EXPECT_EQ(codes[col], 0);
  EXPECT_EQ(codes[128], 0x7e);
  EXPECT_EQ(codes[129], 0xfe);
  EXPECT_EQ(codes[130], 0x38);
  EXPECT_EQ(codes[131], 0x3a);
  // Replacing reciprocal multiplication with division shifts this scale by one ULP and flips the tie.
  EXPECT_EQ(actual_scales[2], 0.65625f * (1.0f / 448.0f));
  EXPECT_EQ(codes[257], 0x39);
}

TEST(CUDA_EP_Unittest, QMoEDynamicFp8QuantizationScalesAndNearestEven) {
  TestDynamicFp8Quantization<float>();
  TestDynamicFp8Quantization<half>();
  TestDynamicFp8Quantization<__nv_bfloat16>();
}

TEST(CUDA_EP_Unittest, QMoEDynamicFp8QuantizesActivationWithoutBFloat16RoundTrip) {
  std::array<__nv_bfloat16, 256> input{};
  input[0] = __nv_bfloat16(32.0f);
  input[1] = __nv_bfloat16(14.0f);
  input[2] = __nv_bfloat16(0.0078125f);
  input[3] = __nv_bfloat16(0.25f);
  CudaBuffer source(sizeof(input));
  CudaBuffer quantized(128);
  CudaBuffer scales(sizeof(float));
  source.Upload(input.data());
  qmoe::LaunchQMoEDynamicFp8Quantize(source.As<__nv_bfloat16>(), quantized.As<uint8_t>(), scales.As<float>(),
                                     1, 128, 2, nullptr);
  std::array<uint8_t, 128> codes{};
  quantized.Download(codes.data());
  // The FP32 product is just above the E4M3 zero/subnormal midpoint; BF16 would round to the midpoint.
  EXPECT_EQ(codes[0], 0x7e);
  EXPECT_EQ(codes[1], 0x01);
}

TEST(CUDA_EP_Unittest, QMoEDynamicFp8FloatNormalizationPreservesMidpoint) {
  std::array<float, 128> input{};
  input[0] = 0.783361554145813f;
  input[1] = -0.20283471047878265f;
  CudaBuffer source(sizeof(input)), quantized(input.size()), scales(sizeof(float));
  source.Upload(input.data());
  qmoe::LaunchQMoEDynamicFp8Quantize(source.As<float>(), quantized.As<uint8_t>(), scales.As<float>(),
                                     1, 128, 0, nullptr);
  std::array<uint8_t, 128> codes{};
  quantized.Download(codes.data());
  // div.full resolves this to -116 (the midpoint); div.rn yields the next float below -116.
  EXPECT_EQ(codes[1], 0xee);
}

template <typename T>
void TestDynamicFp8RoutingWeightRounding() {
  std::vector<uint8_t> weights(128 * 128, 0x38);
  std::vector<uint8_t> input(128, 0x38);
  const float scale = 1.0f;
  const float coefficient = 0.1f;
  const int route = 0;
  const std::array<int, 2> tiles{0, 1};
  const std::array<int64_t, 2> offsets{0, 1};
  CudaBuffer w(weights.size()), x(input.size()), s(sizeof(float)), r(sizeof(float));
  CudaBuffer map(sizeof(int)), tile_offsets(sizeof(tiles)), expert_offsets(sizeof(offsets));
  CudaBuffer output(128 * sizeof(float));
  w.Upload(weights.data());
  x.Upload(input.data());
  s.Upload(&scale);
  r.Upload(&coefficient);
  map.Upload(&route);
  tile_offsets.Upload(tiles.data());
  expert_offsets.Upload(offsets.data());
  qmoe::QMoEFp8ProjectionParams params;
  params.weights = w.As<uint8_t>();
  params.scales = s.As<float>();
  params.tile_offsets = tile_offsets.As<int>();
  params.expert_offsets = expert_offsets.As<int64_t>();
  params.num_experts = params.num_rows = params.expanded_rows = 1;
  params.n = params.k = params.block_size = 128;
  qmoe::LaunchQMoEDynamicFp8Projection<T, float>(
      params, x.As<uint8_t>(), s.As<float>(), output.As<float>(), r.As<float>(), map.As<int>(), 1, nullptr);
  std::array<float, 128> actual{};
  output.Download(actual.data());
  const float expected = 128.0f * static_cast<float>(T(coefficient));
  if constexpr (!std::is_same_v<T, float>) {
    EXPECT_NE(expected, 128.0f * coefficient);
  }
  for (float value : actual) EXPECT_FLOAT_EQ(value, expected);
}

TEST(CUDA_EP_Unittest, QMoEDynamicFp8RoundsRoutingWeightsBeforeFp32Product) {
  if (!qmoe::QMoEDynamicFp8Supported()) GTEST_SKIP() << "Native FP8 MMA requires an SM89+ kernel.";
  TestDynamicFp8RoutingWeightRounding<half>();
  TestDynamicFp8RoutingWeightRounding<__nv_bfloat16>();
  TestDynamicFp8RoutingWeightRounding<float>();
}

template <typename T>
void TestFp16CompatibleSourceKFragments() {
  std::array<uint8_t, 128> input{};
  std::array<uint8_t, 128 * 128> weights{};
  input[0] = input[16] = 0x7e;  // 448
  input[2] = 0x01;              // 2^-9
  weights[0] = 0x7e;
  weights[16] = 0xfe;  // -448
  weights[2] = 0x38;   // 1
  const float scale = 1.0f;
  const std::array<int, 2> tiles{0, 1};
  const std::array<int64_t, 2> offsets{0, 1};
  CudaBuffer x(sizeof(input)), w(sizeof(weights)), s(sizeof(scale)), output(128 * sizeof(float));
  CudaBuffer tile_offsets(sizeof(tiles)), expert_offsets(sizeof(offsets));
  x.Upload(input.data());
  w.Upload(weights.data());
  s.Upload(&scale);
  tile_offsets.Upload(tiles.data());
  expert_offsets.Upload(offsets.data());
  qmoe::QMoEFp8ProjectionParams params;
  params.weights = w.As<uint8_t>();
  params.scales = s.As<float>();
  params.tile_offsets = tile_offsets.As<int>();
  params.expert_offsets = expert_offsets.As<int64_t>();
  params.num_experts = params.num_rows = params.expanded_rows = 1;
  params.n = params.k = params.block_size = 128;
  qmoe::LaunchQMoEDynamicFp8Projection<T, float>(
      params, x.As<uint8_t>(), s.As<float>(), output.As<float>(), nullptr, nullptr, 1, nullptr, true);
  std::array<float, 128> actual{};
  output.Download(actual.data());
  // Source fragment 0 cancels the large products at K0/K16 before fragment 1
  // adds K2. Contiguous K16 fragments would lose the tiny term before cancellation.
  EXPECT_EQ(actual[0], 0x1p-9f);
  for (size_t col = 1; col < actual.size(); ++col) EXPECT_EQ(actual[col], 0.0f);
}

TEST(CUDA_EP_Unittest, QMoEFp16CompatiblePreservesSourceKFragmentOrder) {
  if (!qmoe::QMoEFp16CompatibleSupported()) GTEST_SKIP() << "SM90 QMoE kernel required.";
  TestFp16CompatibleSourceKFragments<half>();
  TestFp16CompatibleSourceKFragments<__nv_bfloat16>();
}

template <typename T>
void TestFp16CompatibleTiedRouting() {
  std::array<T, 256> logits{};
  CudaBuffer source(sizeof(logits)), scales(8 * sizeof(float)), indices(8 * sizeof(int));
  source.Upload(logits.data());
  qmoe::LaunchQMoEFp16CompatibleRouting(source.As<T>(), scales.As<float>(), indices.As<int>(), 1, nullptr);
  std::array<int, 8> actual_indices{};
  std::array<float, 8> actual_scales{};
  indices.Download(actual_indices.data());
  scales.Download(actual_scales.data());
  // Independent Torch 2.8 CUDA softmax(...).topk(8, sorted=True) oracle.
  const std::array<int, 8> expected{7, 6, 4, 5, 1, 0, 2, 3};
  EXPECT_EQ(actual_indices, expected);
  for (float value : actual_scales) EXPECT_EQ(value, 0.125f);
}

TEST(CUDA_EP_Unittest, QMoEFp16CompatibleThresholdTies) {
  if (!qmoe::QMoEFp16CompatibleSupported()) GTEST_SKIP() << "SM90 QMoE kernel required.";
  TestFp16CompatibleTiedRouting<half>();
  TestFp16CompatibleTiedRouting<__nv_bfloat16>();
}

template <typename T>
void TestFp16CompatibleNormalization(const std::array<uint32_t, 8>& expected_bits) {
  std::array<T, 256> logits{};
  const std::array<float, 8> values{
      2.84765625f, 2.013671875f, 1.68359375f, 2.6953125f, 1.3330078125f, 2.517578125f, 2.861328125f, 1.7353515625f};
  for (size_t i = 0; i < values.size(); ++i) logits[i] = T(values[i]);
  CudaBuffer source(sizeof(logits)), scales(8 * sizeof(float)), indices(8 * sizeof(int));
  source.Upload(logits.data());
  qmoe::LaunchQMoEFp16CompatibleRouting(source.As<T>(), scales.As<float>(), indices.As<int>(), 1, nullptr);
  std::array<int, 8> actual_indices{};
  std::array<uint32_t, 8> actual_bits{};
  indices.Download(actual_indices.data());
  scales.Download(actual_bits.data());
  const std::array<int, 8> expected_indices{6, 0, 3, 5, 1, 7, 2, 4};
  EXPECT_EQ(actual_indices, expected_indices);
  EXPECT_EQ(actual_bits, expected_bits);
}

TEST(CUDA_EP_Unittest, QMoEFp16CompatibleSourceNormalizationBeforeTypedRound) {
  if (!qmoe::QMoEFp16CompatibleSupported()) GTEST_SKIP() << "SM90 QMoE kernel required.";
  // Synthetic Torch 2.8 CUDA oracles. The last FP16 coefficient rounds to
  // 0x29c3; reciprocal-based native normalization instead rounds to 0x29c2.
  TestFp16CompatibleNormalization<half>(
      {1045721238, 1045532189, 1043591563, 1041671282, 1035339434, 1032441271, 1031985732, 1027100673});
  TestFp16CompatibleNormalization<__nv_bfloat16>(
      {1045727040, 1045511103, 1043527552, 1041675396, 1035391114, 1032453862, 1032039721, 1027164856});
}

template <typename T>
void TestFp16CompatibleVt4() {
  std::array<float, 8 * 128> input{};
  const std::array<float, 8> contributions{16777216.0f, 1.0f, 0.0f, 0.0f, -16777216.0f, 1.0f, 0.0f, 0.0f};
  const std::array<int, 8> map{0, 1, 2, 3, 4, 5, 6, 7};
  for (size_t slot = 0; slot < map.size(); ++slot) input[slot * 128] = contributions[slot];
  CudaBuffer source(sizeof(input)), row_map(sizeof(map)), output(128 * sizeof(T));
  source.Upload(input.data());
  row_map.Upload(map.data());
  qmoe::LaunchQMoEDynamicFp8Finalize(source.As<float>(), output.As<T>(), row_map.As<int>(),
                                     1, 128, 8, nullptr, true);
  std::array<T, 128> actual{};
  output.Download(actual.data());
  EXPECT_EQ(static_cast<float>(actual[0]), 2.0f);
  for (int col = 1; col < 128; ++col) EXPECT_EQ(static_cast<float>(actual[col]), 0.0f);
  qmoe::LaunchQMoEDynamicFp8Finalize(source.As<float>(), output.As<T>(), row_map.As<int>(),
                                     1, 128, 8, nullptr);
  output.Download(actual.data());
  EXPECT_EQ(static_cast<float>(actual[0]), 1.0f);
}

TEST(CUDA_EP_Unittest, QMoEFp16CompatibleVt4NotSequentialReduction) {
  if (!qmoe::QMoEFp16CompatibleSupported()) GTEST_SKIP() << "SM90 QMoE kernel required.";
  TestFp16CompatibleVt4<half>();
  TestFp16CompatibleVt4<__nv_bfloat16>();
}

TEST(CUDA_EP_Unittest, QMoEFp16CompatibleProductionExpertMaps) {
  if (!qmoe::QMoEFp16CompatibleSupported()) GTEST_SKIP() << "SM90 QMoE kernel required.";
  namespace ck = onnxruntime::llm::kernels::cutlass_kernels;
  constexpr int experts = 256;
  constexpr int top_k = 8;
  for (int rows : {1, 30, 33, 65, 257, 8191}) {
    for (bool concentrated : {false, true}) {
      SCOPED_TRACE(std::to_string(rows) + ":" + std::to_string(concentrated));
      const int routes = rows * top_k;
      std::vector<int> ids(routes), order(routes), expected_map(routes), expected_inverse(routes);
      std::array<int64_t, experts + 1> expected_offsets{};
      std::array<int, experts + 1> expected_tiles{};
      for (int token = 0; token < rows; ++token) {
        for (int slot = 0; slot < top_k; ++slot) {
          ids[token * top_k + slot] = ((concentrated ? 0 : token * 37) + slot * 23) % experts;
          ++expected_offsets[ids[token * top_k + slot] + 1];
        }
      }
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return ids[a] < ids[b]; });
      for (int index = 0; index < routes; ++index) {
        expected_map[index] = order[index] / top_k + (order[index] % top_k) * rows;
        expected_inverse[expected_map[index]] = index;
      }
      for (int expert = 0; expert < experts; ++expert) {
        expected_tiles[expert + 1] = expected_tiles[expert] + static_cast<int>((expected_offsets[expert + 1] + 15) / 16);
        expected_offsets[expert + 1] += expected_offsets[expert];
      }
      CudaBuffer source(routes * sizeof(int)), map(routes * sizeof(int)), inverse(routes * sizeof(int));
      CudaBuffer sorted(routes * sizeof(int)), offsets(sizeof(expected_offsets)), tiles(sizeof(expected_tiles));
      source.Upload(ids.data());
      if (!ck::fusedBuildExpertMapsSortFirstToken(source.As<int>(), map.As<int>(), inverse.As<int>(), sorted.As<int>(),
                                                  offsets.As<int64_t>(), rows, experts, top_k, 0, experts, nullptr)) {
        const int64_t tokens_per_block = ck::computeNumTokensPerBlock(rows, experts);
        const int64_t blocks = (rows + tokens_per_block - 1) / tokens_per_block;
        CudaBuffer counts(experts * blocks * sizeof(int)), cumulative(experts * blocks * sizeof(int));
        CudaBuffer blocked_rows(experts * rows * sizeof(int));
        ck::threeStepBuildExpertMapsSortFirstToken(
            source.As<int>(), sorted.As<int>(), map.As<int>(), inverse.As<int>(), offsets.As<int64_t>(),
            counts.As<int>(), cumulative.As<int>(), blocked_rows.As<int>(), rows, experts, top_k, 0, nullptr);
      }
      qmoe::LaunchQMoEFp8ExpertTiles(offsets.As<int64_t>(), tiles.As<int>(), experts, nullptr);
      std::vector<int> actual_map(routes), actual_inverse(routes);
      std::array<int64_t, experts + 1> actual_offsets{};
      std::array<int, experts + 1> actual_tiles{};
      map.Download(actual_map.data());
      inverse.Download(actual_inverse.data());
      offsets.Download(actual_offsets.data());
      tiles.Download(actual_tiles.data());
      EXPECT_EQ(actual_map, expected_map);
      EXPECT_EQ(actual_inverse, expected_inverse);
      EXPECT_EQ(actual_offsets, expected_offsets);
      EXPECT_EQ(actual_tiles, expected_tiles);
    }
  }
}

TEST(CUDA_EP_Unittest, QMoEFp8CompactExpertsPreservesRoutesAndClearsUnusedSlots) {
  constexpr int num_experts = 512;
  constexpr int capacity = 6;
  CudaBuffer indices(capacity * sizeof(int));
  CudaBuffer compact_indices(capacity * sizeof(int));
  CudaBuffer expert_map(num_experts * sizeof(int));
  CudaBuffer slot_map(capacity * sizeof(int));

  for (const auto& routes : {std::array<int, capacity>{511, 7, 511, 7, 2, 7},
                             std::array<int, capacity>{0, 0, 0, 0, 0, 0}}) {
    indices.Upload(routes.data());
    qmoe::LaunchQMoECompactExperts(indices.As<int>(), compact_indices.As<int>(),
                                   expert_map.As<int>(), slot_map.As<int>(),
                                   num_experts, capacity, capacity, nullptr);
    std::array<int, capacity> original{};
    std::array<int, capacity> compact{};
    std::array<int, capacity> slots{};
    std::array<int, num_experts> reverse{};
    indices.Download(original.data());
    compact_indices.Download(compact.data());
    slot_map.Download(slots.data());
    expert_map.Download(reverse.data());
    EXPECT_EQ(original, routes);
    int used = 0;
    for (int expert = 0; expert < num_experts; ++expert) {
      if (reverse[expert] >= 0) {
        EXPECT_EQ(reverse[expert], used);
        EXPECT_EQ(slots[used++], expert);
      }
    }
    EXPECT_EQ(used, routes[0] == 511 ? 3 : 1);
    for (int slot = used; slot < capacity; ++slot) {
      EXPECT_EQ(slots[slot], -1);
    }
    for (int route = 0; route < capacity; ++route) {
      ASSERT_GE(compact[route], 0);
      ASSERT_LT(compact[route], used);
      EXPECT_EQ(slots[compact[route]], routes[route]);
    }
  }
}

template <typename T>
void TestSelectedDequantization(int layout) {
  constexpr int source_experts = 5;
  constexpr int capacity = 3;
  constexpr int n = 6;
  constexpr int k = 5;
  constexpr int block_size = 4;
  const bool split = layout == 1;
  const int output_n = split ? 2 * n : n;
  const std::array<int, capacity> experts{4, 1, -1};
  std::vector<uint8_t> weights(source_experts * n * k, 0x7f);
  std::vector<float> scales(source_experts * 4);
  std::vector<T> bias(source_experts * n);
  for (int expert = 0; expert < source_experts; ++expert) {
    for (int i = 0; i < n * k; ++i) {
      if (expert == 4 || expert == 1) {
        weights[expert * n * k + i] = 0x38;  // E4M3 1.0; unselected experts contain NaNs.
      }
    }
    for (int i = 0; i < 4; ++i) {
      scales[expert * 4 + i] = expert + 1.0f + i * 0.25f;
    }
    for (int row = 0; row < n; ++row) {
      bias[expert * n + row] = T(static_cast<float>(expert * n + row));
    }
  }
  std::vector<T> output(capacity * output_n * k, T(-123.0f));
  std::vector<T> output_bias(capacity * output_n, T(-123.0f));
  CudaBuffer device_weights(weights.size());
  CudaBuffer device_scales(scales.size() * sizeof(float));
  CudaBuffer device_bias(bias.size() * sizeof(T));
  CudaBuffer device_output(output.size() * sizeof(T));
  CudaBuffer device_output_bias(output_bias.size() * sizeof(T));
  CudaBuffer device_experts(experts.size() * sizeof(int));
  device_weights.Upload(weights.data());
  device_scales.Upload(scales.data());
  device_bias.Upload(bias.data());
  device_output.Upload(output.data());
  device_output_bias.Upload(output_bias.data());
  device_experts.Upload(experts.data());

  for (int projection = 0; projection < (split ? 2 : 1); ++projection) {
    qmoe::LaunchQMoEDequantizeFp8Weights(
        device_weights.As<uint8_t>(), device_scales.As<float>(), device_output.As<T>() + projection * k,
        capacity, n, k, nullptr, block_size, split, layout == 2, device_experts.As<int>(),
        device_bias.As<T>(), device_output_bias.As<T>() + projection);
  }
  device_output.Download(output.data());
  device_output_bias.Download(output_bias.data());
  for (int slot = 0; slot < capacity; ++slot) {
    for (int output_row = 0; output_row < output_n; ++output_row) {
      const int row = split ? output_row / 2
                            : (layout == 2 ? output_row / 2 + (output_row % 2) * (n / 2) : output_row);
      const int expert = experts[slot];
      const float expected_bias = expert < 0 ? -123.0f : static_cast<float>(bias[expert * n + row]);
      EXPECT_EQ(static_cast<float>(output_bias[slot * output_n + output_row]), expected_bias);
      for (int col = 0; col < k; ++col) {
        const float expected = expert < 0 ? -123.0f
                                          : scales[expert * 4 + (row / block_size) * 2 + col / block_size];
        EXPECT_EQ(static_cast<float>(output[(slot * output_n + output_row) * k + col]), expected);
      }
    }
  }
}

TEST(CUDA_EP_Unittest, QMoEFp8DequantizesOnlySelectedExpertsAndGathersBias) {
  for (int layout : {0, 1, 2}) {
    SCOPED_TRACE(layout);
    TestSelectedDequantization<half>(layout);
    TestSelectedDequantization<__nv_bfloat16>(layout);
  }
}

[[maybe_unused]] float FiniteFp8Reference(int code) {
  const int exponent = (code >> 3) & 15;
  const int mantissa = code & 7;
  const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
                                        : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
  return code & 128 ? -magnitude : magnitude;
}

#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
bool SupportsFp8Gemv() {
  int device = 0;
  cudaDeviceProp properties{};
  CUDA_CALL_THROW(cudaGetDevice(&device));
  CUDA_CALL_THROW(cudaGetDeviceProperties(&properties, device));
  return properties.major >= 8;
}
#endif

template <typename T, typename S>
void TestFp8GemvCodeDecoding(int scale_type, int block_size, int k, int fusion, bool graph) {
  constexpr int n = 256;
  const int source_n = fusion == 1 ? n / 2 : n;
  const int scales_per_expert =
      ((source_n + block_size - 1) / block_size) * ((k + block_size - 1) / block_size);
  std::vector<uint8_t> weights(2 * source_n * k);
  std::vector<uint8_t> up_weights(weights.size());
  for (int expert = 0; expert < 2; ++expert) {
    for (int row = 0; row < source_n; ++row) {
      const int code = fusion == 1 ? row * 2 : row;
      std::fill_n(weights.begin() + (expert * source_n + row) * k, k, static_cast<uint8_t>(code));
      std::fill_n(up_weights.begin() + (expert * source_n + row) * k, k, static_cast<uint8_t>(code + 1));
    }
  }
  const int k_blocks = (k + block_size - 1) / block_size;
  std::vector<S> scales(2 * scales_per_expert);
  for (int expert = 0; expert < 2; ++expert) {
    for (int block = 0; block < scales_per_expert; ++block) {
      scales[expert * scales_per_expert + block] =
          S((expert + 1) * 0.25f + ((block / k_blocks) % 4) * 0.125f + (block % k_blocks) * 0.25f);
    }
  }
  std::vector<T> input(2 * k);
  for (int col = 0; col < k; ++col) {
    input[col] = T(col == k - 1 ? 2.0f : (1 + col % 4) * 0.03125f);
    input[k + col] = T(col == k - 1 ? -4.0f : -(1 + col % 3) * 0.0625f);
  }
  const std::array<int, 2> experts{1, 0};
  const std::array<int, 2> source_rows{1, 0};
  CudaBuffer device_weights(weights.size());
  CudaBuffer device_up_weights(up_weights.size());
  CudaBuffer device_scales(scales.size() * sizeof(S));
  CudaBuffer device_input(input.size() * sizeof(T));
  CudaBuffer device_experts(sizeof(experts));
  CudaBuffer device_source_rows(sizeof(source_rows));
  CudaBuffer device_output(2 * n * sizeof(T));
  device_weights.Upload(weights.data());
  device_up_weights.Upload(up_weights.data());
  device_scales.Upload(scales.data());
  device_input.Upload(input.data());
  device_experts.Upload(experts.data());
  device_source_rows.Upload(source_rows.data());
  qmoe::QMoEFp8ProjectionParams params;
  params.weights = device_weights.As<uint8_t>();
  params.scales = device_scales.data;
  params.scale_type = scale_type;
  if (fusion == 1) {
    params.up_weights = device_up_weights.As<uint8_t>();
    params.up_scales = device_scales.data;
    params.up_scale_type = scale_type;
  }
  params.fusion = fusion;
  params.experts = device_experts.As<int>();
  params.row_to_unpermuted = device_source_rows.As<int>();
  params.num_experts = 2;
  params.num_rows = 2;
  params.expanded_rows = 2;
  params.n = n;
  params.k = k;
  params.block_size = block_size;
  cudaStream_t stream;
  CUDA_CALL_THROW(cudaStreamCreate(&stream));
  if (graph) {
    cudaGraph_t captured;
    cudaGraphExec_t executable;
    CUDA_CALL_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    qmoe::LaunchQMoEFp8Projection(params, device_input.As<T>(), device_output.As<T>(), stream);
    CUDA_CALL_THROW(cudaStreamEndCapture(stream, &captured));
    size_t node_count = 0;
    CUDA_CALL_THROW(cudaGraphGetNodes(captured, nullptr, &node_count));
    std::vector<cudaGraphNode_t> nodes(node_count);
    CUDA_CALL_THROW(cudaGraphGetNodes(captured, nodes.data(), &node_count));
    EXPECT_EQ(node_count, 1u);
    if (node_count == 1) {
      cudaKernelNodeParams node{};
      CUDA_CALL_THROW(cudaGraphKernelNodeGetParams(nodes[0], &node));
      EXPECT_EQ(qmoe::QMoEFp8GemvKernelUsesNativeDecode(node.func),
                ParseEnvironmentVariableWithDefault<bool>("ORT_QMOE_FP8_GEMV_NATIVE_DECODE", false));
    }
    CUDA_CALL_THROW(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    CUDA_CALL_THROW(cudaGraphLaunch(executable, stream));
    CUDA_CALL_THROW(cudaStreamSynchronize(stream));
    CUDA_CALL_THROW(cudaGraphExecDestroy(executable));
    CUDA_CALL_THROW(cudaGraphDestroy(captured));
  } else {
    qmoe::LaunchQMoEFp8Projection(params, device_input.As<T>(), device_output.As<T>(), stream);
    CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  }
  CUDA_CALL_THROW(cudaStreamDestroy(stream));
  std::array<T, 2 * n> actual{};
  device_output.Download(actual.data());
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < n; ++col) {
      const int code = fusion == 2 ? col / 2 + (col % 2) * (n / 2) : col;
      SCOPED_TRACE(row);
      SCOPED_TRACE(code);
      const float value = static_cast<float>(actual[row * n + col]);
      if ((code & 127) == 127) {
        EXPECT_TRUE(std::isnan(value));
        continue;
      }
      const int source_col = fusion == 1 ? col / 2 : code;
      double sum = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        const float scale = static_cast<float>(scales[experts[row] * scales_per_expert +
                                                      (source_col / block_size) * k_blocks + inner / block_size]);
        const T scaled_weight(FiniteFp8Reference(code) * scale);
        sum += static_cast<float>(scaled_weight) * static_cast<float>(input[source_rows[row] * k + inner]);
      }
      const float expected = static_cast<float>(T(static_cast<float>(sum)));
      const float tolerance = std::is_same_v<T, half> ? 0.001f : 0.008f;
      EXPECT_NEAR(value, expected, std::max(1e-7f, std::abs(expected) * tolerance));
    }
  }
}

void TestFp8GemvCodeConfigurations() {
  for (int block_size : {16, 128}) {
    for (int fusion : {0, 1, 2}) {
      for (bool graph : {false, true}) {
        SCOPED_TRACE(block_size);
        SCOPED_TRACE(fusion);
        SCOPED_TRACE(graph);
        const int k = block_size * 2 + 1;
        TestFp8GemvCodeDecoding<half, float>(0, block_size, k, fusion, graph);
        TestFp8GemvCodeDecoding<half, half>(1, block_size, k, fusion, graph);
        TestFp8GemvCodeDecoding<half, __nv_bfloat16>(2, block_size, k, fusion, graph);
        TestFp8GemvCodeDecoding<__nv_bfloat16, float>(0, block_size, k, fusion, graph);
        TestFp8GemvCodeDecoding<__nv_bfloat16, half>(1, block_size, k, fusion, graph);
        TestFp8GemvCodeDecoding<__nv_bfloat16, __nv_bfloat16>(2, block_size, k, fusion, graph);
      }
    }
  }
}

TEST(CUDA_EP_Unittest, QMoEFp8GemvAllCodesWithMappedRowsPartialKBlocksAndGraphReplay) {
  TestFp8GemvCodeConfigurations();
}

TEST(CUDA_EP_Unittest, QMoEFp8GemvNativeDecode) {
  if (!ParseEnvironmentVariableWithDefault<bool>("ORT_QMOE_FP8_GEMV_NATIVE_DECODE", false)) {
    GTEST_SKIP() << "Native decoding coverage runs in its flag-enabled CTest process.";
  }
  if (!qmoe::QMoEFp8GemvSupportsNativeDecode()) {
    GTEST_SKIP() << "Native E4M3 decoding requires FP8-enabled kernels compiled for SM90+.";
  }
  TestFp8GemvCodeConfigurations();
}

#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
template <typename T>
void TestNvfp4GemvScaleCodes(int k) {
  constexpr int n = 256;
  std::vector<uint8_t> weights(n * k / 2, 0);
  std::vector<uint8_t> scales(n * k / 16);
  const int blocks = k / 16;
  for (int row = 0; row < n; ++row) {
    for (int block = 0; block < blocks; ++block) {
      weights[row * k / 2 + block * 8] = 0x22;
      scales[row * blocks + block] =
          block == 0 ? static_cast<uint8_t>(row) : static_cast<uint8_t>(0x30 + (row + block) % 16);
    }
  }
  std::vector<T> input(k, T(0.0f));
  for (int block = 0; block < blocks; ++block) {
    input[block * 16] = T(1.0f + (block % 3) * 0.5f);
    input[block * 16 + 1] = T(0.5f + (block % 2) * 0.25f);
  }
  const float global = 0.37f;
  const int expert = 0;
  CudaBuffer device_weights(weights.size());
  CudaBuffer device_scales(scales.size());
  CudaBuffer device_input(input.size() * sizeof(T));
  CudaBuffer device_global(sizeof(global));
  CudaBuffer device_expert(sizeof(expert));
  CudaBuffer device_output(n * sizeof(T));
  device_weights.Upload(weights.data());
  device_scales.Upload(scales.data());
  device_input.Upload(input.data());
  device_global.Upload(&global);
  device_expert.Upload(&expert);
  using namespace onnxruntime::llm::kernels::moe_gemv;
  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  CUDA_CALL_THROW(cudaStreamCreate(&stream));
  CUDA_CALL_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  launch_moe_gemv_fp4_symmetric<T>(
      device_input.As<T>(), device_weights.As<uint8_t>(), nullptr, device_scales.As<uint8_t>(),
      device_global.As<float>(), nullptr, device_output.As<T>(), nullptr, device_expert.As<int>(),
      1, 1, n, k, 16, 90, MoeGemvConfig::kDefault, false, true, stream, true);
  CUDA_CALL_THROW(cudaStreamEndCapture(stream, &graph));
  size_t node_count = 0;
  CUDA_CALL_THROW(cudaGraphGetNodes(graph, nullptr, &node_count));
  std::vector<cudaGraphNode_t> nodes(node_count);
  CUDA_CALL_THROW(cudaGraphGetNodes(graph, nodes.data(), &node_count));
  EXPECT_EQ(node_count, 1u);
  if (node_count == 1) {
    cudaKernelNodeParams node{};
    CUDA_CALL_THROW(cudaGraphKernelNodeGetParams(nodes[0], &node));
    EXPECT_EQ(Fp4MoeGemvKernelUsesNativeScale(node.func),
              ParseEnvironmentVariableWithDefault<bool>("ORT_NVFP4_GEMV_NATIVE_SCALE", false));
  }
  CUDA_CALL_THROW(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CALL_THROW(cudaGraphLaunch(executable, stream));
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  CUDA_CALL_THROW(cudaGraphExecDestroy(executable));
  CUDA_CALL_THROW(cudaGraphDestroy(graph));
  CUDA_CALL_THROW(cudaStreamDestroy(stream));
  std::array<T, n> actual{};
  device_output.Download(actual.data());
  for (int code = 0; code < 256; ++code) {
    SCOPED_TRACE(code);
    const float value = static_cast<float>(actual[code]);
    if ((code & 127) == 127) {
      EXPECT_TRUE(std::isnan(value));
    } else {
      double sum = 0.0;
      for (int block = 0; block < blocks; ++block) {
        const float scale = static_cast<float>(T(FiniteFp8Reference(scales[code * blocks + block]) * global));
        sum += scale * static_cast<float>(input[block * 16]);
        sum += scale * static_cast<float>(input[block * 16 + 1]);
      }
      const float expected = static_cast<float>(T(static_cast<float>(sum)));
      const float tolerance = std::is_same_v<T, half> ? 0.001f : 0.008f;
      EXPECT_NEAR(value, expected, std::max(1e-7f, std::abs(expected) * tolerance));
    }
  }
}

void TestNvfp4GemvScaleCodeConfigurations() {
  for (int k : {16, 80, 1040}) {
    SCOPED_TRACE(k);
    TestNvfp4GemvScaleCodes<half>(k);
    TestNvfp4GemvScaleCodes<__nv_bfloat16>(k);
  }
}

TEST(CUDA_EP_Unittest, QMoENvfp4GemvAllScaleCodesPreserveScaledValuesAndNaNs) {
  if (!SupportsFp8Gemv()) {
    GTEST_SKIP() << "NVFP4 GEMV requires SM80+.";
  }
  TestNvfp4GemvScaleCodeConfigurations();
}

TEST(CUDA_EP_Unittest, QMoENvfp4GemvNativeScale) {
  if (!ParseEnvironmentVariableWithDefault<bool>("ORT_NVFP4_GEMV_NATIVE_SCALE", false)) {
    GTEST_SKIP() << "Native-scale coverage runs in its flag-enabled CTest process.";
  }
  if (!SupportsFp8Gemv() || !onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvSupportsNativeScale()) {
    GTEST_SKIP() << "Native E4M3 scale conversion requires FP8-enabled kernels compiled for SM90+.";
  }
  TestNvfp4GemvScaleCodeConfigurations();
}
#endif

}  // namespace
}  // namespace test
}  // namespace onnxruntime
