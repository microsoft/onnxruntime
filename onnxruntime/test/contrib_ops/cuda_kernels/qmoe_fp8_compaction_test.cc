// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#ifdef ORT_TEST_NVFP4_GEMV
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.h"
#endif
#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"

#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.h"
#endif

namespace onnxruntime {
namespace test {
namespace {

namespace qmoe = contrib::cuda;
#if defined(ORT_TEST_NVFP4_GEMV) || (defined(ENABLE_FP4) && defined(USE_FP4_QMOE))
bool SupportsSm80OrLater() {
  int device = 0;
  cudaDeviceProp properties{};
  CUDA_CALL_THROW(cudaGetDevice(&device));
  CUDA_CALL_THROW(cudaGetDeviceProperties(&properties, device));
  return properties.major >= 8;
}
#endif

#ifdef ORT_TEST_NVFP4_GEMV
namespace gemv = llm::kernels::moe_gemv;
#endif

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

#ifdef ORT_TEST_NVFP4_GEMV
template <typename T>
void TestRoutingMaps(int rows, int experts, int k, bool normalize, bool tied) {
  std::vector<T> logits(rows * experts);
  for (int i = 0; i < rows * experts; ++i) {
    logits[i] = T(tied ? 0.0f : float((i * 73 + i / experts * 11) % 101 - 50) / 16.0f);
  }
  const int expanded = rows * k;
  CudaBuffer input(logits.size() * sizeof(T));
  CudaBuffer scales(expanded * sizeof(float)), indices(expanded * sizeof(int));
  CudaBuffer r2u(expanded * sizeof(int)), u2r(expanded * sizeof(int)), sorted(expanded * sizeof(int));
  CudaBuffer offsets((experts + 1) * sizeof(int64_t));
  input.Upload(logits.data());
  qmoe::LaunchSoftmaxTopK(input.As<T>(), scales.As<float>(), indices.As<int>(),
                          rows, experts, k, normalize, nullptr);
  ASSERT_TRUE(qmoe::TryLaunchSmallDecodeExpertMaps(
      indices.As<int>(), r2u.As<int>(), u2r.As<int>(), sorted.As<int>(),
      offsets.As<int64_t>(), rows, experts, k, nullptr));
  std::vector<int> expected_indices(expanded), actual_r2u(expanded),
      actual_u2r(expanded), actual_sorted(expanded);
  std::vector<int64_t> actual_offsets(experts + 1);
  indices.Download(expected_indices.data());
  r2u.Download(actual_r2u.data());
  u2r.Download(actual_u2r.data());
  sorted.Download(actual_sorted.data());
  offsets.Download(actual_offsets.data());
  std::vector<int> order(expanded);
  for (int i = 0; i < expanded; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return expected_indices[a] < expected_indices[b];
  });
  for (int rank = 0; rank < expanded; ++rank) {
    const int original = order[rank];
    const int source = (original % k) * rows + original / k;
    EXPECT_EQ(actual_r2u[rank], source);
    EXPECT_EQ(actual_u2r[source], rank);
    EXPECT_EQ(actual_sorted[rank], expected_indices[original]);
  }
  for (int expert = 0; expert <= experts; ++expert) {
    EXPECT_EQ(actual_offsets[expert],
              std::count_if(expected_indices.begin(), expected_indices.end(),
                            [&](int selected) { return selected < expert; }));
  }
}

TEST(CUDA_EP_Unittest, QMoENvfp4SmallDecodeMapsPreserveStableOrder) {
  for (int rows : {1, 2}) {
    for (int experts : {16, 257, 512}) {
      for (int k : {1, 10, 16}) {
        for (bool normalize : {false, true}) {
          for (bool tied : {false, true}) {
            TestRoutingMaps<half>(rows, experts, k, normalize, tied);
            TestRoutingMaps<__nv_bfloat16>(rows, experts, k, normalize, tied);
          }
        }
      }
    }
  }
  EXPECT_FALSE(qmoe::TryLaunchSmallDecodeExpertMaps(
      nullptr, nullptr, nullptr, nullptr, nullptr, 3, 512, 10, nullptr));
  EXPECT_FALSE(qmoe::TryLaunchSmallDecodeExpertMaps(
      nullptr, nullptr, nullptr, nullptr, nullptr, 2, 513, 10, nullptr));
  EXPECT_FALSE(qmoe::TryLaunchSmallDecodeExpertMaps(
      nullptr, nullptr, nullptr, nullptr, nullptr, 0, 512, 10, nullptr));
  EXPECT_FALSE(qmoe::TryLaunchSmallDecodeExpertMaps(
      nullptr, nullptr, nullptr, nullptr, nullptr, 2, 512, 17, nullptr));
  EXPECT_FALSE(qmoe::TryLaunchSmallDecodeExpertMaps(
      nullptr, nullptr, nullptr, nullptr, nullptr, 2, 8, 10, nullptr));
}

template <typename T>
void TestNvfp4KPackedGateUp(int k, bool use_bias, bool varied_scales = false, int n = 48) {
  if (!SupportsSm80OrLater()) {
    GTEST_SKIP() << "NVFP4 K-packed gate/up requires SM80 or newer";
  }
  constexpr int experts = 3;
  constexpr int rows = 4;
  constexpr std::array<int, rows> routes{2, 0, 2, 1};
  constexpr std::array<int, rows> source_rows{1, 0, 0, 1};
  constexpr std::array<float, experts> globals{0.25f, 0.125f, 0.5f};
  constexpr std::array<float, 8> magnitudes{0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  std::vector<T> activations(2 * k), bias(experts * n);
  std::vector<uint8_t> weights(experts * n * k / 2), scales(experts * n * k / 16);
  for (size_t i = 0; i < activations.size(); ++i) {
    activations[i] = T(float(int(i % 11) - 5) / 64.0f);
  }
  for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<uint8_t>(i * 37 + 19);
  for (size_t i = 0; i < scales.size(); ++i) {
    scales[i] = varied_scales ? static_cast<uint8_t>(i % 127) : (i % 2 ? 0x38 : 0x30);
  }
  for (size_t i = 0; i < bias.size(); ++i) bias[i] = T(float(int(i % 7) - 3) / 16.0f);
  CudaBuffer a(activations.size() * sizeof(T)), w(weights.size()), s(scales.size());
  CudaBuffer g(sizeof(globals)), b(bias.size() * sizeof(T)), r(sizeof(routes)), sources(sizeof(source_rows));
  CudaBuffer output(rows * n / 2 * sizeof(T));
  a.Upload(activations.data());
  w.Upload(weights.data());
  s.Upload(scales.data());
  g.Upload(globals.data());
  b.Upload(bias.data());
  r.Upload(routes.data());
  sources.Upload(source_rows.data());
  llm::kernels::cutlass_kernels::ActivationParams activation;
  activation.alpha = 1.0f;
  activation.beta = 0.25f;
  activation.limit = 7.0f;
  gemv::launch_moe_gemv_fp4_symmetric_interleaved_swiglu<T>(
      a.As<T>(), w.As<uint8_t>(), nullptr, s.As<uint8_t>(), g.As<float>(),
      use_bias ? b.As<T>() : nullptr, output.As<T>(), nullptr, r.As<int>(),
      experts, rows, n / 2, k, 16, 90, activation, gemv::MoeGemvConfig::kDefault,
      false, true, sources.As<int>(), 2, nullptr, true);
  CUDA_CALL_THROW(cudaGetLastError());
  std::vector<T> actual(rows * n / 2);
  output.Download(actual.data());
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < n; col += 2) {
      float sums[2]{};
      for (int pair = 0; pair < 2; ++pair) {
        const int column = col + pair;
        for (int d = 0; d < k; ++d) {
          const uint8_t packed = weights[(routes[row] * n + column) * (k / 2) + d / 2];
          const int code = (d % 2 ? packed >> 4 : packed) & 15;
          const uint8_t scale_code = scales[(routes[row] * n + column) * (k / 16) + d / 16];
          const int exponent = scale_code >> 3;
          const int mantissa = scale_code & 7;
          const float block_scale = exponent == 0 ? std::ldexp(float(mantissa), -9)
                                                  : std::ldexp(1.0f + mantissa * 0.125f, exponent - 7);
          const float rounded_scale = static_cast<float>(T(block_scale * globals[routes[row]]));
          const float weight = static_cast<float>(T(magnitudes[code & 7] * (code & 8 ? -1 : 1) * rounded_scale));
          sums[pair] += weight * static_cast<float>(activations[source_rows[row] * k + d]);
        }
        if (use_bias) sums[pair] += static_cast<float>(bias[routes[row] * n + column]);
      }
      const float gate = std::min(sums[0], activation.limit);
      const float up = std::clamp(sums[1], -activation.limit, activation.limit) + activation.beta;
      const float expected = static_cast<float>(T(gate / (1.0f + std::exp(-gate)) * up));
      EXPECT_NEAR(static_cast<float>(actual[row * n / 2 + col / 2]), expected,
                  0.001f + std::abs(expected) * 0.01f)
          << "row=" << row << ", column=" << col << ", K=" << k << ", bias=" << use_bias;
    }
  }
}

TEST(CUDA_EP_Unittest, QMoENvfp4KPackedGateUpTiling) {
  for (int k : {144, 2560}) {
    for (bool use_bias : {false, true}) {
      TestNvfp4KPackedGateUp<half>(k, use_bias);
      TestNvfp4KPackedGateUp<__nv_bfloat16>(k, use_bias);
    }
  }
}

TEST(CUDA_EP_Unittest, QMoENvfp4KPackedGateUpScaleRange) {
  TestNvfp4KPackedGateUp<half>(2560, true, true);
  TestNvfp4KPackedGateUp<__nv_bfloat16>(2560, true, true);
}

TEST(CUDA_EP_Unittest, QMoENvfp4KPackedGateUpPartialTiles) {
  for (int n : {2, 10}) {
    for (int k : {16, 272}) {
      for (bool use_bias : {false, true}) {
        TestNvfp4KPackedGateUp<half>(k, use_bias, false, n);
        TestNvfp4KPackedGateUp<__nv_bfloat16>(k, use_bias, false, n);
      }
    }
  }
}

template <typename T>
void TestNvfp4FusedFinalize(int num_rows, int topk, int k, bool use_bias) {
  if (!SupportsSm80OrLater()) {
    GTEST_SKIP() << "NVFP4 fused finalization requires SM80 or newer";
  }
  constexpr int n = 50, experts = 3;
  const int expanded = num_rows * topk;
  std::vector<T> activations(expanded * k), biases(experts * n);
  std::vector<uint8_t> weights(experts * n * k / 2), scales(experts * n * k / 16, 0x30);
  std::vector<float> globals{0.3f, 0.17f, 0.51f}, routing(expanded);
  std::vector<int> mapping(expanded), expert_ids(expanded);
  for (size_t i = 0; i < activations.size(); ++i) activations[i] = T(float(int(i % 11) - 5) / 64.0f);
  for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<uint8_t>(i * 37 + 19);
  for (size_t i = 0; i < biases.size(); ++i) biases[i] = T(float(int(i % 7) - 3) / 16.0f);
  for (int i = 0; i < expanded; ++i) {
    mapping[i] = (i * 7) % expanded;
    expert_ids[i] = i % experts;
    routing[i] = float(i % topk + 1) / float(topk * (topk + 1) / 2);
  }
  if (topk > 1) mapping[expanded - 1] = -1;
  if (expanded > 1) expert_ids[expanded / 2] = experts;
  CudaBuffer a(activations.size() * sizeof(T)), w(weights.size()), s(scales.size());
  CudaBuffer g(globals.size() * sizeof(float)), b(biases.size() * sizeof(T));
  CudaBuffer m(mapping.size() * sizeof(int)), e(expert_ids.size() * sizeof(int)), r(routing.size() * sizeof(float));
  CudaBuffer reference(expanded * n * sizeof(T)), output(num_rows * n * sizeof(T));
  a.Upload(activations.data());
  w.Upload(weights.data());
  s.Upload(scales.data());
  g.Upload(globals.data());
  b.Upload(biases.data());
  m.Upload(mapping.data());
  e.Upload(expert_ids.data());
  r.Upload(routing.data());
  gemv::launch_moe_gemv_fp4_symmetric<T>(
      a.As<T>(), w.As<uint8_t>(), nullptr, s.As<uint8_t>(), g.As<float>(),
      use_bias ? b.As<T>() : nullptr, reference.As<T>(), nullptr, e.As<int>(),
      experts, expanded, n, k, 16, 90, gemv::MoeGemvConfig::kDefault, false, true, nullptr, true);
  ASSERT_TRUE(gemv::try_launch_moe_gemv_nvfp4_fused_finalize<T>(
      a.As<T>(), w.As<uint8_t>(), s.As<uint8_t>(), g.As<float>(),
      use_bias ? b.As<T>() : nullptr, output.As<T>(), m.As<int>(), e.As<int>(), r.As<float>(),
      experts, num_rows, topk, n, k, nullptr));
  CUDA_CALL_THROW(cudaGetLastError());
  std::vector<T> reference_values(expanded * n), actual(num_rows * n);
  reference.Download(reference_values.data());
  output.Download(actual.data());
  for (int row = 0; row < num_rows; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int rank = 0; rank < topk; ++rank) {
        const int permuted = mapping[rank * num_rows + row];
        if (permuted >= 0 && expert_ids[permuted] < experts) {
          sum += routing[row * topk + rank] * static_cast<float>(reference_values[permuted * n + col]);
        }
      }
      const float expected = static_cast<float>(T(sum));
      EXPECT_NEAR(static_cast<float>(actual[row * n + col]), expected, 1e-5f + std::abs(expected) * 0.01f)
          << "row=" << row << ", col=" << col << ", topk=" << topk << ", bias=" << use_bias;
    }
  }
}

TEST(CUDA_EP_Unittest, QMoENvfp4FusedFinalizePreservesRoutingAndRounding) {
  for (int rows : {1, 2}) {
    for (int topk : {1, 4, 10, 16}) {
      for (int k : {144, 640, 1152}) {
        for (bool use_bias : {false, true}) {
          TestNvfp4FusedFinalize<half>(rows, topk, k, use_bias);
          TestNvfp4FusedFinalize<__nv_bfloat16>(rows, topk, k, use_bias);
        }
      }
    }
  }

  if (!SupportsSm80OrLater()) {
    GTEST_SKIP() << "NVFP4 fused finalization requires SM80 or newer";
  }
  // Cancellation-sensitive parity check for K <= 1024 (4 K-lanes):
  // lane 0 accumulates offsets 0, 64, and 128 in order: (100139008 + 1) + (-100139008) == 0.
  constexpr int k = 144, n = 2, experts = 1, num_rows = 1, topk = 1;
  std::vector<__nv_bfloat16> activations(k, __nv_bfloat16(0.0f));
  activations[0] = __nv_bfloat16(100139008.0f);
  activations[64] = __nv_bfloat16(1.0f);
  activations[128] = __nv_bfloat16(-100139008.0f);
  // E2M1 code 2 is +1.0, so 0x22 packs two +1.0 weights; E4M3 code 0x38 is 1.0.
  const std::vector<uint8_t> weights(n * k / 2, 0x22), scales(n * k / 16, 0x38);
  const std::vector<float> globals{1.0f}, routing{1.0f};
  const std::vector<int> mapping{0}, expert_ids{0};
  CudaBuffer a(activations.size() * sizeof(__nv_bfloat16)), w(weights.size()), s(scales.size());
  CudaBuffer g(sizeof(float)), m(sizeof(int)), e(sizeof(int)), r(sizeof(float));
  CudaBuffer reference(n * sizeof(__nv_bfloat16)), output(n * sizeof(__nv_bfloat16));
  a.Upload(activations.data());
  w.Upload(weights.data());
  s.Upload(scales.data());
  g.Upload(globals.data());
  m.Upload(mapping.data());
  e.Upload(expert_ids.data());
  r.Upload(routing.data());
  gemv::launch_moe_gemv_fp4_symmetric<__nv_bfloat16>(
      a.As<__nv_bfloat16>(), w.As<uint8_t>(), nullptr, s.As<uint8_t>(), g.As<float>(),
      nullptr, reference.As<__nv_bfloat16>(), nullptr, e.As<int>(),
      experts, 1, n, k, 16, 90, gemv::MoeGemvConfig::kDefault, false, true, nullptr, true);
  ASSERT_TRUE(gemv::try_launch_moe_gemv_nvfp4_fused_finalize<__nv_bfloat16>(
      a.As<__nv_bfloat16>(), w.As<uint8_t>(), s.As<uint8_t>(), g.As<float>(),
      nullptr, output.As<__nv_bfloat16>(), m.As<int>(), e.As<int>(), r.As<float>(),
      experts, num_rows, topk, n, k, nullptr));
  CUDA_CALL_THROW(cudaGetLastError());
  std::vector<__nv_bfloat16> ref_host(n), out_host(n);
  reference.Download(ref_host.data());
  output.Download(out_host.data());
  for (int col = 0; col < n; ++col) {
    EXPECT_EQ(static_cast<float>(ref_host[col]), 0.0f);
    EXPECT_EQ(static_cast<float>(out_host[col]), static_cast<float>(ref_host[col]));
  }
}
#endif

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

template <typename Call>
void ExpectThrowsContaining(Call&& call, const char* fragment) {
  try {
    call();
    ADD_FAILURE() << "Expected an exception containing \"" << fragment << "\".";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
  }
}

// Eight experts and six routes that use four distinct experts {1, 3, 5, 6}.
struct CompactExpertsCase {
  static constexpr int kExperts = 8;
  static constexpr int kRoutes = 6;
  static constexpr std::array<int, kRoutes> kRouteExperts{5, 1, 5, 6, 1, 3};

  CompactExpertsCase()
      : indices(kRoutes * sizeof(int)),
        compact_indices(kRoutes * sizeof(int)),
        expert_map(kExperts * sizeof(int)),
        slot_map(kExperts * sizeof(int)) {
    indices.Upload(kRouteExperts.data());
  }

  void Launch(int capacity, int pass_offset, bool partial_pass) {
    qmoe::LaunchQMoECompactExperts(indices.As<int>(), compact_indices.As<int>(), expert_map.As<int>(),
                                   slot_map.As<int>(), kExperts, capacity, kRoutes, nullptr, pass_offset,
                                   partial_pass);
  }

  std::array<int, kRoutes> Compact() {
    std::array<int, kRoutes> values{};
    compact_indices.Download(values.data());
    return values;
  }

  std::vector<int> Slots(int capacity) {
    std::array<int, kExperts> values{};
    slot_map.Download(values.data());
    return std::vector<int>(values.begin(), values.begin() + capacity);
  }

  CudaBuffer indices;
  CudaBuffer compact_indices;
  CudaBuffer expert_map;
  CudaBuffer slot_map;
};

TEST(CUDA_EP_Unittest, QMoECompactExpertsSinglePassRequiresCapacityForEveryRoutedExpert) {
  CompactExpertsCase run;
  // A single pass cannot drop routes, so it needs min(num_experts, num_routes) = 6 slots even though the routes use
  // only four distinct experts.
  for (int capacity : {1, 3, 5}) {
    SCOPED_TRACE(capacity);
    ExpectThrowsContaining([&] { run.Launch(capacity, 0, false); }, "cannot hold every routed expert");
  }
  ExpectThrowsContaining([&] { run.Launch(6, 2, false); }, "requires a partial pass");
  ExpectThrowsContaining([&] { run.Launch(0, 0, false); }, "Invalid QMoE compact expert capacity");
  ExpectThrowsContaining([&] { run.Launch(CompactExpertsCase::kExperts + 1, 0, false); },
                         "Invalid QMoE compact expert capacity");

  run.Launch(6, 0, false);
  EXPECT_EQ(run.Compact(), (std::array<int, 6>{2, 0, 2, 3, 0, 1}));
  EXPECT_EQ(run.Slots(6), (std::vector<int>{1, 3, 5, 6, -1, -1}));
}

TEST(CUDA_EP_Unittest, QMoECompactExpertsPartialPassesMapOnlyTheirRankRange) {
  CompactExpertsCase run;
  // Two slots per pass over the four routed experts {1, 3, 5, 6}, ranked in ascending expert order.
  run.Launch(2, 0, true);
  EXPECT_EQ(run.Compact(), (std::array<int, 6>{-1, 0, -1, -1, 0, 1}));
  EXPECT_EQ(run.Slots(2), (std::vector<int>{1, 3}));

  run.Launch(2, 2, true);
  EXPECT_EQ(run.Compact(), (std::array<int, 6>{0, -1, 0, 1, -1, -1}));
  EXPECT_EQ(run.Slots(2), (std::vector<int>{5, 6}));

  // A pass that starts beyond the last routed expert is valid and maps nothing.
  run.Launch(2, CompactExpertsCase::kExperts - 1, true);
  EXPECT_EQ(run.Compact(), (std::array<int, 6>{-1, -1, -1, -1, -1, -1}));
  EXPECT_EQ(run.Slots(2), (std::vector<int>{-1, -1}));
}

TEST(CUDA_EP_Unittest, QMoECompactExpertsPartialPassRejectsInvalidOffsetOrCapacity) {
  CompactExpertsCase run;
  ExpectThrowsContaining([&] { run.Launch(2, -1, true); }, "Invalid QMoE compact expert pass offset");
  ExpectThrowsContaining([&] { run.Launch(2, CompactExpertsCase::kExperts, true); },
                         "Invalid QMoE compact expert pass offset");
  ExpectThrowsContaining([&] { run.Launch(0, 0, true); }, "Invalid QMoE compact expert capacity");
  ExpectThrowsContaining([&] { run.Launch(CompactExpertsCase::kExperts + 1, 0, true); },
                         "Invalid QMoE compact expert capacity");
}

TEST(CUDA_EP_Unittest, QMoEAccumulateExpertHistogramCountsExpertsBeyondTheSharedHistogram) {
  // The block-local histogram covers the first 1024 experts, so larger expert banks must count the rest directly.
  constexpr int num_experts = 2048;
  constexpr int shared_experts = 1024;
  constexpr int num_routes = 3000;
  constexpr int baseline = 5;
  std::vector<int> routes(num_routes);
  for (int route = 0; route < num_routes; ++route) {
    // Covers every expert and also produces -1 and num_experts, which no histogram bin may count.
    routes[route] = (route * 37) % (num_experts + 2) - 1;
  }
  routes[10] = routes[11] = 1500;
  routes[12] = num_experts - 1;

  std::vector<int> expected(num_experts, baseline);
  int tail_routes = 0;
  for (int expert : routes) {
    if (expert >= 0 && expert < num_experts) {
      ++expected[expert];
      tail_routes += expert >= shared_experts;
    }
  }
  ASSERT_GT(tail_routes, 0);

  CudaBuffer device_routes(routes.size() * sizeof(int));
  CudaBuffer device_hist(expected.size() * sizeof(int));
  device_routes.Upload(routes.data());
  std::vector<int> initial(num_experts, baseline);
  device_hist.Upload(initial.data());

  qmoe::LaunchQMoEAccumulateExpertHistogram(device_routes.As<int>(), num_routes, num_experts,
                                            device_hist.As<int>(), nullptr);
  std::vector<int> actual(num_experts);
  device_hist.Download(actual.data());
  int mismatches = 0;
  for (int expert = 0; expert < num_experts; ++expert) {
    if (actual[expert] != expected[expert] && ++mismatches <= 8) {
      ADD_FAILURE() << "expert " << expert << ": counted " << actual[expert] << ", expected " << expected[expert];
    }
  }
  EXPECT_EQ(mismatches, 0);
}

// A slot table placed between guard words that look like a resident slot, so a read just outside the table is
// classified as a hit instead of an incidental zero.
struct GuardedSlotTable {
  static constexpr int kGuardWords = 2;
  static constexpr int kGuardSlot = 7;

  explicit GuardedSlotTable(const std::vector<int>& slots)
      : buffer((slots.size() + 2 * kGuardWords) * sizeof(int)) {
    std::vector<int> host(slots.size() + 2 * kGuardWords, kGuardSlot);
    std::copy(slots.begin(), slots.end(), host.begin() + kGuardWords);
    buffer.Upload(host.data());
  }

  const int* Slots() { return buffer.As<int>() + kGuardWords; }

  CudaBuffer buffer;
};

// Expert 2 is resident in slot 1 and expert 5 in slot 0. The routes also hold what the router leaves for a row without
// a finite probability (-1) and indices beyond the expert bank.
constexpr int kClassifyExperts = 8;
constexpr int kClassifyRoutes = 6;
const std::array<int, kClassifyRoutes> kClassifyRouteExperts{2, -1, 5, kClassifyExperts, kClassifyExperts + 1, 3};
const std::vector<int> kClassifyExpertSlots{-1, -1, 1, -1, -1, 0, -1, -1};

TEST(CUDA_EP_Unittest, QMoEOffloadClassifyTreatsOutOfRangeExpertsAsMissesAndForwardsThem) {
  constexpr int k = 2;
  constexpr int dummy_slot = 4;
  GuardedSlotTable table(kClassifyExpertSlots);
  CudaBuffer topk(sizeof(kClassifyRouteExperts));
  CudaBuffer hit_ids(sizeof(kClassifyRouteExperts));
  CudaBuffer hit_compact(sizeof(kClassifyRouteExperts));
  CudaBuffer hit_flag(sizeof(kClassifyRouteExperts));
  CudaBuffer host_indices(sizeof(kClassifyRouteExperts));
  topk.Upload(kClassifyRouteExperts.data());

  CUDA_CALL_THROW(qmoe::LaunchQMoEOffloadClassify(
      topk.As<int>(), table.Slots(), kClassifyExperts, kClassifyRoutes, kClassifyRoutes, hit_ids.As<int>(),
      hit_compact.As<int>(), hit_flag.As<int>(), nullptr, dummy_slot, nullptr, nullptr, k, host_indices.As<int>()));

  std::array<int, kClassifyRoutes> ids{};
  std::array<int, kClassifyRoutes> compact{};
  std::array<int, kClassifyRoutes> flags{};
  std::array<int, kClassifyRoutes> forwarded{};
  hit_ids.Download(ids.data());
  hit_compact.Download(compact.data());
  hit_flag.Download(flags.data());
  host_indices.Download(forwarded.data());
  EXPECT_EQ(flags, (std::array<int, 6>{1, 0, 1, 0, 0, 0}));
  EXPECT_EQ(compact, (std::array<int, 6>{1, dummy_slot, 0, dummy_slot, dummy_slot, dummy_slot}));
  EXPECT_EQ(ids, (std::array<int, 6>{0, 0, 1, 1, 2, 2}));
  EXPECT_EQ(forwarded, kClassifyRouteExperts);
}

// Host memory that kernels write and the CPU polls, like the offload doorbell.
struct MappedHostBuffer {
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MappedHostBuffer);

  explicit MappedHostBuffer(size_t size) {
    CUDA_CALL_THROW(cudaHostAlloc(&host, size, cudaHostAllocMapped));
    std::memset(host, 0, size);
    CUDA_CALL_THROW(cudaHostGetDevicePointer(&device, host, 0));
  }

  ~MappedHostBuffer() noexcept {
    const cudaError_t status = cudaFreeHost(host);
    if (status != cudaSuccess) {
      std::fprintf(stderr, "cudaFreeHost failed during QMoE test cleanup: %s\n", cudaGetErrorString(status));
    }
  }

  template <typename T>
  T* Host() { return static_cast<T*>(host); }

  template <typename T>
  T* Device() { return static_cast<T*>(device); }

  void* host = nullptr;
  void* device = nullptr;
};

TEST(CUDA_EP_Unittest, QMoEOffloadClassifyDoorbellForwardsRoutesAndRingsOncePerLaunch) {
  constexpr int tile_rows = 3;
  GuardedSlotTable table(kClassifyExpertSlots);
  CudaBuffer topk(sizeof(kClassifyRouteExperts));
  CudaBuffer hit_ids(sizeof(kClassifyRouteExperts));
  CudaBuffer hit_compact(sizeof(kClassifyRouteExperts));
  CudaBuffer hit_flag(sizeof(kClassifyRouteExperts));
  CudaBuffer doorbell_seq(sizeof(uint64_t));
  MappedHostBuffer host_indices(sizeof(kClassifyRouteExperts));
  MappedHostBuffer doorbell_request(sizeof(uint64_t));
  MappedHostBuffer doorbell_tile_rows(sizeof(int32_t));
  topk.Upload(kClassifyRouteExperts.data());
  const uint64_t initial_seq = 0;
  doorbell_seq.Upload(&initial_seq);

  // The doorbell path runs one block of 256 threads, almost all of them idle for six routes; every thread must still
  // reach the fence and barrier that precede the doorbell.
  for (uint64_t launch = 1; launch <= 2; ++launch) {
    SCOPED_TRACE(launch);
    CUDA_CALL_THROW(qmoe::LaunchQMoEOffloadClassify(
        topk.As<int>(), table.Slots(), kClassifyExperts, kClassifyRoutes, kClassifyRoutes, hit_ids.As<int>(),
        hit_compact.As<int>(), hit_flag.As<int>(), nullptr, 0, nullptr, nullptr, 0, host_indices.Device<int>(),
        nullptr, nullptr, 0, doorbell_seq.As<uint64_t>(), doorbell_request.Device<uint64_t>(),
        doorbell_tile_rows.Device<int32_t>(), tile_rows));
    CUDA_CALL_THROW(cudaDeviceSynchronize());

    uint64_t seq = 0;
    doorbell_seq.Download(&seq);
    const uint64_t request = *doorbell_request.Host<volatile uint64_t>();
    const int32_t published_tile_rows = *doorbell_tile_rows.Host<volatile int32_t>();
    EXPECT_EQ(seq, launch);
    EXPECT_EQ(request, launch);
    EXPECT_EQ(published_tile_rows, tile_rows);
    for (int route = 0; route < kClassifyRoutes; ++route) {
      EXPECT_EQ(host_indices.Host<int>()[route], kClassifyRouteExperts[route]) << "route " << route;
    }
  }

  std::array<int, kClassifyRoutes> flags{};
  hit_flag.Download(flags.data());
  EXPECT_EQ(flags, (std::array<int, 6>{1, 0, 1, 0, 0, 0}));
}

template <typename T>
void TestScatterBiasOfOutOfRangeExperts() {
  constexpr int source_rows = 3;
  constexpr int route_count = 4;
  constexpr int hidden_size = 5;
  // The bias table sits between rows of a nonzero sentinel, so an unchecked read below or above it would copy the
  // sentinel instead of the zero row.
  std::vector<T> bias((source_rows + 2) * hidden_size, T(77.0f));
  for (int row = 0; row < source_rows; ++row) {
    for (int col = 0; col < hidden_size; ++col) {
      bias[(row + 1) * hidden_size + col] = T(static_cast<float>(1 + row * 10 + col));
    }
  }
  const std::array<int, route_count> experts{2, -1, 0, source_rows};
  const std::array<int, route_count> route_ids{3, 2, 1, 0};
  const std::vector<T> route_bias_init(route_count * hidden_size, T(-5.0f));

  CudaBuffer device_bias(bias.size() * sizeof(T));
  CudaBuffer device_experts(sizeof(experts));
  CudaBuffer device_route_ids(sizeof(route_ids));
  CudaBuffer device_route_bias(route_bias_init.size() * sizeof(T));
  device_bias.Upload(bias.data());
  device_experts.Upload(experts.data());
  device_route_ids.Upload(route_ids.data());
  device_route_bias.Upload(route_bias_init.data());

  CUDA_CALL_THROW(qmoe::LaunchQMoEOffloadScatterBias(
      device_bias.As<T>() + hidden_size, device_experts.As<int>(), device_route_ids.As<int>(),
      device_route_bias.As<T>(), route_count, hidden_size, source_rows, nullptr));

  std::vector<T> route_bias(route_bias_init.size());
  device_route_bias.Download(route_bias.data());
  for (int route = 0; route < route_count; ++route) {
    const int expert = experts[route];
    const bool in_range = expert >= 0 && expert < source_rows;
    for (int col = 0; col < hidden_size; ++col) {
      const float expected = in_range ? static_cast<float>(1 + expert * 10 + col) : 0.0f;
      EXPECT_EQ(static_cast<float>(route_bias[route_ids[route] * hidden_size + col]), expected)
          << "route " << route << " expert " << expert << " column " << col;
    }
  }
}

TEST(CUDA_EP_Unittest, QMoEOffloadScatterBiasZeroesRowsOfOutOfRangeExperts) {
  TestScatterBiasOfOutOfRangeExperts<half>();
  TestScatterBiasOfOutOfRangeExperts<__nv_bfloat16>();
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
  if (!SupportsSm80OrLater()) {
    GTEST_SKIP() << "NVFP4 GEMV requires SM80+.";
  }
  TestNvfp4GemvScaleCodeConfigurations();
}

TEST(CUDA_EP_Unittest, QMoENvfp4GemvNativeScale) {
  if (!ParseEnvironmentVariableWithDefault<bool>("ORT_NVFP4_GEMV_NATIVE_SCALE", false)) {
    GTEST_SKIP() << "Native-scale coverage runs in its flag-enabled CTest process.";
  }
  if (!SupportsSm80OrLater() || !onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvSupportsNativeScale()) {
    GTEST_SKIP() << "Native E4M3 scale conversion requires FP8-enabled kernels compiled for SM90+.";
  }
  TestNvfp4GemvScaleCodeConfigurations();
}
#endif

}  // namespace
}  // namespace test
}  // namespace onnxruntime
