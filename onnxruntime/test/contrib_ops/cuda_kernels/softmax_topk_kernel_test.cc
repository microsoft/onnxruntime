#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "core/providers/cuda/cuda_common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace onnxruntime {
namespace test {
namespace {

struct CudaBuffer {
  void* data = nullptr;
  size_t bytes = 0;

  explicit CudaBuffer(size_t size_in_bytes) : bytes(size_in_bytes) {
    CUDA_CALL_THROW(cudaMalloc(&data, bytes));
  }

  ~CudaBuffer() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }

  template <typename T>
  T* As() {
    return reinterpret_cast<T*>(data);
  }

  void CopyFromHost(const void* src) {
    CUDA_CALL_THROW(cudaMemcpy(data, src, bytes, cudaMemcpyHostToDevice));
  }

  void CopyToHost(void* dst) const {
    CUDA_CALL_THROW(cudaMemcpy(dst, data, bytes, cudaMemcpyDeviceToHost));
  }
};

struct ExpectedTopK {
  std::vector<float> scales;
  std::vector<int> indices;
};

template <typename T>
float ToFloat(T value) {
  return static_cast<float>(value);
}

template <typename T>
T FromFloat(float value) {
  return static_cast<T>(value);
}

template <typename T>
ExpectedTopK ReferenceSoftmaxTopK(const std::vector<T>& logits, int num_rows, int num_experts, int k,
                                  bool normalize_scales) {
  ExpectedTopK expected;
  expected.scales.resize(static_cast<size_t>(num_rows) * k);
  expected.indices.resize(static_cast<size_t>(num_rows) * k);

  for (int row = 0; row < num_rows; ++row) {
    const T* row_logits = logits.data() + static_cast<size_t>(row) * num_experts;
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int expert = 0; expert < num_experts; ++expert) {
      max_logit = std::max(max_logit, ToFloat(row_logits[expert]));
    }

    float sum_exp = 0.0f;
    for (int expert = 0; expert < num_experts; ++expert) {
      sum_exp += std::exp(ToFloat(row_logits[expert]) - max_logit);
    }

    std::vector<int> order(num_experts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
      const float lhs_logit = ToFloat(row_logits[lhs]);
      const float rhs_logit = ToFloat(row_logits[rhs]);
      return lhs_logit > rhs_logit || (lhs_logit == rhs_logit && lhs < rhs);
    });

    float topk_sum = 0.0f;
    for (int rank = 0; rank < k; ++rank) {
      const int expert = order[rank];
      const float scale = std::exp(ToFloat(row_logits[expert]) - max_logit) / sum_exp;
      expected.scales[static_cast<size_t>(row) * k + rank] = scale;
      expected.indices[static_cast<size_t>(row) * k + rank] = expert;
      topk_sum += scale;
    }

    if (normalize_scales && topk_sum > 1e-6f) {
      for (int rank = 0; rank < k; ++rank) {
        expected.scales[static_cast<size_t>(row) * k + rank] /= topk_sum;
      }
    }
  }

  return expected;
}

template <typename T>
void RunSoftmaxTopKTest(int num_rows, int num_experts, int k, bool normalize_scales,
                        const std::vector<float>& logits_float, float tolerance) {
  ASSERT_EQ(logits_float.size(), static_cast<size_t>(num_rows) * num_experts);

  std::vector<T> logits(logits_float.size());
  std::transform(logits_float.begin(), logits_float.end(), logits.begin(), [](float value) {
    return FromFloat<T>(value);
  });

  CudaBuffer d_logits(logits.size() * sizeof(T));
  CudaBuffer d_scales(static_cast<size_t>(num_rows) * k * sizeof(float));
  CudaBuffer d_indices(static_cast<size_t>(num_rows) * k * sizeof(int));
  d_logits.CopyFromHost(logits.data());

  cudaStream_t stream = nullptr;
  CUDA_CALL_THROW(cudaStreamCreate(&stream));
  onnxruntime::contrib::cuda::LaunchSoftmaxTopK(
      d_logits.As<T>(), d_scales.As<float>(), d_indices.As<int>(), num_rows, num_experts, k, normalize_scales, stream);
  CUDA_CALL_THROW(cudaGetLastError());
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  CUDA_CALL_THROW(cudaStreamDestroy(stream));

  std::vector<float> actual_scales(static_cast<size_t>(num_rows) * k);
  std::vector<int> actual_indices(static_cast<size_t>(num_rows) * k);
  d_scales.CopyToHost(actual_scales.data());
  d_indices.CopyToHost(actual_indices.data());

  const ExpectedTopK expected = ReferenceSoftmaxTopK(logits, num_rows, num_experts, k, normalize_scales);
  ASSERT_EQ(actual_indices, expected.indices);
  ASSERT_EQ(actual_scales.size(), expected.scales.size());
  for (size_t i = 0; i < actual_scales.size(); ++i) {
    EXPECT_NEAR(actual_scales[i], expected.scales[i], tolerance) << "at flattened top-k index " << i;
  }
}

std::vector<float> MakeLogits(int num_rows, int num_experts) {
  std::vector<float> logits(static_cast<size_t>(num_rows) * num_experts);
  for (int row = 0; row < num_rows; ++row) {
    for (int expert = 0; expert < num_experts; ++expert) {
      const int v = (expert * 17 + row * 11) % 29;
      logits[static_cast<size_t>(row) * num_experts + expert] = 0.125f * static_cast<float>(v - 14);
    }
  }
  return logits;
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_WarpBitonicStableTiesFloat) {
  constexpr int num_rows = 9;
  constexpr int num_experts = 8;
  constexpr int k = 4;
  std::vector<float> logits = {
      4.0f, 1.0f, 4.0f, 0.0f, 3.0f, 4.0f, -2.0f, 4.0f,
      -1.0f, 2.0f, 2.0f, 5.0f, 5.0f, 5.0f, 0.5f, -3.0f,
      0.0f, 0.0f, 0.0f, -1.0f, -1.0f, 2.0f, 2.0f, 2.0f,
      3.0f, 1.0f, 2.0f, 3.0f, 3.0f, -4.0f, -5.0f, 0.0f,
      1.5f, 1.5f, 1.0f, 2.0f, 0.0f, 2.0f, 2.0f, -2.0f,
      -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f,
      7.0f, 7.0f, 6.0f, 6.0f, 5.0f, 5.0f, 4.0f, 4.0f,
      -1.0f, 8.0f, -1.0f, 8.0f, 3.0f, 8.0f, 2.0f, 1.0f,
      0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f};

  RunSoftmaxTopKTest<float>(num_rows, num_experts, k, false, logits, 1e-6f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_WarpBitonicBoundaryNormalizeHalf) {
  constexpr int num_rows = 10;
  constexpr int num_experts = 32;
  constexpr int k = 8;
  auto logits = MakeLogits(num_rows, num_experts);
  logits[3 * num_experts + 0] = 6.0f;
  logits[3 * num_experts + 5] = 6.0f;
  logits[3 * num_experts + 17] = 6.0f;

  RunSoftmaxTopKTest<half>(num_rows, num_experts, k, true, logits, 2e-3f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_WarpMergeStableTiesFloat) {
  constexpr int num_rows = 5;
  constexpr int num_experts = 64;
  constexpr int k = 8;
  auto logits = MakeLogits(num_rows, num_experts);
  for (int expert : {2, 13, 31, 63}) {
    logits[static_cast<size_t>(1) * num_experts + expert] = 5.0f;
  }
  for (int expert : {0, 16, 32, 48}) {
    logits[static_cast<size_t>(4) * num_experts + expert] = 4.5f;
  }

  RunSoftmaxTopKTest<float>(num_rows, num_experts, k, false, logits, 1e-6f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_WarpMergeNormalizeBFloat16) {
  constexpr int num_rows = 6;
  constexpr int num_experts = 64;
  constexpr int k = 16;
  auto logits = MakeLogits(num_rows, num_experts);
  logits[2 * num_experts + 4] = 7.0f;
  logits[2 * num_experts + 9] = 7.0f;
  logits[2 * num_experts + 47] = 7.0f;

  RunSoftmaxTopKTest<__nv_bfloat16>(num_rows, num_experts, k, true, logits, 2e-2f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_BlockMergeStillMatchesReference) {
  constexpr int num_rows = 4;
  constexpr int num_experts = 128;
  constexpr int k = 8;
  auto logits = MakeLogits(num_rows, num_experts);
  logits[0] = 9.0f;
  logits[7] = 9.0f;
  logits[65] = 9.0f;

  RunSoftmaxTopKTest<float>(num_rows, num_experts, k, true, logits, 1e-6f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_WarpBitonicHandlesNegativeInfinityPadding) {
  constexpr int num_rows = 2;
  constexpr int num_experts = 8;
  constexpr int k = 8;
  constexpr float neg_inf = -std::numeric_limits<float>::infinity();
  std::vector<float> logits = {
      3.0f, neg_inf, 2.0f, neg_inf, 1.0f, neg_inf, 0.0f, neg_inf,
      neg_inf, 4.0f, neg_inf, neg_inf, 3.0f, neg_inf, neg_inf, 2.0f};

  RunSoftmaxTopKTest<float>(num_rows, num_experts, k, false, logits, 1e-6f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_WarpMergeHandlesNegativeInfinityPadding) {
  constexpr int num_rows = 2;
  constexpr int num_experts = 40;
  constexpr int k = 40;
  constexpr float neg_inf = -std::numeric_limits<float>::infinity();
  std::vector<float> logits(static_cast<size_t>(num_rows) * num_experts, neg_inf);
  for (int expert = 0; expert < 10; ++expert) {
    logits[expert * 3] = static_cast<float>(expert) * 0.25f;
    logits[static_cast<size_t>(1) * num_experts + expert * 2] = 3.0f - static_cast<float>(expert) * 0.125f;
  }

  RunSoftmaxTopKTest<float>(num_rows, num_experts, k, true, logits, 1e-6f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_BlockMergeHandlesNegativeInfinityPadding) {
  constexpr int num_rows = 2;
  constexpr int num_experts = 80;
  constexpr int k = 64;
  constexpr float neg_inf = -std::numeric_limits<float>::infinity();
  std::vector<float> logits(static_cast<size_t>(num_rows) * num_experts, neg_inf);
  for (int expert = 0; expert < 20; ++expert) {
    logits[expert * 2] = static_cast<float>(expert) * 0.125f;
    logits[static_cast<size_t>(1) * num_experts + expert * 3] = 5.0f - static_cast<float>(expert) * 0.25f;
  }

  RunSoftmaxTopKTest<float>(num_rows, num_experts, k, true, logits, 1e-6f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_RejectsKGreaterThanExperts) {
  EXPECT_THROW(onnxruntime::contrib::cuda::LaunchSoftmaxTopK(
                   static_cast<const float*>(nullptr), nullptr, nullptr, 1, 8, 9, false, nullptr),
               OnnxRuntimeException);
}

TEST(CUDA_EP_Unittest, FusedRouteMaps_CanUsePredicate) {
  using onnxruntime::contrib::cuda::CanUseFusedRouteMaps;
  for (int64_t rows : {1, 2, 4, 7, 8}) {
    for (int64_t k : {1, 2, 4, 6, 8, 10}) {
      EXPECT_TRUE(CanUseFusedRouteMaps(rows, 512, k)) << "rows=" << rows << ", k=" << k;
    }
  }
  EXPECT_FALSE(CanUseFusedRouteMaps(0, 512, 10));
  EXPECT_FALSE(CanUseFusedRouteMaps(9, 512, 10));
  EXPECT_FALSE(CanUseFusedRouteMaps(8, 256, 10));
  EXPECT_FALSE(CanUseFusedRouteMaps(8, 32, 4));
  EXPECT_FALSE(CanUseFusedRouteMaps(8, 512, 3));
  EXPECT_FALSE(CanUseFusedRouteMaps(8, 512, 5));
  EXPECT_FALSE(CanUseFusedRouteMaps(8, 512, 12));
}

template <typename T>
void RunFusedRouteMapsParityAndFallbackTest(bool normalize_scales) {
  constexpr int num_rows = 8;
  constexpr int num_experts = 512;
  constexpr int k = 10;
  constexpr int num_pairs = num_rows * k;

  auto logits_float = MakeLogits(num_rows, num_experts);
  // Include exact ties across rows to verify stable tie-breaking toward lower expert indices.
  logits_float[0 * num_experts + 3] = 6.0f;
  logits_float[0 * num_experts + 19] = 6.0f;
  logits_float[5 * num_experts + 128] = 5.5f;
  logits_float[5 * num_experts + 257] = 5.5f;

  // Allocate +1 element so we can also test a 2-byte misaligned pointer (declined launch).
  std::vector<T> logits_padded(logits_float.size() + 1);
  logits_padded[0] = FromFloat<T>(0.0f);
  for (size_t i = 0; i < logits_float.size(); ++i) {
    logits_padded[i] = FromFloat<T>(logits_float[i]);
  }

  CudaBuffer d_logits(logits_padded.size() * sizeof(T));
  d_logits.CopyFromHost(logits_padded.data());

  CudaBuffer d_ref_scales(static_cast<size_t>(num_pairs) * sizeof(float));
  CudaBuffer d_ref_indices(static_cast<size_t>(num_pairs) * sizeof(int));
  CudaBuffer d_scales(static_cast<size_t>(num_pairs) * sizeof(float));
  CudaBuffer d_indices(static_cast<size_t>(num_pairs) * sizeof(int));
  CudaBuffer d_r2u(static_cast<size_t>(num_pairs) * sizeof(int));
  CudaBuffer d_u2r(static_cast<size_t>(num_pairs) * sizeof(int));
  CudaBuffer d_expert(static_cast<size_t>(num_pairs) * sizeof(int));
  CudaBuffer d_offsets(static_cast<size_t>(num_experts + 1) * sizeof(int64_t));

  cudaStream_t stream = nullptr;
  CUDA_CALL_THROW(cudaStreamCreate(&stream));

  // Unaligned logits pointer (offset by 2 bytes) must be declined without launching.
  const T* unaligned_logits = d_logits.As<T>() + 1;
  EXPECT_FALSE(onnxruntime::contrib::cuda::LaunchFusedRouteMaps(
      unaligned_logits, d_scales.As<float>(), d_indices.As<int>(), d_r2u.As<int>(), d_u2r.As<int>(),
      d_expert.As<int>(), d_offsets.As<int64_t>(), num_rows, num_experts, k, normalize_scales, false, stream));

  onnxruntime::contrib::cuda::LaunchSoftmaxTopK(
      d_logits.As<T>(), d_ref_scales.As<float>(), d_ref_indices.As<int>(), num_rows, num_experts, k,
      normalize_scales, stream);

  const bool launched = onnxruntime::contrib::cuda::LaunchFusedRouteMaps(
      d_logits.As<T>(), d_scales.As<float>(), d_indices.As<int>(), d_r2u.As<int>(), d_u2r.As<int>(),
      d_expert.As<int>(), d_offsets.As<int64_t>(), num_rows, num_experts, k, normalize_scales, false, stream);
  CUDA_CALL_THROW(cudaGetLastError());
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  CUDA_CALL_THROW(cudaStreamDestroy(stream));

  if (!launched) {
    GTEST_SKIP() << "SM90 fused route maps kernel is not available on this device/build.";
  }

  std::vector<float> ref_scales(num_pairs);
  std::vector<int> ref_indices(num_pairs);
  std::vector<float> actual_scales(num_pairs);
  std::vector<int> actual_indices(num_pairs);
  std::vector<int> actual_r2u(num_pairs);
  std::vector<int> actual_u2r(num_pairs);
  std::vector<int> actual_expert(num_pairs);
  std::vector<int64_t> actual_offsets(num_experts + 1);

  d_ref_scales.CopyToHost(ref_scales.data());
  d_ref_indices.CopyToHost(ref_indices.data());
  d_scales.CopyToHost(actual_scales.data());
  d_indices.CopyToHost(actual_indices.data());
  d_r2u.CopyToHost(actual_r2u.data());
  d_u2r.CopyToHost(actual_u2r.data());
  d_expert.CopyToHost(actual_expert.data());
  d_offsets.CopyToHost(actual_offsets.data());

  EXPECT_EQ(actual_indices, ref_indices);
  EXPECT_EQ(actual_scales, ref_scales);

  struct PairItem {
    int expert;
    int token;
    int k_idx;
  };
  std::vector<PairItem> pairs(num_pairs);
  for (int p = 0; p < num_pairs; ++p) {
    pairs[p] = PairItem{ref_indices[p], p / k, p % k};
  }
  std::sort(pairs.begin(), pairs.end(), [](const PairItem& a, const PairItem& b) {
    return a.expert < b.expert || (a.expert == b.expert && a.token < b.token);
  });

  std::vector<int> expected_r2u(num_pairs);
  std::vector<int> expected_u2r(num_pairs);
  std::vector<int> expected_expert(num_pairs);
  std::vector<int64_t> expected_offsets(num_experts + 1, 0);
  for (int rank = 0; rank < num_pairs; ++rank) {
    const int unpermuted_row = pairs[rank].k_idx * num_rows + pairs[rank].token;
    expected_r2u[rank] = unpermuted_row;
    expected_u2r[unpermuted_row] = rank;
    expected_expert[rank] = pairs[rank].expert;
  }
  for (int e = 0; e <= num_experts; ++e) {
    expected_offsets[e] = std::count_if(pairs.begin(), pairs.end(),
                                        [e](const PairItem& item) { return item.expert < e; });
  }

  EXPECT_EQ(actual_r2u, expected_r2u);
  EXPECT_EQ(actual_u2r, expected_u2r);
  EXPECT_EQ(actual_expert, expected_expert);
  EXPECT_EQ(actual_offsets, expected_offsets);
}

TEST(CUDA_EP_Unittest, FusedRouteMaps_S8E512K10_Half) {
  RunFusedRouteMapsParityAndFallbackTest<half>(true);
  RunFusedRouteMapsParityAndFallbackTest<half>(false);
}

TEST(CUDA_EP_Unittest, FusedRouteMaps_S8E512K10_BFloat16) {
  RunFusedRouteMapsParityAndFallbackTest<__nv_bfloat16>(true);
  RunFusedRouteMapsParityAndFallbackTest<__nv_bfloat16>(false);
}

void VerifyPairedFirstRouteGroups(const std::vector<int>& indices, const std::vector<int2>& groups) {
  const int num_routes = static_cast<int>(indices.size());
  ASSERT_EQ(static_cast<int>(groups.size()), num_routes);

  std::vector<int> visit_count(static_cast<size_t>(num_routes), 0);
  std::vector<int> expert_total_routes(512, 0);
  std::vector<int> expert_singleton_routes(512, 0);
  for (int e : indices) {
    ASSERT_GE(e, 0);
    ASSERT_LT(e, 512);
    ++expert_total_routes[static_cast<size_t>(e)];
  }

  int phase = 0;  // 0: paired (r0 >= 0 && r1 >= 0), 1: singleton (r0 >= 0 && r1 == -1), 2: padding (-1, -1)
  int total_grouped_routes = 0;
  for (int i = 0; i < num_routes; ++i) {
    const int2 grp = groups[static_cast<size_t>(i)];
    if (grp.x >= 0 && grp.y >= 0) {
      EXPECT_EQ(phase, 0) << "Paired group found after singleton/padding at index " << i;
      ASSERT_LT(grp.x, num_routes);
      ASSERT_LT(grp.y, num_routes);
      EXPECT_LT(grp.x, grp.y) << "Paired routes must be ordered ascending within each pair";
      EXPECT_EQ(indices[static_cast<size_t>(grp.x)], indices[static_cast<size_t>(grp.y)])
          << "Paired routes must select the same expert at group " << i;
      ++visit_count[static_cast<size_t>(grp.x)];
      ++visit_count[static_cast<size_t>(grp.y)];
      total_grouped_routes += 2;
    } else if (grp.x >= 0 && grp.y == -1) {
      if (phase == 0) phase = 1;
      EXPECT_EQ(phase, 1) << "Singleton group found after padding at index " << i;
      ASSERT_LT(grp.x, num_routes);
      ++visit_count[static_cast<size_t>(grp.x)];
      ++expert_singleton_routes[static_cast<size_t>(indices[static_cast<size_t>(grp.x)])];
      total_grouped_routes += 1;
    } else {
      EXPECT_EQ(grp.x, -1);
      EXPECT_EQ(grp.y, -1);
      phase = 2;
    }
  }

  EXPECT_EQ(total_grouped_routes, num_routes);
  for (int r = 0; r < num_routes; ++r) {
    EXPECT_EQ(visit_count[static_cast<size_t>(r)], 1) << "Route " << r << " was not visited exactly once";
  }
  for (int e = 0; e < 512; ++e) {
    EXPECT_EQ(expert_singleton_routes[static_cast<size_t>(e)], expert_total_routes[static_cast<size_t>(e)] & 1)
        << "Expert " << e << " with count " << expert_total_routes[static_cast<size_t>(e)]
        << " should have singleton count (count % 2)";
  }
}

template <typename T>
void RunFusedInt3RouteGroupsTest(int num_rows, int k, bool normalize_scales) {
  constexpr int num_experts = 512;
  const int num_routes = num_rows * k;

  // Build logits with tied values and mixed paired/singleton expert runs across the 8 rows:
  // - Expert 7 selected by 4 rows (2 pairs)
  // - Expert 42 selected by 3 rows (1 pair + 1 singleton)
  // - Expert 199 selected by 2 rows (1 pair)
  // - Expert 311 selected by 5 rows (2 pairs + 1 singleton)
  // - Tied high logits on each row to exercise deterministic lower-index-first tie breaking
  std::vector<float> logits_float = MakeLogits(num_rows, num_experts);
  for (int r = 0; r < num_rows; ++r) {
    if (r < 4) logits_float[static_cast<size_t>(r) * num_experts + 7] = 12.0f;
    if (r < 3) logits_float[static_cast<size_t>(r) * num_experts + 42] = 11.5f;
    if (r < 2) logits_float[static_cast<size_t>(r) * num_experts + 199] = 11.0f;
    if (r < 5) logits_float[static_cast<size_t>(r) * num_experts + 311] = 10.5f;
    // Inject 4-way exact ties per row at distinct expert indices across the 4 warp chunks (0..511).
    const int tie_base = 10 + r * 3;
    logits_float[static_cast<size_t>(r) * num_experts + tie_base] = 9.25f;
    logits_float[static_cast<size_t>(r) * num_experts + tie_base + 128] = 9.25f;
    logits_float[static_cast<size_t>(r) * num_experts + tie_base + 256] = 9.25f;
    logits_float[static_cast<size_t>(r) * num_experts + tie_base + 384] = 9.25f;
  }

  std::vector<T> logits(logits_float.size());
  std::transform(logits_float.begin(), logits_float.end(), logits.begin(), [](float value) {
    return FromFloat<T>(value);
  });

  CudaBuffer d_logits(logits.size() * sizeof(T));
  CudaBuffer d_fused_scales(static_cast<size_t>(num_routes) * sizeof(float));
  CudaBuffer d_fused_indices(static_cast<size_t>(num_routes) * sizeof(int));
  CudaBuffer d_fused_groups(static_cast<size_t>(num_routes) * sizeof(int2));
  CudaBuffer d_ref_scales(static_cast<size_t>(num_routes) * sizeof(float));
  CudaBuffer d_ref_indices(static_cast<size_t>(num_routes) * sizeof(int));
  CudaBuffer d_unfused_groups(static_cast<size_t>(num_routes) * sizeof(int2));
  d_logits.CopyFromHost(logits.data());

  cudaStream_t stream = nullptr;
  CUDA_CALL_THROW(cudaStreamCreate(&stream));

  const bool launched = onnxruntime::contrib::cuda::LaunchFusedInt3RouteGroups(
      d_logits.As<T>(), d_fused_scales.As<float>(), d_fused_indices.As<int>(), d_fused_groups.As<int2>(),
      num_rows, num_experts, k, normalize_scales, stream);
  ASSERT_TRUE(launched);

  onnxruntime::contrib::cuda::LaunchSoftmaxTopK(
      d_logits.As<T>(), d_ref_scales.As<float>(), d_ref_indices.As<int>(),
      num_rows, num_experts, k, normalize_scales, stream);
  onnxruntime::contrib::cuda::LaunchBuildInt3RouteGroups(
      d_ref_indices.As<int>(), d_unfused_groups.As<int2>(), num_routes, stream);
  CUDA_CALL_THROW(cudaGetLastError());
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  CUDA_CALL_THROW(cudaStreamDestroy(stream));

  std::vector<float> fused_scales(static_cast<size_t>(num_routes));
  std::vector<int> fused_indices(static_cast<size_t>(num_routes));
  std::vector<int2> fused_groups(static_cast<size_t>(num_routes));
  std::vector<float> ref_scales(static_cast<size_t>(num_routes));
  std::vector<int> ref_indices(static_cast<size_t>(num_routes));
  std::vector<int2> unfused_groups(static_cast<size_t>(num_routes));

  d_fused_scales.CopyToHost(fused_scales.data());
  d_fused_indices.CopyToHost(fused_indices.data());
  d_fused_groups.CopyToHost(fused_groups.data());
  d_ref_scales.CopyToHost(ref_scales.data());
  d_ref_indices.CopyToHost(ref_indices.data());
  d_unfused_groups.CopyToHost(unfused_groups.data());

  ASSERT_EQ(fused_indices, ref_indices);
  for (int i = 0; i < num_routes; ++i) {
    EXPECT_FLOAT_EQ(fused_scales[static_cast<size_t>(i)], ref_scales[static_cast<size_t>(i)])
        << "Fused vs existing router scale mismatch at route " << i;
  }

  VerifyPairedFirstRouteGroups(fused_indices, fused_groups);
  VerifyPairedFirstRouteGroups(ref_indices, unfused_groups);
  for (int i = 0; i < num_routes; ++i) {
    EXPECT_EQ(fused_groups[static_cast<size_t>(i)].x, unfused_groups[static_cast<size_t>(i)].x);
    EXPECT_EQ(fused_groups[static_cast<size_t>(i)].y, unfused_groups[static_cast<size_t>(i)].y);
  }
}

TEST(CUDA_EP_Unittest, FusedInt3RouteGroups_512ExpertsParityAndGrouping) {
  for (int k : {10, 16}) {
    for (bool normalize_scales : {false, true}) {
      SCOPED_TRACE(testing::Message() << "k=" << k << " normalize=" << normalize_scales);
      RunFusedInt3RouteGroupsTest<half>(8, k, normalize_scales);
      RunFusedInt3RouteGroupsTest<__nv_bfloat16>(8, k, normalize_scales);
    }
  }
}

// Only the vectorized INT3 decode GEMV kernels read route groups, and QMoE builds them only when this selector picks
// that family. The selection inspects shapes and pointer alignment, so the addresses need no backing allocation.
TEST(CUDA_EP_Unittest, FusedInt3RouteGroups_DecodeKernelSelection) {
  using onnxruntime::contrib::cuda::Int3DecodeKernel;
  using onnxruntime::contrib::cuda::Int3MoeParams;
  using onnxruntime::contrib::cuda::SelectInt3DecodeKernel;

  const auto bytes = [](uintptr_t address) { return reinterpret_cast<const uint8_t*>(address); };
  struct Case {
    const char* name;
    void (*mutate)(Int3MoeParams&);
    Int3DecodeKernel expected;
  };
  const Case cases[] = {
      {"vectorized shape", [](Int3MoeParams&) {}, Int3DecodeKernel::kFastVectorized},
      {"route groups and router weights are ignored",
       [](Int3MoeParams& p) {
         p.route_groups = reinterpret_cast<const int2*>(uintptr_t{0x70000});
         p.router_weights = reinterpret_cast<const void*>(uintptr_t{0x80000});
       },
       Int3DecodeKernel::kFastVectorized},
      {"4-byte aligned weights", [](Int3MoeParams& p) { p.fc1.weights = reinterpret_cast<const uint8_t*>(uintptr_t{0x50004}); },
       Int3DecodeKernel::kFastVectorized},
      {"hidden at the shared memory limit", [](Int3MoeParams& p) { p.hidden = 12288; },
       Int3DecodeKernel::kFastVectorized},
      {"block size 16", [](Int3MoeParams& p) { p.block_size = 16; }, Int3DecodeKernel::kFast},
      {"input offset by 2 bytes", [](Int3MoeParams& p) { p.input = reinterpret_cast<const void*>(uintptr_t{0x10002}); },
       Int3DecodeKernel::kFast},
      {"output offset by 2 bytes", [](Int3MoeParams& p) { p.output = reinterpret_cast<void*>(uintptr_t{0x20002}); },
       Int3DecodeKernel::kFast},
      {"activation offset by 4 bytes", [](Int3MoeParams& p) { p.activation = reinterpret_cast<void*>(uintptr_t{0x30004}); },
       Int3DecodeKernel::kFast},
      {"fc2 partial offset by 8 bytes", [](Int3MoeParams& p) { p.fc2_partial = reinterpret_cast<void*>(uintptr_t{0x40008}); },
       Int3DecodeKernel::kFast},
      {"fc1 weights offset by 1 byte", [](Int3MoeParams& p) { p.fc1.weights = reinterpret_cast<const uint8_t*>(uintptr_t{0x50001}); },
       Int3DecodeKernel::kScalar},
      {"fc2 weights offset by 2 bytes", [](Int3MoeParams& p) { p.fc2.weights = reinterpret_cast<const uint8_t*>(uintptr_t{0x60002}); },
       Int3DecodeKernel::kScalar},
      {"fc1 zero points", [](Int3MoeParams& p) { p.fc1.zeros = reinterpret_cast<const uint8_t*>(uintptr_t{0x90000}); },
       Int3DecodeKernel::kScalar},
      {"fc2 zero points", [](Int3MoeParams& p) { p.fc2.zeros = reinterpret_cast<const uint8_t*>(uintptr_t{0x90000}); },
       Int3DecodeKernel::kScalar},
      {"no fc2 partial buffer", [](Int3MoeParams& p) { p.fc2_partial = nullptr; }, Int3DecodeKernel::kScalar},
      {"hidden 128", [](Int3MoeParams& p) { p.hidden = 128; }, Int3DecodeKernel::kScalar},
      {"intermediate 192", [](Int3MoeParams& p) { p.intermediate = 192; }, Int3DecodeKernel::kScalar},
      {"block size 12", [](Int3MoeParams& p) { p.block_size = 12; }, Int3DecodeKernel::kScalar},
      {"routes beyond the grid limit", [](Int3MoeParams& p) { p.rows = 40000; }, Int3DecodeKernel::kScalar},
      {"hidden above the shared memory limit", [](Int3MoeParams& p) { p.hidden = 12544; },
       Int3DecodeKernel::kScalar},
  };
  for (const Case& test_case : cases) {
    Int3MoeParams params{};
    params.input = reinterpret_cast<const void*>(uintptr_t{0x10000});
    params.output = reinterpret_cast<void*>(uintptr_t{0x20000});
    params.activation = reinterpret_cast<void*>(uintptr_t{0x30000});
    params.fc2_partial = reinterpret_cast<void*>(uintptr_t{0x40000});
    params.fc1.weights = bytes(0x50000);
    params.fc2.weights = bytes(0x60000);
    params.fc1.bits = 3;
    params.fc2.bits = 3;
    params.rows = 8;
    params.top_k = 2;
    params.hidden = 256;
    params.intermediate = 128;
    params.block_size = 64;
    test_case.mutate(params);
    EXPECT_EQ(static_cast<int>(SelectInt3DecodeKernel(params)), static_cast<int>(test_case.expected))
        << test_case.name;
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
