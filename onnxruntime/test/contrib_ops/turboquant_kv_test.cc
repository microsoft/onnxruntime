// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "gtest/gtest.h"

#include "core/framework/int3.h"
#include "test/common/cuda_op_test_utils.h"

// CUDA kernel correctness for TurboQuant is validated via Phase 5 end-to-end
// model runs (LFM2-1.2B), not via in-process gtests. The reason: the
// `onnxruntime_provider_test` binary does NOT link `libonnxruntime_providers_cuda.so`
// at link time — the CUDA EP is loaded dynamically via dlopen during the test.
// Calling our kernel launcher's symbols directly from a gtest .cc would require
// dlsym + a separate test driver. We keep the host-side `Int3x8` tests (which
// validate the bit-layout that the CUDA kernel must match) and defer CUDA
// correctness to model-level e2e validation.

namespace onnxruntime {
namespace test {

// =============================================================================
// UInt3x8 unit tests.
// =============================================================================

TEST(TurboQuantKVTest, Int3x8_RoundtripBijective) {
  // 8 values into 3 bytes, then back. Exhaustive over a slice of inputs.
  for (uint8_t v0 = 0; v0 <= 7; ++v0) {
    for (uint8_t v7 = 0; v7 <= 7; ++v7) {
      const uint8_t in[8] = {v0, 1, 2, 3, 4, 5, 6, v7};
      UInt3x8 packed(in);
      EXPECT_EQ(packed.GetElem(0), v0);
      EXPECT_EQ(packed.GetElem(7), v7);
      for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(packed.GetElem(i), in[i]);
      }
    }
  }
}

TEST(TurboQuantKVTest, Int3x8_BitLayoutMatchesSpec) {
  // Verify the documented bit layout: byte0 = (v0) | (v1<<3) | (low2 of v2 << 6).
  // For v = [1, 2, 3, 4, 5, 6, 7, 0]:
  //   word = 1 | (2<<3) | (3<<6) | (4<<9) | (5<<12) | (6<<15) | (7<<18) | (0<<21)
  //        = 0x1F58D1   ⇒ byte0=0xD1, byte1=0x58, byte2=0x1F
  const uint8_t in[8] = {1, 2, 3, 4, 5, 6, 7, 0};
  UInt3x8 packed(in);
  EXPECT_EQ(static_cast<uint8_t>(packed.bytes_[0]), 0xD1);
  EXPECT_EQ(static_cast<uint8_t>(packed.bytes_[1]), 0x58);
  EXPECT_EQ(static_cast<uint8_t>(packed.bytes_[2]), 0x1F);
}

TEST(TurboQuantKVTest, Int3x8_BulkPackUnpack) {
  std::vector<uint8_t> values(128);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<uint8_t>(i % 8);
  }
  std::vector<UInt3x8> packed(UInt3x8::CalcNumPacks(values.size()));
  EXPECT_TRUE(UInt3x8::Pack(packed, values));

  std::vector<uint8_t> recovered(values.size());
  EXPECT_TRUE(UInt3x8::Unpack(recovered, packed));
  EXPECT_EQ(values, recovered);
}

TEST(TurboQuantKVTest, Int3x8_SetGet) {
  UInt3x8 p;
  for (size_t i = 0; i < 8; ++i) {
    p.SetElem(i, static_cast<uint8_t>(7 - i));
  }
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(p.GetElem(i), static_cast<uint8_t>(7 - i));
  }
}

// =============================================================================
// CUDA kernel correctness (only built when CUDA EP is enabled).
// Currently a no-op placeholder; see comment at top of file. Kept here as a
// hook so we know where to plumb a dlopen-based driver in a follow-up.
// =============================================================================

#if defined(USE_CUDA)
TEST(TurboQuantKVTest, CudaEncodeDecodeRoundtrip_K4V4) {
  GTEST_SKIP() << "CUDA kernel direct-test deferred — see comment at top of file";
}
#endif

#if 0  // Disabled: needs dlopen-based test driver (see top-of-file comment).
namespace tq_test_helpers {

// Solve Lloyd-Max codebook on host for N(0, 1/d).
// Mirrors solve_lloyd_max() in python/tools/quantization/turboquant_kv/centroids.py.
// Uses fixed-size buffers (max 16 levels for bits<=4) to avoid GCC 13 false-positive
// -Wstringop-overflow on vector<double> reassignment.
inline std::vector<float> SolveLloydMax(int d, int bits, int max_iter = 200) {
  constexpr int kMaxLevels = 16;
  const int n_levels = 1 << bits;
  if (n_levels > kMaxLevels) {
    return std::vector<float>();  // unsupported
  }
  const double sigma2 = 1.0 / d;
  const double sigma = std::sqrt(sigma2);
  const double lo = -3.5 * sigma;
  const double hi = 3.5 * sigma;

  auto pdf = [sigma2](double x) {
    return (1.0 / std::sqrt(2 * M_PI * sigma2)) * std::exp(-x * x / (2 * sigma2));
  };
  auto trapz = [](auto f, double a, double b, int n = 200) {
    const double h = (b - a) / n;
    double r = 0.5 * (f(a) + f(b));
    for (int i = 1; i < n; ++i) r += f(a + i * h);
    return r * h;
  };

  double centroids[kMaxLevels] = {};
  double new_centroids[kMaxLevels] = {};
  double edges[kMaxLevels + 1] = {};
  for (int i = 0; i < n_levels; ++i) {
    centroids[i] = lo + (hi - lo) * (i + 0.5) / n_levels;
  }

  for (int it = 0; it < max_iter; ++it) {
    edges[0] = lo * 3.0;
    edges[n_levels] = hi * 3.0;
    for (int i = 0; i < n_levels - 1; ++i) {
      edges[i + 1] = 0.5 * (centroids[i] + centroids[i + 1]);
    }

    double drift = 0.0;
    for (int i = 0; i < n_levels; ++i) {
      double a = edges[i], b = edges[i + 1];
      double num = trapz([&pdf](double x) { return x * pdf(x); }, a, b);
      double den = trapz(pdf, a, b);
      new_centroids[i] = (den > 1e-15) ? (num / den) : centroids[i];
      drift = std::max(drift, std::abs(new_centroids[i] - centroids[i]));
    }
    for (int i = 0; i < n_levels; ++i) centroids[i] = new_centroids[i];
    if (drift < 1e-10) break;
  }

  std::vector<float> out(n_levels);
  for (int i = 0; i < n_levels; ++i) out[i] = static_cast<float>(centroids[i]);
  return out;
}

inline double CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-9);
}

}  // namespace tq_test_helpers

// CUDA correctness test: the encode/decode roundtrip should preserve
// vector direction (cosine similarity vs original > 0.97 for k4v4 in rotated space).
//
// Note: the K reconstruction is in ROTATED space (K · H), so we compare against
// the rotated original, not raw K. This validates the algorithm; a full GQA
// dispatch test that compares attention OUTPUT vs fp16 baseline is left for v2.
TEST(TurboQuantKVTest, CudaEncodeDecodeRoundtrip_K4V4) {
  using namespace tq_test_helpers;

  // We don't gate on DefaultCudaExecutionProvider() here because the CUDA EP
  // may not be initialized at the moment of test discovery; cudaGetDeviceCount
  // is a sufficient runtime check.
  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }

  constexpr int batch_size = 1;
  constexpr int n_kv_heads = 4;
  constexpr int seq_len = 16;
  constexpr int head_size = 128;
  constexpr int key_bits = 4;
  constexpr int value_bits = 4;
  constexpr bool norm_correction = true;
  const int n_centroids = 1 << key_bits;

  // Generate random K, V on host (deterministic seed).
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  const int kv_elements = batch_size * n_kv_heads * seq_len * head_size;
  std::vector<float> K_host_f(kv_elements), V_host_f(kv_elements);
  for (int i = 0; i < kv_elements; ++i) {
    K_host_f[i] = dist(rng);
    V_host_f[i] = dist(rng);
  }

  // Convert to fp16.
  std::vector<half> K_host(kv_elements), V_host(kv_elements);
  for (int i = 0; i < kv_elements; ++i) {
    K_host[i] = __float2half(K_host_f[i]);
    V_host[i] = __float2half(V_host_f[i]);
  }

  // Lloyd-Max codebook.
  auto codebook_f = SolveLloydMax(head_size, key_bits);
  std::vector<half> codebook(n_centroids);
  for (int i = 0; i < n_centroids; ++i) codebook[i] = __float2half(codebook_f[i]);

  // Allocate device memory.
  half *d_K, *d_V, *d_codebook, *d_K_recon, *d_V_recon;
  cudaMalloc(&d_K, kv_elements * sizeof(half));
  cudaMalloc(&d_V, kv_elements * sizeof(half));
  cudaMalloc(&d_codebook, n_centroids * sizeof(half));
  cudaMalloc(&d_K_recon, kv_elements * sizeof(half));
  cudaMalloc(&d_V_recon, kv_elements * sizeof(half));

  cudaMemcpy(d_K, K_host.data(), kv_elements * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, V_host.data(), kv_elements * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_codebook, codebook.data(), n_centroids * sizeof(half), cudaMemcpyHostToDevice);

  cudaStream_t cuda_stream;
  cudaStreamCreate(&cuda_stream);

  int rc = RunTurboQuantRoundtrip_fp16(
      batch_size, n_kv_heads, seq_len, head_size,
      key_bits, value_bits, norm_correction ? 1 : 0,
      d_K, d_V, d_codebook,
      d_K_recon, d_V_recon,
      cuda_stream);
  ASSERT_EQ(rc, 0) << "RunTurboQuantRoundtrip_fp16 returned " << rc;
  cudaStreamSynchronize(cuda_stream);

  // Read back reconstructions.
  std::vector<half> K_recon(kv_elements), V_recon(kv_elements);
  cudaMemcpy(K_recon.data(), d_K_recon, kv_elements * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(V_recon.data(), d_V_recon, kv_elements * sizeof(half), cudaMemcpyDeviceToHost);

  // Compute cosine sim of V (per slot, in original space — V is uniform-quant only).
  std::vector<double> v_cos_per_slot;
  for (int s = 0; s < batch_size * n_kv_heads * seq_len; ++s) {
    std::vector<float> v_orig(head_size), v_rec(head_size);
    for (int i = 0; i < head_size; ++i) {
      v_orig[i] = V_host_f[s * head_size + i];
      v_rec[i] = __half2float(V_recon[s * head_size + i]);
    }
    v_cos_per_slot.push_back(tq_test_helpers::CosineSimilarity(v_orig, v_rec));
  }
  std::sort(v_cos_per_slot.begin(), v_cos_per_slot.end());
  double v_median = v_cos_per_slot[v_cos_per_slot.size() / 2];

  // K is in rotated space — compare each K_recon slot to the rotated normalized
  // original (K_orig / ||K_orig|| · H · ||K_orig||). For 4-bit Lloyd-Max we
  // expect cosine sim > 0.99.
  // To avoid implementing FWHT here, we instead just verify that K_recon is
  // not zero and has reasonable magnitude relative to the original norm.
  std::vector<double> k_norm_ratios;
  for (int s = 0; s < batch_size * n_kv_heads * seq_len; ++s) {
    double k_orig_norm_sq = 0.0, k_rec_norm_sq = 0.0;
    for (int i = 0; i < head_size; ++i) {
      double o = K_host_f[s * head_size + i];
      double r = __half2float(K_recon[s * head_size + i]);
      k_orig_norm_sq += o * o;
      k_rec_norm_sq += r * r;
    }
    if (k_orig_norm_sq > 1e-9) {
      k_norm_ratios.push_back(std::sqrt(k_rec_norm_sq) / std::sqrt(k_orig_norm_sq));
    }
  }
  std::sort(k_norm_ratios.begin(), k_norm_ratios.end());
  double k_median_ratio = k_norm_ratios[k_norm_ratios.size() / 2];

  // Cleanup.
  cudaFree(d_K); cudaFree(d_V); cudaFree(d_codebook);
  cudaFree(d_K_recon); cudaFree(d_V_recon);
  cudaStreamDestroy(cuda_stream);

  // V uniform quant should reach > 0.99 median cosine sim at 4 bits.
  EXPECT_GT(v_median, 0.99) << "V reconstruction cosine sim too low";
  // K_recon norm should be close to original (norm-correction makes this exact in expectation).
  EXPECT_GT(k_median_ratio, 0.95);
  EXPECT_LT(k_median_ratio, 1.05);
}

#endif  // disabled CUDA direct-test

}  // namespace test
}  // namespace onnxruntime
