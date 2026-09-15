/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_sqnbitgemm_2bit.cpp

Abstract:

    Unit tests for the block-group W2 path
    (sqnbitgemm_kernel_avx512_2bit.{h,cpp}).

    End-to-end pack + scalar GEMM correctness coverage These
    tests do NOT exercise any SIMD path -- they validate the layout, the
    pack 3-call sequence, and the scalar oracle kernel that will back the
    SIMD kernel.

    Tests deliberately use the same shapes as the production W2 tests so a
    side-by-side comparison is straightforward.

--*/

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "core/mlas/inc/mlas_qnbit.h"
#include "core/mlas/lib/qnbitgemm.h"
#include "core/mlas/lib/mlasi.h"
#include "core/mlas/lib/sqnbitgemm_kernel_avx512_2bit.h"
#include "core/mlas/lib/sqnbitgemm_kernel_avx2_2bit.h"

namespace {

namespace sq2 = onnxruntime::mlas::sq2bit_avx512;
namespace sq2a = onnxruntime::mlas::sq2bit_avx2;

constexpr size_t kBlkLen = sq2::kBlkLen;      // 64
constexpr size_t kBlkBytes = sq2::kBlkBytes;  // 16

// Standard ONNX 2-bit source packing (1 byte = 4 weights).
void PackSourceBlock_BlkLen64(const uint8_t weights[kBlkLen], std::byte* src_out) {
  for (size_t i = 0; i < kBlkBytes; ++i) {
    const uint8_t v0 = weights[4 * i + 0] & 0x03u;
    const uint8_t v1 = weights[4 * i + 1] & 0x03u;
    const uint8_t v2 = weights[4 * i + 2] & 0x03u;
    const uint8_t v3 = weights[4 * i + 3] & 0x03u;
    src_out[i] = static_cast<std::byte>(
        static_cast<uint8_t>(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)));
  }
}

// Bit-exact mirror of MlasQNBitGemm's per-block int8 quantizer (amax/127,
// round-half-to-even via std::nearbyint, scale_recip = 127/amax).
void QuantizeA_Reference(size_t M, size_t K, const float* A,
                         int8_t* QuantAData, float* QuantAScale) {
  const size_t BlockCountK = (K + kBlkLen - 1) / kBlkLen;
  for (size_t m = 0; m < M; ++m) {
    for (size_t k = 0, k_blk = 0; k < K; k += kBlkLen, ++k_blk) {
      const size_t local_len = std::min(K - k, kBlkLen);
      float amax = 0.0f;
      for (size_t kk = 0; kk < local_len; ++kk) {
        amax = std::max(amax, std::fabs(A[m * K + k + kk]));
      }
      constexpr float range_max = 127.0f;
      const float scale = amax / range_max;
      const float scale_recip = amax != 0.0f ? range_max / amax : 0.0f;
      QuantAScale[m * BlockCountK + k_blk] = scale;
      for (size_t kk = 0; kk < kBlkLen; ++kk) {
        const float a = (kk < local_len) ? A[m * K + k + kk] : 0.0f;
        const float q = std::nearbyint(a * scale_recip);
        QuantAData[m * BlockCountK * kBlkLen + k + kk] =
            static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
      }
    }
  }
}

//
// Integer-domain GEMM oracle: bit-exact match to the math the MLAS W2 path
// performs (kernel int8 GEMM + SGEMM zero-point correction collapsed into a
// single direct dot of (qa * (qb - zp))).
//
void ReferenceGemm_W2_CompInt8(size_t M, size_t N, size_t K,
                               const float* A,
                               const std::vector<uint8_t>& BWeights,  // [N * K] in [0, 3]
                               const float* QuantBScale,              // [N * BlockCountK]
                               const uint8_t* BZeroPoints,            // [N * BlockCountK] or nullptr
                               const float* Bias,
                               float* C) {
  const size_t BlockCountK = (K + kBlkLen - 1) / kBlkLen;
  std::vector<int8_t> QuantAData(M * BlockCountK * kBlkLen, int8_t{0});
  std::vector<float> QuantAScale(M * BlockCountK, 0.0f);
  QuantizeA_Reference(M, K, A, QuantAData.data(), QuantAScale.data());

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float acc = (Bias != nullptr) ? Bias[n] : 0.0f;
      for (size_t k = 0, blk = 0; k < K; k += kBlkLen, ++blk) {
        const size_t local_len = std::min(K - k, kBlkLen);
        const float a_scale = QuantAScale[m * BlockCountK + blk];
        const float b_scale = QuantBScale[n * BlockCountK + blk];
        const int32_t zp = BZeroPoints != nullptr
                               ? static_cast<int32_t>(BZeroPoints[n * BlockCountK + blk])
                               : static_cast<int32_t>(sq2::kDefaultSymmetricZeroPoint2Bit);
        int32_t dot = 0;
        for (size_t kk = 0; kk < local_len; ++kk) {
          const int8_t qa = QuantAData[m * BlockCountK * kBlkLen + k + kk];
          const int32_t qb =
              static_cast<int32_t>(BWeights[n * K + k + kk]) - zp;
          dot += static_cast<int32_t>(qa) * qb;
        }
        acc += static_cast<float>(dot) * a_scale * b_scale;
      }
      C[m * N + n] = acc;
    }
  }
}

//
// Pack per-block W2 zero points into the standard ONNX byte stream
// (4 zp per byte along K, row-major in N).
//
std::vector<std::byte>
PackW2ZeroPoints(size_t N, size_t BlockCountK, const std::vector<uint8_t>& BZeroPoints) {
  const size_t ZPCountK = (BlockCountK + 3) / 4;
  std::vector<std::byte> packed(N * ZPCountK, std::byte{0});
  for (size_t n = 0; n < N; ++n) {
    for (size_t blk = 0; blk < BlockCountK; ++blk) {
      const uint8_t zp = BZeroPoints[n * BlockCountK + blk] & 0x03u;
      const size_t byte_idx = n * ZPCountK + (blk / 4);
      const size_t bit_off = (blk % 4) * 2;
      packed[byte_idx] = static_cast<std::byte>(
          static_cast<uint8_t>(packed[byte_idx]) | (zp << bit_off));
    }
  }
  return packed;
}

//
// Test harness that drives the block-group path directly (without going through
// MlasQNBitGemmBatch). Builds the same packed buffer the dispatcher would
// construct, runs the chosen block-group kernel (scalar / AVX-512BW / VNNI),
// and compares to ReferenceGemm_W2_CompInt8.
//
// `KernelFn` matches the SQ4BitGemmKernel_BlkSum_CompInt8_Fn signature, which
// every block-group kernel variant honors via direct-call forwarders declared
// in sqnbitgemm_kernel_avx512_2bit.h.
//
using W2KernelFn = size_t(MLASCALL*)(
    size_t, const std::byte*, const float*, const std::byte*, const float*,
    const std::byte*, float*, size_t, size_t, size_t, size_t,
    const float*, size_t, const float*, const float*);

[[maybe_unused]] void RunW2Case(size_t M, size_t N, size_t K, bool WithBias, uint32_t seed,
                                bool WithZeroPoints, W2KernelFn kernel,
                                const char* kernel_name) {
  const size_t BlockCountK = (K + kBlkLen - 1) / kBlkLen;
  ASSERT_EQ(K % kBlkLen, 0u) << "Test K must be a multiple of BlkLen=64";
  // BlockCountK no longer required to be a multiple of kBlockGroupBlks --
  // the K-tail handler picks up the trailing 1-3 blocks.

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);
  std::uniform_int_distribution<uint32_t> w_dist(0, 3);
  std::uniform_real_distribution<float> s_dist(0.05f, 0.5f);

  std::vector<float> A(M * K);
  for (auto& v : A) v = a_dist(rng);

  std::vector<uint8_t> BWeights(N * K);
  for (auto& v : BWeights) v = static_cast<uint8_t>(w_dist(rng));

  // Source-packed B (standard ONNX layout) -- the input to the pack helper.
  std::vector<std::byte> QuantBData(N * BlockCountK * kBlkBytes, std::byte{0});
  for (size_t n = 0; n < N; ++n) {
    for (size_t blk = 0; blk < BlockCountK; ++blk) {
      uint8_t blk_weights[kBlkLen];
      for (size_t kk = 0; kk < kBlkLen; ++kk) {
        blk_weights[kk] = BWeights[n * K + blk * kBlkLen + kk];
      }
      PackSourceBlock_BlkLen64(blk_weights,
                               QuantBData.data() + (n * BlockCountK + blk) * kBlkBytes);
    }
  }

  std::vector<float> QuantBScale(N * BlockCountK);
  for (auto& v : QuantBScale) v = s_dist(rng);

  std::vector<uint8_t> BZeroPoints;
  std::vector<std::byte> BZeroPointsPacked;
  const uint8_t* BZeroPointsRef = nullptr;
  const std::byte* BZeroPointsMlas = nullptr;
  if (WithZeroPoints) {
    BZeroPoints.resize(N * BlockCountK);
    for (auto& v : BZeroPoints) v = static_cast<uint8_t>(w_dist(rng));
    BZeroPointsRef = BZeroPoints.data();
    BZeroPointsPacked = PackW2ZeroPoints(N, BlockCountK, BZeroPoints);
    BZeroPointsMlas = BZeroPointsPacked.data();
  }

  std::vector<float> Bias;
  const float* BiasPtr = nullptr;
  if (WithBias) {
    Bias.resize(N);
    for (auto& v : Bias) v = a_dist(rng);
    BiasPtr = Bias.data();
  }

  // Allocate the packed-B buffer (same total size as the production path).
  const size_t PackedSize = sq2::Q2BitGemmPackQuantBDataSize_Avx512(
      N, K, kBlkLen, WithZeroPoints, SQNBIT_CompInt8, nullptr);
  ASSERT_GT(PackedSize, 0u) << "block-group pack size unsupported for the chosen shape";

  std::vector<std::byte> PackedQuantBBuf(PackedSize, std::byte{0});
  // The W2 PackedQuantBDataStruct constructor pads BlockCountK to a multiple
  // of 4 internally (see qnbitgemm.h) so the slab layout matches what the
  // block-group pack helper writes regardless of whether the caller passes
  // the logical or padded BlockCountK. We pass the logical value to mirror
  // exactly what matmul_nbits.cc does in production.
  PackedQuantBDataStruct<float, 2> packed_b(
      PackedQuantBBuf.data(), N, BlockCountK, kBlkLen, /*QuantAUnsigned=*/false);

  // Mirror the matmul_nbits.cc prepack 3-call pattern (B, scales, ZP) so the
  // pack code path is exercised exactly as the production dispatcher would.
  sq2::SQ2BitGemmPackQuantBDataAndBlkSum_Scalar(
      N, K, kBlkLen, SQNBIT_CompInt8,
      QuantBData.data(), /*scales=*/nullptr,
      WithZeroPoints, /*zp=*/nullptr,
      packed_b, /*tp=*/nullptr, /*cfg=*/nullptr);
  sq2::SQ2BitGemmPackQuantBDataAndBlkSum_Scalar(
      N, K, kBlkLen, SQNBIT_CompInt8,
      /*B=*/nullptr, QuantBScale.data(),
      WithZeroPoints, /*zp=*/nullptr,
      packed_b, /*tp=*/nullptr, /*cfg=*/nullptr);
  if (WithZeroPoints) {
    sq2::SQ2BitGemmPackQuantBDataAndBlkSum_Scalar(
        N, K, kBlkLen, SQNBIT_CompInt8,
        /*B=*/nullptr, /*scales=*/nullptr,
        WithZeroPoints, BZeroPointsMlas,
        packed_b, /*tp=*/nullptr, /*cfg=*/nullptr);
  }

  // Quantize A the same way MLAS would (per-block amax/127, banker rounding).
  std::vector<int8_t> QuantAData(M * BlockCountK * kBlkLen, int8_t{0});
  std::vector<float> QuantAScale(M * BlockCountK, 0.0f);
  QuantizeA_Reference(M, K, A.data(), QuantAData.data(), QuantAScale.data());

  std::vector<float> ABlockSum(M * BlockCountK, 0.0f);
  for (size_t m = 0; m < M; ++m) {
    for (size_t blk = 0; blk < BlockCountK; ++blk) {
      int32_t sum = 0;
      for (size_t kk = 0; kk < kBlkLen; ++kk) {
        sum += static_cast<int32_t>(
            QuantAData[m * BlockCountK * kBlkLen + blk * kBlkLen + kk]);
      }
      ABlockSum[m * BlockCountK + blk] =
          QuantAScale[m * BlockCountK + blk] * static_cast<float>(sum);
    }
  }

  std::vector<float> C(M * N, 0.0f);
  kernel(
      kBlkLen,
      reinterpret_cast<const std::byte*>(QuantAData.data()),
      QuantAScale.data(),
      packed_b.PackedQuantBData,
      packed_b.PackedQuantBScale,
      /*QuantBZeroPoint=*/nullptr,
      C.data(),
      M, N, /*CountK=*/K, BlockCountK,
      BiasPtr,
      /*ldc=*/N,
      ABlockSum.data(),
      packed_b.QuantBBlkSum);

  std::vector<float> CRef(M * N, 0.0f);
  ReferenceGemm_W2_CompInt8(M, N, K, A.data(), BWeights, QuantBScale.data(),
                            BZeroPointsRef, BiasPtr, CRef.data());

  const float abs_tol = 1e-4f;
  const float rel_tol = 1e-4f;
  for (size_t i = 0; i < M * N; ++i) {
    const float diff = std::fabs(C[i] - CRef[i]);
    const float bound = abs_tol + rel_tol * std::fabs(CRef[i]);
    ASSERT_LE(diff, bound)
        << "block-group " << kernel_name << " mismatch at i=" << i
        << " (m=" << (i / N) << ", n=" << (i % N) << ")"
        << " out=" << C[i] << " ref=" << CRef[i]
        << " M=" << M << " N=" << N << " K=" << K
        << " WithBias=" << WithBias
        << " WithZeroPoints=" << WithZeroPoints;
  }
}

}  // namespace

#if defined(MLAS_TARGET_AMD64)

//
// Scalar block-group test, no zero-points. Covers the same small synthetic
// shapes + representative prefill sizes used by the production W2 tests. All
// shapes have K as a multiple of (kBlkLen * kBlockGroupBlks) = 256. K=384 is
// NOT a multiple of 256 so it's excluded; that shape will need a tail
// handler in a follow-up.
//
TEST(MlasSq2BitTest, Scalar_BlkLen64) {
  if (!GetMlasPlatform().Avx512Supported_) {
    GTEST_SKIP() << "AVX-512 not available on this host";
  }
  struct Shape {
    size_t M, N, K;
  };
  constexpr Shape shapes[] = {
      {1, 16, 256},
      {1, 32, 256},
      {1, 64, 512},
      {4, 16, 256},
      {4, 33, 256},
      {7, 17, 256},
      {16, 64, 512},
      {32, 128, 256},
      // Representative prefill (only the K values that are multiples of 256).
      {1, 1024, 1024},
      {1, 192, 1024},
      {1, 384, 1024},
      {1, 4096, 1024},
      {1, 1024, 4096},
      {128, 1024, 1024},
      {128, 192, 1024},
      {128, 384, 1024},
      {128, 4096, 1024},
      {128, 1024, 4096},
  };

  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const Shape& s : shapes) {
      for (bool bias : {false, true}) {
        RunW2Case(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                  /*WithZeroPoints=*/false,
                  sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Scalar,
                  "scalar");
      }
    }
  }
}

//
// Same coverage with per-block non-default zero points.
//
TEST(MlasSq2BitTest, Scalar_BlkLen64_WithZeroPoints) {
  if (!GetMlasPlatform().Avx512Supported_) {
    GTEST_SKIP() << "AVX-512 not available on this host";
  }
  struct Shape {
    size_t M, N, K;
  };
  constexpr Shape shapes[] = {
      {1, 16, 256},
      {1, 32, 256},
      {1, 64, 512},
      {4, 16, 256},
      {4, 33, 256},
      {7, 17, 256},
      {16, 64, 512},
      {32, 128, 256},
      {1, 1024, 1024},
      {1, 192, 1024},
      {1, 384, 1024},
      {1, 4096, 1024},
      {1, 1024, 4096},
      {128, 1024, 1024},
      {128, 192, 1024},
      {128, 384, 1024},
      {128, 4096, 1024},
      {128, 1024, 4096},
  };

  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const Shape& s : shapes) {
      for (bool bias : {false, true}) {
        RunW2Case(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                  /*WithZeroPoints=*/true,
                  sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Scalar,
                  "scalar");
      }
    }
  }
}

#endif  // defined(MLAS_TARGET_AMD64) -- kSimdShapes itself is cross-arch (also used by the ARM64 NEON DotProd W2 tests further down)

//
// SIMD block-group shape coverage. Phase-3+K-tail kernel requires:
//   * BlkLen == 64
//   * CountN a multiple of kNCols4 (=4)
// CountM and BlockCountK have NO alignment requirements:
//   - R2xC4 handles the M-aligned head; a single R1xC4 picks up the optional
//     trailing odd row. CountM == 1 dispatches directly to R1xC4.
//   - The K-loop iterates `BlockCountK / 4` full block-groups plus a partial
//     "tail group" of 1-3 trailing K-blocks. The pack helpers zero-pad the
//     trailing slots so they contribute 0 to the dot product; the tile loads
//     only valid A blocks (zero ZMM for missing ones) to avoid OOB.
//
// Representative prefill shapes (M in {1, 128}, N in {192, 384, 1024, 4096},
// K in
// {1024, 4096}) are all covered. M=3 and M=5 exercise the M-tail path.
//
// K-tail handler (BlockCountK not a multiple of kBlockGroupBlks=4): the
// pack helper zero-pads the trailing 1-3 K-block slots; the SIMD K-loop
// processes them via the 4-block accumulator with zero ZMM for the missing
// A blocks. GroupStride uses BlockCountKPadded so N-group advances land on
// the right packed-B address regardless of K % 4. K=384 and the
// synthetic K=320, K=448 shapes exercise this path.
//
[[maybe_unused]] constexpr struct {
  size_t M, N, K;
} kSimdShapes[] = {
    {1, 16, 256},     // R1 only
    {1, 192, 1024},   // R1 only, representative N
    {1, 1024, 4096},  // R1 only, representative N
    {2, 16, 256},
    {2, 32, 256},
    {2, 64, 512},
    {3, 16, 256},  // R2 head (1 pair) + R1 tail
    {3, 384, 1024},
    {4, 16, 256},
    {4, 32, 256},
    {5, 64, 512},  // R2 head (2 pairs) + R1 tail
    {16, 64, 512},
    {32, 128, 256},
    // Representative prefill (M=128) at all (K, N) pairs, including K=384.
    {128, 1024, 384},  // K-tail: BlockCountK=6, 1 full group + tail of 2 blocks
    {128, 1024, 1024},
    {128, 192, 1024},
    {128, 384, 1024},
    {128, 4096, 1024},
    {128, 1024, 4096},
    // Representative decode (M=1) at K=384 (the case the K%4 gate previously blocked).
    {1, 1024, 384},
    // Synthetic K-tail stress shapes covering all (TailBlocks in {1, 2, 3}).
    {2, 16, 320},  // tail=1
    {4, 16, 320},
    {128, 1024, 320},
    {2, 16, 448},  // tail=3
    {4, 16, 448},
    {128, 1024, 448},
    // N-tail stress (CountN % 4 != 0). The R2/R1 main tiles handle the
    // NMain = floor(CountN/4)*4 cols; the per-1-col tail tile picks up
    // the trailing 1-3 cols against the column-major tail region of the
    // packed buffer. NMain = 0 cases (N in {1,2,3}) exercise the tail
    // tile in isolation.
    {1, 1, 256},   // NMain=0, NTail=1, single-column decode
    {1, 3, 256},   // NMain=0, NTail=3
    {4, 3, 256},   // NMain=0, NTail=3, R2+R1 head still empty
    {1, 17, 256},  // NMain=16, NTail=1, decode
    {4, 17, 256},
    {128, 17, 256},
    {1, 33, 256},  // NMain=32, NTail=1
    {4, 33, 256},  // exact shape that failed the dispatch swap
    {128, 33, 256},
    {1, 18, 256},  // NMain=16, NTail=2
    {4, 18, 256},
    {128, 19, 256},  // NMain=16, NTail=3
    // N-tail combined with K-tail (the most generic case).
    {1, 17, 384},
    {4, 33, 384},
    {128, 19, 448},
    // R*xC4 tile coverage (CountN % 8 in [4..7]). The per-BlkLen NEON tile
    // dispatcher splits CountN into NMain8 + NMain4 + NTail; without these
    // shapes the NMain4 region (R1xC4 / R2xC4 tiles) is never exercised.
    // M={1,2,3} covers M-tail-only / M-pair-only / mixed M-pair+M-tail.
    {1, 4, 256},   // R1xC4 only
    {2, 4, 256},   // R2xC4 only
    {3, 4, 256},   // R2xC4 + R1xC4
    {1, 5, 256},   // R1xC4 + R1xC1
    {2, 5, 256},   // R2xC4 + R2xC1
    {3, 5, 256},   // R2xC4 + R2xC1 + R1xC4 + R1xC1
    {1, 12, 256},  // R1xC8 + R1xC4
    {2, 12, 256},  // R2xC8 + R2xC4
    {3, 12, 256},  // R2xC8 + R2xC4 + R1xC8 + R1xC4
    {1, 13, 256},  // R1xC8 + R1xC4 + R1xC1
    {3, 13, 256},  // ALL 6 tiles (R2xC8 + R2xC4 + R2xC1 + R1xC8 + R1xC4 + R1xC1)
    {3, 21, 384},  // ALL 6 tiles + K-tail
};

#if defined(MLAS_TARGET_AMD64)

//
// AVX-512BW (non-VNNI) SIMD block-group kernel.
//
TEST(MlasSq2BitTest, BlkLen64_Avx512) {
  if (!GetMlasPlatform().Avx512Supported_) {
    GTEST_SKIP() << "AVX-512BW (DQ/VL) not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes) {
      for (bool bias : {false, true}) {
        RunW2Case(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                  /*WithZeroPoints=*/false,
                  sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Avx512_Dispatch,
                  "AVX-512BW");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen64_Avx512_WithZeroPoints) {
  if (!GetMlasPlatform().Avx512Supported_) {
    GTEST_SKIP() << "AVX-512BW (DQ/VL) not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes) {
      for (bool bias : {false, true}) {
        RunW2Case(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                  /*WithZeroPoints=*/true,
                  sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Avx512_Dispatch,
                  "AVX-512BW");
      }
    }
  }
}

//
// AVX-512-VNNI SIMD block-group kernel. Gated on the platform having selected
// the VNNI dispatch table (the SIMD path uses `_mm512_dpbusd_epi32`).
//
TEST(MlasSq2BitTest, BlkLen64_Avx512Vnni) {
  if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx512vnni) {
    GTEST_SKIP() << "AVX-512-VNNI not selected as the active dispatch on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes) {
      for (bool bias : {false, true}) {
        RunW2Case(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                  /*WithZeroPoints=*/false,
                  sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Avx512Vnni_Dispatch,
                  "AVX-512-VNNI");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen64_Avx512Vnni_WithZeroPoints) {
  if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx512vnni) {
    GTEST_SKIP() << "AVX-512-VNNI not selected as the active dispatch on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes) {
      for (bool bias : {false, true}) {
        RunW2Case(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                  /*WithZeroPoints=*/true,
                  sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Avx512Vnni_Dispatch,
                  "AVX-512-VNNI");
      }
    }
  }
}

#endif  // defined(MLAS_TARGET_AMD64)

// =============================================================================
// BlkLen=128 coverage. Mirrors the BlkLen=64 tests above. The helpers are
// duplicated rather than templated to keep the BlkLen=64 path bit-identical;
// the diff is purely additive.
//
// The helper namespace is intentionally cross-arch (not under MLAS_TARGET_AMD64)
// so the ARM64 NEON DotProd tests in this file can reuse RunW2Case_BlkLen128.
// =============================================================================

namespace {

constexpr size_t kBlkLen128 = sq2::kBlkLen128;      // 128
constexpr size_t kBlkBytes128 = sq2::kBlkBytes128;  // 32

void PackSourceBlock_BlkLen128(const uint8_t weights[kBlkLen128], std::byte* src_out) {
  for (size_t i = 0; i < kBlkBytes128; ++i) {
    const uint8_t v0 = weights[4 * i + 0] & 0x03u;
    const uint8_t v1 = weights[4 * i + 1] & 0x03u;
    const uint8_t v2 = weights[4 * i + 2] & 0x03u;
    const uint8_t v3 = weights[4 * i + 3] & 0x03u;
    src_out[i] = static_cast<std::byte>(
        static_cast<uint8_t>(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)));
  }
}

void QuantizeA_Reference_BlkLen128(size_t M, size_t K, const float* A,
                                   int8_t* QuantAData, float* QuantAScale) {
  const size_t BlockCountK = (K + kBlkLen128 - 1) / kBlkLen128;
  for (size_t m = 0; m < M; ++m) {
    for (size_t k = 0, k_blk = 0; k < K; k += kBlkLen128, ++k_blk) {
      const size_t local_len = std::min(K - k, kBlkLen128);
      float amax = 0.0f;
      for (size_t kk = 0; kk < local_len; ++kk) {
        amax = std::max(amax, std::fabs(A[m * K + k + kk]));
      }
      constexpr float range_max = 127.0f;
      const float scale = amax / range_max;
      const float scale_recip = amax != 0.0f ? range_max / amax : 0.0f;
      QuantAScale[m * BlockCountK + k_blk] = scale;
      for (size_t kk = 0; kk < kBlkLen128; ++kk) {
        const float a = (kk < local_len) ? A[m * K + k + kk] : 0.0f;
        const float q = std::nearbyint(a * scale_recip);
        QuantAData[m * BlockCountK * kBlkLen128 + k + kk] =
            static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
      }
    }
  }
}

void ReferenceGemm_W2_CompInt8_BlkLen128(size_t M, size_t N, size_t K,
                                         const float* A,
                                         const std::vector<uint8_t>& BWeights,
                                         const float* QuantBScale,
                                         const uint8_t* BZeroPoints,
                                         const float* Bias,
                                         float* C) {
  const size_t BlockCountK = (K + kBlkLen128 - 1) / kBlkLen128;
  std::vector<int8_t> QuantAData(M * BlockCountK * kBlkLen128, int8_t{0});
  std::vector<float> QuantAScale(M * BlockCountK, 0.0f);
  QuantizeA_Reference_BlkLen128(M, K, A, QuantAData.data(), QuantAScale.data());

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float acc = (Bias != nullptr) ? Bias[n] : 0.0f;
      for (size_t k = 0, blk = 0; k < K; k += kBlkLen128, ++blk) {
        const size_t local_len = std::min(K - k, kBlkLen128);
        const float a_scale = QuantAScale[m * BlockCountK + blk];
        const float b_scale = QuantBScale[n * BlockCountK + blk];
        const int32_t zp = BZeroPoints != nullptr
                               ? static_cast<int32_t>(BZeroPoints[n * BlockCountK + blk])
                               : static_cast<int32_t>(sq2::kDefaultSymmetricZeroPoint2Bit);
        int32_t dot = 0;
        for (size_t kk = 0; kk < local_len; ++kk) {
          const int8_t qa = QuantAData[m * BlockCountK * kBlkLen128 + k + kk];
          const int32_t qb =
              static_cast<int32_t>(BWeights[n * K + k + kk]) - zp;
          dot += static_cast<int32_t>(qa) * qb;
        }
        acc += static_cast<float>(dot) * a_scale * b_scale;
      }
      C[m * N + n] = acc;
    }
  }
}

[[maybe_unused]] void RunW2Case_BlkLen128(size_t M, size_t N, size_t K, bool WithBias, uint32_t seed,
                                          bool WithZeroPoints, W2KernelFn kernel,
                                          const char* kernel_name) {
  const size_t BlockCountK = (K + kBlkLen128 - 1) / kBlkLen128;
  ASSERT_EQ(K % kBlkLen128, 0u) << "BlkLen128 test K must be a multiple of 128";

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);
  std::uniform_int_distribution<uint32_t> w_dist(0, 3);
  std::uniform_real_distribution<float> s_dist(0.05f, 0.5f);

  std::vector<float> A(M * K);
  for (auto& v : A) v = a_dist(rng);

  std::vector<uint8_t> BWeights(N * K);
  for (auto& v : BWeights) v = static_cast<uint8_t>(w_dist(rng));

  // Source-packed B in standard ONNX layout (32 bytes per block at BlkLen=128).
  std::vector<std::byte> QuantBData(N * BlockCountK * kBlkBytes128, std::byte{0});
  for (size_t n = 0; n < N; ++n) {
    for (size_t blk = 0; blk < BlockCountK; ++blk) {
      uint8_t blk_weights[kBlkLen128];
      for (size_t kk = 0; kk < kBlkLen128; ++kk) {
        blk_weights[kk] = BWeights[n * K + blk * kBlkLen128 + kk];
      }
      PackSourceBlock_BlkLen128(blk_weights,
                                QuantBData.data() + (n * BlockCountK + blk) * kBlkBytes128);
    }
  }

  std::vector<float> QuantBScale(N * BlockCountK);
  for (auto& v : QuantBScale) v = s_dist(rng);

  std::vector<uint8_t> BZeroPoints;
  std::vector<std::byte> BZeroPointsPacked;
  const uint8_t* BZeroPointsRef = nullptr;
  const std::byte* BZeroPointsMlas = nullptr;
  if (WithZeroPoints) {
    BZeroPoints.resize(N * BlockCountK);
    for (auto& v : BZeroPoints) v = static_cast<uint8_t>(w_dist(rng));
    BZeroPointsRef = BZeroPoints.data();
    BZeroPointsPacked = PackW2ZeroPoints(N, BlockCountK, BZeroPoints);
    BZeroPointsMlas = BZeroPointsPacked.data();
  }

  std::vector<float> Bias;
  const float* BiasPtr = nullptr;
  if (WithBias) {
    Bias.resize(N);
    for (auto& v : Bias) v = a_dist(rng);
    BiasPtr = Bias.data();
  }

  const size_t PackedSize = sq2::Q2BitGemmPackQuantBDataSize_Avx512(
      N, K, kBlkLen128, WithZeroPoints, SQNBIT_CompInt8, nullptr);
  ASSERT_GT(PackedSize, 0u) << "BlkLen128 block-group pack size unsupported for shape";

  std::vector<std::byte> PackedQuantBBuf(PackedSize, std::byte{0});
  PackedQuantBDataStruct<float, 2> packed_b(
      PackedQuantBBuf.data(), N, BlockCountK, kBlkLen128, /*QuantAUnsigned=*/false);

  sq2::SQ2BitGemmPackQuantBDataAndBlkSum_Scalar(
      N, K, kBlkLen128, SQNBIT_CompInt8,
      QuantBData.data(), /*scales=*/nullptr,
      WithZeroPoints, /*zp=*/nullptr,
      packed_b, /*tp=*/nullptr, /*cfg=*/nullptr);
  sq2::SQ2BitGemmPackQuantBDataAndBlkSum_Scalar(
      N, K, kBlkLen128, SQNBIT_CompInt8,
      /*B=*/nullptr, QuantBScale.data(),
      WithZeroPoints, /*zp=*/nullptr,
      packed_b, /*tp=*/nullptr, /*cfg=*/nullptr);
  if (WithZeroPoints) {
    sq2::SQ2BitGemmPackQuantBDataAndBlkSum_Scalar(
        N, K, kBlkLen128, SQNBIT_CompInt8,
        /*B=*/nullptr, /*scales=*/nullptr,
        WithZeroPoints, BZeroPointsMlas,
        packed_b, /*tp=*/nullptr, /*cfg=*/nullptr);
  }

  std::vector<int8_t> QuantAData(M * BlockCountK * kBlkLen128, int8_t{0});
  std::vector<float> QuantAScale(M * BlockCountK, 0.0f);
  QuantizeA_Reference_BlkLen128(M, K, A.data(), QuantAData.data(), QuantAScale.data());

  std::vector<float> ABlockSum(M * BlockCountK, 0.0f);
  for (size_t m = 0; m < M; ++m) {
    for (size_t blk = 0; blk < BlockCountK; ++blk) {
      int32_t sum = 0;
      for (size_t kk = 0; kk < kBlkLen128; ++kk) {
        sum += static_cast<int32_t>(
            QuantAData[m * BlockCountK * kBlkLen128 + blk * kBlkLen128 + kk]);
      }
      ABlockSum[m * BlockCountK + blk] =
          QuantAScale[m * BlockCountK + blk] * static_cast<float>(sum);
    }
  }

  std::vector<float> C(M * N, 0.0f);
  kernel(
      kBlkLen128,
      reinterpret_cast<const std::byte*>(QuantAData.data()),
      QuantAScale.data(),
      packed_b.PackedQuantBData,
      packed_b.PackedQuantBScale,
      /*QuantBZeroPoint=*/nullptr,
      C.data(),
      M, N, /*CountK=*/K, BlockCountK,
      BiasPtr,
      /*ldc=*/N,
      ABlockSum.data(),
      packed_b.QuantBBlkSum);

  std::vector<float> CRef(M * N, 0.0f);
  ReferenceGemm_W2_CompInt8_BlkLen128(M, N, K, A.data(), BWeights, QuantBScale.data(),
                                      BZeroPointsRef, BiasPtr, CRef.data());

  const float abs_tol = 1e-4f;
  const float rel_tol = 1e-4f;
  for (size_t i = 0; i < M * N; ++i) {
    const float diff = std::fabs(C[i] - CRef[i]);
    const float bound = abs_tol + rel_tol * std::fabs(CRef[i]);
    ASSERT_LE(diff, bound)
        << "BlkLen128 " << kernel_name << " mismatch at i=" << i
        << " (m=" << (i / N) << ", n=" << (i % N) << ")"
        << " out=" << C[i] << " ref=" << CRef[i]
        << " M=" << M << " N=" << N << " K=" << K
        << " WithBias=" << WithBias
        << " WithZeroPoints=" << WithZeroPoints;
  }
}

//
// Shape set for BlkLen=128. K must be a multiple of 128 (BlkLen=128 constraint).
// Covers the same regimes as the BlkLen=64 shape set:
//   * R1 / R2 tiles, M=1 decode + larger M prefill
//   * BlockCountK in {1, 2, 3, 4, 8, 16, 32} -- full + K-tail variants
//   * N-tail (NMain=0 and various NTail) combined with K-tail
//
[[maybe_unused]] constexpr struct {
  size_t M, N, K;
} kSimdShapes_BlkLen128[] = {
    {1, 16, 128},     // R1, BlockCountK=1
    {1, 32, 256},     // R1, BlockCountK=2 (K-tail, no full group)
    {1, 1024, 1024},  // R1, BlockCountK=8
    {1, 1024, 4096},  // R1, BlockCountK=32 (representative N)
    // K < BlkLen coverage lives at the operator layer
    // (MatMul2Bits.Float32_2b_BlkLen128_Accuracy4), which goes through the
    // quantizer's K-pad-up path. This direct-kernel runner intentionally
    // enforces K % BlkLen == 0.
    {2, 16, 128},
    {2, 32, 256},
    {2, 64, 512},  // BlockCountK=4 (one full group)
    {3, 16, 256},  // R2 head + R1 tail
    {3, 384, 1024},
    {4, 16, 256},
    {4, 32, 256},
    {16, 64, 512},
    {32, 128, 256},
    // M=128 prefill at representative shapes (K multiples of 128)
    {128, 1024, 1024},
    {128, 1024, 4096},
    {128, 192, 1024},
    {128, 384, 1024},
    // K-tail stress (BlockCountK not a multiple of 4)
    {2, 16, 384},  // tail=3
    {4, 16, 384},
    {128, 1024, 384},  // tail=3 at representative M
    {2, 16, 640},      // tail=1
    {4, 16, 640},
    {2, 16, 768},  // tail=2
    {4, 16, 768},
    // N-tail stress
    {1, 1, 256},
    {1, 3, 256},
    {4, 3, 256},
    {1, 17, 256},
    {4, 17, 256},
    {128, 17, 256},
    {1, 33, 256},
    {4, 33, 256},
    {128, 33, 256},
    {1, 18, 256},
    {4, 18, 256},
    {128, 19, 256},
    // N-tail combined with K-tail (most generic)
    {1, 17, 384},
    {4, 33, 384},
    {128, 19, 640},
    // R*xC4 tile coverage (CountN % 8 in [4..7]). The per-BlkLen NEON tile
    // dispatcher splits CountN into NMain8 + NMain4 + NTail; without these
    // shapes the NMain4 region (R1xC4 / R2xC4 tiles) is never exercised.
    // M={1,2,3} covers M-tail-only / M-pair-only / mixed M-pair+M-tail.
    {1, 4, 256},   // R1xC4 only
    {2, 4, 256},   // R2xC4 only
    {3, 4, 256},   // R2xC4 + R1xC4
    {1, 5, 256},   // R1xC4 + R1xC1
    {2, 5, 256},   // R2xC4 + R2xC1
    {3, 5, 256},   // R2xC4 + R2xC1 + R1xC4 + R1xC1
    {1, 12, 256},  // R1xC8 + R1xC4
    {2, 12, 256},  // R2xC8 + R2xC4
    {3, 12, 256},  // R2xC8 + R2xC4 + R1xC8 + R1xC4
    {1, 13, 256},  // R1xC8 + R1xC4 + R1xC1
    {3, 13, 256},  // ALL 6 tiles (R2xC8 + R2xC4 + R2xC1 + R1xC8 + R1xC4 + R1xC1)
    {3, 21, 384},  // ALL 6 tiles + K-tail
};

}  // namespace

#if defined(MLAS_TARGET_AMD64)

TEST(MlasSq2BitTest, Scalar_BlkLen128) {
  if (!GetMlasPlatform().Avx512Supported_) {
    GTEST_SKIP() << "AVX-512 not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen128) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen128(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                            /*WithZeroPoints=*/false,
                            sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Scalar,
                            "Scalar");
      }
    }
  }
}

TEST(MlasSq2BitTest, Scalar_BlkLen128_WithZeroPoints) {
  if (!GetMlasPlatform().Avx512Supported_) {
    GTEST_SKIP() << "AVX-512 not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen128) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen128(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                            /*WithZeroPoints=*/true,
                            sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Scalar,
                            "Scalar");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen128_Avx512) {
  if (!GetMlasPlatform().Avx512Supported_) {
    GTEST_SKIP() << "AVX-512BW (DQ/VL) not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen128) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen128(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                            /*WithZeroPoints=*/false,
                            sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Avx512_Dispatch,
                            "AVX-512BW");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen128_Avx512_WithZeroPoints) {
  if (!GetMlasPlatform().Avx512Supported_) {
    GTEST_SKIP() << "AVX-512BW (DQ/VL) not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen128) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen128(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                            /*WithZeroPoints=*/true,
                            sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Avx512_Dispatch,
                            "AVX-512BW");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen128_Avx512Vnni) {
  if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx512vnni) {
    GTEST_SKIP() << "AVX-512-VNNI not selected as the active dispatch on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen128) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen128(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                            /*WithZeroPoints=*/false,
                            sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Avx512Vnni_Dispatch,
                            "AVX-512-VNNI");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen128_Avx512Vnni_WithZeroPoints) {
  if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx512vnni) {
    GTEST_SKIP() << "AVX-512-VNNI not selected as the active dispatch on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen128) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen128(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                            /*WithZeroPoints=*/true,
                            sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Avx512Vnni_Dispatch,
                            "AVX-512-VNNI");
      }
    }
  }
}

#endif  // defined(MLAS_TARGET_AMD64)

// =============================================================================
// BlkLen=32 coverage. Mirrors BlkLen=128 above with per-block byte width 8
// and per-group bytes 32. Helpers duplicated (rather than templated) to keep
// the BlkLen=64 hot path bit-identical.
//
// The helper namespace is intentionally cross-arch (not under MLAS_TARGET_AMD64)
// so the ARM64 NEON DotProd tests in this file can reuse RunW2Case_BlkLen32.
// =============================================================================

namespace {

constexpr size_t kBlkLen32 = sq2::kBlkLen32;      // 32
constexpr size_t kBlkBytes32 = sq2::kBlkBytes32;  // 8

void PackSourceBlock_BlkLen32(const uint8_t weights[kBlkLen32], std::byte* src_out) {
  for (size_t i = 0; i < kBlkBytes32; ++i) {
    const uint8_t v0 = weights[4 * i + 0] & 0x03u;
    const uint8_t v1 = weights[4 * i + 1] & 0x03u;
    const uint8_t v2 = weights[4 * i + 2] & 0x03u;
    const uint8_t v3 = weights[4 * i + 3] & 0x03u;
    src_out[i] = static_cast<std::byte>(
        static_cast<uint8_t>(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)));
  }
}

void QuantizeA_Reference_BlkLen32(size_t M, size_t K, const float* A,
                                  int8_t* QuantAData, float* QuantAScale) {
  const size_t BlockCountK = (K + kBlkLen32 - 1) / kBlkLen32;
  for (size_t m = 0; m < M; ++m) {
    for (size_t k = 0, k_blk = 0; k < K; k += kBlkLen32, ++k_blk) {
      const size_t local_len = std::min(K - k, kBlkLen32);
      float amax = 0.0f;
      for (size_t kk = 0; kk < local_len; ++kk) {
        amax = std::max(amax, std::fabs(A[m * K + k + kk]));
      }
      constexpr float range_max = 127.0f;
      const float scale = amax / range_max;
      const float scale_recip = amax != 0.0f ? range_max / amax : 0.0f;
      QuantAScale[m * BlockCountK + k_blk] = scale;
      for (size_t kk = 0; kk < kBlkLen32; ++kk) {
        const float a = (kk < local_len) ? A[m * K + k + kk] : 0.0f;
        const float q = std::nearbyint(a * scale_recip);
        QuantAData[m * BlockCountK * kBlkLen32 + k + kk] =
            static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
      }
    }
  }
}

void ReferenceGemm_W2_CompInt8_BlkLen32(size_t M, size_t N, size_t K,
                                        const float* A,
                                        const std::vector<uint8_t>& BWeights,
                                        const float* QuantBScale,
                                        const uint8_t* BZeroPoints,
                                        const float* Bias,
                                        float* C) {
  const size_t BlockCountK = (K + kBlkLen32 - 1) / kBlkLen32;
  std::vector<int8_t> QuantAData(M * BlockCountK * kBlkLen32, int8_t{0});
  std::vector<float> QuantAScale(M * BlockCountK, 0.0f);
  QuantizeA_Reference_BlkLen32(M, K, A, QuantAData.data(), QuantAScale.data());

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float acc = (Bias != nullptr) ? Bias[n] : 0.0f;
      for (size_t k = 0, blk = 0; k < K; k += kBlkLen32, ++blk) {
        const size_t local_len = std::min(K - k, kBlkLen32);
        const float a_scale = QuantAScale[m * BlockCountK + blk];
        const float b_scale = QuantBScale[n * BlockCountK + blk];
        const int32_t zp = BZeroPoints != nullptr
                               ? static_cast<int32_t>(BZeroPoints[n * BlockCountK + blk])
                               : static_cast<int32_t>(sq2::kDefaultSymmetricZeroPoint2Bit);
        int32_t dot = 0;
        for (size_t kk = 0; kk < local_len; ++kk) {
          const int8_t qa = QuantAData[m * BlockCountK * kBlkLen32 + k + kk];
          const int32_t qb =
              static_cast<int32_t>(BWeights[n * K + k + kk]) - zp;
          dot += static_cast<int32_t>(qa) * qb;
        }
        acc += static_cast<float>(dot) * a_scale * b_scale;
      }
      C[m * N + n] = acc;
    }
  }
}

[[maybe_unused]] void RunW2Case_BlkLen32(size_t M, size_t N, size_t K, bool WithBias, uint32_t seed,
                                         bool WithZeroPoints, W2KernelFn kernel,
                                         const char* kernel_name) {
  const size_t BlockCountK = (K + kBlkLen32 - 1) / kBlkLen32;
  ASSERT_EQ(K % kBlkLen32, 0u) << "BlkLen32 test K must be a multiple of 32";

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);
  std::uniform_int_distribution<uint32_t> w_dist(0, 3);
  std::uniform_real_distribution<float> s_dist(0.05f, 0.5f);

  std::vector<float> A(M * K);
  for (auto& v : A) v = a_dist(rng);

  std::vector<uint8_t> BWeights(N * K);
  for (auto& v : BWeights) v = static_cast<uint8_t>(w_dist(rng));

  std::vector<std::byte> QuantBData(N * BlockCountK * kBlkBytes32, std::byte{0});
  for (size_t n = 0; n < N; ++n) {
    for (size_t blk = 0; blk < BlockCountK; ++blk) {
      uint8_t blk_weights[kBlkLen32];
      for (size_t kk = 0; kk < kBlkLen32; ++kk) {
        blk_weights[kk] = BWeights[n * K + blk * kBlkLen32 + kk];
      }
      PackSourceBlock_BlkLen32(blk_weights,
                               QuantBData.data() + (n * BlockCountK + blk) * kBlkBytes32);
    }
  }

  std::vector<float> QuantBScale(N * BlockCountK);
  for (auto& v : QuantBScale) v = s_dist(rng);

  std::vector<uint8_t> BZeroPoints;
  std::vector<std::byte> BZeroPointsPacked;
  const uint8_t* BZeroPointsRef = nullptr;
  const std::byte* BZeroPointsMlas = nullptr;
  if (WithZeroPoints) {
    BZeroPoints.resize(N * BlockCountK);
    for (auto& v : BZeroPoints) v = static_cast<uint8_t>(w_dist(rng));
    BZeroPointsRef = BZeroPoints.data();
    BZeroPointsPacked = PackW2ZeroPoints(N, BlockCountK, BZeroPoints);
    BZeroPointsMlas = BZeroPointsPacked.data();
  }

  std::vector<float> Bias;
  const float* BiasPtr = nullptr;
  if (WithBias) {
    Bias.resize(N);
    for (auto& v : Bias) v = a_dist(rng);
    BiasPtr = Bias.data();
  }

  const size_t PackedSize = sq2::Q2BitGemmPackQuantBDataSize_Avx512(
      N, K, kBlkLen32, WithZeroPoints, SQNBIT_CompInt8, nullptr);
  ASSERT_GT(PackedSize, 0u) << "BlkLen32 pack size unsupported for shape";

  std::vector<std::byte> PackedQuantBBuf(PackedSize, std::byte{0});
  PackedQuantBDataStruct<float, 2> packed_b(
      PackedQuantBBuf.data(), N, BlockCountK, kBlkLen32, /*QuantAUnsigned=*/false);

  sq2::SQ2BitGemmPackQuantBDataAndBlkSum_Scalar(
      N, K, kBlkLen32, SQNBIT_CompInt8,
      QuantBData.data(), /*scales=*/nullptr,
      WithZeroPoints, /*zp=*/nullptr,
      packed_b, /*tp=*/nullptr, /*cfg=*/nullptr);
  sq2::SQ2BitGemmPackQuantBDataAndBlkSum_Scalar(
      N, K, kBlkLen32, SQNBIT_CompInt8,
      /*B=*/nullptr, QuantBScale.data(),
      WithZeroPoints, /*zp=*/nullptr,
      packed_b, /*tp=*/nullptr, /*cfg=*/nullptr);
  if (WithZeroPoints) {
    sq2::SQ2BitGemmPackQuantBDataAndBlkSum_Scalar(
        N, K, kBlkLen32, SQNBIT_CompInt8,
        /*B=*/nullptr, /*scales=*/nullptr,
        WithZeroPoints, BZeroPointsMlas,
        packed_b, /*tp=*/nullptr, /*cfg=*/nullptr);
  }

  std::vector<int8_t> QuantAData(M * BlockCountK * kBlkLen32, int8_t{0});
  std::vector<float> QuantAScale(M * BlockCountK, 0.0f);
  QuantizeA_Reference_BlkLen32(M, K, A.data(), QuantAData.data(), QuantAScale.data());

  std::vector<float> ABlockSum(M * BlockCountK, 0.0f);
  for (size_t m = 0; m < M; ++m) {
    for (size_t blk = 0; blk < BlockCountK; ++blk) {
      int32_t sum = 0;
      for (size_t kk = 0; kk < kBlkLen32; ++kk) {
        sum += static_cast<int32_t>(
            QuantAData[m * BlockCountK * kBlkLen32 + blk * kBlkLen32 + kk]);
      }
      ABlockSum[m * BlockCountK + blk] =
          QuantAScale[m * BlockCountK + blk] * static_cast<float>(sum);
    }
  }

  std::vector<float> C(M * N, 0.0f);
  kernel(
      kBlkLen32,
      reinterpret_cast<const std::byte*>(QuantAData.data()),
      QuantAScale.data(),
      packed_b.PackedQuantBData,
      packed_b.PackedQuantBScale,
      /*QuantBZeroPoint=*/nullptr,
      C.data(),
      M, N, /*CountK=*/K, BlockCountK,
      BiasPtr,
      /*ldc=*/N,
      ABlockSum.data(),
      packed_b.QuantBBlkSum);

  std::vector<float> CRef(M * N, 0.0f);
  ReferenceGemm_W2_CompInt8_BlkLen32(M, N, K, A.data(), BWeights, QuantBScale.data(),
                                     BZeroPointsRef, BiasPtr, CRef.data());

  const float abs_tol = 1e-4f;
  const float rel_tol = 1e-4f;
  for (size_t i = 0; i < M * N; ++i) {
    const float diff = std::fabs(C[i] - CRef[i]);
    const float bound = abs_tol + rel_tol * std::fabs(CRef[i]);
    ASSERT_LE(diff, bound)
        << "BlkLen32 " << kernel_name << " mismatch at i=" << i
        << " (m=" << (i / N) << ", n=" << (i % N) << ")"
        << " out=" << C[i] << " ref=" << CRef[i]
        << " M=" << M << " N=" << N << " K=" << K
        << " WithBias=" << WithBias
        << " WithZeroPoints=" << WithZeroPoints;
  }
}

// K-shape constraint: K multiple of 32 (BlkLen=32). Covers BlockCountK in
// {1, 2, 3, 4, 8, 16, 32, 64} -- both K-tail variants (BlockCountK not a
// multiple of 4) and exact block-group multiples.
[[maybe_unused]] constexpr struct {
  size_t M, N, K;
} kSimdShapes_BlkLen32[] = {
    {1, 16, 32},      // R1, BlockCountK=1
    {1, 32, 64},      // R1, BlockCountK=2 (K-tail, no full group)
    {1, 1024, 256},   // R1, BlockCountK=8
    {1, 1024, 1024},  // R1, BlockCountK=32
    {2, 16, 32},
    {2, 32, 64},
    {2, 64, 128},  // BlockCountK=4 (one full group)
    {3, 16, 64},   // R2 head + R1 tail
    {3, 384, 256},
    {4, 16, 64},
    {4, 32, 64},
    {16, 64, 128},
    {32, 128, 128},
    // M=128 prefill
    {128, 1024, 256},
    {128, 1024, 1024},
    {128, 192, 256},
    {128, 384, 256},
    // K-tail (BlockCountK not multiple of 4)
    {2, 16, 96},  // tail=3
    {4, 16, 96},
    {128, 1024, 96},
    {2, 16, 160},  // tail=1
    {4, 16, 160},
    {2, 16, 192},  // tail=2 (BlockCountK=6)
    {4, 16, 192},
    // N-tail
    {1, 1, 64},
    {1, 3, 64},
    {4, 3, 64},
    {1, 17, 64},
    {4, 17, 64},
    {128, 17, 64},
    {1, 33, 64},
    {4, 33, 64},
    {128, 33, 64},
    {1, 18, 64},
    {4, 18, 64},
    {128, 19, 64},
    // N-tail + K-tail
    {1, 17, 96},
    {4, 33, 96},
    {128, 19, 160},
    // R*xC4 tile coverage (CountN % 8 in [4..7]). The per-BlkLen NEON tile
    // dispatcher splits CountN into NMain8 + NMain4 + NTail; without these
    // shapes the NMain4 region (R1xC4 / R2xC4 tiles) is never exercised.
    // M={1,2,3} covers M-tail-only / M-pair-only / mixed M-pair+M-tail.
    {1, 4, 128},   // R1xC4 only
    {2, 4, 128},   // R2xC4 only
    {3, 4, 128},   // R2xC4 + R1xC4
    {1, 5, 128},   // R1xC4 + R1xC1
    {2, 5, 128},   // R2xC4 + R2xC1
    {3, 5, 128},   // R2xC4 + R2xC1 + R1xC4 + R1xC1
    {1, 12, 128},  // R1xC8 + R1xC4
    {2, 12, 128},  // R2xC8 + R2xC4
    {3, 12, 128},  // R2xC8 + R2xC4 + R1xC8 + R1xC4
    {1, 13, 128},  // R1xC8 + R1xC4 + R1xC1
    {3, 13, 128},  // ALL 6 tiles (R2xC8 + R2xC4 + R2xC1 + R1xC8 + R1xC4 + R1xC1)
    {3, 21, 160},  // ALL 6 tiles + K-tail
};

}  // namespace

#if defined(MLAS_TARGET_AMD64)

TEST(MlasSq2BitTest, Scalar_BlkLen32) {
  if (!GetMlasPlatform().Avx512Supported_) {
    GTEST_SKIP() << "AVX-512 not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen32) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen32(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                           /*WithZeroPoints=*/false,
                           sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Scalar,
                           "Scalar");
      }
    }
  }
}

TEST(MlasSq2BitTest, Scalar_BlkLen32_WithZeroPoints) {
  if (!GetMlasPlatform().Avx512Supported_) {
    GTEST_SKIP() << "AVX-512 not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen32) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen32(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                           /*WithZeroPoints=*/true,
                           sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Scalar,
                           "Scalar");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen32_Avx512) {
  if (!GetMlasPlatform().Avx512Supported_) {
    GTEST_SKIP() << "AVX-512BW (DQ/VL) not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen32) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen32(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                           /*WithZeroPoints=*/false,
                           sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Avx512_Dispatch,
                           "AVX-512BW");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen32_Avx512_WithZeroPoints) {
  if (!GetMlasPlatform().Avx512Supported_) {
    GTEST_SKIP() << "AVX-512BW (DQ/VL) not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen32) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen32(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                           /*WithZeroPoints=*/true,
                           sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Avx512_Dispatch,
                           "AVX-512BW");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen32_Avx512Vnni) {
  if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx512vnni) {
    GTEST_SKIP() << "AVX-512-VNNI not selected as the active dispatch on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen32) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen32(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                           /*WithZeroPoints=*/false,
                           sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Avx512Vnni_Dispatch,
                           "AVX-512-VNNI");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen32_Avx512Vnni_WithZeroPoints) {
  if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx512vnni) {
    GTEST_SKIP() << "AVX-512-VNNI not selected as the active dispatch on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen32) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen32(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                           /*WithZeroPoints=*/true,
                           sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Avx512Vnni_Dispatch,
                           "AVX-512-VNNI");
      }
    }
  }
}

#endif  // defined(MLAS_TARGET_AMD64)

#if defined(MLAS_TARGET_ARM64)

#include "core/mlas/lib/qnbitgemm_kernel_neon.h"

//
// W2 NEON DotProd kernel direct-call tests. Mirror the AVX-512 layout
// above: BlkLen 64 / 128 / 32, each with a no-zero-points and a
// with-zero-points variant, all driven from the shared kSimdShapes*
// shape tables and the cross-arch RunW2Case* helpers (which build the
// packed B and quantized A in the layout the kernel expects).
//
// End-to-end coverage through MlasQNBitGemmBatch is provided by the
// MatMulNBits operator tests; this file only verifies the inner kernel.
//
// Skipped on ARM64 hosts that lack FEAT_DotProd -- the NEON DotProd
// kernel emits SDOT instructions and would SIGILL on a pre-armv8.2 core.
//

//
// NEON DotProd SIMD block-group kernel, BlkLen=64.
//
TEST(MlasSq2BitTest, BlkLen64_NeonDotProd) {
  if (!MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeonDot()) {
    GTEST_SKIP() << "ARM NEON FEAT_DotProd not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes) {
      for (bool bias : {false, true}) {
        RunW2Case(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                  /*WithZeroPoints=*/false,
                  sqnbitgemm_neon::SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd,
                  "NEON-DotProd");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen64_NeonDotProd_WithZeroPoints) {
  if (!MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeonDot()) {
    GTEST_SKIP() << "ARM NEON FEAT_DotProd not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes) {
      for (bool bias : {false, true}) {
        RunW2Case(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                  /*WithZeroPoints=*/true,
                  sqnbitgemm_neon::SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd,
                  "NEON-DotProd");
      }
    }
  }
}

//
// NEON DotProd SIMD block-group kernel, BlkLen=128.
//
TEST(MlasSq2BitTest, BlkLen128_NeonDotProd) {
  if (!MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeonDot()) {
    GTEST_SKIP() << "ARM NEON FEAT_DotProd not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen128) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen128(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                            /*WithZeroPoints=*/false,
                            sqnbitgemm_neon::SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd,
                            "NEON-DotProd");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen128_NeonDotProd_WithZeroPoints) {
  if (!MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeonDot()) {
    GTEST_SKIP() << "ARM NEON FEAT_DotProd not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen128) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen128(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                            /*WithZeroPoints=*/true,
                            sqnbitgemm_neon::SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd,
                            "NEON-DotProd");
      }
    }
  }
}

//
// NEON DotProd SIMD block-group kernel, BlkLen=32.
//
TEST(MlasSq2BitTest, BlkLen32_NeonDotProd) {
  if (!MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeonDot()) {
    GTEST_SKIP() << "ARM NEON FEAT_DotProd not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen32) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen32(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                           /*WithZeroPoints=*/false,
                           sqnbitgemm_neon::SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd,
                           "NEON-DotProd");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen32_NeonDotProd_WithZeroPoints) {
  if (!MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeonDot()) {
    GTEST_SKIP() << "ARM NEON FEAT_DotProd not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen32) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen32(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                           /*WithZeroPoints=*/true,
                           sqnbitgemm_neon::SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd,
                           "NEON-DotProd");
      }
    }
  }
}

#endif  // defined(MLAS_TARGET_ARM64)

//
// Availability contract test for W2 + SQNBIT_CompInt8.
//
// The W2 native kernel only implements BlkLen ∈ {32, 64, 128}.
// MlasIsQNBitGemmAvailable must report this truthfully so direct MLAS callers
// can rely on it as the support contract. Previously the variant gate also
// admitted BlkLen 16 and 256 (since they're valid for W4 / W8), which made
// availability return true for shapes that Q2BitGemmPackQuantBDataSize_Avx512
// would refuse to size (returning 0).
//
// The contract is platform-agnostic: it must hold wherever a W2 dispatch is
// installed (AVX-512 on x86_64, NEON+DotProd or i8mm on ARM64). We skip on
// hosts where no W2 dispatch is wired in (e.g. pure-NEON-only ARM, or x86
// without AVX-512) -- there is nothing to assert about a feature that is
// uniformly unavailable.
//
TEST(MlasSq2BitTest, AvailabilityContract_BlkLens) {
  // Probe a representative supported BlkLen to detect whether ANY W2 dispatch
  // is installed on this host. This mirrors the dispatch-pointer-identity
  // guard pattern that the W4/W8 tests use.
  if (!MlasIsQNBitGemmAvailable(2, 64, SQNBIT_CompInt8)) {
    GTEST_SKIP() << "No W2 native dispatch on this host";
  }

  // Supported BlkLens.
  EXPECT_TRUE(MlasIsQNBitGemmAvailable(2, 32, SQNBIT_CompInt8));
  EXPECT_TRUE(MlasIsQNBitGemmAvailable(2, 64, SQNBIT_CompInt8));
  EXPECT_TRUE(MlasIsQNBitGemmAvailable(2, 128, SQNBIT_CompInt8));

  // Unsupported BlkLens for W2 (valid for W4/W8 but not implemented for W2).
  EXPECT_FALSE(MlasIsQNBitGemmAvailable(2, 16, SQNBIT_CompInt8));
  EXPECT_FALSE(MlasIsQNBitGemmAvailable(2, 256, SQNBIT_CompInt8));

  // Compute types not implemented for W2.
  EXPECT_FALSE(MlasIsQNBitGemmAvailable(2, 64, SQNBIT_CompFp32));
  EXPECT_FALSE(MlasIsQNBitGemmAvailable(2, 64, HQNBIT_CompFp16));
}

#if defined(MLAS_TARGET_AMD64)

// =============================================================================
// AVX2 / AVX2-VNNI W2 coverage. Reuses the same RunW2Case* harnesses and
// kSimdShapes* tables as the AVX-512 tests above. The non-VNNI tests guard on
// Avx2Supported_, so they also run on AVX-512 hosts and exercise the
// vpmaddubsw + vpmaddwd fallback there. The VNNI tests guard on the AVX2-VNNI
// dispatch being the active one (the SIMD path uses _mm256_dpbusds_avx_epi32).
// =============================================================================

TEST(MlasSq2BitTest, BlkLen64_Avx2) {
  if (!GetMlasPlatform().Avx2Supported_) {
    GTEST_SKIP() << "AVX2 not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes) {
      for (bool bias : {false, true}) {
        RunW2Case(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                  /*WithZeroPoints=*/false,
                  sq2a::SQ2BitGemmKernel_BlkSum_CompInt8_Avx2_Dispatch, "AVX2");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen64_Avx2_WithZeroPoints) {
  if (!GetMlasPlatform().Avx2Supported_) {
    GTEST_SKIP() << "AVX2 not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes) {
      for (bool bias : {false, true}) {
        RunW2Case(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                  /*WithZeroPoints=*/true,
                  sq2a::SQ2BitGemmKernel_BlkSum_CompInt8_Avx2_Dispatch, "AVX2");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen64_Avx2Vnni) {
  if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx2vnni) {
    GTEST_SKIP() << "AVX2-VNNI not selected as the active dispatch on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes) {
      for (bool bias : {false, true}) {
        RunW2Case(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                  /*WithZeroPoints=*/false,
                  sq2a::SQ2BitGemmKernel_BlkSum_CompInt8_Avx2Vnni_Dispatch, "AVX2-VNNI");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen64_Avx2Vnni_WithZeroPoints) {
  if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx2vnni) {
    GTEST_SKIP() << "AVX2-VNNI not selected as the active dispatch on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes) {
      for (bool bias : {false, true}) {
        RunW2Case(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                  /*WithZeroPoints=*/true,
                  sq2a::SQ2BitGemmKernel_BlkSum_CompInt8_Avx2Vnni_Dispatch, "AVX2-VNNI");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen128_Avx2) {
  if (!GetMlasPlatform().Avx2Supported_) {
    GTEST_SKIP() << "AVX2 not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen128) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen128(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                            /*WithZeroPoints=*/false,
                            sq2a::SQ2BitGemmKernel_BlkSum_CompInt8_Avx2_Dispatch, "AVX2");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen128_Avx2_WithZeroPoints) {
  if (!GetMlasPlatform().Avx2Supported_) {
    GTEST_SKIP() << "AVX2 not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen128) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen128(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                            /*WithZeroPoints=*/true,
                            sq2a::SQ2BitGemmKernel_BlkSum_CompInt8_Avx2_Dispatch, "AVX2");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen128_Avx2Vnni) {
  if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx2vnni) {
    GTEST_SKIP() << "AVX2-VNNI not selected as the active dispatch on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen128) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen128(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                            /*WithZeroPoints=*/false,
                            sq2a::SQ2BitGemmKernel_BlkSum_CompInt8_Avx2Vnni_Dispatch, "AVX2-VNNI");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen128_Avx2Vnni_WithZeroPoints) {
  if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx2vnni) {
    GTEST_SKIP() << "AVX2-VNNI not selected as the active dispatch on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen128) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen128(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                            /*WithZeroPoints=*/true,
                            sq2a::SQ2BitGemmKernel_BlkSum_CompInt8_Avx2Vnni_Dispatch, "AVX2-VNNI");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen32_Avx2) {
  if (!GetMlasPlatform().Avx2Supported_) {
    GTEST_SKIP() << "AVX2 not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen32) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen32(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                           /*WithZeroPoints=*/false,
                           sq2a::SQ2BitGemmKernel_BlkSum_CompInt8_Avx2_Dispatch, "AVX2");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen32_Avx2_WithZeroPoints) {
  if (!GetMlasPlatform().Avx2Supported_) {
    GTEST_SKIP() << "AVX2 not available on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen32) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen32(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                           /*WithZeroPoints=*/true,
                           sq2a::SQ2BitGemmKernel_BlkSum_CompInt8_Avx2_Dispatch, "AVX2");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen32_Avx2Vnni) {
  if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx2vnni) {
    GTEST_SKIP() << "AVX2-VNNI not selected as the active dispatch on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen32) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen32(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                           /*WithZeroPoints=*/false,
                           sq2a::SQ2BitGemmKernel_BlkSum_CompInt8_Avx2Vnni_Dispatch, "AVX2-VNNI");
      }
    }
  }
}

TEST(MlasSq2BitTest, BlkLen32_Avx2Vnni_WithZeroPoints) {
  if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx2vnni) {
    GTEST_SKIP() << "AVX2-VNNI not selected as the active dispatch on this host";
  }
  for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
    for (const auto& s : kSimdShapes_BlkLen32) {
      for (bool bias : {false, true}) {
        RunW2Case_BlkLen32(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                           /*WithZeroPoints=*/true,
                           sq2a::SQ2BitGemmKernel_BlkSum_CompInt8_Avx2Vnni_Dispatch, "AVX2-VNNI");
      }
    }
  }
}

#endif  // defined(MLAS_TARGET_AMD64)
