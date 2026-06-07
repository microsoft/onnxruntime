/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_sqnbitgemm_2bit_gemm.cpp

Abstract:

    Numerical correctness tests for the 2-bit weight CompInt8 GEMM path on
    AVX-512 hosts. Covers three execution routes:

      1) MlasQNBitGemmBatch (public API) -> platform-selected dispatch ->
         AVX-512-VNNI W2 kernel. This is what production callers hit on a
         VNNI host.

      2) AVX-512-VNNI W2 kernel via direct test-entry forwarder, bypassing
         the platform dispatcher. Same kernel as (1); validates the
         forwarder mechanism used by (3).

      3) AVX-512BW (non-VNNI) W2 kernel via direct test-entry forwarder.
         Validates the kernel a non-VNNI AVX-512 host would normally run.
         On a VNNI host the platform dispatcher never picks this path, so
         the direct-call route is the only way to exercise it.

    All three are compared against a single bit-exact integer-domain
    reference (`ReferenceGemm_W2_CompInt8`) that reproduces the same per-
    block int8 A quantization MLAS uses internally (amax/127, symmetric)
    and the same dequant-free integer dot product against raw 2-bit
    weights with an implicit zero-point of 2.

--*/

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "core/mlas/inc/mlas_qnbit.h"
#include "core/mlas/lib/sqnbitgemm_kernel_avx512_2bit.h"
#include "core/mlas/lib/qnbitgemm.h"  // for PackedQuantBDataStruct (test direct-call path)
#include "core/mlas/lib/mlasi.h"     // for GetMlasPlatform().Avx512Supported_

namespace {

namespace sq2 = onnxruntime::mlas::sq2bit_avx512;

constexpr size_t kBlkLen = sq2::kBlkLen;            // 64
constexpr size_t kBlkBytes = sq2::kBlkBytes;        // 16 source bytes per block
constexpr size_t kBlkBitWidth = 2;
constexpr MLAS_QNBIT_GEMM_COMPUTE_TYPE kComputeType = SQNBIT_CompInt8;

// Standard ONNX 2-bit packing: byte_i = w[4i] | w[4i+1]<<2 | w[4i+2]<<4 | w[4i+3]<<6.
inline void
PackSourceBlock_BlkLen64(const uint8_t weights[kBlkLen], std::byte* src_out)
{
    for (size_t i = 0; i < kBlkBytes; ++i) {
        const uint8_t v0 = weights[4 * i + 0] & 0x03u;
        const uint8_t v1 = weights[4 * i + 1] & 0x03u;
        const uint8_t v2 = weights[4 * i + 2] & 0x03u;
        const uint8_t v3 = weights[4 * i + 3] & 0x03u;
        src_out[i] = static_cast<std::byte>(
            static_cast<uint8_t>(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6))
        );
    }
}

//
// Mirror the MLAS per-row int8 block quantizer used by the CompInt8 path:
// per-block symmetric scale = amax / 127, round-to-nearest, clamp to [-127, 127].
//
void
QuantizeA_Reference(size_t M,
                    size_t K,
                    const float* A,
                    int8_t* QuantAData,
                    float* QuantAScale)
{
    const size_t BlockCountK = (K + kBlkLen - 1) / kBlkLen;
    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0, k_blk = 0; k < K; k += kBlkLen, ++k_blk) {
            const size_t local_len = std::min(K - k, kBlkLen);

            float amax = 0.0f;
            for (size_t kk = 0; kk < local_len; ++kk) {
                amax = std::max(amax, std::fabs(A[m * K + k + kk]));
            }

            constexpr float range_max = static_cast<float>((1 << 7) - 1);
            const float scale = amax / range_max;
            // Match MLAS's QuantizeARow_CompInt8_avx512 bit-for-bit: it computes
            // `inverse_scale = 127 / amax` directly from amax (single op), not
            // `1 / (amax/127)` (two ops). The two are equal in real math but
            // differ by 1 ULP in float, which is enough to flip the rounded
            // int8 by ±1 for A values that land right on a half-integer.
            const float scale_recip = amax != 0.0f ? range_max / amax : 0.0f;

            QuantAScale[m * BlockCountK + k_blk] = scale;

            for (size_t kk = 0; kk < kBlkLen; ++kk) {
                const float a = (kk < local_len) ? A[m * K + k + kk] : 0.0f;
                // `std::nearbyint` uses the current floating-point rounding mode
                // (default round-half-to-even) and matches what MLAS's
                // `_mm512_roundscale_ps(v, _MM_ROUND_NEAREST)` does at .5
                // boundaries. `std::round` would diverge (rounds half away from
                // zero), introducing systematic mismatches in the public-API path
                // where MLAS quantises A and the reference computes ABlockSum
                // from its own quantization.
                const float q = std::nearbyint(a * scale_recip);
                QuantAData[m * BlockCountK * kBlkLen + k + kk] =
                    static_cast<int8_t>(
                        std::clamp(q,
                                   static_cast<float>(std::numeric_limits<int8_t>::min()),
                                   static_cast<float>(std::numeric_limits<int8_t>::max())));
            }
        }
    }
}

//
// Reference GEMM that exactly mirrors the math performed by the MLAS W2
// CompInt8 path:
//
//   C[m,n] = bias[n]
//          + sum_blk( scale_a[m,blk] * scale_b[n,blk]
//                     * dot(qa[m,blk,:], (qb[n,blk,:] - zp[n,blk])) )
//
// When BZeroPoints is null the symmetric default ZP = 2 is used for every
// block (matches the kernel's behavior when no zero-point tensor is supplied).
// When non-null, BZeroPoints[n * BlockCountK + blk] gives the per-block ZP in
// [0, 3].
//
void
ReferenceGemm_W2_CompInt8(size_t M,
                          size_t N,
                          size_t K,
                          const float* A,
                          const std::vector<uint8_t>& BWeights,  // [N * K] in [0,3]
                          const float* QuantBScale,
                          const uint8_t* BZeroPoints,            // [N * BlockCountK] in [0,3] or nullptr
                          const float* Bias,
                          float* C)
{
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
                    const int32_t qb = static_cast<int32_t>(BWeights[n * K + k + kk]) - zp;
                    dot += static_cast<int32_t>(qa) * qb;
                }
                acc += static_cast<float>(dot) * a_scale * b_scale;
            }
            C[m * N + n] = acc;
        }
    }
}

//
// Pack per-block 2-bit zero points into the standard ONNX MatMulNBits W2
// layout: 4 ZPs per byte along K, row-major in N. Row stride is
// ceil(BlockCountK / 4) bytes.
//
inline std::vector<std::byte>
PackW2ZeroPoints(size_t N, size_t BlockCountK, const std::vector<uint8_t>& BZeroPoints)
{
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

class MlasSQ2BitGemmTest {
 public:
    static void Run(size_t M, size_t N, size_t K, bool WithBias, uint32_t seed,
                    bool WithZeroPoints = false)
    {
        const size_t BlockCountK = (K + kBlkLen - 1) / kBlkLen;
        ASSERT_EQ(K % kBlkLen, 0u) << "Test K must be a multiple of BlkLen=64";

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);
        std::uniform_int_distribution<uint32_t> w_dist(0, 3);
        std::uniform_real_distribution<float> s_dist(0.05f, 0.5f);

        std::vector<float> A(M * K);
        for (auto& v : A) v = a_dist(rng);

        // Raw weights in [0,3], natural [n, k] order; the test owns this oracle
        // copy and is the source of truth for the reference math.
        std::vector<uint8_t> BWeights(N * K);
        for (auto& v : BWeights) v = static_cast<uint8_t>(w_dist(rng));

        // Source-packed B in the layout that MlasQNBitGemmPackQuantBData consumes:
        // column-major in N, kBlkBytes per block, standard ONNX 4-weights-per-byte.
        std::vector<std::byte> QuantBData(N * BlockCountK * kBlkBytes, std::byte{0});
        for (size_t n = 0; n < N; ++n) {
            for (size_t blk = 0; blk < BlockCountK; ++blk) {
                uint8_t blk_weights[kBlkLen];
                for (size_t kk = 0; kk < kBlkLen; ++kk) {
                    blk_weights[kk] = BWeights[n * K + blk * kBlkLen + kk];
                }
                PackSourceBlock_BlkLen64(
                    blk_weights,
                    QuantBData.data() + (n * BlockCountK + blk) * kBlkBytes);
            }
        }

        std::vector<float> QuantBScale(N * BlockCountK);
        for (auto& v : QuantBScale) v = s_dist(rng);

        // Per-block zero points (W2: each ZP in [0,3]). The reference path
        // uses the raw [N * BlockCountK] uint8 vector; MLAS gets the standard
        // ONNX-packed byte stream produced by PackW2ZeroPoints.
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

        // Pack B through the public API.
        const size_t PackedSize = MlasQNBitGemmPackQuantBDataSize(
            N, K, kBlkBitWidth, kBlkLen, WithZeroPoints, kComputeType, nullptr);
        ASSERT_GT(PackedSize, 0u);
        std::vector<std::byte> PackedQuantB(PackedSize, std::byte{0});

        // Mirror the matmul_nbits.cc prepack flow on AMD64: three separate
        // calls, one per input (B data, scales, zero_points). Each call passes
        // only its own input and nullptr for the others. The pack function
        // must update the BlkSum buffer on the zero_points call by reading
        // scales from the already-packed buffer.
        MlasQNBitGemmPackQuantBData(
            N, K, kBlkBitWidth, kBlkLen, kComputeType,
            QuantBData.data(), PackedQuantB.data(),
            /*QuantBScale=*/nullptr, WithZeroPoints, /*QuantBZeroPoint=*/nullptr,
            nullptr, nullptr);
        MlasQNBitGemmPackQuantBData(
            N, K, kBlkBitWidth, kBlkLen, kComputeType,
            /*QuantBData=*/nullptr, PackedQuantB.data(),
            QuantBScale.data(), WithZeroPoints, /*QuantBZeroPoint=*/nullptr,
            nullptr, nullptr);
        if (WithZeroPoints) {
            MlasQNBitGemmPackQuantBData(
                N, K, kBlkBitWidth, kBlkLen, kComputeType,
                /*QuantBData=*/nullptr, PackedQuantB.data(),
                /*QuantBScale=*/nullptr, WithZeroPoints, BZeroPointsMlas,
                nullptr, nullptr);
        }

        const size_t WorkspaceSize = MlasQNBitGemmBatchWorkspaceSize(
            M, N, K, 1, kBlkBitWidth, kBlkLen, WithZeroPoints, kComputeType, nullptr);
        std::vector<std::byte> Workspace(std::max<size_t>(WorkspaceSize, 1), std::byte{0});

        std::vector<float> C(M * N, 0.0f);

        MLAS_QNBIT_GEMM_DATA_PARAMS<float> params{};
        params.A = A.data();
        params.lda = K;
        params.QuantBDataWorkspace = PackedQuantB.data();
        params.PackedQuantBData = PackedQuantB.data();
        params.QuantBScale = QuantBScale.data();
        params.QuantBZeroPoint = BZeroPointsMlas;
        params.Bias = BiasPtr;
        params.C = C.data();
        params.ldc = N;
        params.PostProcessor = nullptr;

        MlasQNBitGemmBatch(M, N, K, 1, kBlkBitWidth, kBlkLen, kComputeType,
                           &params, Workspace.data(), nullptr, nullptr);

        std::vector<float> CRef(M * N, 0.0f);
        ReferenceGemm_W2_CompInt8(M, N, K, A.data(), BWeights, QuantBScale.data(),
                                  BZeroPointsRef, BiasPtr, CRef.data());

        // Both paths perform the identical integer-domain dot product followed
        // by the same float multiply-add chain, so the result should agree to
        // a small relative tolerance driven only by float accumulation order.
        const float abs_tol = 1e-4f;
        const float rel_tol = 1e-4f;
        for (size_t i = 0; i < M * N; ++i) {
            const float diff = std::fabs(C[i] - CRef[i]);
            const float bound = abs_tol + rel_tol * std::fabs(CRef[i]);
            ASSERT_LE(diff, bound)
                << "Mismatch at i=" << i
                << " (m=" << (i / N) << ", n=" << (i % N) << ")"
                << " MLAS=" << C[i] << " Ref=" << CRef[i]
                << " M=" << M << " N=" << N << " K=" << K
                << " WithBias=" << WithBias
                << " WithZeroPoints=" << WithZeroPoints;
        }
    }
};

}  // namespace

//
// Public-API correctness test. On a VNNI host the platform dispatcher
// resolves to the AVX-512-VNNI W2 kernel. Skips if no W2 path is available.
//
TEST(MlasSq2BitTest, GemmCompInt8_BlkLen64_PublicApi)
{
    if (!MlasIsQNBitGemmAvailable(kBlkBitWidth, kBlkLen, kComputeType)) {
        GTEST_SKIP() << "MlasQNBitGemm W2/BlkLen=64/CompInt8 not available on this host";
    }

    struct Shape { size_t M, N, K; };
    constexpr Shape shapes[] = {
        {1,   16,  64},
        {1,   32, 128},
        {1,   64, 256},
        {4,   16,  64},
        {4,   33, 192},
        {7,   17, 128},
        {16,  64, 512},
        {32, 128, 256},
        // Customer model shapes routed through the full MlasQNBitGemmBatch
        // dispatcher (threading, PerGemmQuantAWorkspace, packed-B). Both decode
        // (M=1) and prefill (M=128) sizes; M=128 forces the multi-threaded
        // path that the direct-call test cannot reach.
        {  1, 1024,  384}, {  1,  192, 1024}, {  1,  384, 1024}, {  1, 4096, 1024}, {  1, 1024, 4096},
        {128, 1024,  384}, {128,  192, 1024}, {128,  384, 1024}, {128, 4096, 1024}, {128, 1024, 4096},
    };

    for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
        for (const Shape& s : shapes) {
            for (bool bias : {false, true}) {
                MlasSQ2BitGemmTest::Run(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u));
            }
        }
    }
}

//
// Same coverage as GemmCompInt8_BlkLen64_PublicApi but with per-block
// non-default zero points (random ZP in [0, 3] per block). This is the
// configuration the customer model uses: 4-input MatMulNBits nodes with an
// explicit zero_points initializer.
//
TEST(MlasSq2BitTest, GemmCompInt8_BlkLen64_PublicApi_WithZeroPoints)
{
    if (!MlasIsQNBitGemmAvailable(kBlkBitWidth, kBlkLen, kComputeType)) {
        GTEST_SKIP() << "MlasQNBitGemm W2/BlkLen=64/CompInt8 not available on this host";
    }

    struct Shape { size_t M, N, K; };
    constexpr Shape shapes[] = {
        {1,   16,  64},
        {1,   32, 128},
        {1,   64, 256},
        {4,   16,  64},
        {4,   33, 192},
        {7,   17, 128},
        {16,  64, 512},
        {32, 128, 256},
        // Customer model shapes (BlkLen=64, asymmetric ZP).
        {  1, 1024,  384}, {  1,  192, 1024}, {  1,  384, 1024}, {  1, 4096, 1024}, {  1, 1024, 4096},
        {128, 1024,  384}, {128,  192, 1024}, {128,  384, 1024}, {128, 4096, 1024}, {128, 1024, 4096},
    };

    for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
        for (const Shape& s : shapes) {
            for (bool bias : {false, true}) {
                MlasSQ2BitGemmTest::Run(s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                                        /*WithZeroPoints=*/true);
            }
        }
    }
}

//
// Direct-call test harness for the AVX-512 W2 kernel variants. Calls the
// kernel through its non-inline test-entry forwarder (declared in
// sqnbitgemm_kernel_avx512_2bit.h), bypassing the platform dispatcher.
// This is the only way to exercise the non-VNNI kernel on a VNNI host.
//
// Setup re-uses the public pack API (MlasQNBitGemmPackQuantBData) and the
// reference int8 A-quantizer in this file (QuantizeA_Reference, which is
// bit-identical to QuantizeARow_CompInt8_avx512). Output is compared
// against ReferenceGemm_W2_CompInt8 within a tight tolerance.
//
class MlasSQ2BitGemmDirectCallTest {
 public:
    static void Run(size_t M, size_t N, size_t K, bool WithBias, uint32_t seed,
                    bool TestVnni, bool WithZeroPoints = false)
    {
        const size_t BlockCountK = (K + kBlkLen - 1) / kBlkLen;
        ASSERT_EQ(K % kBlkLen, 0u) << "Test K must be a multiple of BlkLen=64";

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);
        std::uniform_int_distribution<uint32_t> w_dist(0, 3);
        std::uniform_real_distribution<float> s_dist(0.05f, 0.5f);

        std::vector<float> A(M * K);
        for (auto& v : A) v = a_dist(rng);

        std::vector<uint8_t> BWeights(N * K);
        for (auto& v : BWeights) v = static_cast<uint8_t>(w_dist(rng));

        std::vector<std::byte> QuantBData(N * BlockCountK * kBlkBytes, std::byte{0});
        for (size_t n = 0; n < N; ++n) {
            for (size_t blk = 0; blk < BlockCountK; ++blk) {
                uint8_t blk_weights[kBlkLen];
                for (size_t kk = 0; kk < kBlkLen; ++kk) {
                    blk_weights[kk] = BWeights[n * K + blk * kBlkLen + kk];
                }
                PackSourceBlock_BlkLen64(
                    blk_weights,
                    QuantBData.data() + (n * BlockCountK + blk) * kBlkBytes);
            }
        }

        std::vector<float> QuantBScale(N * BlockCountK);
        for (auto& v : QuantBScale) v = s_dist(rng);

        // Per-block zero points (same conventions as the public-API harness).
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

        // Pack B through the public API (produces the same buffer for both
        // VNNI and non-VNNI consumers; the kernel doesn't care which one).
        const size_t PackedSize = MlasQNBitGemmPackQuantBDataSize(
            N, K, kBlkBitWidth, kBlkLen, WithZeroPoints, kComputeType, nullptr);
        ASSERT_GT(PackedSize, 0u);
        std::vector<std::byte> PackedQuantBBuf(PackedSize, std::byte{0});

        MlasQNBitGemmPackQuantBData(
            N, K, kBlkBitWidth, kBlkLen, kComputeType,
            QuantBData.data(), PackedQuantBBuf.data(),
            QuantBScale.data(), WithZeroPoints, BZeroPointsMlas,
            nullptr, nullptr);

        // Reconstruct the packed-B view so we can pass the right sub-pointers
        // to the kernel forwarder (PackedQuantBData, PackedQuantBScale,
        // QuantBBlkSum). The struct is a layout overlay over the buffer.
        PackedQuantBDataStruct<float, 2> packed_b(
            PackedQuantBBuf.data(), N, BlockCountK, kBlkLen, /*QuantAUnsigned=*/false);

        // Quantize A (bit-identical to MLAS's QuantizeARow_CompInt8_avx512 for
        // BlkLen=64: per-block symmetric scale = amax / 127). Also compute the
        // scaled block sums the kernel's BlkSum correction needs.
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

        // Run the kernel under test directly via the test-entry forwarder.
        std::vector<float> C(M * N, 0.0f);
        if (TestVnni) {
            onnxruntime::mlas::sq2bit_avx512::SQ2BitGemmKernel_BlkSum_CompInt8_Avx512Vnni_TestEntry(
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
        } else {
            onnxruntime::mlas::sq2bit_avx512::SQ2BitGemmKernel_BlkSum_CompInt8_Avx512_TestEntry(
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
        }

        // Reference: bit-exact integer-domain math.
        std::vector<float> CRef(M * N, 0.0f);
        ReferenceGemm_W2_CompInt8(M, N, K, A.data(), BWeights, QuantBScale.data(),
                                  BZeroPointsRef, BiasPtr, CRef.data());

        const float abs_tol = 1e-4f;
        const float rel_tol = 1e-4f;
        for (size_t i = 0; i < M * N; ++i) {
            const float diff = std::fabs(C[i] - CRef[i]);
            const float bound = abs_tol + rel_tol * std::fabs(CRef[i]);
            ASSERT_LE(diff, bound)
                << (TestVnni ? "VNNI" : "non-VNNI") << " direct-call mismatch at i=" << i
                << " (m=" << (i / N) << ", n=" << (i % N) << ")"
                << " MLAS=" << C[i] << " Ref=" << CRef[i]
                << " M=" << M << " N=" << N << " K=" << K
                << " WithBias=" << WithBias
                << " WithZeroPoints=" << WithZeroPoints;
        }
    }
};

//
// Exercises the non-VNNI W2 kernel (vpmaddubsw + vpmaddwd + vpaddd MAC chain).
// Gated on AVX-512BW availability. Runs even on VNNI hosts where the platform
// dispatcher would never select this kernel naturally.
//
TEST(MlasSq2BitTest, GemmCompInt8_BlkLen64_Avx512)
{
    if (!GetMlasPlatform().Avx512Supported_) {
        GTEST_SKIP() << "AVX-512BW (DQ/VL) not available on this host";
    }

    struct Shape { size_t M, N, K; };
    constexpr Shape shapes[] = {
        {1,   16,  64},
        {1,   32, 128},
        {1,   64, 256},
        {4,   16,  64},
        {4,   33, 192},
        {7,   17, 128},
        {16,  64, 512},
        {32, 128, 256},
    };

    for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
        for (const Shape& s : shapes) {
            for (bool bias : {false, true}) {
                MlasSQ2BitGemmDirectCallTest::Run(
                    s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                    /*TestVnni=*/false);
            }
        }
    }
}

//
// Same as GemmCompInt8_BlkLen64_Avx512 but with random per-block zero points.
//
TEST(MlasSq2BitTest, GemmCompInt8_BlkLen64_Avx512_WithZeroPoints)
{
    if (!GetMlasPlatform().Avx512Supported_) {
        GTEST_SKIP() << "AVX-512BW (DQ/VL) not available on this host";
    }

    struct Shape { size_t M, N, K; };
    constexpr Shape shapes[] = {
        {1,   16,  64},
        {1,   32, 128},
        {1,   64, 256},
        {4,   16,  64},
        {4,   33, 192},
        {7,   17, 128},
        {16,  64, 512},
        {32, 128, 256},
    };

    for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
        for (const Shape& s : shapes) {
            for (bool bias : {false, true}) {
                MlasSQ2BitGemmDirectCallTest::Run(
                    s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                    /*TestVnni=*/false, /*WithZeroPoints=*/true);
            }
        }
    }
}

//
// Exercises the AVX-512-VNNI W2 kernel (_mm512_dpbusd_epi32 MAC) via the
// direct-call forwarder. On a VNNI host this is the same kernel that the
// public-API test above hits through the dispatcher; the explicit invocation
// here keeps the harness symmetric and validates the forwarder mechanism.
//
// Gating: requires the platform to have actually selected the VNNI dispatch
// table. MlasIsQNBitGemmAvailable alone is insufficient because the W2 path
// is now registered into both AVX-512 and AVX-512-VNNI dispatch tables.
//
TEST(MlasSq2BitTest, GemmCompInt8_BlkLen64_Avx512Vnni)
{
    if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx512vnni) {
        GTEST_SKIP() << "AVX-512-VNNI not selected as the active dispatch on this host";
    }

    struct Shape { size_t M, N, K; };
    constexpr Shape shapes[] = {
        {1,   16,  64},
        {1,   32, 128},
        {1,   64, 256},
        {4,   16,  64},
        {4,   33, 192},
        {7,   17, 128},
        {16,  64, 512},
        {32, 128, 256},
        // Customer model shapes (BlkLen=64). Both decode (M=1) and prefill
        // (M=128) rows; these are far larger than the small-N shapes above
        // and exercise the R1xC4 / R2xC4 tile paths at production sizes.
        {  1, 1024,  384}, {  1,  192, 1024}, {  1,  384, 1024}, {  1, 4096, 1024}, {  1, 1024, 4096},
        {128, 1024,  384}, {128,  192, 1024}, {128,  384, 1024}, {128, 4096, 1024}, {128, 1024, 4096},
    };

    for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
        for (const Shape& s : shapes) {
            for (bool bias : {false, true}) {
                MlasSQ2BitGemmDirectCallTest::Run(
                    s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                    /*TestVnni=*/true);
            }
        }
    }
}

//
// Same as GemmCompInt8_BlkLen64_Avx512Vnni but with random per-block zero points.
//
TEST(MlasSq2BitTest, GemmCompInt8_BlkLen64_Avx512Vnni_WithZeroPoints)
{
    if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx512vnni) {
        GTEST_SKIP() << "AVX-512-VNNI not selected as the active dispatch on this host";
    }

    struct Shape { size_t M, N, K; };
    constexpr Shape shapes[] = {
        {1,   16,  64},
        {1,   32, 128},
        {1,   64, 256},
        {4,   16,  64},
        {4,   33, 192},
        {7,   17, 128},
        {16,  64, 512},
        {32, 128, 256},
        // Customer model shapes with explicit per-block zero points (matches the
        // production accuracy concern).
        {  1, 1024,  384}, {  1,  192, 1024}, {  1,  384, 1024}, {  1, 4096, 1024}, {  1, 1024, 4096},
        {128, 1024,  384}, {128,  192, 1024}, {128,  384, 1024}, {128, 4096, 1024}, {128, 1024, 4096},
    };

    for (uint32_t seed : {0xC0FFEEu, 0xBADC0DEu}) {
        for (const Shape& s : shapes) {
            for (bool bias : {false, true}) {
                MlasSQ2BitGemmDirectCallTest::Run(
                    s.M, s.N, s.K, bias, seed + (bias ? 1u : 0u),
                    /*TestVnni=*/true, /*WithZeroPoints=*/true);
            }
        }
    }
}
