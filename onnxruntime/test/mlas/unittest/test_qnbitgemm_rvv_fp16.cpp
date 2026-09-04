/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_qnbitgemm_rvv_fp16.cpp

Abstract:

    End-to-end tests for the RISC-V RVV HQNBIT_CompFp16 (fp16 activation) path of
    MatMulNBits: pack quantized B, dequantize it through the platform dispatch,
    run the fp16 GEMM kernel, and compare against a reference. Requires Zvfh.

--*/

#include "test_util.h"
#include "core/mlas/lib/mlasi.h"
#include "core/mlas/lib/qnbitgemm.h"
#include "mlas_qnbit.h"

#if defined(MLAS_TARGET_RISCV64) && defined(MLAS_USE_RVV_ZVFH)

class MlasRvvFp16HQ4GemmTest : public MlasTestBase {
 private:
  unsigned int seed_ = 1234;
  std::mt19937 gen_{seed_};
  MatrixGuardBuffer<uint8_t> rawB_, packedB_, zp_;
  MatrixGuardBuffer<MLAS_FP16> A_, scale_, bias_, dequantB_, C_;

  static float H2F(MLAS_FP16 h) { return h.ToFloat(); }

  template <size_t M, size_t N, size_t K, size_t BlkLen, bool HasZp, bool HasBias>
  void RunOne() {
    if (!MlasIsQNBitGemmAvailable(4, BlkLen, HQNBIT_CompFp16)) {
      GTEST_SKIP() << "HQNBIT_CompFp16 not available";
    }
    const auto* dispatch = GetMlasPlatform().QNBitGemmDispatch;
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t BCK = (K + BlkLen - 1) / BlkLen;
    constexpr size_t ldb = BCK * BlkLen;
    constexpr size_t BlkDataSize = BlkLen / 2;  // 4-bit
    const size_t ZpBytesPerCol = (BCK + 1) / 2;

    std::uniform_int_distribution<int> u8(0, 255);

    // Raw 4-bit B (two nibbles per byte), row-major [N][BCK*BlkDataSize].
    auto* rawB = rawB_.GetBuffer(N * BCK * BlkDataSize, true);
    for (size_t i = 0; i < N * BCK * BlkDataSize; ++i) rawB[i] = static_cast<uint8_t>(u8(gen_));

    auto* scale = scale_.GetBuffer(N * BCK, true);
    for (size_t i = 0; i < N * BCK; ++i) scale[i] = MLAS_FP16(0.02f + (u8(gen_) % 32) * 0.01f);

    uint8_t* zp = nullptr;
    if (HasZp) {
      zp = zp_.GetBuffer(N * ZpBytesPerCol, true);
      for (size_t i = 0; i < N * ZpBytesPerCol; ++i) zp[i] = static_cast<uint8_t>(u8(gen_));
    }

    auto* A = A_.GetBuffer(M * K, true);
    for (size_t i = 0; i < M * K; ++i) A[i] = MLAS_FP16(-1.5f + (u8(gen_) % 96) * 0.03f);
    auto* bias = bias_.GetBuffer(N, true);
    for (size_t i = 0; i < N; ++i) bias[i] = MLAS_FP16(-0.5f + (u8(gen_) % 64) * 0.02f);

    // Pack B through the real pack entry point.
    const size_t packedSize = MlasQNBitGemmPackQuantBDataSize(N, K, BlkBitWidth, BlkLen, HasZp, HQNBIT_CompFp16, nullptr);
    ASSERT_GT(packedSize, size_t{0});
    auto* packedB = packedB_.GetBuffer(packedSize, true);
    MlasQNBitGemmPackQuantBData(N, K, BlkBitWidth, BlkLen, HQNBIT_CompFp16,
                                rawB, packedB, /*scale*/ nullptr, HasZp, /*zp*/ nullptr, nullptr, nullptr);

    // Dequantize B to fp16 through the dispatch.
    auto* dequantB = dequantB_.GetBuffer(N * ldb, true);
    dispatch->HQ4BitBlkDequantBForHgemm_CompFp16(
        BlkLen, dequantB, reinterpret_cast<const std::byte*>(packedB), scale,
        HasZp ? reinterpret_cast<const std::byte*>(zp) : nullptr, N, K, BCK);

    // fp16 GEMM through the dispatch.
    auto* C = C_.GetBuffer(M * N, true);
    dispatch->HQ4BitGemmKernel_CompFp16(A, dequantB, HasBias ? bias : nullptr, C, M, N, K, K, ldb, N);

    // Reference: fp32 accumulate over the same fp16 A and fp16 dequantized B.
    double maxrel = 0.0;
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        double acc = HasBias ? H2F(bias[n]) : 0.0;
        double absum = HasBias ? std::fabs(H2F(bias[n])) : 0.0;
        for (size_t k = 0; k < K; ++k) {
          double t = static_cast<double>(H2F(A[m * K + k])) * H2F(dequantB[n * ldb + k]);
          acc += t;
          absum += std::fabs(t);
        }
        const double got = H2F(C[m * N + n]);
        maxrel = std::max(maxrel, std::fabs(got - acc) / (absum + 1e-3));
      }
    }
    ASSERT_LT(maxrel, 6e-3) << " M=" << M << " N=" << N << " K=" << K << " BlkLen=" << BlkLen
                            << " zp=" << HasZp << " bias=" << HasBias;
  }

  // HQ8: 8-bit weights, fp16 activations (plain 8-bit pack + HQ8 dequant + shared fp16 kernel).
  template <size_t M, size_t N, size_t K, size_t BlkLen, bool HasZp, bool HasBias>
  void RunOne8() {
    if (!MlasIsQNBitGemmAvailable(8, BlkLen, HQNBIT_CompFp16)) {
      GTEST_SKIP() << "HQNBIT_CompFp16 (8-bit) not available";
    }
    const auto* dispatch = GetMlasPlatform().QNBitGemmDispatch;
    constexpr size_t BlkBitWidth = 8;
    constexpr size_t BCK = (K + BlkLen - 1) / BlkLen;
    constexpr size_t ldb = BCK * BlkLen;  // 8-bit: one byte per weight

    std::uniform_int_distribution<int> u8(0, 255);

    auto* rawB = rawB_.GetBuffer(N * ldb, true);
    for (size_t i = 0; i < N * ldb; ++i) rawB[i] = static_cast<uint8_t>(u8(gen_));
    auto* scale = scale_.GetBuffer(N * BCK, true);
    for (size_t i = 0; i < N * BCK; ++i) scale[i] = MLAS_FP16(0.02f + (u8(gen_) % 32) * 0.01f);
    uint8_t* zp = nullptr;
    if (HasZp) {
      zp = zp_.GetBuffer(N * BCK, true);  // 8-bit: one zp byte per block
      for (size_t i = 0; i < N * BCK; ++i) zp[i] = static_cast<uint8_t>(u8(gen_));
    }
    auto* A = A_.GetBuffer(M * K, true);
    for (size_t i = 0; i < M * K; ++i) A[i] = MLAS_FP16(-1.5f + (u8(gen_) % 96) * 0.03f);
    auto* bias = bias_.GetBuffer(N, true);
    for (size_t i = 0; i < N; ++i) bias[i] = MLAS_FP16(-0.5f + (u8(gen_) % 64) * 0.02f);

    const size_t packedSize = MlasQNBitGemmPackQuantBDataSize(N, K, BlkBitWidth, BlkLen, HasZp, HQNBIT_CompFp16, nullptr);
    ASSERT_GT(packedSize, size_t{0});
    auto* packedB = packedB_.GetBuffer(packedSize, true);
    MlasQNBitGemmPackQuantBData(N, K, BlkBitWidth, BlkLen, HQNBIT_CompFp16,
                                rawB, packedB, nullptr, HasZp, nullptr, nullptr, nullptr);

    auto* dequantB = dequantB_.GetBuffer(N * ldb, true);
    dispatch->HQ8BitBlkDequantBForHgemm_CompFp16(
        BlkLen, dequantB, reinterpret_cast<const std::byte*>(packedB), scale,
        HasZp ? reinterpret_cast<const std::byte*>(zp) : nullptr, N, K, BCK);

    auto* C = C_.GetBuffer(M * N, true);
    dispatch->HQ4BitGemmKernel_CompFp16(A, dequantB, HasBias ? bias : nullptr, C, M, N, K, K, ldb, N);

    double maxrel = 0.0;
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        double acc = HasBias ? H2F(bias[n]) : 0.0;
        double absum = HasBias ? std::fabs(H2F(bias[n])) : 0.0;
        for (size_t k = 0; k < K; ++k) {
          double t = static_cast<double>(H2F(A[m * K + k])) * H2F(dequantB[n * ldb + k]);
          acc += t;
          absum += std::fabs(t);
        }
        maxrel = std::max(maxrel, std::fabs(H2F(C[m * N + n]) - acc) / (absum + 1e-3));
      }
    }
    ASSERT_LT(maxrel, 6e-3) << " HQ8 M=" << M << " N=" << N << " K=" << K << " BlkLen=" << BlkLen
                            << " zp=" << HasZp << " bias=" << HasBias;
  }

 public:
  static const char* GetTestSuiteName() { return "RvvFp16HQ4Gemm"; }

  void ExecuteShort(void) override {
    RunOne<1, 1, 16, 16, false, false>();
    RunOne<1, 8, 32, 32, false, true>();
    RunOne<4, 16, 64, 32, true, false>();
    RunOne<4, 17, 64, 64, true, true>();
    RunOne<8, 33, 128, 64, false, true>();
    RunOne<8, 32, 256, 128, true, true>();
    RunOne<2, 15, 88, 32, true, false>();    // K not multiple of BlkLen
    RunOne<5, 16, 4096, 128, true, true>();  // large K

    RunOne8<1, 1, 16, 16, false, false>();
    RunOne8<1, 8, 32, 32, false, true>();
    RunOne8<4, 16, 64, 32, true, false>();
    RunOne8<4, 17, 64, 64, true, true>();
    RunOne8<8, 33, 128, 64, false, true>();
    RunOne8<8, 32, 256, 128, true, true>();
    RunOne8<2, 15, 88, 32, true, false>();    // K not multiple of BlkLen
    RunOne8<5, 16, 4096, 128, true, true>();  // large K
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) -> size_t {
  if (is_short_execute) {
    return MlasDirectShortExecuteTests<MlasRvvFp16HQ4GemmTest>::RegisterShortExecute();
  }
  return 0;
});

#endif  // defined(MLAS_TARGET_RISCV64) && defined(MLAS_USE_RVV_ZVFH)
