/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_sqlutgemm.cpp

Abstract:

    Tests for MLAS LUT-based n-bit GEMM (TMAC/LUT path) for 2-bit..

--*/

#include "test_util.h"
#include "mlas_qnbit.h"
#include "mlas_q4.h"

// Generic template to future-proof for different bit widths; instantiate with 2 for now.
template <size_t BlkBitWidth, size_t BlkLen>
class MlasSQLutGemmTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferA;
  MatrixGuardBuffer<float> BufferB;
  MatrixGuardBuffer<float> BufferC;
  MatrixGuardBuffer<float> BufferCReference;

  MatrixGuardBuffer<uint8_t> BufferQuantBData;
  MatrixGuardBuffer<float> BufferQuantBScale;
  MatrixGuardBuffer<uint8_t> BufferQuantBZeroPoint;
  MatrixGuardBuffer<float> BufferDequantizedB;
  MatrixGuardBuffer<std::byte> BufferPackedB;  // Single buffer for packed weights and scales

  void CallReferenceGemm(size_t M,
                         size_t N,
                         size_t K,
                         const float* A,
                         const uint8_t* QuantBData,
                         const float* QuantBScale,
                         const uint8_t* QuantBZeroPoint,
                         float* C) {
    float* DequantizedBData = BufferDequantizedB.GetBuffer(K * N);
    MlasDequantizeBlockwise<float, BlkBitWidth>(
        DequantizedBData, QuantBData, QuantBScale, QuantBZeroPoint, BlkLen, /* columnwise */ true,
        static_cast<int>(K), static_cast<int>(N), GetMlasThreadPool());

    // Note: DequantizedBData is in column major layout.

    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        const float* a = A + m * K;
        const float* b = DequantizedBData + n * K;
        float* c = C + (m * N) + n;
        float sum = 0.0f;
        for (size_t k = 0; k < K; k++) {
          sum += (*a) * (*b);
          b += 1;
          a += 1;
        }
        *c = sum;
      }
    }
  }

 public:
  void Test(size_t M, size_t N, size_t K, bool WithThreadpool, bool Symmetric) {
    MLAS_THREADPOOL* tp = WithThreadpool ? GetMlasThreadPool() : nullptr;

    // Clear config cache to ensure fresh config for each test case
    MlasClearLutGemmKernelConfig();

    const float* A = BufferA.GetBuffer(K * M);
    const float* B = BufferB.GetBuffer(N * K);
    float* C = BufferC.GetBuffer(N * M, true);
    float* CReference = BufferCReference.GetBuffer(N * M, true);

    // quantize B
    uint8_t* QuantBData = nullptr;
    float* QuantBScale = nullptr;
    uint8_t* QuantBZeroPoint = nullptr;

    {
      size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
      MlasBlockwiseQuantizedBufferSizes<BlkBitWidth>(BlkLen, /* columnwise */ true,
                                                     static_cast<int>(K), static_cast<int>(N),
                                                     QuantBDataSizeInBytes, QuantBScaleSize, &QuantBZeroPointSizeInBytes);

      QuantBData = BufferQuantBData.GetBuffer(QuantBDataSizeInBytes);
      QuantBScale = BufferQuantBScale.GetBuffer(QuantBScaleSize);
      if (!Symmetric) {
        QuantBZeroPoint = BufferQuantBZeroPoint.GetBuffer(QuantBZeroPointSizeInBytes);
      }

      MlasQuantizeBlockwise<float, BlkBitWidth>(QuantBData, QuantBScale, QuantBZeroPoint,
                                                B, BlkLen,
                                                /* columnwise */ true,
                                                static_cast<int>(K), static_cast<int>(N),
                                                static_cast<int>(N),
                                                GetMlasThreadPool());
    }

    MlasInitLutGemmKernelConfig(N, K, BlkBitWidth, BlkLen, !Symmetric);

    // Use unified packing - single buffer for weights and scales/zp
    size_t PackedBufSize = MlasLutGemmPackedSize(N, K, BlkBitWidth, BlkLen, !Symmetric);
    std::byte* PackedBuf = BufferPackedB.GetBuffer(PackedBufSize);

    MlasLutGemmPack(
        N,
        K,
        BlkBitWidth,
        BlkLen,
        !Symmetric,
        reinterpret_cast<std::byte*>(QuantBData),
        QuantBScale,
        QuantBZeroPoint,
        false,  // IsFloatZeroPoint
        PackedBuf,
        tp);

    MlasLutGemm(
        A,
        BlkLen,
        PackedBuf,
        C,
        static_cast<int>(K),
        static_cast<int>(M),
        static_cast<int>(N),
        !Symmetric,
        tp);

    // Reference computation
    CallReferenceGemm(M, N, K, A, QuantBData, QuantBScale, QuantBZeroPoint, CReference);

    size_t f = 0;
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++, f++) {
        ASSERT_TRUE(CloseEnough(C[f], CReference[f]))
            << "Expected: " << CReference[f] << " Actual: " << C[f] << "@[" << m << "x" << n << "], "
            << "M=" << M << ", N=" << N << ", K=" << K;
      }
    }
  }

  void TestFloatZeroPoint(size_t M, size_t N, size_t K, bool WithThreadpool, float zp_value) {
    MLAS_THREADPOOL* tp = WithThreadpool ? GetMlasThreadPool() : nullptr;

    MlasClearLutGemmKernelConfig();

    const float* A = BufferA.GetBuffer(K * M);
    const float* B = BufferB.GetBuffer(N * K);
    float* C = BufferC.GetBuffer(N * M, true);
    float* CReference = BufferCReference.GetBuffer(N * M, true);

    // Quantize B (symmetric — we'll supply our own float ZP)
    uint8_t* QuantBData = nullptr;
    float* QuantBScale = nullptr;

    size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
    MlasBlockwiseQuantizedBufferSizes<BlkBitWidth>(BlkLen, true,
                                                   static_cast<int>(K), static_cast<int>(N),
                                                   QuantBDataSizeInBytes, QuantBScaleSize, &QuantBZeroPointSizeInBytes);

    QuantBData = BufferQuantBData.GetBuffer(QuantBDataSizeInBytes);
    QuantBScale = BufferQuantBScale.GetBuffer(QuantBScaleSize);

    MlasQuantizeBlockwise<float, BlkBitWidth>(QuantBData, QuantBScale, nullptr,
                                              B, BlkLen, true,
                                              static_cast<int>(K), static_cast<int>(N),
                                              static_cast<int>(N), GetMlasThreadPool());

    // Create float zero point array — one per quantization group, all set to zp_value
    size_t k_blocks = (K + BlkLen - 1) / BlkLen;
    std::vector<float> FloatZeroPoints(N * k_blocks, zp_value);

    MlasInitLutGemmKernelConfig(N, K, BlkBitWidth, BlkLen, true);  // HasZeroPoint = true

    size_t PackedBufSize = MlasLutGemmPackedSize(N, K, BlkBitWidth, BlkLen, true);
    std::byte* PackedBuf = BufferPackedB.GetBuffer(PackedBufSize);

    MlasLutGemmPack(
        N, K, BlkBitWidth, BlkLen, true,
        reinterpret_cast<std::byte*>(QuantBData),
        QuantBScale,
        FloatZeroPoints.data(),
        true,  // IsFloatZeroPoint
        PackedBuf, tp);

    MlasLutGemm(A, BlkLen, PackedBuf, C,
                static_cast<int>(K), static_cast<int>(M), static_cast<int>(N),
                true, tp);

    // Reference: scalar dequant with float ZP, then GEMM
    float* DequantizedBData = BufferDequantizedB.GetBuffer(K * N);
    const size_t elems_per_byte = 8 / BlkBitWidth;
    size_t packed_k = k_blocks * BlkLen;
    size_t bytes_per_col = packed_k / elems_per_byte;
    for (size_t n = 0; n < N; n++) {
      for (size_t k = 0; k < K; k++) {
        size_t block_idx = k / BlkLen;
        float scale = QuantBScale[n * k_blocks + block_idx];
        size_t packed_idx = n * bytes_per_col + k / elems_per_byte;
        size_t bit_offset = (k % elems_per_byte) * BlkBitWidth;
        uint8_t q = (QuantBData[packed_idx] >> bit_offset) & ((1 << BlkBitWidth) - 1);
        // Use float ZP directly, not the midpoint-based default
        DequantizedBData[n * K + k] = (static_cast<float>(q) - zp_value) * scale;
      }
    }

    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; k++) {
          sum += A[m * K + k] * DequantizedBData[n * K + k];
        }
        CReference[m * N + n] = sum;
      }
    }

    size_t f = 0;
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++, f++) {
        ASSERT_TRUE(CloseEnough(C[f], CReference[f]))
            << "Expected: " << CReference[f] << " Actual: " << C[f] << "@[" << m << "x" << n << "], "
            << "M=" << M << ", N=" << N << ", K=" << K << ", FloatZP=" << zp_value;
      }
    }
  }

  // Verifies that the bias argument to MlasLutGemm is correctly broadcast-added per row.
  // Bias has shape [N] and must be added to every row of the [M, N] output.
  void TestWithBias(size_t M, size_t N, size_t K, bool WithThreadpool, bool Symmetric) {
    MLAS_THREADPOOL* tp = WithThreadpool ? GetMlasThreadPool() : nullptr;

    MlasClearLutGemmKernelConfig();

    const float* A = BufferA.GetBuffer(K * M);
    const float* B = BufferB.GetBuffer(N * K);
    float* C = BufferC.GetBuffer(N * M, true);
    float* CReference = BufferCReference.GetBuffer(N * M, true);

    uint8_t* QuantBData = nullptr;
    float* QuantBScale = nullptr;
    uint8_t* QuantBZeroPoint = nullptr;

    {
      size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
      MlasBlockwiseQuantizedBufferSizes<BlkBitWidth>(BlkLen, /* columnwise */ true,
                                                     static_cast<int>(K), static_cast<int>(N),
                                                     QuantBDataSizeInBytes, QuantBScaleSize, &QuantBZeroPointSizeInBytes);

      QuantBData = BufferQuantBData.GetBuffer(QuantBDataSizeInBytes);
      QuantBScale = BufferQuantBScale.GetBuffer(QuantBScaleSize);
      if (!Symmetric) {
        QuantBZeroPoint = BufferQuantBZeroPoint.GetBuffer(QuantBZeroPointSizeInBytes);
      }

      MlasQuantizeBlockwise<float, BlkBitWidth>(QuantBData, QuantBScale, QuantBZeroPoint,
                                                B, BlkLen, /* columnwise */ true,
                                                static_cast<int>(K), static_cast<int>(N),
                                                static_cast<int>(N), GetMlasThreadPool());
    }

    MlasInitLutGemmKernelConfig(N, K, BlkBitWidth, BlkLen, !Symmetric);

    size_t PackedBufSize = MlasLutGemmPackedSize(N, K, BlkBitWidth, BlkLen, !Symmetric);
    std::byte* PackedBuf = BufferPackedB.GetBuffer(PackedBufSize);

    MlasLutGemmPack(
        N, K, BlkBitWidth, BlkLen, !Symmetric,
        reinterpret_cast<std::byte*>(QuantBData),
        QuantBScale,
        QuantBZeroPoint,
        false,  // IsFloatZeroPoint
        PackedBuf, tp);

    // Build a deterministic per-output-feature bias vector with non-trivial variation across N
    // so that any stride/transpose bug in the fused bias add will be caught.
    std::vector<float> Bias(N);
    for (size_t n = 0; n < N; ++n) {
      Bias[n] = 0.125f + 0.5f * static_cast<float>(n % 7) - 0.25f * static_cast<float>(n % 11);
    }

    MlasLutGemm(
        A, BlkLen, PackedBuf, C,
        static_cast<int>(K), static_cast<int>(M), static_cast<int>(N),
        !Symmetric, tp, Bias.data());

    // Reference: same GEMM as the no-bias path, then broadcast-add Bias.
    CallReferenceGemm(M, N, K, A, QuantBData, QuantBScale, QuantBZeroPoint, CReference);
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        CReference[m * N + n] += Bias[n];
      }
    }

    size_t f = 0;
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++, f++) {
        ASSERT_TRUE(CloseEnough(C[f], CReference[f]))
            << "Expected: " << CReference[f] << " Actual: " << C[f] << "@[" << m << "x" << n << "], "
            << "M=" << M << ", N=" << N << ", K=" << K << ", Symmetric=" << Symmetric << ", WithBias=true";
      }
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("SQLutGemm") +
                                    "BlkBitWidth" + std::to_string(BlkBitWidth) +
                                    "BlkLen" + std::to_string(BlkLen);
    return suite_name.c_str();
  }
};

// Fixture to register parameterized tests quickly
template <size_t BlkBitWidth, size_t BlkLen>
class SQLutGemmShortExecuteTest : public MlasTestFixture<MlasSQLutGemmTest<BlkBitWidth, BlkLen>> {
 public:
  explicit SQLutGemmShortExecuteTest(size_t M, size_t N, size_t K,
                                     bool WithThreadpool, bool Symmetric)
      : M_(M),
        N_(N),
        K_(K),
        WithThreadpool_(WithThreadpool),
        Symmetric_(Symmetric) {
  }

  void TestBody() override {
    MlasTestFixture<MlasSQLutGemmTest<BlkBitWidth, BlkLen>>::mlas_tester->Test(
        M_, N_, K_, WithThreadpool_, Symmetric_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, bool WithThreadpool, bool Symmetric) {
    if (!MlasIsLutGemmAvailable(N, K, BlkBitWidth, BlkLen)) {
      return 0;
    }

    if (N < BlkLen) {
      return 0;
    }

    std::stringstream ss;
    ss << (WithThreadpool ? "Threaded" : "SingleThread")
       << "/isSymmetric" << Symmetric
       << "/M" << M << "xN" << N << "xK" << K;

    auto test_name = ss.str();

    testing::RegisterTest(
        MlasSQLutGemmTest<BlkBitWidth, BlkLen>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasSQLutGemmTest<BlkBitWidth, BlkLen>>* {
          return new SQLutGemmShortExecuteTest<BlkBitWidth, BlkLen>(
              M, N, K, WithThreadpool, Symmetric);
        });

    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t count = 0;
    for (bool with_threadpool : {true}) {
      for (bool symmetric : {true, false}) {  // Test both symmetric and asymmetric
        for (size_t b = 256; b < 320; b += 32) {
          count += RegisterSingleTest(b, b, b, with_threadpool, symmetric);
        }

        count += RegisterSingleTest(64, 128, 128, with_threadpool, symmetric);
        count += RegisterSingleTest(128, 256, 256, with_threadpool, symmetric);

        count += RegisterSingleTest(1, 128, 128, with_threadpool, symmetric);
        count += RegisterSingleTest(1, 1024, 1024, with_threadpool, symmetric);
      }
    }
    return count;
  }

 private:
  size_t M_, N_, K_;
  bool WithThreadpool_, Symmetric_;
};

// Fixture for float zero point tests
template <size_t BlkBitWidth, size_t BlkLen>
class SQLutGemmFloatZPTest : public MlasTestFixture<MlasSQLutGemmTest<BlkBitWidth, BlkLen>> {
 public:
  explicit SQLutGemmFloatZPTest(size_t M, size_t N, size_t K, bool WithThreadpool, float ZPValue)
      : M_(M), N_(N), K_(K), WithThreadpool_(WithThreadpool), ZPValue_(ZPValue) {}

  void TestBody() override {
    MlasTestFixture<MlasSQLutGemmTest<BlkBitWidth, BlkLen>>::mlas_tester->TestFloatZeroPoint(
        M_, N_, K_, WithThreadpool_, ZPValue_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, bool WithThreadpool, float ZPValue) {
    if (!MlasIsLutGemmAvailable(N, K, BlkBitWidth, BlkLen)) {
      return 0;
    }
    if (N < BlkLen) {
      return 0;
    }

    std::stringstream ss;
    ss << (WithThreadpool ? "Threaded" : "SingleThread")
       << "/FloatZP" << ZPValue
       << "/M" << M << "xN" << N << "xK" << K;

    auto test_name = ss.str();

    testing::RegisterTest(
        MlasSQLutGemmTest<BlkBitWidth, BlkLen>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        [=]() -> MlasTestFixture<MlasSQLutGemmTest<BlkBitWidth, BlkLen>>* {
          return new SQLutGemmFloatZPTest<BlkBitWidth, BlkLen>(M, N, K, WithThreadpool, ZPValue);
        });

    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t count = 0;
    // Test QAD-specific ZP=1.5 and other values
    for (float zp : {1.5f, 0.0f, 2.0f, 3.0f}) {
      count += RegisterSingleTest(1, 128, 128, true, zp);
      count += RegisterSingleTest(1, 256, 256, true, zp);
      count += RegisterSingleTest(128, 256, 256, true, zp);
    }
    return count;
  }

 private:
  size_t M_, N_, K_;
  bool WithThreadpool_;
  float ZPValue_;
};

// Fixture for the fused bias path (MlasLutGemm Bias parameter).
template <size_t BlkBitWidth, size_t BlkLen>
class SQLutGemmBiasTest : public MlasTestFixture<MlasSQLutGemmTest<BlkBitWidth, BlkLen>> {
 public:
  explicit SQLutGemmBiasTest(size_t M, size_t N, size_t K, bool WithThreadpool, bool Symmetric)
      : M_(M), N_(N), K_(K), WithThreadpool_(WithThreadpool), Symmetric_(Symmetric) {}

  void TestBody() override {
    MlasTestFixture<MlasSQLutGemmTest<BlkBitWidth, BlkLen>>::mlas_tester->TestWithBias(
        M_, N_, K_, WithThreadpool_, Symmetric_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, bool WithThreadpool, bool Symmetric) {
    if (!MlasIsLutGemmAvailable(N, K, BlkBitWidth, BlkLen)) {
      return 0;
    }
    if (N < BlkLen) {
      return 0;
    }

    std::stringstream ss;
    ss << (WithThreadpool ? "Threaded" : "SingleThread")
       << "/Bias/isSymmetric" << Symmetric
       << "/M" << M << "xN" << N << "xK" << K;

    auto test_name = ss.str();

    testing::RegisterTest(
        MlasSQLutGemmTest<BlkBitWidth, BlkLen>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        [=]() -> MlasTestFixture<MlasSQLutGemmTest<BlkBitWidth, BlkLen>>* {
          return new SQLutGemmBiasTest<BlkBitWidth, BlkLen>(M, N, K, WithThreadpool, Symmetric);
        });

    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t count = 0;
    for (bool with_threadpool : {true}) {
      for (bool symmetric : {true, false}) {
        // Cover M=1 (decode-like) and M>1 (prefill/batched) paths, and multiple chunked-N sizes
        // so we exercise the per-tile bias slice arithmetic across more than one tile.
        count += RegisterSingleTest(1, 128, 128, with_threadpool, symmetric);
        count += RegisterSingleTest(1, 256, 256, with_threadpool, symmetric);
        count += RegisterSingleTest(1, 1024, 1024, with_threadpool, symmetric);
        count += RegisterSingleTest(32, 128, 128, with_threadpool, symmetric);
        count += RegisterSingleTest(32, 256, 256, with_threadpool, symmetric);
      }
    }
    return count;
  }

 private:
  size_t M_, N_, K_;
  bool WithThreadpool_;
  bool Symmetric_;
};

static size_t SQLutGemmRegisterAllShortExecuteTests() {
  size_t count = 0;
  count += SQLutGemmShortExecuteTest<2, 32>::RegisterShortExecuteTests();
  count += SQLutGemmShortExecuteTest<2, 64>::RegisterShortExecuteTests();
  count += SQLutGemmShortExecuteTest<2, 128>::RegisterShortExecuteTests();
  count += SQLutGemmShortExecuteTest<2, 256>::RegisterShortExecuteTests();
  // Float zero point tests
  count += SQLutGemmFloatZPTest<2, 32>::RegisterShortExecuteTests();
  count += SQLutGemmFloatZPTest<2, 64>::RegisterShortExecuteTests();
  count += SQLutGemmFloatZPTest<2, 128>::RegisterShortExecuteTests();
  // Fused bias tests
  count += SQLutGemmBiasTest<2, 32>::RegisterShortExecuteTests();
  count += SQLutGemmBiasTest<2, 64>::RegisterShortExecuteTests();
  count += SQLutGemmBiasTest<2, 128>::RegisterShortExecuteTests();
  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        return SQLutGemmRegisterAllShortExecuteTests();
      }
      return 0;
    });
