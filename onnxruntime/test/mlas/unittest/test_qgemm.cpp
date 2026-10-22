// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "test_qgemm.h"
#include "test_qgemm_fixture.h"

#include <array>
#include <cstdint>

#if defined(MLAS_KLEIDIAI_INT8_OPS_AVAILABLE)
#include "core/mlas/lib/kleidiai/mlasi_kleidiai.h"
#include "core/mlas/lib/qgemm.h"

TEST(QGemmKleidiAIOps, PackedWeightsPreserveQuantizationAndAccumulation) {
  constexpr size_t N = 17, K = 13, MaxM = 65, lda = K + 3, ldb = N + 3, ldc = N + 4;
  std::array<uint8_t, MaxM * lda> a{};
  std::array<uint8_t, K * ldb> b{};
  std::array<uint8_t, N> zero_points{};
  for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint8_t>(i * 37);
  for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<uint8_t>(i * 53);
  for (size_t i = 0; i < N; ++i) zero_points[i] = static_cast<uint8_t>(i * 17 + 5);

  const size_t packed_size = ArmKleidiAI::MlasQGemmPackBSizeKleidiAIOps(N, K, true, true);
  if (packed_size == 0) GTEST_SKIP() << "No compatible KleidiAI Ops signed INT8 packer";
  ASSERT_EQ(MlasGemmPackBSize(N, K, true, true, nullptr), packed_size);
  MatrixGuardBuffer<uint8_t> packed_buffer;
  void* packed_b = packed_buffer.GetBuffer(packed_size);
  MlasGemmPackB(N, K, b.data(), ldb, true, true, packed_b);

  MLAS_GEMM_QUANT_SHAPE_PARAMS shape{};
  shape.N = N;
  shape.K = K;
  shape.AIsSigned = true;
  shape.BIsSigned = true;
  MLAS_GEMM_QUANT_DATA_PARAMS data{};
  data.A = a.data();
  data.lda = lda;
  data.ZeroPointA = static_cast<uint8_t>(-3);
  data.B = packed_b;
  data.BIsPacked = true;
  data.ZeroPointB = zero_points.data();
  data.ldc = ldc;
  std::array<int32_t, MaxM * ldc> output{}, expected{};

  for (size_t m : {size_t(1), MaxM}) {
    shape.M = m;
    for (bool per_column : {false, true}) {
      data.PerColumnZeroPoints = per_column;
      for (bool accumulate : {false, true}) {
        shape.IsAccumulateMode = accumulate;
        expected.fill(19);
        for (size_t row = 0; row < m; ++row) {
          for (size_t col = 0; col < N; ++col) {
            int32_t sum = accumulate ? 19 : 0;
            const int32_t off_b = static_cast<int8_t>(zero_points[per_column ? col : 0]);
            for (size_t depth = 0; depth < K; ++depth) {
              sum += (static_cast<int8_t>(a[row * lda + depth]) + 3) *
                     (static_cast<int8_t>(b[depth * ldb + col]) - off_b);
            }
            expected[row * ldc + col] = sum;
          }
        }
        for (MLAS_THREADPOOL* pool : {static_cast<MLAS_THREADPOOL*>(nullptr), GetMlasThreadPool()}) {
          SCOPED_TRACE(testing::Message() << "M=" << m << " per_column=" << per_column
                                          << " accumulate=" << accumulate << " threaded=" << (pool != nullptr));
          output.fill(19);
          data.C = output.data();
          ASSERT_TRUE(ArmKleidiAI::MlasQGemmBatchKleidiAIOps(shape, &data, 1, pool));
          EXPECT_EQ(output, expected);
          output.fill(19);
          MlasGemmBatch(shape, &data, 1, pool);
          EXPECT_EQ(output, expected);
        }
      }
    }
  }
}

TEST(QGemmKleidiAIOps, MixedPackedBatchFallsBackWithoutChangingUnsupportedEntries) {
  constexpr size_t M = 3, N = 17, K = 13;
  std::array<uint8_t, M * K> a{};
  std::array<uint8_t, K * N> b{};
  a.fill(2);
  b.fill(3);
  const uint8_t zero_point = 0;
  const size_t packed_size = ArmKleidiAI::MlasQGemmPackBSizeKleidiAIOps(N, K, true, true);
  if (packed_size == 0) GTEST_SKIP() << "No compatible KleidiAI Ops signed INT8 packer";
  MatrixGuardBuffer<uint8_t> packed_buffer;
  void* packed_b = packed_buffer.GetBuffer(packed_size);
  MlasGemmPackB(N, K, b.data(), N, true, true, packed_b);
  MLAS_GEMM_QUANT_SHAPE_PARAMS shape{};
  shape.M = M;
  shape.N = N;
  shape.K = K;
  shape.AIsSigned = true;
  shape.BIsSigned = true;
  std::array<int32_t, M * N * 2> output{};
  output.fill(-11);
  std::array<MLAS_GEMM_QUANT_DATA_PARAMS, 2> data{};
  for (size_t i = 0; i < data.size(); ++i) {
    data[i].A = a.data();
    data[i].lda = K;
    data[i].B = i == 0 ? packed_b : b.data();
    data[i].ldb = N;
    data[i].BIsPacked = i == 0;
    data[i].ZeroPointB = &zero_point;
    data[i].C = output.data() + M * N * i;
    data[i].ldc = N;
  }
  EXPECT_FALSE(ArmKleidiAI::MlasQGemmBatchKleidiAIOps(shape, data.data(), data.size(), nullptr));
  for (int32_t value : output) EXPECT_EQ(value, -11);
  MlasGemmBatch(shape, data.data(), data.size(), GetMlasThreadPool());
  for (int32_t value : output) EXPECT_EQ(value, int32_t(6 * K));

  data[0] = data[1];
  data[0].BIsPacked = true;
  output.fill(-11);
  EXPECT_FALSE(ArmKleidiAI::MlasQGemmBatchKleidiAIOps(shape, data.data(), 1, nullptr));
  for (int32_t value : output) EXPECT_EQ(value, -11);
  data[0].BIsPacked = false;
  for (bool signed_b : {false, true}) {
    shape.AIsSigned = !signed_b;
    shape.BIsSigned = signed_b;
    output.fill(-11);
    EXPECT_EQ(ArmKleidiAI::MlasQGemmPackBSizeKleidiAIOps(N, K, shape.AIsSigned, shape.BIsSigned), 0u);
    EXPECT_FALSE(ArmKleidiAI::MlasQGemmBatchKleidiAIOps(shape, data.data(), 1, nullptr));
    for (int32_t value : output) EXPECT_EQ(value, -11);
  }
}
#endif

TEST(QGemmPackB, UsesKleidiAIBackendSelector) {
  constexpr size_t M = 3, N = 17, K = 13;
  std::array<uint8_t, M * K> a{};
  std::array<uint8_t, K * N> b{};
  std::array<int32_t, M * N> output{};
  a.fill(2);
  b.fill(3);
  const uint8_t zero_point = 0;
  MLAS_GEMM_QUANT_SHAPE_PARAMS shape{};
  shape.M = M;
  shape.N = N;
  shape.K = K;
  shape.AIsSigned = true;
  shape.BIsSigned = true;
  MLAS_GEMM_QUANT_DATA_PARAMS data{};
  data.A = a.data();
  data.lda = K;
  data.BIsPacked = true;
  data.ZeroPointB = &zero_point;
  data.C = output.data();
  data.ldc = N;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG selector_config{};
#if defined(MLAS_KLEIDIAI_INT8_OPS_AVAILABLE)
  const auto* dispatch = MlasGemmQuantGetDispatch(true, true);
  const size_t aligned_n = (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
  const size_t aligned_k = dispatch->PackedK == 0 ? 0 : (K + dispatch->PackedK - 1) & ~(dispatch->PackedK - 1);
  const size_t alignment = MlasGetPreferredBufferAlignment();
  const size_t native_size = dispatch->PackedStrideK == 0 ? 0 : (aligned_n * (sizeof(int32_t) + aligned_k) + alignment - 1) & ~(alignment - 1);
#endif
  for (bool enabled : {false, true, false}) {
    SCOPED_TRACE(testing::Message() << "use_kleidiai=" << enabled);
    selector_config.use_kleidiai = enabled;
    const size_t packed_size = MlasGemmPackBSize(N, K, true, true, &selector_config);
#if defined(MLAS_KLEIDIAI_INT8_OPS_AVAILABLE)
    // Correct native packing alone would not catch an oversized Ops allocation.
    if (!enabled) EXPECT_EQ(packed_size, native_size);
#endif
    if (packed_size == 0) GTEST_SKIP() << "Signed INT8 packing is not supported";
    if (enabled) EXPECT_EQ(packed_size, MlasGemmPackBSize(N, K, true, true, nullptr));
    MatrixGuardBuffer<uint8_t> packed_buffer;
    void* packed_b = packed_buffer.GetBuffer(packed_size);
    data.B = packed_b;
    MlasGemmPackB(N, K, b.data(), N, true, true, packed_b, &selector_config);
    output.fill(-11);
#if defined(MLAS_KLEIDIAI_INT8_OPS_AVAILABLE)
    const bool uses_ops = enabled && ArmKleidiAI::MlasQGemmPackBSizeKleidiAIOps(N, K, true, true) != 0;
    ASSERT_EQ(ArmKleidiAI::MlasQGemmBatchKleidiAIOps(shape, &data, 1, nullptr), uses_ops);
    for (int32_t value : output) EXPECT_EQ(value, uses_ops ? int32_t(6 * K) : -11);
    output.fill(-11);
#endif
    MlasGemmBatch(shape, &data, 1, nullptr);
    for (int32_t value : output) EXPECT_EQ(value, int32_t(6 * K));
  }
}

static size_t QGemmRegistLongExecute() {
  size_t count = 0;

  count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, int8_t, int32_t, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, int8_t, int32_t, true, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, uint8_t, int32_t, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, uint8_t, int32_t, true, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmTest<int8_t, int8_t, int32_t, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmTest<int8_t, int8_t, int32_t, true, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmTest<int8_t, uint8_t, int32_t, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmTest<int8_t, uint8_t, int32_t, true, false>>::RegisterLongExecute();

  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, int8_t, int32_t, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, int8_t, int32_t, true, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, uint8_t, int32_t, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, uint8_t, int32_t, true, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmTest<int8_t, int8_t, int32_t, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmTest<int8_t, int8_t, int32_t, true, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmTest<int8_t, uint8_t, int32_t, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmTest<int8_t, uint8_t, int32_t, true, true>>::RegisterLongExecute();
  }

  return count;
}

static size_t QGemmRegistShortExecute() {
  size_t count = 0;

  count += QgemmShortExecuteTest<uint8_t, int8_t, float, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<uint8_t, uint8_t, float, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<uint8_t, int8_t, int32_t, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<uint8_t, uint8_t, int32_t, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<int8_t, int8_t, float, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<int8_t, int8_t, int32_t, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<int8_t, uint8_t, float, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<int8_t, uint8_t, int32_t, false, false>::RegisterShortExecuteTests();
  count += QgemmS8U8SignedInputTest<false>::RegisterShortExecuteTests();
  if (MlasGemmPackBSize(128, 128, false /*AIsSigned*/, false /*BIsSigned*/, nullptr) > 0) {
    // QGEMM U8U8=float packed tests
    count += QgemmShortExecuteTest<uint8_t, uint8_t, float, true, false>::RegisterShortExecuteTests();
    // QGEMM U8U8=int32_t packed tests
    count += QgemmShortExecuteTest<uint8_t, uint8_t, int32_t, true, false>::RegisterShortExecuteTests();
  }
  if (MlasGemmPackBSize(128, 128, false /*AIsSigned*/, true /*BIsSigned*/, nullptr) > 0) {
    // QGEMM U8S8=float packed tests
    count += QgemmShortExecuteTest<uint8_t, int8_t, float, true, false>::RegisterShortExecuteTests();
    // QGEMM U8S8=int32_t packed tests
    count += QgemmShortExecuteTest<uint8_t, int8_t, int32_t, true, false>::RegisterShortExecuteTests();
  }
  if (MlasGemmPackBSize(128, 128, true /*AIsSigned*/, true /*BIsSigned*/, nullptr) > 0) {
    // QGEMM S8S8=float packed tests
    count += QgemmShortExecuteTest<int8_t, int8_t, float, true, false>::RegisterShortExecuteTests();
    // QGEMM S8S8=int32_t packed tests
    count += QgemmShortExecuteTest<int8_t, int8_t, int32_t, true, false>::RegisterShortExecuteTests();
  }
  if (MlasGemmPackBSize(128, 128, true /*AIsSigned*/, false /*BIsSigned*/, nullptr) > 0) {
    // QGEMM S8U8=float packed tests
    count += QgemmShortExecuteTest<int8_t, uint8_t, float, true, false>::RegisterShortExecuteTests();
    // QGEMM S8U8=int32_t packed tests
    count += QgemmShortExecuteTest<int8_t, uint8_t, int32_t, true, false>::RegisterShortExecuteTests();
  }

  if (GetMlasThreadPool() != nullptr) {
    count += QgemmShortExecuteTest<uint8_t, int8_t, float, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<uint8_t, uint8_t, float, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<uint8_t, int8_t, int32_t, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<uint8_t, uint8_t, int32_t, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<int8_t, int8_t, float, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<int8_t, int8_t, int32_t, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<int8_t, uint8_t, float, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<int8_t, uint8_t, int32_t, false, true>::RegisterShortExecuteTests();
    count += QgemmS8U8SignedInputTest<true>::RegisterShortExecuteTests();
    if (MlasGemmPackBSize(128, 128, false /*AIsSigned*/, false /*BIsSigned*/, nullptr) > 0) {
      count += QgemmShortExecuteTest<uint8_t, uint8_t, float, true, true>::RegisterShortExecuteTests();
      count += QgemmShortExecuteTest<uint8_t, uint8_t, int32_t, true, true>::RegisterShortExecuteTests();
    }
    if (MlasGemmPackBSize(128, 128, false /*AIsSigned*/, true /*BIsSigned*/, nullptr) > 0) {
      count += QgemmShortExecuteTest<uint8_t, int8_t, float, true, true>::RegisterShortExecuteTests();
      count += QgemmShortExecuteTest<uint8_t, int8_t, int32_t, true, true>::RegisterShortExecuteTests();
    }
    if (MlasGemmPackBSize(128, 128, true /*AIsSigned*/, true /*BIsSigned*/, nullptr) > 0) {
      count += QgemmShortExecuteTest<int8_t, int8_t, float, true, true>::RegisterShortExecuteTests();
      count += QgemmShortExecuteTest<int8_t, int8_t, int32_t, true, true>::RegisterShortExecuteTests();
    }
    if (MlasGemmPackBSize(128, 128, true /*AIsSigned*/, false /*BIsSigned*/, nullptr) > 0) {
      count += QgemmShortExecuteTest<int8_t, uint8_t, float, true, true>::RegisterShortExecuteTests();
      count += QgemmShortExecuteTest<int8_t, uint8_t, int32_t, true, true>::RegisterShortExecuteTests();
    }
  }

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? QGemmRegistShortExecute() : QGemmRegistLongExecute();
});
