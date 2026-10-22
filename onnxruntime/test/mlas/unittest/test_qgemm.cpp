#include <array>

#include "test_qgemm.h"
#include "test_qgemm_fixture.h"

namespace {

class QGemmOffsetOutputProcessor final : public MLAS_QGEMM_OUTPUT_PROCESSOR {
 public:
  QGemmOffsetOutputProcessor(int32_t* output, size_t stride, int32_t offset)
      : output_(output), stride_(stride), offset_(offset) {}

  void Process(const int32_t* input, size_t start_m, size_t start_n,
               size_t count_m, size_t count_n, size_t ldc) const override {
    for (size_t m = start_m; m < start_m + count_m; ++m) {
      for (size_t n = start_n; n < start_n + count_n; ++n) {
        output_[m * stride_ + n] = input[m * ldc + n] + offset_;
      }
    }
  }

 private:
  int32_t* output_;
  size_t stride_;
  int32_t offset_;
};

void CheckQGemmReduction(size_t k, bool packed) {
  constexpr size_t M = 2;
  constexpr size_t N = 3;
  constexpr size_t Ldc = 5;
  constexpr size_t OutputStride = 4;
  constexpr size_t BatchCount = 2;
  constexpr int32_t Padding = -777;
  constexpr int32_t OutputSentinel = -999;
  const uint8_t zero_point_b = 11;
  const size_t packed_size = packed ? MlasGemmPackBSize(N, k, false, false, nullptr) : 0;
  if (packed && packed_size == 0) {
    GTEST_SKIP() << "Packed U8U8 QGEMM is not available";
  }

  for (bool accumulate : {false, true}) {
    for (bool use_processor : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "K=" << k << ", packed=" << packed
                                        << ", accumulate=" << accumulate
                                        << ", output processor=" << use_processor);
      MLAS_GEMM_QUANT_SHAPE_PARAMS shape;
      shape.M = M;
      shape.N = N;
      shape.K = k;
      shape.IsAccumulateMode = accumulate;

      std::array<std::vector<uint8_t>, BatchCount> a;
      std::array<std::vector<uint8_t>, BatchCount> b;
      std::array<std::vector<int32_t>, BatchCount> c;
      std::array<std::vector<int32_t>, BatchCount> output;
      MatrixGuardBuffer<uint8_t> packed_buffers[BatchCount];
      std::array<MLAS_GEMM_QUANT_DATA_PARAMS, BatchCount> params;
      std::vector<QGemmOffsetOutputProcessor> processors;
      processors.reserve(BatchCount);

      for (size_t batch = 0; batch < BatchCount; ++batch) {
        a[batch].assign(M * std::max(k, size_t{1}), static_cast<uint8_t>(5 + batch));
        b[batch].assign(N * std::max(k, size_t{1}), static_cast<uint8_t>(7 + batch));
        c[batch].assign(M * Ldc, Padding);
        output[batch].assign(M * OutputStride, OutputSentinel);
        for (size_t m = 0; m < M; ++m) {
          for (size_t n = 0; n < N; ++n) {
            c[batch][m * Ldc + n] = static_cast<int32_t>(100 + batch * 17 + m * N + n);
          }
        }

        auto& data = params[batch];
        data.A = a[batch].data();
        data.lda = k;
        data.ZeroPointA = 3;
        data.ZeroPointB = &zero_point_b;
        data.C = c[batch].data();
        data.ldc = Ldc;
        data.ldb = N;
        if (packed) {
          auto* packed_b = packed_buffers[batch].GetBuffer(packed_size);
          MlasGemmPackB(N, k, b[batch].data(), N, false, false, packed_b);
          data.B = packed_b;
          data.BIsPacked = true;
        } else {
          data.B = b[batch].data();
        }
        if (use_processor) {
          processors.emplace_back(output[batch].data(), OutputStride,
                                  static_cast<int32_t>(13 + batch * 7));
          data.OutputProcessor = &processors.back();
        }
      }

      MlasGemmBatch(shape, params.data(), BatchCount, nullptr);

      for (size_t batch = 0; batch < BatchCount; ++batch) {
        const int32_t dot_product = static_cast<int32_t>(k) *
                                    (static_cast<int32_t>(5 + batch) - 3) *
                                    (static_cast<int32_t>(7 + batch) - zero_point_b);
        for (size_t m = 0; m < M; ++m) {
          for (size_t n = 0; n < N; ++n) {
            const int32_t initial = static_cast<int32_t>(100 + batch * 17 + m * N + n);
            const int32_t expected = dot_product + (accumulate ? initial : 0);
            ASSERT_EQ(c[batch][m * Ldc + n], expected) << batch << ", " << m << ", " << n;
            const int32_t expected_output = use_processor
                                                ? expected + static_cast<int32_t>(13 + batch * 7)
                                                : OutputSentinel;
            ASSERT_EQ(output[batch][m * OutputStride + n], expected_output);
          }
          for (size_t n = N; n < Ldc; ++n) {
            ASSERT_EQ(c[batch][m * Ldc + n], Padding);
          }
          ASSERT_EQ(output[batch][m * OutputStride + N], OutputSentinel);
        }
      }
    }
  }
}

}  // namespace

TEST(QGemmEmptyReduction, Unpacked) { CheckQGemmReduction(0, false); }
TEST(QGemmEmptyReduction, Packed) { CheckQGemmReduction(0, true); }
TEST(QGemmEmptyReduction, NonEmptyUnpackedControl) { CheckQGemmReduction(1, false); }
TEST(QGemmEmptyReduction, NonEmptyPackedControl) { CheckQGemmReduction(1, true); }

TEST(QGemmPackB, SizeDoesNotIncludeDynamicQGemmPacking) {
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG selector_config{};
  selector_config.use_kleidiai = false;
  const size_t standard_packed_size = MlasGemmPackBSize(
      1024, 1024, true /* AIsSigned */, true /* BIsSigned */, &selector_config);

  selector_config.use_kleidiai = true;
  const size_t packed_size_with_kleidiai_enabled = MlasGemmPackBSize(
      1024, 1024, true /* AIsSigned */, true /* BIsSigned */, &selector_config);

  ASSERT_EQ(packed_size_with_kleidiai_enabled, standard_packed_size);
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
