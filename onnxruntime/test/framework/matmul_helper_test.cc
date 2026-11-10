// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/providers/cpu/math/matmul_helper.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {

void CheckBatchTransposeOffsets(int64_t batch, int64_t m, int64_t k, int64_t n) {
  for (bool trans_a : {false, true}) {
    for (bool trans_b : {false, true}) {
      for (bool batch_a : {false, true}) {
        for (bool batch_b : {false, true}) {
          const int64_t a_first = trans_a ? k : m;
          const int64_t a_last = trans_a ? m : k;
          const int64_t b_first = trans_b ? n : k;
          const int64_t b_last = trans_b ? k : n;
          const TensorShape a = batch_a ? TensorShape({a_first, batch, a_last})
                                        : TensorShape({batch, a_first, a_last});
          const TensorShape b = batch_b ? TensorShape({b_first, batch, b_last})
                                        : TensorShape({batch, b_first, b_last});
          for (bool fill_offsets : {false, true}) {
            SCOPED_TRACE(MakeString("A=", a, ", B=", b, ", transA=", trans_a, ", transB=", trans_b,
                                    ", transBatchA=", batch_a, ", transBatchB=", batch_b,
                                    ", fill_offsets=", fill_offsets));
            MatMulComputeHelper helper;
            ASSERT_STATUS_OK(helper.Compute(a, b, trans_a, trans_b, batch_a, batch_b, fill_offsets));
            ASSERT_EQ(helper.OutputShape(), TensorShape({batch, m, n}));
            if (!fill_offsets) {
              helper.FillOffsets();
            }
            ASSERT_EQ(helper.LeftOffsets().size(), static_cast<size_t>(batch));
            ASSERT_EQ(helper.RightOffsets().size(), static_cast<size_t>(batch));
            ASSERT_EQ(helper.OutputOffsets().size(), static_cast<size_t>(batch));
            for (size_t i = 0; i < static_cast<size_t>(batch); ++i) {
              const auto a_stride = m == 0 || k == 0 ? 0 : (batch_a ? a_last : m * k);
              const auto b_stride = k == 0 || n == 0 ? 0 : (batch_b ? b_last : k * n);
              EXPECT_EQ(helper.LeftOffsets()[i], i * static_cast<size_t>(a_stride));
              EXPECT_EQ(helper.RightOffsets()[i], i * static_cast<size_t>(b_stride));
              EXPECT_EQ(helper.OutputOffsets()[i], i * static_cast<size_t>(m * n));
            }
          }
        }
      }
    }
  }
}

TEST(MatMulComputeHelperTest, BatchTransposeEmptyReduction) {
  CheckBatchTransposeOffsets(3, 2, 0, 4);
}

TEST(MatMulComputeHelperTest, BatchTransposeEmptyOutput) {
  CheckBatchTransposeOffsets(3, 0, 2, 4);
  CheckBatchTransposeOffsets(3, 2, 4, 0);
}

TEST(MatMulComputeHelperTest, BatchTransposeEmptyBatch) {
  CheckBatchTransposeOffsets(0, 2, 3, 4);
}

TEST(MatMulComputeHelperTest, BatchTransposeNonempty) {
  CheckBatchTransposeOffsets(3, 2, 3, 4);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
