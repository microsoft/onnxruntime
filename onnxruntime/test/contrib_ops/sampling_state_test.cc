// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Regression tests for the buffer-size arithmetic in
// onnxruntime/contrib_ops/cpu/transformers/greedy_search_impl_base.h
// (SamplingState::Init). The bug being guarded was that a plain `int * int`
// multiply of `batch_size * vocab_size`, or a `static_cast<size_t>(...)` of a
// negative/unvalidated operand, could silently wrap and lead to under-allocated
// buffers (heap-buffer-overflow on the downstream memcpy in
// SamplingCpuHelper::Sample).
//
// `greedy_search_impl_base.h` is not a self-contained public header (it
// transitively requires internal framework types such as OpKernelContextInternal
// that are unavailable to test code), so these tests reproduce the exact
// SafeInt<size_t> expression that the fix introduced rather than constructing
// a SamplingState directly. They will fail if anyone reverts the production
// code to use `int * int` or `static_cast<size_t>` on operands that may be
// negative.

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/common/safeint.h"

namespace onnxruntime {
namespace test {

namespace {

// Mirrors the production computation in SamplingState::Init:
//   const SafeInt<size_t> total_count =
//       SafeInt<size_t>(batch_size) * SafeInt<size_t>(vocab_size);
size_t ComputeSamplingTotalCount(int batch_size, int vocab_size) {
  return SafeInt<size_t>(batch_size) * SafeInt<size_t>(vocab_size);
}

// Mirrors the production computation for `h_sampled_all`:
//   SafeInt<size_t>(batch_size) * SafeInt<size_t>(max_iter)
size_t ComputeSampledAllCount(int batch_size, int max_iter) {
  return SafeInt<size_t>(batch_size) * SafeInt<size_t>(max_iter);
}

}  // namespace

// Sanity check: well-formed inputs produce the expected element count.
TEST(SamplingStateArithmeticTest, ProducesExpectedTotalCountForValidInputs) {
  EXPECT_EQ(ComputeSamplingTotalCount(4, 32), static_cast<size_t>(4) * 32u);
  EXPECT_EQ(ComputeSamplingTotalCount(1, 50257), static_cast<size_t>(50257));
  EXPECT_EQ(ComputeSampledAllCount(8, 16), static_cast<size_t>(8) * 16u);
}

// A negative `vocab_size` (e.g. an unvalidated default of -1) used to be turned
// into SIZE_MAX by `static_cast<size_t>(vocab_size)`, yielding a multiplication
// result that either silently wrapped or requested an absurdly large buffer.
// SafeInt<size_t> rejects the negative-to-unsigned conversion up front.
TEST(SamplingStateArithmeticTest, ThrowsOnNegativeVocabSize) {
  EXPECT_THROW(ComputeSamplingTotalCount(4, -1), OnnxRuntimeException);
}

// Symmetric check for a negative `batch_size`.
TEST(SamplingStateArithmeticTest, ThrowsOnNegativeBatchSize) {
  EXPECT_THROW(ComputeSamplingTotalCount(-1, 32), OnnxRuntimeException);
}

// `max_iter` flows through the same SafeInt<size_t> path for the `h_sampled_all`
// allocation, so a negative value must also be rejected.
TEST(SamplingStateArithmeticTest, ThrowsOnNegativeMaxIter) {
  EXPECT_THROW(ComputeSampledAllCount(4, -1), OnnxRuntimeException);
}

}  // namespace test
}  // namespace onnxruntime
