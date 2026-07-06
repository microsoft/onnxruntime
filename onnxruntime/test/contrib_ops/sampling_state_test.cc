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
// The production computation lives in `sampling_buffer_element_count.h` as
// `SamplingBufferElementCount`, which `SamplingState::Init` calls directly.
// These tests call the same helper, so reverting the production code to
// `int * int` or `static_cast<size_t>` on operands that may be negative will
// fail these tests.

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "contrib_ops/cpu/transformers/sampling_buffer_element_count.h"

namespace onnxruntime {
namespace test {

using contrib::transformers::SamplingBufferElementCount;

// Sanity check: well-formed inputs produce the expected element count.
TEST(SamplingStateArithmeticTest, ProducesExpectedTotalCountForValidInputs) {
  EXPECT_EQ(SamplingBufferElementCount(4, 32), static_cast<size_t>(4) * 32u);
  EXPECT_EQ(SamplingBufferElementCount(1, 50257), static_cast<size_t>(50257));
  // `h_sampled_all` uses the same helper with `max_iter` as the second operand.
  EXPECT_EQ(SamplingBufferElementCount(8, 16), static_cast<size_t>(8) * 16u);
}

// A negative `vocab_size` (e.g. an unvalidated default of -1) used to be turned
// into SIZE_MAX by `static_cast<size_t>(vocab_size)`, yielding a multiplication
// result that either silently wrapped or requested an absurdly large buffer.
// SafeInt<size_t> rejects the negative-to-unsigned conversion up front.
TEST(SamplingStateArithmeticTest, ThrowsOnNegativeVocabSize) {
  EXPECT_THROW(SamplingBufferElementCount(4, -1), OnnxRuntimeException);
}

// Symmetric check for a negative `batch_size`.
TEST(SamplingStateArithmeticTest, ThrowsOnNegativeBatchSize) {
  EXPECT_THROW(SamplingBufferElementCount(-1, 32), OnnxRuntimeException);
}

// `max_iter` flows through the same helper for the `h_sampled_all`
// allocation, so a negative value must also be rejected.
TEST(SamplingStateArithmeticTest, ThrowsOnNegativeMaxIter) {
  EXPECT_THROW(SamplingBufferElementCount(4, -1), OnnxRuntimeException);
}

}  // namespace test
}  // namespace onnxruntime
