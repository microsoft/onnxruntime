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
// allocation, so a negative value in the second operand slot must also be
// rejected. Uses a valid `batch_size` and a negative `max_iter` to exercise
// the operand distinctly from `ThrowsOnNegativeVocabSize` above.
TEST(SamplingStateArithmeticTest, ThrowsOnNegativeMaxIter) {
  constexpr int batch_size = 4;
  constexpr int max_iter = -1;
  EXPECT_THROW(SamplingBufferElementCount(batch_size, max_iter), OnnxRuntimeException);
}

// The core bug this PR guards against: a plain `int * int` multiply of
// `batch_size * vocab_size` silently wraps once the mathematical product
// exceeds `INT_MAX`, producing an under-allocated buffer. `SafeMul<size_t>`
// promotes both operands to `size_t` before multiplying, so the helper must
// return the correct non-wrapped product. Reverting the production code to
// `int * int` (or `static_cast<size_t>(int_product)`) fails this test.
TEST(SamplingStateArithmeticTest, ReturnsCorrectProductWhenIntMultiplyWouldOverflow) {
  // 50000 * 50000 = 2.5e9, which exceeds INT_MAX (~2.147e9) and would
  // signed-overflow if computed in `int`, but still fits in a 32-bit
  // `size_t` so the test is portable to 32-bit Windows builds where
  // `size_t` is 32-bit (SIZE_MAX ~ 4.29e9).
  constexpr int large_operand = 50000;
  const size_t expected = static_cast<size_t>(large_operand) * static_cast<size_t>(large_operand);
  EXPECT_EQ(SamplingBufferElementCount(large_operand, large_operand), expected);
}

}  // namespace test
}  // namespace onnxruntime
