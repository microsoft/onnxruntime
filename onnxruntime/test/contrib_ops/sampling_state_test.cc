// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "contrib_ops/cpu/transformers/greedy_search_impl_base.h"

namespace onnxruntime {
namespace test {

namespace {

// Returns a process-wide CPU allocator suitable for the SamplingState buffers.
AllocatorPtr GetCpuAllocator() {
  return CPUAllocator::DefaultInstance();
}

}  // namespace

// Sanity test: SamplingState::Init on the CPU path allocates buffers whose element
// counts match `batch_size * vocab_size` when inputs are well within range.
TEST(SamplingStateTest, Init_AllocatesExpectedSizesOnCpu) {
  contrib::transformers::SamplingState<float> state;
  AllocatorPtr cpu_allocator = GetCpuAllocator();

  constexpr int batch_size = 4;
  constexpr int vocab_size = 32;
  constexpr int max_iter = 0;  // unused on the CPU path
  constexpr int seed = 123;
  constexpr bool is_cuda = false;

  state.Init(cpu_allocator, cpu_allocator, batch_size, vocab_size, max_iter, seed, is_cuda,
             /*stream=*/nullptr);

  const size_t expected = static_cast<size_t>(batch_size) * static_cast<size_t>(vocab_size);
  EXPECT_EQ(state.h_softmaxed_score.size(), expected);
  EXPECT_EQ(state.sorted_scores.size(), expected);
  EXPECT_EQ(state.cumulative_probs.size(), expected);
}

// Regression test for the heap-buffer-overflow fix: prior to the change, an
// `int * int` multiply with a negative `vocab_size` (which can happen before
// upstream validation) would either silently wrap or be turned into an enormous
// size_t via static_cast, producing an under-allocated or absurdly large buffer.
// With SafeInt<size_t> on both operands, constructing SafeInt<size_t> from a
// negative int must throw OnnxRuntimeException ("Integer overflow").
TEST(SamplingStateTest, Init_ThrowsOnNegativeVocabSize) {
  contrib::transformers::SamplingState<float> state;
  AllocatorPtr cpu_allocator = GetCpuAllocator();

  constexpr int batch_size = 4;
  constexpr int vocab_size = -1;  // model-controlled / unvalidated input
  constexpr int max_iter = 0;
  constexpr int seed = 0;
  constexpr bool is_cuda = false;

  EXPECT_THROW(state.Init(cpu_allocator, cpu_allocator, batch_size, vocab_size, max_iter, seed,
                          is_cuda, /*stream=*/nullptr),
               OnnxRuntimeException);
}

// Companion to the above: a negative `batch_size` must also be rejected by the
// SafeInt<size_t> conversion rather than silently wrapping.
TEST(SamplingStateTest, Init_ThrowsOnNegativeBatchSize) {
  contrib::transformers::SamplingState<float> state;
  AllocatorPtr cpu_allocator = GetCpuAllocator();

  constexpr int batch_size = -1;
  constexpr int vocab_size = 32;
  constexpr int max_iter = 0;
  constexpr int seed = 0;
  constexpr bool is_cuda = false;

  EXPECT_THROW(state.Init(cpu_allocator, cpu_allocator, batch_size, vocab_size, max_iter, seed,
                          is_cuda, /*stream=*/nullptr),
               OnnxRuntimeException);
}

// On 32-bit size_t platforms, an `int * int` multiply that overflows INT_MAX
// would previously wrap to a small/negative value and lead to an under-allocated
// buffer. Verify SafeInt<size_t> catches the overflow on such platforms. On
// 64-bit platforms the multiplication fits in size_t and the call would attempt
// a huge allocation, so we skip the check there.
TEST(SamplingStateTest, Init_ThrowsOnVocabBatchProductOverflow) {
  if constexpr (sizeof(size_t) > 4) {
    GTEST_SKIP() << "Product fits in 64-bit size_t; overflow path not reachable here.";
  } else {
    contrib::transformers::SamplingState<float> state;
    AllocatorPtr cpu_allocator = GetCpuAllocator();

    constexpr int batch_size = 1 << 20;
    constexpr int vocab_size = 1 << 20;  // 2^40, overflows 32-bit size_t
    constexpr int max_iter = 0;
    constexpr int seed = 0;
    constexpr bool is_cuda = false;

    EXPECT_THROW(state.Init(cpu_allocator, cpu_allocator, batch_size, vocab_size, max_iter, seed,
                            is_cuda, /*stream=*/nullptr),
                 OnnxRuntimeException);
  }
}

}  // namespace test
}  // namespace onnxruntime
