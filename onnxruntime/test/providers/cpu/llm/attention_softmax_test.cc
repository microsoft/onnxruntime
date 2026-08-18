// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_NO_EXCEPTIONS)

#include <exception>
#include <limits>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/framework/allocator.h"
#include "core/providers/cpu/llm/attention_softmax.h"

namespace onnxruntime {
namespace test {

// Regression test for integer overflow in FP16 softmax allocation.
// ComputeAttentionSoftmaxInplace<MLFloat16> previously used int for N and D, so N*D could overflow int32.
// The fix changed parameters to size_t and uses SafeInt for the multiplication.
//
// This test calls ComputeAttentionSoftmaxInplace<MLFloat16> directly with overflow-triggering dimensions
// (N=46341, D=46341, where N*D > INT_MAX).
// A custom allocator intercepts the Alloc call to verify the requested size is computed correctly with size_t
// arithmetic, without actually allocating the ~8GB buffer.
//
// On 32-bit builds, SafeInt<size_t> will signal an overflow for the requested size.
TEST(AttentionSoftmaxTest, Fp16OverflowAllocation) {
  // Custom exception thrown by the allocator to distinguish it from SafeInt overflow.
  struct AllocationIntercepted : std::exception {
    const char* what() const noexcept override { return "allocation intercepted"; }
  };

  // Custom allocator that records the requested allocation size and throws to avoid actually allocating the
  // (very large) buffer.
  class OverflowCheckAllocator : public IAllocator {
   public:
    OverflowCheckAllocator()
        : IAllocator(OrtMemoryInfo(CPU, OrtDeviceAllocator)) {}
    void* Alloc(size_t size) override {
      last_alloc_size_ = size;
      throw AllocationIntercepted();
    }
    void Free(void*) override {}
    size_t LastAllocSize() const { return last_alloc_size_; }

   private:
    size_t last_alloc_size_ = 0;
  };

  constexpr size_t N = 46341;
  constexpr size_t D = 46341;

  // Verify at compile time that these dimensions would overflow int32.
  static_assert(int64_t{N} * int64_t{D} > int64_t{std::numeric_limits<int>::max()},
                "Test dimensions must cause int32 overflow in N*D");

  auto alloc = std::make_shared<OverflowCheckAllocator>();
  MLFloat16 dummy_score{0.0f};

  // The allocation size must reflect correct size_t arithmetic: N * D * sizeof(float).
  // With the old int parameters, N * D would overflow to a small/negative value, producing a wrong allocation size.
  constexpr uintmax_t expected_allocation_size = uintmax_t{N} * D * sizeof(float);

  if constexpr (expected_allocation_size <= uintmax_t{std::numeric_limits<size_t>::max()}) {
    // Allocation size fits in size_t. The function reaches Alloc, which records the requested size and throws
    // AllocationIntercepted.
    EXPECT_THROW(ComputeAttentionSoftmaxInplace<MLFloat16>(&dummy_score, N, D, nullptr, alloc),
                 AllocationIntercepted);

    EXPECT_EQ(alloc->LastAllocSize(), static_cast<size_t>(expected_allocation_size));
  } else {
    // Allocation size overflows size_t (i.e., in a 32-bit build), so SafeInt<size_t> will throw an exception.
    try {
      ComputeAttentionSoftmaxInplace<MLFloat16>(&dummy_score, N, D, nullptr, alloc);
      FAIL() << "Expected OnnxRuntimeException to be thrown";
    } catch (const OnnxRuntimeException& e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Integer overflow"));
    }
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_NO_EXCEPTIONS)
