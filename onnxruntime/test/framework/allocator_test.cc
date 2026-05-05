// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <absl/base/config.h>

#include "core/framework/allocator.h"
#include "core/framework/allocator_utils.h"
#include "core/session/allocator_adapters.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/ort_apis.h"

#include "test/unittest_util/framework_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
TEST(AllocatorTest, CPUAllocatorTest) {
  auto cpu_arena = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  ASSERT_STREQ(cpu_arena->Info().name.c_str(), CPU);
  EXPECT_EQ(cpu_arena->Info().device.Id(), 0);

  const auto expected_allocator_type = DoesCpuAllocatorSupportArenaUsage()
                                           ? OrtAllocatorType::OrtArenaAllocator
                                           : OrtAllocatorType::OrtDeviceAllocator;
  EXPECT_EQ(cpu_arena->Info().alloc_type, expected_allocator_type);

  size_t size = 1024;
  auto bytes = cpu_arena->Alloc(size);
  EXPECT_TRUE(bytes);
  // test the bytes are ok for read/write
  memset(bytes, -1, 1024);

  EXPECT_EQ(*((int*)bytes), -1);
  cpu_arena->Free(bytes);
  // todo: test the used / max api.
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26400)
#endif
// helper class to validate values in Alloc and Free calls made via IAllocator::MakeUniquePtr
class TestAllocator : public IAllocator {
 public:
  TestAllocator(size_t expected_size)
      : IAllocator(OrtMemoryInfo("test", OrtDeviceAllocator)),
        expected_size_{expected_size} {
  }

  void* Reserve(size_t size) override {
    reserve_is_called_ = true;
    return Alloc(size);
  }
  void* Alloc(size_t size) override {
    EXPECT_EQ(size, expected_size_);
    // return a pointer to the expected size in the result.
    // this isn't valid as a real allocator would return memory of the correct size,
    // however the unit test won't be using the memory and via this mechanism we can validate
    // that the Free argument matches.
    size_t* result = new size_t(size);
    return result;
  }

  void Free(void* p) override {
    // the IAllocatorUniquePtr should be calling this with the contents of what was returned from the Alloc
    size_t* p_sizet = (size_t*)p;
    EXPECT_EQ(*p_sizet, expected_size_);
    delete p_sizet;
  }

  bool reserve_is_called_ = false;

 private:
  size_t expected_size_;
};

// test that IAllocator::MakeUniquePtr allocates buffers of the expected size
TEST(AllocatorTest, MakeUniquePtrTest) {
  // test float creates buffer of size * sizeof(float)
  size_t num_floats = 16;
  for (bool use_reserve : {true, false}) {
    // create allocator that will check the call to Alloc matches the expected size
    auto allocator = std::make_shared<TestAllocator>(num_floats * sizeof(float));
    IAllocatorUniquePtr<float> float_ptr = IAllocator::MakeUniquePtr<float>(allocator, num_floats, use_reserve);
    float_ptr = nullptr;  // reset so TestAllocator.Free is called here
    ASSERT_EQ(allocator->reserve_is_called_, use_reserve);
    // void should create buffer of size 16 for void*
    // Create new TestAllocator to validate that.
    allocator = std::make_shared<TestAllocator>(16);
    auto void_ptr = IAllocator::MakeUniquePtr<void>(allocator, 16);
    void_ptr = nullptr;
  }
}

TEST(AllocatorTest, TestOverflowChecks) {
  size_t size;
  size_t element_size = sizeof(float);
  size_t num_elements = std::numeric_limits<size_t>::max() / element_size;

  EXPECT_TRUE(IAllocator::CalcMemSizeForArrayWithAlignment<0>(num_elements, element_size, &size));
  EXPECT_FALSE(IAllocator::CalcMemSizeForArrayWithAlignment<0>(num_elements + 1, element_size, &size));

  // we need to add kAllocAlignment-1 bytes to apply the alignment mask, so num_elements * element_size must be kAllocAlignment-bytes short of the max
  EXPECT_TRUE(IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(num_elements - (kAllocAlignment / element_size), element_size, &size));
  EXPECT_FALSE(IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(num_elements, element_size, &size));

  element_size = std::numeric_limits<size_t>::max() / 8;
  num_elements = 8;

  EXPECT_TRUE(IAllocator::CalcMemSizeForArrayWithAlignment<0>(num_elements, element_size, &size));
  EXPECT_FALSE(IAllocator::CalcMemSizeForArrayWithAlignment<0>(num_elements + 1, element_size, &size));

  // we need to add kAllocAlignment-1 bytes to apply the alignment mask, so num_elements * element_size must be kAllocAlignment-bytes short of the max
  EXPECT_TRUE(IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(num_elements, element_size - (kAllocAlignment / num_elements), &size));
  EXPECT_FALSE(IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(num_elements, element_size, &size));
}

// --- AsArena / SafeArenaCast tests ---

TEST(AllocatorTest, AsArena_ReturnsNullForNonArena) {
  auto cpu_allocator = std::make_shared<CPUAllocator>();
  EXPECT_EQ(cpu_allocator->AsArena(), nullptr);
  EXPECT_EQ(static_cast<const IAllocator*>(cpu_allocator.get())->AsArena(), nullptr);
  EXPECT_EQ(IArena::SafeArenaCast(cpu_allocator.get()), nullptr);
}

TEST(AllocatorTest, AsArena_ReturnsNonNullForArena) {
  if (!DoesCpuAllocatorSupportArenaUsage()) {
    GTEST_SKIP() << "CPU arena not enabled in this build";
  }
  auto cpu_arena = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  EXPECT_NE(cpu_arena->AsArena(), nullptr);
  EXPECT_EQ(cpu_arena->AsArena(), IArena::SafeArenaCast(cpu_arena.get()));
}

TEST(AllocatorTest, SafeArenaCast_NullInput) {
  EXPECT_EQ(IArena::SafeArenaCast(nullptr), nullptr);
}

// --- IArenaImplWrappingOrtAllocator tests ---

namespace {
// Minimal OrtAllocator with arena-like Shrink support for unit testing.
struct MockArenaOrtAllocator : OrtAllocator {
  int alloc_count = 0;
  int free_count = 0;
  int reserve_count = 0;
  int shrink_count = 0;
  bool shrink_should_fail = false;

  static OrtMemoryInfo mem_info_;

  MockArenaOrtAllocator() {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = ReserveImpl;
    GetStats = GetStatsImpl;
    AllocOnStream = nullptr;
    Shrink = ShrinkImpl;
  }

  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_, size_t size) {
    auto& self = *static_cast<MockArenaOrtAllocator*>(this_);
    self.alloc_count++;
    if (size == 0) return nullptr;
    return malloc(size);
  }

  static void ORT_API_CALL FreeImpl(OrtAllocator* this_, void* p) {
    auto& self = *static_cast<MockArenaOrtAllocator*>(this_);
    self.free_count++;
    free(p);
  }

  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* /*this_*/) {
    return &mem_info_;
  }

  static void* ORT_API_CALL ReserveImpl(OrtAllocator* this_, size_t size) {
    auto& self = *static_cast<MockArenaOrtAllocator*>(this_);
    self.reserve_count++;
    if (size == 0) return nullptr;
    return malloc(size);
  }

  static OrtStatusPtr ORT_API_CALL GetStatsImpl(const OrtAllocator* this_, OrtKeyValuePairs** out) noexcept {
    auto& self = *static_cast<const MockArenaOrtAllocator*>(this_);
    auto kvp = std::make_unique<OrtKeyValuePairs>();
    kvp->CopyFromMap(std::map<std::string, std::string>{
        {"NumAllocs", std::to_string(self.alloc_count)},
        {"NumArenaShrinkages", std::to_string(self.shrink_count)},
        {"InUse", "0"},
        {"TotalAllocated", "0"},
        {"MaxInUse", "0"},
        {"Limit", "0"},
        {"NumReserves", std::to_string(self.reserve_count)},
        {"NumArenaExtensions", "0"},
        {"MaxAllocSize", "0"},
    });
    *out = kvp.release();
    return nullptr;
  }

  static OrtStatusPtr ORT_API_CALL ShrinkImpl(OrtAllocator* this_) noexcept {
    auto& self = *static_cast<MockArenaOrtAllocator*>(this_);
    if (self.shrink_should_fail) {
      return OrtApis::CreateStatus(ORT_EP_FAIL, "Mock shrink failure");
    }
    self.shrink_count++;
    return nullptr;
  }
};

OrtMemoryInfo MockArenaOrtAllocator::mem_info_{"MockArena", OrtAllocatorType::OrtDeviceAllocator};
}  // namespace

TEST(AllocatorTest, IArenaWrapper_AsArenaReturnsThis) {
  MockArenaOrtAllocator mock;
  auto wrapper = std::make_shared<IArenaImplWrappingOrtAllocator>(
      OrtAllocatorUniquePtr(&mock, [](OrtAllocator*) {}));

  EXPECT_NE(wrapper->AsArena(), nullptr);
  EXPECT_EQ(wrapper->AsArena(), wrapper.get());
  EXPECT_EQ(IArena::SafeArenaCast(wrapper.get()), wrapper.get());
}

TEST(AllocatorTest, IArenaWrapper_AllocFreeReserve) {
  MockArenaOrtAllocator mock;
  auto wrapper = std::make_shared<IArenaImplWrappingOrtAllocator>(
      OrtAllocatorUniquePtr(&mock, [](OrtAllocator*) {}));

  void* p = wrapper->Alloc(256);
  EXPECT_NE(p, nullptr);
  EXPECT_EQ(mock.alloc_count, 1);

  wrapper->Free(p);
  EXPECT_EQ(mock.free_count, 1);

  void* r = wrapper->Reserve(512);
  EXPECT_NE(r, nullptr);
  EXPECT_EQ(mock.reserve_count, 1);
  wrapper->Free(r);
}

TEST(AllocatorTest, IArenaWrapper_ShrinkForwards) {
  MockArenaOrtAllocator mock;
  auto wrapper = std::make_shared<IArenaImplWrappingOrtAllocator>(
      OrtAllocatorUniquePtr(&mock, [](OrtAllocator*) {}));

  auto status = wrapper->Shrink();
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(mock.shrink_count, 1);
}

TEST(AllocatorTest, IArenaWrapper_ShrinkPropagatesError) {
  MockArenaOrtAllocator mock;
  mock.shrink_should_fail = true;
  auto wrapper = std::make_shared<IArenaImplWrappingOrtAllocator>(
      OrtAllocatorUniquePtr(&mock, [](OrtAllocator*) {}));

  auto status = wrapper->Shrink();
  EXPECT_FALSE(status.IsOK());
}

TEST(AllocatorTest, IArenaWrapper_GetStatsRoundTrip) {
  MockArenaOrtAllocator mock;
  // Do some operations to populate counters
  void* p = MockArenaOrtAllocator::AllocImpl(&mock, 100);
  MockArenaOrtAllocator::FreeImpl(&mock, p);
  void* r = MockArenaOrtAllocator::ReserveImpl(&mock, 200);
  MockArenaOrtAllocator::FreeImpl(&mock, r);
  MockArenaOrtAllocator::ShrinkImpl(&mock);

  auto wrapper = std::make_shared<IArenaImplWrappingOrtAllocator>(
      OrtAllocatorUniquePtr(&mock, [](OrtAllocator*) {}));

  AllocatorStats stats{};
  wrapper->GetStats(&stats);
  EXPECT_EQ(stats.num_allocs, 1);
  EXPECT_EQ(stats.num_reserves, 1);
  EXPECT_EQ(stats.num_arena_shrinkages, 1);
}

TEST(AllocatorTest, IArenaWrapper_ReleaseStreamBuffersIsNoop) {
  MockArenaOrtAllocator mock;
  auto wrapper = std::make_shared<IArenaImplWrappingOrtAllocator>(
      OrtAllocatorUniquePtr(&mock, [](OrtAllocator*) {}));

  // Should not crash — ReleaseStreamBuffers is inherited no-op from IArena
  wrapper->ReleaseStreamBuffers(nullptr);
}

}  // namespace test
}  // namespace onnxruntime
