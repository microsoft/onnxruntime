// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// registration/selection is only supported on windows as there's no device discovery on other platforms
#ifdef _WIN32

#include <algorithm>
#include <gsl/gsl>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "core/framework/allocator.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "test/autoep/test_autoep_utils.h"
#include "test/shared_lib/utils.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

namespace {
struct DummyAllocator : OrtAllocator {
  DummyAllocator(const OrtMemoryInfo* mem_info)
      : memory_info{mem_info} {
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;      // no special reserve logic and most likely unnecessary unless you have your own arena
    GetStats = nullptr;       // this can be set to nullptr if not implemented
    AllocOnStream = nullptr;  // optional
  }

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size) {
    auto& impl = *static_cast<DummyAllocator*>(this_);
    ++impl.stats.num_allocs;
    impl.stats.max_alloc_size = std::max<int64_t>(size, impl.stats.max_alloc_size);

    return malloc(size);
  }

  static void ORT_API_CALL FreeImpl(struct OrtAllocator* /*this_*/, void* p) {
    return free(p);
  }

  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    const DummyAllocator& impl = *static_cast<const DummyAllocator*>(this_);
    return impl.memory_info;
  }

 private:
  const OrtMemoryInfo* memory_info;
  AllocatorStats stats{};
};
}  // namespace

// validate CreateSharedAllocator allows adding an arena to the shared allocator
TEST(SharedAllocators, AddArenaToSharedAllocator) {
  RegisteredEpDeviceUniquePtr example_ep;
  Utils::RegisterAndGetExampleEp(*ort_env, example_ep);

  Ort::ConstEpDevice example_ep_device{example_ep.get()};

  auto ep_memory_info = example_ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);

  // validate there is a shared allocator
  auto allocator = ort_env->GetSharedAllocator(ep_memory_info);
  ASSERT_NE(allocator, nullptr);

  // call CreateSharedAllocator to replace with arena based allocator. arena is configured with kvps
  Ort::KeyValuePairs allocator_options;
  auto initial_chunk_size = "25600";  // arena allocates in 256 byte amounts
  allocator_options.Add(OrtArenaCfg::ConfigKeyNames::InitialChunkSizeBytes, initial_chunk_size);

  allocator = ort_env->CreateSharedAllocator(example_ep.get(), OrtDeviceMemoryType_DEFAULT,
                                             // allocator is internally added by EP.
                                             // OrtArenaAllocator can only be used for the internal BFCArena
                                             OrtDeviceAllocator,
                                             allocator_options);

  // first allocation should init the arena to the initial chunk size
  void* mem = allocator.Alloc(16);
  allocator.Free(mem);

  // stats should prove the arena was used
  auto allocator_stats = allocator.GetStats();

  using ::testing::Contains;
  using ::testing::Pair;
  const auto& stats = static_cast<OrtKeyValuePairs*>(allocator_stats)->Entries();
  EXPECT_THAT(stats, Contains(Pair("NumAllocs", "1")));
  EXPECT_THAT(stats, Contains(Pair("NumArenaExtensions", "1")));
  EXPECT_THAT(stats, Contains(Pair("TotalAllocated", initial_chunk_size)));

  // optional. ORT owns the allocator but we want to test the release implementation
  ort_env->ReleaseSharedAllocator(example_ep.get(), OrtDeviceMemoryType_DEFAULT);
}

TEST(SharedAllocators, GetSharedAllocator) {
  // default CPU allocator should be available.
  // create a memory info with a different name to validate the shared allocator lookup ignores the name
  auto test_cpu_memory_info = Ort::MemoryInfo("dummy", OrtMemoryInfoDeviceType_CPU, 0, 0,
                                              OrtDeviceMemoryType_DEFAULT, 0, OrtDeviceAllocator);

  const auto get_allocator_and_check_name = [&](const std::string& expected_name) {
    auto allocator = ort_env->GetSharedAllocator(test_cpu_memory_info);
    ASSERT_NE(allocator, nullptr);

    auto ort_cpu_memory_info = allocator.GetInfo();
    auto allocator_name = ort_cpu_memory_info.GetAllocatorName();
    ASSERT_EQ(expected_name, allocator_name);  // Default ORT CPU allocator
  };

  // check we get the default ORT CPU allocator initially
  get_allocator_and_check_name(onnxruntime::CPU);

  // register custom allocator and make sure that is accessible by exact match
  DummyAllocator dummy_alloc{test_cpu_memory_info};
  ort_env->RegisterAllocator(&dummy_alloc);

  // GetSharedAllocator should now match the custom allocator
  get_allocator_and_check_name("dummy");

  // unregister custom allocator
  ort_env->UnregisterAllocator(test_cpu_memory_info);

  // there should always be a CPU allocator available
  get_allocator_and_check_name(onnxruntime::CPU);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // _WIN32
