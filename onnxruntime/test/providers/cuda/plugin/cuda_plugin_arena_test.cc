// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the CUDA plugin EP arena allocator integration.
// Validates that CreateAllocatorImpl wraps raw allocators in CudaArenaAllocator,
// arena stats are reported, and CUDA device/pinned memory is properly managed.

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/file_util.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

constexpr const char* kCudaPluginEpRegistrationName = "CudaPluginArenaTest";

// Helper: get a stat value as string from allocator stats, or empty if not found.
std::string GetStatValue(const Ort::KeyValuePairs& stats, const char* key) {
  const char* v = stats.GetValue(key);
  return v ? std::string(v) : std::string{};
}

// Helper: get a stat value as int64, returning 0 if not found.
int64_t GetStatInt(const Ort::KeyValuePairs& stats, const char* key) {
  const char* v = stats.GetValue(key);
  return v ? std::stoll(v) : 0;
}

// Resolve the CUDA plugin EP shared library path.
std::filesystem::path GetCudaPluginLibraryPath() {
  return GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_cuda_plugin"));
}

// RAII handle that registers/unregisters the CUDA plugin EP library.
class ScopedCudaPluginRegistration {
 public:
  ScopedCudaPluginRegistration(Ort::Env& env, const char* registration_name)
      : env_(env), name_(registration_name) {
    auto lib_path = GetCudaPluginLibraryPath();
    if (!std::filesystem::exists(lib_path)) {
      available_ = false;
      return;
    }
    env_.RegisterExecutionProviderLibrary(name_.c_str(), lib_path.c_str());
    available_ = true;
  }

  ~ScopedCudaPluginRegistration() {
    if (available_) {
      try {
        env_.UnregisterExecutionProviderLibrary(name_.c_str());
      } catch (...) {
      }
    }
  }

  bool IsAvailable() const { return available_; }

  ScopedCudaPluginRegistration(const ScopedCudaPluginRegistration&) = delete;
  ScopedCudaPluginRegistration& operator=(const ScopedCudaPluginRegistration&) = delete;

 private:
  Ort::Env& env_;
  std::string name_;
  bool available_ = false;
};

// Find the CUDA plugin EP device after registration.
Ort::ConstEpDevice FindCudaPluginDevice(Ort::Env& env) {
  auto ep_devices = env.GetEpDevices();
  for (const auto& device : ep_devices) {
    if (strcmp(device.EpName(), "CudaPluginExecutionProvider") == 0) {
      return device;
    }
  }
  return Ort::ConstEpDevice{nullptr};
}

}  // namespace

class CudaPluginArenaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available.";
    }

    registration_ = std::make_unique<ScopedCudaPluginRegistration>(
        *ort_env, kCudaPluginEpRegistrationName);
    if (!registration_->IsAvailable()) {
      GTEST_SKIP() << "CUDA plugin EP library not found.";
    }

    cuda_device_ = FindCudaPluginDevice(*ort_env);
    if (!cuda_device_) {
      GTEST_SKIP() << "No CUDA plugin EP device found after registration.";
    }
  }

  void TearDown() override {
    registration_.reset();
    cudaDeviceSynchronize();
  }

  std::unique_ptr<ScopedCudaPluginRegistration> registration_;
  Ort::ConstEpDevice cuda_device_{nullptr};
};

// Verify that the shared device allocator is backed by an arena.
TEST_F(CudaPluginArenaTest, DeviceAllocator_IsArena) {
  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  void* p = allocator.Alloc(1024);
  ASSERT_NE(p, nullptr);
  allocator.Free(p);

  auto stats = allocator.GetStats();
  EXPECT_FALSE(GetStatValue(stats, "NumAllocs").empty());
  EXPECT_FALSE(GetStatValue(stats, "NumArenaExtensions").empty());
  EXPECT_GE(GetStatInt(stats, "NumArenaExtensions"), 1);
}

// Verify that CUDA device memory allocated through the arena is usable.
TEST_F(CudaPluginArenaTest, DeviceAllocator_CudaMemoryIsValid) {
  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  constexpr size_t kBytes = 4096;
  void* gpu_ptr = allocator.Alloc(kBytes);
  ASSERT_NE(gpu_ptr, nullptr);
  auto gpu_ptr_guard = std::unique_ptr<void, std::function<void(void*)>>(
      gpu_ptr, [&allocator](void* p) { allocator.Free(p); });

  ASSERT_EQ(cudaSuccess, cudaMemset(gpu_ptr, 0xAB, kBytes));
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  std::vector<unsigned char> host_buf(kBytes);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(host_buf.data(), gpu_ptr, kBytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < kBytes; ++i) {
    ASSERT_EQ(host_buf[i], 0xAB) << "Mismatch at byte " << i;
  }
}

// Verify that multiple alloc/free cycles reuse arena memory (no new extensions).
TEST_F(CudaPluginArenaTest, DeviceAllocator_ArenaReusesMemory) {
  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  constexpr size_t kBytes = 512;

  void* p1 = allocator.Alloc(kBytes);
  ASSERT_NE(p1, nullptr);
  allocator.Free(p1);

  auto stats1 = allocator.GetStats();
  int64_t extensions_after_first = GetStatInt(stats1, "NumArenaExtensions");

  void* p2 = allocator.Alloc(kBytes);
  ASSERT_NE(p2, nullptr);
  allocator.Free(p2);

  auto stats2 = allocator.GetStats();
  int64_t extensions_after_second = GetStatInt(stats2, "NumArenaExtensions");

  EXPECT_EQ(extensions_after_first, extensions_after_second)
      << "Arena should reuse previously freed chunk without extending.";
}

// Verify multiple concurrent allocations from the arena.
TEST_F(CudaPluginArenaTest, DeviceAllocator_MultipleAllocations) {
  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  constexpr int kNumAllocs = 10;
  constexpr size_t kBytes = 2048;
  std::vector<void*> ptrs;
  ptrs.reserve(kNumAllocs);

  for (int i = 0; i < kNumAllocs; ++i) {
    void* p = allocator.Alloc(kBytes);
    ASSERT_NE(p, nullptr) << "Allocation " << i << " failed.";
    ASSERT_EQ(cudaSuccess, cudaMemset(p, static_cast<int>(i & 0xFF), kBytes));
    ptrs.push_back(p);
  }

  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  std::vector<unsigned char> host_buf(kBytes);
  for (int i = 0; i < kNumAllocs; ++i) {
    ASSERT_EQ(cudaSuccess, cudaMemcpy(host_buf.data(), ptrs[i], kBytes, cudaMemcpyDeviceToHost));
    unsigned char expected = static_cast<unsigned char>(i & 0xFF);
    for (size_t j = 0; j < kBytes; ++j) {
      ASSERT_EQ(host_buf[j], expected) << "Mismatch at alloc " << i << " byte " << j;
    }
  }

  for (void* p : ptrs) {
    allocator.Free(p);
  }

  auto stats = allocator.GetStats();
  EXPECT_GE(GetStatInt(stats, "NumAllocs"), kNumAllocs);
}

// Verify that the pinned allocator is also backed by an arena.
TEST_F(CudaPluginArenaTest, PinnedAllocator_IsArena) {
  auto pinned_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_HOST_ACCESSIBLE);
  if (!pinned_memory_info) {
    GTEST_SKIP() << "No pinned memory info available for this device.";
  }

  auto allocator = ort_env->GetSharedAllocator(pinned_memory_info);
  if (!allocator) {
    GTEST_SKIP() << "No shared pinned allocator available.";
  }

  void* p = allocator.Alloc(1024);
  ASSERT_NE(p, nullptr);

  std::memset(p, 0xCD, 1024);
  auto* bytes = static_cast<unsigned char*>(p);
  EXPECT_EQ(bytes[0], 0xCD);
  EXPECT_EQ(bytes[1023], 0xCD);

  allocator.Free(p);

  auto stats = allocator.GetStats();
  EXPECT_GE(GetStatInt(stats, "NumArenaExtensions"), 1);
}

// Verify arena can handle zero-size allocation.
TEST_F(CudaPluginArenaTest, DeviceAllocator_ZeroSizeAlloc) {
  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  void* p = allocator.Alloc(0);
  EXPECT_EQ(p, nullptr);

  allocator.Free(nullptr);
}

// Verify arena handles a large allocation.
TEST_F(CudaPluginArenaTest, DeviceAllocator_LargeAllocation) {
  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  constexpr size_t kLargeSize = 32 * 1024 * 1024;
  void* p = allocator.Alloc(kLargeSize);
  ASSERT_NE(p, nullptr);
  auto p_guard = std::unique_ptr<void, std::function<void(void*)>>(
      p, [&allocator](void* ptr) { allocator.Free(ptr); });

  ASSERT_EQ(cudaSuccess, cudaMemset(p, 0xFF, kLargeSize));
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
}

// Verify GetStats reports InUse correctly during allocation lifecycle.
TEST_F(CudaPluginArenaTest, DeviceAllocator_StatsTrackBytesInUse) {
  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  auto stats_before = allocator.GetStats();
  int64_t inuse_before = GetStatInt(stats_before, "InUse");

  constexpr size_t kBytes = 4096;
  void* p = allocator.Alloc(kBytes);
  ASSERT_NE(p, nullptr);

  auto stats_during = allocator.GetStats();
  int64_t inuse_during = GetStatInt(stats_during, "InUse");
  EXPECT_GT(inuse_during, inuse_before);

  allocator.Free(p);

  auto stats_after = allocator.GetStats();
  int64_t inuse_after = GetStatInt(stats_after, "InUse");
  EXPECT_LE(inuse_after, inuse_before);
}

// Verify arena can be replaced via CreateSharedAllocator with custom config.
// Restores the default allocator at the end to avoid affecting shuffled test ordering.
TEST_F(CudaPluginArenaTest, DeviceAllocator_ReplaceWithCustomConfig) {
  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  Ort::KeyValuePairs allocator_options;
  allocator_options.Add("arena.initial_chunk_size_bytes", "25600");

  auto new_allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      allocator_options);
  ASSERT_NE(new_allocator, nullptr);

  void* p = new_allocator.Alloc(256);
  ASSERT_NE(p, nullptr);
  new_allocator.Free(p);

  auto stats = new_allocator.GetStats();
  int64_t total_allocated = GetStatInt(stats, "TotalAllocated");
  EXPECT_EQ(total_allocated, 25600);

  // Restore the default shared allocator so subsequent tests (under --gtest_shuffle)
  // can call GetSharedAllocator without hitting an empty slot.
  ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator, {});
}

// --- Negative / defensive tests ---

TEST_F(CudaPluginArenaTest, DeviceAllocator_FreeNullptrIsSafe) {
  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  // Free(nullptr) should be a no-op; must not crash.
  allocator.Free(nullptr);
}

TEST_F(CudaPluginArenaTest, DeviceAllocator_InvalidConfigIsRejected) {
  // Providing a non-numeric value for a numeric arena config key should
  // result in an invalid ArenaConfig (IsValid() == false) which causes
  // CreateSharedAllocator to return an error.
  Ort::KeyValuePairs bad_options;
  bad_options.Add("arena.initial_chunk_size_bytes", "not_a_number");

  try {
    auto bad_alloc = ort_env->CreateSharedAllocator(
        cuda_device_, OrtDeviceMemoryType_DEFAULT,
        OrtDeviceAllocator,
        bad_options);
    // If we get here, the allocator was created — that's wrong.
    // Clean up and fail.
    ort_env->CreateSharedAllocator(
        cuda_device_, OrtDeviceMemoryType_DEFAULT,
        OrtDeviceAllocator, {});
    FAIL() << "Expected CreateSharedAllocator to reject invalid config.";
  } catch (const Ort::Exception&) {
    // Expected: invalid config should produce an error.
  }

  // Restore the default shared allocator.
  ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator, {});
}

TEST_F(CudaPluginArenaTest, DeviceAllocator_NegativeConfigIsRejected) {
  // Negative values for arena config should fail validation.
  Ort::KeyValuePairs bad_options;
  bad_options.Add("arena.initial_chunk_size_bytes", "-100");

  try {
    auto bad_alloc = ort_env->CreateSharedAllocator(
        cuda_device_, OrtDeviceMemoryType_DEFAULT,
        OrtDeviceAllocator,
        bad_options);
    ort_env->CreateSharedAllocator(
        cuda_device_, OrtDeviceMemoryType_DEFAULT,
        OrtDeviceAllocator, {});
    FAIL() << "Expected CreateSharedAllocator to reject negative config value.";
  } catch (const Ort::Exception&) {
    // Expected
  }

  ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator, {});
}

TEST_F(CudaPluginArenaTest, DeviceAllocator_MaxMemZeroTreatedAsUnlimited) {
  // arena.max_mem=0 should be treated as unlimited (SIZE_MAX).
  // The arena should create successfully and allow allocations.
  Ort::KeyValuePairs options;
  options.Add("arena.max_mem", "0");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  void* p = allocator.Alloc(1024);
  ASSERT_NE(p, nullptr);
  allocator.Free(p);

  // Restore default.
  ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator, {});
}

TEST_F(CudaPluginArenaTest, DeviceAllocator_ReserveRespectsBudget) {
  // Set a small max_mem budget and verify Reserve returns nullptr
  // when allocation would exceed it.
  Ort::KeyValuePairs options;
  options.Add("arena.max_mem", "65536");
  options.Add("arena.initial_chunk_size_bytes", "4096");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  // Reserve more than the budget should return nullptr.
  // Call through the C function pointer since Ort::Allocator doesn't wrap Reserve.
  OrtAllocator* raw = allocator;
  ASSERT_NE(raw->Reserve, nullptr);
  void* p = raw->Reserve(raw, 128 * 1024);
  EXPECT_EQ(p, nullptr);

  // Restore default.
  ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator, {});
}

}  // namespace test
}  // namespace onnxruntime

#endif  // defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
