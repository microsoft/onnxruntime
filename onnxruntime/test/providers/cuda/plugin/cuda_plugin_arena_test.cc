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

  // RAII cleanup: free all pointers on early exit.
  auto cleanup = [&]() {
    for (void* ptr : ptrs) allocator.Free(ptr);
  };
  auto guard = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) { cleanup(); });

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

  // Guard will free remaining pointers; clear to avoid double-free.
  cleanup();
  ptrs.clear();

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
    ORT_IGNORE_RETURN_VALUE(ort_env->CreateSharedAllocator(
        cuda_device_, OrtDeviceMemoryType_DEFAULT,
        OrtDeviceAllocator,
        bad_options));
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
    ORT_IGNORE_RETURN_VALUE(ort_env->CreateSharedAllocator(
        cuda_device_, OrtDeviceMemoryType_DEFAULT,
        OrtDeviceAllocator,
        bad_options));
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
  void* p = allocator.Reserve(128 * 1024);
  EXPECT_EQ(p, nullptr);

  // Restore default.
  ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator, {});
}

// ---------------------------------------------------------------------------
// CudaMempoolOrtAllocator tests
// ---------------------------------------------------------------------------

TEST_F(CudaPluginArenaTest, Mempool_BasicAllocFree) {
  // Enable mempool and verify basic alloc/free roundtrip on device memory.
  Ort::KeyValuePairs options;
  options.Add("arena.use_cuda_mempool", "1");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  // RAII: restore default allocator on any exit path.
  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  constexpr size_t kBytes = 4096;
  void* p = allocator.Alloc(kBytes);
  ASSERT_NE(p, nullptr);
  auto p_guard = std::unique_ptr<void, std::function<void(void*)>>(
      p, [&allocator](void* ptr) { allocator.Free(ptr); });

  // Verify the memory is usable on the GPU.
  ASSERT_EQ(cudaSuccess, cudaMemset(p, 0xAB, kBytes));
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  std::vector<unsigned char> host_buf(kBytes);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(host_buf.data(), p, kBytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < kBytes; ++i) {
    ASSERT_EQ(host_buf[i], 0xAB) << "Mismatch at byte " << i;
  }
}

TEST_F(CudaPluginArenaTest, Mempool_MultipleAllocations) {
  Ort::KeyValuePairs options;
  options.Add("arena.use_cuda_mempool", "1");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  // RAII: restore default allocator on any exit path.
  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  constexpr int kNumAllocs = 8;
  constexpr size_t kBytes = 2048;
  std::vector<void*> ptrs;
  ptrs.reserve(kNumAllocs);

  auto cleanup_ptrs = [&]() {
    for (void* ptr : ptrs) allocator.Free(ptr);
  };
  auto ptrs_guard = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) { cleanup_ptrs(); });

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

  // Explicit cleanup; clear to prevent guard double-free.
  cleanup_ptrs();
  ptrs.clear();
}

TEST_F(CudaPluginArenaTest, Mempool_StatsAreReported) {
  Ort::KeyValuePairs options;
  options.Add("arena.use_cuda_mempool", "1");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  constexpr size_t kBytes = 1024;
  void* p = allocator.Alloc(kBytes);
  ASSERT_NE(p, nullptr);
  auto p_guard = std::unique_ptr<void, std::function<void(void*)>>(
      p, [&allocator](void* ptr) { allocator.Free(ptr); });

  auto stats = allocator.GetStats();
  EXPECT_GE(GetStatInt(stats, "NumAllocs"), 1);
  EXPECT_GT(GetStatInt(stats, "InUse"), 0);

  p_guard.reset();  // Free p

  auto stats_after = allocator.GetStats();
  EXPECT_EQ(GetStatInt(stats_after, "InUse"), 0);
}

TEST_F(CudaPluginArenaTest, Mempool_ZeroSizeAllocReturnsNull) {
  Ort::KeyValuePairs options;
  options.Add("arena.use_cuda_mempool", "1");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  void* p = allocator.Alloc(0);
  EXPECT_EQ(p, nullptr);

  // Free(nullptr) should be safe.
  allocator.Free(nullptr);
}

TEST_F(CudaPluginArenaTest, Mempool_LargeAllocation) {
  Ort::KeyValuePairs options;
  options.Add("arena.use_cuda_mempool", "1");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  // RAII: restore default allocator on any exit path.
  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  constexpr size_t kLargeSize = 32 * 1024 * 1024;  // 32 MB
  void* p = allocator.Alloc(kLargeSize);
  ASSERT_NE(p, nullptr);
  auto p_guard = std::unique_ptr<void, std::function<void(void*)>>(
      p, [&allocator](void* ptr) { allocator.Free(ptr); });

  ASSERT_EQ(cudaSuccess, cudaMemset(p, 0xFF, kLargeSize));
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
}

TEST_F(CudaPluginArenaTest, Mempool_CustomReleaseThreshold) {
  // Verify mempool can be created with a custom release threshold.
  Ort::KeyValuePairs options;
  options.Add("arena.use_cuda_mempool", "1");
  options.Add("arena.cuda_mempool_release_threshold", "1048576");  // 1 MB

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  void* p = allocator.Alloc(4096);
  ASSERT_NE(p, nullptr);
  allocator.Free(p);
}

TEST_F(CudaPluginArenaTest, Mempool_FreeNullptrIsSafe) {
  Ort::KeyValuePairs options;
  options.Add("arena.use_cuda_mempool", "1");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  // Must not crash.
  allocator.Free(nullptr);
}

// ---------------------------------------------------------------------------
// Arena config coverage tests
// ---------------------------------------------------------------------------

// Verify kSameAsRequested extend strategy allocates exactly the requested amount.
TEST_F(CudaPluginArenaTest, DeviceAllocator_SameAsRequestedStrategy) {
  Ort::KeyValuePairs options;
  options.Add("arena.extend_strategy", "1");  // kSameAsRequested
  options.Add("arena.initial_chunk_size_bytes", "4096");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  void* p = allocator.Alloc(2048);
  ASSERT_NE(p, nullptr);
  allocator.Free(p);

  auto stats = allocator.GetStats();
  // kSameAsRequested: each extension allocates exactly what's needed (rounded to kMinAllocationSize).
  EXPECT_GE(GetStatInt(stats, "NumArenaExtensions"), 1);
}

// Verify max_dead_bytes_per_chunk config is accepted and arena works.
TEST_F(CudaPluginArenaTest, DeviceAllocator_MaxDeadBytesConfig) {
  Ort::KeyValuePairs options;
  options.Add("arena.max_dead_bytes_per_chunk", "1024");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  // A small max_dead_bytes forces more aggressive splitting.
  void* p1 = allocator.Alloc(512);
  ASSERT_NE(p1, nullptr);
  void* p2 = allocator.Alloc(256);
  ASSERT_NE(p2, nullptr);
  allocator.Free(p1);
  allocator.Free(p2);
}

// Verify initial_growth_chunk_size_bytes config is accepted.
TEST_F(CudaPluginArenaTest, DeviceAllocator_InitialGrowthChunkSizeConfig) {
  Ort::KeyValuePairs options;
  options.Add("arena.initial_growth_chunk_size_bytes", "8192");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  void* p = allocator.Alloc(1024);
  ASSERT_NE(p, nullptr);
  allocator.Free(p);
}

// Verify max_power_of_two_extend_bytes config is accepted.
TEST_F(CudaPluginArenaTest, DeviceAllocator_MaxPowerOfTwoExtendConfig) {
  Ort::KeyValuePairs options;
  options.Add("arena.max_power_of_two_extend_bytes", "1048576");  // 1 MB cap

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  void* p = allocator.Alloc(2048);
  ASSERT_NE(p, nullptr);
  allocator.Free(p);
}

// Verify multiple config keys combined.
TEST_F(CudaPluginArenaTest, DeviceAllocator_CombinedConfig) {
  Ort::KeyValuePairs options;
  options.Add("arena.extend_strategy", "1");  // kSameAsRequested
  options.Add("arena.initial_chunk_size_bytes", "8192");
  options.Add("arena.max_dead_bytes_per_chunk", "512");
  options.Add("arena.max_mem", "2097152");  // 2 MB

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  void* p = allocator.Alloc(4096);
  ASSERT_NE(p, nullptr);
  allocator.Free(p);

  auto stats = allocator.GetStats();
  EXPECT_GE(GetStatInt(stats, "NumAllocs"), 1);
}

// Verify arena chunk splitting: allocate a large chunk then a small one.
// The second allocation should reuse a split portion of the first free chunk.
TEST_F(CudaPluginArenaTest, DeviceAllocator_ChunkSplitting) {
  Ort::KeyValuePairs options;
  options.Add("arena.initial_chunk_size_bytes", "65536");
  options.Add("arena.max_dead_bytes_per_chunk", "256");  // force aggressive splitting

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  // First alloc triggers arena extension.
  void* p1 = allocator.Alloc(256);
  ASSERT_NE(p1, nullptr);

  auto stats1 = allocator.GetStats();
  int64_t ext1 = GetStatInt(stats1, "NumArenaExtensions");

  // Second alloc should reuse the remainder of the first chunk (no new extension).
  void* p2 = allocator.Alloc(256);
  ASSERT_NE(p2, nullptr);

  auto stats2 = allocator.GetStats();
  int64_t ext2 = GetStatInt(stats2, "NumArenaExtensions");
  EXPECT_EQ(ext1, ext2) << "Second alloc should split from existing chunk, not extend.";

  allocator.Free(p1);
  allocator.Free(p2);
}

// Verify chunk coalescing: alloc two adjacent chunks, free both, then alloc a large one
// that only fits if the two free chunks are merged.
TEST_F(CudaPluginArenaTest, DeviceAllocator_ChunkCoalescing) {
  Ort::KeyValuePairs options;
  // Use kNextPowerOfTwo (default) so that both small allocations come from
  // a single extension region and their freed chunks are contiguous.
  options.Add("arena.initial_chunk_size_bytes", "16384");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  constexpr size_t kSize = 4096;
  void* p1 = allocator.Alloc(kSize);
  void* p2 = allocator.Alloc(kSize);
  ASSERT_NE(p1, nullptr);
  ASSERT_NE(p2, nullptr);

  auto stats_before = allocator.GetStats();
  int64_t ext_before = GetStatInt(stats_before, "NumArenaExtensions");

  // Free both — the arena should coalesce them into a single free chunk.
  allocator.Free(p1);
  allocator.Free(p2);

  // Allocate a size that fits into the coalesced free chunk.
  void* p3 = allocator.Alloc(kSize * 2);
  ASSERT_NE(p3, nullptr);

  auto stats_after = allocator.GetStats();
  int64_t ext_after = GetStatInt(stats_after, "NumArenaExtensions");
  // Coalescing: the large alloc should reuse the merged free chunk without extending.
  EXPECT_EQ(ext_before, ext_after) << "Coalesced free chunk should serve the large alloc.";

  allocator.Free(p3);
}

// Verify Reserve within budget succeeds and the reserved memory is freed correctly.
TEST_F(CudaPluginArenaTest, DeviceAllocator_ReserveWithinBudget) {
  Ort::KeyValuePairs options;
  options.Add("arena.max_mem", "2097152");  // 2 MB
  options.Add("arena.initial_chunk_size_bytes", "4096");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  void* p = allocator.Reserve(4096);
  ASSERT_NE(p, nullptr);

  // Reserved memory contributes to InUse.
  auto stats = allocator.GetStats();
  EXPECT_GT(GetStatInt(stats, "InUse"), 0);
  EXPECT_GE(GetStatInt(stats, "NumReserves"), 1);

  // Free the reserved chunk.
  allocator.Free(p);

  auto stats_after = allocator.GetStats();
  EXPECT_EQ(GetStatInt(stats_after, "InUse"), 0);
}

// Verify max_mem exactly exhausted: alloc up to the limit, then one more should fail.
TEST_F(CudaPluginArenaTest, DeviceAllocator_MaxMemExhaustion) {
  constexpr size_t kMaxMem = 65536;
  Ort::KeyValuePairs options;
  options.Add("arena.max_mem", std::to_string(kMaxMem).c_str());
  options.Add("arena.initial_chunk_size_bytes", std::to_string(kMaxMem).c_str());

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  // Exhaust the arena.
  void* p1 = allocator.Alloc(kMaxMem);
  ASSERT_NE(p1, nullptr);

  // Arena is full — next alloc should return nullptr (not crash).
  void* p2 = allocator.Alloc(256);
  EXPECT_EQ(p2, nullptr);

  allocator.Free(p1);
}

// Verify non-numeric max_mem is rejected.
TEST_F(CudaPluginArenaTest, DeviceAllocator_InvalidMaxMemIsRejected) {
  Ort::KeyValuePairs bad_options;
  bad_options.Add("arena.max_mem", "abc");

  try {
    ORT_IGNORE_RETURN_VALUE(ort_env->CreateSharedAllocator(
        cuda_device_, OrtDeviceMemoryType_DEFAULT,
        OrtDeviceAllocator,
        bad_options));
    ort_env->CreateSharedAllocator(
        cuda_device_, OrtDeviceMemoryType_DEFAULT,
        OrtDeviceAllocator, {});
    FAIL() << "Expected CreateSharedAllocator to reject invalid max_mem.";
  } catch (const Ort::Exception&) {
    // Expected
  }

  ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator, {});
}

// Verify pinned allocator with custom config.
TEST_F(CudaPluginArenaTest, PinnedAllocator_CustomConfig) {
  auto pinned_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_HOST_ACCESSIBLE);
  if (!pinned_memory_info) {
    GTEST_SKIP() << "No pinned memory info available for this device.";
  }

  Ort::KeyValuePairs options;
  options.Add("arena.initial_chunk_size_bytes", "16384");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_HOST_ACCESSIBLE,
      OrtDeviceAllocator,
      options);
  if (!allocator) {
    GTEST_SKIP() << "No shared pinned allocator from CreateSharedAllocator.";
  }

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_HOST_ACCESSIBLE,
            OrtDeviceAllocator, {});
      });

  void* p = allocator.Alloc(1024);
  ASSERT_NE(p, nullptr);

  // Pinned memory should be directly usable from host.
  std::memset(p, 0xAA, 1024);
  auto* bytes = static_cast<unsigned char*>(p);
  EXPECT_EQ(bytes[0], 0xAA);
  EXPECT_EQ(bytes[1023], 0xAA);

  allocator.Free(p);

  auto stats = allocator.GetStats();
  EXPECT_EQ(GetStatInt(stats, "TotalAllocated"), 16384);
}

// Verify pinned: alloc, free, realloc reuses memory.
TEST_F(CudaPluginArenaTest, PinnedAllocator_Reuse) {
  auto pinned_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_HOST_ACCESSIBLE);
  if (!pinned_memory_info) {
    GTEST_SKIP() << "No pinned memory info available for this device.";
  }

  auto allocator = ort_env->GetSharedAllocator(pinned_memory_info);
  if (!allocator) {
    GTEST_SKIP() << "No shared pinned allocator available.";
  }

  void* p1 = allocator.Alloc(512);
  ASSERT_NE(p1, nullptr);
  allocator.Free(p1);

  auto stats1 = allocator.GetStats();
  int64_t ext1 = GetStatInt(stats1, "NumArenaExtensions");

  void* p2 = allocator.Alloc(512);
  ASSERT_NE(p2, nullptr);
  allocator.Free(p2);

  auto stats2 = allocator.GetStats();
  int64_t ext2 = GetStatInt(stats2, "NumArenaExtensions");
  EXPECT_EQ(ext1, ext2) << "Pinned arena should reuse freed chunk.";
}

// Verify all stat keys are reported for the device arena.
TEST_F(CudaPluginArenaTest, DeviceAllocator_AllStatsKeysPresent) {
  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  void* p = allocator.Alloc(1024);
  ASSERT_NE(p, nullptr);
  allocator.Free(p);

  auto stats = allocator.GetStats();
  // All known stat keys should be present.
  EXPECT_FALSE(GetStatValue(stats, "Limit").empty());
  EXPECT_FALSE(GetStatValue(stats, "InUse").empty());
  EXPECT_FALSE(GetStatValue(stats, "TotalAllocated").empty());
  EXPECT_FALSE(GetStatValue(stats, "MaxInUse").empty());
  EXPECT_FALSE(GetStatValue(stats, "NumAllocs").empty());
  EXPECT_FALSE(GetStatValue(stats, "NumReserves").empty());
  EXPECT_FALSE(GetStatValue(stats, "NumArenaExtensions").empty());
  EXPECT_FALSE(GetStatValue(stats, "NumArenaShrinkages").empty());
  EXPECT_FALSE(GetStatValue(stats, "MaxAllocSize").empty());
}

// Verify mempool bytes_to_keep_on_shrink config is accepted.
TEST_F(CudaPluginArenaTest, Mempool_BytesToKeepOnShrinkConfig) {
  Ort::KeyValuePairs options;
  options.Add("arena.use_cuda_mempool", "1");
  options.Add("arena.cuda_mempool_bytes_to_keep_on_shrink", "65536");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  void* p = allocator.Alloc(4096);
  ASSERT_NE(p, nullptr);
  allocator.Free(p);
}

// Verify mempool all stat keys present.
TEST_F(CudaPluginArenaTest, Mempool_AllStatsKeysPresent) {
  Ort::KeyValuePairs options;
  options.Add("arena.use_cuda_mempool", "1");

  auto allocator = ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  void* p = allocator.Alloc(256);
  ASSERT_NE(p, nullptr);
  allocator.Free(p);

  auto stats = allocator.GetStats();
  EXPECT_FALSE(GetStatValue(stats, "NumAllocs").empty());
  EXPECT_FALSE(GetStatValue(stats, "TotalAllocated").empty());
  EXPECT_FALSE(GetStatValue(stats, "InUse").empty());
  EXPECT_FALSE(GetStatValue(stats, "MaxInUse").empty());
  EXPECT_FALSE(GetStatValue(stats, "MaxAllocSize").empty());
}

// Verify that Shrink on the device arena frees unused regions and updates stats.
TEST_F(CudaPluginArenaTest, DeviceAllocator_ShrinkFreesUnusedRegions) {
  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);

  // Create a fresh allocator so stats are clean regardless of test order.
  ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator, {});
  auto restore = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  // Allocate and free to create a region.
  constexpr size_t kBytes = 4096;
  void* p = allocator.Alloc(kBytes);
  ASSERT_NE(p, nullptr);
  allocator.Free(p);

  auto stats_before = allocator.GetStats();
  int64_t total_before = GetStatInt(stats_before, "TotalAllocated");
  ASSERT_GT(total_before, 0);

  // Shrink should free the (now entirely free) region.
  allocator.Shrink();

  auto stats_after = allocator.GetStats();
  int64_t total_after = GetStatInt(stats_after, "TotalAllocated");
  EXPECT_LT(total_after, total_before);
  EXPECT_GE(GetStatInt(stats_after, "NumArenaShrinkages"), 1);
}

// Verify that Shrink does not free regions that have live allocations.
TEST_F(CudaPluginArenaTest, DeviceAllocator_ShrinkKeepsLiveRegions) {
  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);

  // Fresh allocator for isolation.
  ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator, {});
  auto restore = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  constexpr size_t kBytes = 4096;
  void* p = allocator.Alloc(kBytes);
  ASSERT_NE(p, nullptr);
  auto p_guard = std::unique_ptr<void, std::function<void(void*)>>(
      p, [&allocator](void* ptr) { allocator.Free(ptr); });

  auto stats_before = allocator.GetStats();
  int64_t total_before = GetStatInt(stats_before, "TotalAllocated");

  // Shrink while allocation is live — nothing should change.
  allocator.Shrink();

  auto stats_after = allocator.GetStats();
  EXPECT_EQ(GetStatInt(stats_after, "TotalAllocated"), total_before);
}

// Verify that Shrink on the pinned arena works.
TEST_F(CudaPluginArenaTest, PinnedAllocator_ShrinkFreesUnusedRegions) {
  auto pinned_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_HOST_ACCESSIBLE);
  if (!pinned_memory_info) {
    GTEST_SKIP() << "No pinned memory info available for this device.";
  }

  // Fresh allocator for isolation.
  ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_HOST_ACCESSIBLE,
      OrtDeviceAllocator, {});
  auto restore = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_HOST_ACCESSIBLE,
            OrtDeviceAllocator, {});
      });

  auto allocator = ort_env->GetSharedAllocator(pinned_memory_info);
  if (!allocator) {
    GTEST_SKIP() << "No shared pinned allocator available.";
  }

  void* p = allocator.Alloc(1024);
  ASSERT_NE(p, nullptr);
  allocator.Free(p);

  auto stats_before = allocator.GetStats();
  int64_t total_before = GetStatInt(stats_before, "TotalAllocated");
  ASSERT_GT(total_before, 0);

  allocator.Shrink();

  auto stats_after = allocator.GetStats();
  EXPECT_LT(GetStatInt(stats_after, "TotalAllocated"), total_before);
  EXPECT_GE(GetStatInt(stats_after, "NumArenaShrinkages"), 1);
}

// Verify that Shrink on the mempool allocator increments shrinkage counter.
TEST_F(CudaPluginArenaTest, MempoolAllocator_ShrinkTrimsPool) {
  // Create a mempool-based allocator via session config.
  Ort::KeyValuePairs options;
  options.Add("arena.use_cuda_mempool", "1");

  ort_env->CreateSharedAllocator(
      cuda_device_, OrtDeviceMemoryType_DEFAULT,
      OrtDeviceAllocator,
      options);

  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  auto restore_default = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1), [&](void*) {
        ort_env->CreateSharedAllocator(
            cuda_device_, OrtDeviceMemoryType_DEFAULT,
            OrtDeviceAllocator, {});
      });

  // Allocate and free to make the pool non-empty.
  void* p = allocator.Alloc(1024);
  ASSERT_NE(p, nullptr);
  allocator.Free(p);
  cudaDeviceSynchronize();

  auto stats_before = allocator.GetStats();
  int64_t shrinkages_before = GetStatInt(stats_before, "NumArenaShrinkages");

  allocator.Shrink();

  auto stats_after = allocator.GetStats();
  EXPECT_EQ(GetStatInt(stats_after, "NumArenaShrinkages"), shrinkages_before + 1);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
