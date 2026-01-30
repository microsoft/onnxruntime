// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_CUDA

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "core/common/inlined_containers.h"  // InlinedVector
#include "core/framework/allocator.h"        // OrtMemoryInfo, IAllocator, AllocatorStats, onnxruntime::CUDA
#include "core/framework/execution_provider.h"
#include "core/framework/stream_handles.h"  // onnxruntime::Stream (interface)
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_provider_factory_creator.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

// --------- Helpers ---------

// cuda errors are sticky and may affect subsequent API calls.
// we want to clear the error if when supported check fails.
void ClearCudaError() {
  ORT_IGNORE_RETURN_VALUE(::cudaGetLastError());
}

static bool IsCudaMemPoolSupported() {
  int ort_cuda_rt_version = 0;
  cudaError_t cuda_status = cudaRuntimeGetVersion(&ort_cuda_rt_version);
  if (cuda_status != cudaSuccess) {
    ClearCudaError();
    return false;
  }

  if (ort_cuda_rt_version < 11020) {
    return false;
  }

  int ort_cuda_driver_version = 0;
  cuda_status = cudaDriverGetVersion(&ort_cuda_driver_version);
  if (cuda_status != cudaSuccess) {
    ClearCudaError();
    return false;
  }

  if (ort_cuda_driver_version < 11020) {
    return false;
  }

  // Check if the driver version supports the runtime version
  if (ort_cuda_rt_version >= 12000 && ort_cuda_driver_version < 12000) {
    return false;
  }

  if (ort_cuda_rt_version >= 13000 && ort_cuda_driver_version < 13000) {
    return false;
  }

  // Creating a cuda mempool in some pipelines fails with
  // CUDA failure 801: operation not supported ; GPU=0 ; hostname=af14bbb1c000000 ;
  // Even though CUDA version may be 12.8 possibly due to the driver.
  cudaMemPoolProps props{};
  // Pinned is not the same as pinned allocator, cudaMemLocationTypeDevice actually does not exist
  // even though is present in some internet docs.
  props.allocType = cudaMemAllocationTypePinned;
  props.handleTypes = cudaMemHandleTypeNone;        // local to process
  props.location.type = cudaMemLocationTypeDevice;  // Device memory
  props.location.id = 0;                            // test device 0
  cudaMemPool_t pool;
  auto cuda_error = cudaMemPoolCreate(&pool, &props);
  if (cuda_error != cudaSuccess) {
    ClearCudaError();
    return false;
  }
  ORT_IGNORE_RETURN_VALUE(cudaMemPoolDestroy(pool));

  return true;
}

static ::cudaStream_t NewCudaStream() {
  ::cudaStream_t s{};
  const cudaError_t st = ::cudaStreamCreate(&s);
  EXPECT_EQ(st, cudaSuccess);
  return s;
}

static void DestroyCudaStream(::cudaStream_t s) {
  if (s) {
    EXPECT_EQ(cudaSuccess, ::cudaStreamDestroy(s));
  }
}

static void TouchDevice(void* p, size_t bytes, ::cudaStream_t s, unsigned char value = 0xAB) {
  ASSERT_NE(p, nullptr);
  ASSERT_EQ(::cudaSuccess, ::cudaMemsetAsync(p, static_cast<int>(value), bytes, s));
}

// --------- Test parameters ---------

struct MPArenaParams {
  uint64_t release_threshold = 1ull << 20;  // 1 MB (recommended in allocator docs)
  size_t bytes_to_keep = 4ull << 20;        // 4 MB (small trim target for tests)
};

OrtArenaCfg CreateArenaCfgFromParams(const MPArenaParams& params) {
  OrtArenaCfg cfg;
  cfg.initial_chunk_size_bytes = 0;  // Make BFCArena for CUDAPinned not to allocate anything here
  cfg.use_cuda_mempool = 1;          // Key switch
  cfg.cuda_mempool_release_threshold = params.release_threshold;
  cfg.cuda_mempool_bytes_to_keep_on_shrink = params.bytes_to_keep;
  return cfg;
}

std::unique_ptr<IExecutionProvider> CreateCudaExecutionProvider(OrtArenaCfg& arena_cfg) {
  OrtCUDAProviderOptionsV2 cuda_options;
  cuda_options.device_id = 0;  // single-device tests
  cuda_options.default_memory_arena_cfg = &arena_cfg;
  cuda_options.do_copy_in_default_stream = true;
  cuda_options.use_tf32 = false;
  if (auto factory = CudaProviderFactoryCreator::Create(&cuda_options))
    return factory->CreateProvider();
  return nullptr;
}

AllocatorPtr GetCudaMempoolArena(IExecutionProvider& cuda_ep) {
  auto allocators = cuda_ep.CreatePreferredAllocators();
  EXPECT_EQ(allocators.size(), 2u);
  const auto& mem_info = allocators[0]->Info();
  EXPECT_EQ("CUDAMemPoolArena", mem_info.name);
  return allocators[0];
}

// --------- Minimal test Stream adapter ---------
//
// Adapts a cudaStream_t to ORT's Stream interface.
// If your Stream interface has additional pure virtuals on the work branch,
// add trivial overrides here (returning defaults / no-ops) so tests compile.
class TestCudaStream final : public onnxruntime::Stream {
 public:
  TestCudaStream(::cudaStream_t s, const OrtDevice& device) : Stream(s, device) {}

  ~TestCudaStream() {
    DestroyCudaStream(static_cast<::cudaStream_t>(GetHandle()));
  }

  void* GetHandle() const {
    // ORT expects GetHandle() to return the native handle (cast to void*).
    return Stream::GetHandle();
  }
};

// --------- Test fixture ---------

class CudaMempoolArenaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!IsCudaMemPoolSupported()) {
      GTEST_SKIP() << "CUDA memory pools not supported on this device/driver.";
    }

    const auto& logger = onnxruntime::logging::LoggingManager::DefaultLogger();
    orig_severity_ = logger.GetSeverity();
    orig_verbosity_ = logger.VLOGMaxLevel();
    logging::LoggingManager::SetDefaultLoggerSeverity(logging::Severity::kVERBOSE);
    logging::LoggingManager::SetDefaultLoggerVerbosity(0);
    cuda_ep_ = CreateCudaExecutionProvider(arena_cfg_);
    cuda_ep_->SetLogger(&logger);
    arena_ = GetCudaMempoolArena(*cuda_ep_);
    mem_info_ = arena_->Info();
  }

  void TearDown() override {
    arena_.reset();
    cuda_ep_.reset();
    ::cudaDeviceSynchronize();
    logging::LoggingManager::SetDefaultLoggerSeverity(orig_severity_);
    logging::LoggingManager::SetDefaultLoggerVerbosity(orig_verbosity_);
  }

  logging::Severity orig_severity_;
  int orig_verbosity_;
  OrtArenaCfg arena_cfg_ = CreateArenaCfgFromParams(MPArenaParams());
  std::unique_ptr<IExecutionProvider> cuda_ep_;
  AllocatorPtr arena_;
  OrtMemoryInfo mem_info_;
};

// --------- Tests ---------

TEST_F(CudaMempoolArenaTest, AllocAndFree_OnDefaultStream) {
  const size_t kBytes = 1 << 20;  // 1 MB
  void* p = arena_->Alloc(kBytes);
  ASSERT_NE(p, nullptr);

  // default (legacy) stream 0
  ASSERT_EQ(::cudaSuccess, ::cudaMemsetAsync(p, 0xCD, kBytes, /*stream=*/0));
  arena_->Free(p);

  ASSERT_EQ(::cudaSuccess, ::cudaDeviceSynchronize());

  onnxruntime::AllocatorStats stats{};
  arena_->GetStats(&stats);
  EXPECT_GE(stats.num_allocs, 1u);
}

TEST_F(CudaMempoolArenaTest, AllocOnTwoStreams_OrderedFree) {
  const size_t kBytes = 2 << 20;  // 2 MB

  ::cudaStream_t s0 = NewCudaStream();
  ::cudaStream_t s1 = NewCudaStream();
  {
    TestCudaStream ort_s0(s0, mem_info_.device);
    TestCudaStream ort_s1(s1, mem_info_.device);

    void* p0 = arena_->AllocOnStream(kBytes, &ort_s0);
    void* p1 = arena_->AllocOnStream(kBytes, &ort_s1);
    ASSERT_NE(p0, nullptr);
    ASSERT_NE(p1, nullptr);

    TouchDevice(p0, kBytes, s0, 0x11);
    TouchDevice(p1, kBytes, s1, 0x22);

    // Enqueue ordered frees (no sync needed here).
    arena_->Free(p0);
    arena_->Free(p1);

    // Ensure queued frees completed on each stream.
    ASSERT_EQ(::cudaSuccess, ::cudaStreamSynchronize(s0));
    ASSERT_EQ(::cudaSuccess, ::cudaStreamSynchronize(s1));

    // Destroy streams here
  }

  ASSERT_EQ(::cudaSuccess, ::cudaGetLastError());
}

TEST_F(CudaMempoolArenaTest, Shrink_TrimsPool_And_AllowsFurtherUse) {
  const size_t kBytes = 2 << 20;

  InlinedVector<void*> ptrs;
  for (size_t i = 0; i < ptrs.capacity(); ++i) {
    void* p = arena_->Alloc(kBytes);
    ASSERT_NE(p, nullptr);
    ASSERT_EQ(::cudaSuccess, ::cudaMemsetAsync(p, 0xEF, kBytes, /*stream=*/0));
    ptrs.push_back(p);
  }
  ASSERT_EQ(::cudaSuccess, ::cudaDeviceSynchronize());

  for (void* p : ptrs) {
    arena_->Free(p);
  }
  ASSERT_EQ(::cudaSuccess, ::cudaDeviceSynchronize());

  // Trim and sanity-check future allocations still work.
  auto* arena_cast = IArena::SafeArenaCast(arena_.get());
  ASSERT_STATUS_OK(arena_cast->Shrink());

  void* p_check = arena_->Alloc(kBytes);
  ASSERT_NE(p_check, nullptr);
  arena_->Free(p_check);
  ASSERT_EQ(::cudaSuccess, ::cudaDeviceSynchronize());
}

TEST_F(CudaMempoolArenaTest, Reserve_DelegatesToAlloc) {
  const size_t kBytes = 512 * 1024;
  void* p = arena_->Reserve(kBytes);
  ASSERT_NE(p, nullptr);
  arena_->Free(p);
  ASSERT_EQ(::cudaSuccess, ::cudaDeviceSynchronize());
}

// Validates allocator dtor guarantees completion of queued frees even when
// streams are destroyed prior to allocator destruction.
TEST_F(CudaMempoolArenaTest, Destructor_CompletesQueuedFrees_EvenIfStreamDestroyed) {
  const size_t kBytes = 1 << 20;
  ::cudaStream_t s = NewCudaStream();

  {
    auto cuda_prov = CreateCudaExecutionProvider(arena_cfg_);
    cuda_prov->SetLogger(&onnxruntime::logging::LoggingManager::DefaultLogger());
    auto alloc = GetCudaMempoolArena(*cuda_ep_);
    {
      TestCudaStream ort_s(s, mem_info_.device);

      InlinedVector<void*> ptrs;
      for (size_t i = 0; i < ptrs.capacity(); ++i) {
        void* p = alloc->AllocOnStream(kBytes, &ort_s);
        ASSERT_NE(p, nullptr);
        TouchDevice(p, kBytes, s);
        ptrs.push_back(p);
      }

      for (void* p : ptrs) {
        alloc->Free(p);
      }

      // Destroy the stream *before* the frees have a chance to run.
    }

    // arena goes out of scope here; its destructor must:
    //   - sync known streams (best-effort),
    //   - device-wide synchronize as a safety net,
    //   - then trim and destroy the pool.
  }

  ASSERT_EQ(::cudaSuccess, ::cudaGetLastError());
  ASSERT_EQ(::cudaSuccess, ::cudaDeviceSynchronize());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_CUDA
