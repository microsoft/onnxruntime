// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocator.h"
#include "core/framework/utils.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_apis.h"
#include <assert.h>
#include "core/framework/allocatormgr.h"

struct OrtAllocatorForDevice : public OrtAllocator {
  explicit OrtAllocatorForDevice(onnxruntime::AllocatorPtr&& dev_allocator)
      : device_allocator_(std::move(dev_allocator)) {
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<OrtAllocatorForDevice*>(this_)->Alloc(size); };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<OrtAllocatorForDevice*>(this_)->Free(p); };
    OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const OrtAllocatorForDevice*>(this_)->Info(); };
  }

  ~OrtAllocatorForDevice() = default;

  void* Alloc(size_t size) const {
    return device_allocator_->Alloc(size);
  }
  void Free(void* p) const {
    device_allocator_->Free(p);
  }

  const OrtMemoryInfo* Info() const {
    return &device_allocator_->Info();
  }

  OrtAllocatorForDevice(const OrtAllocatorForDevice&) = delete;
  OrtAllocatorForDevice& operator=(const OrtAllocatorForDevice&) = delete;

 private:
  onnxruntime::AllocatorPtr device_allocator_;
};

#define API_IMPL_BEGIN try {
#define API_IMPL_END                                                \
  }                                                                 \
  catch (const std::exception& ex) {                                \
    return OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what()); \
  }

ORT_API_STATUS_IMPL(OrtApis::CreateAllocator, const OrtSession* sess, const OrtMemoryInfo* mem_info, _Outptr_ OrtAllocator** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  auto allocator_ptr = session->GetAllocator(*mem_info);
  if (!allocator_ptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "No requested allocator available");
  }
  *out = new OrtAllocatorForDevice(std::move(allocator_ptr));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateAllocatorForSharing, _In_ const OrtMemoryInfo* mem_info,
                    _In_ const OrtArenaCfg* arena_cfg, _Outptr_ OrtAllocator** out) {
  using namespace onnxruntime;
  if (!mem_info) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "OrtMemoryInfo is null");
  }

  // TODO should we allow sharing of non-CPU allocators?
  if (mem_info->device.Type() != OrtDevice::CPU) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Only CPU devices are supported for now.");
  }

  // determine if arena should be used
  bool create_arena = mem_info->alloc_type == OrtArenaAllocator;

#ifdef USE_JEMALLOC
#if defined(USE_MIMALLOC_ARENA_ALLOCATOR) || defined(USE_MIMALLOC_STL_ALLOCATOR)
#error jemalloc and mimalloc should not both be enabled
#endif
  //JEMalloc already has memory pool, so just use device allocator.
  create_arena = false;
#elif !(defined(__amd64__) || defined(_M_AMD64))
  //Disable Arena allocator for x86_32 build because it may run into infinite loop when integer overflow happens
  create_arena = false;
#endif

  AllocatorPtr allocator_ptr;
  size_t max_mem = std::numeric_limits<size_t>::max();

  // create appropriate DeviceAllocatorRegistrationInfo and allocator based on create_arena
  if (create_arena) {
    ArenaExtendStrategy arena_extend_strategy = BFCArena::DEFAULT_ARENA_EXTEND_STRATEGY;
    int initial_chunk_size_bytes = BFCArena::DEFAULT_INITIAL_CHUNK_SIZE_BYTES;
    int max_dead_bytes_per_chunk = BFCArena::DEFAULT_MAX_DEAD_BYTES_PER_CHUNK;
    if (arena_cfg) {
      if (arena_cfg->max_mem != -1) max_mem = arena_cfg->max_mem;
      if (arena_cfg->arena_extend_strategy == 0) {
        arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo;
      } else if (arena_cfg->arena_extend_strategy == 1) {
        arena_extend_strategy = ArenaExtendStrategy::kSameAsRequested;
      }
      if (arena_cfg->initial_chunk_size_bytes != -1) initial_chunk_size_bytes = arena_cfg->initial_chunk_size_bytes;
      if (arena_cfg->max_dead_bytes_per_chunk != -1) max_dead_bytes_per_chunk = arena_cfg->max_dead_bytes_per_chunk;
    }

    DeviceAllocatorRegistrationInfo device_info{
        OrtMemTypeDefault,
        [mem_info](int) { return onnxruntime::make_unique<TAllocator>(*mem_info); },
        max_mem,
        arena_extend_strategy,
        initial_chunk_size_bytes,
        max_dead_bytes_per_chunk};
    allocator_ptr = CreateAllocator(device_info, 0, create_arena);
  } else {
    DeviceAllocatorRegistrationInfo device_info{OrtMemTypeDefault,
                                                [](int) { return onnxruntime::make_unique<TAllocator>(); },
                                                max_mem};
    allocator_ptr = CreateAllocator(device_info, 0, create_arena);
  }

  *out = new OrtAllocatorForDevice(std::move(allocator_ptr));
  return nullptr;
}

ORT_API(void, OrtApis::ReleaseAllocator, _Frees_ptr_opt_ OrtAllocator* allocator) {
  delete reinterpret_cast<OrtAllocatorForDevice*>(allocator);
}
