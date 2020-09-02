// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "device_allocator.h"
#include "core/session/inference_session.h"
#include "core/session/ort_env.h"
#include "core/session/allocator_impl.h"

#ifndef ORT_NO_EXCEPTIONS

#define API_IMPL_BEGIN try {
#define API_IMPL_END                                                \
  }                                                                 \
  catch (const std::exception& ex) {                                \
    return OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what()); \
  }

#else

#define API_IMPL_BEGIN {
#define API_IMPL_END }

#endif

ORT_API_STATUS_IMPL(OrtApis::CreateAllocator, const OrtSession* sess, const OrtMemoryInfo* mem_info, _Outptr_ OrtAllocator** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  auto allocator_ptr = session->GetAllocator(*mem_info);
  if (!allocator_ptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "No requested allocator available");
  }
  *out = new onnxruntime::OrtAllocatorForDevice(std::move(allocator_ptr));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateAndRegisterAllocator, _Inout_ OrtEnv* env, _In_ const OrtMemoryInfo* mem_info,
                    _In_ const OrtArenaCfg* arena_cfg) {
  using namespace onnxruntime;
  if (!env) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Env is null");
  }

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
  // create appropriate DeviceAllocatorRegistrationInfo and allocator based on create_arena
  if (create_arena) {
    // defaults in case arena_cfg is nullptr (not supplied by the user)
    size_t max_mem = 0;
    int arena_extend_strategy = -1;
    int initial_chunk_size_bytes = -1;
    int max_dead_bytes_per_chunk = -1;

    // override with values from the user supplied arena_cfg object
    if (arena_cfg) {
      max_mem = arena_cfg->max_mem;

      arena_extend_strategy = arena_cfg->arena_extend_strategy;
      // validate the value here
      if (!(arena_extend_strategy == -1 || arena_extend_strategy == 0 || arena_extend_strategy == 1)) {
        return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                     "Received invalid value for arena extend strategy."
                                     " Valid values can be either 0, 1 or -1.");
      }

      initial_chunk_size_bytes = arena_cfg->initial_chunk_size_bytes;
      max_dead_bytes_per_chunk = arena_cfg->max_dead_bytes_per_chunk;
    }

    OrtArenaCfg l_arena_cfg{max_mem, arena_extend_strategy, initial_chunk_size_bytes, max_dead_bytes_per_chunk};
    AllocatorCreationInfo alloc_creation_info{
        [mem_info](int) { return onnxruntime::make_unique<TAllocator>(*mem_info); },
        0,
        create_arena,
        l_arena_cfg};
    allocator_ptr = CreateAllocator(alloc_creation_info);
  } else {
    AllocatorCreationInfo alloc_creation_info{[](int) { return onnxruntime::make_unique<TAllocator>(); },
                                              0, create_arena};
    allocator_ptr = CreateAllocator(alloc_creation_info);
  }

  auto st = env->RegisterAllocator(allocator_ptr);
  if (!st.IsOK()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, st.ErrorMessage().c_str());
  }
  return nullptr;
}

ORT_API(void, OrtApis::ReleaseAllocator, _Frees_ptr_opt_ OrtAllocator* allocator) {
  delete reinterpret_cast<onnxruntime::OrtAllocatorForDevice*>(allocator);
}
