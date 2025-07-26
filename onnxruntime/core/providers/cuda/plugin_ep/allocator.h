// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once

#include <memory>

#include "core/session/onnxruntime_c_api.h"

namespace cuda_plugin_ep {
struct CudaOrtAllocator : OrtAllocator {
  CudaOrtAllocator(const OrtMemoryInfo* mem_info, const OrtApi& api);

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size) noexcept;
  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p) noexcept;
  static void* ORT_API_CALL PinnedAllocImpl(struct OrtAllocator* this_, size_t size) noexcept;
  static void ORT_API_CALL PinnedFreeImpl(struct OrtAllocator* this_, void* p) noexcept;

  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) noexcept;

 private:
  const OrtMemoryInfo& memory_info_;
  const OrtMemoryDevice& memory_device_;
};

}  // namespace cuda_plugin_ep
