// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
namespace ep_allocator_utils {

// Creates a plugin EP allocator using OrtEp::CreateAllocator when available, otherwise OrtEpFactory::CreateAllocator,
// and wraps the result as an IAllocator.
// Outputs are modified only on success.
Status CreateAndWrapEpAllocator(OrtEp* ep,
                                OrtEpFactory& ep_factory,
                                const OrtMemoryInfo& memory_info,
                                const OrtKeyValuePairs* ep_factory_allocator_options,
                                AllocatorPtr& allocator_out,
                                OrtAllocator** raw_allocator_out = nullptr);

// Creates a plugin EP allocator using OrtEpFactory::CreateAllocator and wraps the result as an IAllocator.
// Outputs are modified only on success.
inline Status CreateAndWrapEpAllocator(OrtEpFactory& ep_factory,
                                       const OrtMemoryInfo& memory_info,
                                       const OrtKeyValuePairs* ep_factory_allocator_options,
                                       AllocatorPtr& allocator_out,
                                       OrtAllocator** raw_allocator_out = nullptr) {
  return CreateAndWrapEpAllocator(/*ep*/ nullptr, ep_factory, memory_info, ep_factory_allocator_options,
                                  allocator_out, raw_allocator_out);
}

}  // namespace ep_allocator_utils
}  // namespace onnxruntime
