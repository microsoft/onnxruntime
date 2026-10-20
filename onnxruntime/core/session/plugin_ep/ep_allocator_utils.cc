// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_allocator_utils.h"

#include <functional>
#include <memory>
#include <string_view>

#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/session/allocator_adapters.h"

namespace onnxruntime {
namespace ep_allocator_utils {

Status CreateAndWrapEpAllocator(OrtEp* ep,
                                OrtEpFactory& ep_factory,
                                const OrtMemoryInfo& memory_info,
                                const OrtKeyValuePairs* ep_factory_allocator_options,
                                AllocatorPtr& allocator_out,
                                OrtAllocator** raw_allocator_out) {
  const bool has_ep_create = ep != nullptr && ep->CreateAllocator != nullptr;
  const bool has_factory_create = ep_factory.CreateAllocator != nullptr;

  ORT_RETURN_IF_NOT((!has_ep_create && !has_factory_create) || ep_factory.ReleaseAllocator != nullptr,
                    "OrtEpFactory must implement ReleaseAllocator when OrtEp::CreateAllocator or "
                    "OrtEpFactory::CreateAllocator is implemented.");

  if (!has_ep_create && !has_factory_create) {
    allocator_out = nullptr;
    if (raw_allocator_out != nullptr) {
      *raw_allocator_out = nullptr;
    }
    return Status::OK();
  }

  std::function<void(OrtAllocator*)> allocator_deleter = [&ep_factory](OrtAllocator* allocator) {
    ep_factory.ReleaseAllocator(&ep_factory, allocator);
  };

  OrtAllocator* raw_allocator = nullptr;
  OrtStatus* creation_ort_status =
      has_ep_create
          ? ep->CreateAllocator(ep, &memory_info, &raw_allocator)
          : ep_factory.CreateAllocator(&ep_factory, &memory_info, ep_factory_allocator_options, &raw_allocator);
  OrtAllocatorUniquePtr owned_allocator{raw_allocator, std::move(allocator_deleter)};

  if (creation_ort_status != nullptr) {
    return ToStatusAndRelease(creation_ort_status);
  }

  AllocatorPtr wrapped_allocator;
  if (owned_allocator != nullptr) {
    const std::string_view creator = has_ep_create ? "OrtEp" : "OrtEpFactory";

    const OrtMemoryInfo* allocator_memory_info = owned_allocator->Info(owned_allocator.get());
    ORT_RETURN_IF_NOT(allocator_memory_info != nullptr, creator, " returned an allocator with null memory info.");

    ORT_RETURN_IF_NOT(allocator_memory_info->alloc_type != OrtAllocatorType::OrtArenaAllocator,
                      creator,
                      " returned an allocator with OrtAllocatorType of OrtArenaAllocator. "
                      "This type is reserved for ONNX Runtime internal usage only, as any arena usage by the "
                      "EP library should be opaque to ORT");

    constexpr uint32_t kOrtAllocatorShrinkMinVersion = 25;
    if (owned_allocator->version >= kOrtAllocatorShrinkMinVersion && owned_allocator->Shrink != nullptr) {
      wrapped_allocator = std::make_shared<IArenaImplWrappingOrtAllocator>(std::move(owned_allocator));
    } else {
      wrapped_allocator = std::make_shared<IAllocatorImplWrappingOrtAllocator>(std::move(owned_allocator));
    }
  }

  if (raw_allocator_out != nullptr) {
    *raw_allocator_out = raw_allocator;
  }
  allocator_out = std::move(wrapped_allocator);
  return Status::OK();
}

}  // namespace ep_allocator_utils
}  // namespace onnxruntime
