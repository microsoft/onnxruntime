// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "adapter/pch.h"

#include "winml_adapter_c_api.h"
#include "core/session/ort_apis.h"
#include "winml_adapter_apis.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"

namespace winmla = Windows::AI::MachineLearning::Adapter;

struct OrtAllocatorWrapper : public OrtAllocator {
 public:
  OrtAllocatorWrapper(onnxruntime::AllocatorPtr impl) : impl_(impl) {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
  }

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size) {
    return static_cast<OrtAllocatorWrapper*>(this_)->impl_->Alloc(size);
  }
  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p) {
    return static_cast<OrtAllocatorWrapper*>(this_)->impl_->Free(p);
  }
  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    return &(static_cast<const OrtAllocatorWrapper*>(this_)->impl_->Info());
  }

 private:
  onnxruntime::AllocatorPtr impl_;
};

ORT_API_STATUS_IMPL(winmla::ExecutionProviderSync, _In_ OrtExecutionProvider* provider) {
  API_IMPL_BEGIN
  const auto execution_provider = reinterpret_cast<onnxruntime::IExecutionProvider*>(provider);
  ORT_API_RETURN_IF_STATUS_NOT_OK(execution_provider->Sync());
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::GetProviderAllocator, _In_ OrtExecutionProvider* provider, OrtAllocator** allocator) {
  API_IMPL_BEGIN
  const auto execution_provider = reinterpret_cast<onnxruntime::IExecutionProvider*>(provider);
  auto allocator_ptr = execution_provider->GetAllocator(0, ::OrtMemType::OrtMemTypeDefault);
  *allocator = new (std::nothrow) OrtAllocatorWrapper(allocator_ptr);
  if (*allocator == nullptr) {
    return OrtApis::CreateStatus(ORT_FAIL, "Out of memory");
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::GetProviderMemoryInfo, _In_ OrtExecutionProvider* provider, OrtMemoryInfo** memory_info) {
  API_IMPL_BEGIN
  const auto execution_provider = reinterpret_cast<onnxruntime::IExecutionProvider*>(provider);

  auto allocator = execution_provider->GetAllocator(0, ::OrtMemType::OrtMemTypeDefault);

  const auto& info = allocator->Info();
  *memory_info = new (std::nothrow) OrtMemoryInfo(info.name, info.alloc_type, info.device, info.id, info.mem_type);
  if (*memory_info == nullptr) {
    return OrtApis::CreateStatus(ORT_FAIL, "Out of memory");
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::FreeProviderAllocator, _In_ OrtAllocator* allocator) {
  API_IMPL_BEGIN
  delete static_cast<OrtAllocatorWrapper*>(allocator);
  return nullptr;
  API_IMPL_END
}