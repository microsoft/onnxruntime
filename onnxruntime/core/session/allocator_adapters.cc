// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "allocator_adapters.h"
#include "core/session/inference_session.h"
#include "core/session/ort_env.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"

namespace onnxruntime {
OrtAllocatorImplWrappingIAllocator::OrtAllocatorImplWrappingIAllocator(onnxruntime::AllocatorPtr&& i_allocator)
    : i_allocator_(std::move(i_allocator)) {
  OrtAllocator::version = ORT_API_VERSION;
  OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<OrtAllocatorImplWrappingIAllocator*>(this_)->Alloc(size); };
  OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<OrtAllocatorImplWrappingIAllocator*>(this_)->Free(p); };
  OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const OrtAllocatorImplWrappingIAllocator*>(this_)->Info(); };
}

OrtAllocatorImplWrappingIAllocator ::~OrtAllocatorImplWrappingIAllocator() = default;

void* OrtAllocatorImplWrappingIAllocator::Alloc(size_t size) {
  return i_allocator_->Alloc(size);
}

void OrtAllocatorImplWrappingIAllocator::Free(void* p) {
  i_allocator_->Free(p);
}

const OrtMemoryInfo* OrtAllocatorImplWrappingIAllocator::Info() const {
  return &i_allocator_->Info();
}

IAllocatorImplWrappingOrtAllocator::IAllocatorImplWrappingOrtAllocator(OrtAllocator* ort_allocator)
    : IAllocator(*ort_allocator->Info(ort_allocator)), ort_allocator_(ort_allocator) {}

IAllocatorImplWrappingOrtAllocator::IAllocatorImplWrappingOrtAllocator(OrtAllocatorV2* ort_allocator)
    : IAllocator(*ort_allocator->Info(ort_allocator)), ort_allocator_v2_(ort_allocator) {}

IAllocatorImplWrappingOrtAllocator::~IAllocatorImplWrappingOrtAllocator() = default;

void* IAllocatorImplWrappingOrtAllocator::Alloc(size_t size) {
  return ort_allocator_
             ? ort_allocator_->Alloc(ort_allocator_, size)
             : ort_allocator_v2_->Alloc(ort_allocator_v2_, size);
}

void IAllocatorImplWrappingOrtAllocator::Free(void* p) {
  return ort_allocator_
             ? ort_allocator_->Free(ort_allocator_, p)
             : ort_allocator_v2_->Free(ort_allocator_v2_, p);
}

void* IAllocatorImplWrappingOrtAllocator::Reserve(size_t size) {
  // OrtAllocator has no callback for Reserve()
  if (ort_allocator_ != nullptr) {
    return ort_allocator_->Alloc(ort_allocator_, size);
  }

  if (ort_allocator_v2_->Reserve != nullptr) {
    return ort_allocator_v2_->Reserve(ort_allocator_v2_, size);
  }

  return ort_allocator_v2_->Alloc(ort_allocator_v2_, size);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtApis::CreateAllocator, const OrtSession* sess, const OrtMemoryInfo* mem_info, _Outptr_ OrtAllocator** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  auto allocator_ptr = session->GetAllocator(*mem_info);
  if (!allocator_ptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "No requested allocator available");
  }
  *out = new onnxruntime::OrtAllocatorImplWrappingIAllocator(std::move(allocator_ptr));
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

  auto st = env->CreateAndRegisterAllocator(*mem_info, arena_cfg);

  if (!st.IsOK()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, st.ErrorMessage().c_str());
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RegisterAllocator, _Inout_ OrtEnv* env, _In_ OrtAllocatorV2* allocator) {
  using namespace onnxruntime;
  if (!env) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Env is null");
  }

  if (!allocator) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Provided allocator is null");
  }

  if (allocator->Info(allocator)->alloc_type == OrtAllocatorType::OrtArenaAllocator) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Please register the allocator as OrtDeviceAllocator "
                                 "even if the provided allocator has arena logic built-in. "
                                 "OrtArenaAllocator is reserved for internal arena logic based "
                                 "allocators only.");
  }

  std::shared_ptr<IAllocator> i_alloc_ptr =
      std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(allocator);

  auto st = env->RegisterAllocator(i_alloc_ptr);

  if (!st.IsOK()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, st.ErrorMessage().c_str());
  }
  return nullptr;
}

ORT_API(void, OrtApis::ReleaseAllocator, _Frees_ptr_opt_ OrtAllocator* allocator) {
  delete static_cast<onnxruntime::OrtAllocatorImpl*>(allocator);
}
