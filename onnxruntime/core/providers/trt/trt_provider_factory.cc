// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/trt/trt_provider_factory.h"
#include "trt_allocator.h"
#include <atomic>
#include "trt_execution_provider.h"

using namespace onnxruntime;

namespace
{
struct TRTProviderFactory
{
    const OrtProviderFactoryInterface* const cls;
    std::atomic_int ref_count;
    int device_id;
    TRTProviderFactory();
};

OrtStatus* ORT_API_CALL CreateTRT(void* this_, OrtProvider** out)
{
    TRTExecutionProviderInfo info;
    TRTProviderFactory* this_ptr = static_cast<TRTProviderFactory*>(this_);
    info.device_id = this_ptr->device_id;
    TRTExecutionProvider* ret = new TRTExecutionProvider();
    *out = (OrtProvider*)ret;
    return nullptr;
}

uint32_t ORT_API_CALL ReleaseTRT(void* this_)
{
    TRTProviderFactory* this_ptr = static_cast<TRTProviderFactory*>(this_);
    if (--this_ptr->ref_count == 0)
        delete this_ptr;
    return 0;
}

uint32_t ORT_API_CALL AddRefTRT(void* this_)
{
    TRTProviderFactory* this_ptr = static_cast<TRTProviderFactory*>(this_);
    ++this_ptr->ref_count;
    return 0;
}

constexpr OrtProviderFactoryInterface trt_cls =
{
    AddRefTRT,
    ReleaseTRT,
    CreateTRT,
};

TRTProviderFactory::TRTProviderFactory() : cls(&trt_cls), ref_count(1), device_id(0) {}
}  // namespace

ORT_API_STATUS_IMPL(OrtCreateTRTExecutionProviderFactory, int device_id, _Out_ OrtProviderFactoryInterface*** out)
{
    TRTProviderFactory* ret = new TRTProviderFactory();
    ret->device_id = device_id;
    *out = (OrtProviderFactoryInterface**)ret;
    return nullptr;
}

ORT_API_STATUS_IMPL(OrtCreateTRTAllocatorInfo, enum OrtAllocatorType type, enum OrtMemType mem_type, _Out_ OrtAllocatorInfo** out)
{
    return OrtCreateAllocatorInfo(onnxruntime::TRT, type, 0, mem_type, out);
}

