// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/cpu_provider_factory.h"
#include <atomic>
#include "cpu_execution_provider.h"

using namespace onnxruntime;

namespace {
struct CpuProviderFactory {
  const OrtProviderFactoryInterface* const cls;
  std::atomic_int ref_count;
  bool create_arena;
  CpuProviderFactory();
};

ONNXStatus* ORT_API_CALL CreateCpu(void* this_, OrtProvider** out) {
  CPUExecutionProviderInfo info;
  CpuProviderFactory* this_ptr = (CpuProviderFactory*)this_;
  info.create_arena = this_ptr->create_arena;
  CPUExecutionProvider* ret = new CPUExecutionProvider(info);
  *out = (OrtProvider*)ret;
  return nullptr;
}

uint32_t ORT_API_CALL ReleaseCpu(void* this_) {
  CpuProviderFactory* this_ptr = (CpuProviderFactory*)this_;
  if (--this_ptr->ref_count == 0)
    delete this_ptr;
  return 0;
}

uint32_t ORT_API_CALL AddRefCpu(void* this_) {
  CpuProviderFactory* this_ptr = (CpuProviderFactory*)this_;
  ++this_ptr->ref_count;
  return 0;
}

constexpr OrtProviderFactoryInterface cpu_cls = {
    {AddRefCpu,
     ReleaseCpu},
    CreateCpu,
};

CpuProviderFactory::CpuProviderFactory() : cls(&cpu_cls), ref_count(1), create_arena(true) {}
}  // namespace

ORT_API_STATUS_IMPL(OrtCreateCpuExecutionProviderFactory, int use_arena, _Out_ OrtProviderFactoryInterface*** out) {
  CpuProviderFactory* ret = new CpuProviderFactory();
  ret->create_arena = (use_arena != 0);
  *out = (OrtProviderFactoryInterface**)ret;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtCreateCpuAllocatorInfo, enum OrtAllocatorType type, enum OrtMemType mem_type, _Out_ OrtAllocatorInfo** out) {
  return OrtCreateAllocatorInfo(onnxruntime::CPU, type, 0, mem_type, out);
}
