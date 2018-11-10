// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/cpu_provider_factory.h"
#include <atomic>
#include "cpu_execution_provider.h"

using namespace onnxruntime;

namespace {
struct CpuProviderFactory {
  const ONNXRuntimeProviderFactoryInterface* const cls;
  std::atomic_int ref_count;
  bool create_arena;
  CpuProviderFactory();
};

ONNXStatusPtr ONNXRUNTIME_API_STATUSCALL CreateCpu(void* this_, ONNXRuntimeProviderPtr* out) {
  CPUExecutionProviderInfo info;
  CpuProviderFactory* this_ptr = (CpuProviderFactory*)this_;
  info.create_arena = this_ptr->create_arena;
  CPUExecutionProvider* ret = new CPUExecutionProvider(info);
  *out = (ONNXRuntimeProviderPtr)ret;
  return nullptr;
}

uint32_t ONNXRUNTIME_API_STATUSCALL ReleaseCpu(void* this_) {
  CpuProviderFactory* this_ptr = (CpuProviderFactory*)this_;
  if (--this_ptr->ref_count == 0)
    delete this_ptr;
  return 0;
}

uint32_t ONNXRUNTIME_API_STATUSCALL AddRefCpu(void* this_) {
  CpuProviderFactory* this_ptr = (CpuProviderFactory*)this_;
  ++this_ptr->ref_count;
  return 0;
}

constexpr ONNXRuntimeProviderFactoryInterface cpu_cls = {
    {AddRefCpu,
     ReleaseCpu},
    CreateCpu,
};

CpuProviderFactory::CpuProviderFactory() : cls(&cpu_cls), ref_count(1), create_arena(true) {}
}  // namespace

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeCreateCpuExecutionProviderFactory, int use_arena, _Out_ ONNXRuntimeProviderFactoryPtr** out) {
  CpuProviderFactory* ret = new CpuProviderFactory();
  ret->create_arena = (use_arena != 0);
  *out = (ONNXRuntimeProviderFactoryPtr*)ret;
  return nullptr;
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeCreateCpuAllocatorInfo, enum ONNXRuntimeAllocatorType type, enum ONNXRuntimeMemType mem_type, _Out_ ONNXRuntimeAllocatorInfo** out) {
  return ONNXRuntimeCreateAllocatorInfo(onnxruntime::CPU, type, 0, mem_type, out);
}