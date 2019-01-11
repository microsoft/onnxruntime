// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/cpu_provider_factory.h"
#include <atomic>
#include "cpu_execution_provider.h"

using namespace onnxruntime;

struct CpuProviderFactory : OrtProviderFactoryImpl {
  CpuProviderFactory(bool create_arena);

 private:
  bool create_arena_;

  OrtStatus* CreateProvider(OrtProvider** out);
};

OrtStatus* CpuProviderFactory::CreateProvider(OrtProvider** out) {
  CPUExecutionProviderInfo info;
  info.create_arena = create_arena_;
  CPUExecutionProvider* ret = new CPUExecutionProvider(info);
  *out = (OrtProvider*)ret;
  return nullptr;
}

CpuProviderFactory::CpuProviderFactory(bool create_arena) : create_arena_(create_arena) {
  OrtProviderFactory::CreateProvider = [](OrtProviderFactory* this_, OrtProvider** out) { return static_cast<CpuProviderFactory*>(this_)->CreateProvider(out); };
}

ORT_API_STATUS_IMPL(OrtCreateCpuExecutionProviderFactory, int use_arena, _Out_ OrtProviderFactory** out) {
  *out = new CpuProviderFactory(use_arena != 0);
  return nullptr;
}

ORT_API(void, OrtReleaseProviderFactory, OrtProviderFactory* in) {
  delete static_cast<OrtProviderFactoryImpl*>(in);
}

ORT_API_STATUS_IMPL(OrtCreateCpuAllocatorInfo, enum OrtAllocatorType type, enum OrtMemType mem_type, _Out_ OrtAllocatorInfo** out) {
  return OrtCreateAllocatorInfo(onnxruntime::CPU, type, 0, mem_type, out);
}
