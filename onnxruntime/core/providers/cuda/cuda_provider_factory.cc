// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_provider_factory.h"
#include <atomic>
#include "cuda_execution_provider.h"

using namespace onnxruntime;

namespace {
struct CUDAProviderFactory {
  const OrtProviderFactoryInterface* const cls;
  std::atomic_int ref_count;
  int device_id;
  CUDAProviderFactory();
};

OrtStatus* ORT_API_CALL CreateCuda(void* this_, OrtProvider** out) {
  CUDAExecutionProviderInfo info;
  CUDAProviderFactory* this_ptr = (CUDAProviderFactory*)this_;
  info.device_id = this_ptr->device_id;
  CUDAExecutionProvider* ret = new CUDAExecutionProvider(info);
  *out = (OrtProvider*)ret;
  return nullptr;
}

uint32_t ORT_API_CALL ReleaseCuda(void* this_) {
  CUDAProviderFactory* this_ptr = (CUDAProviderFactory*)this_;
  if (--this_ptr->ref_count == 0)
    delete this_ptr;
  return 0;
}

uint32_t ORT_API_CALL AddRefCuda(void* this_) {
  CUDAProviderFactory* this_ptr = (CUDAProviderFactory*)this_;
  ++this_ptr->ref_count;
  return 0;
}

constexpr OrtProviderFactoryInterface cuda_cls = {
    AddRefCuda,
    ReleaseCuda,
    CreateCuda,
};

CUDAProviderFactory::CUDAProviderFactory() : cls(&cuda_cls), ref_count(1), device_id(0) {}
}  // namespace

ORT_API_STATUS(OrtCreateCUDAExecutionProviderFactory, int device_id, _Out_ OrtProviderFactoryInterface*** out) {
  CUDAProviderFactory* ret = new CUDAProviderFactory();
  ret->device_id = device_id;
  *out = (OrtProviderFactoryInterface**)ret;
  return nullptr;
}
