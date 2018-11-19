// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/mkldnn/mkldnn_provider_factory.h"
#include <atomic>
#include "mkldnn_execution_provider.h"

using namespace onnxruntime;

namespace {
struct MkldnnProviderFactory {
  const ONNXRuntimeProviderFactoryInterface* const cls;
  std::atomic_int ref_count;
  bool create_arena;
  MkldnnProviderFactory();
};

ONNXStatusPtr ONNXRUNTIME_API_STATUSCALL CreateMkldnn(void* this_, ONNXRuntimeProviderPtr* out) {
  MKLDNNExecutionProviderInfo info;
  MkldnnProviderFactory* this_ptr = (MkldnnProviderFactory*)this_;
  info.create_arena = this_ptr->create_arena;
  MKLDNNExecutionProvider* ret = new MKLDNNExecutionProvider(info);
  *out = (ONNXRuntimeProviderPtr)ret;
  return nullptr;
}

uint32_t ONNXRUNTIME_API_STATUSCALL ReleaseMkldnn(void* this_) {
  MkldnnProviderFactory* this_ptr = (MkldnnProviderFactory*)this_;
  if (--this_ptr->ref_count == 0)
    delete this_ptr;
  return 0;
}

uint32_t ONNXRUNTIME_API_STATUSCALL AddRefMkldnn(void* this_) {
  MkldnnProviderFactory* this_ptr = (MkldnnProviderFactory*)this_;
  ++this_ptr->ref_count;
  return 0;
}

constexpr ONNXRuntimeProviderFactoryInterface mkl_cls = {
    {AddRefMkldnn,
     ReleaseMkldnn},
    CreateMkldnn,
};

MkldnnProviderFactory::MkldnnProviderFactory() : cls(&mkl_cls), ref_count(1), create_arena(true) {}
}  // namespace

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeCreateMkldnnExecutionProviderFactory, int use_arena, _Out_ ONNXRuntimeProviderFactoryPtr** out) {
  MkldnnProviderFactory* ret = new MkldnnProviderFactory();
  ret->create_arena = (use_arena != 0);
  *out = (ONNXRuntimeProviderFactoryPtr*)ret;
  return nullptr;
}