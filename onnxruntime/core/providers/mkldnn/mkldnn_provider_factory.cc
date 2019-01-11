// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/mkldnn/mkldnn_provider_factory.h"
#include <atomic>
#include "mkldnn_execution_provider.h"

using namespace onnxruntime;

namespace {
struct MkldnnProviderFactory : OrtProviderFactoryImpl {
  MkldnnProviderFactory(bool create_arena);

 private:
  bool create_arena_;
  OrtStatus* CreateProvider(OrtProvider** out);
};

OrtStatus* ORT_API_CALL MkldnnProviderFactory::CreateProvider(OrtProvider** out) {
  MKLDNNExecutionProviderInfo info;
  info.create_arena = create_arena_;
  MKLDNNExecutionProvider* ret = new MKLDNNExecutionProvider(info);
  *out = (OrtProvider*)ret;
  return nullptr;
}

MkldnnProviderFactory::MkldnnProviderFactory(bool create_arena) : create_arena_(create_arena) {
  OrtProviderFactory::CreateProvider = [](OrtProviderFactory* this_, OrtProvider** out) { return static_cast<MkldnnProviderFactory*>(this_)->CreateProvider(out); };
}
}  // namespace

ORT_API_STATUS_IMPL(OrtCreateMkldnnExecutionProviderFactory, int use_arena, _Out_ OrtProviderFactory** out) {
  *out = new MkldnnProviderFactory(use_arena != 0);
  return nullptr;
}
