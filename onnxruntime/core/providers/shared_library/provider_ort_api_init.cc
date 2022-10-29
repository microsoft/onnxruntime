// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the provider DLL side of the provider API to let providers be built
// as a DLL

// Typically, this should be a part of the provider_bridge_provider.cc
// However, we choose to put this into a separate unit. Reason being, it is for
// providers that are implemented as a CustomOp and make use of the public API in conjunction with
// ORT_API_MANUAL_INIT, so they need to initialize their global API ptr before doing any work.
// This is not needed for providers that are implemented as a shared library and make use of the internal
// ORT Apis which are provider via provider_api.h and implemented in provider_bridge_provider.cc where we do not
// want to include public API headers.

#include "provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"

#include <mutex>

namespace onnxruntime {
namespace {
std::once_flag init;
}  // namespace

void InitProviderOrtApi() {
  std::call_once(init, []() { Ort::Global<void>::api_ = Provider_GetHost()->OrtGetApiBase()->GetApi(ORT_API_VERSION); });
}

}  // namespace onnxruntime