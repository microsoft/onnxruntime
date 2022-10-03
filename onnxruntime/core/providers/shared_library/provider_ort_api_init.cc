// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the provider DLL side of the provider API to let providers be built
// as a DLL

#include "provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace {
// will run exactly once
struct ApiInitializer {
  explicit ApiInitializer(const OrtApi* api) {
    Ort::Global<void>::api_ = api;
  }
};
}  // namespace

void InitProviderOrtApi() {
  static ApiInitializer api_initializer(Provider_GetHost()->OrtGetApiBase()->GetApi(ORT_API_VERSION));
}

}  // namespace onnxruntime