// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#if !defined(USE_ROCM)

namespace onnxruntime {
struct Provider;
struct CUDA_Provider;
CUDA_Provider* GetProvider();
}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return reinterpret_cast<onnxruntime::Provider*>(onnxruntime::GetProvider());
}
}

#endif
