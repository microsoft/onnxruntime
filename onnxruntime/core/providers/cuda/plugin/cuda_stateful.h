// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"
#include <cuda_runtime_api.h>

namespace onnxruntime::cuda_plugin {

OrtStatus* RegisterCudaStatefulProvider(OrtEp* ep, int device_id, cudaStream_t stream) noexcept;

}  // namespace onnxruntime::cuda_plugin
