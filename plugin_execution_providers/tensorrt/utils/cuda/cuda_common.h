// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_call.h"

namespace onnxruntime {
namespace cuda {

#define CUDA_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(CUDA_CALL(expr))

}  // namespace cuda
}  // namespace onnxruntime
