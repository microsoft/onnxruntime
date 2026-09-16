// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace functors {

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
// Activations supported by ElementWiseRangedTransform<MLFloat16>::Create().
bool IsFp16FusableActivation(const std::string& type);
#endif

}  // namespace functors
}  // namespace onnxruntime
