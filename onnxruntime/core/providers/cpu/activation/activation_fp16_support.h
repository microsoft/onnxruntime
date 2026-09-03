// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace functors {

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
// True for the activations ElementWiseRangedTransform<MLFloat16>::Create() can
// build. GemmActivationFusion consults this so it never fuses an FP16
// activation that FusedGemm<MLFloat16> would then fail to construct.
//
// Deliberately kept in a header with no Eigen dependency so the graph
// optimizer can include it without pulling in the CPU kernel headers.
bool IsFp16FusableActivation(const std::string& type);
#endif

}  // namespace functors
}  // namespace onnxruntime
