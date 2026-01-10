// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

common::Status GetFusedActivationAttr(const OpKernelInfo& info, MLAS_ACTIVATION& activation);

}  // namespace onnxruntime
