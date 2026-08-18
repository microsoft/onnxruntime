// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <memory>

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/nn/conv.h"

namespace onnxruntime {
namespace webgpu {

// Creates a subgroup-matrix Conv 1x1 implementation on devices whose vendor policy supports
// the subgroup-matrix kernel; returns nullptr otherwise, so the caller falls back to the
// normal Conv path. The impl reads the Conv attributes from `parent`.
template <bool is_channels_last, bool is_fused>
std::unique_ptr<typename Conv<is_channels_last, is_fused>::ConvOptImpl> CreateSubgroupMatrixConvImpl(
    const Conv<is_channels_last, is_fused>& parent,
    const ComputeContextBase& context);

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
