// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

// This header is only used by the in-tree CUDA EP kernel registration tables,
// which are compiled with the shared-provider bridge. Use the bridged types via
// provider_api.h (KernelDef, MLDataType, DataTypeImpl) rather than the real
// framework headers. The CUDA plugin EP does not use this header.
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_op_allowlist.h"

namespace onnxruntime {
namespace cuda {

// Build-time filter that decides whether a CUDA kernel should be dropped at
// registration time. It combines the Stage 1 data-type filter (e.g. drop
// `double` when DISABLE_DOUBLE_TYPE is defined) and the Stage 2 operator
// allow-list (drop ops not present in the configured allow-list). It is a no-op
// unless one of those build options is enabled.
//
// Note: this removes the kernel *registration* only. Reclaiming the disabled
// kernels' device code additionally relies on linker dead-code elimination
// (--gc-sections) or on excluding the kernel sources from compilation.
inline bool IsCudaKernelDisabledByType([[maybe_unused]] const KernelDef& kernel_def) {
  // Stage 2: operator allow-list. Drop kernels whose op type is not allowed.
  if (!IsCudaOpAllowed(kernel_def.OpName())) {
    return true;
  }

#if defined(DISABLE_DOUBLE_TYPE)
  const MLDataType disabled_type = DataTypeImpl::GetTensorType<double>();

  // Drop the kernel if any of its type constraints can only be satisfied by a
  // disabled type (e.g. a `double`-only kernel instantiation). Kernels that allow
  // the disabled type among other still-enabled types are kept.
  for (const auto& type_constraint : kernel_def.TypeConstraints()) {
    const std::vector<MLDataType>& allowed_types = type_constraint.second;
    if (allowed_types.empty()) {
      continue;
    }

    bool all_disabled = true;
    for (const MLDataType allowed_type : allowed_types) {
      if (allowed_type != disabled_type) {
        all_disabled = false;
        break;
      }
    }

    if (all_disabled) {
      return true;
    }
  }
#endif
  return false;
}

}  // namespace cuda
}  // namespace onnxruntime
