// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

// This header is only used by the in-tree CUDA EP kernel registration tables,
// which are compiled with the shared-provider bridge. Use the bridged types via
// provider_api.h (KernelDef, MLDataType, DataTypeImpl) rather than the real
// framework headers. The CUDA plugin EP does not use this header.
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace cuda {

// Stage 1 build-time data-type filter for CUDA kernels.
//
// When a floating point element type is disabled at build time (currently only
// `double`, enabled via the `--disable_types double` build option which defines
// the DISABLE_DOUBLE_TYPE macro), CUDA kernels that can only run in that type are
// skipped at registration time, so the type becomes unavailable on the CUDA EP.
//
// This helper is used by the in-tree CUDA EP kernel registration tables. The CUDA
// plugin EP achieves the same effect inside its adapter KernelDefBuilder (which
// returns a null KernelDef for disabled-type kernels), because the plugin's
// registration loop only sees an opaque KernelDef that does not expose kernel type
// constraints.
//
// Note: this removes the kernel *registration* only. Reclaiming the disabled
// kernels' device code additionally relies on linker dead-code elimination
// (--gc-sections) or on excluding the kernel sources from compilation.
inline bool IsCudaKernelDisabledByType([[maybe_unused]] const KernelDef& kernel_def) {
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
