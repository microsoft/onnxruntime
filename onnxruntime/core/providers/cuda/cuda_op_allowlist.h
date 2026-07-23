// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#if defined(ORT_ENABLE_CUDA_OP_ALLOWLIST)
#include <unordered_set>
#endif

namespace onnxruntime {
namespace cuda {

// Stage 2 build-time operator allow-list for the CUDA EP.
//
// When `ORT_ENABLE_CUDA_OP_ALLOWLIST` is defined (enabled via the
// `--cuda_op_allowlist <file>` build option), only the operator types listed in
// the generated allow-list are registered on the CUDA EP; every other CUDA
// operator is skipped at registration time. This applies to both the in-tree
// CUDA EP and the CUDA plugin EP, which share the same kernel sources.
//
// When no allow-list is configured (the default), every operator is allowed and
// this is a no-op.
//
// The allow-list matches on operator type only (domain-agnostic), so listing
// e.g. `LayerNormalization` keeps it in every domain that registers it.
//
// Note: excluded operators are removed from the CUDA kernel registry, so the
// CUDA EP no longer claims those nodes (they fall back to another EP, typically
// CPU). Graph fusions that would produce a CUDA-only excluded operator are also
// skipped (see core/optimizer/graph_transformer_utils.cc).
inline bool IsCudaOpAllowed([[maybe_unused]] const std::string& op_type) {
#if defined(ORT_ENABLE_CUDA_OP_ALLOWLIST)
  static const std::unordered_set<std::string> allowed = {
#include "cuda_op_allowlist_data.inc"
  };
  return allowed.find(op_type) != allowed.end();
#else
  return true;
#endif
}

}  // namespace cuda
}  // namespace onnxruntime
