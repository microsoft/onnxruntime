// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <utility>

#include "core/providers/rocm/tunable/gemm_ck.cuh"
#include "core/providers/rocm/tunable/gemm_rocblas.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/tunable.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {
namespace internal {

template <typename T, typename ALayout, typename BLayout>
class GemmTunableOp : public tunable::TunableOp<GemmParams<T>> {
 public:
  GemmTunableOp() {
    this->ops_.emplace_back(RocBlasGemmOp<T>);
    for (auto&& [_, op] : GetCKGemmTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->ops_.emplace_back(std::move(op));
    }
  }
};

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
