/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "cutlass/detail/dependent_false.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class DispatchPolicy, class TileShape, class ElementA, class StrideA, class ElementB, class StrideB,
          class TiledMma, class GmemTiledCopyA, class SmemLayoutAtomA, class SmemCopyAtomA, class TransformA,
          class GmemTiledCopyB, class SmemLayoutAtomB, class SmemCopyAtomB, class TransformB,
          template <class /* ElementCompute */> class Activation, bool SwapAB = false>
struct CollectiveMmaGated {
  static_assert(cutlass::detail::dependent_false<ElementA>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/collective/sm90_mma_gated_tma_gmma_ss_warpspecialized.hpp"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/collective/sm90_mma_gated_tma_gmma_ss_warpspecialized_fp8.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////
