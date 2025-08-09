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

#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

////////////////////////////////////////////////////////////////////////////////

/*
 * Stateless universal device GEMM kernel type that treats GEMM as
 * a composition of a collective mainloop and a collective epilogue.
 *
 * Supports both the 2.x and 3.x APIs based on whether the first type is
 * a cute::tuple<> or not.
 * 2.x API implementation: cutlass/gemm/kernel/gemm_universal.h
 * 3.x API implementation: cutlass/gemm/kernel/gemm_*.hpp
 *
 * In the following declaration, the name preceding the 'Or' refers to
 * 3.x API type argument order, and the name succeeding the 'Or' refers to
 * 2.x API type argument order. Template arguments without two names
 * belong to the 3.x API only.
 **/
template <class ProblemShapeOrThreadblockMma_,  // (m, n, k) or (m, n, k, l)
          class CollectiveMainloopOrEpilogue_, class CollectiveEpilogueOrThreadblockSwizzle_, class TileScheduler_ = void,
          class Enable = void>
class GemmUniversalGated;

////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::kernel

////////////////////////////////////////////////////////////////////////////////

#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/kernel/sm90_gemm_gated_tma_warpspecialized_cooperative.hpp"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/kernel/sm90_gemm_gated_tma_warpspecialized_pingpong.hpp"
////////////////////////////////////////////////////////////////////////////////
