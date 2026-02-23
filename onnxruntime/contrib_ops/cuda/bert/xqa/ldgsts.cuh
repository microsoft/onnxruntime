/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cuda_hint.cuh"
#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif
#include "barriers.cuh"

namespace ldgsts {
// @fixme: prefetch makes it slower on sm_86. Try on other platforms.
template <uint32_t size>
__device__ inline void copyAsync(
    void* dst, void const* src, uint32_t srcSize = size)  // srcSize == 0 means filling with zeros.
{
  static_assert(size == 4 || size == 8 || size == 16);
  if constexpr (size == 16) {
    // asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::
    // "l"(__cvta_generic_to_shared(dst)), "l"(src), "r"(srcSize));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"l"(__cvta_generic_to_shared(dst)), "l"(src),
                 "r"(srcSize));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"l"(__cvta_generic_to_shared(dst)), "l"(src),
                 "n"(size), "r"(srcSize));
  }
}

__device__ inline void commitGroup() {
  asm volatile("cp.async.commit_group;\n");
}

// wait until only targetNbInFlightGroups groups are still in-flight.
template <uint32_t targetNbInFlightGroups>
__device__ inline void waitGroup() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(targetNbInFlightGroups));
}

// noInc = false:  increase expected arrive count, in additional to increasing arrive count
// noInc = true: increases arrive count but does not modify expected arrive count
__device__ inline void barArrive(CtaBarrier& bar, bool noInc = false) {
  if (noInc) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"l"(__cvta_generic_to_shared(&bar)));
  } else {
    asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];\n" ::"l"(__cvta_generic_to_shared(&bar)));
  }
}

}  // namespace ldgsts
