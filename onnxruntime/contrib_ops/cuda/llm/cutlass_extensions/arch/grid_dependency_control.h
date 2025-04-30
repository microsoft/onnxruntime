/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

/*! \file
    \brief Grid dependent control (GDC) helpers for programmatic dependent launches (PDL).
*/

#pragma once

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/conv/dispatch_policy.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#ifndef CUTLASS_GDC_ENABLED
#if (defined(CUTLASS_ENABLE_GDC_FOR_SM90) && __CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__)                      \
    && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL))
#define CUTLASS_GDC_ENABLED
#endif
#endif

namespace cutlass
{
namespace arch
{

// Issuing the launch_dependents instruction hints a dependent kernel to launch earlier
// launch_dependents doesn't impact the functionality but the performance:
// Launching a dependent kernel too early can compete with current kernels,
// while launching too late can lead to a long latency.
CUTLASS_DEVICE
void launch_dependent_grids()
{
#if (defined(CUTLASS_GDC_ENABLED))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Issuing the griddepcontrol.wait instruction enforces no global memory access
// prior to this istruction. This ensures the correctness of global memory access
// when launching a dependent kernel earlier.
CUTLASS_DEVICE
void wait_on_dependent_grids()
{
#if (defined(CUTLASS_GDC_ENABLED))
    asm volatile("griddepcontrol.wait;");
#endif
}

// Enable kernel-level query regarding whether the GDC feature is turned on
#if (defined(CUTLASS_GDC_ENABLED))
static constexpr bool IsGdcGloballyEnabled = true;
#else
static constexpr bool IsGdcGloballyEnabled = false;
#endif

} // namespace arch
} // namespace cutlass
