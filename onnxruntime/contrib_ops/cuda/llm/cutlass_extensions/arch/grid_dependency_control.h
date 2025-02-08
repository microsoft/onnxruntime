/***************************************************************************************************
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

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
