/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_fp16.h>
#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/config.hpp>

namespace cute
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDACC_VER_MAJOR__ >= 12))
#define CUTE_ARCH_MULTIMEM_SM90_ENABLED
#endif

////////////////////////////////////
// FP16 precision                 //
////////////////////////////////////
struct SM90_MULTIMEM_LDREDUCE_F16x8
{
    // used for unpacking tensor into registers
    using DRegisters = uint32_t[4];

    CUTE_HOST_DEVICE static void copy(
        void const* gmem_mc_ptr, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
    {
#if defined(CUTE_ARCH_MULTIMEM_SM90_ENABLED)
        uint64_t gmem_addr = reinterpret_cast<uint64_t>(gmem_mc_ptr);
        // Load gmem_addr from each GPU, reduce in switch, and return.
        asm volatile("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
                     : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
                     : "l"(gmem_addr)
                     : "memory");
#else
        CUTE_INVALID_CONTROL_PATH("Trying to use multimem.red without CUTE_ARCH_NVLS_SM90_ENABLED.");
#endif
    }
};

struct SM90_MULTIMEM_ST_F16x8
{
    // used for unpacking tensor into registers
    using SRegisters = uint32_t[4];

    CUTE_HOST_DEVICE static void copy(
        void const* gmem_mc_ptr, uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3)
    {
#if defined(CUTE_ARCH_MULTIMEM_SM90_ENABLED)
        uint64_t gmem_addr = reinterpret_cast<uint64_t>(gmem_mc_ptr);
        // Load gmem_addr from each GPU, add srcX, and store back.
        asm volatile("multimem.st.global.v4.f16x2 [%0], {%1,%2,%3,%4};" ::"l"(gmem_addr), "r"(src0), "r"(src1),
                     "r"(src2), "r"(src3)
                     : "memory");
#else
        CUTE_INVALID_CONTROL_PATH("Trying to use multimem.red without CUTE_ARCH_NVLS_SM90_ENABLED.");
#endif
    }
};

////////////////////////////////////
// BF16 precision                 //
////////////////////////////////////
struct SM90_MULTIMEM_LDREDUCE_BF16x8
{
    // used for unpacking tensor into registers
    using DRegisters = uint32_t[4];

    CUTE_HOST_DEVICE static void copy(
        void const* gmem_mc_ptr, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
    {
#if defined(CUTE_ARCH_MULTIMEM_SM90_ENABLED)
        uint64_t gmem_addr = reinterpret_cast<uint64_t>(gmem_mc_ptr);
        // Load gmem_addr from each GPU, reduce in switch, and return.
        asm volatile("multimem.ld_reduce.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                     : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
                     : "l"(gmem_addr)
                     : "memory");
#else
        CUTE_INVALID_CONTROL_PATH("Trying to use multimem.red without CUTE_ARCH_NVLS_SM90_ENABLED.");
#endif
    }
};

struct SM90_MULTIMEM_ST_BF16x8
{
    // used for unpacking tensor into registers
    using SRegisters = uint32_t[4];

    CUTE_HOST_DEVICE static void copy(
        void const* gmem_mc_ptr, uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3)
    {
#if defined(CUTE_ARCH_MULTIMEM_SM90_ENABLED)
        uint64_t gmem_addr = reinterpret_cast<uint64_t>(gmem_mc_ptr);
        // Load gmem_addr from each GPU, add srcX, and store back.
        asm volatile("multimem.st.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(gmem_addr), "r"(src0), "r"(src1),
                     "r"(src2), "r"(src3)
                     : "memory");
#else
        CUTE_INVALID_CONTROL_PATH("Trying to use multimem.red without CUTE_ARCH_NVLS_SM90_ENABLED.");
#endif
    }
};

} // namespace cute
