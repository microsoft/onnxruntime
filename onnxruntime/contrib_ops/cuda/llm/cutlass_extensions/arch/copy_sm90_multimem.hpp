/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
