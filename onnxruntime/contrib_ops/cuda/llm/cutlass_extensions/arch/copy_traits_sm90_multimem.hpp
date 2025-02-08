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

#include "copy_sm90_multimem.hpp"
#include <cute/atom/copy_traits.hpp>
#include <cute/numeric/integral_ratio.hpp>
#include <cute/tensor.hpp>

namespace cute
{
// Utility for unpacking tensor into registers for multimem CopyOp
template <class CopyOp>
struct MULTIMEM_RED_Unpack
{
    template <class... Args, class TS, class SLayout, class TD, class DLayout>
    CUTE_HOST_DEVICE friend constexpr void copy_unpack(
        Copy_Traits<CopyOp, Args...> const& traits, Tensor<TS, SLayout> const& src, Tensor<TD, DLayout>& dst)
    {
        static_assert(is_rmem<TS>::value, "Expected RMEM src.");

        void const* dst_ptr = raw_pointer_cast(dst.data());

        using RegTypeS = typename remove_extent<typename CopyOp::SRegisters>::type;
        // NOTE: extent returns compile-time dimension of given type.
        constexpr int RegNumS = extent<typename CopyOp::SRegisters>::value;
        Tensor rS = recast<RegTypeS>(src);
        CUTE_STATIC_ASSERT_V(size(rS) == Int<RegNumS>{},
            "In CopyAtom, src layout doesn't vectorize into registers. This src layout is incompatible with this "
            "CopyOp.");

        return detail::explode(CopyOp::copy, &dst_ptr, seq<0>{}, rS, make_seq<RegNumS>{});
    }
};

// Utility for unpacking tensor into registers for multimem CopyOp
template <class CopyOp>
struct MULTIMEM_LDREDUCE_Unpack
{
    template <class... Args, class TS, class SLayout, class TD, class DLayout>
    CUTE_HOST_DEVICE friend constexpr void copy_unpack(
        Copy_Traits<CopyOp, Args...> const& traits, Tensor<TS, SLayout> const& src, Tensor<TD, DLayout>& dst)
    {
        static_assert(is_rmem<TS>::value, "Expected RMEM src.");

        void const* src_ptr = raw_pointer_cast(src.data());

        using RegTypeD = typename remove_extent<typename CopyOp::DRegisters>::type;
        // NOTE: extent returns compile-time dimension of given type.
        constexpr int RegNumD = extent<typename CopyOp::DRegisters>::value;
        Tensor rD = recast<RegTypeD>(dst);
        CUTE_STATIC_ASSERT_V(size(rD) == Int<RegNumD>{},
            "In CopyAtom, dst layout doesn't vectorize into registers. This dst layout is incompatible with this "
            "CopyOp.");

        return detail::explode(CopyOp::copy, &src_ptr, seq<0>{}, rD, make_seq<RegNumD>{});
    }
};

///////////////////////////////////////////
// LD_REDUCE
//////////////////////////////////////////
template <>
struct Copy_Traits<SM90_MULTIMEM_LDREDUCE_F16x8> : MULTIMEM_LDREDUCE_Unpack<SM90_MULTIMEM_LDREDUCE_F16x8>
{
    // Logical thread id to thread idx (one-thread)
    using ThrID = Layout<_1>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, _128>>; // 128-bits
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = SrcLayout;
    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

template <>
struct Copy_Traits<SM90_MULTIMEM_LDREDUCE_BF16x8> : MULTIMEM_LDREDUCE_Unpack<SM90_MULTIMEM_LDREDUCE_BF16x8>
{
    // Logical thread id to thread idx (one-thread)
    using ThrID = Layout<_1>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, _128>>; // 128-bits
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = SrcLayout;
    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

///////////////////////////////////////////
// STORE
//////////////////////////////////////////
template <>
struct Copy_Traits<SM90_MULTIMEM_ST_F16x8> : MULTIMEM_RED_Unpack<SM90_MULTIMEM_ST_F16x8>
{
    // Logical thread id to thread idx (one-thread)
    using ThrID = Layout<_1>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, _128>>; // 128-bits
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = SrcLayout;
    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

template <>
struct Copy_Traits<SM90_MULTIMEM_ST_BF16x8> : MULTIMEM_RED_Unpack<SM90_MULTIMEM_ST_BF16x8>
{
    // Logical thread id to thread idx (one-thread)
    using ThrID = Layout<_1>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, _128>>; // 128-bits
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = SrcLayout;
    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

// T = data type
// N = vector size
template <class T, int N>
CUTE_HOST_RTC constexpr auto get_multimem_ldreduce_copy_atom()
{
    // FP16
    if constexpr (cute::is_same_v<T, cutlass::half_t> && N == 8)
    {
        return Copy_Atom<SM90_MULTIMEM_LDREDUCE_F16x8, T>{};
    }
    // BF16
    else if constexpr (cute::is_same_v<T, cutlass::bfloat16_t> && N == 8)
    {
        return Copy_Atom<SM90_MULTIMEM_LDREDUCE_BF16x8, T>{};
    }
    else
    {
        static_assert(dependent_false<T>, "No multimem instruction match.");
    }
}

// T = data type
// N = vector size
template <class T, int N>
CUTE_HOST_RTC constexpr auto get_multimem_st_copy_atom()
{
    // FP16
    if constexpr (cute::is_same_v<T, cutlass::half_t> && N == 8)
    {
        return Copy_Atom<SM90_MULTIMEM_ST_F16x8, T>{};
    }
    // BF16
    else if constexpr (cute::is_same_v<T, cutlass::bfloat16_t> && N == 8)
    {
        return Copy_Atom<SM90_MULTIMEM_ST_BF16x8, T>{};
    }
    else
    {
        static_assert(dependent_false<T>, "No multimem instruction match.");
    }
}

} // namespace cute
