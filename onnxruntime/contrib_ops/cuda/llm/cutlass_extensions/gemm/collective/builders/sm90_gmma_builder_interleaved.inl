/*
 * Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "cutlass/arch/mma.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

// SM90 Collective Builders should be used only starting CUDA 12.0
#if (__CUDACC_VER_MAJOR__ >= 12)
#define CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective
{

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_RS Mixed Scaled GEMM
template <class ElementPairA_, class GmemLayoutATag_, int AlignmentA, class ElementPairB_, class GmemLayoutBTag_,
    int AlignmentB, class ElementAccumulator, class TileShape_MNK, class ClusterShape_MNK, class StageCountType,
    class KernelScheduleType>
struct CollectiveBuilderInterleaved<arch::Sm90, arch::OpClassTensorOp, ElementPairA_, GmemLayoutATag_, AlignmentA,
    ElementPairB_, GmemLayoutBTag_, AlignmentB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType,
    KernelScheduleType,
    cute::enable_if_t<(cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecialized>
        || cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpong>
        || cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative>)>>
{

private:
    using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementPairA_>;
    using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementPairB_>;
    using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementPairA_>;
    using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementPairB_>;
    static constexpr bool NeitherIsTuple
        = !cute::is_tuple<ElementPairA_>::value && !cute::is_tuple<ElementPairB_>::value;

public:
    using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementPairA_>;
    using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementPairB_>;
    static_assert(cute::is_tuple<ElementPairA_>::value ^ cute::is_tuple<ElementPairB_>::value
            || (NeitherIsTuple && (sizeof_bits<ElementA>::value != sizeof_bits<ElementB>::value)),
        "Either A OR B must be a tuple or the widths of A and B must be different.");

    static constexpr bool IsANarrow = sizeof_bits<ElementA>::value < sizeof_bits<ElementB>::value;

    using GmemLayoutATag = GmemLayoutATag_;
    using GmemLayoutBTag = GmemLayoutBTag_;

    using ElementPairA = cute::conditional_t<IsANarrow && NeitherIsTuple, cute::tuple<ElementA>, ElementPairA_>;
    using ElementPairB = cute::conditional_t<!IsANarrow && NeitherIsTuple, cute::tuple<ElementB>, ElementPairB_>;

    static constexpr bool IsATransformed = cute::is_tuple<ElementPairA>::value;
    using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
    using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;

    static_assert(is_static<TileShape_MNK>::value);
    static_assert(is_static<ClusterShape_MNK>::value);
    static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
        "Should meet TMA alignment requirement\n");
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
    static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
    static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_rs_tag_to_major_A<GmemLayoutATag>();
    static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_rs_tag_to_major_B<GmemLayoutBTag>();
    static constexpr bool IsWarpSpecializedTransposeB = detail::is_warpspecialized_transpose_B<ElementA, GmemLayoutATag,
        ElementB, GmemLayoutBTag, KernelScheduleType>();
    static_assert(!IsWarpSpecializedTransposeB, "Mixed input GEMM does not support WS transpose B.");

    // If A is scaled, then we don't need to swap. Otherwise, we must ensure B goes to RF and we must swap the operands.
    static constexpr bool SwapAB = !IsATransformed;

    // When we relax the above assertion, we must handle setting the tile mma GmmaMajorB correctly.
    static constexpr cute::GMMA::Major TiledMmaGmmaMajorB = SwapAB ? GmmaMajorA : GmmaMajorB;

    using ElementMma = cute::conditional_t<IsATransformed, ElementB, ElementA>;
    using AtomLayoutMNK = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative>,
        Layout<Shape<_2, _1, _1>>, Layout<Shape<_1, _1, _1>>>;

    using TiledMma
        = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<ElementMma, ElementMma, ElementAccumulator,
                                            TileShape_MNK, GMMA::Major::K, TiledMmaGmmaMajorB>(),
            AtomLayoutMNK{}));

    using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
    using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

    using SmemLayoutAtomA
        = decltype(detail::rs_smem_selector<GmmaMajorA, ElementA, decltype(cute::get<0>(TileShape_MNK{})),
            decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());
    using SmemLayoutAtomB
        = decltype(detail::rs_smem_selector<GmmaMajorB, ElementB, decltype(cute::get<1>(TileShape_MNK{})),
            decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());

    using RealElementA = cute::conditional_t<SwapAB, ElementB, ElementA>;
    using RealElementB = cute::conditional_t<SwapAB, ElementA, ElementB>;
    static constexpr int PipelineStages
        = detail::compute_stage_count_or_override_single_affine_transformed_input<detail::sm90_smem_capacity_bytes,
            RealElementA, RealElementB, ElementScale, ElementZero, TileShape_MNK>(StageCountType{});

    using SmemCopyAtomA = cute::conditional_t<SwapAB, void, Copy_Atom<cute::AutoVectorizingCopy, ElementA>>;
    using SmemCopyAtomB = cute::conditional_t<SwapAB, Copy_Atom<cute::AutoVectorizingCopy, ElementB>, void>;

    using DispatchPolicy
        = MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput<PipelineStages, ClusterShape_MNK, KernelScheduleType>;

    // We pack the scale data with the operand that will be optionally scaled and converted before MMA.
    using StrideA = TagToStrideA_t<GmemLayoutATag>;
    using StrideB = TagToStrideB_t<GmemLayoutBTag>;

    using CollectiveOp = CollectiveMmaInterleaved<DispatchPolicy, TileShape_MNK, ElementPairA, StrideA, ElementPairB,
        StrideB, TiledMma, GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity, GmemTiledCopyB,
        SmemLayoutAtomB, SmemCopyAtomB, cute::identity>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
