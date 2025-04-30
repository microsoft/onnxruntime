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
/*! \file
  \brief Visitor tree store operations for the sm90 AllReduce TMA warp-specialized (ws) epilogue
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"

#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion
{

using namespace cute;
using namespace detail;

template <int Stages, typename EpilogueTile, typename ElementT, typename StrideMNL, typename SmemLayoutAtom,
    FloatRoundStyle RoundStyle, typename CopyOpR2S, typename TileShape, typename SystemBarrier_, bool OneShot>
struct Sm90AuxAllReduce
{
    using ElementAux = ElementT; // required for compilation
    using SystemBarrier = SystemBarrier_;

    static constexpr int kAlignment = 128 / sizeof_bits_v<ElementT>;
    constexpr static bool is_m_major = epilogue::collective::detail::is_m_major<StrideMNL>();
    // Find the max contiguous layout usable by TMA (if EpilogueTile is a non-compact tiler)
    // This should not be needed... {$nv-internal-release}
    using SmemShapeTma = decltype(make_shape(
        max_common_vector(make_layout(get<0>(EpilogueTile{})), make_layout(get<0>(EpilogueTile{}))),
        max_common_vector(make_layout(get<1>(EpilogueTile{})), make_layout(get<1>(EpilogueTile{})))));
    using SmemLayoutTma = decltype(tile_to_shape(
        SmemLayoutAtom{}, SmemShapeTma{}, cute::conditional_t<is_m_major, Step<_2, _1>, Step<_1, _2>>{}));
    using SmemLayout = decltype(tile_to_shape(SmemLayoutTma{},
        make_shape(size<0>(shape(EpilogueTile{})), size<1>(shape(EpilogueTile{})), Int<Stages>{}),
        cute::conditional_t<is_m_major, Step<_2, _1, _3>, Step<_1, _2, _3>>{}));

    struct SharedStorage
    {
        alignas(
            cutlass::detail::alignment_for_swizzle(SmemLayout{})) array_aligned<ElementT, size(SmemLayout{})> smem_aux;
    };

    struct Arguments
    {
        ElementT* multicast_ptr_aux = nullptr;
        ElementT* unicast_ptr_aux = nullptr;
        StrideMNL dAux = {};
        typename SystemBarrier::Params barrier_params;
        int rank = 0;
        int world_size = 1;
    };

    static constexpr auto get_TMA_store_op()
    {
        if constexpr (OneShot)
        {
            return SM90_TMA_REDUCE_ADD{};
        }
        else
        {
            return SM90_TMA_STORE{};
        }
    }

    struct Params
    {
        using TMA_Aux = decltype(make_tma_copy(get_TMA_store_op(),
            make_tensor(static_cast<ElementT*>(nullptr), repeat_like(StrideMNL{}, int32_t(0)), StrideMNL{}),
            SmemLayoutTma{}));
        TMA_Aux tma_store_aux;
        ElementT* multicast_ptr_aux;         // for MC instructions
        StrideMNL dAux;
        Layout<Shape<int, int>> tile_layout; // (TILE_M, TILE_N)
        typename SystemBarrier::Params barrier_params;
        int rank;
        int world_size;
    };

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(
        ProblemShape const& problem_shape, Arguments const& args, void* workspace)
    {
        // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
        auto problem_shape_mnkl = append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_mnkl;

        auto dst_ptr = OneShot ? args.multicast_ptr_aux : args.unicast_ptr_aux;
        Tensor tensor_aux = make_tensor(dst_ptr, make_layout(make_shape(M, N, L), args.dAux));
        typename Params::TMA_Aux tma_store_aux = make_tma_copy(get_TMA_store_op(), tensor_aux, SmemLayoutTma{});

        int m_tiles = ceil_div(M, size<0>(TileShape{}));
        int n_tiles = ceil_div(N, size<1>(TileShape{}));
        auto tile_layout = make_layout(make_shape(m_tiles, n_tiles));

        return {tma_store_aux, args.multicast_ptr_aux, args.dAux, tile_layout, args.barrier_params, args.rank,
            args.world_size};
    }

    template <class ProblemShape>
    static bool can_implement(ProblemShape const& problem_shape, Arguments const& args)
    {
        return true;
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args)
    {
        return 0;
    }

    template <class ProblemShape>
    static cutlass::Status initialize_workspace(ProblemShape const& problem_shape, Arguments const& args,
        void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr)
    {
        return cutlass::Status::kSuccess;
    }

    CUTLASS_HOST_DEVICE
    Sm90AuxAllReduce() {}

    CUTLASS_HOST_DEVICE
    Sm90AuxAllReduce(Params const& params, SharedStorage const& shared_storage)
        : params_ptr(&params)
        , smem_aux(const_cast<ElementT*>(shared_storage.smem_aux.data()))
    {
    }

    Params const* params_ptr; // pointer to Params from kernel(Params) (constant mem)
    ElementT* smem_aux;

    CUTLASS_DEVICE bool is_producer_load_needed() const
    {
        return false;
    }

    CUTLASS_DEVICE bool is_C_load_needed() const
    {
        return false;
    }

    template <class... Args>
    CUTLASS_DEVICE auto get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args)
    {
        return EmptyProducerLoadCallbacks{};
    }

    template <int NumThreads, // number of threads that cooperatively execute this EVT node function
        class RTensor, class TiledR2S, class STensorR2S, class STensorS2G, class GTensorS2G, class ProblemShapeMNL,
        class TileCoordMNL>
    struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks
    {
        CUTLASS_DEVICE
        ConsumerStoreCallbacks(RTensor&& tC_rAux, TiledR2S tiled_r2s, STensorR2S&& tRS_sAux, STensorS2G&& bSG_sAux,
            GTensorS2G&& bSG_gAux, Params const* params_ptr, ProblemShapeMNL problem_shape_mnl,
            TileCoordMNL tile_coord_mnl, int const thread_idx)
            : issued_tma_store(false)
            , tiled_r2s(tiled_r2s)
            , tC_rAux(cute::forward<RTensor>(tC_rAux))
            , tRS_sAux(cute::forward<STensorR2S>(tRS_sAux))
            , bSG_sAux(cute::forward<STensorS2G>(bSG_sAux))
            , bSG_gAux(cute::forward<GTensorS2G>(bSG_gAux))
            , problem_shape_mnl(problem_shape_mnl)
            , tile_coord_mnl(tile_coord_mnl)
            , thread_idx(thread_idx)
            , params_ptr(params_ptr)
        {
        }

        bool issued_tma_store;
        TiledR2S tiled_r2s;
        RTensor tC_rAux;     // (CPY,CPY_M,CPY_N)
        STensorR2S tRS_sAux; // (R2S,R2S_M,R2S_N,PIPE)
        STensorS2G bSG_sAux; // (S2G,S2G_M,S2G_N,PIPE)
        GTensorS2G bSG_gAux; // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)
        ProblemShapeMNL problem_shape_mnl;
        TileCoordMNL tile_coord_mnl;
        int thread_idx;
        Params const* params_ptr;

        // Wait until at most Count committed TMA_STOREs are pending and all prior commits are complete and visible in
        // gmem
        template <int Count>
        CUTLASS_DEVICE static void tma_store_wait()
        {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
            asm volatile("cp.async.bulk.wait_group %0;" : : "n"(Count) : "memory");
#endif
        }

        template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
        CUTLASS_DEVICE auto visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m,
            int epi_n, Array<ElementInput, FragmentSize> const& frg_input)
        {
            using ConvertInput = NumericArrayConverter<ElementT, ElementInput, FragmentSize, RoundStyle>;
            ConvertInput convert_input{};

            Tensor tC_rAux_frg = recast<Array<ElementT, FragmentSize>>(coalesce(tC_rAux)); // (EPI_V)
            tC_rAux_frg(epi_v) = convert_input(frg_input);

            return frg_input;
        }

        CUTLASS_DEVICE void postreduce(int epi_m, int epi_n, int store_iteration, bool issue_smem_store)
        {

            using RLayoutR2S = decltype(cute::layout(TiledR2S{}.get_slice(0).retile_S(RTensor{})));
            Tensor tRS_rAux = make_tensor(tC_rAux.data(), RLayoutR2S{}); // (R2S,R2S_M,R2S_N)

            if (issue_smem_store)
            {
                int store_pipe_index = store_iteration % Stages;
                copy(tiled_r2s, tRS_rAux, tRS_sAux(_, _, _, store_pipe_index));
            }
        }

        CUTLASS_DEVICE void tma_store(int epi_m, int epi_n, int store_iteration, bool issue_tma_store)
        {
            if (issue_tma_store && thread_idx == 0)
            {
                // Issue the TMA store
                int store_pipe_index = store_iteration % Stages;
                copy(params_ptr->tma_store_aux, bSG_sAux(_, _, _, store_pipe_index), bSG_gAux(_, _, _, epi_m, epi_n));
            }
            issued_tma_store = issue_tma_store;
        }

        // Tile end
        CUTLASS_DEVICE void end()
        {
            if constexpr (OneShot)
            {
                return;
            }

            auto [m, n, l] = tile_coord_mnl;
            if (m >= size<0>(params_ptr->tile_layout.shape()) || n >= size<1>(params_ptr->tile_layout.shape()))
            {
                // early exit if out of bound
                return;
            }

            if (params_ptr->world_size == 1)
            {
                return; // single-GPU doesn't need AR
            }

            // if (issued_tma_store)
            // {
            //     assert(params_ptr->world_size <= warpSize);
            // Process for ensuring TMA store is visible to all threads in (g)mem.
            // 1. Issue TMA op                (executing thread)
            // 2. cp.async.bulk.commit_group  (executing thread)
            // 3. cp.async.bulk.wait_group    (executing thread)
            // 4. thread synchronize          (all threads)
            tma_store_wait<0>();

            int tile_idx = params_ptr->tile_layout(m, n);
            SystemBarrier::arrive_inc(
                params_ptr->barrier_params, thread_idx, tile_idx, params_ptr->rank, params_ptr->world_size);
        }
    };

    template <bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
        class... Args>
    CUTLASS_DEVICE auto get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args)
    {

        auto [M, N, K, L] = args.problem_shape_mnkl;
        auto [m, n, k, l] = args.tile_coord_mnkl;

        auto problem_shape_mnl = make_shape(M, N, L);
        auto tile_coord_mnl = make_coord(m, n, l);

        Tensor mAux = params_ptr->tma_store_aux.get_tma_tensor(problem_shape_mnl);       // (M,N,L)
        Tensor gAux = local_tile(mAux, take<0, 2>(args.tile_shape_mnk), tile_coord_mnl); // (CTA_M,CTA_N)

        Tensor tC_gAux = sm90_partition_for_epilogue<ReferenceSrc>(         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
            gAux, args.epi_tile, args.tiled_copy, args.thread_idx);
        Tensor tC_rAux = make_tensor<ElementT>(take<0, 3>(shape(tC_gAux))); // (CPY,CPY_M,CPY_N)

        Tensor sAux_epi = cute::as_position_independent_swizzle_tensor(
            make_tensor(make_smem_ptr(smem_aux), SmemLayout{})); // (EPI_TILE_M,EPI_TILE_N,PIPE)
        Tensor gAux_epi = flat_divide(gAux, args.epi_tile);      // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

        auto tiled_r2s
            = conditional_return<ReferenceSrc>(make_tiled_copy_S(Copy_Atom<CopyOpR2S, ElementT>{}, args.tiled_copy),
                make_tiled_copy_D(Copy_Atom<CopyOpR2S, ElementT>{}, args.tiled_copy));
        auto tRS_sAux = tiled_r2s.get_slice(args.thread_idx).partition_D(sAux_epi); // (R2S,R2S_M,R2S_N,PIPE)

        ThrCopy thrblk_s2g = params_ptr->tma_store_aux.get_slice(_0{});
        Tensor bSG_sAux = thrblk_s2g.partition_S(sAux_epi);          // (TMA,TMA_M,TMA_N,PIPE)
        Tensor bSG_gAux = thrblk_s2g.partition_D(gAux_epi);          // (TMA,TMA_M,TMA_N,EPI_M,EPI_N)

        constexpr int NumThreads = size(decltype(args.tiled_mma){}); // sync threads

        return ConsumerStoreCallbacks<NumThreads, decltype(tC_rAux), decltype(tiled_r2s), decltype(tRS_sAux),
            decltype(bSG_sAux), decltype(bSG_gAux), decltype(problem_shape_mnl), decltype(tile_coord_mnl)>(
            cute::move(tC_rAux), tiled_r2s, cute::move(tRS_sAux), cute::move(bSG_sAux), cute::move(bSG_gAux),
            params_ptr, problem_shape_mnl, tile_coord_mnl, args.thread_idx);
    }
};

// D = AllReduce(activation(alpha * acc + beta * C))
template <bool IsOneShot_, class SystemBarrier_, class GmemLayoutTagAux_, class ElementAux_, class ElementCompute_,
    class ElementSource_ = ElementAux_, class ElementScalar_ = ElementCompute_,
    FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest>
struct Sm90LinCombAuxAllReduce
    : LinearCombination<ElementAux_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>
{
    using ElementAux = ElementAux_;
    using GmemLayoutTagAux = GmemLayoutTagAux_;
    static constexpr int AlignmentAux = 128 / cute::sizeof_bits_v<ElementAux_>;
    static constexpr bool IsAuxOutSupported = true;
};

template <int StagesD, bool IsOneShot, class SystemBarrier, class GmemLayoutTagOutput, class ElementOutput,
    class ElementCompute, class ElementSource, class ElementScalar, FloatRoundStyle RoundStyle, class CtaTileShapeMNK,
    class EpilogueTile, class SmemLayoutAtom, class CopyOpR2S>
using Sm90LinearCombAuxAllReduce
    = Sm90EVT<Sm90AuxAllReduce<StagesD, EpilogueTile, ElementOutput, cutlass::gemm::TagToStrideC_t<GmemLayoutTagOutput>,
                  SmemLayoutAtom, RoundStyle, CopyOpR2S, CtaTileShapeMNK, SystemBarrier, IsOneShot>,   // Aux AR
        Sm90LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C +
                                                                                                       // (alpha * acc)
        >;

template <
    // Dispatch policy arguments
    int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
    // Fusion Op arguments
    bool IsOneShot, class SystemBarrier, class GmemLayoutTagD, class ElementD, class ElementCompute, class ElementC,
    class ElementScalar, FloatRoundStyle RoundStyle,
    // Epilogue arguments
    class CtaTileShapeMNK, class EpilogueTile, class SmemLayoutAtom, class CopyOpR2S>
struct FusionCallbacks<epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    Sm90LinCombAuxAllReduce<IsOneShot, SystemBarrier, GmemLayoutTagD, ElementD, ElementCompute, ElementC, ElementScalar,
        RoundStyle>,
    CtaTileShapeMNK, EpilogueTile, SmemLayoutAtom, CopyOpR2S>
    : Sm90LinearCombAuxAllReduce<StagesD, IsOneShot, SystemBarrier, GmemLayoutTagD, ElementD, ElementCompute, ElementC,
          ElementScalar, RoundStyle, CtaTileShapeMNK, EpilogueTile, SmemLayoutAtom, CopyOpR2S>
{

    using Impl = Sm90LinearCombAuxAllReduce<StagesD, IsOneShot, SystemBarrier, GmemLayoutTagD, ElementD, ElementCompute,
        ElementC, ElementScalar, RoundStyle, CtaTileShapeMNK, EpilogueTile, SmemLayoutAtom, CopyOpR2S>;
    using Operation = Sm90LinCombAuxAllReduce<IsOneShot, SystemBarrier, GmemLayoutTagD, ElementD, ElementCompute,
        ElementC, ElementScalar, RoundStyle>;

    struct Arguments
    {
        using StrideD = cutlass::gemm::TagToStrideC_t<GmemLayoutTagD>;
        ElementScalar alpha = ElementScalar(1);
        ElementScalar beta = ElementScalar(0);
        ElementScalar const* alpha_ptr = nullptr;
        ElementScalar const* beta_ptr = nullptr;
        ElementD* multicast_ptr_aux = nullptr;
        ElementD* ptr_aux = nullptr;
        StrideD dAux = {};
        typename SystemBarrier::Params barrier_params{};
        int rank = 0;
        int num_ranks = 1;
        using StrideAlpha = Stride<_0, _0, int64_t>;
        using StrideBeta = Stride<_0, _0, int64_t>;
        StrideAlpha dAlpha = {_0{}, _0{}, 0};
        StrideBeta dBeta = {_0{}, _0{}, 0};

        operator typename Impl::Arguments() const
        {
            return {{
                        // ternary op : beta * C + (alpha * acc)
                        {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
                        {},                            // leaf args : C
                        {
                            // binary op : alpha * acc
                            {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                            {},                               // leaf args : acc
                            {}                                // binary args : multiplies
                        },                                    // end binary op
                        {}                                    // ternary args : multiply_add
                    },                                        // end ternary op
                {multicast_ptr_aux, ptr_aux, dAux, barrier_params, rank, num_ranks}};
        }
    };

    // Ctor inheritance
    using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
