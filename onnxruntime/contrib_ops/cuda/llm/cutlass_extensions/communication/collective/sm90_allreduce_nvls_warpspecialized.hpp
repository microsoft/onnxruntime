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

#include "cutlass/cutlass.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/arch/copy_traits_sm90_multimem.hpp"
#include "contrib_ops/cuda/llm/cutlass_extensions/system_barrier.h"

namespace cutlass::communication::collective
{
using namespace cute;

template <class ElementT_, int ThreadCount_, int Unroll_, class TileShape_, class StrideMNL_, class SystemBarrier_,
    class LayoutD_, bool OneShot_>
class CollectiveAllReduceMulticastWarpSpecialized
{
public:
    // Type aliases
    using ElementT = ElementT_;
    using TileShape = TileShape_;
    using StrideMNL = StrideMNL_;
    using SystemBarrier = SystemBarrier_;

    static constexpr bool OneShot = OneShot_;
    static constexpr int ThreadCount = ThreadCount_;
    static constexpr int VecWidth = 128 / sizeof_bits_v<ElementT>; // multimem has max vec instructions
    static constexpr int MaxRanksPerCollective = 8;

    static constexpr bool is_m_major = cutlass::is_same_v<LayoutD_, cutlass::layout::ColumnMajor>;

    static constexpr auto get_reduce_tile()
    {
        // Clamp registers per thread to <R>
        constexpr int R = VecWidth * Unroll_;
        constexpr int MaxTileSize = R * ThreadCount;

        if constexpr (is_m_major)
        {
            constexpr int ReduceTileM = size<0>(TileShape{});
            static_assert(MaxTileSize % ReduceTileM == 0);

            constexpr int ReduceTileN = cute::min(size<1>(TileShape{}), MaxTileSize / ReduceTileM);
            return Shape<Int<ReduceTileM>, Int<ReduceTileN>>{};
        }
        else
        {
            constexpr int ReduceTileN = size<1>(TileShape{});
            static_assert(MaxTileSize % ReduceTileN == 0);

            constexpr int ReduceTileM = cute::min(size<0>(TileShape{}), MaxTileSize / ReduceTileN);
            return Shape<Int<ReduceTileM>, Int<ReduceTileN>>{};
        }
    }

    using ReduceTile = decltype(get_reduce_tile());

    static_assert(cute::product(ReduceTile{}) % ThreadCount == 0);
    static_assert(cute::product(ReduceTile{}) / ThreadCount >= VecWidth);

    struct Arguments
    {
        ElementT* multicast_ptr_aux = nullptr; // for MC instructions
        ElementT** ipc_ptr_aux = nullptr;      // for UC instructions
        StrideMNL stride;
        typename SystemBarrier::Params barrier_params;
        typename SystemBarrier::Params barrier_params_final_sync;
        int rank;
        int world_size;
        bool switch_reduction;
    };

    struct Params
    {
        ElementT* multicast_ptr_aux = nullptr;
        ElementT* ipc_ptr_aux[MaxRanksPerCollective];
        StrideMNL stride;
        typename SystemBarrier::Params barrier_params;
        typename SystemBarrier::Params barrier_params_final_sync;
        int rank;
        int world_size;
        bool switch_reduction;
        Layout<Shape<int, int>> tile_layout;
    };

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args)
    {
        // Append 1s until problem shape is rank-4
        auto problem_shape_mnkl = append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_mnkl;

        int m_tiles = ceil_div(M, size<0>(TileShape{}));
        int n_tiles = ceil_div(N, size<1>(TileShape{}));
        auto tile_layout = make_layout(make_shape(m_tiles, n_tiles));

        Params params;
        params.multicast_ptr_aux = args.multicast_ptr_aux;
        if (args.ipc_ptr_aux != nullptr)
        {
            for (int i = 0; i < args.world_size; i++)
            {
                params.ipc_ptr_aux[i] = args.ipc_ptr_aux[i];
            }
        }
        params.stride = args.stride;
        params.barrier_params = args.barrier_params, params.barrier_params_final_sync = args.barrier_params_final_sync;
        params.rank = args.rank;
        params.world_size = args.world_size;
        params.switch_reduction = args.switch_reduction;
        params.tile_layout = tile_layout;
        return params;
    }

    Params const* params_ptr;
    uint32_t named_barrier;

    CUTLASS_HOST_DEVICE
    CollectiveAllReduceMulticastWarpSpecialized() {}

    CUTLASS_HOST_DEVICE
    CollectiveAllReduceMulticastWarpSpecialized(Params const& params, uint32_t named_barrier)
        : params_ptr(&params)
        , named_barrier(named_barrier)
    {
    }

    template <typename CopyAtom>
    constexpr auto make_AR_tiled_copy()
    {
        if constexpr (is_m_major)
        {
            constexpr int DimM = cute::min(ThreadCount, size<0>(ReduceTile{}) / VecWidth);
            constexpr int DimN = ThreadCount / DimM;
            static_assert(ThreadCount % DimM == 0);
            static_assert(DimN > 0);

            using ThreadLayout = Layout<Shape<Int<DimM>, Int<DimN>>>; // No stride as col-major by default;
            using ValueLayout = Layout<Shape<Int<VecWidth>>>;

            return make_tiled_copy(CopyAtom{}, ThreadLayout{}, ValueLayout{});
        }
        else // n-major
        {
            constexpr int DimN = cute::min(ThreadCount, size<1>(ReduceTile{}) / VecWidth);
            constexpr int DimM = ThreadCount / DimN;
            static_assert(ThreadCount % DimN == 0);
            static_assert(DimM > 0);

            using ThreadLayout = Layout<Shape<Int<DimM>, Int<DimN>>, Stride<Int<DimN>, _1>>;
            using ValueLayout = Layout<Shape<_1, Int<VecWidth>>, Stride<Int<VecWidth>, _1>>;

            return make_tiled_copy(CopyAtom{}, ThreadLayout{}, ValueLayout{});
        }
    }

    // Out-of-bounds check
    CUTLASS_DEVICE bool tile_valid(int m, int n)
    {
        auto tiles_mn = params_ptr->tile_layout.shape();
        return m < size<0>(tiles_mn) && n < size<1>(tiles_mn);
    }

    // Determines which 1/Nth of tiles each rank should process
    CUTLASS_DEVICE bool should_process_tile(int m, int n)
    {
        int tile_index = params_ptr->tile_layout(m, n);
        if constexpr (is_m_major)
        {
            int tiles_per_rank = cute::ceil_div(cute::product(params_ptr->tile_layout.shape()), params_ptr->world_size);
            return tile_index / tiles_per_rank == params_ptr->rank;
        }
        else
        {
            return tile_index % params_ptr->world_size == params_ptr->rank;
        }
    }

    CUTLASS_DEVICE void sync_threads()
    {
        cutlass::arch::NamedBarrier::sync(ThreadCount, named_barrier);
    }

    template <class ProblemShapeMNKL, class TileCoordMNKL>
    CUTLASS_DEVICE void tile_global_sync(
        ProblemShapeMNKL const& problem_shape, TileCoordMNKL const& tile_coord, int thread_idx)
    {
        auto [M, N, K, L] = problem_shape;
        auto [m, n, k, l] = tile_coord;

        if (!tile_valid(m, n) || params_ptr->world_size == 1)
        {
            return; // nothing to do
        }

        int tile_index = params_ptr->tile_layout(m, n);

        sync_threads();

        // Wait for all multicast writes to be visible to us.
        // This is safe between phases.
        SystemBarrier::arrive_and_wait(
            params_ptr->barrier_params_final_sync, thread_idx, tile_index, params_ptr->rank, params_ptr->world_size);
    }

    template <class ProblemShapeMNKL, class TileCoordMNKL>
    CUTLASS_DEVICE void gather_reduce_broadcast(
        ProblemShapeMNKL const& problem_shape, TileCoordMNKL const& tile_coord, int thread_idx)
    {
        if (params_ptr->switch_reduction)
        {
            allreduce_in_switch(problem_shape, tile_coord, thread_idx);
        }
        else
        {
            allreduce_in_sm(problem_shape, tile_coord, thread_idx);
        }
    }

    template <class ProblemShapeMNKL, class TileCoordMNKL>
    CUTLASS_DEVICE void allreduce_in_switch(
        ProblemShapeMNKL const& problem_shape, TileCoordMNKL const& tile_coord, int thread_idx)
    {
        if constexpr (OneShot)
        {
            return; // Nothing to do.
        }

        auto [M, N, K, L] = problem_shape;
        auto [m, n, k, l] = tile_coord;

        if (!tile_valid(m, n) || params_ptr->world_size == 1)
        {
            return; // nothing to do
        }

        int tile_index = params_ptr->tile_layout(m, n);

        // Wait for the tile to be ready across all ranks
        SystemBarrier::wait_eq_reset(
            params_ptr->barrier_params, thread_idx, tile_index, params_ptr->rank, params_ptr->world_size);

        if (!should_process_tile(m, n))
        {
            return; // nothing to do
        }

        // Synchronize threads to ensure TMA stores of D across all ranks are visible to all threads
        sync_threads();

        // Setup tensors
        Tensor mD_mc = make_tensor(
            params_ptr->multicast_ptr_aux, make_layout(make_shape(M, N, L), params_ptr->stride)); // (M,N,L)
        Tensor gD_mc = local_tile(mD_mc, take<0, 2>(TileShape{}), make_coord(m, n, l));           // (TILE_M,TILE_N)
        Tensor gD_mc_red = flat_divide(gD_mc, ReduceTile{}); // (RED_TILE_M,RED_TILE_N,RED_M,RED_N)

        // Predication tensor
        Tensor coordD = make_identity_tensor(shape(mD_mc));
        Tensor pD = local_tile(coordD, take<0, 2>(TileShape{}), make_coord(m, n, l)); // (CTA_M,CTA_N)
        Tensor pD_red = flat_divide(pD, ReduceTile{}); // (RED_TILE_M,RED_TILE_N,RED_M,RED_N)

        using CopyAtomG2R = decltype(get_multimem_ldreduce_copy_atom<ElementT, VecWidth>()); // reduce in switch
        using CopyAtomR2G = decltype(get_multimem_st_copy_atom<ElementT, VecWidth>());       // multicast store

        auto tiled_g2r = make_AR_tiled_copy<CopyAtomG2R>();
        auto tiled_r2g = make_AR_tiled_copy<CopyAtomR2G>();

        auto thread_g2r = tiled_g2r.get_slice(thread_idx);
        auto thread_r2g = tiled_r2g.get_slice(thread_idx);

        Tensor tGR_pD = thread_g2r.partition_S(pD_red);    // ((Atom,AtomNum),TiledCopy_M,TiledCopy_N,RED_M,RED_N)
        Tensor tGR_gD = thread_g2r.partition_S(gD_mc_red); // ((Atom,AtomNum),TiledCopy_M,TiledCopy_N,RED_M,RED_N)
        Tensor tRG_gD = thread_r2g.partition_D(gD_mc_red); // ((Atom,AtomNum),TiledCopy_M,TiledCopy_N,RED_M,RED_N)

        // Allocate intermediate registers for a single subtile
        Tensor tGR_rD = make_fragment_like(tGR_gD(_, _, _, 0, 0)); // ((G2R,G2R_V),G2R_M,G2R_N)

        // problem shape bounds check
        auto pred_fn = [&](auto const&... coords) { return elem_less(tGR_pD(coords...), problem_shape); };

        // reduce subtile loop
        CUTLASS_PRAGMA_UNROLL
        for (int red_n = 0; red_n < size<3>(gD_mc_red); ++red_n)
        {
            // reduce subtile loop
            CUTLASS_PRAGMA_UNROLL
            for (int red_m = 0; red_m < size<2>(gD_mc_red); ++red_m)
            {
                // load-reduce in switch
                cute::copy_if(CopyAtomG2R{}, pred_fn, tGR_gD(_, _, _, red_m, red_n), tGR_rD);
                // store switch multicast
                cute::copy_if(CopyAtomR2G{}, pred_fn, tGR_rD, tRG_gD(_, _, _, red_m, red_n));
            }
        }
    }

    template <class ProblemShapeMNKL, class TileCoordMNKL>
    CUTLASS_DEVICE void allreduce_in_sm(
        ProblemShapeMNKL const& problem_shape, TileCoordMNKL const& tile_coord, int thread_idx)
    {
        if constexpr (OneShot)
        {
            return; // Nothing to do.
        }

        auto [M, N, K, L] = problem_shape;
        auto [m, n, k, l] = tile_coord;

        if (!tile_valid(m, n) || params_ptr->world_size == 1)
        {
            return; // nothing to do
        }

        int tile_index = params_ptr->tile_layout(m, n);

        // Wait for the tile to be ready across all ranks
        SystemBarrier::wait_eq_reset(
            params_ptr->barrier_params, thread_idx, tile_index, params_ptr->rank, params_ptr->world_size);

        if (!should_process_tile(m, n))
        {
            return; // nothing to do
        }

        sync_threads();

        // load/store in ring-order to reduce NVL bus contention
        auto get_step_D_ptr = [&](int step) CUTLASS_LAMBDA_FUNC_INLINE
        {
            int rank = params_ptr->rank + step + 1;
            if (rank >= params_ptr->world_size)
            {
                rank -= params_ptr->world_size;
            }
            return params_ptr->ipc_ptr_aux[rank];
        };

        // Setup tensors
        Tensor mD0 = make_tensor(get_step_D_ptr(0), make_layout(make_shape(M, N, L), params_ptr->stride)); // (M,N,L)
        Tensor gD0 = local_tile(mD0, take<0, 2>(TileShape{}), make_coord(m, n, l)); // (TILE_M,TILE_N)
        Tensor gD0_red = flat_divide(gD0, ReduceTile{}); // (RED_TILE_M,RED_TILE_N,RED_M,RED_N)

        // Predication tensor
        Tensor coordD = make_identity_tensor(shape(mD0));
        Tensor pD = local_tile(coordD, take<0, 2>(TileShape{}), make_coord(m, n, l)); // (CTA_M,CTA_N)
        Tensor pD_red = flat_divide(pD, ReduceTile{}); // (RED_TILE_M,RED_TILE_N,RED_M,RED_N)

        using CopyAtomG2R = Copy_Atom<UniversalCopy<AlignedArray<ElementT, VecWidth>>, ElementT>;

        auto tiled_g2r = make_AR_tiled_copy<CopyAtomG2R>();
        auto thread_g2r = tiled_g2r.get_slice(thread_idx);

        Tensor tGR_pD = thread_g2r.partition_S(pD_red);   // ((Atom,AtomNum),TiledCopy_M,TiledCopy_N,RED_M,RED_N)
        Tensor tGR_gD0 = thread_g2r.partition_S(gD0_red); // ((Atom,AtomNum),TiledCopy_M,TiledCopy_N,RED_M,RED_N)

        // Allocate intermediate registers for a single subtile
        Tensor tGR_rD0_mn = make_fragment_like(tGR_gD0(_, _, _, 0, 0)); // ((G2R,G2R_V),G2R_M,G2R_N)
        Tensor tGR_rD1_mn = make_fragment_like(tGR_gD0(_, _, _, 0, 0)); // ((G2R,G2R_V),G2R_M,G2R_N)

        // partition ptr offset
        auto gmem_offset = static_cast<ptrdiff_t>(tGR_gD0.data() - get_step_D_ptr(0));

        // problem shape bounds check
        auto pred_fn = [&](auto const&... coords) CUTLASS_LAMBDA_FUNC_INLINE
        { return elem_less(tGR_pD(coords...), problem_shape); };

        // reduce subtile loop
        CUTLASS_PRAGMA_UNROLL
        for (int red_n = 0; red_n < size<3>(gD0_red); ++red_n)
        {
            // reduce subtile loop
            CUTLASS_PRAGMA_UNROLL
            for (int red_m = 0; red_m < size<2>(gD0_red); ++red_m)
            {
                // init tGR_rD0 with first iteration to prevent filling with zeros
                cute::copy_if(CopyAtomG2R{}, pred_fn, tGR_gD0(_, _, _, red_m, red_n), tGR_rD0_mn);

                // load D tile from each rank and accumulate.
                for (int step = 1; step < params_ptr->world_size; ++step)
                {
                    Tensor tGR_gD = make_tensor(get_step_D_ptr(step) + gmem_offset, tGR_gD0.layout());

                    // Load D tile
                    cute::copy_if(CopyAtomG2R{}, pred_fn, tGR_gD(_, _, _, red_m, red_n), tGR_rD1_mn);

                    // Reduce
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < size(tGR_rD0_mn); i++)
                    {
                        tGR_rD0_mn(i) += tGR_rD1_mn(i);
                    }
                }

                // store D tile to each rank.
                for (int step = 0; step < params_ptr->world_size; ++step)
                {
                    Tensor tRG_gD = make_tensor(get_step_D_ptr(step) + gmem_offset, tGR_gD0.layout());
                    cute::copy_if(CopyAtomG2R{}, pred_fn, tGR_rD0_mn, tRG_gD(_, _, _, red_m, red_n));
                }
            }
        }
    }

    static int get_num_barrier_flags(int const problem_m, int const problem_n)
    {
        int tile_m = get<0>(TileShape{});
        int tile_n = get<1>(TileShape{});
        int n_tiles = cute::ceil_div(problem_m, tile_m) * cute::ceil_div(problem_n, tile_n);
        // Each tile needs a barrier flag
        return n_tiles;
    }
};

} // namespace cutlass::communication::collective
