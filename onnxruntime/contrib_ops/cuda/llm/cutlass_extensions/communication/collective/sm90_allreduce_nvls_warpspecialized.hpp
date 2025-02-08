/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass/cutlass.h"
#include "cutlass_extensions/arch/copy_traits_sm90_multimem.hpp"
#include "cutlass_extensions/system_barrier.h"

namespace cutlass::communication::collective
{
using namespace cute;

template <class ElementT_, int Threads_, class TileShape_, class StrideMNL_, class SystemBarrier_, class LayoutD_,
    bool OneShot_>
class CollectiveAllReduceMulticastWarpSpecialized
{
public:
    // Type aliases
    using ElementT = ElementT_;
    using TileShape = TileShape_;
    using StrideMNL = StrideMNL_;
    using SystemBarrier = SystemBarrier_;

    static constexpr bool OneShot = OneShot_;
    static constexpr int Threads = Threads_;
    static constexpr int VecWidth = 128 / sizeof_bits_v<ElementT>; // multimem has max vec instructions

    static constexpr bool is_m_major = cutlass::is_same_v<LayoutD_, cutlass::layout::ColumnMajor>;

    static constexpr auto get_reduce_tile()
    {
        // Clamp registers per thread to <R>
        constexpr int R = 32;
        constexpr int MaxTileSize = R * Threads;

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

    static_assert(cute::product(ReduceTile{}) % Threads == 0);
    static_assert(cute::product(ReduceTile{}) / Threads >= VecWidth);

    struct Arguments
    {
        ElementT* multicast_ptr_aux = nullptr; // for MC instructions
        ElementT* ptr_aux = nullptr;           // for UC instructions
        StrideMNL stride;
        typename SystemBarrier::Params barrier_params;
        typename SystemBarrier::Params barrier_params_final_sync;
        int rank;
        int world_size;
    };

    struct Params
    {
        ElementT* multicast_ptr_aux = nullptr; // for MC instructions
        ElementT* ptr_aux = nullptr;
        StrideMNL stride;
        typename SystemBarrier::Params barrier_params;
        typename SystemBarrier::Params barrier_params_final_sync;
        int rank;
        int world_size;
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

        return {
            args.multicast_ptr_aux,
            args.ptr_aux,
            args.stride,
            args.barrier_params,
            args.barrier_params_final_sync,
            args.rank,
            args.world_size,
            tile_layout,
        };
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
            constexpr int DimM = cute::min(Threads, size<0>(ReduceTile{}) / VecWidth);
            constexpr int DimN = Threads / DimM;
            static_assert(Threads % DimM == 0);
            static_assert(DimN > 0);

            using ThreadLayout = Layout<Shape<Int<DimM>, Int<DimN>>>; // No stride as col-major by default;
            using ValueLayout = Layout<Shape<Int<VecWidth>>>;

            return make_tiled_copy(CopyAtom{}, ThreadLayout{}, ValueLayout{});
        }
        else // n-major
        {
            constexpr int DimN = cute::min(Threads, size<1>(ReduceTile{}) / VecWidth);
            constexpr int DimM = Threads / DimN;
            static_assert(Threads % DimN == 0);
            static_assert(DimM > 0);

            using ThreadLayout = Layout<Shape<Int<DimM>, Int<DimN>>, Stride<Int<DimN>, _1>>;
            using ValueLayout = Layout<Shape<_1, Int<VecWidth>>, Stride<Int<VecWidth>, _1>>;

            return make_tiled_copy(CopyAtom{}, ThreadLayout{}, ValueLayout{});
        }
    }

    template <class ProblemShapeMNKL, class TileCoordMNKL>
    CUTLASS_DEVICE void tile_global_sync(
        ProblemShapeMNKL const& problem_shape, TileCoordMNKL const& tile_coord, int thread_idx)
    {

        auto [M, N, K, L] = problem_shape;
        auto [m, n, k, l] = tile_coord;

        if (m >= size<0>(params_ptr->tile_layout.shape()) || n >= size<1>(params_ptr->tile_layout.shape()))
        {
            // early exit if out of bound
            return;
        }

        if (params_ptr->world_size == 1)
        {
            return; // single-GPU doesn't need AR
        }

        int tile_index = params_ptr->tile_layout(m, n);

        cutlass::arch::NamedBarrier::sync(Threads, named_barrier);
        // Wait for all multicast writes to be visible to us.
        // This is safe between phases.
        SystemBarrier::arrive_and_wait(
            params_ptr->barrier_params_final_sync, thread_idx, tile_index, params_ptr->rank, params_ptr->world_size);
    }

    template <class ProblemShapeMNKL, class TileCoordMNKL>
    CUTLASS_DEVICE void gather_reduce_broadcast(
        ProblemShapeMNKL const& problem_shape, TileCoordMNKL const& tile_coord, int thread_idx)
    {
        if constexpr (OneShot)
        {
            return; // Nothing to do.
        }

        if (params_ptr->world_size == 1)
        {
            return; // single-GPU doesn't need AR
        }

        auto [M, N, K, L] = problem_shape;
        auto [m, n, k, l] = tile_coord;

        if (m >= size<0>(params_ptr->tile_layout.shape()) || n >= size<1>(params_ptr->tile_layout.shape()))
        {
            // early exit if out of bound
            return;
        }

        int tile_index = params_ptr->tile_layout(m, n);
        int tiles_per_rank = cute::ceil_div(cute::product(params_ptr->tile_layout.shape()), params_ptr->world_size);

        // Wait for the tile to be ready across all ranks
        SystemBarrier::wait_eq_reset(
            params_ptr->barrier_params, thread_idx, tile_index, params_ptr->rank, params_ptr->world_size);

        if (tile_index / tiles_per_rank != params_ptr->rank)
        {
            // not our tile to process
            return;
        }

        // Synchronize threads to ensure TMA stores of D across all ranks are visible to all threads
        cutlass::arch::NamedBarrier::sync(Threads, named_barrier);

        // Setup tensors
        Tensor mAux = make_tensor(
            params_ptr->multicast_ptr_aux, make_layout(make_shape(M, N, L), params_ptr->stride)); // (M,N,L)
        Tensor gAux = local_tile(mAux, take<0, 2>(TileShape{}), make_coord(m, n, l));             // (TILE_M,TILE_N)
        Tensor gAux_red = flat_divide(gAux, ReduceTile{}); // (RED_TILE_M,RED_TILE_N,RED_M,RED_N)

        // Predication tensor
        Tensor coordAux = make_identity_tensor(shape(mAux));
        Tensor pAux = local_tile(coordAux, take<0, 2>(TileShape{}), make_coord(m, n, l)); // (CTA_M,CTA_N)
        Tensor pAux_red = flat_divide(pAux, ReduceTile{}); // (RED_TILE_M,RED_TILE_N,RED_M,RED_N)

        using CopyAtomG2R = decltype(get_multimem_ldreduce_copy_atom<ElementT, VecWidth>()); // reduce in switch
        using CopyAtomR2G = decltype(get_multimem_st_copy_atom<ElementT, VecWidth>());       // multicast store

        auto tiled_g2r = make_AR_tiled_copy<CopyAtomG2R>();
        auto tiled_r2g = make_AR_tiled_copy<CopyAtomR2G>();

        auto thread_g2r = tiled_g2r.get_slice(thread_idx);
        auto thread_r2g = tiled_r2g.get_slice(thread_idx);

        Tensor tGR_pAux_red = thread_g2r.partition_S(pAux_red); // ((Atom,AtomNum),TiledCopy_M,TiledCopy_N,RED_M,RED_N)
        Tensor tGR_gAux_red = thread_g2r.partition_S(gAux_red); // ((Atom,AtomNum),TiledCopy_M,TiledCopy_N,RED_M,RED_N)
        Tensor tRG_gAux_red = thread_r2g.partition_D(gAux_red); // ((Atom,AtomNum),TiledCopy_M,TiledCopy_N,RED_M,RED_N)

        CUTLASS_PRAGMA_UNROLL
        for (int red_n = 0; red_n < size<3>(gAux_red); ++red_n)
        {
            CUTLASS_PRAGMA_UNROLL
            for (int red_m = 0; red_m < size<2>(gAux_red); ++red_m)
            {
                constexpr int V = VecWidth;
                auto Vec = coalesce(Layout<Shape<_1, Int<V>>, Stride<Int<V>, _1>>{});

                // Predication happens on units of Vec
                Tensor tGR_gAux_red_vec = zipped_divide(tGR_gAux_red(_, _, _, red_m, red_n), Vec); // (Vec, Rest)
                Tensor tGR_pAux_red_vec = zipped_divide(tGR_pAux_red(_, _, _, red_m, red_n), Vec); // (Vec, Rest)
                Tensor tRG_gAux_red_vec = zipped_divide(tRG_gAux_red(_, _, _, red_m, red_n), Vec); // (Vec, Rest)

                // problem shape bounds check
                auto pred_fn = [&](auto const&... coords)
                { return elem_less(tGR_pAux_red_vec(Int<0>{}, coords...), problem_shape); };

                Tensor fragment = make_fragment_like(tRG_gAux_red_vec);
                cute::copy_if(CopyAtomG2R{}, pred_fn, tGR_gAux_red_vec, fragment); // (g)mem -> reduce -> reg
                cute::copy_if(CopyAtomR2G{}, pred_fn, fragment, tRG_gAux_red_vec); // reg -> (g)mem
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
