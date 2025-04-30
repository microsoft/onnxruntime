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

#include "cute/arch/cluster_sm90.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass/workspace.h"
#include "gemm_universal_allreduce.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel
{

///////////////////////////////////////////////////////////////////////////////

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class CollectiveAllReduce_,
    class TileScheduler_>
class GemmARUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, CollectiveAllReduce_, TileScheduler_,
    cute::enable_if_t<cute::is_base_of_v<KernelTmaWarpSpecializedCooperative,
        typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
    //
    // Type Aliases
    //
    using ProblemShape = ProblemShape_;
    static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
        "ProblemShape{} should be <M,N,K> or <M,N,K,L>");
    // Mainloop derived types
    using CollectiveMainloop = CollectiveMainloop_;
    using TileShape = typename CollectiveMainloop::TileShape;
    using TiledMma = typename CollectiveMainloop::TiledMma;
    using ArchTag = typename CollectiveMainloop::ArchTag;
    using ElementA = typename CollectiveMainloop::ElementA;
    using StrideA = typename CollectiveMainloop::StrideA;
    using ElementB = typename CollectiveMainloop::ElementB;
    using StrideB = typename CollectiveMainloop::StrideB;
    using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
    using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
    using ClusterShape = typename DispatchPolicy::ClusterShape;
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    using MainloopParams = typename CollectiveMainloop::Params;

    // Epilogue derived types
    using CollectiveEpilogue = CollectiveEpilogue_;
    using ElementC = typename CollectiveEpilogue::ElementC;
    using StrideC = typename CollectiveEpilogue::StrideC;
    using ElementD = typename CollectiveEpilogue::ElementD;
    using StrideD = typename CollectiveEpilogue::StrideD;
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams = typename CollectiveEpilogue::Params;

    // AllReduce derived types
    using CollectiveAllReduce = CollectiveAllReduce_;
    using AllReduceArguments = typename CollectiveAllReduce::Arguments;
    using AllReduceParams = typename CollectiveAllReduce::Params;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    using TileSchedulerTag = TileScheduler_;
    using TileScheduler =
        typename detail::TileSchedulerSelector<TileScheduler_, ArchTag, TileShape, ClusterShape>::Scheduler;
    using TileSchedulerArguments = typename TileScheduler::Arguments;
    using TileSchedulerParams = typename TileScheduler::Params;

    static constexpr uint32_t NumLoadWarpGroups = 1;
    static constexpr uint32_t NumMmaWarpGroups = CUTE_STATIC_V(size(TiledMma{})) / NumThreadsPerWarpGroup;
    static constexpr uint32_t MaxThreadsPerBlock
        = CUTE_STATIC_V(size(TiledMma{})) + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

    /// Register requirement for Load and Math WGs
    static constexpr uint32_t LoadRegisterRequirement = 40;
    static constexpr uint32_t MmaRegisterRequirement = 232;

    // 1 stage ordered sequence between mainloop and epilogue producer load threads
    using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1, 2>;

    // Kernel level shared memory storage
    struct SharedStorage
    {
        struct PipelineStorage : cute::aligned_struct<16>
        {
            using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
            using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;

            alignas(16) MainloopPipelineStorage mainloop;
            alignas(16) EpiLoadPipelineStorage epi_load;
            alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
        } pipelines;

        struct TensorStorage : cute::aligned_struct<128>
        {
            using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
            using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

            EpilogueTensorStorage epilogue;
            MainloopTensorStorage mainloop;
        } tensors;
    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // Device side arguments
    struct Arguments
    {
        GemmUniversalMode mode{};
        ProblemShape problem_shape{};
        MainloopArguments mainloop{};
        EpilogueArguments epilogue{};
        AllReduceArguments all_reduce{};
        KernelHardwareInfo hw_info{};
        TileSchedulerArguments scheduler{};
    };

    // Kernel entry point API
    struct Params
    {
        GemmUniversalMode mode{};
        ProblemShape problem_shape{};
        MainloopParams mainloop{};
        EpilogueParams epilogue{};
        AllReduceParams all_reduce{};
        KernelHardwareInfo hw_info{};
        TileSchedulerParams scheduler{};
        void* workspace{nullptr};
    };

    //
    // Methods
    //

    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    static Params to_underlying_arguments(Arguments const& args, void* workspace)
    {
        CUTLASS_TRACE_HOST("to_underlying_arguments():");

        auto problem_shape = args.problem_shape;
        if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>)
        {
            // swap M/N
            get<0>(problem_shape) = get<1>(args.problem_shape);
            get<1>(problem_shape) = get<0>(args.problem_shape);
        }
        auto problem_shape_MNKL = append<4>(problem_shape, 1);

        // Get SM count if needed, otherwise use user supplied SM count
        int sm_count = args.hw_info.sm_count;
        if (sm_count <= 0)
        {
            CUTLASS_TRACE_HOST(
                "  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

        KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

        // Calculate workspace pointers
        uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
        size_t workspace_offset = 0;

        void* scheduler_workspace = workspace_ptr;
        workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
            args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
        workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);

        void* epilogue_workspace = workspace_ptr + workspace_offset;
        workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
        workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);

        void* mainloop_workspace = nullptr;
        // Precompute the sub tiles numbers in epilogue, pass into tile scheduler.  Therefore it will be used
        // in separate reduction scheme for streamk case, NumEpilogueSubTiles default value is 1, which means
        // subtile will not be used, therefore separate reduction will not be enabled.
        constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});
        TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(problem_shape_MNKL, TileShape{},
            ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles);

        return {args.mode, problem_shape,
            CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace),
            CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
            CollectiveAllReduce::to_underlying_arguments(args.problem_shape, args.all_reduce), hw_info, scheduler,
            workspace};
    }

    static bool can_implement(Arguments const& args)
    {
        bool implementable = (args.mode == GemmUniversalMode::kGemm)
            or (args.mode == GemmUniversalMode::kBatched && cute::rank(ProblemShape{}) == 4);
        if (!implementable)
        {
            CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
            return implementable;
        }
        implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
        implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
        implementable &= TileScheduler::can_implement(args.scheduler);
        return implementable;
    }

    static size_t get_workspace_size(Arguments const& args)
    {
        size_t workspace_size = 0;
        constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});

        workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
            args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
        workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

        workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
        workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

        return workspace_size;
    }

    static cutlass::Status initialize_workspace(Arguments const& args, void* workspace = nullptr,
        cudaStream_t stream = nullptr, CudaHostAdapter* cuda_adapter = nullptr)
    {
        Status status = Status::kSuccess;
        uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
        size_t workspace_offset = 0;
        constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});

        status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(args.scheduler,
            workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info, NumMmaWarpGroups,
            NumEpilogueSubTiles, cuda_adapter);
        workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
            args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
        workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
        if (status != Status::kSuccess)
        {
            return status;
        }

        status = CollectiveEpilogue::initialize_workspace(
            args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
        workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
        workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
        if (status != Status::kSuccess)
        {
            return status;
        }

        return status;
    }

    // Computes the kernel launch grid shape based on runtime parameters
    static dim3 get_grid_shape(Params const& params)
    {
        // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
        TileSchedulerArguments args{};
        if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>)
        {
            args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
        }
        args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN
            ? TileScheduler::RasterOrderOptions::AlongN
            : TileScheduler::RasterOrderOptions::AlongM;
        return TileScheduler::get_grid_shape(params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
    }

    static dim3 get_block_shape()
    {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }

    CUTLASS_DEVICE
    void operator()(Params const& params, char* smem_buf)
    {
        using namespace cute;
        using X = Underscore;

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define ENABLE_SM90_KERNEL_LEVEL 1
#endif
// Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
#if !defined(ENABLE_SM90_KERNEL_LEVEL)
        printf(
            "ERROR : Arch conditional MMA instruction used without targeting appropriate compute capability. "
            "Aborting.\n");
#else

        // Preconditions
        static_assert(size(TiledMma{}) == 256, "Cooperative kernel must have TiledMMA operating using 256 threads.");
        static_assert(size<0>(TileShape{}) >= 128,
            "Cooperative kernel requires Tile Size to be greater than or equal to 128 along the M-dimension.");

        static_assert(cute::rank(StrideA{}) == 3,
            "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
        static_assert(cute::rank(StrideB{}) == 3,
            "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
        static_assert(cute::rank(StrideC{}) == 3,
            "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
        static_assert(cute::rank(StrideD{}) == 3,
            "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

        /* In the Cooperative kernel, Consumer0 and Consumer1 collaborate on the same tile */
        enum class WarpGroupRole
        {
            Producer = 0,
            Consumer0 = 1,
            Consumer1 = 2
        };
        enum class ProducerWarpRole
        {
            Mainloop = 0,
            Warp1 = 1,
            Epilogue = 2,
            Warp3 = 3
        };

        // Kernel level shared memory storage
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        int thread_idx = int(threadIdx.x);
        int lane_idx = canonical_lane_idx();
        int warp_idx = canonical_warp_idx_sync();
        int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
        int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
        int mma_thread_idx = thread_idx % size(TiledMma{});
        auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
        auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
        int lane_predicate = cute::elect_one_sync();
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

        // Issue Tma Descriptor Prefetch from a single thread
        if ((warp_idx == 0) && lane_predicate)
        {
            CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
            CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
        }

        // Mainloop Load pipeline
        using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
        typename MainloopPipeline::Params mainloop_pipeline_params;
        if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Mainloop)
        {
            mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
        }
        if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1)
        {
            mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
        }
        mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
        mainloop_pipeline_params.num_consumers = size(TiledMma{});
        mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
        MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});

        // Epilogue Load pipeline
        using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
        typename EpiLoadPipeline::Params epi_load_pipeline_params;
        if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Epilogue)
        {
            epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
        }
        if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1)
        {
            epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
        }
        epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
        epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
        epi_load_pipeline_params.consumer_arv_count = size(TiledMma{});
        if constexpr (CollectiveEpilogue::RequiresTransactionBytes)
        {
            epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
        }
        EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

        // Epilogue Store pipeline
        using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
        typename EpiStorePipeline::Params epi_store_pipeline_params;
        epi_store_pipeline_params.always_wait = true;
        EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

        typename LoadWarpOrderBarrier::Params params_load_order_barrier;
        params_load_order_barrier.group_id = producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
        params_load_order_barrier.group_size = NumThreadsPerWarp;
        LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, params_load_order_barrier);

        // Initialize starting pipeline states for the collectives
        // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
        typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
        typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

        // For the DMA Load (producer) we start with an opposite phase
        // i.e., we skip all waits since we know that the buffer is indeed empty
        PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
        PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
        PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

        auto cluster_wait_fn = []()
        {
            // We need this to guarantee that the Pipeline init is visible
            // To all producers and consumer thread blocks in the Cluster
            if constexpr (size(ClusterShape{}) > 1)
            {
                cute::cluster_arrive_relaxed();
                return []() { cute::cluster_wait(); };
            }
            else
            {
                __syncthreads();
                return []() {}; // do nothing
            }
        }();

        // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
        auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});

        // Get the appropriate blocks for this thread block -- potential for thread block locality
        TiledMma tiled_mma;
        auto blk_shape = TileShape{}; // (BLK_M,BLK_N,BLK_K)

        TileScheduler scheduler{params.scheduler};
        auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});

        // In a warp specialized kernel, collectives expose data movement and compute operations separately
        CollectiveMainloop collective_mainloop;

        // Prepare and partition the input tensors. Expects a tuple of tensors where:
        // get<0>(load_inputs) is the tma tensor A after local tiling so that it has shape (BLK_M,BLK_K,m,k,l)
        // get<1>(load_inputs) is the tma tensor B after local tiling so that it has shape (BLK_N,BLK_K,n,k,l)
        auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
        static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2,
            "Output of load_init must have at least two elements (A, B)");

        // Extract out partitioned A and B.
        Tensor gA_mkl = get<0>(load_inputs);
        Tensor gB_nkl = get<1>(load_inputs);

        // Wait for all thread blocks in the Cluster
        cluster_wait_fn();

        if (warp_group_role == WarpGroupRole::Producer)
        {
            cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

            CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

            // Mainloop Producer Warp
            if (producer_warp_role == ProducerWarpRole::Mainloop)
            {
                bool do_load_order_arrive = true;
                while (work_tile_info.is_valid())
                {
                    if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info))
                    {
                        work_tile_info = scheduler.fetch_next_work(work_tile_info);
                        continue;
                    }

                    // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
                    auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
                    auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
                    auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
                    auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

                    // Get the number of K tiles to compute for this work as well as the starting K tile offset of the
                    // work.
                    auto work_k_tile_count
                        = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
                    auto work_k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
                    auto k_tile_iter
                        = cute::make_coord_iterator(idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

                    collective_mainloop.load(params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state,
                        load_inputs, blk_coord, k_tile_iter, work_k_tile_count, lane_idx, block_rank_in_cluster,
                        shared_storage.tensors.mainloop);
                    // Update starting pipeline state for the next tile
                    mainloop_pipe_producer_state.advance(work_k_tile_count);

                    // Signal for the epilogue load warp to begin
                    if (do_load_order_arrive)
                    {
                        load_order_barrier.arrive();
                        do_load_order_arrive = false;
                    }

                    // Get next work tile
                    work_tile_info = scheduler.fetch_next_work(work_tile_info);
                } // Scheduler work fetch loop

                // Make sure all Consumer Warp Groups have been waited upon
                collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

            } // Mainloop Producer Warp End

            // Epilogue Producer Warp
            else if (producer_warp_role == ProducerWarpRole::Epilogue && collective_epilogue.is_producer_load_needed())
            {

                if (!TileScheduler::requires_separate_reduction(params.scheduler) && work_tile_info.is_valid())
                {
                    load_order_barrier.wait();
                }
                while (work_tile_info.is_valid())
                {
                    if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler))
                    {
                        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
                        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
                        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
                        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
                        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

                        epi_load_pipe_producer_state = collective_epilogue.load(epi_load_pipeline,
                            epi_load_pipe_producer_state, problem_shape_MNKL, blk_shape, blk_coord, tiled_mma, lane_idx,
                            shared_storage.tensors.epilogue, work_tile_info.reduction_subtile_idx());
                    }

                    // Get next work tile
                    work_tile_info = scheduler.fetch_next_work(work_tile_info);
                } // Scheduler work fetch loop

                // Make sure all Consumer Warp Groups have been waited upon
                collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
            } // Epilogue Producer Warp End
        }     // Producer Warp Group End

        else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1)
        {
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

            constexpr uint32_t AllReduceBarrier = uint32_t(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier) + 1;
            CollectiveAllReduce collective_all_reduce(params.all_reduce, AllReduceBarrier);
            CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

            // Do we potentially issue tail arrives for TMA stores, if epilogue load is waiting for it
            bool do_store = false;
            cute::tuple<int32_t, int32_t, cute::Underscore, int32_t> prev_blk_coord;
            while (work_tile_info.is_valid())
            {
                // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
                auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
                auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
                auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
                auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
                auto work_k_tile_count
                    = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);

                // Allocate the accumulators for the (M,N) blk_shape
                //
                // MSVC CTAD breaks if we say "Tensor" here, so we use "auto" instead.
                auto accumulators = partition_fragment_C(tiled_mma, take<0, 2>(blk_shape)); // (MMA,MMA_M,MMA_N)
                if (TileScheduler::valid_warpgroup_in_work_tile(work_tile_info))
                {
                    collective_mainloop.mma(mainloop_pipeline, mainloop_pipe_consumer_state, accumulators,
                        work_k_tile_count, mma_thread_idx, shared_storage.tensors.mainloop, params.mainloop);

                    // Make sure the math instructions are done and free buffers before entering the epilogue
                    collective_mainloop.mma_tail(mainloop_pipeline, mainloop_pipe_consumer_state, work_k_tile_count);

                    // Update starting mainloop pipeline state for the next tile
                    mainloop_pipe_consumer_state.advance(work_k_tile_count);
                }
                // Index of warp group within consumer warp groups
                int consumer_warp_group_idx = canonical_warp_group_idx() - NumLoadWarpGroups;

                // Perform reduction across splits, if needed
                TileScheduler::fixup(
                    params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups, consumer_warp_group_idx);

                if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler))
                {
                    // Epilogue and write to gD
                    auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next]
                        = collective_epilogue.store(epi_load_pipeline, epi_load_pipe_consumer_state, epi_store_pipeline,
                            epi_store_pipe_producer_state, problem_shape_MNKL, blk_shape, blk_coord, accumulators,
                            tiled_mma, mma_thread_idx, shared_storage.tensors.epilogue,
                            work_tile_info.reduction_subtile_idx());
                    epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;
                    epi_store_pipe_producer_state = epi_store_pipe_producer_state_next;
                    do_store = true;
                    prev_blk_coord = blk_coord;

                    collective_epilogue.store_tail(epi_load_pipeline, epi_load_pipe_consumer_state, epi_store_pipeline,
                        epi_store_pipe_producer_state);
                }

                collective_all_reduce.gather_reduce_broadcast(problem_shape_MNKL, blk_coord, mma_thread_idx);

                // Get next work tile
                work_tile_info = scheduler.fetch_next_work(work_tile_info);
            } // Scheduler work fetch loop

            if (do_store)
            {
                collective_all_reduce.tile_global_sync(problem_shape_MNKL, prev_blk_coord, mma_thread_idx);
            }
        } // Consumer Warp Groups End
#endif
    }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
