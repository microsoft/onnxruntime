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

/*! \file
    \brief Implementation of a system-wide barrier for inter-GPU synchronization.
*/
#pragma once

#include "cutlass/barrier.h"

namespace cutlass
{

namespace detail
{

struct SyncNoOp
{
    CUTLASS_DEVICE
    static void sync() {}
};

__forceinline__ __device__ uint32_t atomicCAS_system_acq(uint32_t* p, uint32_t compare, uint32_t val)
{
    uint32_t result;
    asm volatile("atom.acquire.sys.global.cas.b32 %0, [%1], %2, %3;" : "=r"(result) : "l"(p), "r"(compare), "r"(val));
    return result;
}

} // namespace detail

template <class Sync, bool SafeBetweenPhases>
struct MulticastSystemBarrier : public GenericBarrier<Sync>
{

    using T = uint32_t;

    struct Params
    {
        T* mc_barrier_ptr;
        T* uc_barrier_ptr;
    };

protected:
    /// Reduce into flag, with release pattern (int specialization)
    CUTLASS_DEVICE
    static void red_release(T* mc_ptr, int val)
    {
#if defined(CUTE_ARCH_MULTIMEM_SM90_ENABLED)
        // atomic reduction to all replicas
        // this can be conceptually thought of as __threadfence_system(); atomicAdd_system(arrival_counter_mc, 1);
        // See
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red
        // for multimem PTX doc
        asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" ::"l"(mc_ptr), "r"(val) : "memory");

        // Need a fence between MC and UC access to the same memory:
        // - fence.proxy instructions establish an ordering between memory accesses that may happen through different
        // proxies
        // - Value .alias of the .proxykind qualifier refers to memory accesses performed using virtually aliased
        // addresses to the same memory location. from
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar
        asm volatile("fence.proxy.alias;" ::: "memory");
#else
        CUTE_INVALID_CONTROL_PATH("Trying to use multimem.red without CUTE_ARCH_NVLS_SM90_ENABLED.");
#endif
    }

    CUTLASS_DEVICE
    static bool bar_has_flipped(T old_arrive, T current_arrive)
    {
        return (((old_arrive ^ current_arrive) & 0x80000000) != 0);
    }

public:
    CUTLASS_DEVICE
    static void wait(T old_arrive, T* uc_ptr, int thread_idx, int flag_idx)
    {
        T* uc_barrier_ptr = uc_ptr + flag_idx;

        if (thread_idx == 0)
        {
            T current_arrive;
            do
            {
                asm volatile("ld.acquire.sys.u32 %0,[%1];" : "=r"(current_arrive) : "l"(uc_barrier_ptr) : "memory");
            } while (!bar_has_flipped(old_arrive, current_arrive));
        }

        Sync::sync();
    }

    CUTLASS_DEVICE
    static void wait_eq_reset(T* uc_ptr, int thread_idx, int flag_idx, int val)
    {
        T* uc_barrier_ptr = uc_ptr + flag_idx;

        if (thread_idx == 0)
        {
// Spin-loop
#pragma unroll 1
            while (detail::atomicCAS_system_acq(uc_barrier_ptr, val, 0) != val)
            {
            }
        }

        Sync::sync();
    }

    CUTLASS_DEVICE
    static T arrive_inc_get(T* mc_ptr, T* uc_ptr, int thread_idx, int flag_idx, int rank, int world_size)
    {
        T* mc_barrier_ptr = mc_ptr + flag_idx;
        T* uc_barrier_ptr = uc_ptr + flag_idx;

        Sync::sync();

        int old_arrive = 0;
        if (thread_idx == 0)
        {
            asm volatile("ld.acquire.sys.u32 %0,[%1];" : "=r"(old_arrive) : "l"(uc_barrier_ptr) : "memory");
            // Core of the addition: The total sum of
            // all participants must equal 0x8000000, so
            // that adding this to the existing entry will
            // cause an integer overflow.

            // N-1 blocks will add 1
            // Last block will add (0x80000000 - (N-1))
            // The end result is that the additive sum is 0x80000000

            // Effectively, the MSB will flip every time
            // the barrier finishes.

            // The practical implication of this is that the barrier
            // can be immediately reused.
            bool master = rank == 0;
            int val = master ? 0x80000000 - (world_size - 1) : 1;
            red_release(mc_barrier_ptr, val);
        }
        return old_arrive;
    }

    CUTLASS_DEVICE
    static void arrive_inc(Params const& params, int thread_idx, int flag_idx, int rank, int world_size)
    {
        T* mc_barrier = params.mc_barrier_ptr + flag_idx;

        Sync::sync();

        if (thread_idx == 0)
        {
            red_release(mc_barrier, 1);
        }
    }

    CUTLASS_DEVICE
    static void arrive_and_wait(Params const& params, int thread_idx, int flag_idx, int rank, int world_size)
    {
        auto mc_ptr = params.mc_barrier_ptr;
        auto uc_ptr = params.uc_barrier_ptr;
        if constexpr (SafeBetweenPhases)
        {
            auto old_arrive = arrive_inc_get(mc_ptr, uc_ptr, thread_idx, flag_idx, rank, world_size);
            wait(old_arrive, uc_ptr, thread_idx, flag_idx);
        }
        else
        {
            arrive_inc(params, thread_idx, flag_idx, rank, world_size);
            wait_eq_reset(uc_ptr, thread_idx, flag_idx, world_size);
        }
    }

    CUTLASS_DEVICE
    static void wait_eq_reset(Params const& params, int thread_idx, int flag_idx, int rank, int val)
    {
        T* barrier_ptr = params.uc_barrier_ptr + flag_idx;

        if (thread_idx == 0)
        {
// Spin-loop
#pragma unroll 1
            while (detail::atomicCAS_system_acq(barrier_ptr, val, 0) != val)
            {
            }
        }

        Sync::sync();
    }
};

template <class Sync, bool SafeBetweenPhases>
struct SystemBarrier : public GenericBarrier<Sync>
{
    using T = uint32_t;

    struct Params
    {
        T** barrier_ptrs;
    };

protected:
    /// Reduce into flag, with release pattern (int specialization)
    CUTLASS_DEVICE
    static T red_release(T* ptr, int val)
    {
        T old_arrive = 0;
        asm volatile("atom.add.release.sys.u32 %0,[%1],%2;" : "=r"(old_arrive) : "l"(ptr), "r"(val) : "memory");
        return old_arrive;
    }

    CUTLASS_DEVICE
    static T ld_acquire(T* ptr)
    {
        T state = 0;
        asm volatile("ld.acquire.sys.u32 %0,[%1];" : "=r"(state) : "l"(ptr) : "memory");
        return state;
    }

    CUTLASS_DEVICE
    static bool bar_has_flipped(T old_arrive, T current_arrive)
    {
        return (((old_arrive ^ current_arrive) & 0x80000000) != 0);
    }

public:
    CUTLASS_DEVICE
    static void wait(T old_arrive, T* barriers, int thread_idx, int flag_idx)
    {
        T* barrier = barriers + flag_idx;

        if (thread_idx == 0)
        {
#pragma unroll 1
            while (!bar_has_flipped(old_arrive, ld_acquire(barrier)))
            {
            }
        }

        Sync::sync();
    }

    CUTLASS_DEVICE
    static T arrive_inc_get(T** barrier_ptrs, int thread_idx, int flag_idx, int rank, int world_size)
    {

        Sync::sync();

        int old_arrive = 0;
        if (thread_idx < world_size)
        {
            bool master = rank == 0;
            int val = master ? 0x80000000 - (world_size - 1) : 1;
            T* barrier_ptr = barrier_ptrs[thread_idx] + flag_idx;
            old_arrive = red_release(barrier_ptr, val);
        }
        return old_arrive;
    }

    CUTLASS_DEVICE
    static void arrive_and_wait(Params const& params, int thread_idx, int flag_idx, int rank, int world_size)
    {
        if constexpr (SafeBetweenPhases)
        {
            auto old_arrive = arrive_inc_get(params.barrier_ptrs, thread_idx, flag_idx, rank, world_size);
            wait(old_arrive, params.barrier_ptrs[rank], thread_idx, flag_idx);
        }
        else
        {
            arrive_inc(params, thread_idx, flag_idx, rank, world_size);
            wait_eq_reset(params, thread_idx, flag_idx, rank, world_size);
        }
    }

    CUTLASS_DEVICE
    static void arrive_inc(Params const& params, int thread_idx, int flag_idx, int rank, int world_size)
    {
        Sync::sync();

        if (thread_idx < world_size)
        {
            T* barrier_ptr = params.barrier_ptrs[thread_idx] + flag_idx;
            [[maybe_unsed]] T old_arrive = red_release(barrier_ptr, 1);
        }
    }

    CUTLASS_DEVICE
    static void wait_eq_reset(Params const& params, int thread_idx, int flag_idx, int rank, int val)
    {
        if (thread_idx == 0)
        {
            T* barrier_ptr = params.barrier_ptrs[rank] + flag_idx;
#pragma unroll 1
            while (detail::atomicCAS_system_acq(barrier_ptr, val, 0) != val)
            {
            }
        }

        Sync::sync();
    }
};

} // namespace cutlass
