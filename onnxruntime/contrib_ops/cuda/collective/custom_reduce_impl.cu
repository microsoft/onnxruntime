/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/shared_library/provider_api.h"
#include "custom_reduce_impl.h"
#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <tuple>
#include <type_traits>

namespace onnxruntime {
namespace cuda {
namespace collective {

#if defined(USE_MPI) || defined(ORT_USE_NCCL)

using namespace onnxruntime;
using namespace onnxruntime::cuda;

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void st_flag_release(uint32_t const& flag, uint32_t* flag_addr) {
#if __CUDA_ARCH__ >= 700
  asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#else
  __threadfence_system();
  asm volatile("st.global.volatile.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t ld_flag_acquire(uint32_t* flag_addr) {
  uint32_t flag;
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
  asm volatile("ld.global.volatile.b32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#endif
  return flag;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Type Converter that packs data format to 128 bits data type
//
using PackedFloat = union {
  int4 packed;
  float unpacked[4];
};

using PackedHalf = union {
  int4 packed;
  half2 unpacked[4];
};

using PackedBFloat16 = union {
  int4 packed;
  __nv_bfloat162 unpacked[4];
};

template <typename T>
struct PackedOn16Bytes {};

template <>
struct PackedOn16Bytes<float> {
  using Type = PackedFloat;
};

template <>
struct PackedOn16Bytes<half> {
  using Type = PackedHalf;
};

template <>
struct PackedOn16Bytes<__nv_bfloat16> {
  using Type = PackedBFloat16;
};

// add two 128b data
template <typename T>
inline __device__ int4 add128b(T& a, T& b) {
  T c;
  c.unpacked[0] = a.unpacked[0] + b.unpacked[0];
  c.unpacked[1] = a.unpacked[1] + b.unpacked[1];
  c.unpacked[2] = a.unpacked[2] + b.unpacked[2];
  c.unpacked[3] = a.unpacked[3] + b.unpacked[3];
  return c.packed;
}

// Sum the per-rank packed vectors, always starting from rank 0 so the result does not depend on
// which rank runs the kernel.
//
// bfloat16 carries 8 mantissa bits, so accumulating eight partial sums in bfloat16 rounds after
// every addition. This all-reduce sits on the residual stream of all 43 layers, and measurably
// costs MMLU-Pro accuracy at that width, so the bfloat16 case accumulates in fp32 and rounds once.
template <typename T, int RANKS_PER_NODE, typename PackedStruct>
inline __device__ int4 reduce_packed(PackedStruct (&vals)[RANKS_PER_NODE], int local_rank) {
  if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    float2 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc[j] = make_float2(0.f, 0.f);
    }
#pragma unroll
    for (int rank = 0; rank < RANKS_PER_NODE; ++rank) {
      int const ii = (rank + RANKS_PER_NODE - local_rank) % RANKS_PER_NODE;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 const v = __bfloat1622float2(vals[ii].unpacked[j]);
        acc[j].x += v.x;
        acc[j].y += v.y;
      }
    }
    PackedStruct out;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out.unpacked[j] = __float22bfloat162_rn(acc[j]);
    }
    return out.packed;
  } else {
    PackedStruct sums;
    sums.packed = {0, 0, 0, 0};
#pragma unroll
    for (int rank = 0; rank < RANKS_PER_NODE; ++rank) {
      int const ii = (rank + RANKS_PER_NODE - local_rank) % RANKS_PER_NODE;
      sums.packed = add128b(sums, vals[ii]);
    }
    return sums.packed;
  }
}

// True once `observed` has caught up with `expected`. The flags only ever move forward, so a
// signed difference is both the "reached or passed" test and wrap-around safe: a peer that has
// already started the *next* all-reduce, and so has published expected + 1, must not leave this
// rank spinning for a value that will never appear again.
__inline__ __device__ bool flag_reached(uint32_t observed, uint32_t expected) {
  return static_cast<int32_t>(observed - expected) >= 0;
}

// Each block owns one counter in its own barrier buffer, holding the flag value it last used.
// See BarrierFlagCounterOffset for why this lives on the device rather than in the kernel
// arguments. The counters are only ever touched by the owning block on the owning rank, so
// reading at entry and writing at exit needs no atomics.
__inline__ __device__ uint32_t* local_flag_counters(AllReduceParams const& params) {
  return params.peer_barrier_ptrs_in[params.local_rank] + BarrierFlagCounterOffset(params.ranks_per_node);
}

__inline__ __device__ void multi_gpu_barrier(uint32_t** signals, uint32_t const flag, size_t const local_rank,
                                             size_t const world_size, int const tidx, int const bidx) {
  // After this function, at least one block in each GPU has reached the barrier
  if (tidx < world_size) {
    // we can think of signals having the shape [world_size, world_size]
    // Dimension 0 is the "listening" dimension, dimension 2 is "emitting" dimension

    // Block 0 broadcasts its flag (local_rank on emitting dimension) to all receivers
    if (bidx == 0) {
      signals[tidx][local_rank] = flag;
    }

    // All blocks check that corresponding block 0 on other GPUs have set the flag
    // No deadlock because block #0 is always the first block started
    uint32_t volatile* my_signals = signals[local_rank];
    while (!flag_reached(my_signals[tidx], flag)) {
    }
  }

  __syncthreads();
}

__inline__ __device__ void block_barrier(uint32_t** signals, uint32_t const flag, size_t const local_rank,
                                         size_t const world_size, int const tidx, int const bidx) {
  // After this function, the block of id == bidx of each GPU has reached the barrier
  if (tidx < world_size) {
    // we can think of signals having the shape [world_size, num_blocks, world_size]
    // (+ an offset on dim 1 to account for flags used in multi_gpu_barrier)
    // Dimension 0 is the "listening" dimension, dimension 2 is "emitting" dimension

    // Block broadcast its flag (local_rank on emitting dimension) to all receivers
    uint32_t flag_block_offset = world_size + bidx * world_size;
    st_flag_release(flag, signals[tidx] + flag_block_offset + local_rank);

    // Blocks check that corresponding blocks on other GPUs have also set the flag
    uint32_t* peer_barrier_d = signals[local_rank] + flag_block_offset + tidx;
    while (!flag_reached(ld_flag_acquire(peer_barrier_d), flag)) {
    }
  }

  __syncthreads();
}

template <typename T, int RANKS_PER_NODE, bool COPY_INPUT = true, bool PUSH_MODE = false>
static __global__ void oneShotAllReduceKernel(AllReduceParams params) {
  // Suppose that two GPUs participate in the AR exchange, and we start four blocks.
  // The message is partitioned into chunks as detailed below:
  //               message
  //       |-------------------|
  // GPU 0 | B0 | B1 | B2 | B3 |
  // GPU 1 | B0 | B1 | B2 | B3 |
  //
  // Here the step-by-step behavior of one block:
  // 1. B0 copies the chunk it  is responsible for, from local_input to shareable buffer
  // 2. B0 on GPU 0 and B0 on GPU 1 wait for each other (block_barrier)
  // 3. B0 on GPU 0 pull and sum the chunk from GPU 1, writes the result to local_output
  //
  // With COPY_INPUT == false, skip step 1. and use gpu_barrier instead of block barrier during step 2.
  // We only to know if the other GPU as arrived at the AR kernel, that would mean that data is ready
  //
  // With PUSH_MODE, we consider that the shared buffer is of size:
  // params.peer_comm_buffer_ptrs: [world_size, world_size, message_size]
  //
  // Here the step-by-step behavior of one block:
  // 1. B0 push the chunk is it responsible for into all other GPUs:
  //    params.peer_comm_buffer_ptrs[:, local_gpu, B0 slice]
  // 2. block sync so the block is shared by other GPUs
  // 3. Reduce along second dimension params.peer_comm_buffer_ptrs[local_gpu, :, B0 slice]

  int const bidx = blockIdx.x;
  int const tidx = threadIdx.x;

  uint32_t* const flag_counters = local_flag_counters(params);
  uint32_t const barrier_flag = flag_counters[bidx] + 1;

  // The number of elements packed into one for comms
  static constexpr int PACKED_ELTS = 16 / sizeof(T);
  using PackedStruct = typename PackedOn16Bytes<T>::Type;

  [[maybe_unused]] T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
  [[maybe_unused]] T* local_shared_buffer = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank]);
  [[maybe_unused]] T* local_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);

  // Start and end offsets of the thread
  size_t const chunk_start = bidx * params.elts_per_block + tidx * PACKED_ELTS;
  size_t const chunk_end = std::min((bidx + 1) * params.elts_per_block, params.elts_total);

  T* buffers[RANKS_PER_NODE];
#pragma unroll
  for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
    // buffers[0] is always the local buffers. Helps load balancing reads.
    int rank = (params.local_rank + ii) % RANKS_PER_NODE;
    buffers[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank]);
  }

  if constexpr (PUSH_MODE || COPY_INPUT) {
    // Copy from local buffer to shareable buffer
    for (size_t iter_offset = chunk_start; iter_offset < chunk_end; iter_offset += blockDim.x * PACKED_ELTS) {
      if constexpr (PUSH_MODE) {
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
          *reinterpret_cast<int4*>(&buffers[ii][params.local_rank * params.elts_total + iter_offset]) =
              *reinterpret_cast<int4 const*>(&local_input_buffer[iter_offset]);
        }
      } else {
        *reinterpret_cast<int4*>(&local_shared_buffer[iter_offset]) =
            *reinterpret_cast<int4 const*>(&local_input_buffer[iter_offset]);
      }
    }
    // wait for equivalent blocks of other GPUs to have copied data to their shareable buffer
    block_barrier(params.peer_barrier_ptrs_in, barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx);
  } else {
    // In the non-copy case, we assume that once the kernel has been started, data is ready to be consumed
    multi_gpu_barrier(params.peer_barrier_ptrs_in, barrier_flag, params.local_rank, RANKS_PER_NODE, tidx,
                      bidx);
  }

  // Each block accumulates the values from the different GPUs on the same node.
  for (size_t iter_offset = chunk_start; iter_offset < chunk_end; iter_offset += blockDim.x * PACKED_ELTS) {
    // Iterate over the different ranks/devices on the node to load the values.
    PackedStruct vals[RANKS_PER_NODE];
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
      if constexpr (PUSH_MODE) {
        vals[ii].packed =
            *reinterpret_cast<int4 const*>(&buffers[params.local_rank][ii * params.elts_total + iter_offset]);
      } else {
        vals[ii].packed = *reinterpret_cast<int4 const*>(&buffers[ii][iter_offset]);
      }
    }

    // Sum the values from the different ranks.
    PackedStruct sums;
    sums.packed = reduce_packed<T, RANKS_PER_NODE>(vals, static_cast<int>(params.local_rank));

    // Store to the destination buffer.
    *reinterpret_cast<int4*>(&local_output_buffer[iter_offset]) = sums.packed;
  }

  // Nothing above stops a rank that finishes early from returning, letting the next all-reduce on
  // its stream overwrite the shared buffer that slower peers are still reading. DSV4 issues 86
  // back-to-back all-reduces per decode step, so that window is hit often enough to corrupt the
  // residual stream. Hold every rank here until all of them have finished reading. The exit
  // barrier uses the "out" buffer, so it cannot be confused with the entry barrier above.
  block_barrier(params.peer_barrier_ptrs_out, barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx);

  // Publish the flag this block just used, so the next launch picks up from here. Safe without a
  // barrier: no other block or rank ever reads this slot.
  if (tidx == 0) {
    flag_counters[bidx] = barrier_flag;
  }
}

template <typename T, int RANKS_PER_NODE, bool COPY_INPUT = true, bool PUSH_MODE = false>
static __global__ void twoShotAllReduceKernel(AllReduceParams params) {
  // Suppose that two GPUs participate in the AR exchange, and we start two blocks.
  // The message is partitioned into chunks as detailed below:
  //               message
  //       |-------------------|
  //       |--GPU 0--|--GPU 1--| (GPU responsibility parts)
  // GPU 0 | B0 | B1 | B0 | B1 |
  // GPU 1 | B0 | B1 | B0 | B1 |
  //
  // Here the step-by-step behavior of one block:
  // 1. B0 copies all chunks is it responsible for, from local_input to shareable buffer
  // 2. B0 on GPU 0 and B0 on GPU 1 wait for each other (block_barrier #0)
  // 3. B0 on GPU 0 gather and sum the B0 chunks from GPU 1, that are in the GPU 0 responsibility
  //    part (the first half of the message, see GPU responsibility row above)
  // 3bis. Likewise, B0 on GPU 1 copies and sum the chunks for GPU 0,
  //       where GPU 1 is responsible: the second half of the message.
  // 4. B0 on GPU 0 and B0 on GPU 1 wait for each other (block_barrier #1)
  // 5. B0 writes result to local_output. It gathers each chunk from its responsible GPU.
  //    For example, here it reads the first chunk from GPU 0 and second chunk from GPU 1.
  //
  // With COPY_INPUT == false, skip step 1. and use gpu_barrier instead of block barrier during step 2.
  // We only to know if the other GPU as arrived at the AR kernel, that would mean that data is ready
  // to be read.
  //
  // Note that compared to one-shot, one block (CTA) writes multiple input chunks and write multiple output chunks.
  // However, it's only responsible for the summation of a single chunk.
  //
  // With PUSH_MODE, we consider that the shared buffer is of size:
  // params.peer_comm_buffer_ptrs: [world_size, world_size, message_size / world_size]
  //
  // Here the step-by-step behavior of one block:
  // 1. B0 push the chunks is it responsible for into the corresponding GPUs:
  //    params.peer_comm_buffer_ptrs[target_gpu, local_gpu, current B0 slice]
  // 2. block sync so the blocks have been shared by other GPUs
  // 3. Reduce along second dimension params.peer_comm_buffer_ptrs[local_gpu, :, B0 slice]
  // 4. block barrier (corresponding blocks have finished reduction)
  // 5. pull and write on local buffer, by reading params.peer_comm_buffer_ptrs[:, 0, B0 slice] (reduction result is
  //    written at index 0 of 2nd dim)

  int const bidx = blockIdx.x;
  int const tidx = threadIdx.x;

  uint32_t* const flag_counters = local_flag_counters(params);
  uint32_t const barrier_flag = flag_counters[bidx] + 1;

  // The number of elements packed into one for comms
  static constexpr int PACKED_ELTS = 16 / sizeof(T);
  using PackedType = typename PackedOn16Bytes<T>::Type;

  [[maybe_unused]] T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
  [[maybe_unused]] T* local_shared_buffer = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank]);
  [[maybe_unused]] T* local_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);

  size_t const chunk_start = bidx * params.elts_per_block + tidx * PACKED_ELTS;
  size_t const chunk_end = min(chunk_start + params.elts_per_block, params.elts_per_rank);

  T* buffers[RANKS_PER_NODE];
  int ranks[RANKS_PER_NODE];
#pragma unroll
  for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
    // A mapping of the ranks to scatter reads as much as possible
    int rank = (params.local_rank + ii) % RANKS_PER_NODE;
    ranks[ii] = rank;
    buffers[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank]);
  }

  if constexpr (PUSH_MODE || COPY_INPUT) {
    // Copy all blocks from local buffer to shareable buffer
    for (size_t local_offset = chunk_start; local_offset < chunk_end; local_offset += blockDim.x * PACKED_ELTS) {
#pragma unroll
      for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
        size_t offset_rank = ii * params.elts_per_rank + local_offset;
        if (offset_rank >= params.elts_total) {
          continue;
        }

        if constexpr (PUSH_MODE) {
          *reinterpret_cast<int4*>(&buffers[ii][params.local_rank * params.elts_per_rank + local_offset]) =
              *reinterpret_cast<int4 const*>(&local_input_buffer[offset_rank]);
        } else {
          *reinterpret_cast<int4*>(&local_shared_buffer[offset_rank]) =
              *reinterpret_cast<int4 const*>(&local_input_buffer[offset_rank]);
        }
      }
    }
    block_barrier(params.peer_barrier_ptrs_in, barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx);
  } else {
    // In the non-copy case, we assume that once the kernel has been started, data is ready to be consumed
    multi_gpu_barrier(params.peer_barrier_ptrs_in, barrier_flag, params.local_rank, RANKS_PER_NODE, tidx,
                      bidx);
  }

  // Each block accumulates the values from the different GPUs on the same node.
  for (size_t local_offset = chunk_start; local_offset < chunk_end; local_offset += blockDim.x * PACKED_ELTS) {
    size_t const responsible_block_offset = local_offset + params.rank_offset;

    // Iterate over the different ranks/devices on the node to load the values.
    PackedType vals[RANKS_PER_NODE];
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
      if constexpr (PUSH_MODE) {
        vals[ii].packed =
            *reinterpret_cast<int4 const*>(&local_shared_buffer[ii * params.elts_per_rank + local_offset]);
      } else {
        vals[ii].packed = *reinterpret_cast<int4 const*>(&buffers[ii][responsible_block_offset]);
      }
    }

    // Sum the values from the different ranks.
    PackedType sums;
    sums.packed = reduce_packed<T, RANKS_PER_NODE>(vals, static_cast<int>(params.local_rank));

    // Store to the local buffer.
    if constexpr (PUSH_MODE) {
      *reinterpret_cast<int4*>(&local_shared_buffer[local_offset]) = sums.packed;
    } else {
      *reinterpret_cast<int4*>(&local_shared_buffer[responsible_block_offset]) = sums.packed;
    }
  }

  block_barrier(params.peer_barrier_ptrs_out, barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx);

  // Gather all needed elts from other intra-node ranks
  for (size_t local_offset = chunk_start; local_offset < chunk_end; local_offset += blockDim.x * PACKED_ELTS) {
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
      // use round-robin gathering from other ranks
      size_t offset_rank = ranks[ii] * params.elts_per_rank + local_offset;
      if (offset_rank >= params.elts_total) {
        continue;
      }

      if constexpr (PUSH_MODE) {
        *reinterpret_cast<int4*>(&local_output_buffer[offset_rank]) =
            *reinterpret_cast<int4*>(&buffers[ii][local_offset]);
      } else {
        *reinterpret_cast<int4*>(&local_output_buffer[offset_rank]) =
            *reinterpret_cast<int4*>(&buffers[ii][offset_rank]);
      }
    }
  }

  // Same exit barrier as in oneShotAllReduceKernel: the gather above reads from peers, so no rank
  // may return until all of them are done. Both barrier buffers already carry barrier_flag at this
  // point, so reuse the "in" buffer one step further along.
  block_barrier(params.peer_barrier_ptrs_in, barrier_flag + 1, params.local_rank, RANKS_PER_NODE, tidx, bidx);

  // See the matching comment in oneShotAllReduceKernel.
  if (tidx == 0) {
    flag_counters[bidx] = barrier_flag + 1;
  }
}

bool ConfigurationSupported(AllReduceStrategyType algo, size_t msg_size, size_t world_size,
                            onnxruntime::MLDataType type) {
  size_t elts_per_thread = 16 / type->Size();
  int const msg_align = (algo == AllReduceStrategyType::TWOSHOT) ? world_size * elts_per_thread : elts_per_thread;
  bool supported_algo = (algo == AllReduceStrategyType::ONESHOT || algo == AllReduceStrategyType::TWOSHOT);
  return supported_algo && (msg_size % msg_align == 0);
}

std::tuple<int, int> kernelLaunchConfig(AllReduceStrategyType algo, AllReduceParams& param, size_t elts_per_thread) {
  int blocks_per_grid = 1, threads_per_block = DEFAULT_BLOCK_SIZE;

  switch (algo) {
    case AllReduceStrategyType::ONESHOT: {
      ORT_ENFORCE(param.elts_total % elts_per_thread == 0);
      size_t const total_threads = roundUp(param.elts_total / elts_per_thread, WARP_SIZE);
      threads_per_block = std::min(DEFAULT_BLOCK_SIZE, total_threads);
      blocks_per_grid = std::min(static_cast<size_t>(MAX_ALL_REDUCE_BLOCKS),
                                 divUp(total_threads, static_cast<size_t>(threads_per_block)));
      param.elts_per_block = roundUp(divUp(param.elts_total, static_cast<size_t>(blocks_per_grid)), elts_per_thread);
      break;
    }
    case AllReduceStrategyType::TWOSHOT: {
      ORT_ENFORCE(param.elts_total % (elts_per_thread * param.ranks_per_node) == 0);
      size_t const total_threads = roundUp(param.elts_total / (elts_per_thread * param.ranks_per_node), WARP_SIZE);

      /*
      threads_per_block = std::min(DEFAULT_BLOCK_SIZE, total_threads);
      blocks_per_grid = std::min(static_cast<size_t>(MAX_ALL_REDUCE_BLOCKS), divUp(total_threads, threads_per_block));
      */

      while (total_threads % blocks_per_grid != 0 || total_threads / blocks_per_grid > DEFAULT_BLOCK_SIZE) {
        blocks_per_grid += 1;
      }

      threads_per_block = total_threads / blocks_per_grid;

      // NOTE: need to adjust here
      if (static_cast<size_t>(blocks_per_grid) > MAX_ALL_REDUCE_BLOCKS) {
        size_t iter_factor = 1;
        while (blocks_per_grid / iter_factor > MAX_ALL_REDUCE_BLOCKS || blocks_per_grid % iter_factor) {
          iter_factor += 1;
        }
        blocks_per_grid /= iter_factor;
      }
      param.elts_per_rank = param.elts_total / param.ranks_per_node;
      param.rank_offset = param.local_rank * param.elts_per_rank;
      param.elts_per_block =
          roundUp(divUp(param.elts_per_rank, static_cast<size_t>(blocks_per_grid)), elts_per_thread);
      break;
    }
    default:
      ORT_THROW("Algorithm not supported here.");
  }

  return std::make_tuple(blocks_per_grid, threads_per_block);
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false, bool USE_MEMCPY = false>
void AllReduceDispatchMemcpy(AllReduceStrategyType algo, AllReduceStrategyConfig config, AllReduceParams& param,
                             cudaStream_t stream) {
  ORT_ENFORCE(!(USE_MEMCPY && PUSH_MODE), "Memcpy cannot be used with PUSH_MODE.");
  size_t elts_per_thread = 16 / sizeof(T);
  auto [blocks_per_grid, threads_per_block] = kernelLaunchConfig(algo, param, elts_per_thread);

  if (USE_MEMCPY) {
    cudaMemcpyAsync(param.peer_comm_buffer_ptrs[param.local_rank], param.local_input_buffer_ptr,
                    param.elts_total * sizeof(T), cudaMemcpyDeviceToDevice, stream);
  }

  if (algo == AllReduceStrategyType::ONESHOT) {
    oneShotAllReduceKernel<T, RANKS_PER_NODE, !USE_MEMCPY, PUSH_MODE>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(param);
  } else {
    twoShotAllReduceKernel<T, RANKS_PER_NODE, !USE_MEMCPY, PUSH_MODE>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(param);
  }
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false>
void AllReduceDispatchPushMode(AllReduceStrategyType algo, AllReduceStrategyConfig config, AllReduceParams& param,
                               cudaStream_t stream) {
  if (static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(config) &
      static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(AllReduceStrategyConfig::USE_MEMCPY)) {
    AllReduceDispatchMemcpy<T, RANKS_PER_NODE, PUSH_MODE, true>(algo, config, param, stream);
  } else {
    AllReduceDispatchMemcpy<T, RANKS_PER_NODE, PUSH_MODE, false>(algo, config, param, stream);
  }
}

template <typename T, int RANKS_PER_NODE>  //, bool USE_MEMCPY = false, bool PUSH_MODE = false>
void AllReduceDispatchRanksPerNode(AllReduceStrategyType algo, AllReduceStrategyConfig config, AllReduceParams& param,
                                   cudaStream_t stream) {
  if (static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(config) &
      static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(AllReduceStrategyConfig::PUSH_MODE)) {
    AllReduceDispatchPushMode<T, RANKS_PER_NODE, true>(algo, config, param, stream);
  } else {
    AllReduceDispatchPushMode<T, RANKS_PER_NODE, false>(algo, config, param, stream);
  }
}

template <typename T>
void AllReduceDispatchType(AllReduceParams& param, AllReduceStrategyType strategy, AllReduceStrategyConfig config,
                           cudaStream_t stream) {
  switch (param.ranks_per_node) {
    case 2:
      AllReduceDispatchRanksPerNode<T, 2>(strategy, config, param, stream);
      break;
    case 4:
      AllReduceDispatchRanksPerNode<T, 4>(strategy, config, param, stream);
      break;
    case 6:
      AllReduceDispatchRanksPerNode<T, 6>(strategy, config, param, stream);
      break;
    case 8:
      AllReduceDispatchRanksPerNode<T, 8>(strategy, config, param, stream);
      break;
    default:
      ORT_THROW("Custom all reduce only supported on {2, 4, 6, 8} GPUs per node.");
  }
}

AllReduceParams AllReduceParams::deserialize(const int32_t* buffer, size_t tp_size, size_t tp_rank) {
  void* const* buffer_ptrs = reinterpret_cast<void* const*>(buffer);
  AllReduceParams params;

  for (size_t i = 0; i < tp_size; ++i) {
    params.peer_comm_buffer_ptrs[i] = buffer_ptrs[i];
  }
  for (size_t i = 0; i < tp_size; ++i) {
    params.peer_barrier_ptrs_in[i] = reinterpret_cast<uint32_t*>(buffer_ptrs[tp_size + i]);
  }
  for (size_t i = 0; i < tp_size; ++i) {
    params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(buffer_ptrs[2 * tp_size + i]);
  }
  params.ranks_per_node = tp_size;
  params.rank = tp_rank;
  params.local_rank = tp_rank;

  return params;
}

void CustomAllReduce(AllReduceParams& params, onnxruntime::MLDataType data_type, AllReduceStrategyType strategy,
                     AllReduceStrategyConfig config, cudaStream_t stream) {
  ORT_ENFORCE(ConfigurationSupported(strategy, params.elts_total, params.ranks_per_node, data_type),
              "Custom all-reduce configuration unsupported");
  // USE_MEMCPY stages the input with cudaMemcpyAsync and then synchronizes with multi_gpu_barrier,
  // where block 0 publishes a single flag on behalf of every block. That is incompatible with the
  // per-block device flag counters the barrier now uses, since a block cannot read block 0's
  // counter without racing block 0's write of it. The in-kernel copy is a wash at these sizes and
  // saves a launch, so the memcpy variant is simply not offered.
  ORT_ENFORCE(!(static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(config) &
                static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(AllReduceStrategyConfig::USE_MEMCPY)),
              "Custom all-reduce no longer supports USE_MEMCPY");
  if (data_type == onnxruntime::DataTypeImpl::GetType<float>()) {
    AllReduceDispatchType<float>(params, strategy, config, stream);
  } else if (data_type == onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>()) {
    AllReduceDispatchType<half>(params, strategy, config, stream);
  } else if (data_type == onnxruntime::DataTypeImpl::GetType<onnxruntime::BFloat16>()) {
    AllReduceDispatchType<__nv_bfloat16>(params, strategy, config, stream);
  } else {
    ORT_THROW("Unsupported data type for CustomAllReduce");
  }
}

size_t GetMaxRequiredWorkspaceSize(int world_size) {
  if (world_size <= 2) {
    return 16 * 1000 * 1000;
  }
  return 8 * 1000 * 1000;
}

Status SetPeerAccess(int rank, int world_size, bool enable, int& can_access_peer) {
  const int src_node = rank;

  // The peers are identified by device ordinal, which only works when every device is visible to
  // this process. Launchers that pin one GPU per rank with CUDA_VISIBLE_DEVICES leave a single
  // ordinal here, so there is nothing to probe: peer access is instead established when the
  // workspace handles are imported with cudaIpcMemLazyEnablePeerAccess, which reports its own
  // failure if the devices turn out not to be peer-capable.
  int device_count = 0;
  CUDA_RETURN_IF_ERROR(cudaGetDeviceCount(&device_count));
  if (device_count < world_size) {
    can_access_peer = 1;
    return Status::OK();
  }

  for (int dst_node = 0; dst_node < world_size; dst_node++) {
    if (dst_node == src_node) {
      continue;
    }

    CUDA_RETURN_IF_ERROR(cudaDeviceCanAccessPeer(&can_access_peer, src_node, dst_node));

    if (!can_access_peer) {
      return Status::OK();
    }

    if (enable) {
      cudaDeviceEnablePeerAccess(dst_node, 0);
    } else {
      cudaDeviceDisablePeerAccess(dst_node);
    }

    auto const error = cudaGetLastError();
    if (error != cudaErrorPeerAccessAlreadyEnabled && error != cudaErrorPeerAccessNotEnabled) {
      CUDA_RETURN_IF_ERROR(error);
    }
  }

  return Status::OK();
}

AllReduceStrategyType SelectImplementation(size_t message_size, int rank, int world_size,
                                           onnxruntime::MLDataType type) {
  AllReduceStrategyType strategy = AllReduceStrategyType::NCCL;
  if (type != onnxruntime::DataTypeImpl::GetType<float>() &&
      type != onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>() &&
      type != onnxruntime::DataTypeImpl::GetType<onnxruntime::BFloat16>()) {
    return strategy;
  }

  if (world_size != 2 && world_size != 4 && world_size != 6 && world_size != 8) {
    return strategy;
  }

  // cudaDeviceCanAccessPeer / cudaDeviceEnablePeerAccess are pure host calls, but this runs on
  // every eager AllReduce, so keep only the first answer.
  static const int cached_can_access_peer = [&]() {
    int can_access = 0;
    ORT_ENFORCE(SetPeerAccess(rank, world_size, true, can_access) == Status::OK());
    return can_access;
  }();
  // If P2P is not enabled, we cannot use the custom allreduce, so default to NCCL.
  if (!cached_can_access_peer) {
    return strategy;
  }

  const size_t maxWorkspaceSize = GetMaxRequiredWorkspaceSize(world_size);
  const size_t message_size_bytes = message_size * type->Size();

  if (message_size_bytes <= maxWorkspaceSize) {
    if (world_size <= 2) {
      strategy = AllReduceStrategyType::ONESHOT;
    } else if (world_size <= 4) {
      if (message_size_bytes < 1 * 1000 * 1000) {
        strategy = AllReduceStrategyType::ONESHOT;
      } else {
        strategy = AllReduceStrategyType::TWOSHOT;
      }
    } else {
      if (message_size_bytes < 500 * 1000) {
        strategy = AllReduceStrategyType::ONESHOT;
      } else {
        strategy = AllReduceStrategyType::TWOSHOT;
      }
    }
  }

  if (!ConfigurationSupported(strategy, message_size, world_size, type)) {
    strategy = AllReduceStrategyType::NCCL;
  }

  return strategy;
}

#endif

}  // namespace collective
}  // namespace cuda
}  // namespace onnxruntime
