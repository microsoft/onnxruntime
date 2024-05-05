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

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include <cuda_fp16.h>

namespace ort_trtllm {

constexpr size_t WARP_SIZE = 32;
constexpr size_t MAX_ALL_REDUCE_BLOCKS = 24;
constexpr size_t MAX_RANKS_PER_NODE = 8;
constexpr size_t DEFAULT_BLOCK_SIZE = 1024;

// Warning: python definition is in tensorrt_llm/functional.py
// they must be kept in sync
enum class AllReduceStrategyType : int8_t {
    NCCL = 0,
    ONESHOT = 1,
    TWOSHOT = 2,
    AUTO = 3,
};

enum class AllReduceStrategyConfig : int8_t {
    USE_MEMCPY = 1 << 0,
    PUSH_MODE = 1 << 1,
};

struct AllReduceParams {
    size_t elts_total;
    size_t elts_per_rank;
    size_t elts_per_block;
    size_t rank_offset;
    size_t ranks_per_node, rank, local_rank;
    uint32_t barrier_flag;
    uint32_t *peer_barrier_ptrs_in[MAX_RANKS_PER_NODE];
    uint32_t *peer_barrier_ptrs_out[MAX_RANKS_PER_NODE];
    void *peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE];
    void *local_output_buffer_ptr;
    void const *local_input_buffer_ptr;

    static AllReduceParams deserialize(int32_t const *buffer, size_t tpSize, size_t tpRank);
};

bool configurationSupported(AllReduceStrategyType algo, size_t msg_size, size_t n_ranks, onnxruntime::MLDataType type);

void customAllReduce(AllReduceParams &params, onnxruntime::MLDataType dataType, AllReduceStrategyType strat,
                     AllReduceStrategyConfig config, cudaStream_t stream);

} // namespace ort_trtllm
