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

bool ConfigurationSupported(AllReduceStrategyType algo, size_t msg_size, size_t world_size,
                            onnxruntime::MLDataType type);

void CustomAllReduce(AllReduceParams &params, onnxruntime::MLDataType data_type, AllReduceStrategyType strat,
                     AllReduceStrategyConfig config, cudaStream_t stream);

inline size_t GetMaxRequiredWorkspaceSize(int world_size) noexcept {
    if (world_size <= 2) {
        return 16 * 1000 * 1000;
    }
    return 8 * 1000 * 1000;
}

inline AllReduceStrategyType SelectImplementation(size_t message_size, int world_size,
                                                  onnxruntime::MLDataType type) noexcept {
    const size_t maxWorkspaceSize = GetMaxRequiredWorkspaceSize(world_size);

    AllReduceStrategyType strat = AllReduceStrategyType::NCCL;
    const size_t message_size_bytes = message_size * type->Size();

    if (message_size_bytes <= maxWorkspaceSize) {
        if (world_size <= 2) {
            strat = AllReduceStrategyType::ONESHOT;
        } else if (world_size <= 4) {
            if (message_size_bytes < 1 * 1000 * 1000) {
                strat = AllReduceStrategyType::ONESHOT;
            } else {
                strat = AllReduceStrategyType::TWOSHOT;
            }
        } else {
            if (message_size_bytes < 500 * 1000) {
                strat = AllReduceStrategyType::ONESHOT;
            } else {
                strat = AllReduceStrategyType::TWOSHOT;
            }
        }
    }

    if (!ConfigurationSupported(strat, message_size, world_size, type)) {
        strat = AllReduceStrategyType::NCCL;
    }

    return strat;
}

} // namespace ort_trtllm
