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

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include <cuda_fp16.h>

namespace onnxruntime {
namespace cuda {
namespace collective {

#if defined(USE_MPI) || defined(ORT_USE_NCCL)

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
  uint32_t* peer_barrier_ptrs_in[MAX_RANKS_PER_NODE];
  uint32_t* peer_barrier_ptrs_out[MAX_RANKS_PER_NODE];
  void* peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE];
  void* local_output_buffer_ptr;
  const void* local_input_buffer_ptr;

  static AllReduceParams deserialize(const int32_t* buffer, size_t tp_size, size_t tp_rank, uint32_t flag);
};

bool ConfigurationSupported(AllReduceStrategyType algo, size_t msg_size, size_t world_size,
                            onnxruntime::MLDataType type);

void CustomAllReduce(AllReduceParams& params, onnxruntime::MLDataType data_type, AllReduceStrategyType strategy,
                     AllReduceStrategyConfig config, cudaStream_t stream);

size_t GetMaxRequiredWorkspaceSize(int world_size);

Status SetPeerAccess(int rank, int world_size, bool enable, int& can_access_peer);

AllReduceStrategyType SelectImplementation(size_t message_size, int rank, int world_size, onnxruntime::MLDataType type);

#endif

}  // namespace collective
}  // namespace cuda
}  // namespace onnxruntime
