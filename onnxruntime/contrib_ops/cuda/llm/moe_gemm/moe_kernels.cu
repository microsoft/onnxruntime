/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "core/providers/cuda/curand_wrapper.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>

// Ignore CUTLASS warnings about type punning
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/epilogue/thread/fused_activations.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "core/common/common.h"
#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/common/memory_utils.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/common/data_type.h"
// #include "contrib_ops/cuda/llm/common/env_utils.h"
#include "contrib_ops/cuda/llm/common/workspace.h"
#include "contrib_ops/cuda/llm/cutlass_type_conversion.h"
#include "contrib_ops/cuda/llm/kernels/pre_quant_scale_kernel.h"
#include "contrib_ops/cuda/llm/kernels/quantization.cuh"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_util_kernels.h"

#include <cub/cub.cuh>
#include <curand_philox4x32_x.h>

using namespace onnxruntime::llm::kernels;
using namespace onnxruntime::llm::common;

namespace onnxruntime::llm::kernels::cutlass_kernels {
/**
 * Takes the input maps and prepares the expanded maps for min latency
 * @param num_active_experts_per_node: Number of active experts on current node
 * @param experts_to_token_scores: The score of each token for each activated expert. 0 if the expert is not chosen by
 * the token. Only the first num_active_experts_per_ rows are valid
 * @param active_expert_global_ids: The global expert id for each activated expert
 * Only the first num_active_experts_per_ values are valid
 * @param expert_first_token_offset: Store the first token offset for each expert
 */
template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void initTensor(T* value, int const tid, int const total_num, T const init_value) {
  for (int i = tid; i < total_num; i += BLOCK_SIZE) {
    value[i] = init_value;
  }
}

template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void setLocalExperts(int* s_local_experts, T const* token_selected_experts,
                                                int const total_num_experts, int const tid, int const start_expert, int const end_expert) {
  for (int i = tid; i < total_num_experts; i += BLOCK_SIZE) {
    int const expert = token_selected_experts[i];

    // If expert is in the current node, subtract start_expert to shift the range to [0, num_experts_per_node)
    bool is_valid_expert = expert >= start_expert && expert < end_expert;
    if (is_valid_expert) {
      int local_expert_id = expert - start_expert;
      if (s_local_experts[local_expert_id] == 0) {
        s_local_experts[local_expert_id] = 1;  // @TODO: Make sure that we allow duplicated write here
      }
    }
  }
  __syncthreads();
}

template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void prefixSum(T* out, T* in, int const num, int const tid) {
  typedef cub::BlockScan<T, BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage tempStorage;

  T threadData = 0;
  if (tid < num) {
    threadData = in[tid];
  }

  BlockScan(tempStorage).InclusiveSum(threadData, threadData);
  __syncthreads();

  if (tid < num) {
    out[tid] = threadData;
  }
  __syncthreads();
}

__device__ __forceinline__ void setActiveNum(int& num_active, int& num_active_offset_start, int& num_active_offset_end,
                                             int const cluster_size, int const cluster_rank) {
  int num_remainder = num_active % cluster_size;
  int num_active_per_node = max(0, num_active - 1) / cluster_size;  // num_active_per_node shouldn't be neg
  if (cluster_rank < num_remainder) {
    num_active = num_active_per_node + 1;
    num_active_offset_start = cluster_rank * num_active;
  } else {
    num_active = num_active_per_node;
    num_active_offset_start = cluster_rank * num_active_per_node + num_remainder;
  }
  num_active_offset_end = num_active_offset_start + num_active;
}

template <int BLOCK_SIZE>
__global__ void buildMinLatencyActiveExpertMapsKernel(int* num_active_experts_per_node, float* experts_to_token_scores,
                                                      int* active_expert_global_ids, int64_t* expert_first_token_offset, int const* token_selected_experts,
                                                      float const* token_final_scales, int64_t const num_tokens, int const num_experts_per_token, int const start_expert,
                                                      int const end_expert, int const num_experts_per_node, bool const smart_routing, int const cluster_rank,
                                                      int const cluster_size, int const num_experts_smem) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  // Use one block to process the min latency case
  int tid = threadIdx.x;
  // 0. init the global memory experts_to_token_scores [num_experts_per_node, num_token]
  int const total_local_scales = num_experts_per_node * num_tokens;
  initTensor<float, BLOCK_SIZE>(experts_to_token_scores, tid, total_local_scales, 0.0f);
  initTensor<int, BLOCK_SIZE>(active_expert_global_ids, tid, num_experts_per_node, -1);

  __threadfence();  //@Todo: check do I need this fence for previous zero setting

  // 1. mask for the active expert: 1 stands for active
  extern __shared__ int s_local_experts[];
  int* s_store_experts = s_local_experts + num_experts_smem;
  initTensor<int, BLOCK_SIZE>(s_local_experts, tid, num_experts_smem, 0);
  __syncthreads();

  // 2. set the shared array s_local_experts[]
  int const total_num_experts = num_tokens * num_experts_per_token;
  setLocalExperts<int, BLOCK_SIZE>(
      s_local_experts, token_selected_experts, total_num_experts, tid, start_expert, end_expert);

  // 3. perform prefix sum to acquire the store position and total active experts
  //@TODO: Use cub first, might need to change it to self-defined api
  prefixSum<int, BLOCK_SIZE>(s_store_experts, s_local_experts, num_experts_smem, tid);

  // 4. store the num of active experts
  int num_active = s_store_experts[num_experts_smem - 1];
  int num_active_offset_start = 0;
  int num_active_offset_end = 0;

  if (smart_routing) {
    setActiveNum(num_active, num_active_offset_start, num_active_offset_end, cluster_size, cluster_rank);
  }

  if (tid == 0) {
    *num_active_experts_per_node = num_active;
  }

  // 5. store the global expert id for each expert
  if (smart_routing) {
    for (int i = tid; i < num_experts_smem; i += BLOCK_SIZE) {
      if (s_local_experts[i]) {
        int offset = s_store_experts[i] - 1;
        if (offset >= num_active_offset_start && offset < num_active_offset_end) {
          active_expert_global_ids[offset - num_active_offset_start] = i;
        } else {
          s_local_experts[i] = 0;
        }
      }
    }
    __syncthreads();  // Need sync to update the s_local_experts
  } else {
    for (int i = tid; i < num_experts_smem; i += BLOCK_SIZE) {
      if (s_local_experts[i]) {
        int offset = s_store_experts[i] - 1;
        active_expert_global_ids[offset] = i + start_expert;
      }
    }
  }

  // 6. store the scale values
  __threadfence();  //@Todo: check do I need this fence for previous zero setting
  for (int i = tid; i < total_num_experts; i += BLOCK_SIZE) {
    int const expert = token_selected_experts[i];

    // If expert is not in the current node, set it to num_experts_per_node
    // If expert is in the current node, subtract start_expert to shift the range to [0, num_experts_per_node)
    bool is_valid_expert = smart_routing ? s_local_experts[expert] : (expert >= start_expert && expert < end_expert);

    if (is_valid_expert) {
      int token = i / num_experts_per_token;
      float const scale = token_final_scales[i];
      int offset = s_store_experts[expert - start_expert] - 1 - num_active_offset_start;
      experts_to_token_scores[offset * num_tokens + token] = scale;
    }
  }
  // 7. set default value for redundant memory
  for (int i_exp = num_active + tid; i_exp < num_experts_per_node; i_exp += BLOCK_SIZE) {
    active_expert_global_ids[i_exp] = -1;
  }
  // 8. set expert_first_token_offset
  for (int i_exp = tid; i_exp < num_experts_per_node + 1; i_exp += BLOCK_SIZE) {
    if (i_exp < num_active) {
      expert_first_token_offset[i_exp] = i_exp * num_tokens;
    } else {
      expert_first_token_offset[i_exp] = num_active * num_tokens;
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void buildMinLatencyActiveExpertMaps(int* num_active_experts_per_node, float* experts_to_token_scores,
                                     int* active_expert_global_ids, int64_t* expert_first_token_offset, int const* token_selected_experts,
                                     float const* token_final_scales, int64_t const num_tokens, int const experts_per_token, int const start_expert,
                                     int const end_expert, int const num_experts_per_node, int const cluster_rank, int const cluster_size,
                                     int const num_experts_smem, cudaStream_t const stream) {
  ORT_ENFORCE(num_experts_per_node == (end_expert - start_expert),
              "num_experts_per_node must be equal to end_expert - start_expert");

  ORT_ENFORCE(num_experts_per_node <= 256, "don't support num_experts_per_node > 256 cases");

  int const threads = 256;
  int const blocks = 1;
  bool const smart_routing = cluster_size > 1;

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = num_experts_smem * sizeof(int) * 2;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, buildMinLatencyActiveExpertMapsKernel<threads>, num_active_experts_per_node,
                     experts_to_token_scores, active_expert_global_ids, expert_first_token_offset, token_selected_experts,
                     token_final_scales, num_tokens, experts_per_token, start_expert, end_expert, num_experts_per_node,
                     smart_routing, cluster_rank, cluster_size, num_experts_smem);
}

template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
__global__ void fusedBuildExpertMapsSortFirstTokenKernel(int const* const token_selected_experts,
                                                         int* const permuted_row_to_unpermuted_row, int* const unpermuted_row_to_permuted_row,
                                                         int64_t* const expert_first_token_offset, int64_t const num_tokens, int const experts_per_token,
                                                         int const start_expert, int const end_expert, int const num_experts_per_node) {
  // Only using block wise collective so we can only have one block
  assert(gridDim.x == 1);

  assert(start_expert <= end_expert);
  assert(num_experts_per_node == (end_expert - start_expert));
  assert(num_experts_per_node <= (1 << LOG2_NUM_EXPERTS));

  int const token = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  bool is_valid_token = token < num_tokens;

  // This is the masked expert id for this token
  int local_token_selected_experts[EXPERTS_PER_TOKEN];
  // This is the final permuted rank of this token (ranked by selected expert)
  int local_token_permuted_indices[EXPERTS_PER_TOKEN];

  // Wait PDL before reading token_selected_experts
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

// build expert map
// we need to populate expert ids for all threads, even if there are
// fewer tokens
#pragma unroll
  for (int i = 0; i < EXPERTS_PER_TOKEN; i++) {
    int const expert = is_valid_token ? token_selected_experts[token * EXPERTS_PER_TOKEN + i] : num_experts_per_node;

    // If the token is not valid, set the expert id to num_experts_per_node + 1
    // If expert is not in the current node, set it to num_experts_per_node
    // If expert is in the current node, subtract start_expert to shift the range to [0, num_experts_per_node)
    bool is_valid_expert = expert >= start_expert && expert < end_expert;
    local_token_selected_experts[i] = !is_valid_token   ? num_experts_per_node + 1
                                      : is_valid_expert ? (expert - start_expert)
                                                        : num_experts_per_node;
  }

  // TODO: decompose cub's sort to expose the bucket starts, and just return
  // that to elide the binary search

  // sort the expert map
  using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
  extern __shared__ unsigned char temp_storage[];
  auto& sort_temp = *reinterpret_cast<typename BlockRadixRank::TempStorage*>(temp_storage);

  // Sanity check that the number of bins do correspond to the number of experts
  static_assert(BlockRadixRank::BINS_TRACKED_PER_THREAD * BLOCK_SIZE >= (1 << LOG2_NUM_EXPERTS));
  assert(BlockRadixRank::BINS_TRACKED_PER_THREAD * BLOCK_SIZE >= num_experts_per_node);

  int local_expert_first_token_offset[BlockRadixRank::BINS_TRACKED_PER_THREAD];

  cub::BFEDigitExtractor<int> extractor(0, LOG2_NUM_EXPERTS);
  BlockRadixRank(sort_temp).RankKeys(
      local_token_selected_experts, local_token_permuted_indices, extractor, local_expert_first_token_offset);

// We are done with compute, launch the dependent kernels while the stores are in flight
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  // write to shared memory and global memory
  if (is_valid_token) {
#pragma unroll
    for (int i = 0; i < EXPERTS_PER_TOKEN; i++) {
      int const unpermuted_row = i * num_tokens + token;
      int const permuted_row = local_token_permuted_indices[i];
      permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
      unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
    }
  }

#pragma unroll
  for (int expert_id = 0; expert_id < BlockRadixRank::BINS_TRACKED_PER_THREAD; expert_id++) {
    int out_expert_id = expert_id + token * BlockRadixRank::BINS_TRACKED_PER_THREAD;
    if (out_expert_id < num_experts_per_node + 1) {
      expert_first_token_offset[out_expert_id] = local_expert_first_token_offset[expert_id];
    }
  }
}

template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenDispatch(int const* token_selected_experts, int* permuted_row_to_unpermuted_row,
                                                int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset, int64_t const num_tokens,
                                                int const num_experts_per_node, int const experts_per_token, int const start_expert, int const end_expert,
                                                cudaStream_t stream) {
  ORT_ENFORCE(num_experts_per_node == (end_expert - start_expert),
              "num_experts_per_node must be equal to end_expert - start_expert");
  int const threads = BLOCK_SIZE;
  int const blocks = (num_tokens + threads - 1) / threads;
  ORT_ENFORCE(blocks == 1, "Current implementation requires single block");

  using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
  size_t shared_size = sizeof(typename BlockRadixRank::TempStorage);

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = shared_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;

  auto kernel = &fusedBuildExpertMapsSortFirstTokenKernel<BLOCK_SIZE, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;

  int device = 0;
  int max_smem_per_block = 0;
  CUDA_CALL_THROW(cudaGetDevice(&device));
  CUDA_CALL_THROW(cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  if (shared_size >= static_cast<size_t>(max_smem_per_block)) {
    // This should mean that
    // cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)
    // wouldn't work.
    return false;
  }

  CUDA_CALL_THROW(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));
  CUDA_CALL_THROW(cudaLaunchKernelEx(&config, kernel, token_selected_experts, permuted_row_to_unpermuted_row,
                                     unpermuted_row_to_permuted_row, expert_first_token_offset, num_tokens, experts_per_token, start_expert,
                                     end_expert, num_experts_per_node));

  return true;
}

template <int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenBlockSize(int const* token_selected_experts, int* permuted_row_to_unpermuted_row,
                                                 int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset, int64_t const num_tokens,
                                                 int const num_experts_per_node, int const experts_per_token, int const start_expert, int const end_expert,
                                                 cudaStream_t stream) {
  int const block_size = num_tokens;
  if (num_tokens > 256) {
    ORT_LLM_LOG_TRACE(onnxruntime::MakeString("Number of tokens ", num_tokens, " is greater than 256, which is not supported for fused moe prologues"));
    return false;
  }

  auto func = &fusedBuildExpertMapsSortFirstTokenDispatch<32, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  if (block_size > 32 && block_size <= 64) {
    func = &fusedBuildExpertMapsSortFirstTokenDispatch<64, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  } else if (block_size > 64 && block_size <= 128) {
    func = &fusedBuildExpertMapsSortFirstTokenDispatch<128, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  } else if (block_size > 128 && block_size <= 256) {
    func = &fusedBuildExpertMapsSortFirstTokenDispatch<256, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  }

  return func(token_selected_experts, permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row,
              expert_first_token_offset, num_tokens, num_experts_per_node, experts_per_token, start_expert, end_expert,
              stream);
}

template <int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenBlockSize(int const* token_selected_experts, int* permuted_row_to_unpermuted_row,
                                                 int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset, int64_t const num_tokens,
                                                 int const num_experts_per_node, int const experts_per_token, int const start_expert, int const end_expert,
                                                 cudaStream_t stream) {
  auto func = &fusedBuildExpertMapsSortFirstTokenBlockSize<1, LOG2_NUM_EXPERTS>;
  switch (experts_per_token) {
    case 1: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<1, LOG2_NUM_EXPERTS>;
      break;
    }
    case 2: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<2, LOG2_NUM_EXPERTS>;
      break;
    }
    case 4: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<4, LOG2_NUM_EXPERTS>;
      break;
    }
    case 6: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<6, LOG2_NUM_EXPERTS>;
      break;
    }
    case 8: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<8, LOG2_NUM_EXPERTS>;
      break;
    }
    default: {
      ORT_LLM_LOG_TRACE(onnxruntime::MakeString("Top-K value ", experts_per_token, " does not have supported fused moe prologues"));
      return false;
    }
  }
  return func(token_selected_experts, permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row,
              expert_first_token_offset, num_tokens, num_experts_per_node, experts_per_token, start_expert, end_expert,
              stream);
}

bool fusedBuildExpertMapsSortFirstToken(int const* token_selected_experts, int* permuted_row_to_unpermuted_row,
                                        int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset, int64_t const num_tokens,
                                        int const num_experts_per_node, int const experts_per_token, int const start_expert, int const end_expert,
                                        cudaStream_t stream) {
  // We need enough bits to represent [0, num_experts_per_node+1] (inclusive) i.e. num_experts_per_node + 2 values
  // This is floor(log2(num_experts_per_node+1)) + 1
  int expert_log = static_cast<int>(log2(num_experts_per_node + 1)) + 1;
  if (expert_log <= 9) {
    auto funcs = std::array{&fusedBuildExpertMapsSortFirstTokenBlockSize<1>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<2>, &fusedBuildExpertMapsSortFirstTokenBlockSize<3>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<4>, &fusedBuildExpertMapsSortFirstTokenBlockSize<5>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<6>, &fusedBuildExpertMapsSortFirstTokenBlockSize<7>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<8>, &fusedBuildExpertMapsSortFirstTokenBlockSize<9>};

    return funcs[expert_log - 1](token_selected_experts, permuted_row_to_unpermuted_row,
                                 unpermuted_row_to_permuted_row, expert_first_token_offset, num_tokens, num_experts_per_node,
                                 experts_per_token, start_expert, end_expert, stream);
  }
  ORT_LLM_LOG_TRACE(onnxruntime::MakeString("Experts per node ", num_experts_per_node, " does not have supported fused moe prologues"));
  return false;
}

int64_t computeNumTokensPerBlock(int64_t const num_tokens, int64_t const num_experts_per_node) {
  for (int64_t num_tokens_per_block = 32; num_tokens_per_block <= 1024; num_tokens_per_block *= 2) {
    int64_t const num_blocks_per_seq = onnxruntime::llm::common::ceilDiv(num_tokens, num_tokens_per_block);
    if (num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block) {
      return num_tokens_per_block;
    }
  }
  return 1024;
}

template <int kNumTokensPerBlock>
__global__ void blockExpertPrefixSumKernel(int const* token_selected_experts, int* blocked_expert_counts,
                                           int* blocked_row_to_unpermuted_row, int64_t const num_tokens, int64_t const num_experts_per_token,
                                           int const start_expert_id) {
  using BlockScan = cub::BlockScan<int, kNumTokensPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // target_expert_id and expert_id are offset by start_expert_id
  int const target_expert_id = blockIdx.x;
  int const block_id = blockIdx.y;
  int const num_blocks_per_seq = gridDim.y;
  int const token_id = block_id * kNumTokensPerBlock + threadIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  int expanded_token_id = -1;
  if (token_id < num_tokens) {
    for (int i = 0; i < num_experts_per_token; i++) {
      // TODO(enweiz): Fix uncoalesced access with shared memory.
      int const expert_id = token_selected_experts[token_id * num_experts_per_token + i] - start_expert_id;
      if (expert_id == target_expert_id) {
        expanded_token_id = i * num_tokens + token_id;
        break;
      }
    }
  }

  int const has_matched = expanded_token_id >= 0 ? 1 : 0;
  int index;
  BlockScan(temp_storage).ExclusiveSum(has_matched, index);

  if (has_matched) {
    blocked_row_to_unpermuted_row[target_expert_id * num_tokens + block_id * kNumTokensPerBlock + index] = expanded_token_id;
  }
  if (threadIdx.x == kNumTokensPerBlock - 1) {
    blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id] = index + has_matched;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void blockExpertPrefixSum(int const* token_selected_experts, int* blocked_expert_counts,
                          int* blocked_row_to_unpermuted_row, int64_t const num_tokens, int64_t const num_experts_per_node,
                          int64_t const num_experts_per_token, int64_t const num_tokens_per_block, int64_t const num_blocks_per_seq,
                          int const start_expert_id, cudaStream_t stream) {
  dim3 const blocks(num_experts_per_node, num_blocks_per_seq);
  dim3 const threads(num_tokens_per_block);

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;

  auto func = blockExpertPrefixSumKernel<1024>;
  if (num_tokens_per_block <= 32) {
    func = blockExpertPrefixSumKernel<32>;
  } else if (num_tokens_per_block <= 64) {
    func = blockExpertPrefixSumKernel<64>;
  } else if (num_tokens_per_block <= 128) {
    func = blockExpertPrefixSumKernel<128>;
  } else if (num_tokens_per_block <= 256) {
    func = blockExpertPrefixSumKernel<256>;
  } else if (num_tokens_per_block <= 512) {
    func = blockExpertPrefixSumKernel<512>;
  }
  cudaLaunchKernelEx(&config, func, token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row,
                     num_tokens, num_experts_per_token, start_expert_id);
}

template <int kNumThreadsPerBlock>
__global__ void globalExpertPrefixSumLargeKernel(int const* blocked_expert_counts, int* blocked_expert_counts_cumsum,
                                                 int64_t* expert_first_token_offset, int64_t const num_experts_per_node, int64_t const num_blocks_per_seq,
                                                 int64_t const num_elem_per_thread) {
  using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int offset = threadIdx.x * num_elem_per_thread;
  int cnt = 0;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // Note: Because of limited registers, cannot store thread-level prefix sum or enable #pragma unroll
  for (int i = 0; i < num_elem_per_thread; i++) {
    // TODO(enweiz): Fix uncoalesced access with shared memory.
    if (offset + i < num_experts_per_node * num_blocks_per_seq) {
      cnt += blocked_expert_counts[offset + i];
    }
  }

  int cumsum;
  BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

  for (int i = 0; i < num_elem_per_thread; i++) {
    if (offset + i < num_experts_per_node * num_blocks_per_seq) {
      blocked_expert_counts_cumsum[offset + i] = cumsum;
      if ((offset + i) % num_blocks_per_seq == 0) {
        expert_first_token_offset[(offset + i) / num_blocks_per_seq] = cumsum;
      }
      cumsum += blocked_expert_counts[offset + i];
      if ((offset + i) == num_experts_per_node * num_blocks_per_seq - 1) {
        expert_first_token_offset[num_experts_per_node] = cumsum;
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <int kNumThreadsPerBlock>
__global__ void globalExpertPrefixSumKernel(int const* blocked_expert_counts, int* blocked_expert_counts_cumsum,
                                            int64_t* expert_first_token_offset, int64_t const num_experts_per_node, int64_t const num_blocks_per_seq) {
  using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  int const cnt = threadIdx.x < num_experts_per_node * num_blocks_per_seq ? blocked_expert_counts[threadIdx.x] : 0;
  int cumsum;
  BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

  if (threadIdx.x < num_experts_per_node * num_blocks_per_seq) {
    blocked_expert_counts_cumsum[threadIdx.x] = cumsum;
    if (threadIdx.x % num_blocks_per_seq == 0) {
      expert_first_token_offset[threadIdx.x / num_blocks_per_seq] = cumsum;
    }
    if (threadIdx.x == num_experts_per_node * num_blocks_per_seq - 1) {
      expert_first_token_offset[num_experts_per_node] = cumsum + cnt;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void globalExpertPrefixSum(int const* blocked_expert_counts, int* blocked_expert_counts_cumsum,
                           int64_t* expert_first_token_offset, int64_t const num_experts_per_node, int64_t const num_tokens_per_block,
                           int64_t const num_blocks_per_seq, cudaStream_t stream) {
  int64_t const num_elements = num_experts_per_node * num_blocks_per_seq;

  cudaLaunchConfig_t config;
  config.gridDim = 1;
  config.blockDim = 1024;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;

  if (num_elements <= 1024) {
    auto func = globalExpertPrefixSumKernel<1024>;
    if (num_elements <= 32) {
      func = globalExpertPrefixSumKernel<32>;
      config.blockDim = 32;
    } else if (num_elements <= 64) {
      func = globalExpertPrefixSumKernel<64>;
      config.blockDim = 64;
    } else if (num_elements <= 128) {
      func = globalExpertPrefixSumKernel<128>;
      config.blockDim = 128;
    } else if (num_elements <= 256) {
      func = globalExpertPrefixSumKernel<256>;
      config.blockDim = 256;
    } else if (num_elements <= 512) {
      func = globalExpertPrefixSumKernel<512>;
      config.blockDim = 512;
    }
    cudaLaunchKernelEx(&config, func, blocked_expert_counts, blocked_expert_counts_cumsum,
                       expert_first_token_offset, num_experts_per_node, num_blocks_per_seq);
  } else {
    auto func = globalExpertPrefixSumLargeKernel<1024>;
    int64_t const num_elem_per_thread = onnxruntime::llm::common::ceilDiv(num_elements, 1024);
    cudaLaunchKernelEx(&config, func, blocked_expert_counts, blocked_expert_counts_cumsum,
                       expert_first_token_offset, num_experts_per_node, num_blocks_per_seq, num_elem_per_thread);
  }
}

__global__ void mergeExpertPrefixSumKernel(int const* blocked_expert_counts, int const* blocked_expert_counts_cumsum,
                                           int const* blocked_row_to_unpermuted_row, int* permuted_token_selected_experts, int* permuted_row_to_unpermuted_row,
                                           int* unpermuted_row_to_permuted_row, int const num_tokens) {
  int const target_expert_id = blockIdx.x;
  int const block_id = blockIdx.y;
  int const num_blocks_per_seq = gridDim.y;
  int const token_id = block_id * blockDim.x + threadIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  int const cnt = blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id];
  int const offset = blocked_expert_counts_cumsum[target_expert_id * num_blocks_per_seq + block_id];
  if (threadIdx.x < cnt) {
    int const unpermuted_row = blocked_row_to_unpermuted_row[target_expert_id * num_tokens + token_id];
    int const permuted_row = offset + threadIdx.x;
    permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
    permuted_token_selected_experts[permuted_row] = target_expert_id;
    unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void mergeExpertPrefixSum(int const* blocked_expert_counts, int const* blocked_expert_counts_cumsum,
                          int const* blocked_row_to_unpermuted_row, int* permuted_token_selected_experts, int* permuted_row_to_unpermuted_row,
                          int* unpermuted_row_to_permuted_row, int64_t const num_tokens, int64_t const num_experts_per_node,
                          int64_t const num_tokens_per_block, int64_t const num_blocks_per_seq, cudaStream_t stream) {
  dim3 const blocks(num_experts_per_node, num_blocks_per_seq);
  dim3 const threads(num_tokens_per_block);

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaLaunchKernelEx(&config, mergeExpertPrefixSumKernel, blocked_expert_counts, blocked_expert_counts_cumsum,
                     blocked_row_to_unpermuted_row, permuted_token_selected_experts, permuted_row_to_unpermuted_row,
                     unpermuted_row_to_permuted_row, num_tokens);
}

// threeStepBuildExpertMapsSortFirstToken uses three kernels to achieve the sort of token_selected_experts

// 1. blockExpertPrefixSumKernel launches [num_experts_per_node, num_blocks_per_seq] CTAs; each CTA has
// num_tokens_per_block threads. blocked_row_to_unpermuted_row points to a 2D buffer of size [num_experts_per_node,
// num_tokens], which can be viewed as [num_experts_per_node, num_blocks_per_seq] blocks, and each block has
// num_tokens_per_block tokens. Note that each CTA corresponds to a block in blocked_row_to_unpermuted_row. Within each
// CTA, the threads leverage cub::BlockScan to compute the offsets of tokens that activate the target expert. If a
// thread's token activates the target expert, the thread stores its unpermuted_row to the buffer block with the offset.
// In addition, the kernel also stores the expert counts for each block to another 2D buffer blocked_expert_counts of
// size [num_experts_per_node, num_blocks_per_seq].

// 2. globalExpertPrefixSumKernel launches 1 CTA; that CTA has num_experts_per_node * num_blocks_per_seq threads.
// The kernel views blocked_expert_counts as a 1D buffer, and leverages cub::BlockScan to compute the prefix sum of the
// expert counts for each block. The prefix sum is stored to blocked_expert_counts_cumsum.

// 3. mergeExpertPrefixSumKernel launches [num_experts_per_node, num_blocks_per_seq] CTAs; each CTA has
// num_tokens_per_block threads. Each CTA obtains the block-level offset from blocked_expert_counts_cumsum, and thus
// compacts blocked_row_to_unpermuted_row to permuted_row_to_unpermuted_row. In addition, with the block-level offsets,
// the kernel fills permuted_token_selected_experts.

// computeNumTokensPerBlock decides num_tokens_per_block. Note that both blockExpertPrefixSumKernel and
// globalExpertPrefixSumKernel leverage cub::BlockScan, and their CTA sizes are num_tokens_per_block and
// num_experts_per_node * num_blocks_per_seq, respectively. computeNumTokensPerBlock tries to find a minimum CTA size
// for both kernels, so that the block-leval cub::BlockScan can be efficient.

void threeStepBuildExpertMapsSortFirstToken(int const* token_selected_experts, int* permuted_token_selected_experts,
                                            int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
                                            int* blocked_expert_counts, int* blocked_expert_counts_cumsum, int* blocked_row_to_unpermuted_row,
                                            int64_t const num_tokens, int64_t const num_experts_per_node, int64_t const num_experts_per_token,
                                            int const start_expert_id, cudaStream_t stream) {
  int64_t const num_tokens_per_block = computeNumTokensPerBlock(num_tokens, num_experts_per_node);
  int64_t const num_blocks_per_seq = onnxruntime::llm::common::ceilDiv(num_tokens, num_tokens_per_block);

  blockExpertPrefixSum(token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row, num_tokens,
                       num_experts_per_node, num_experts_per_token, num_tokens_per_block, num_blocks_per_seq, start_expert_id, stream);
  sync_check_cuda_error(stream);

  globalExpertPrefixSum(blocked_expert_counts, blocked_expert_counts_cumsum, expert_first_token_offset,
                        num_experts_per_node, num_tokens_per_block, num_blocks_per_seq, stream);
  sync_check_cuda_error(stream);

  mergeExpertPrefixSum(blocked_expert_counts, blocked_expert_counts_cumsum, blocked_row_to_unpermuted_row,
                       permuted_token_selected_experts, permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row, num_tokens,
                       num_experts_per_node, num_tokens_per_block, num_blocks_per_seq, stream);
}

// ============================== Infer GEMM sizes =================================
// TODO Could linear search be better for small # experts
template <class T>
__device__ inline int64_t findTotalEltsLessThanTarget(T const* sorted_indices, int64_t const arr_length, T const target) {
  int64_t low = 0, high = arr_length - 1, target_location = -1;
  while (low <= high) {
    int64_t mid = (low + high) / 2;

    if (sorted_indices[mid] >= target) {
      high = mid - 1;
    } else {
      low = mid + 1;
      target_location = mid;
    }
  }
  return target_location + 1;
}

template <class T>
using sizeof_bits = cutlass::sizeof_bits<typename cutlass_kernels::CudaToCutlassTypeAdapter<std::remove_cv_t<T>>::type>;

// Function to safely offset an pointer that may contain sub-byte types (FP4/INT4)
template <class T>
__host__ __device__ constexpr T* safe_inc_ptr(T* ptr, size_t offset) {
  constexpr int adjustment = (sizeof_bits<T>::value < 8) ? (8 / sizeof_bits<T>::value) : 1;
  assert(offset % adjustment == 0 && "Attempt to offset index to sub-byte");
  return ptr + offset / adjustment;
}

__host__ __device__ constexpr int64_t getOffsetWeightSF(int64_t expert_id, int64_t gemm_n, int64_t gemm_k,
                                                        TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType scaling_type) {
  auto function = [=](int64_t min_n_dim_alignment, int64_t min_k_dim_alignment, int64_t block_size) {
    int64_t padded_gemm_n = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(gemm_n, min_n_dim_alignment);
    int64_t padded_gemm_k = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(gemm_k, min_k_dim_alignment);
    assert(gemm_k % block_size == 0);
    return expert_id * padded_gemm_n * padded_gemm_k / block_size;
  };
  switch (scaling_type) {
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE:
      return 0;  // No scaling factors, no offset
  }

  assert(false && "Unrecognized scaling type");
  return 0;
}

__host__ __device__ constexpr int64_t getOffsetActivationSF(int64_t expert_id, int64_t token_offset, int64_t gemm_k,
                                                            TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType scaling_type) {
  auto function = [=](int64_t min_n_dim_alignment, int64_t min_k_dim_alignment, int64_t block_size) {
    // This formulation ensures that:
    // `sf_offset[i + 1] - sf_offset[i] >= padded(token_offset[i + 1] - token_offset[i])`
    // is true for all possible token distributions.
    int64_t padded_sf_start_offset = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
        token_offset + expert_id * (min_n_dim_alignment - 1), min_n_dim_alignment);
    int64_t padded_gemm_k = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(gemm_k, min_k_dim_alignment);
    assert(gemm_k % block_size == 0);
    assert(padded_gemm_k % block_size == 0);
    return padded_sf_start_offset * padded_gemm_k / block_size;
  };
  switch (scaling_type) {
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE:
      return 0;  // No scaling factors, no offset
  }

  assert(false && "Unrecognized scaling type");
  return 0;
}

template <class GemmOutputType, class QuantizedType, class ComputeElem, int VecSize>
__device__ auto quantizePackedFPXValue(ComputeElem& post_act_val, float global_scale_val,
                                       int64_t num_tokens_before_expert, int64_t expert_id, int64_t token_id, int64_t elem_idx, int64_t num_cols,
                                       TmaWarpSpecializedGroupedGemmInput::ElementSF* act_sf_flat,
                                       TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType scaling_type) {
  constexpr bool is_fp8 = std::is_same_v<QuantizedType, __nv_fp8_e4m3>;
  static constexpr int NumThreadsPerSF = VecSize / CVT_FP4_ELTS_PER_THREAD;
  // Quantize the input to FP4
  static_assert(std::is_same_v<GemmOutputType, __nv_bfloat16> || std::is_same_v<GemmOutputType, half>);
  static_assert(ComputeElem::kElements == CVT_FP4_ELTS_PER_THREAD);
  PackedVec<GemmOutputType> packed_vec{};
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    packed_vec.elts[i].x = static_cast<GemmOutputType>(post_act_val[i * 2 + 0]);
    packed_vec.elts[i].y = static_cast<GemmOutputType>(post_act_val[i * 2 + 1]);
  }

  // We need to offset into the scaling factors for just this expert
  auto act_sf_expert = act_sf_flat + getOffsetActivationSF(expert_id, num_tokens_before_expert, num_cols, scaling_type);

  // Use `token - num_tokens_before_expert` because we want this to be relative to the start of this expert
  auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF, NumThreadsPerSF, VecSize>(
      std::nullopt /* batchIdx */, token_id - num_tokens_before_expert, elem_idx, std::nullopt /* numRows */,
      num_cols, act_sf_expert, FP4QuantizationSFLayout::SWIZZLED);

  // Do the conversion and set the output and scaling factor
  auto func = [&]() {
    if constexpr (is_fp8) {
      return [](PackedVec<GemmOutputType>& vec, float /* ignored */, uint8_t* SFout) -> uint64_t {
        static_assert(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize == VecSize);
        return cvt_warp_fp16_to_mxfp8<GemmOutputType, VecSize>(vec, SFout);
      };
    } else {
      return (scaling_type == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4)
                 ? &cvt_warp_fp16_to_fp4<GemmOutputType, VecSize, false>
                 : &cvt_warp_fp16_to_fp4<GemmOutputType, VecSize, true>;
    }
  }();

  return func(packed_vec, global_scale_val, sf_out);
}

template <int VecSize, int ElementsPerThread>
__device__ void writeSF(int64_t num_tokens_before_expert, int64_t expert_id, int64_t source_token_id, int64_t token_id,
                        int64_t elem_idx, int64_t num_cols, TmaWarpSpecializedGroupedGemmInput::ElementSF* act_sf_flat,
                        TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf) {
  static constexpr int NumThreadsPerSF = VecSize / ElementsPerThread;

  // We need to offset into the scaling factors for just this expert
  auto act_sf_expert = act_sf_flat + getOffsetActivationSF(expert_id, num_tokens_before_expert, num_cols,
                                                           (VecSize == TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize)
                                                               ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                                                               : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);

  // Use `token - num_tokens_before_expert` because we want this to be relative to the start of this expert
  auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF, NumThreadsPerSF, VecSize>(
      std::nullopt /* batchIdx */, token_id - num_tokens_before_expert, elem_idx, std::nullopt /* numRows */,
      num_cols, act_sf_expert, FP4QuantizationSFLayout::SWIZZLED);
  if (sf_out) {
    if (input_sf) {
      auto const sf_in = cvt_quant_to_fp4_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF, NumThreadsPerSF,
                                                            VecSize>(std::nullopt /* batchIdx */, source_token_id, elem_idx, std::nullopt /* numRows */,
                                                                     num_cols, const_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(input_sf),
                                                                     FP4QuantizationSFLayout::SWIZZLED);
      *sf_out = *sf_in;
    } else {
      *sf_out = 0x00;
    }
  }
}

// ====================== Compute FP8 dequant scale only ===============================
__global__ void computeFP8DequantScaleKernel(
    float const** alpha_scale_ptr_array, int64_t const num_experts_per_node, float const* fp8_dequant) {
  // First, compute the global tid. We only need 1 thread per expert.
  int const expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts_per_node) {
    return;
  }

  assert(fp8_dequant != nullptr);
  alpha_scale_ptr_array[expert] = fp8_dequant + expert;
}

float const** computeFP8DequantScale(
    float const** alpha_scale_ptr_array, int const num_experts_per_node, float const* fp8_dequant, cudaStream_t stream) {
  if (!fp8_dequant) {
    return nullptr;
  }

  int const threads = std::min(1024, num_experts_per_node);
  int const blocks = (num_experts_per_node + threads - 1) / threads;

  computeFP8DequantScaleKernel<<<blocks, threads, 0, stream>>>(
      alpha_scale_ptr_array, num_experts_per_node, fp8_dequant);

  return alpha_scale_ptr_array;
}

template <class BSConfig>
__device__ void setupFP4BlockScalingFactors(TmaWarpSpecializedGroupedGemmInput& layout_info, int expert, int gemm_m,
                                            int gemm_n, int gemm_k, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat,
                                            TmaWarpSpecializedGroupedGemmInput::ElementSF const* weight_block_scale, int64_t num_tokens_before_expert) {
  assert(layout_info.fpX_block_scaling_factors_stride_A);
  assert(layout_info.fpX_block_scaling_factors_stride_B);

  // M & N swapped for transpose
  auto stride_a_ptr = reinterpret_cast<typename BSConfig::LayoutSF*>(layout_info.fpX_block_scaling_factors_stride_A);
  auto stride_b_ptr = reinterpret_cast<typename BSConfig::LayoutSF*>(layout_info.fpX_block_scaling_factors_stride_B);
  stride_a_ptr[expert] = BSConfig::tile_atom_to_shape_SFB(cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, (int)1));
  stride_b_ptr[expert] = BSConfig::tile_atom_to_shape_SFA(cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, (int)1));

  // This assert validates our current assumption that A&B can be safely transposed without needing to modify
  assert(BSConfig::tile_atom_to_shape_SFB(cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, 1)) == BSConfig::tile_atom_to_shape_SFA(cute::make_shape((int)gemm_m, (int)gemm_n, (int)gemm_k, 1)));

  auto scaling_type = std::is_same_v<BSConfig, TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig>
                          ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                          : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
  layout_info.fpX_block_scaling_factors_A[expert] = fp4_act_flat + getOffsetActivationSF(expert, num_tokens_before_expert, gemm_k, scaling_type);

  layout_info.fpX_block_scaling_factors_B[expert] = weight_block_scale + getOffsetWeightSF(expert, gemm_n, gemm_k, scaling_type);
}

__device__ void computeTmaWarpSpecializedInputStrides(
    TmaWarpSpecializedGroupedGemmInput& layout_info, int gemm_m, int gemm_n, int gemm_k, int64_t out_idx) {
  layout_info.stride_a[out_idx] = cutlass::make_cute_packed_stride(
      TmaWarpSpecializedGroupedGemmInput::StrideA{}, cute::make_shape(gemm_m, gemm_k, 1));
  layout_info.stride_b[out_idx] = cutlass::make_cute_packed_stride(
      TmaWarpSpecializedGroupedGemmInput::StrideB{}, cute::make_shape(gemm_n, gemm_k, 1));
  if (layout_info.stride_c) {
    assert(false && "CUTLASS does not support a 1xN bias");
    //        layout_info.stride_c[out_idx] = cute::make_stride(0, cute::Int<1>{}, 0);
    layout_info.stride_c[out_idx] = cutlass::make_cute_packed_stride(
        TmaWarpSpecializedGroupedGemmInput::StrideC{}, cute::make_shape(1, gemm_n, 1));
  }
  if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE) {
    layout_info.default_epilogue.stride_d[out_idx] = cutlass::make_cute_packed_stride(
        TmaWarpSpecializedGroupedGemmInput::DefaultEpilogue::StrideD{}, cute::make_shape(gemm_n, gemm_m, 1));
  }
  if (layout_info.int4_groupwise_params.enabled) {
    layout_info.int4_groupwise_params.stride_s_a[out_idx] = cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::StrideSFA{},
                                                                                             cute::make_shape(gemm_n, gemm_k / 128, 1));
  }
}

template <class T, class WeightType, class OutputType, class ScaleBiasType>
__device__ void computeTmaWarpSpecializedInputPointers(TmaWarpSpecializedGroupedGemmInput& layout_info, int64_t gemm_m,
                                                       int64_t gemm_n, int64_t gemm_k, int num_tokens_before_expert, int64_t expert, T const* in,
                                                       WeightType const* weights, TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA const* w4a8_weight_scale,
                                                       ScaleBiasType const* bias, OutputType* output, int64_t const out_idx) {
  // The input prior to this contains K elements per token, with `num_tokens_before_expert` tokens
  layout_info.ptr_a[out_idx] = safe_inc_ptr(in, num_tokens_before_expert * gemm_k);

  // Each expert's weight matrix is a constant size NxK, get the matrix at index `expert`
  layout_info.ptr_b[out_idx] = safe_inc_ptr(weights, expert * (gemm_n * gemm_k));

  if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE) {
    // The output prior to this contains N elements per token, with `num_tokens_before_expert` tokens
    layout_info.default_epilogue.ptr_d[out_idx] = safe_inc_ptr(output, num_tokens_before_expert * gemm_n);
  }
  if (layout_info.int4_groupwise_params.enabled) {
    layout_info.int4_groupwise_params.ptr_s_a[out_idx] = safe_inc_ptr(w4a8_weight_scale, expert * (gemm_n * gemm_k / 128));
  }
}

// TODO Some of this setup could be cached
template <class T, class WeightType, class OutputType, class ScaleBiasType>
__global__ void computeStridesTmaWarpSpecializedKernel(int64_t const* expert_first_token_offset,
                                                       TmaWarpSpecializedGroupedGemmInput layout_info1, TmaWarpSpecializedGroupedGemmInput layout_info2,
                                                       int64_t num_tokens, int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k,
                                                       int64_t const num_experts_per_node, T const* gemm1_in, T const* gemm2_in, WeightType const* weights1,
                                                       WeightType const* weights2, float const* alpha_scale_flat1, float const* alpha_scale_flat2,
                                                       TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
                                                       TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2, QuantParams quant_params,
                                                       ScaleBiasType const* bias1, ScaleBiasType const* bias2, OutputType* gemm1_output, OutputType* gemm2_output) {
  // First, compute the global tid. We only need 1 thread per expert.
  int const expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts_per_node) {
    return;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // Both gemms use the same token offset
  auto const num_tokens_before_expert = expert_first_token_offset[expert];
  auto const num_tokens_including_expert = expert_first_token_offset[expert + 1];
  auto const num_tokens_to_expert = num_tokens_including_expert - num_tokens_before_expert;
  auto const gemm_m = num_tokens_to_expert;

  // M and N transposed since we are using the #tokens as the N dimension
  layout_info1.shape_info.problem_shapes[expert] = TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm1_n, gemm_m, gemm1_k);
  layout_info2.shape_info.problem_shapes[expert] = TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm2_n, gemm_m, gemm2_k);

  if (layout_info1.int4_groupwise_params.enabled) {
    layout_info1.int4_groupwise_params.shape.problem_shapes[expert] = TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::ProblemShapeInt::UnderlyingProblemShape(
        gemm1_n, gemm_m, gemm1_k);
  }

  if (layout_info2.int4_groupwise_params.enabled) {
    layout_info2.int4_groupwise_params.shape.problem_shapes[expert] = TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::ProblemShapeInt::UnderlyingProblemShape(
        gemm2_n, gemm_m, gemm2_k);
  }

  if (alpha_scale_flat1 && alpha_scale_flat2) {
    layout_info1.alpha_scale_ptr_array[expert] = alpha_scale_flat1 + expert;
    layout_info2.alpha_scale_ptr_array[expert] = alpha_scale_flat2 + expert;
  }

  auto setupIfSelected = [&](auto bs_config, auto quant_type) {
    if (quant_type.fc1.weight_block_scale) {
      setupFP4BlockScalingFactors<decltype(bs_config)>(layout_info1, expert, gemm_m, gemm1_n, gemm1_k,
                                                       fp4_act_flat1, quant_type.fc1.weight_block_scale, num_tokens_before_expert);
    }
    if (quant_type.fc2.weight_block_scale) {
      setupFP4BlockScalingFactors<decltype(bs_config)>(layout_info2, expert, gemm_m, gemm2_n, gemm2_k,
                                                       fp4_act_flat2, quant_type.fc2.weight_block_scale, num_tokens_before_expert);
    }
  };

  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig{}, quant_params.fp4);
  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaledConfig{}, quant_params.fp8_mxfp4);
  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaledConfig{}, quant_params.mxfp8_mxfp4);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
  assert(gemm_m <= INT32_MAX);
  assert(gemm1_n > 0 && gemm1_n <= INT32_MAX);
  assert(gemm1_k > 0 && gemm1_k <= INT32_MAX);
  assert(gemm2_n > 0 && gemm2_n <= INT32_MAX);
  assert(gemm2_k > 0 && gemm2_k <= INT32_MAX);
  computeTmaWarpSpecializedInputStrides(layout_info1, gemm_m, gemm1_n, gemm1_k, expert);
  computeTmaWarpSpecializedInputStrides(layout_info2, gemm_m, gemm2_n, gemm2_k, expert);

  computeTmaWarpSpecializedInputPointers(layout_info1, gemm_m, gemm1_n, gemm1_k, num_tokens_before_expert, expert,
                                         gemm1_in, weights1,
                                         reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA const*>(
                                             quant_params.groupwise.fc1.weight_scales),
                                         bias1, gemm1_output, expert);
  computeTmaWarpSpecializedInputPointers(layout_info2, gemm_m, gemm2_n, gemm2_k, num_tokens_before_expert, expert,
                                         gemm2_in, weights2,
                                         reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA const*>(
                                             quant_params.groupwise.fc2.weight_scales),
                                         bias2, gemm2_output, expert);
}

template <class T, class WeightType, class OutputType, class ScaleBiasType>
__global__ void computeStridesTmaWarpSpecializedLowLatencyKernel(TmaWarpSpecializedGroupedGemmInput layout_info1,
                                                                 TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens, int64_t gemm1_n, int64_t gemm1_k,
                                                                 int64_t gemm2_n, int64_t gemm2_k, int64_t const num_experts_per_node, T const* in1, T const* in2,
                                                                 WeightType const* weights1, WeightType const* weights2, float const* alpha_scale_flat1,
                                                                 float const* alpha_scale_flat2, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
                                                                 TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2, QuantParams quant_params,
                                                                 ScaleBiasType const* bias1, ScaleBiasType const* bias2, OutputType* output1, OutputType* output2,
                                                                 int const* num_active_experts_per, int const* active_expert_global_ids, int start_expert) {
  // First, compute the global tid. We only need 1 thread per expert.
  int const expert = blockIdx.x * blockDim.x + threadIdx.x;

  if (expert >= num_experts_per_node) {
    return;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // Note: expert is used to calculate the offset of the input and output
  // local_expert is used to calculate the offset of the weight
  auto const num_tokens_before_expert = expert * num_tokens;
  bool const is_active_expert = expert < *num_active_experts_per;
  int const local_expert = is_active_expert ? active_expert_global_ids[expert] - start_expert : -1;
  auto const gemm_m = is_active_expert ? num_tokens : 0;

  // M and N transposed since we are using the #tokens as the N dimension
  layout_info1.shape_info.problem_shapes[expert] = TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm1_n, gemm_m, gemm1_k);
  layout_info2.shape_info.problem_shapes[expert] = TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm2_n, gemm_m, gemm2_k);

  if (alpha_scale_flat1) {
    assert(alpha_scale_flat2);
    if (is_active_expert) {
      layout_info1.alpha_scale_ptr_array[expert] = alpha_scale_flat1 + local_expert;
      layout_info2.alpha_scale_ptr_array[expert] = alpha_scale_flat2 + local_expert;
    } else {
      layout_info1.alpha_scale_ptr_array[expert] = nullptr;
      layout_info2.alpha_scale_ptr_array[expert] = nullptr;
    }
  }

  if (quant_params.fp4.fc1.weight_block_scale) {
    setupFP4BlockScalingFactors<TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig>(layout_info1, expert,
                                                                                            gemm_m, gemm1_n, gemm1_k, fp4_act_flat1, quant_params.fp4.fc1.weight_block_scale, num_tokens_before_expert);

    // Override the scaling factors, fc1 uses the same A input for all experts and the scaling factor B offsets from
    // the local expert index
    if (is_active_expert) {
      layout_info1.fpX_block_scaling_factors_A[expert] = fp4_act_flat1;
      layout_info1.fpX_block_scaling_factors_B[expert] = quant_params.fp4.fc1.weight_block_scale + getOffsetWeightSF(
                                                                                                       local_expert, gemm1_n, gemm1_k, TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4);
    } else {
      layout_info1.fpX_block_scaling_factors_A[expert] = nullptr;
      layout_info1.fpX_block_scaling_factors_B[expert] = nullptr;
    }
  }

  if (quant_params.fp4.fc2.weight_block_scale) {
    setupFP4BlockScalingFactors<TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig>(layout_info2, expert,
                                                                                            gemm_m, gemm2_n, gemm2_k, fp4_act_flat2, quant_params.fp4.fc2.weight_block_scale, num_tokens_before_expert);

    // Override the scaling factors, fc2 scaling factor B offsets by the local expert index
    if (is_active_expert) {
      layout_info2.fpX_block_scaling_factors_B[expert] = quant_params.fp4.fc2.weight_block_scale + getOffsetWeightSF(
                                                                                                       local_expert, gemm2_n, gemm2_k, TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4);
    } else {
      layout_info2.fpX_block_scaling_factors_A[expert] = nullptr;
      layout_info2.fpX_block_scaling_factors_B[expert] = nullptr;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  assert(gemm_m <= INT32_MAX);
  assert(gemm1_n > 0 && gemm1_n <= INT32_MAX);
  assert(gemm1_k > 0 && gemm1_k <= INT32_MAX);
  assert(gemm2_n > 0 && gemm2_n <= INT32_MAX);
  assert(gemm2_k > 0 && gemm2_k <= INT32_MAX);
  computeTmaWarpSpecializedInputStrides(layout_info1, gemm_m, gemm1_n, gemm1_k, expert);
  computeTmaWarpSpecializedInputStrides(layout_info2, gemm_m, gemm2_n, gemm2_k, expert);

  if (is_active_expert) {
    // Note: under low latency mode, we use the same input for all experts
    // so for gemm1, the inputs are the same,
    // for gemm2, we use the input generated by gemm1
    layout_info1.ptr_a[expert] = in1;
    layout_info2.ptr_a[expert] = safe_inc_ptr(in2, expert * num_tokens * gemm2_k);

    // Each expert's weight matrix is a constant size NxK, get the matrix at index `expert`
    layout_info1.ptr_b[expert] = safe_inc_ptr(weights1, local_expert * (gemm1_n * gemm2_k));
    layout_info2.ptr_b[expert] = safe_inc_ptr(weights2, local_expert * (gemm1_n * gemm2_k));

    assert(layout_info1.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE);
    layout_info1.default_epilogue.ptr_d[expert] = safe_inc_ptr(output1, expert * num_tokens * gemm1_n);

    if (layout_info2.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE) {
      // The output prior to this contains N elements per token, with `num_tokens` tokens
      layout_info2.default_epilogue.ptr_d[expert] = safe_inc_ptr(output2, expert * num_tokens * gemm2_n);
    }
  } else {
    layout_info1.ptr_a[expert] = nullptr;
    layout_info2.ptr_a[expert] = nullptr;
    layout_info1.ptr_b[expert] = nullptr;
    layout_info2.ptr_b[expert] = nullptr;

    layout_info1.default_epilogue.ptr_d[expert] = nullptr;
    if (layout_info2.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE) {
      layout_info2.default_epilogue.ptr_d[expert] = nullptr;
    }
  }
}

// ========================== Permutation things =======================================

template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input) {
  cutlass::NumericArrayConverter<typename U::Element, typename T::Element, U::kElements> converter;
  return converter(input);
}

// Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since we will have to
// duplicate some rows in the input matrix to match the dimensions. Duplicates will always get routed to separate
// experts in the end.

// Note that the permuted_row_to_unpermuted_row map referred to here has indices in the range (0,
// k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ... (k-1)*rows_in_input
// all map to row 0 in the original matrix. Thus, to know where to read in the source matrix, we simply take the modulus
// of the expanded index.

constexpr static int EXPAND_THREADS_PER_BLOCK = 256;

template <class InputActivationsType, class ExpandedActivationsType,
          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType BlockScalingType, bool PRE_QUANT_AWQ>
__global__ void expandInputRowsKernel(InputActivationsType const* unpermuted_input,
                                      ExpandedActivationsType* permuted_output, float const* unpermuted_scales, float* permuted_scales,
                                      int const* permuted_row_to_unpermuted_row, int64_t const num_tokens, int64_t const hidden_size, int64_t const k,
                                      float const* fc1_act_global_scale, bool use_per_expert_act_scale, int64_t const* expert_first_token_offset,
                                      TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
                                      TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, int64_t const num_experts_per_node,
                                      InputActivationsType const* prequant_scales = nullptr) {
  static_assert(BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE || !PRE_QUANT_AWQ,
                "AWQ and Block Scaling are mutually exclusive");
#ifdef ENABLE_FP4
  constexpr bool is_mxfp8 = std::is_same_v<ExpandedActivationsType, __nv_fp8_e4m3> && BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX && !PRE_QUANT_AWQ;
  constexpr bool is_mxfp8_input = is_mxfp8 && std::is_same_v<InputActivationsType, __nv_fp8_e4m3>;
  constexpr bool need_mxfp8_quant = is_mxfp8 && !is_mxfp8_input;
  constexpr bool is_nvfp4 = std::is_same_v<ExpandedActivationsType, __nv_fp4_e2m1> && BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4 && !PRE_QUANT_AWQ;
  constexpr bool is_nvfp4_input = is_nvfp4 && std::is_same_v<InputActivationsType, __nv_fp4_e2m1>;
  constexpr bool need_nvfp4_quant = is_nvfp4 && !is_nvfp4_input;
#else
  constexpr bool is_mxfp8 = false;
  constexpr bool is_mxfp8_input = false;
  constexpr bool need_mxfp8_quant = false;
  constexpr bool is_nvfp4 = false;
  constexpr bool is_nvfp4_input = false;
  constexpr bool need_nvfp4_quant = false;
#endif

  static_assert(need_nvfp4_quant || need_mxfp8_quant || PRE_QUANT_AWQ || std::is_same_v<InputActivationsType, ExpandedActivationsType>,
                "Only NVFP4, MXFP8 and WINT4_AFP8 supports outputting a different format as part of the expansion");

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  constexpr int VecSize = is_nvfp4 ? TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize
                                   : TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize;

  constexpr int64_t ELEM_PER_THREAD = (is_nvfp4 || is_mxfp8) ? CVT_FP4_ELTS_PER_THREAD : (128 / sizeof_bits<InputActivationsType>::value);

  // This should be VecSize * 4 elements
  // We assume at least VecSize alignment or the quantization will fail
  constexpr int64_t min_k_dim_alignment = is_nvfp4 ? TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4
                                                   : TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX;
  int64_t const padded_hidden_size = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(hidden_size, min_k_dim_alignment);

  int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];
  for (int64_t permuted_row = blockIdx.x; permuted_row < num_valid_tokens; permuted_row += gridDim.x) {
    int64_t const unpermuted_row = permuted_row_to_unpermuted_row[permuted_row];

    // Load 128-bits per thread

    constexpr int64_t ELEM_PER_BYTE = is_nvfp4_input ? 2 : 1;
    using DataElem = std::conditional_t<is_nvfp4_input, uint32_t,
                                        std::conditional_t<is_mxfp8_input, uint64_t, cutlass::Array<InputActivationsType, ELEM_PER_THREAD>>>;
    using OutputElem = std::conditional_t<is_nvfp4, uint32_t,
                                          std::conditional_t<is_mxfp8, uint64_t, cutlass::Array<ExpandedActivationsType, ELEM_PER_THREAD>>>;

    // Duplicate and permute rows
    int64_t const source_k_rank = unpermuted_row / num_tokens;
    int64_t const source_row = unpermuted_row % num_tokens;

    auto const* source_row_ptr = reinterpret_cast<DataElem const*>(unpermuted_input + source_row * hidden_size / ELEM_PER_BYTE);
    // Cast first to handle when this is FP4
    auto* dest_row_ptr = reinterpret_cast<OutputElem*>(permuted_output) + permuted_row * hidden_size / ELEM_PER_THREAD;

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = EXPAND_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = hidden_size / ELEM_PER_THREAD;
    assert(hidden_size % ELEM_PER_THREAD == 0);
    assert(hidden_size % VecSize == 0);

    if constexpr (is_nvfp4 || is_mxfp8) {
      static_assert(ELEM_PER_THREAD == 8, "Expecting 8 elements per thread for quantized types");
      int64_t expert = findTotalEltsLessThanTarget(
                           expert_first_token_offset, num_experts_per_node, (int64_t)permuted_row + 1) -
                       1;

      assert(!fc1_act_global_scale || is_nvfp4 && "Global scale is only supported for NVFP4");
      size_t act_scale_idx = use_per_expert_act_scale ? expert : 0;
      float global_scale_val = fc1_act_global_scale ? fc1_act_global_scale[act_scale_idx] : 1.0f;
      int64_t num_tokens_before_expert = expert_first_token_offset[expert];

      for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        auto in_vec = source_row_ptr[elem_index];
        if constexpr (need_nvfp4_quant || need_mxfp8_quant) {
          auto res = quantizePackedFPXValue<InputActivationsType, ExpandedActivationsType, DataElem, VecSize>(
              in_vec, global_scale_val, num_tokens_before_expert, expert, permuted_row, elem_index,
              padded_hidden_size, fc1_act_sf_flat,
              is_nvfp4 ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                       : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);
          static_assert(
              sizeof(res) == sizeof(*dest_row_ptr), "Quantized value must be the same size as the output");
          dest_row_ptr[elem_index] = res;
        } else {
          assert(act_scale_idx == 0 && "Cannot use per-expert act scale for pre-quantized activations");
          writeSF<VecSize, ELEM_PER_THREAD>(num_tokens_before_expert, expert, source_row, permuted_row,
                                            elem_index, padded_hidden_size, fc1_act_sf_flat, input_sf);
          dest_row_ptr[elem_index] = in_vec;
        }
      }

      // Pad zeros in the extra SFs along the K dimension, we do this to ensure there are no nan values in the
      // padded SF atom Use VecSize per thread since we are just writing out zeros so every thread can process a
      // whole vector
      size_t padding_start_offset = hidden_size / VecSize + start_offset;
      size_t padding_elems_in_col = padded_hidden_size / VecSize;
      for (int64_t elem_index = padding_start_offset; elem_index < padding_elems_in_col; elem_index += stride) {
        writeSF<VecSize, VecSize>(num_tokens_before_expert, expert, /*source_row*/ -1, permuted_row, elem_index,
                                  padded_hidden_size, fc1_act_sf_flat,
                                  /* input_sf */ nullptr);  // Pass nulltpr input_sf so we write 0
      }
    } else if constexpr (PRE_QUANT_AWQ) {
      static_assert(!is_nvfp4 && !is_mxfp8, "NVFP4 and MXFP8 are not supported for AWQ");
      static_assert(!std::is_same_v<InputActivationsType, ExpandedActivationsType>,
                    "Input and output types must be different for AWQ");
      for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        auto frag_elems = source_row_ptr[elem_index];

        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < ELEM_PER_THREAD; e++) {
          frag_elems[e] = frag_elems[e] * prequant_scales[elem_index * ELEM_PER_THREAD + e];
        }

        dest_row_ptr[elem_index] = arrayConvert<DataElem, OutputElem>(frag_elems);
      }
    } else {
      for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        dest_row_ptr[elem_index] = source_row_ptr[elem_index];
      }
    }

    if (permuted_scales && threadIdx.x == 0) {
      int64_t const source_k_idx = source_row * k + source_k_rank;
      permuted_scales[permuted_row] = unpermuted_scales ? unpermuted_scales[source_k_idx] : 1.0f;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  // Pad zeros in the extra SFs along the N dimension, we do this to ensure there are no nan values in the padded SF
  // atom
  if constexpr (is_nvfp4 || is_mxfp8) {
    int64_t const start_offset = threadIdx.x;
    int64_t const stride = EXPAND_THREADS_PER_BLOCK;
    // Use VecSize per thread since we are just writing out zeros so every thread can process a whole vector
    int64_t const padded_num_elems_in_col = padded_hidden_size / VecSize;
    assert(padded_hidden_size % VecSize == 0);

    constexpr int min_num_tokens_alignment = is_nvfp4 ? TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4
                                                      : TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX;
    static_assert((min_num_tokens_alignment & (min_num_tokens_alignment - 1)) == 0,
                  "Min num tokens alignment must be a power of two");
    // Since we don't know a priori how much padding is needed we assume the max per expert
    // NOTE: we don't use (min_num_tokens_alignment-1) to be able to do power of two divisions
    int64_t num_padding_tokens = min_num_tokens_alignment * num_experts_per_node;

    for (int64_t padding_token = blockIdx.x; padding_token < num_padding_tokens; padding_token += gridDim.x) {
      int64_t expert = padding_token / min_num_tokens_alignment;
      int64_t num_tokens_before_expert = expert_first_token_offset[expert];
      int64_t num_tokens_after_expert = expert_first_token_offset[expert + 1];
      int64_t tokens_to_expert = num_tokens_after_expert - num_tokens_before_expert;
      int64_t padding_to_expert = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(tokens_to_expert, min_num_tokens_alignment) - tokens_to_expert;
      int64_t expert_pad_idx = padding_token % min_num_tokens_alignment;
      if (expert_pad_idx < padding_to_expert) {
        for (int64_t elem_index = start_offset; elem_index < padded_num_elems_in_col; elem_index += stride) {
          writeSF<VecSize, VecSize>(num_tokens_before_expert, expert, /*source_row*/ -1,
                                    num_tokens_after_expert + expert_pad_idx, elem_index, padded_hidden_size, fc1_act_sf_flat,
                                    /* input_sf */ nullptr);  // Pass nulltpr input_sf so we write 0
        }
      }
    }
  }
}

template <class InputActivationsType, class ExpandedActivationsType>
void expandInputRowsKernelLauncher(InputActivationsType const* unpermuted_input,
                                   ExpandedActivationsType* permuted_output, float const* unpermuted_scales, float* permuted_scales,
                                   int const* permuted_row_to_unpermuted_row, int64_t const num_rows, int64_t const hidden_size, int const k,
                                   int const num_experts_per_node, QuantParams const& quant_params, bool use_per_expert_act_scale,
                                   int64_t* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
                                   TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, void const* prequant_scales, cudaStream_t stream) {
#ifdef ENABLE_FP4
  ORT_ENFORCE(
      (std::is_same_v<ExpandedActivationsType, __nv_fp4_e2m1> && fc1_act_sf_flat) || !use_per_expert_act_scale,
      "Per-expert act scale for FC1 is only supported for NVFP4 activations");
  constexpr int64_t min_num_tokens_alignment = std::is_same_v<ExpandedActivationsType, __nv_fp4_e2m1>
                                                   ? TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4
                                                   : TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX;
  int64_t num_padding_tokens = min_num_tokens_alignment * num_experts_per_node;
#else
  int64_t num_padding_tokens = 0;
#endif

  static int64_t const smCount = onnxruntime::llm::common::getMultiProcessorCount();
  // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
  int64_t const blocks = std::min(smCount * 8, std::max(num_rows * k, num_padding_tokens));
  int64_t const threads = EXPAND_THREADS_PER_BLOCK;

  auto func = [&]() {
#ifdef ENABLE_FP8
    // Always MXFP8
    if constexpr (std::is_same_v<ExpandedActivationsType, __nv_fp8_e4m3> && !std::is_same_v<InputActivationsType, __nv_fp8_e4m3>) {
      ORT_ENFORCE(quant_params.mxfp8_mxfp4.fc1.weight_block_scale || prequant_scales,
                  "MXFP8xMXFP4 block scaling or prequant_scales or prequant_scales parameters not provided");
      return prequant_scales ? &expandInputRowsKernel<InputActivationsType, ExpandedActivationsType,
                                                      TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE, true>
                             : &expandInputRowsKernel<InputActivationsType, ExpandedActivationsType,
                                                      TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX, false>;
    }
    // Could be either regular FP8 or MXFP8
    else if constexpr (std::is_same_v<ExpandedActivationsType, __nv_fp8_e4m3> && std::is_same_v<InputActivationsType, __nv_fp8_e4m3>) {
      ORT_ENFORCE(!prequant_scales, "NVFP4 is not supported for AWQ");
      return quant_params.mxfp8_mxfp4.fc1.weight_block_scale
                 ? &expandInputRowsKernel<InputActivationsType, ExpandedActivationsType,
                                          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX, false>
                 : &expandInputRowsKernel<InputActivationsType, ExpandedActivationsType,
                                          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE, false>;
    } else
#endif
#ifdef ENABLE_FP4
        if constexpr (std::is_same_v<ExpandedActivationsType, __nv_fp4_e2m1>) {
      ORT_ENFORCE(
          quant_params.fp4.fc1.weight_block_scale, "NVFP4 block scaling is expected for FP4xFP4");
      ORT_ENFORCE(!prequant_scales, "NVFP4 is not supported for AWQ");
      return &expandInputRowsKernel<InputActivationsType, ExpandedActivationsType,
                                    TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4, false>;
    } else
#endif
    {
      ORT_ENFORCE(!prequant_scales, "w4afp8 Prequant scales provided for non-FP8 data type");
      return &expandInputRowsKernel<InputActivationsType, ExpandedActivationsType,
                                    TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE, false>;
    }
  }();

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, func, unpermuted_input, permuted_output, unpermuted_scales, permuted_scales,
                     permuted_row_to_unpermuted_row, num_rows, hidden_size, k, quant_params.fp4.fc1.act_global_scale,
                     use_per_expert_act_scale, expert_first_token_offset, fc1_act_sf_flat, input_sf, num_experts_per_node,
                     reinterpret_cast<InputActivationsType const*>(prequant_scales));
}

#define INSTANTIATE_EXPAND_INPUT_ROWS(InputActivationsType, ExpandedActivationsType)                      \
  template void expandInputRowsKernelLauncher<InputActivationsType, ExpandedActivationsType>(             \
      InputActivationsType const* unpermuted_input, ExpandedActivationsType* permuted_output,             \
      float const* unpermuted_scales, float* permuted_scales, int const* permuted_row_to_unpermuted_row,  \
      int64_t const num_rows, int64_t const hidden_size, int const k, int const num_experts_per_node,     \
      QuantParams const& quant_params, bool use_per_expert_act_scale, int64_t* expert_first_token_offset, \
      TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,                                     \
      TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, void const* prequant_scales,         \
      cudaStream_t stream)

// Instantiate the data types that are used by the external pytorch op
INSTANTIATE_EXPAND_INPUT_ROWS(float, float);
INSTANTIATE_EXPAND_INPUT_ROWS(half, half);
#ifdef ENABLE_BF16
INSTANTIATE_EXPAND_INPUT_ROWS(__nv_bfloat16, __nv_bfloat16);
#endif

enum class ScaleMode : int {
  NO_SCALE = 0,
  DEFAULT = 1,
};

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE>
__global__ void finalizeMoeRoutingKernel(GemmOutputType const* expanded_permuted_rows,
                                         OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* scales,
                                         int const* unpermuted_row_to_permuted_row, int const* token_selected_experts, int64_t const orig_cols,
                                         int64_t const experts_per_token, int const num_experts_per_node, int const start_expert_id) {
  assert(orig_cols % 4 == 0);
  int64_t const original_row = blockIdx.x;
  int64_t const num_rows = gridDim.x;
  auto const offset = original_row * orig_cols;
  OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;

  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t FINALIZE_ELEM_PER_THREAD = 128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);

  int64_t const start_offset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
  int64_t const num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

  using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
  using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
  auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
  auto const* expanded_permuted_rows_v = reinterpret_cast<InputElem const*>(expanded_permuted_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll
  for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    ComputeElem thread_output;
    thread_output.fill(0);
    for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
      int64_t const k_offset = original_row * experts_per_token + k_idx;
      int64_t const expert_id = token_selected_experts[k_offset] - start_expert_id;
      if (expert_id < 0 || expert_id >= num_experts_per_node) {
        continue;
      }

      int64_t const expanded_original_row = original_row + k_idx * num_rows;
      int64_t const expanded_permuted_row = unpermuted_row_to_permuted_row[expanded_original_row];

      float const row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];

      auto const* expanded_permuted_rows_row_ptr = expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;

      ComputeElem expert_result = arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);
      if (bias) {
        auto const* bias_ptr = bias_v + expert_id * num_elems_in_col;
        expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
      }

      thread_output = thread_output + row_scale * expert_result;
    }

    OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
    reduced_row_ptr_v[elem_index] = output_elem;
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE>
__global__ void finalizeMoeRoutingNoFillingKernel(GemmOutputType const* expanded_permuted_rows,
                                                  OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* scales,
                                                  int const* const unpermuted_row_to_permuted_row, int const* permuted_row_to_unpermuted_row,
                                                  int const* token_selected_experts, int64_t const* expert_first_token_offset, int64_t const num_rows,
                                                  int64_t const orig_cols, int64_t const experts_per_token, int const num_experts_per_node, int const start_expert_id) {
  assert(orig_cols % 4 == 0);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];
  for (int64_t expanded_permuted_row = blockIdx.x; expanded_permuted_row < num_valid_tokens;
       expanded_permuted_row += gridDim.x) {
    int64_t unpermuted_row = permuted_row_to_unpermuted_row[expanded_permuted_row];

    // Duplicate and permute rows
    int64_t const source_k_rank = unpermuted_row / num_rows;
    int64_t const source_row = unpermuted_row % num_rows;

    // If the expert is the first selected (valid) one of the corresponding token on the current EP rank, do
    // reduction; otherwise, skip.
    bool is_first_selected_expert = true;
    for (int k_idx = 0; k_idx < source_k_rank; ++k_idx) {
      int const expert_id = token_selected_experts[source_row * experts_per_token + k_idx] - start_expert_id;
      if (expert_id >= 0 && expert_id < num_experts_per_node) {
        is_first_selected_expert = false;
        break;
      }
    }
    if (!is_first_selected_expert) {
      continue;
    }

    OutputType* reduced_row_ptr = reduced_unpermuted_output + source_row * orig_cols;

    // Load 128-bits per thread, according to the smallest data type we read/write
    constexpr int64_t FINALIZE_ELEM_PER_THREAD = 128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

    using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
    using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
    auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
    auto const* expanded_permuted_rows_v = reinterpret_cast<InputElem const*>(expanded_permuted_rows);
    auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

    for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      ComputeElem thread_output;
      thread_output.fill(0);
      for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
        int64_t const k_offset = source_row * experts_per_token + k_idx;
        int64_t const expert_id = token_selected_experts[k_offset] - start_expert_id;
        if (expert_id < 0 || expert_id >= num_experts_per_node) {
          continue;
        }

        int64_t const expanded_permuted_row_from_k_idx = unpermuted_row_to_permuted_row[source_row + k_idx * num_rows];

        float const row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];

        auto const* expanded_permuted_rows_row_ptr = expanded_permuted_rows_v + expanded_permuted_row_from_k_idx * num_elems_in_col;

        ComputeElem expert_result = arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);

        if (bias) {
          auto const* bias_ptr = bias_v + expert_id * num_elems_in_col;
          expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
        }

        thread_output = thread_output + row_scale * expert_result;
      }
      OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
      reduced_row_ptr_v[elem_index] = output_elem;
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingKernelLauncher(GemmOutputType const* expanded_permuted_rows,
                                      OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* final_scales,
                                      int const* unpermuted_row_to_permuted_row, int const* permuted_row_to_unpermuted_row,
                                      int const* token_selected_experts, int64_t const* expert_first_token_offset, int64_t const num_rows,
                                      int64_t const cols, int64_t const experts_per_token, int64_t const num_experts_per_node,
                                      MOEParallelismConfig parallelism_config, bool const enable_alltoall, cudaStream_t stream) {
  // Only add bias on rank 0 for tensor parallelism
  bool const is_rank_0 = parallelism_config.tp_rank == 0;
  ScaleBiasType const* bias_ptr = is_rank_0 ? bias : nullptr;
  int const start_expert_id = num_experts_per_node * parallelism_config.ep_rank;

  cudaLaunchConfig_t config;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;

  if (parallelism_config.ep_size > 1 && enable_alltoall) {
    // If all-to-all comm is enabled, finalizeMoeRouting doesn't need to fill the invalid output tokens with zeros.
    static int const smCount = onnxruntime::llm::common::getMultiProcessorCount();
    // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
    int64_t const blocks = smCount * 8;
    int64_t const threads = FINALIZE_THREADS_PER_BLOCK;
    config.gridDim = blocks;
    config.blockDim = threads;
    auto func = final_scales
                    ? &finalizeMoeRoutingNoFillingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT>
                    : &finalizeMoeRoutingNoFillingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE>;
    cudaLaunchKernelEx(&config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr, final_scales,
                       unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row, token_selected_experts,
                       expert_first_token_offset, num_rows, cols, experts_per_token, num_experts_per_node, start_expert_id);
  } else {
    // If all-gather reduce-scatter is used, finalizeMoeRouting must fill invalid output tokens with zeros.
    int64_t const blocks = num_rows;
    int64_t const threads = FINALIZE_THREADS_PER_BLOCK;
    config.gridDim = blocks;
    config.blockDim = threads;
    auto func = final_scales
                    ? &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT>
                    : &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE>;
    cudaLaunchKernelEx(&config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr, final_scales,
                       unpermuted_row_to_permuted_row, token_selected_experts, cols, experts_per_token, num_experts_per_node,
                       start_expert_id);
  }
}

#define INSTANTIATE_FINALIZE_MOE_ROUTING(OutputT, GemmOutputT, ScaleBiasT)                                          \
  template void finalizeMoeRoutingKernelLauncher<OutputT, GemmOutputT, ScaleBiasT>(                                 \
      GemmOutputT const* expanded_permuted_rows, OutputT* reduced_unpermuted_output, ScaleBiasT const* bias,        \
      float const* final_scales, int const* expanded_source_row_to_expanded_dest_row,                               \
      int const* expanded_dest_row_to_expanded_source_row, int const* expert_for_source_row,                        \
      int64_t const* expert_first_token_offset, int64_t const num_rows, int64_t const cols,                         \
      int64_t const experts_per_token, int64_t const num_experts_per_node, MOEParallelismConfig parallelism_config, \
      bool const enable_alltoall, cudaStream_t stream);

// Instantiate the data types that are used by the external pytorch op
INSTANTIATE_FINALIZE_MOE_ROUTING(half, half, half);
INSTANTIATE_FINALIZE_MOE_ROUTING(float, float, float);
#ifdef ENABLE_BF16
INSTANTIATE_FINALIZE_MOE_ROUTING(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
#endif

// ============================== Gated Activation =================================
constexpr static int ACTIVATION_THREADS_PER_BLOCK = 256;

template <class ActivationOutputType, class GemmOutputType, template <class> class ActFn>
__global__ void doGatedActivationKernel(ActivationOutputType* output, GemmOutputType const* gemm_result,
                                        int64_t const* num_valid_tokens_ptr, int64_t inter_size) {
  int64_t const tid = threadIdx.x;
  int64_t const token = blockIdx.x;
  if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr) {
    return;
  }

  output = output + token * inter_size;
  gemm_result = gemm_result + token * inter_size * 2;

  constexpr int64_t ACTIVATION_ELEM_PER_THREAD = 128 / sizeof_bits<ActivationOutputType>::value;

  using OutputElem = cutlass::Array<ActivationOutputType, ACTIVATION_ELEM_PER_THREAD>;
  using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
  auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result);
  auto output_vec = reinterpret_cast<OutputElem*>(output);
  int64_t const start_offset = tid;
  int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
  assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
  int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
  int64_t const inter_size_vec = inter_size / ACTIVATION_ELEM_PER_THREAD;

  ActFn<ComputeElem> fn{};
  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto fc1_value = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index]);
    // BF16 isn't supported, use FP32 for activation function
    auto gate_value = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index + inter_size_vec]);
    auto gate_act = fn(gate_value);
    output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(fc1_value * gate_act);
  }
}

template <typename ActivationOutputType, typename GemmOutputType>
void doGatedActivation(ActivationOutputType* output, GemmOutputType const* gemm_result,
                       int64_t const* num_valid_tokens_ptr, int64_t inter_size, int64_t num_tokens, ActivationType activation_type,
                       cudaStream_t stream) {
  int64_t const blocks = num_tokens;
  int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;

  auto* fn = activation_type == ActivationType::Swiglu
                 ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType, cutlass::epilogue::thread::SiLu>
                 : &doGatedActivationKernel<ActivationOutputType, GemmOutputType, cutlass::epilogue::thread::GELU>;
  fn<<<blocks, threads, 0, stream>>>(output, gemm_result, num_valid_tokens_ptr, inter_size);
}

// ============================== Activation =================================

template <class T, class GemmOutputType, class ScaleBiasType, template <class> class ActFn,
          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType BlockScalingType>
__global__ void doActivationKernel(T* output, GemmOutputType const* gemm_result, float const* fp8_quant,
                                   ScaleBiasType const* bias_ptr, bool bias_is_broadcast, int64_t const* expert_first_token_offset,
                                   int num_experts_per_node, int64_t inter_size, bool gated, float const* fc2_act_global_scale,
                                   bool use_per_expert_act_scale, TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_act_sf_flat) {
#ifdef ENABLE_FP4
  constexpr bool IsNVFP4 = std::is_same_v<T, __nv_fp4_e2m1> && BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4;
  constexpr bool IsMXFP8 = std::is_same_v<T, __nv_fp8_e4m3> && BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
#else
  constexpr bool IsNVFP4 = cute::dependent_false<T>;
  constexpr bool IsMXFP8 = cute::dependent_false<T>;
#endif

  int64_t const tid = threadIdx.x;
  size_t const gated_size_mul = gated ? 2 : 1;
  size_t const gated_off = gated ? inter_size : 0;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  constexpr int64_t VecSize = IsNVFP4 ? TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize
                                      : TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize;
  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t ACTIVATION_ELEM_PER_THREAD = (IsNVFP4 || IsMXFP8)
                                                     ? CVT_FP4_ELTS_PER_THREAD
                                                     : (128 / std::min(sizeof_bits<T>::value, sizeof_bits<GemmOutputType>::value));

  // This should be VecSize * 4 elements
  // We assume at least VecSize alignment or the quantization will fail
  int64_t const min_k_dim_alignment = IsNVFP4 ? TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4
                                              : TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX;
  int64_t const padded_inter_size = ceilDiv(inter_size, min_k_dim_alignment) * min_k_dim_alignment;

  int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];

  for (int64_t token = blockIdx.x; token < num_valid_tokens; token += gridDim.x) {
    size_t gemm_result_offset = token * inter_size * gated_size_mul;
    size_t output_offset = token * inter_size;

    int64_t expert = 0;
    if (bias_ptr || IsNVFP4 || IsMXFP8 || use_per_expert_act_scale) {
      // TODO this is almost certainly faster as a linear scan
      expert = findTotalEltsLessThanTarget(expert_first_token_offset, num_experts_per_node, token + 1) - 1;
    }

    size_t act_scale_idx = use_per_expert_act_scale ? expert : 0;
    float const quant_scale = fp8_quant ? fp8_quant[act_scale_idx] : 1.f;

    // Some globals for FP4
    float global_scale_val = fc2_act_global_scale ? fc2_act_global_scale[act_scale_idx] : 1.0f;
    int64_t num_tokens_before_expert = (IsNVFP4 || IsMXFP8) ? expert_first_token_offset[expert] : 0;

    size_t bias_offset = 0;
    if (bias_ptr) {
      bias_offset = (bias_is_broadcast ? expert * inter_size * gated_size_mul : gemm_result_offset);
    }

    using BiasElem = cutlass::Array<ScaleBiasType, ACTIVATION_ELEM_PER_THREAD>;
    using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
    using OutputElem = std::conditional_t<IsNVFP4, uint32_t,
                                          std::conditional_t<IsMXFP8, uint64_t, cutlass::Array<T, ACTIVATION_ELEM_PER_THREAD>>>;
    using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
    // Aliases gemm_result for non-gated, non-fp8 cases
    auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result + gemm_result_offset);
    auto output_vec = reinterpret_cast<OutputElem*>(safe_inc_ptr(output, output_offset));
    auto bias_ptr_vec = reinterpret_cast<BiasElem const*>(bias_ptr + bias_offset);
    int64_t const start_offset = tid;
    int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
    assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
    assert(gated_off % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const gated_off_vec = gated_off / ACTIVATION_ELEM_PER_THREAD;

    ActFn<ComputeElem> fn{};
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      auto fc1_value = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index + gated_off_vec]);
      if (bias_ptr) {
        fc1_value = fc1_value + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index + gated_off_vec]);
      }

      auto gate_act = fn(fc1_value);

      if (gated) {
        auto gate_mul = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index]);
        if (bias_ptr_vec) {
          gate_mul = gate_mul + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index]);
        }
        gate_act = gate_act * gate_mul;
      }

      auto post_act_val = gate_act * quant_scale;

      if constexpr (IsNVFP4 || IsMXFP8) {
        // We use GemmOutputType as the intermediate compute type as that should always be unquantized
        auto res = quantizePackedFPXValue<GemmOutputType, T, ComputeElem, VecSize>(post_act_val,
                                                                                   global_scale_val, num_tokens_before_expert, expert, token, elem_index, inter_size, fc2_act_sf_flat,
                                                                                   IsNVFP4 ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                                                                                           : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);
        static_assert(
            sizeof(res) == sizeof(*output_vec), "Quantized value must be the same size as the output");
        output_vec[elem_index] = res;
      } else {
        output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(post_act_val);
      }
    }

    // Pad zeros in the extra SFs along the K dimension, we do this to ensure there are no nan values in the padded
    // SF atom
    if constexpr (IsNVFP4 || IsMXFP8) {
      // Use VecSize per thread since we are just writing out zeros so every thread can process a whole vector
      size_t padding_start_offset = inter_size / VecSize + start_offset;
      size_t padding_elems_in_col = padded_inter_size / VecSize;
      for (int64_t elem_index = padding_start_offset; elem_index < padding_elems_in_col; elem_index += stride) {
        writeSF<VecSize, VecSize>(num_tokens_before_expert, expert, /*source_row*/ -1, token, elem_index,
                                  padded_inter_size, fc2_act_sf_flat, /* input_sf */ nullptr);  // Pass nulltpr input_sf so we write 0
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  // Pad zeros in the extra SFs along the N dimension, we do this to ensure there are no nan values in the padded SF
  // atom
  if constexpr (IsNVFP4 || IsMXFP8) {
    int64_t const start_offset = threadIdx.x;
    int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
    // Use VecSize per thread since we are just writing out zeros so every thread can process a whole vector
    int64_t const padded_num_elems_in_col = padded_inter_size / VecSize;
    assert(padded_inter_size % VecSize == 0);

    constexpr int64_t min_num_tokens_alignment = IsNVFP4
                                                     ? TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4
                                                     : TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX;
    static_assert((min_num_tokens_alignment & (min_num_tokens_alignment - 1)) == 0,
                  "Min num tokens alignment must be a power of two");
    // Since we don't know a priori how much padding is needed we assume the max per expert
    // NOTE: we don't (min_num_tokens_alignment-1) to have power of two divisions
    int64_t num_padding_tokens = min_num_tokens_alignment * num_experts_per_node;

    for (int64_t padding_token = blockIdx.x; padding_token < num_padding_tokens; padding_token += gridDim.x) {
      int64_t expert = padding_token / min_num_tokens_alignment;
      int64_t num_tokens_before_expert = expert_first_token_offset[expert];
      int64_t num_tokens_after_expert = expert_first_token_offset[expert + 1];
      int64_t tokens_to_expert = num_tokens_after_expert - num_tokens_before_expert;
      int64_t padding_to_expert = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(tokens_to_expert, min_num_tokens_alignment) - tokens_to_expert;
      int64_t expert_pad_idx = padding_token % min_num_tokens_alignment;
      if (expert_pad_idx < padding_to_expert) {
        for (int64_t elem_index = start_offset; elem_index < padded_num_elems_in_col; elem_index += stride) {
          // The SF buffer is padded to a multiple of MinNDimAlignment for each expert
          // This means we can safely write to offset num_tokens_after_expert + padded_token, since the next
          // expert will leave space for the padding
          writeSF<VecSize, VecSize>(num_tokens_before_expert, expert, /*source_row*/ -1,
                                    num_tokens_after_expert + expert_pad_idx, elem_index, padded_inter_size, fc2_act_sf_flat,
                                    /* input_sf */ nullptr);  // Pass nulltpr input_sf so we write 0
        }
      }
    }
  }
}

template <class T, class GemmOutputType, class ScaleBiasType>
void doActivation(T* output, GemmOutputType const* gemm_result, float const* fp8_quant, ScaleBiasType const* bias,
                  bool bias_is_broadcast, int64_t const* expert_first_token_offset, int num_experts_per_node, int64_t inter_size,
                  int64_t expanded_num_tokens, ActivationType activation_type, QuantParams const& quant_params,
                  bool use_per_expert_act_scale, TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_act_sf_flat, cudaStream_t stream) {
#ifdef ENABLE_FP4
  constexpr int64_t min_num_tokens_alignment = std::is_same_v<T, __nv_fp4_e2m1>
                                                   ? TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4
                                                   : TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX;
  int64_t num_padding_tokens = min_num_tokens_alignment * num_experts_per_node;
#else
  int64_t num_padding_tokens = 0;
#endif

  static int64_t const smCount = onnxruntime::llm::common::getMultiProcessorCount();
  // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
  int64_t const blocks = std::min(smCount * 8, std::max(expanded_num_tokens, num_padding_tokens));
  int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;

  auto fn = [&]() {
    auto fn = [&](auto block_scaling_type) {
      auto fn_list = std::array{
          &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::GELU,
                              decltype(block_scaling_type)::value>,  // Gelu
          &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::ReLu,
                              decltype(block_scaling_type)::value>,  // Relu
          &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::SiLu,
                              decltype(block_scaling_type)::value>,  // Silu
          &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::SiLu,
                              decltype(block_scaling_type)::value>,  // Swiglu
          &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::GELU,
                              decltype(block_scaling_type)::value>,  // Geglu
          &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::Identity,
                              decltype(block_scaling_type)::value>  // Identity
      };
      return fn_list[static_cast<int>(activation_type)];
    };
    auto NVFP4 = onnxruntime::llm::common::ConstExprWrapper<TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType,
                                                            TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4>{};
    auto MXFPX = onnxruntime::llm::common::ConstExprWrapper<TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType,
                                                            TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX>{};
    auto NONE = onnxruntime::llm::common::ConstExprWrapper<TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType,
                                                           TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE>{};
#ifdef ENABLE_FP4
    if constexpr (std::is_same_v<T, __nv_fp4_e2m1>) {
      ORT_ENFORCE(
          quant_params.fp4.fc2.weight_block_scale, "NVFP4 block scaling is expected for FP4xFP4");
      return fn(NVFP4);
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      return quant_params.mxfp8_mxfp4.fc2.weight_block_scale ? fn(MXFPX) : fn(NONE);
    } else
#endif
    {
      return fn(NONE);
    }
  }();

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, fn, output, gemm_result, fp8_quant, bias, bias_is_broadcast, expert_first_token_offset,
                     num_experts_per_node, inter_size, isGatedActivation(activation_type), quant_params.fp4.fc2.act_global_scale,
                     use_per_expert_act_scale, fc2_act_sf_flat);
}

// ============================== Lora Add Bias =================================
constexpr static int LORA_KERNELS_THREADS_PER_BLOCK = 256;

template <class ScaleBiasType, class LoraType, bool IsGated>
__global__ void loraAddBiasKernel(ScaleBiasType* output, LoraType const* lora_result, ScaleBiasType const* bias,
                                  int64_t const* num_valid_tokens_ptr, int* permuted_token_selected_experts, int64_t inter_size) {
  int64_t const tid = threadIdx.x;
  int64_t const token = blockIdx.x;
  int64_t const num_tokens = gridDim.x;
  if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr) {
    return;
  }

  LoraType const* lora_result_1 = lora_result + token * inter_size;
  int expert_id = permuted_token_selected_experts[token];
  if constexpr (IsGated) {
    output = output + token * inter_size * 2;
    bias = bias + expert_id * inter_size * 2;
  } else {
    output = output + token * inter_size;
    bias = bias + expert_id * inter_size;
  }

  constexpr int64_t LORA_ADD_BIAS_ELEM_PER_THREAD = 128 / sizeof_bits<LoraType>::value;

  using DataElem = cutlass::Array<LoraType, LORA_ADD_BIAS_ELEM_PER_THREAD>;
  using BiasElem = cutlass::Array<ScaleBiasType, LORA_ADD_BIAS_ELEM_PER_THREAD>;
  auto lora_result_1_vec = reinterpret_cast<DataElem const*>(lora_result_1);
  auto bias_vec = reinterpret_cast<BiasElem const*>(bias);
  auto output_vec = reinterpret_cast<BiasElem*>(output);

  int64_t const start_offset = tid;
  int64_t const stride = LORA_KERNELS_THREADS_PER_BLOCK;
  assert(inter_size % LORA_ADD_BIAS_ELEM_PER_THREAD == 0);
  int64_t const num_elems_in_col = inter_size / LORA_ADD_BIAS_ELEM_PER_THREAD;

  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto lora_value = lora_result_1_vec[elem_index];
    auto bias_value = bias_vec[elem_index];
    output_vec[elem_index] = bias_value + arrayConvert<DataElem, BiasElem>(lora_value);
  }

  if constexpr (IsGated) {
    auto lora_result_2_vec = reinterpret_cast<DataElem const*>(lora_result_1 + num_tokens * inter_size);
    int64_t const inter_size_vec = inter_size / LORA_ADD_BIAS_ELEM_PER_THREAD;
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      auto lora_value = lora_result_2_vec[elem_index];
      auto bias_value = bias_vec[elem_index + inter_size_vec];
      output_vec[elem_index + inter_size_vec] = bias_value + arrayConvert<DataElem, BiasElem>(lora_value);
    }
  }
}

template <class ScaleBiasType, class LoraType>
void loraAddBias(ScaleBiasType* output, LoraType const* lora_result, ScaleBiasType const* bias,
                 int64_t const* num_valid_tokens_ptr, int64_t inter_size, int* permuted_token_selected_experts, int64_t num_tokens,
                 bool is_gated_activation, cudaStream_t stream) {
  int64_t const blocks = num_tokens;
  int64_t const threads = LORA_KERNELS_THREADS_PER_BLOCK;

  auto selected_fn = is_gated_activation ? loraAddBiasKernel<ScaleBiasType, LoraType, true>
                                         : loraAddBiasKernel<ScaleBiasType, LoraType, false>;
  selected_fn<<<blocks, threads, 0, stream>>>(
      output, lora_result, bias, num_valid_tokens_ptr, permuted_token_selected_experts, inter_size);
}

template <class T>
__global__ void loraReorderKernel(
    T* output, T const* lora_result, int64_t const* num_valid_tokens_ptr, int64_t inter_size) {
  int64_t const tid = threadIdx.x;
  int64_t const token = blockIdx.x;
  int64_t const num_tokens = gridDim.x;
  if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr) {
    return;
  }

  T const* lora_result_1 = lora_result + token * inter_size;
  output = output + token * inter_size * 2;

  constexpr int64_t LORA_REORDER_ELEM_PER_THREAD = 128 / sizeof_bits<T>::value;

  using DataElem = cutlass::Array<T, LORA_REORDER_ELEM_PER_THREAD>;
  auto lora_result_1_vec = reinterpret_cast<DataElem const*>(lora_result_1);
  auto output_vec = reinterpret_cast<DataElem*>(output);

  int64_t const start_offset = tid;
  int64_t const stride = LORA_KERNELS_THREADS_PER_BLOCK;
  assert(inter_size % LORA_REORDER_ELEM_PER_THREAD == 0);
  int64_t const num_elems_in_col = inter_size / LORA_REORDER_ELEM_PER_THREAD;

  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto lora_value = lora_result_1_vec[elem_index];
    output_vec[elem_index] = lora_value;
  }

  auto lora_result_2_vec = reinterpret_cast<DataElem const*>(lora_result_1 + num_tokens * inter_size);
  int64_t const inter_size_vec = inter_size / LORA_REORDER_ELEM_PER_THREAD;
  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto lora_value = lora_result_2_vec[elem_index];
    output_vec[elem_index + inter_size_vec] = lora_value;
  }
}

template <class T>
void loraReorder(T* output, T const* lora_result, int64_t const* num_valid_tokens_ptr, int64_t inter_size,
                 int64_t num_tokens, cudaStream_t stream) {
  int64_t const blocks = num_tokens;
  int64_t const threads = LORA_KERNELS_THREADS_PER_BLOCK;

  loraReorderKernel<T><<<blocks, threads, 0, stream>>>(output, lora_result, num_valid_tokens_ptr, inter_size);
}

// ============================== DEQUANT_FP8 =================================
constexpr static int DEQUANT_KERNELS_THREADS_PER_BLOCK = 256;

template <class OutputType, class InputType>
__global__ void dequantFP8Kernel(OutputType* output, InputType const* input, int64_t const* num_valid_tokens_ptr,
                                 int64_t inter_size, float const* scale, bool scale_is_dequant) {
  int64_t const tid = threadIdx.x;
  int64_t const token = blockIdx.x;
  if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr) {
    return;
  }

  output = output + token * inter_size;
  input = input + token * inter_size;

  constexpr int64_t DEQUANT_ELEM_PER_THREAD = 128 / sizeof_bits<InputType>::value;

  using DataElem = cutlass::Array<InputType, DEQUANT_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, DEQUANT_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, DEQUANT_ELEM_PER_THREAD>;
  auto input_vec = reinterpret_cast<DataElem const*>(input);
  auto output_vec = reinterpret_cast<OutputElem*>(output);

  int64_t const start_offset = tid;
  int64_t const stride = DEQUANT_KERNELS_THREADS_PER_BLOCK;
  assert(inter_size % DEQUANT_ELEM_PER_THREAD == 0);
  int64_t const num_elems_in_col = inter_size / DEQUANT_ELEM_PER_THREAD;

  ComputeElem deqaunt_scale_value;
  float dequant_scale = scale[0];
  if (!scale_is_dequant) {
    dequant_scale = 1.f / dequant_scale;
  }
  deqaunt_scale_value.fill(dequant_scale);

  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto input_value = arrayConvert<DataElem, ComputeElem>(input_vec[elem_index]);
    output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(input_value * deqaunt_scale_value);
  }
}

template <class OutputType, class InputType>
void dequantFP8(OutputType* output, InputType const* input, int64_t const* num_valid_tokens_ptr, int64_t inter_size,
                int64_t num_tokens, float const* scale, bool scale_is_dequant, cudaStream_t stream) {
  int64_t const blocks = num_tokens;
  int64_t const threads = DEQUANT_KERNELS_THREADS_PER_BLOCK;

  dequantFP8Kernel<OutputType, InputType>
      <<<blocks, threads, 0, stream>>>(output, input, num_valid_tokens_ptr, inter_size, scale, scale_is_dequant);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::CutlassMoeFCRunner(
    int sm_version,
    ActivationType activation_type,
    bool has_fc3,
    bool normalize_routing_weights,
    bool use_sparse_mixer)
    : sm_(sm_version),
      activation_type_(activation_type),
      has_fc3_(has_fc3),
      normalize_routing_weights_(normalize_routing_weights),
      use_sparse_mixer_(use_sparse_mixer),
      blockscale_gemm_runner_{std::make_unique<kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>>()} {
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
std::map<std::string, std::pair<size_t, size_t>>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::getWorkspaceDeviceBufferSizes(
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
    int const experts_per_token, ActivationType activation_type, bool use_lora, bool use_deepseek_fp8_block_scale,
    bool min_latency_mode, bool use_awq) {
  size_t num_moe_inputs = min_latency_mode ? num_experts_per_node * num_rows : experts_per_token * num_rows;
  size_t const permuted_elems = num_moe_inputs * hidden_size;
  size_t const interbuf_elems = num_moe_inputs * inter_size;
  size_t glu_inter_elems = 0;
  bool is_gated_activation = isGatedActivation(activation_type);
  if (is_gated_activation) {
    glu_inter_elems = interbuf_elems * 2;
  } else if (mayHaveDifferentGEMMOutputType()) {
    // In this case we are using activation quantization, and some intermediate buffers will be unquantized
    // We need to have separate memory for these as we can no longer alias the output buffer for reuse
    glu_inter_elems = interbuf_elems;
  }

  bool using_tma_ws = moe_gemm_runner_.supportsTmaWarpSpecialized();

  size_t const gemm_output_dtype = sizeof(UnfusedGemmOutputType);

  constexpr float dtype_size = act_fp4 ? 0.5f : (use_w4afp8 ? 2.0f : sizeof(T));

  size_t const permuted_row_to_unpermuted_row_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
  size_t const permuted_token_selected_experts_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);

  int64_t const num_tokens_per_block = computeNumTokensPerBlock(num_rows, num_experts_per_node);
  int64_t const num_blocks_per_seq = onnxruntime::llm::common::ceilDiv(num_rows, num_tokens_per_block);
  size_t const blocked_expert_counts_size = min_latency_mode ? 0 : num_experts_per_node * num_blocks_per_seq * sizeof(int);
  size_t const blocked_expert_counts_cumsum_size = blocked_expert_counts_size;
  size_t const blocked_row_to_unpermuted_row_size = min_latency_mode ? 0 : num_experts_per_node * num_rows * sizeof(int);

  size_t const permuted_data_size = permuted_elems * dtype_size;
  size_t const expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t);
  size_t const permuted_token_final_scales_size = mayHaveFinalizeFused() ? num_moe_inputs * sizeof(float) : 0;
  size_t const glu_inter_size = glu_inter_elems * gemm_output_dtype;  // May be an intermediate type for quantization
  size_t const fc1_result_size = interbuf_elems * dtype_size;         // Activation quantizes so back to dtype_size
  size_t const fc2_result_size = min_latency_mode
                                     ? 0
                                     : num_moe_inputs * hidden_size * gemm_output_dtype;  // May be an intermediate type for quantization

  // If topk is greater than num_experts_per_node (i.e. large EP value), then we don't need to allocate for the whole
  // tokens*topk
  auto act_sf_rows = min_latency_mode
                         ? num_moe_inputs
                         : std::min(num_moe_inputs, static_cast<size_t>(num_rows * num_experts_per_node));
  size_t const sf_size = getScalingType() == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX
                             ? sizeof(TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF)
                             : sizeof(TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF);

  size_t const fc1_fp4_act_scale_size = getOffsetActivationSF(num_experts_per_node, act_sf_rows, hidden_size, getScalingType()) * sf_size;
  size_t const fc2_fp4_act_scale_size = getOffsetActivationSF(num_experts_per_node, act_sf_rows, inter_size, getScalingType()) * sf_size;
  size_t const fp4_act_scale_size = std::max(fc1_fp4_act_scale_size, fc2_fp4_act_scale_size);

  size_t const tma_ws_size = using_tma_ws ? TmaWarpSpecializedGroupedGemmInput::workspaceSize(num_experts_per_node, getScalingType()) : 0;

  size_t const gemm_workspace_size = moe_gemm_runner_.getMaxWorkspaceSize(num_experts_per_node);

  // lora related
  size_t const lora_input_size = (use_lora && use_fp8) ? std::max(permuted_elems, interbuf_elems) * sizeof(ScaleBiasType) : 0;
  size_t const lora_fc1_result_size = use_lora
                                          ? (is_gated_activation ? 2 * interbuf_elems * sizeof(ScaleBiasType) : interbuf_elems * sizeof(ScaleBiasType))
                                          : 0;
  size_t const lora_add_bias_size = use_lora ? lora_fc1_result_size : 0;
  size_t const lora_fc2_result_size = use_lora ? permuted_elems * sizeof(ScaleBiasType) : 0;

  // We do some overlapping of the large workspace buffers. Although we could overlap some of the other buffers, they
  // are small enough (i.e no factor of hidden size) they will only be a couple MiB at most, so we don't bother
  // in the case of fused activation we overlap permuted_data and fc2_result
  // in the case of unfused activation we overlap permuted_data and fc1_result
  // we need to calculate the max possible size, so use the max of all three
  size_t overlapped_gemm1_gemm2_inputs_size = std::max(permuted_data_size, fc2_result_size);
  // When glu_inter_elems is 0 we are always fused, otherwise we may need the un-fused case
  if (glu_inter_elems > 0) {
    overlapped_gemm1_gemm2_inputs_size = std::max(overlapped_gemm1_gemm2_inputs_size, fc1_result_size);
  }

  size_t const alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float*);

  // if we have glu_inter we overlap it with fc2_result, otherwise we use fc1_result by itself
  size_t overlapped_gemm1_gemm2_outputs_size = fc1_result_size;
  if (glu_inter_elems > 0) {
    overlapped_gemm1_gemm2_outputs_size = std::max(std::max(glu_inter_size, fc2_result_size), overlapped_gemm1_gemm2_outputs_size);
  }

  size_t smoothed_act_size = use_awq ? std::max(permuted_elems, interbuf_elems) * sizeof(T) * 2
                                     : 0;  // Extra workspace required by AWQ for smoothing activations
  size_t deepseek_fc_workspace_size = 0;
  if (use_deepseek_fp8_block_scale) {
    size_t factor = is_gated_activation ? 2 : 1;
    size_t blockscale_fc1_output_size = factor * interbuf_elems * gemm_output_dtype;
    size_t blockscale_fc2_output_size = permuted_elems * gemm_output_dtype;
    overlapped_gemm1_gemm2_inputs_size = std::max(std::max(permuted_data_size, fc1_result_size), blockscale_fc2_output_size);
    overlapped_gemm1_gemm2_outputs_size = blockscale_fc1_output_size;

    auto* blockscale_gemm_runner = getDeepSeekBlockScaleGemmRunner();
    ORT_ENFORCE(blockscale_gemm_runner != nullptr);
    auto deepseek_fc1_workspace_size = blockscale_gemm_runner->getWorkspaceSize(
        num_rows, factor * inter_size, hidden_size, experts_per_token, num_experts_per_node);
    auto deepseek_fc2_workspace_size = blockscale_gemm_runner->getWorkspaceSize(
        num_rows, hidden_size, inter_size, experts_per_token, num_experts_per_node);
    deepseek_fc_workspace_size = std::max(deepseek_fc1_workspace_size, deepseek_fc2_workspace_size);
  }

  size_t map_offset = 0;
  std::map<std::string, std::pair<size_t, size_t>> out_map;

#define ADD_NAME(name, size)                                                                                \
  do {                                                                                                      \
    auto aligned_size = onnxruntime::llm::common::alignSize(size, onnxruntime::llm::common::kCudaMemAlign); \
    out_map[#name] = std::pair{aligned_size, map_offset};                                                   \
    map_offset += aligned_size;                                                                             \
  } while (false)
#define ADD(name) ADD_NAME(name, name##_size)

  ADD(permuted_row_to_unpermuted_row);
  ADD(permuted_token_selected_experts);
  ADD(blocked_expert_counts);
  ADD(blocked_expert_counts_cumsum);
  ADD(blocked_row_to_unpermuted_row);
  ADD(expert_first_token_offset);
  ADD(permuted_token_final_scales);
  ADD(overlapped_gemm1_gemm2_inputs);
  ADD(overlapped_gemm1_gemm2_outputs);
  ADD_NAME(alpha_scale_ptr_array_fc1, alpha_scale_ptr_array_size);
  ADD_NAME(alpha_scale_ptr_array_fc2, alpha_scale_ptr_array_size);
  ADD(fp4_act_scale);
  ADD_NAME(tma_ws_gemm1_workspace, tma_ws_size);
  ADD_NAME(tma_ws_gemm2_workspace, tma_ws_size);
  ADD(gemm_workspace);
  ADD(lora_input);
  ADD(lora_fc1_result);
  ADD(lora_add_bias);
  ADD(lora_fc2_result);
  ADD(deepseek_fc_workspace);
  ADD(smoothed_act);

  return out_map;

#undef ADD_NAME
#undef ADD
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
size_t CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::getWorkspaceSize(
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts,
    int const experts_per_token, ActivationType activation_type, MOEParallelismConfig parallelism_config, bool use_lora,
    bool use_deepseek_fp8_block_scale, bool min_latency_mode, bool use_awq) {
  int const ep_size = parallelism_config.ep_size;
  ORT_ENFORCE(num_experts % ep_size == 0, "Number of experts must be a multiple of ep size");
  auto sizes_map = getWorkspaceDeviceBufferSizes(num_rows, hidden_size, inter_size, num_experts / ep_size,
                                                 experts_per_token, activation_type, use_lora, use_deepseek_fp8_block_scale, min_latency_mode, use_awq);
  std::vector<size_t> sizes(sizes_map.size());
  std::transform(sizes_map.begin(), sizes_map.end(), sizes.begin(), [](auto& v) { return v.second.first; });
  size_t size = onnxruntime::llm::common::calculateTotalWorkspaceSize(sizes.data(), sizes.size());
  ORT_LLM_LOG_TRACE(onnxruntime::MakeString("Mixture Of Experts Plugin requires workspace of ", size / 1024.f / 1024.f, " MiB"));
  return size;
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::configureWsPtrs(char* ws_ptr,
                                                                                                     int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
                                                                                                     int const experts_per_token, ActivationType activation_type, MOEParallelismConfig parallelism_config, bool use_lora,
                                                                                                     bool use_deepseek_fp8_block_scale, bool min_latency_mode, bool use_awq) {
  auto workspaces = getWorkspaceDeviceBufferSizes(num_rows, hidden_size, inter_size, num_experts_per_node,
                                                  experts_per_token, activation_type, use_lora, use_deepseek_fp8_block_scale, min_latency_mode, use_awq);

  auto getWsPtr = [&](auto type, std::string const& name) {
    return workspaces.at(name).first ? reinterpret_cast<decltype(type)*>(ws_ptr + workspaces.at(name).second)
                                     : nullptr;
  };

  permuted_row_to_unpermuted_row_ = getWsPtr(int{}, "permuted_row_to_unpermuted_row");
  permuted_token_selected_experts_ = getWsPtr(int{}, "permuted_token_selected_experts");
  blocked_expert_counts_ = getWsPtr(int{}, "blocked_expert_counts");
  blocked_expert_counts_cumsum_ = getWsPtr(int{}, "blocked_expert_counts_cumsum");
  blocked_row_to_unpermuted_row_ = getWsPtr(int{}, "blocked_row_to_unpermuted_row");

  expert_first_token_offset_ = getWsPtr(int64_t{}, "expert_first_token_offset");

  // We check if the provided config uses fused finalize and disable it if it does not
  bool const gemm2_using_tma_ws = moe_gemm_runner_.isTmaWarpSpecialized(*gemm2_config_);
  permuted_token_final_scales_ = (gemm2_using_tma_ws && mayHaveFinalizeFused()) ? getWsPtr(float{}, "permuted_token_final_scales") : nullptr;

  bool const is_gated_activation = isGatedActivation(activation_type);
  bool const gemm1_using_fused_moe = moe_gemm_runner_.isFusedGatedActivation(*gemm1_config_, is_gated_activation, inter_size, hidden_size);
  bool const gemm1_using_tma_ws = moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_);
  bool const tma_ws_has_glu = gemm1_using_tma_ws && (mayHaveDifferentGEMMOutputType() || is_gated_activation);
  // We always use fused path if we can
  bool const non_tma_ws_has_glu = !gemm1_using_fused_moe && is_gated_activation;
  bool const has_glu_inter_result = tma_ws_has_glu || non_tma_ws_has_glu || use_fp8;

  // Always same value, but overlapped with either fc1_result_ or fc2_result_
  permuted_data_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
  // Always same value, ignored if not needed
  glu_inter_result_ = has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs") : nullptr;

  // fc1 and fc2 alias one of the above pointers, but it depends on if actfn is fused/unfused which is overlapped
  // NOTE: It is important to get the overlapped pointers correct as the wrong order will cause the buffer to be used
  // as an input and output for the same gemm, which will cause corruption
  fc1_result_ = has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs")
                                     : getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs");
  fc2_result_ = has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs")
                                     : getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");

  if (use_deepseek_fp8_block_scale) {
    permuted_data_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
    fc1_result_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
    glu_inter_result_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs");
    fc2_result_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
  }

  alpha_scale_ptr_array_fc1_ = getWsPtr((float const*)(nullptr), "alpha_scale_ptr_array_fc1");
  alpha_scale_ptr_array_fc2_ = getWsPtr((float const*)(nullptr), "alpha_scale_ptr_array_fc2");

  // NOTE: We alias these, but if we fuse the quantization for GEMM2 into GEMM1 they will need separated
  fc1_fp4_act_scale_ = nullptr;
  fc2_fp4_act_scale_ = nullptr;
  if (use_block_scaling) {
    fc1_fp4_act_scale_ = getWsPtr(TmaWarpSpecializedGroupedGemmInput::ElementSF{}, "fp4_act_scale");
    fc2_fp4_act_scale_ = getWsPtr(TmaWarpSpecializedGroupedGemmInput::ElementSF{}, "fp4_act_scale");
    ORT_ENFORCE(fc1_fp4_act_scale_ != nullptr);
    ORT_ENFORCE(fc2_fp4_act_scale_ != nullptr);
  }

  tma_ws_grouped_gemm1_input_ = {};
  tma_ws_grouped_gemm2_input_ = {};
  if (moe_gemm_runner_.supportsTmaWarpSpecialized()) {
    tma_ws_grouped_gemm1_input_.configureWorkspace(getWsPtr(int8_t{}, "tma_ws_gemm1_workspace"),
                                                   num_experts_per_node, getWsPtr(int8_t{}, "gemm_workspace"), workspaces.at("gemm_workspace").first,
                                                   getScalingType());
    tma_ws_grouped_gemm2_input_.configureWorkspace(getWsPtr(int8_t{}, "tma_ws_gemm2_workspace"),
                                                   num_experts_per_node, getWsPtr(int8_t{}, "gemm_workspace"), workspaces.at("gemm_workspace").first,
                                                   getScalingType());
  }

  lora_fc1_result_ = {};
  lora_add_bias_ = {};
  lora_fc2_result_ = {};

  if (use_lora) {
    lora_input_ = getWsPtr(ScaleBiasType{}, "lora_input");
    lora_fc1_result_ = getWsPtr(ScaleBiasType{}, "lora_fc1_result");
    lora_add_bias_ = getWsPtr(ScaleBiasType{}, "lora_add_bias");
    lora_fc2_result_ = getWsPtr(ScaleBiasType{}, "lora_fc2_result");
    ORT_ENFORCE(!use_fp8 || lora_input_ != nullptr, "LoRA input must not be nullptr if FP8 is enabled");
    ORT_ENFORCE(lora_fc1_result_ != nullptr);
    ORT_ENFORCE(lora_add_bias_ != nullptr);
    ORT_ENFORCE(lora_fc2_result_ != nullptr);
  }

  if (use_deepseek_fp8_block_scale) {
    auto* blockscale_gemm_runner = getDeepSeekBlockScaleGemmRunner();
    ORT_ENFORCE(blockscale_gemm_runner != nullptr);
    blockscale_gemm_runner->configureWorkspace(getWsPtr(char{}, "deepseek_fc_workspace"));
  }

  if (use_awq) {
    smoothed_act_ = getWsPtr(int8_t{}, "smoothed_act");
  }
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunnerInterface*
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::getDeepSeekBlockScaleGemmRunner() const {
  if (!(std::is_same_v<T, __nv_bfloat16> && std::is_same_v<OutputType, __nv_bfloat16>)) {
    ORT_THROW("Block scale GEMM runner only supports BF16 A/output");
  }

  if (!std::is_same_v<WeightType, __nv_fp8_e4m3>) {
    ORT_THROW("Block scale GEMM runner only supports FP8 weights.");
  }

  return blockscale_gemm_runner_.get();
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::BlockScaleFC1(
    DeepSeekBlockScaleGemmRunner& gemm_runner, T const* const input, T* const output, void* const gemm_output,
    int64_t const* const expert_first_token_offset, WeightType const* const fc1_expert_weights,
    ScaleBiasType const* const fc1_expert_biases, float const* const fc2_fp8_quant, int64_t const num_rows,
    int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const num_experts_per_node, ActivationType fc1_activation_type, QuantParams& quant_params, cudaStream_t stream) {
  bool const is_gated_activation = isGatedActivation(fc1_activation_type);

  int shape_n = is_gated_activation ? inter_size * 2 : inter_size;
  int shape_k = hidden_size;

  // NOTE: we assume gemm_runner.configureWorkspace has already been called.
  gemm_runner.moeGemm(gemm_output, input, fc1_expert_weights, expert_first_token_offset, num_experts_per_node,
                      shape_n, shape_k, stream, nullptr, quant_params.fp8_block_scaling.fc1_scales_ptrs);

  sync_check_cuda_error(stream);
  constexpr bool bias_is_broadcast = true;
  constexpr bool use_per_expert_act_scale = false;
  doActivation<T, UnfusedGemmOutputType>(output, static_cast<UnfusedGemmOutputType const*>(gemm_output),
                                         fc2_fp8_quant, fc1_expert_biases, bias_is_broadcast, expert_first_token_offset, num_experts_per_node,
                                         inter_size, expanded_num_rows, fc1_activation_type, quant_params, use_per_expert_act_scale, nullptr, stream);

  sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::BlockScaleFC2(
    DeepSeekBlockScaleGemmRunner& gemm_runner, T const* const input, void* const gemm_output,
    OutputType* const final_output, int64_t const* const expert_first_token_offset,
    WeightType const* const fc2_expert_weights, ScaleBiasType const* const fc2_expert_biases,
    float const* const unpermuted_final_scales, int const* const unpermuted_row_to_permuted_row,

    int const* const permuted_row_to_unpermuted_row, int const* const token_selected_experts,
    int64_t const* const num_valid_tokens_ptr, int64_t const num_rows, int64_t const expanded_num_rows,
    int64_t const hidden_size, int64_t const inter_size, int64_t const num_experts_per_node, int64_t const k,
    MOEParallelismConfig parallelism_config, bool const enable_alltoall, QuantParams& quant_params, cudaStream_t stream) {
  int shape_n = hidden_size;
  int shape_k = inter_size;

  // NOTE: we assume gemm_runner.configureWorkspace has already been called.
  gemm_runner.moeGemm(gemm_output, input, fc2_expert_weights, expert_first_token_offset, num_experts_per_node,
                      shape_n, shape_k, stream, nullptr, quant_params.fp8_block_scaling.fc2_scales_ptrs);

  sync_check_cuda_error(stream);

  finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
      static_cast<UnfusedGemmOutputType const*>(gemm_output), final_output, fc2_expert_biases,
      unpermuted_final_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row, token_selected_experts,
      expert_first_token_offset, num_rows, hidden_size, k, num_experts_per_node, parallelism_config, enable_alltoall,
      stream);
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
T const* CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::applyPrequantScale(
    void* smoothed_act, void const* permuted_data, void const* prequant_scales, int64_t const* num_valid_tokens_ptr,
    int64_t const expanded_num_rows, int64_t const seq_len, bool const use_awq, cudaStream_t stream) {
  T const* gemm_input;
  bool use_prequant_scale_kernel = use_awq && !std::is_same_v<T, WeightType>;
  if (use_prequant_scale_kernel) {
    ORT_ENFORCE(
        (!std::is_same_v<T, WeightType>), "Prequant scales are only used for different weight/activation type!");
    if constexpr (!std::is_same_v<T, WeightType>) {
      onnxruntime::llm::kernels::apply_per_channel_scale_kernel_launcher<UnfusedGemmOutputType, T>(
          reinterpret_cast<T*>(smoothed_act), reinterpret_cast<UnfusedGemmOutputType const*>(permuted_data),
          reinterpret_cast<UnfusedGemmOutputType const*>(prequant_scales), expanded_num_rows, seq_len,
          num_valid_tokens_ptr, stream);
    }
    gemm_input = reinterpret_cast<T const*>(smoothed_act);
  } else {
    gemm_input = reinterpret_cast<T const*>(permuted_data);
  }
  sync_check_cuda_error(stream);
  return gemm_input;
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::gemm1(
    MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner,
    DeepSeekBlockScaleGemmRunner* fp8_blockscale_gemm_runner, T const* const input, T* const output,
    void* const intermediate_result, int64_t const* const expert_first_token_offset,
    TmaWarpSpecializedGroupedGemmInput const tma_ws_input_template, WeightType const* const fc1_expert_weights,
    ScaleBiasType const* const fc1_expert_biases, int64_t const* const num_valid_tokens_ptr,
    ScaleBiasType const* const fc1_int_scales, float const* const fc1_fp8_dequant, float const* const fc2_fp8_quant,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat, QuantParams quant_params, int64_t const num_rows,
    int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const num_experts_per_node, ActivationType fc1_activation_type, float const** alpha_scale_ptr_array,
    bool bias_is_broadcast, cudaStream_t stream, cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode,
    int* num_active_experts_per, int* active_expert_global_ids) {
  if (fp8_blockscale_gemm_runner) {
    ORT_ENFORCE(!min_latency_mode);
    Self::BlockScaleFC1(*fp8_blockscale_gemm_runner, input, output, intermediate_result, expert_first_token_offset,
                        fc1_expert_weights, fc1_expert_biases, fc2_fp8_quant, num_rows, expanded_num_rows, hidden_size, inter_size,
                        num_experts_per_node, fc1_activation_type, quant_params, stream);
    return;
  }

  bool const using_tma_ws_gemm1 = gemm_runner.isTmaWarpSpecialized(config);
  bool const is_gated_activation = isGatedActivation(fc1_activation_type);
  bool const use_ampere_activation_fusion = gemm_runner.isFusedGatedActivation(config, is_gated_activation, inter_size, hidden_size);
  size_t const fc1_out_size = ((!use_ampere_activation_fusion) && is_gated_activation) ? inter_size * 2 : inter_size;

  int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

  if (min_latency_mode) {
    ORT_ENFORCE(using_tma_ws_gemm1, "Only TMA warp specialized GEMM is supported in min latency mode.");
    // TODO: as for bias, need to get the correct expert id according to the active expert global ids
    ORT_ENFORCE(fc1_expert_biases == nullptr, "Min latency mode does not support bias.");
    ORT_ENFORCE(use_fp4);
  }

  if (using_tma_ws_gemm1) {
    ORT_ENFORCE(config.is_tma_warp_specialized);
    ORT_ENFORCE(!use_ampere_activation_fusion);

    ORT_ENFORCE(!use_fp4 || fc1_fp4_act_flat);
    ORT_ENFORCE(!use_fp4 || fc2_fp4_act_flat);

    bool has_different_gemm_output_type = using_tma_ws_gemm1 && !std::is_same_v<T, OutputType>;
    bool const has_intermediate = has_different_gemm_output_type || is_gated_activation;
    ORT_ENFORCE(has_intermediate || input != output, "Input and output buffers are overlapping");
    auto* gemm_output = has_intermediate ? intermediate_result : static_cast<void*>(output);

    auto tma_ws_input = tma_ws_input_template;

    if (use_w4afp8) {
      alpha_scale_ptr_array = computeFP8DequantScale(
          alpha_scale_ptr_array, num_experts_per_node, quant_params.groupwise.fc1.alpha, stream);
    }

    auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input, total_tokens_including_expert,
                                                                                   /*weights*/ nullptr, /*scales*/ nullptr, /*zeros*/ nullptr, /*biases*/ nullptr, /*C*/ nullptr,
                                                                                   alpha_scale_ptr_array, /*occupancy*/ nullptr, fc1_activation_type, num_rows,
                                                                                   /*N*/ int64_t(fc1_out_size),
                                                                                   /*K*/ hidden_size, num_experts_per_node, quant_params.groupwise.group_size, /*bias_is_broadcast*/ true,
                                                                                   /*use_fused_moe*/ false, stream, config};
    gemm_runner.moeGemm(universal_input, tma_ws_input);

    sync_check_cuda_error(stream);

    // TODO: when bias_is_broadcast is false, fuse bias to gemm
    using GatedActOutputType = std::conditional_t<use_w4afp8, BackBoneType, T>;
    bool use_per_expert_act_scale = use_fp4        ? quant_params.fp4.fc2.use_per_expert_act_scale
                                    : use_wfp4afp8 ? quant_params.fp8_mxfp4.fc2.use_per_expert_act_scale
                                    : use_fp8      ? quant_params.fp8.fc2_use_per_expert_act_scale
                                                   : false;

    doActivation<GatedActOutputType, UnfusedGemmOutputType>(reinterpret_cast<GatedActOutputType*>(output),
                                                            static_cast<UnfusedGemmOutputType const*>(gemm_output), fc2_fp8_quant, fc1_expert_biases, bias_is_broadcast,
                                                            expert_first_token_offset, num_experts_per_node, inter_size, expanded_num_rows, fc1_activation_type,
                                                            quant_params, use_per_expert_act_scale, fc2_fp4_act_flat, stream);

    sync_check_cuda_error(stream);
  } else if (use_fp8) {
    ORT_ENFORCE(!use_ampere_activation_fusion);
    ORT_ENFORCE(!config.is_tma_warp_specialized);
    ORT_ENFORCE(!use_block_scaling);

    alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node, quant_params.fp8.dequant_fc1, stream);

    auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input,
                                                                                   total_tokens_including_expert, fc1_expert_weights, /*scales*/ nullptr, /*zeros*/ nullptr,
                                                                                   /*biases*/ nullptr, reinterpret_cast<UnfusedGemmOutputType*>(intermediate_result), alpha_scale_ptr_array,
                                                                                   /*occupancy*/ nullptr, fc1_activation_type, expanded_num_rows, /*N*/ int64_t(fc1_out_size),
                                                                                   /*K*/ hidden_size, num_experts_per_node, quant_params.groupwise.group_size, /*bias_is_broadcast*/ true,
                                                                                   /*use_fused_moe*/ false, stream, config};
    gemm_runner.moeGemm(universal_input, TmaWarpSpecializedGroupedGemmInput{});

    bool use_per_expert_act_scale = use_fp8 ? quant_params.fp8.fc2_use_per_expert_act_scale : false;
    doActivation<T, UnfusedGemmOutputType>(output, static_cast<UnfusedGemmOutputType const*>(intermediate_result),
                                           fc2_fp8_quant, fc1_expert_biases, bias_is_broadcast, expert_first_token_offset, num_experts_per_node,
                                           inter_size, expanded_num_rows, fc1_activation_type, quant_params, use_per_expert_act_scale, nullptr,
                                           stream);

    sync_check_cuda_error(stream);
  } else if (!is_gated_activation) {
    ORT_ENFORCE(!use_ampere_activation_fusion);
    ORT_ENFORCE(!config.is_tma_warp_specialized);
    ORT_ENFORCE(!use_block_scaling);
    if (use_w4afp8) {
      alpha_scale_ptr_array = computeFP8DequantScale(
          alpha_scale_ptr_array, num_experts_per_node, quant_params.groupwise.fc1.alpha, stream);
    }
    auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input,
                                                                                   total_tokens_including_expert, fc1_expert_weights,
                                                                                   /*scales*/ quant_params.groupwise.group_size > 0
                                                                                       ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_scales)
                                                                                       : fc1_int_scales,
                                                                                   /*zeros*/ quant_params.groupwise.group_size > 0
                                                                                       ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_zeros)
                                                                                       : nullptr,
                                                                                   fc1_expert_biases, reinterpret_cast<OutputType*>(output), alpha_scale_ptr_array, /*occupancy*/ nullptr,
                                                                                   fc1_activation_type, expanded_num_rows, /*N*/ int64_t(fc1_out_size),
                                                                                   /*K*/ hidden_size, num_experts_per_node, quant_params.groupwise.group_size, bias_is_broadcast,
                                                                                   /*use_fused_moe*/ false, stream, config};
    gemm_runner.moeGemmBiasAct(universal_input, TmaWarpSpecializedGroupedGemmInput{});

    sync_check_cuda_error(stream);
  } else {
    ORT_ENFORCE(!config.is_tma_warp_specialized);
    ORT_ENFORCE(is_gated_activation);
    ORT_ENFORCE(
        !use_ampere_activation_fusion || input != output, "Input and output buffers are overlapping");
    ORT_ENFORCE(!use_block_scaling);
    if (use_w4afp8) {
      alpha_scale_ptr_array = computeFP8DequantScale(
          alpha_scale_ptr_array, num_experts_per_node, quant_params.groupwise.fc1.alpha, stream);
    }
    // Run the GEMM with activation function overridden with `Identity`, we do the activation separately
    auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input,
                                                                                   total_tokens_including_expert, fc1_expert_weights,
                                                                                   /*scales*/ quant_params.groupwise.group_size > 0
                                                                                       ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_scales)
                                                                                       : fc1_int_scales,
                                                                                   /*zeros*/ quant_params.groupwise.group_size > 0
                                                                                       ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_zeros)
                                                                                       : nullptr,
                                                                                   fc1_expert_biases, static_cast<OutputType*>(use_ampere_activation_fusion ? output : intermediate_result),
                                                                                   alpha_scale_ptr_array, /*occupancy*/ nullptr,
                                                                                   use_ampere_activation_fusion ? fc1_activation_type : ActivationType::Identity, expanded_num_rows,
                                                                                   /*N*/ int64_t(fc1_out_size),
                                                                                   /*K*/ hidden_size, num_experts_per_node, quant_params.groupwise.group_size, bias_is_broadcast,
                                                                                   use_ampere_activation_fusion, stream, config};
    gemm_runner.moeGemmBiasAct(universal_input, TmaWarpSpecializedGroupedGemmInput{});

    sync_check_cuda_error(stream);

    if (!use_ampere_activation_fusion) {
      using GatedActOutputType = std::conditional_t<use_w4afp8, BackBoneType, T>;
      doGatedActivation<GatedActOutputType, UnfusedGemmOutputType>(reinterpret_cast<GatedActOutputType*>(output),
                                                                   static_cast<UnfusedGemmOutputType const*>(intermediate_result), num_valid_tokens_ptr, inter_size,
                                                                   expanded_num_rows, fc1_activation_type, stream);

      sync_check_cuda_error(stream);
    }
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::gemm2(
    MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner,
    DeepSeekBlockScaleGemmRunner* fp8_blockscale_gemm_runner, T const* const input, void* const gemm_output,
    OutputType* const final_output, int64_t const* const expert_first_token_offset,
    TmaWarpSpecializedGroupedGemmInput const tma_ws_input_template, WeightType const* const fc2_expert_weights,
    ScaleBiasType const* const fc2_expert_biases, ScaleBiasType const* const fc2_int_scales,
    float const* const fc2_fp8_dequant, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat,
    QuantParams quant_params, float const* const unpermuted_final_scales, float const* const permuted_final_scales,
    int const* const unpermuted_row_to_permuted_row, int const* permuted_row_to_unpermuted_row,
    int const* const token_selected_experts, int64_t const* const num_valid_tokens_ptr, int64_t const num_rows,
    int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const num_experts_per_node, int64_t const k, float const** alpha_scale_ptr_array, bool use_lora, void* fc2_lora,
    cudaStream_t stream, MOEParallelismConfig parallelism_config, bool const enable_alltoall,
    cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode, int* num_active_experts_per,
    int* active_expert_global_ids) {
  int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

  bool const using_tma_ws_gemm2 = gemm_runner.isTmaWarpSpecialized(config);

  if (min_latency_mode) {
    ORT_ENFORCE(using_tma_ws_gemm2, "Only TMA warp specialized GEMM is supported in min latency mode.");
    ORT_ENFORCE(use_fp4);
  }

  if (fp8_blockscale_gemm_runner) {
    Self::BlockScaleFC2(*fp8_blockscale_gemm_runner, input, gemm_output, final_output, expert_first_token_offset,
                        fc2_expert_weights, fc2_expert_biases, unpermuted_final_scales, unpermuted_row_to_permuted_row,
                        permuted_row_to_unpermuted_row, token_selected_experts, num_valid_tokens_ptr, num_rows, expanded_num_rows,
                        hidden_size, inter_size, num_experts_per_node, k, parallelism_config, enable_alltoall, quant_params,
                        stream);
    return;
  }

  TmaWarpSpecializedGroupedGemmInput tma_ws_input{};
  if (using_tma_ws_gemm2) {
    tma_ws_input = tma_ws_input_template;
    if (tma_ws_input.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE) {
      // TODO For some reason this has to be done here, it should not overlap with anything else, but
      // doing it in setupTmaWarpSpecializedInputs gives a different result. Ideally, we want this to run on a
      // second stream and overlap with everything else
      //
      // This also means it is included in the timing for the profiler, which is probably more representative
      // until we can overlap it
      CUDA_CALL_THROW(cudaMemsetAsync(final_output, 0x0, sizeof(OutputType) * num_rows * hidden_size, stream));
    }
  } else if (use_fp8) {
    alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node, fc2_fp8_dequant, stream);
  }
  if (use_w4afp8) {
    alpha_scale_ptr_array = computeFP8DequantScale(
        alpha_scale_ptr_array, num_experts_per_node, quant_params.groupwise.fc2.alpha, stream);
  }

  bool fuse_lora_bias = use_lora && !(use_fp8 || using_tma_ws_gemm2);
  // Note: expanded_num_rows, to check this value, it's greater than num_rows * num_experts_per_node
  auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input, total_tokens_including_expert,
                                                                                 fc2_expert_weights,
                                                                                 quant_params.groupwise.group_size > 0
                                                                                     ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc2.weight_scales)
                                                                                     : fc2_int_scales,
                                                                                 quant_params.groupwise.group_size > 0
                                                                                     ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc2.weight_zeros)
                                                                                     : nullptr,
                                                                                 fuse_lora_bias ? static_cast<ScaleBiasType*>(fc2_lora) : nullptr, static_cast<OutputType*>(gemm_output),
                                                                                 alpha_scale_ptr_array, /*occupancy*/ nullptr, ActivationType::Identity, expanded_num_rows,
                                                                                 /*N*/ hidden_size,
                                                                                 /*K*/ inter_size, num_experts_per_node, quant_params.groupwise.group_size, /*bias_is_broadcast*/ false,
                                                                                 /*use_fused_moe*/ false, stream, config};
  gemm_runner.moeGemmBiasAct(universal_input, tma_ws_input);
  sync_check_cuda_error(stream);

  if (min_latency_mode)
    return;

  if (use_lora && !fuse_lora_bias) {
    auto loraBiasApplyFunc = doActivation<UnfusedGemmOutputType, UnfusedGemmOutputType, ScaleBiasType>;
    loraBiasApplyFunc(static_cast<UnfusedGemmOutputType*>(gemm_output),
                      static_cast<UnfusedGemmOutputType const*>(gemm_output), nullptr,
                      static_cast<ScaleBiasType const*>(fc2_lora), false, expert_first_token_offset, num_experts_per_node,
                      hidden_size, expanded_num_rows, ActivationType::Identity, {}, false, nullptr, stream);
    sync_check_cuda_error(stream);
  }

  bool has_different_output_type_ampere = (use_w4afp8 || use_fp8) && !using_tma_ws_gemm2;
  bool using_hopper_fused_finalize = tma_ws_input.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
  bool has_different_output_type_tma_ws = !using_hopper_fused_finalize && using_tma_ws_gemm2;

  if (has_different_output_type_ampere || has_different_output_type_tma_ws) {
    finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
        static_cast<UnfusedGemmOutputType const*>(gemm_output), final_output, fc2_expert_biases,
        unpermuted_final_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
        token_selected_experts, expert_first_token_offset, num_rows, hidden_size, k, num_experts_per_node,
        parallelism_config, enable_alltoall, stream);
  } else if (!using_tma_ws_gemm2) {
    finalizeMoeRoutingKernelLauncher<OutputType, T>(static_cast<T const*>(gemm_output), final_output,
                                                    fc2_expert_biases, unpermuted_final_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
                                                    token_selected_experts, expert_first_token_offset, num_rows, hidden_size, k, num_experts_per_node,
                                                    parallelism_config, enable_alltoall, stream);
  }
  sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
bool CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::setupLoraWorkspace(
    int64_t expanded_num_rows, int64_t num_rows, int64_t inter_size, int64_t hidden_size, int start_expert,
    bool is_gated_activation, int num_experts_per_node, bool needs_num_valid, LoraParams& lora_params,
    cudaStream_t stream) {
  std::vector<int>& host_permuted_rows = host_lora_workspace_.host_permuted_rows;
  std::vector<void const*>& host_permuted_fc1_weight_ptrs = host_lora_workspace_.host_permuted_fc1_weight_ptrs;
  std::vector<void const*>& host_permuted_fc2_weight_ptrs = host_lora_workspace_.host_permuted_fc2_weight_ptrs;
  std::vector<void const*>& host_permuted_gated_weight_ptrs = host_lora_workspace_.host_permuted_gated_weight_ptrs;

  std::vector<int32_t>& host_permuted_fc1_lora_ranks = host_lora_workspace_.host_permuted_fc1_lora_ranks;
  std::vector<int32_t>& host_permuted_fc2_lora_ranks = host_lora_workspace_.host_permuted_fc2_lora_ranks;
  std::vector<int32_t>& host_permuted_gated_lora_ranks = host_lora_workspace_.host_permuted_gated_lora_ranks;
  std::vector<int64_t>& host_expert_first_token_offset = host_lora_workspace_.host_expert_first_token_offset;

  bool all_token_without_lora = true;

  host_permuted_fc1_weight_ptrs.resize(expanded_num_rows * 2);
  host_permuted_fc1_lora_ranks.resize(expanded_num_rows);
  host_permuted_fc2_weight_ptrs.resize(expanded_num_rows * 2);
  host_permuted_fc2_lora_ranks.resize(expanded_num_rows);

  if (is_gated_activation) {
    host_permuted_gated_weight_ptrs.resize(expanded_num_rows * 2);
    host_permuted_gated_lora_ranks.resize(expanded_num_rows);
  }

  CUDA_CALL_THROW(cudaEventSynchronize(*(lora_params.memcpy_event_ptr)));

  size_t num_valid_tokens = needs_num_valid ? host_expert_first_token_offset[num_experts_per_node] : expanded_num_rows;

  for (int expert_idx = 0; expert_idx < num_experts_per_node; ++expert_idx) {
    int weight_index = expert_idx + start_expert;
    for (size_t i = host_expert_first_token_offset[expert_idx]; i < host_expert_first_token_offset[expert_idx + 1];
         ++i) {
      int source_index = host_permuted_rows[i] % num_rows;
      int32_t lora_rank = lora_params.fc1_lora_ranks[source_index];
      host_permuted_fc1_weight_ptrs[i * 2] = reinterpret_cast<ScaleBiasType const*>(lora_params.fc1_lora_weight_ptrs[source_index * 2]) + weight_index * hidden_size * lora_rank;
      host_permuted_fc1_weight_ptrs[i * 2 + 1] = reinterpret_cast<ScaleBiasType const*>(lora_params.fc1_lora_weight_ptrs[source_index * 2 + 1]) + weight_index * lora_rank * inter_size;
      host_permuted_fc1_lora_ranks[i] = lora_rank;

      lora_rank = lora_params.fc2_lora_ranks[source_index];
      host_permuted_fc2_weight_ptrs[i * 2] = reinterpret_cast<ScaleBiasType const*>(lora_params.fc2_lora_weight_ptrs[source_index * 2]) + weight_index * inter_size * lora_rank;
      host_permuted_fc2_weight_ptrs[i * 2 + 1] = reinterpret_cast<ScaleBiasType const*>(lora_params.fc2_lora_weight_ptrs[source_index * 2 + 1]) + weight_index * lora_rank * hidden_size;
      host_permuted_fc2_lora_ranks[i] = lora_rank;

      if (host_permuted_fc1_lora_ranks[i] || host_permuted_fc2_lora_ranks[i]) {
        all_token_without_lora = false;
      }

      if (is_gated_activation) {
        lora_rank = lora_params.gated_lora_ranks[source_index];
        host_permuted_gated_weight_ptrs[i * 2] = reinterpret_cast<ScaleBiasType const*>(lora_params.gated_lora_weight_ptrs[source_index * 2]) + weight_index * hidden_size * lora_rank;
        host_permuted_gated_weight_ptrs[i * 2 + 1] = reinterpret_cast<ScaleBiasType const*>(lora_params.gated_lora_weight_ptrs[source_index * 2 + 1]) + weight_index * lora_rank * inter_size;
        host_permuted_gated_lora_ranks[i] = lora_rank;

        if (host_permuted_gated_lora_ranks[i]) {
          all_token_without_lora = false;
        }
      }
    }
  }
  return all_token_without_lora;
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
auto CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::loraFC1(int64_t expanded_num_rows,
                                                                                             int64_t inter_size, int64_t hidden_size, int num_experts_per_node, int start_expert,
                                                                                             int64_t const* num_valid_tokens_ptr, bool is_gated_activation, ScaleBiasType const* fc1_expert_biases,
                                                                                             LoraParams& lora_params, float const* input_fp8_dequant, cudaStream_t stream) -> ScaleBiasType const* {
  ORT_ENFORCE(!act_fp4, "LoRA does not support FP4 activations");
  std::vector<void const*>& host_permuted_fc1_weight_ptrs = host_lora_workspace_.host_permuted_fc1_weight_ptrs;
  std::vector<void const*>& host_permuted_gated_weight_ptrs = host_lora_workspace_.host_permuted_gated_weight_ptrs;

  std::vector<int32_t>& host_permuted_fc1_lora_ranks = host_lora_workspace_.host_permuted_fc1_lora_ranks;
  std::vector<int32_t>& host_permuted_gated_lora_ranks = host_lora_workspace_.host_permuted_gated_lora_ranks;
  std::vector<int64_t>& host_expert_first_token_offset = host_lora_workspace_.host_expert_first_token_offset;

  auto fc1_lora_impl = lora_params.fc1_lora_impl;
  int num_reqs = lora_params.num_reqs;

  ScaleBiasType *lora_gated_out = nullptr, *lora_fc1_result = nullptr;

  if (is_gated_activation) {
    lora_gated_out = lora_fc1_result_;
    lora_fc1_result = lora_fc1_result_ + expanded_num_rows * inter_size;
  } else {
    lora_fc1_result = lora_fc1_result_;
  }

  ScaleBiasType* input{};
  if constexpr (use_fp8) {
    ORT_ENFORCE(lora_input_);
    bool const scale_is_dequant = true;
    dequantFP8<ScaleBiasType, T>(lora_input_, permuted_data_, num_valid_tokens_ptr, hidden_size, expanded_num_rows,
                                 input_fp8_dequant, scale_is_dequant, stream);
    sync_check_cuda_error(stream);
    input = lora_input_;
  } else if constexpr (!act_fp4) {
    ORT_ENFORCE(!lora_input_);
    input = reinterpret_cast<ScaleBiasType*>(permuted_data_);
  }

  void* lora_workspace = lora_params.workspace;
  void* tmp_lora_fc_result = static_cast<void*>(lora_fc1_result);
  int64_t num_valid_tokens = host_expert_first_token_offset[num_experts_per_node];
  int64_t num_reqs_lora = std::min(num_valid_tokens, static_cast<int64_t>(num_reqs * num_experts_per_node));

  ::onnxruntime::llm::kernels::Lora_run(fc1_lora_impl.get(), num_valid_tokens, num_reqs_lora, input,
                                        host_permuted_fc1_lora_ranks.data(), host_permuted_fc1_weight_ptrs.data(), 0, &tmp_lora_fc_result,
                                        lora_workspace, stream);

  if (is_gated_activation) {
    void* tmp_lora_gated_result = static_cast<void*>(lora_gated_out);
    ::onnxruntime::llm::kernels::Lora_run(fc1_lora_impl.get(), num_valid_tokens, num_reqs_lora, input,
                                          host_permuted_gated_lora_ranks.data(), host_permuted_gated_weight_ptrs.data(), 0, &tmp_lora_gated_result,
                                          lora_workspace, stream);
  }

  // add bias and reorder
  if (fc1_expert_biases != nullptr) {
    loraAddBias(lora_add_bias_, lora_fc1_result_, fc1_expert_biases, num_valid_tokens_ptr, inter_size,
                permuted_token_selected_experts_, expanded_num_rows, is_gated_activation, stream);
    return lora_add_bias_;
  } else if (is_gated_activation) {
    loraReorder(lora_add_bias_, lora_fc1_result_, num_valid_tokens_ptr, inter_size, expanded_num_rows, stream);
    return lora_add_bias_;
  } else {
    return lora_fc1_result_;
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::loraFC2(int64_t inter_size,
                                                                                             int64_t hidden_size, int num_experts_per_node, int start_expert, int64_t const* num_valid_tokens_ptr,
                                                                                             int64_t num_tokens, LoraParams& lora_params, float const* fc2_fp8_quant, cudaStream_t stream) {
  std::vector<void const*>& host_permuted_fc2_weight_ptrs = host_lora_workspace_.host_permuted_fc2_weight_ptrs;
  std::vector<int32_t>& host_permuted_fc2_lora_ranks = host_lora_workspace_.host_permuted_fc2_lora_ranks;
  std::vector<int64_t>& host_expert_first_token_offset = host_lora_workspace_.host_expert_first_token_offset;
  auto fc2_lora_impl = lora_params.fc2_lora_impl;
  int num_reqs = lora_params.num_reqs;

  ScaleBiasType* input{};
  if constexpr (use_fp8) {
    ORT_ENFORCE(lora_input_);
    bool const scale_is_dequant = false;
    dequantFP8(lora_input_, fc1_result_, num_valid_tokens_ptr, inter_size, num_tokens, fc2_fp8_quant,
               scale_is_dequant, stream);
    sync_check_cuda_error(stream);
    input = lora_input_;
  } else if constexpr (!act_fp4) {
    ORT_ENFORCE(!lora_input_);
    input = reinterpret_cast<ScaleBiasType*>(fc1_result_);
  }

  void* lora_workspace = lora_params.workspace;
  int64_t num_valid_tokens = host_expert_first_token_offset[num_experts_per_node];
  void* tmp_lora_fc_result = static_cast<void*>(lora_fc2_result_);
  int64_t num_reqs_lora = std::min(num_valid_tokens, static_cast<int64_t>(num_reqs * num_experts_per_node));

  ::onnxruntime::llm::kernels::Lora_run(fc2_lora_impl.get(), num_valid_tokens, num_reqs_lora, input,
                                        host_permuted_fc2_lora_ranks.data(), host_permuted_fc2_weight_ptrs.data(), 0, &tmp_lora_fc_result,
                                        lora_workspace, stream);
  sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::runMoe(
    void const* input_activations_void, void const* input_sf_void, int const* token_selected_experts,
    float const* token_final_scales, void const* fc1_expert_weights_void, void const* fc1_expert_biases_void,
    ActivationType fc1_activation_type, void const* fc2_expert_weights_void, void const* fc2_expert_biases_void,
    QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const full_num_experts, int const experts_per_token, char* workspace_ptr, void* final_output_void,
    int* unpermuted_row_to_permuted_row, MOEParallelismConfig parallelism_config, bool const enable_alltoall,
    bool use_lora, LoraParams& lora_params, bool use_deepseek_fp8_block_scale, bool min_latency_mode,
    MoeMinLatencyParams& min_latency_params, cudaStream_t stream) {
  static constexpr bool int_scales_required = std::is_same<WeightType, uint8_t>::value || std::is_same<WeightType, cutlass::uint4b_t>::value;
  static constexpr bool fp8_scales_required = std::is_same<WeightType, __nv_fp8_e4m3>::value || std::is_same<WeightType, __nv_fp8_e5m2>::value;

  auto const* input_activations = static_cast<InputType const*>(input_activations_void);
  auto const* input_sf = input_sf_void
                             ? reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF const*>(input_sf_void)
                             : nullptr;
  auto const* fc1_expert_weights = static_cast<WeightType const*>(fc1_expert_weights_void);
  auto const* fc1_expert_biases = reinterpret_cast<ScaleBiasType const*>(fc1_expert_biases_void);
  auto const* fc2_expert_weights = static_cast<WeightType const*>(fc2_expert_weights_void);
  auto const* fc1_int_scales = reinterpret_cast<ScaleBiasType const*>(quant_params.wo.fc1_weight_scales);
  auto const* fc2_int_scales = reinterpret_cast<ScaleBiasType const*>(quant_params.wo.fc2_weight_scales);

  auto const* fc1_fp8_dequant = quant_params.fp8.dequant_fc1;
  auto const* fc2_fp8_quant = quant_params.fp8.quant_fc2;
  auto const* fc2_fp8_dequant = quant_params.fp8.dequant_fc2;
  auto const* input_fp8_dequant = quant_params.fp8.dequant_input;

  auto const* fc2_wfp4afp8_quant_scale = quant_params.fp8_mxfp4.fc2.act_global_scale;

  auto const* fc2_expert_biases = reinterpret_cast<ScaleBiasType const*>(fc2_expert_biases_void);
  auto* final_output = static_cast<OutputType*>(final_output_void);
  float const* token_topk_unpermuted_scales = token_final_scales;
  // Note: getDeepSeekBlockScaleGemmRunner will do a sanity check on our template parameters.
  auto* blockscale_gemm_runner = use_deepseek_fp8_block_scale ? getDeepSeekBlockScaleGemmRunner() : nullptr;

  ORT_ENFORCE(input_activations);
  ORT_ENFORCE(token_selected_experts);
  ORT_ENFORCE(fc1_expert_weights);
  ORT_ENFORCE(fc2_expert_weights);
  ORT_ENFORCE(workspace_ptr);
  // ORT_ENFORCE(token_topk_unpermuted_scales);
  ORT_ENFORCE(unpermuted_row_to_permuted_row);
  ORT_ENFORCE(full_num_experts % parallelism_config.ep_size == 0);
  ORT_ENFORCE(full_num_experts % parallelism_config.cluster_size == 0);

  if (quant_params.mxfp8_mxfp4.fc1.weight_block_scale) {
    ORT_ENFORCE(hidden_size % (64 * 8 / sizeof_bits<WeightType>::value) == 0,
                "Hidden size %d does not meet minimum alignment requirements for MXFP8_MXFP4 MOE GEMM %d",
                (int)hidden_size, (int)(64 * 8 / sizeof_bits<WeightType>::value));
    ORT_ENFORCE(inter_size % (64 * 8 / sizeof_bits<WeightType>::value) == 0,
                "Inter size %d does not meet minimum alignment requirements for MXFP8_MXFP4 MOE GEMM %d", (int)inter_size,
                (int)(64 * 8 / sizeof_bits<WeightType>::value));
  } else {
    // Require at least 128 bits of alignment for MOE GEMM
    ORT_ENFORCE(hidden_size % (128 / sizeof_bits<WeightType>::value) == 0,
                "Hidden size %d does not meet minimum alignment requirements for MOE GEMM %d", (int)hidden_size,
                (int)(128 / sizeof_bits<WeightType>::value));
    ORT_ENFORCE(inter_size % (128 / sizeof_bits<WeightType>::value) == 0,
                "Inter size %d does not meet minimum alignment requirements for MOE GEMM %d", (int)inter_size,
                (int)(128 / sizeof_bits<WeightType>::value));
  }

  // These values must fit into an int for building the source maps
  ORT_ENFORCE(num_rows <= std::numeric_limits<int>::max(), "Number of rows is too large");
  ORT_ENFORCE(
      num_rows * full_num_experts <= std::numeric_limits<int>::max(), "Number of rows * num_experts is too large");
  ORT_ENFORCE(experts_per_token * full_num_experts <= std::numeric_limits<int>::max(),
              "experts_per_token * num_experts is too large");

  ORT_ENFORCE(gemm1_config_, "MOE GEMM1 Config is not set");
  ORT_ENFORCE(gemm2_config_, "MOE GEMM2 Config is not set");

  ORT_ENFORCE(!use_lora || !act_fp4, "MOE does not support LoRA with FP4 model");

  if (int_scales_required) {
    if (!(quant_params.groupwise.fc1.weight_scales && quant_params.groupwise.fc2.weight_scales)) {
      ORT_ENFORCE(
          fc1_int_scales != nullptr, "Weight scales expected but scale for first matmul is a null pointer");
      ORT_ENFORCE(
          fc2_int_scales != nullptr, "Weight scales expected but scale for second matmul is a null pointer");
    }
    ORT_ENFORCE(fc1_fp8_dequant == nullptr && fc2_fp8_quant == nullptr && fc2_fp8_dequant == nullptr,
                "FP8 scales are provided for integer quantization");
  } else if (fp8_scales_required && !use_deepseek_fp8_block_scale) {
    ORT_ENFORCE(
        fc1_fp8_dequant != nullptr, "FP8 scales expected but dequant scale for FC1 is a null pointer");
    ORT_ENFORCE(fc2_fp8_quant != nullptr, "FP8 scales expected but quant scale for FC2 is a null pointer");
    ORT_ENFORCE(
        fc2_fp8_dequant != nullptr, "FP8 scales expected but quant scale for FC2 is a null pointer");

    ORT_ENFORCE(
        fc1_int_scales == nullptr && fc2_int_scales == nullptr, "Integer scales are provided for FP8 quantization");
  } else if (use_lora && use_fp8) {
    ORT_ENFORCE(
        input_fp8_dequant != nullptr, "FP8 scales expected but quant scale for input is a null pointer");
  } else {
    ORT_ENFORCE(
        fc1_int_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC1");
    ORT_ENFORCE(
        fc2_int_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC2");
    ORT_ENFORCE(
        fc1_fp8_dequant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received dequant scale for FC1");
    ORT_ENFORCE(
        fc2_fp8_quant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
    ORT_ENFORCE(
        fc2_fp8_dequant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
  }

  bool use_awq = quant_params.groupwise.fc1.act_scales && quant_params.groupwise.fc2.act_scales;
  int const num_experts_per_node = full_num_experts / parallelism_config.ep_size;

  configureWsPtrs(workspace_ptr, num_rows, hidden_size, inter_size, num_experts_per_node, experts_per_token,
                  fc1_activation_type, parallelism_config, use_lora, use_deepseek_fp8_block_scale, min_latency_mode, use_awq);

  int start_expert = num_experts_per_node * parallelism_config.ep_rank;
  int end_expert = start_expert + num_experts_per_node;

  bool const needs_num_valid = parallelism_config.ep_size > 1;
  int64_t const* num_valid_tokens_ptr = needs_num_valid ? expert_first_token_offset_ + num_experts_per_node : nullptr;

  auto expanded_num_rows = num_rows * experts_per_token;

  if (min_latency_mode) {
    ORT_ENFORCE(use_lora == false);
    ORT_ENFORCE(use_awq == false);
    ORT_ENFORCE(use_fp4 == true);

    buildMinLatencyActiveExpertMaps(min_latency_params.num_active_experts_per_node,
                                    min_latency_params.experts_to_token_score, min_latency_params.active_expert_global_ids,
                                    expert_first_token_offset_, token_selected_experts, token_final_scales, num_rows, experts_per_token,
                                    start_expert, end_expert, num_experts_per_node, parallelism_config.cluster_rank,
                                    parallelism_config.cluster_size, full_num_experts, stream);
    sync_check_cuda_error(stream);

    auto [gemm1_tma_ws_input, gemm2_tma_ws_input] = setupTmaWarpSpecializedInputs(num_rows, expanded_num_rows,
                                                                                  fc1_activation_type, hidden_size, inter_size, num_experts_per_node, input_activations_void, input_sf,
                                                                                  final_output, fc1_expert_weights, fc2_expert_weights, quant_params, fc1_expert_biases, fc2_expert_biases,
                                                                                  min_latency_mode, min_latency_params, use_lora, start_expert, parallelism_config, stream);

    // todo: input_activations_void should be nvfp4, waiting for yuxian's mr ready
    Self::gemm1(moe_gemm_runner_, blockscale_gemm_runner, reinterpret_cast<T const*>(input_activations_void),
                fc1_result_, glu_inter_result_, expert_first_token_offset_, gemm1_tma_ws_input, fc1_expert_weights,
                fc1_expert_biases, num_valid_tokens_ptr, fc1_int_scales, fc1_fp8_dequant,
                use_wfp4afp8 ? fc2_wfp4afp8_quant_scale : fc2_fp8_quant, input_sf /*input fp4 scale or expanded fp4 scale*/,
                fc2_fp4_act_scale_, quant_params, num_rows, expanded_num_rows, hidden_size, inter_size,
                num_experts_per_node, fc1_activation_type, alpha_scale_ptr_array_fc1_, !use_lora, stream, *gemm1_config_,
                true, min_latency_params.num_active_experts_per_node, min_latency_params.active_expert_global_ids);
    sync_check_cuda_error(stream);

    auto gemm2_input = applyPrequantScale(smoothed_act_, fc1_result_, quant_params.groupwise.fc2.act_scales,
                                          num_valid_tokens_ptr, expanded_num_rows, inter_size, use_awq, stream);
    Self::gemm2(moe_gemm_runner_, blockscale_gemm_runner, gemm2_input, final_output, nullptr,
                expert_first_token_offset_, gemm2_tma_ws_input, fc2_expert_weights, fc2_expert_biases, fc2_int_scales,
                fc2_fp8_dequant, fc2_fp4_act_scale_, quant_params, token_topk_unpermuted_scales,
                permuted_token_final_scales_, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row_,
                token_selected_experts, num_valid_tokens_ptr, num_rows, expanded_num_rows, hidden_size, inter_size,
                num_experts_per_node, experts_per_token, alpha_scale_ptr_array_fc2_, use_lora, lora_fc2_result_, stream,
                parallelism_config, enable_alltoall, *gemm2_config_, true, min_latency_params.num_active_experts_per_node,
                min_latency_params.active_expert_global_ids);
    sync_check_cuda_error(stream);
  } else {
    bool fused_prologue_result = false;
    if (!use_w4afp8) {
      // WAR: fusedBuildExpertMapsSortFirstToken kernel will lead to illegal memory access for W4AFP8
      fused_prologue_result = fusedBuildExpertMapsSortFirstToken(token_selected_experts,
                                                                 permuted_row_to_unpermuted_row_, unpermuted_row_to_permuted_row, expert_first_token_offset_, num_rows,
                                                                 num_experts_per_node, experts_per_token, start_expert, end_expert, stream);
    }

    if (!fused_prologue_result) {
      ORT_LLM_LOG_TRACE("Falling back to unfused prologue");
      threeStepBuildExpertMapsSortFirstToken(token_selected_experts, permuted_token_selected_experts_,
                                             permuted_row_to_unpermuted_row_, unpermuted_row_to_permuted_row, expert_first_token_offset_,
                                             blocked_expert_counts_, blocked_expert_counts_cumsum_, blocked_row_to_unpermuted_row_, num_rows,
                                             num_experts_per_node, experts_per_token, start_expert, stream);
    }

    sync_check_cuda_error(stream);

    bool is_gated_activation = isGatedActivation(fc1_activation_type);

    if (use_lora) {
      std::vector<int>& host_permuted_rows = host_lora_workspace_.host_permuted_rows;
      std::vector<int64_t>& host_expert_first_token_offset = host_lora_workspace_.host_expert_first_token_offset;
      host_permuted_rows.resize(expanded_num_rows);
      CUDA_CALL_THROW(onnxruntime::llm::common::cudaMemcpyAsyncSanitized(host_permuted_rows.data(),
                                                                         permuted_row_to_unpermuted_row_, expanded_num_rows * sizeof(int), cudaMemcpyDeviceToHost, stream));
      host_expert_first_token_offset.resize(num_experts_per_node + 1);
      CUDA_CALL_THROW(onnxruntime::llm::common::cudaMemcpyAsyncSanitized(host_expert_first_token_offset.data(),
                                                                         expert_first_token_offset_, (num_experts_per_node + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost,
                                                                         stream));
      CUDA_CALL_THROW(cudaEventRecord(*(lora_params.memcpy_event_ptr), stream));
    }

    // Only NVFP4xNVFP4 supports FC1 per-expert act scale
    bool use_per_expert_act_scale = use_fp4 ? quant_params.fp4.fc1.use_per_expert_act_scale : false;
    T* gemm1_input_expand = use_w4afp8 ? reinterpret_cast<T*>(smoothed_act_) : reinterpret_cast<T*>(permuted_data_);
    expandInputRowsKernelLauncher(input_activations, gemm1_input_expand, token_topk_unpermuted_scales,
                                  permuted_token_final_scales_, permuted_row_to_unpermuted_row_, num_rows, hidden_size, experts_per_token,
                                  num_experts_per_node, quant_params, use_per_expert_act_scale, expert_first_token_offset_,
                                  fc1_fp4_act_scale_, input_sf, use_w4afp8 ? quant_params.groupwise.fc1.act_scales : nullptr, stream);
    auto const* gemm1_input = gemm1_input_expand;

    sync_check_cuda_error(stream);

    auto [gemm1_tma_ws_input, gemm2_tma_ws_input] = setupTmaWarpSpecializedInputs(num_rows, expanded_num_rows,
                                                                                  fc1_activation_type, hidden_size, inter_size, num_experts_per_node, input_activations_void, input_sf,
                                                                                  final_output, fc1_expert_weights, fc2_expert_weights, quant_params, fc1_expert_biases, fc2_expert_biases,
                                                                                  min_latency_mode, min_latency_params, use_lora, start_expert, parallelism_config, stream);

    if (use_lora) {
      bool all_token_without_lora = setupLoraWorkspace(expanded_num_rows, num_rows, inter_size, hidden_size,
                                                       start_expert, is_gated_activation, num_experts_per_node, needs_num_valid, lora_params, stream);

      if (!all_token_without_lora) {
        fc1_expert_biases = loraFC1(expanded_num_rows, inter_size, hidden_size, num_experts_per_node,
                                    start_expert, num_valid_tokens_ptr, is_gated_activation, fc1_expert_biases, lora_params,
                                    input_fp8_dequant, stream);
        sync_check_cuda_error(stream);
      } else {
        use_lora = false;
      }
    }

    if constexpr (!use_w4afp8) {
      gemm1_input = applyPrequantScale(smoothed_act_, permuted_data_, quant_params.groupwise.fc1.act_scales,
                                       num_valid_tokens_ptr, expanded_num_rows, hidden_size, use_awq, stream);
    }
    sync_check_cuda_error(stream);
    Self::gemm1(moe_gemm_runner_, blockscale_gemm_runner, gemm1_input, fc1_result_, glu_inter_result_,
                expert_first_token_offset_, gemm1_tma_ws_input, fc1_expert_weights, fc1_expert_biases, num_valid_tokens_ptr,
                fc1_int_scales, fc1_fp8_dequant, use_wfp4afp8 ? fc2_wfp4afp8_quant_scale : fc2_fp8_quant,
                fc1_fp4_act_scale_, fc2_fp4_act_scale_, quant_params, num_rows, expanded_num_rows, hidden_size, inter_size,
                num_experts_per_node, fc1_activation_type, alpha_scale_ptr_array_fc1_, !use_lora, stream, *gemm1_config_,
                false, nullptr, nullptr);
    sync_check_cuda_error(stream);

    if (use_lora) {
      loraFC2(inter_size, hidden_size, num_experts_per_node, start_expert, num_valid_tokens_ptr,
              expanded_num_rows, lora_params, fc2_fp8_quant, stream);
      sync_check_cuda_error(stream);
    }

    auto gemm2_input = applyPrequantScale(smoothed_act_, fc1_result_, quant_params.groupwise.fc2.act_scales,
                                          num_valid_tokens_ptr, expanded_num_rows, inter_size, use_awq, stream);
    sync_check_cuda_error(stream);
    Self::gemm2(moe_gemm_runner_, blockscale_gemm_runner, gemm2_input, fc2_result_, final_output,
                expert_first_token_offset_, gemm2_tma_ws_input, fc2_expert_weights, fc2_expert_biases, fc2_int_scales,
                fc2_fp8_dequant, fc2_fp4_act_scale_, quant_params, token_topk_unpermuted_scales,
                permuted_token_final_scales_, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row_,
                token_selected_experts, num_valid_tokens_ptr, num_rows, expanded_num_rows, hidden_size, inter_size,
                num_experts_per_node, experts_per_token, alpha_scale_ptr_array_fc2_, use_lora, lora_fc2_result_, stream,
                parallelism_config, enable_alltoall, *gemm2_config_, false, nullptr, nullptr);
    sync_check_cuda_error(stream);
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::computeStridesTmaWarpSpecialized(
    int64_t const* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput layout_info1,
    TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens, int64_t expanded_num_tokens, int64_t gemm1_n,
    int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k, int const num_experts_per_node, T const* gemm1_in,
    T const* gemm2_in, WeightType const* weights1, WeightType const* weights2, float const* fp8_dequant1,
    float const* fp8_dequant2, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2, QuantParams quant_params,
    ScaleBiasType const* bias1, ScaleBiasType const* bias2, UnfusedGemmOutputType* gemm1_output,
    UnfusedGemmOutputType* gemm2_output, cudaStream_t stream) {
  // Always nullptr
  layout_info1.ptr_c = nullptr;
  layout_info1.stride_c = nullptr;
  layout_info2.ptr_c = nullptr;
  layout_info2.stride_c = nullptr;

  auto alpha_scale_flat1 = use_fp4        ? quant_params.fp4.fc1.global_scale
                           : use_wfp4afp8 ? quant_params.fp8_mxfp4.fc1.global_scale
                           : use_fp8      ? fp8_dequant1
                                          : nullptr;
  auto alpha_scale_flat2 = use_fp4        ? quant_params.fp4.fc2.global_scale
                           : use_wfp4afp8 ? quant_params.fp8_mxfp4.fc2.global_scale
                           : use_fp8      ? fp8_dequant2
                                          : nullptr;
  if (!alpha_scale_flat1 && !alpha_scale_flat2) {
    layout_info1.alpha_scale_ptr_array = nullptr;
    layout_info2.alpha_scale_ptr_array = nullptr;
  }

  layout_info1.int4_groupwise_params.enabled = use_w4afp8;
  layout_info2.int4_groupwise_params.enabled = use_w4afp8;

  layout_info1.fpX_block_scaling_type = getScalingType();
  layout_info2.fpX_block_scaling_type = getScalingType();

  int const threads = std::min(1024, num_experts_per_node);
  int const blocks = (num_experts_per_node + threads - 1) / threads;

  auto* kernel_instance = &computeStridesTmaWarpSpecializedKernel<T, WeightType, OutputType, ScaleBiasType>;

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, kernel_instance, expert_first_token_offset, layout_info1, layout_info2, num_tokens,
                     expanded_num_tokens, gemm1_n, gemm1_k, gemm2_n, gemm2_k, num_experts_per_node, gemm1_in, gemm2_in, weights1,
                     weights2, alpha_scale_flat1, alpha_scale_flat2, fp4_act_flat1, fp4_act_flat2, quant_params, bias1, bias2,
                     gemm1_output, gemm2_output);

  return std::make_pair(layout_info1, layout_info2);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType,
                   Enable>::computeStridesTmaWarpSpecializedLowLatency(TmaWarpSpecializedGroupedGemmInput layout_info1,
                                                                       TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens, int64_t gemm1_n, int64_t gemm1_k,
                                                                       int64_t gemm2_n, int64_t gemm2_k, int const num_experts, T const* input1, T const* input2,
                                                                       WeightType const* weights1, WeightType const* weights2, float const* fp8_dequant1, float const* fp8_dequant2,
                                                                       TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
                                                                       TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat, QuantParams quant_params,
                                                                       ScaleBiasType const* bias1, ScaleBiasType const* bias2, UnfusedGemmOutputType* output1,
                                                                       UnfusedGemmOutputType* output2, int const* num_active_experts_per, int const* active_expert_global_ids,
                                                                       int start_expert, cudaStream_t stream) {
  ORT_ENFORCE(!use_w4afp8, "W4AFP8 is not supported in low latency mode");

  // Always nullptr
  layout_info1.ptr_c = nullptr;
  layout_info1.stride_c = nullptr;
  layout_info2.ptr_c = nullptr;
  layout_info2.stride_c = nullptr;

  auto alpha_scale_flat1 = use_fp4        ? quant_params.fp4.fc1.global_scale
                           : use_wfp4afp8 ? quant_params.fp8_mxfp4.fc1.global_scale
                                          : fp8_dequant1;
  auto alpha_scale_flat2 = use_fp4        ? quant_params.fp4.fc2.global_scale
                           : use_wfp4afp8 ? quant_params.fp8_mxfp4.fc2.global_scale
                                          : fp8_dequant2;
  if (!alpha_scale_flat1) {
    layout_info1.alpha_scale_ptr_array = nullptr;
  }
  if (!alpha_scale_flat2) {
    layout_info2.alpha_scale_ptr_array = nullptr;
  }

  layout_info1.int4_groupwise_params.enabled = false;
  layout_info2.int4_groupwise_params.enabled = false;

  int const threads = std::min(1024, num_experts);
  int const blocks = (num_experts + threads - 1) / threads;

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config,
                     computeStridesTmaWarpSpecializedLowLatencyKernel<T, WeightType, OutputType, ScaleBiasType>, layout_info1,
                     layout_info2, num_tokens, gemm1_n, gemm1_k, gemm2_n, gemm2_k, num_experts, input1, input2, weights1, weights2,
                     alpha_scale_flat1, alpha_scale_flat2, fc1_fp4_act_flat, fc2_fp4_act_flat, quant_params, bias1, bias2, output1,
                     output2, num_active_experts_per, active_expert_global_ids, start_expert);

  return std::make_pair(layout_info1, layout_info2);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::setupTmaWarpSpecializedInputs(
    int64_t num_rows, int64_t expanded_num_rows, ActivationType fc1_activation_type, int64_t hidden_size,
    int64_t inter_size, int64_t num_experts_per_node, void const* input_activations_void,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, void* final_output,
    WeightType const* fc1_expert_weights, WeightType const* fc2_expert_weights, QuantParams quant_params,
    ScaleBiasType const* fc1_expert_biases, ScaleBiasType const* fc2_expert_biases, bool min_latency_mode,
    MoeMinLatencyParams& min_latency_params, bool use_lora, int start_expert, MOEParallelismConfig parallelism_config,
    cudaStream_t stream) {
  auto gemm1_tma_ws_input = tma_ws_grouped_gemm1_input_;
  auto gemm2_tma_ws_input = tma_ws_grouped_gemm2_input_;
  if (!moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_) && !moe_gemm_runner_.isTmaWarpSpecialized(*gemm2_config_)) {
    return std::make_pair(gemm1_tma_ws_input, gemm2_tma_ws_input);
  }

  bool use_awq = quant_params.groupwise.fc1.act_scales && quant_params.groupwise.fc2.act_scales;

  bool is_gated_activation = isGatedActivation(fc1_activation_type);
  int64_t const fc1_out_size = is_gated_activation ? inter_size * 2 : inter_size;

  bool has_different_gemm_output_type = !std::is_same_v<T, UnfusedGemmOutputType>;
  bool const has_intermediate = has_different_gemm_output_type || is_gated_activation;
  auto* gemm1_output = has_intermediate ? glu_inter_result_ : static_cast<void*>(fc1_result_);

  bool use_prequant_scale_kernel = use_awq && !std::is_same_v<T, WeightType>;
  auto gemm2_input = use_prequant_scale_kernel ? smoothed_act_ : fc1_result_;

  if (min_latency_mode) {
    auto gemm1_input = input_activations_void;

    gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
    gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;

    ORT_ENFORCE(gemm1_input != gemm1_output, "Input and output buffers are overlapping");
    return Self::computeStridesTmaWarpSpecializedLowLatency(gemm1_tma_ws_input, gemm2_tma_ws_input, num_rows,
                                                            fc1_out_size, hidden_size, hidden_size, inter_size, num_experts_per_node,
                                                            reinterpret_cast<T const*>(gemm1_input), reinterpret_cast<T const*>(gemm2_input), fc1_expert_weights,
                                                            fc2_expert_weights, quant_params.fp8.dequant_fc1, quant_params.fp8.dequant_fc2, input_sf,
                                                            fc2_fp4_act_scale_, quant_params, nullptr, nullptr, reinterpret_cast<UnfusedGemmOutputType*>(gemm1_output),
                                                            reinterpret_cast<UnfusedGemmOutputType*>(fc2_result_), min_latency_params.num_active_experts_per_node,
                                                            min_latency_params.active_expert_global_ids, start_expert, stream);
  } else {
    auto gemm1_input = use_prequant_scale_kernel ? smoothed_act_ : permuted_data_;

    gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
    gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;

    bool apply_bias = parallelism_config.tp_rank == 0;
    bool using_hopper_fused_finalize = !use_deterministic_hopper_reduce_ && gemm2_config_->sm_version == 90 && !use_w4afp8 && !use_lora;
    if (using_hopper_fused_finalize) {
      assert(min_latency_mode == false);
      gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
      gemm2_tma_ws_input.setFinalizeFusionParams(final_output, permuted_token_final_scales_,
                                                 expert_first_token_offset_, permuted_row_to_unpermuted_row_, apply_bias ? fc2_expert_biases : nullptr,
                                                 hidden_size, num_rows);
    }

    // fp8_mxfp4 memsets the scaling factors to 1.0f
    if (quant_params.fp8_mxfp4.fc1.weight_block_scale) {
      // We are in FP8 x MXFP4 mode
      ORT_ENFORCE(quant_params.fp8_mxfp4.fc2.weight_block_scale);
      ORT_ENFORCE(fc1_fp4_act_scale_ != nullptr);
      ORT_ENFORCE(fc1_fp4_act_scale_ == fc2_fp4_act_scale_,
                  "WFP4AFP8 expects the scaling factors to be aliased for gemm1 & gemm2");

      TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF weight_block_scale_value_int{};
#ifdef ENABLE_FP8
      __nv_fp8_e8m0 tmp;
      tmp.__x = __nv_cvt_float_to_e8m0(1.0f, __NV_SATFINITE, cudaRoundPosInf);
      std::memcpy(&weight_block_scale_value_int, &tmp, sizeof(tmp));
#endif

      auto act_sf_rows = std::min(expanded_num_rows, num_rows * num_experts_per_node);
      auto fc1_sf_offset = getOffsetActivationSF(num_experts_per_node, act_sf_rows, hidden_size,
                                                 TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);
      auto fc2_sf_offset = getOffsetActivationSF(num_experts_per_node, act_sf_rows, inter_size,
                                                 TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);
      auto max_size = std::max(fc1_sf_offset, fc2_sf_offset) * sizeof(TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF);
      CUDA_CALL_THROW(cudaMemsetAsync(fc1_fp4_act_scale_, weight_block_scale_value_int, max_size, stream));
    }

    ORT_ENFORCE(gemm1_input != gemm1_output, "Input and output buffers are overlapping");
    return Self::computeStridesTmaWarpSpecialized(expert_first_token_offset_, gemm1_tma_ws_input,
                                                  gemm2_tma_ws_input, num_rows, expanded_num_rows, fc1_out_size, hidden_size, hidden_size, inter_size,
                                                  num_experts_per_node, reinterpret_cast<T const*>(gemm1_input), reinterpret_cast<T const*>(gemm2_input),
                                                  fc1_expert_weights, fc2_expert_weights, quant_params.fp8.dequant_fc1, quant_params.fp8.dequant_fc2,
                                                  fc1_fp4_act_scale_, fc2_fp4_act_scale_, quant_params, fc1_expert_biases, fc2_expert_biases,
                                                  reinterpret_cast<UnfusedGemmOutputType*>(gemm1_output),
                                                  reinterpret_cast<UnfusedGemmOutputType*>(fc2_result_), stream);
  }
}

// ==================== Helper for getting load balanced routing for profiling ==================================

__global__ void prepareFakeRouterBuffers(
    int* token_selected_experts, int64_t num_tokens, int64_t k, int64_t num_experts) {
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t sample = blockIdx.y;
  if (tid >= num_tokens) {
    return;
  }

  // Offset the buffers to the start of the sample
  token_selected_experts += sample * num_tokens * k;

  // This is not perf sensitive we just init the state here every time prepare is called
  // This means the first N tokens will always have the same distribution, regardless of num_tokens
  curandStatePhilox4_32_10_t state;
  curand_init(sample, tid, 0, &state);
  for (int k_idx = 0; k_idx < k; k_idx++) {
    while (true) {
      // curand_uniform includes 1 but not 0, so round up and subtract 1
      int expert = std::ceil(static_cast<float>(num_experts) * curand_uniform(&state)) - 1;

      bool valid = true;
      for (int prev_k = 0; prev_k < k_idx; prev_k++) {
        int prev_expert = token_selected_experts[k * tid + prev_k];
        if (expert == prev_expert) {
          valid = false;
          break;
        }
      }

      if (valid) {
        token_selected_experts[k * tid + k_idx] = expert;
        break;
      }
    }
  }
}

__global__ void populateRandomBufferKernel(void* buffer_void, size_t size) {
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size) {
    return;
  }

  curandStatePhilox4_32_10_t state;
  curand_init(size, tid, 0, &state);

  constexpr int elem_per_thread = 128 / sizeof(uint4);
  auto* buffer = reinterpret_cast<uint4*>(buffer_void);
#pragma unroll
  for (int i = 0; i < elem_per_thread; i++)
    buffer[tid * elem_per_thread + i] = curand4(&state);
}

template <int BLOCK_SIZE, int NUM_ROUTING_SAMPLES>
__global__ void prepareMinLatencyBuffer(int* num_active_experts_per_node, int* active_expert_global_ids,
                                        int64_t* expert_first_token_offset, int const num_tokens, int const num_experts_per_token,
                                        int const num_experts_per_node) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  // 0. set offset
  num_active_experts_per_node += bid;
  active_expert_global_ids += bid * num_experts_per_node;
  expert_first_token_offset += bid * (num_experts_per_node + 1);

  // 1. set the num_active_experts_per_node
  int num_active = max(1, (int)(bid * ((float)num_experts_per_node / NUM_ROUTING_SAMPLES)));
  *num_active_experts_per_node = num_active;

  // 2. generate random active experts
  extern __shared__ float s_buf[];
  float* expert_refs = s_buf;
  int* expert_refs_idx = reinterpret_cast<int*>(expert_refs + num_experts_per_node);

  curandState_t local_state;
  curand_init(bid, tid, 0, &local_state);
  for (int i = tid; i < num_experts_per_node; i += BLOCK_SIZE) {
    expert_refs[i] = (float)curand_uniform(&local_state);
    expert_refs_idx[i] = (int)i;
  }
  __syncthreads();

  float thread_key[1];
  int thread_value[1];
  thread_key[0] = std::numeric_limits<float>::max();
  thread_value[0] = num_experts_per_node;

  if (tid < num_experts_per_node) {
    thread_key[0] = expert_refs[tid];
    thread_value[0] = expert_refs_idx[tid];
  }

  using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_SIZE, 1, int>;
  using BlockRadixSortValue = cub::BlockRadixSort<int, BLOCK_SIZE, 1>;

  union TempStorage {
    typename BlockRadixSort::TempStorage key_value;
    typename BlockRadixSortValue::TempStorage value;
  };
  __shared__ union TempStorage temp_storage;

  BlockRadixSort(temp_storage.key_value).Sort(thread_key, thread_value);
  __syncthreads();

  if (tid > num_active) {
    thread_value[0] = std::numeric_limits<int>::max();
  }
  BlockRadixSortValue(temp_storage.value).Sort(thread_value);
  __syncthreads();

  // 3. set the active_expert_global_ids and expert_first_token_offset
  for (int i = tid; i < num_experts_per_node; i += BLOCK_SIZE) {
    if (i < num_active) {
      active_expert_global_ids[i] = thread_value[0];
      expert_first_token_offset[i] = i * num_tokens;
    } else {
      active_expert_global_ids[i] = -1;
      expert_first_token_offset[i] = num_active * num_tokens;
    }
  }
  if (tid == 0) {
    expert_first_token_offset[num_experts_per_node] = num_active * num_tokens;
  }
}

void populateRandomBuffer(void* buffer_void, size_t size, cudaStream_t stream) {
  // Each thread initialises 128 bytes
  ORT_ENFORCE(size % 128 == 0, "Unexpected size alignment");
  auto threads = size / 128;
  populateRandomBufferKernel<<<ceilDiv(threads, 128), 128, 0, stream>>>(buffer_void, threads);
}

std::map<std::string, std::pair<size_t, size_t>> GemmProfilerBackend::getProfilerWorkspaces(
    int maxM, bool is_tma_ws_input) {
  size_t k = mK;
  size_t num_expanded_tokens = mMinLatencyMode ? maxM * mNumExpertsPerNode : maxM * k;

  ORT_ENFORCE(mDType != nvinfer::DataType::kINT4);
  // nvllm still uses int64 because torch doesn't have fp4 yet.
  bool is_4bit_act = mDType == nvinfer::DataType::kFP4 || mDType == nvinfer::DataType::kINT64;
  bool is_4bit_weight = mWType == nvinfer::DataType::kINT4 || mWType == nvinfer::DataType::kFP4 || mWType == nvinfer::DataType::kINT64;
  ORT_ENFORCE(!is_4bit_act || is_4bit_weight, "Cannot have 4-bit activation with non-4-bit weight");
  float dtype_bytes = is_4bit_act
                          ? 0.5f
                          : static_cast<float>(mWType == nvinfer::DataType::kINT4 ? getDTypeSize(mOType) : getDTypeSize(mDType));
  float weight_bytes = is_4bit_weight ? 0.5f : static_cast<float>(getDTypeSize(mWType));
  size_t output_bytes = getDTypeSize(mOType);
  size_t gemm_output_bytes = (mOType == nvinfer::DataType::kFP8)
                                 ? sizeof(TmaWarpSpecializedGroupedGemmInput::OutputTypeAdaptor_t<__nv_fp8_e4m3>)
                                 : output_bytes;

  size_t hidden_size = mExpertHiddenSize;
  size_t inter_size = mExpertInterSize;  // Already divided by TP
  size_t num_experts_per_node = mNumExpertsPerNode;

  size_t fc1_out_size = inter_size;
  if (isGatedActivation(mActivationType)) {
    fc1_out_size = inter_size * 2;
  }

  // TODO Needs updated when gather/finalize fusion is integrated
  size_t input_size1 = hidden_size * num_expanded_tokens * dtype_bytes;
  size_t output_size1 = inter_size * num_expanded_tokens * dtype_bytes;

  size_t input_size2 = inter_size * num_expanded_tokens * dtype_bytes;
  size_t output_size2 = hidden_size * output_bytes;

  size_t input_size = mGemmToProfile == GemmToProfile::GEMM_1 ? input_size1 : input_size2;
  size_t output_size = mGemmToProfile == GemmToProfile::GEMM_1 ? output_size1 : output_size2;

  // This may allocate a pointer when not required. That's fine it will be ignored at the cost of some memory
  size_t intermediate_size1 = fc1_out_size * num_expanded_tokens * gemm_output_bytes;  // Note gemm_output_bytes
  size_t intermediate_size2 = hidden_size * num_expanded_tokens * gemm_output_bytes;   // Note gemm_output_bytes

  size_t intermediate_size = mGemmToProfile == GemmToProfile::GEMM_1 ? intermediate_size1 : intermediate_size2;

  size_t weights_1 = hidden_size * fc1_out_size * num_experts_per_node * weight_bytes;
  size_t bias_1 = mBias ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
  if (mUseLora && !is_tma_ws_input)
    bias_1 = output_size1;
  size_t weights_2 = hidden_size * inter_size * num_experts_per_node * weight_bytes;
  size_t bias_2 = mBias ? hidden_size * num_experts_per_node * dtype_bytes : 0;

  size_t weights_size = mNeedWeights ? (mGemmToProfile == GemmToProfile::GEMM_1 ? weights_1 : weights_2) : 0;
  size_t bias_size = mGemmToProfile == GemmToProfile::GEMM_1 ? bias_1 : bias_2;

  // TODO Make quant 2 & 4 bigger for FP8 if we ever change to scaling per expert
  bool is_int_w_quant = (mWType == nvinfer::DataType::kINT8 || mWType == nvinfer::DataType::kINT4) && mGroupSize <= 0;
  bool is_int_groupwise_w_quant = (mWType == nvinfer::DataType::kINT8 || mWType == nvinfer::DataType::kINT4) && mGroupSize > 0;
  bool is_fp8_act_quant = mDType == nvinfer::DataType::kFP8;
  bool is_fp8_w_quant = mWType == nvinfer::DataType::kFP8;
  // nvllm still uses int64 because torch doesn't have fp4 yet.
  // bool is_fp4_act_quant = mDType == nvinfer::DataType::kFP4 || mDType == nvinfer::DataType::kINT64;
  bool is_fp4_w_quant = mWType == nvinfer::DataType::kFP4 || mWType == nvinfer::DataType::kINT64;
  bool is_w4afp8_quant = is_int_groupwise_w_quant && is_fp8_act_quant;
  // bool is_wfp4afp8_quant = is_fp4_w_quant && is_fp8_act_quant;

  // Int sizes
  size_t quant_1_size = is_int_w_quant ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
  size_t quant_2_size = is_int_w_quant ? hidden_size * num_experts_per_node * dtype_bytes : 0;
  if (is_int_w_quant) {
    quant_1_size = fc1_out_size * num_experts_per_node * dtype_bytes;
    quant_2_size = hidden_size * num_experts_per_node * dtype_bytes;
  } else if (is_int_groupwise_w_quant) {
    quant_1_size = fc1_out_size * num_experts_per_node * dtype_bytes * hidden_size / mGroupSize;
    quant_2_size = hidden_size * num_experts_per_node * dtype_bytes * inter_size / mGroupSize;
  }

  // FP8 sizes
  quant_1_size = is_fp8_w_quant ? num_experts_per_node * sizeof(float) : quant_1_size;
  quant_2_size = is_fp8_w_quant ? sizeof(float) : quant_2_size;
  size_t quant_3_size = is_fp8_w_quant ? num_experts_per_node * sizeof(float) : 0;
  size_t quant_4_size = 0;  // Currently ignored by the GEMM
  if (is_int_groupwise_w_quant) {
    quant_3_size = quant_1_size;
    quant_4_size = quant_2_size;
  }

  // FP4 sizes
  quant_1_size = is_fp4_w_quant ? sizeof(float) : quant_1_size;
  quant_2_size = is_fp4_w_quant ? getOffsetWeightSF(num_experts_per_node, inter_size, hidden_size, mScalingType) * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF)
                                : quant_2_size;
  quant_3_size = is_fp4_w_quant ? num_experts_per_node * sizeof(float) : quant_3_size;
  quant_4_size = is_fp4_w_quant ? sizeof(float) : quant_4_size;
  size_t quant_5_size = is_fp4_w_quant
                            ? getOffsetWeightSF(num_experts_per_node, hidden_size, inter_size, mScalingType) * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF)
                            : 0;
  size_t quant_6_size = is_fp4_w_quant ? num_experts_per_node * sizeof(float) : 0;

  size_t tma_ws_input_workspace_size = 0;
  if (is_tma_ws_input) {
    tma_ws_input_workspace_size = TmaWarpSpecializedGroupedGemmInput::workspaceSize(num_experts_per_node, mScalingType) * (NUM_ROUTING_SAMPLES + 1);

    if (is_w4afp8_quant) {
      quant_3_size = 0;
      quant_4_size = 0;
    }
  }

  auto act_sf_rows = mMinLatencyMode
                         ? num_expanded_tokens
                         : std::min(num_expanded_tokens, static_cast<size_t>(maxM * num_experts_per_node));
  // getOffsetActivationSF returns zero if scaling_type is NONE
  size_t const fc1_fp4_act_scale_size = getOffsetActivationSF(num_experts_per_node, act_sf_rows, hidden_size, mScalingType) * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF);
  size_t const fc2_fp4_act_scale_size = getOffsetActivationSF(num_experts_per_node, act_sf_rows, inter_size, mScalingType) * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF);
  size_t const fp4_act_scale_flat_size = std::max(fc1_fp4_act_scale_size, fc2_fp4_act_scale_size);

  size_t w4a8_alpha_size = is_w4afp8_quant ? num_experts_per_node * sizeof(float) : 0;
  size_t alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float**);
  size_t gemm_workspace_size = mInterface->getGemmWorkspaceSize(num_experts_per_node);

  // Routing info
  size_t expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t) * NUM_ROUTING_SAMPLES;
  size_t map_size = mMinLatencyMode ? 0 : NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
  size_t unpermuted_size = mMinLatencyMode ? 0 : NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
  size_t permuted_size = mMinLatencyMode ? 0 : num_expanded_tokens * sizeof(int);
  size_t token_topk_unpermuted_scales_size = mMinLatencyMode ? 0 : num_expanded_tokens * sizeof(float);

  int64_t const num_tokens_per_block = computeNumTokensPerBlock(maxM, num_experts_per_node);
  int64_t const num_blocks_per_seq = onnxruntime::llm::common::ceilDiv(maxM, num_tokens_per_block);
  size_t const blocked_expert_counts_size = mMinLatencyMode ? 0 : num_experts_per_node * num_blocks_per_seq * sizeof(int);
  size_t const blocked_expert_counts_cumsum_size = blocked_expert_counts_size;
  size_t const blocked_row_to_unpermuted_row_size = mMinLatencyMode ? 0 : num_experts_per_node * maxM * sizeof(int);

  // The follow buffers are used in min_latency_mode
  size_t num_active_experts_per_node_size = mMinLatencyMode ? sizeof(int) * NUM_ROUTING_SAMPLES : 0;  // smaller than or equal to num_experts_per_node
  size_t active_expert_global_ids_size = mMinLatencyMode ? mNumExpertsPerNode * sizeof(int) * NUM_ROUTING_SAMPLES : 0;

  size_t map_offset = 0;
  std::map<std::string, std::pair<size_t, size_t>> out_map;

#define ADD_NAME(name, size)                              \
  do {                                                    \
    auto aligned_size = alignSize(size, kCudaMemAlign);   \
    out_map[#name] = std::pair{aligned_size, map_offset}; \
    map_offset += aligned_size;                           \
  } while (false)
#define ADD(name) ADD_NAME(name, name##_size)

  ADD(expert_first_token_offset);
  ADD_NAME(unpermuted_row_to_permuted_row, map_size);
  ADD_NAME(permuted_row_to_unpermuted_row, map_size);
  ADD_NAME(token_selected_experts, unpermuted_size);
  ADD_NAME(permuted_token_selected_experts, permuted_size);
  ADD(blocked_expert_counts);
  ADD(blocked_expert_counts_cumsum);
  ADD(blocked_row_to_unpermuted_row);
  ADD(token_topk_unpermuted_scales);
  ADD(num_active_experts_per_node);
  ADD(active_expert_global_ids);
  ADD(input);
  ADD(output);
  ADD(intermediate);
  ADD(weights);
  ADD(bias);
  ADD(quant_1);
  ADD(quant_2);
  ADD(quant_3);
  ADD(quant_4);
  ADD(quant_5);
  ADD(quant_6);
  ADD(tma_ws_input_workspace);
  ADD(w4a8_alpha);
  ADD(alpha_scale_ptr_array);
  ADD(fp4_act_scale_flat);
  ADD(gemm_workspace);

#undef ADD_NAME
#undef ADD

  return out_map;
}

void GemmProfilerBackend::prepareRouting(int num_tokens, char* workspace_ptr_char, cudaStream_t stream) {
  auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);
#define GET_WS_PTR_BASE(type, name)                                                                                          \
  auto* name##_base = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                                                  : nullptr)
#define GET_WS_PTR(type, name)                                                                                        \
  auto* name = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                                           : nullptr)

  GET_WS_PTR_BASE(int64_t*, expert_first_token_offset);
  GET_WS_PTR_BASE(int*, unpermuted_row_to_permuted_row);
  GET_WS_PTR_BASE(int*, permuted_row_to_unpermuted_row);
  GET_WS_PTR_BASE(int*, token_selected_experts);
  GET_WS_PTR(int*, permuted_token_selected_experts);
  GET_WS_PTR(int*, blocked_expert_counts);
  GET_WS_PTR(int*, blocked_expert_counts_cumsum);
  GET_WS_PTR(int*, blocked_row_to_unpermuted_row);
  GET_WS_PTR(int*, num_active_experts_per_node);
  GET_WS_PTR(int*, active_expert_global_ids);

#undef GET_WS_PTR_BASE
#undef GET_WS_PTR

  if (mMinLatencyMode) {
    // expert_first_token_offset for each sample
    ORT_ENFORCE(mNumExpertsPerNode <= 256, "Min latency mode only supports #experts < 256");
    prepareMinLatencyBuffer<256, NUM_ROUTING_SAMPLES>
        <<<NUM_ROUTING_SAMPLES, 256, mNumExpertsPerNode * (sizeof(float) + sizeof(int)), stream>>>(
            num_active_experts_per_node, active_expert_global_ids, expert_first_token_offset_base, num_tokens, mK,
            mNumExpertsPerNode);
  } else {
    int64_t const num_expanded_tokens = num_tokens * mK;
    int const start_expert_id = mNumExpertsPerNode * mParallelismConfig.ep_rank;

    uint32_t num_threads = 256;
    dim3 grid_dim{(num_tokens + num_threads - 1) / num_threads, NUM_ROUTING_SAMPLES, 1};
    prepareFakeRouterBuffers<<<grid_dim, num_threads, 0, stream>>>(
        token_selected_experts_base, num_tokens, mK, mNumExperts);
    sync_check_cuda_error(stream);

    for (int64_t i = 0; i < NUM_ROUTING_SAMPLES; i++) {
      int64_t* expert_first_token_offset = expert_first_token_offset_base + i * (mNumExpertsPerNode + 1);
      int* unpermuted_row_to_permuted_row = unpermuted_row_to_permuted_row_base + i * num_expanded_tokens;
      int* permuted_row_to_unpermuted_row = permuted_row_to_unpermuted_row_base + i * num_expanded_tokens;
      int* token_selected_experts = token_selected_experts_base + i * num_expanded_tokens;

      threeStepBuildExpertMapsSortFirstToken(token_selected_experts, permuted_token_selected_experts,
                                             permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row, expert_first_token_offset,
                                             blocked_expert_counts, blocked_expert_counts_cumsum, blocked_row_to_unpermuted_row, num_tokens,
                                             mNumExpertsPerNode, mK, start_expert_id, stream);
      sync_check_cuda_error(stream);
    }
  }
}

void GemmProfilerBackend::prepareQuantParams(int num_tokens, char* workspace_ptr_char, cudaStream_t) {
  auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);
#define GET_WS_PTR(type, name)                                                                                        \
  auto* name = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                                           : nullptr)
  GET_WS_PTR(void const*, quant_1);
  GET_WS_PTR(void const*, quant_2);
  GET_WS_PTR(void const*, quant_3);
  GET_WS_PTR(void const*, quant_4);
  GET_WS_PTR(void const*, quant_5);
  GET_WS_PTR(void const*, quant_6);
  GET_WS_PTR(float const*, w4a8_alpha);
#undef GET_WS_PTR

  if ((mWType == nvinfer::DataType::kINT8 || mWType == nvinfer::DataType::kINT4) && mGroupSize < 0) {
    ORT_ENFORCE(quant_1 && quant_2);
    mQuantParams = QuantParams::Int(quant_1, quant_2);
  } else if (mWType == nvinfer::DataType::kINT4) {
    ORT_ENFORCE(quant_1 && quant_2);
    if (mDType == nvinfer::DataType::kFP8) {
      ORT_ENFORCE(w4a8_alpha);
      mQuantParams = QuantParams::GroupWise(
          mGroupSize, quant_1, quant_2, nullptr, nullptr, quant_3, quant_4, w4a8_alpha, w4a8_alpha);
    } else {
      mQuantParams = QuantParams::GroupWise(mGroupSize, quant_1, quant_2, nullptr, nullptr, quant_3, quant_4);
    }
  } else if (mWType == nvinfer::DataType::kFP8) {
    ORT_ENFORCE(quant_1 && quant_2 && quant_3);
    mQuantParams = QuantParams::FP8(static_cast<float const*>(quant_1), static_cast<float const*>(quant_2),
                                    static_cast<float const*>(quant_3), static_cast<float const*>(quant_4));
  } else if (mDType == nvinfer::DataType::kFP8 && (mWType == nvinfer::DataType::kFP4 || mWType == nvinfer::DataType::kINT64)) {
    ORT_ENFORCE(quant_1 && quant_2 && quant_3 && quant_4 && quant_5 && quant_6);
    mQuantParams = QuantParams::FP8MXFP4(static_cast<float const*>(quant_1),
                                         static_cast<TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const*>(quant_2),
                                         static_cast<float const*>(quant_3), static_cast<float const*>(quant_4),
                                         static_cast<TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const*>(quant_5),
                                         static_cast<float const*>(quant_6));
  } else if ((mDType == nvinfer::DataType::kFP4 || mDType == nvinfer::DataType::kINT64) && (mWType == nvinfer::DataType::kFP4 || mWType == nvinfer::DataType::kINT64)) {
    // nvllm still uses int64 because torch doesn't have fp4 yet.
    ORT_ENFORCE(quant_1 && quant_2 && quant_3 && quant_4 && quant_5 && quant_6);
    mQuantParams = QuantParams::FP4(static_cast<float const*>(quant_1),
                                    static_cast<TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const*>(quant_2),
                                    static_cast<float const*>(quant_3), static_cast<float const*>(quant_4),
                                    static_cast<TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const*>(quant_5),
                                    static_cast<float const*>(quant_6));
  }
}

void GemmProfilerBackend::prepareTmaWsInputs(
    int num_tokens, char* workspace_ptr_char, void const* expert_weights, cudaStream_t stream) {
  if (mSM < 90) {
    return;
  }

  auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);

#define GET_WS_PTR(type, name)                                                                                        \
  auto* name = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                                           : nullptr)

  GET_WS_PTR(int64_t*, expert_first_token_offset);
  int64_t* expert_first_token_offset_base = expert_first_token_offset;
  GET_WS_PTR(int*, permuted_row_to_unpermuted_row);
  int* permuted_row_to_unpermuted_row_base = permuted_row_to_unpermuted_row;
  GET_WS_PTR(void*, input);
  GET_WS_PTR(void*, output);
  GET_WS_PTR(void*, intermediate);
  GET_WS_PTR(void*, weights);
  ORT_ENFORCE(mNeedWeights == (expert_weights == nullptr));
  void const* weights_sel = mNeedWeights ? weights : expert_weights;
  GET_WS_PTR(void*, bias);
  GET_WS_PTR(float*, token_topk_unpermuted_scales);
  GET_WS_PTR(int8_t*, tma_ws_input_workspace);
  GET_WS_PTR(void*, gemm_workspace);
  GET_WS_PTR(float*, alpha_scale_ptr_array);
  GET_WS_PTR(TmaWarpSpecializedGroupedGemmInput::ElementSF*, fp4_act_scale_flat);
  GET_WS_PTR(int*, num_active_experts_per_node);
  GET_WS_PTR(int*, active_expert_global_ids);

#undef GET_WS_PTR

  size_t tma_ws_size = TmaWarpSpecializedGroupedGemmInput::workspaceSize(mNumExpertsPerNode, mScalingType);

  TmaWarpSpecializedGroupedGemmInput dummy_tma_ws_input;
  dummy_tma_ws_input.configureWorkspace(tma_ws_input_workspace, mNumExpertsPerNode, gemm_workspace,
                                        workspaces.at("gemm_workspace").first, mScalingType);
  tma_ws_input_workspace += tma_ws_size;

  size_t num_expanded_tokens = num_tokens * mK;
  for (int64_t i = 0; i < NUM_ROUTING_SAMPLES; i++) {
    mTmaInputCache[i].configureWorkspace(tma_ws_input_workspace, mNumExpertsPerNode, gemm_workspace,
                                         workspaces.at("gemm_workspace").first, mScalingType);
    tma_ws_input_workspace += tma_ws_size;

    int64_t* expert_first_token_offset = expert_first_token_offset_base + i * (mNumExpertsPerNode + 1);
    int* permuted_row_to_unpermuted_row = permuted_row_to_unpermuted_row_base + i * num_expanded_tokens;

    auto& gemm1_tma_ws_input = mGemmToProfile == GemmToProfile::GEMM_1 ? mTmaInputCache[i] : dummy_tma_ws_input;
    auto& gemm2_tma_ws_input = mGemmToProfile == GemmToProfile::GEMM_2 ? mTmaInputCache[i] : dummy_tma_ws_input;
    if (mSM >= 90) {
      /* GEMM1 */
      gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
      gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;

      bool apply_bias = true;
      bool use_w4afp8 = (mDType == nvinfer::DataType::kFP8 && mWType == nvinfer::DataType::kINT4);
      bool using_fused_finalize = !mInterface->use_deterministic_hopper_reduce_ && mSM == 90 && !mMinLatencyMode && !use_w4afp8;
      if (using_fused_finalize) {
        assert(!mMinLatencyMode);
        gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
        gemm2_tma_ws_input.setFinalizeFusionParams(output, token_topk_unpermuted_scales,
                                                   expert_first_token_offset, permuted_row_to_unpermuted_row, apply_bias ? bias : nullptr,
                                                   mExpertHiddenSize, num_tokens);
      }

      auto fc1_output_size = isGatedActivation(mActivationType) ? mExpertInterSize * 2 : mExpertInterSize;
      if (mMinLatencyMode) {
        std::tie(gemm1_tma_ws_input, gemm2_tma_ws_input) = mInterface->computeStridesTmaWarpSpecializedLowLatencyDispatch(gemm1_tma_ws_input,
                                                                                                                          gemm2_tma_ws_input, num_tokens, fc1_output_size, mExpertHiddenSize, mExpertHiddenSize,
                                                                                                                          mExpertInterSize, mNumExpertsPerNode, input, input, weights_sel, weights_sel,
                                                                                                                          mQuantParams.fp8.dequant_fc1, mQuantParams.fp8.dequant_fc2, fp4_act_scale_flat,
                                                                                                                          fp4_act_scale_flat, mQuantParams, nullptr, nullptr, intermediate, intermediate,
                                                                                                                          num_active_experts_per_node, active_expert_global_ids, 0, stream);
      } else {
        std::tie(gemm1_tma_ws_input, gemm2_tma_ws_input) = mInterface->computeStridesTmaWarpSpecializedDispatch(
            expert_first_token_offset, gemm1_tma_ws_input, gemm2_tma_ws_input, num_tokens, num_tokens * mK,
            fc1_output_size, mExpertHiddenSize, mExpertHiddenSize, mExpertInterSize, mNumExpertsPerNode, input,
            input, weights_sel, weights_sel, mQuantParams.fp8.dequant_fc1, mQuantParams.fp8.dequant_fc2,
            fp4_act_scale_flat, fp4_act_scale_flat, mQuantParams, nullptr, nullptr, intermediate, intermediate,
            stream);
      }
      sync_check_cuda_error(stream);
    }
  }
}

void GemmProfilerBackend::prepare(
    int num_tokens, char* workspace_ptr_char, void const* expert_weights, cudaStream_t stream) {
  mAllTacticsSaved = mInterface->getTactics();
  mSampleIndex = 0;

  auto workspace_size = getWorkspaceSize(num_tokens);
  populateRandomBuffer(workspace_ptr_char, workspace_size, stream);

  prepareRouting(num_tokens, workspace_ptr_char, stream);
  prepareQuantParams(num_tokens, workspace_ptr_char, stream);
  prepareTmaWsInputs(num_tokens, workspace_ptr_char, expert_weights, stream);
}

size_t GemmProfilerBackend::getWorkspaceSize(int maxM) {
  auto sizes_map = getProfilerWorkspaces(maxM, mSM >= 90);
  std::vector<size_t> sizes(sizes_map.size());
  std::transform(sizes_map.begin(), sizes_map.end(), sizes.begin(), [](auto& v) { return v.second.first; });
  size_t size = calculateTotalWorkspaceSize(sizes.data(), sizes.size());
  ORT_LLM_LOG_TRACE(onnxruntime::MakeString("MOE profiler workspace size: ", size));
  return size;
}

void GemmProfilerBackend::runProfiler(int original_num_tokens, Config const& tactic, char* workspace_ptr_char,
                                      void const* expert_weights, cudaStream_t const& stream) {
  int64_t expanded_num_tokens = original_num_tokens * mK;
  int64_t num_experts_per_node = mNumExpertsPerNode;

  mSampleIndex = (mSampleIndex + 1) % NUM_ROUTING_SAMPLES;

  auto workspaces = getProfilerWorkspaces(original_num_tokens, tactic.is_tma_warp_specialized);

#define GET_WS_PTR_OFFSET(type, name, offset)                                                             \
  auto* name = (workspaces.at(#name).first                                                                \
                    ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) + (offset) \
                    : nullptr)
#define GET_WS_PTR(type, name)                                                                                        \
  auto* name = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                                           : nullptr)

  GET_WS_PTR_OFFSET(int64_t const*, expert_first_token_offset, (mSampleIndex * (mNumExpertsPerNode + 1)));
  GET_WS_PTR_OFFSET(int const*, unpermuted_row_to_permuted_row, (mSampleIndex * expanded_num_tokens));
  GET_WS_PTR_OFFSET(int const*, permuted_row_to_unpermuted_row, (mSampleIndex * expanded_num_tokens));
  GET_WS_PTR_OFFSET(int const*, token_selected_experts, (mSampleIndex * expanded_num_tokens));

  GET_WS_PTR(float const*, token_topk_unpermuted_scales);
  auto const* token_topk_permuted_scales = token_topk_unpermuted_scales;

  GET_WS_PTR_OFFSET(int*, num_active_experts_per_node, mSampleIndex);
  GET_WS_PTR_OFFSET(int*, active_expert_global_ids, (mSampleIndex * mNumExpertsPerNode));
  GET_WS_PTR(void const*, input);
  GET_WS_PTR(void*, output);
  GET_WS_PTR(void*, intermediate);
  GET_WS_PTR(void const*, weights);
  ORT_ENFORCE(mNeedWeights == (expert_weights == nullptr));
  void const* weights_sel = mNeedWeights ? weights : expert_weights;
  GET_WS_PTR(void const*, bias);

  GET_WS_PTR(float const**, alpha_scale_ptr_array);
  GET_WS_PTR(TmaWarpSpecializedGroupedGemmInput::ElementSF*, fp4_act_scale_flat);
  GET_WS_PTR(void*, gemm_workspace);

#undef GET_WS_PTR_OFFSET
#undef GET_WS_PTR

  TmaWarpSpecializedGroupedGemmInput tma_ws_input_template;
  if (tactic.is_tma_warp_specialized) {
    tma_ws_input_template = mTmaInputCache[mSampleIndex];
  }

  mInterface->is_profiler = true;
  if (mGemmToProfile == GemmToProfile::GEMM_1) {
    mInterface->gemm1(input,                                             //
                      output,                                            //
                      intermediate,                                      //
                      expert_first_token_offset,                         //
                      tma_ws_input_template,                             //
                      weights_sel,                                       //
                      bias,                                              //
                      expert_first_token_offset + num_experts_per_node,  //
                      mQuantParams.wo.fc1_weight_scales,                 //
                      mQuantParams.fp8.dequant_fc1,                      //
                      mQuantParams.fp8_mxfp4.fc2.act_global_scale ? mQuantParams.fp8_mxfp4.fc2.act_global_scale
                                                                  : mQuantParams.fp8.quant_fc2,  //
                      fp4_act_scale_flat,                                                        //
                      fp4_act_scale_flat,                                                        //
                      mQuantParams,                                                              //
                      original_num_tokens,                                                       //
                      expanded_num_tokens,                                                       //
                      mExpertHiddenSize,                                                         //
                      mExpertInterSize,                                                          //
                      num_experts_per_node,                                                      //
                      mActivationType,                                                           //
                      alpha_scale_ptr_array,                                                     //
                      !mUseLora,                                                                 //
                      /*use_deepseek_fp8_block_scale=*/false,                                    //
                      stream,                                                                    //
                      tactic,                                                                    //
                      mMinLatencyMode,                                                           //
                      num_active_experts_per_node,                                               //
                      active_expert_global_ids);                                                 //
  } else {
    ORT_ENFORCE(mGemmToProfile == GemmToProfile::GEMM_2);
    mInterface->gemm2(input,                                           //
                      intermediate,                                    //
                      output,                                          //
                      expert_first_token_offset,                       //
                      tma_ws_input_template,                           //
                      weights_sel,                                     //
                      bias,                                            //
                      mQuantParams.wo.fc2_weight_scales,               //
                      mQuantParams.fp8.dequant_fc2,                    //
                      fp4_act_scale_flat,                              //
                      mQuantParams,                                    //
                      token_topk_unpermuted_scales,                    //
                      token_topk_permuted_scales,                      //
                      unpermuted_row_to_permuted_row,                  //
                      permuted_row_to_unpermuted_row,                  //
                      token_selected_experts,                          //
                      expert_first_token_offset + mNumExpertsPerNode,  //
                      original_num_tokens,                             //
                      expanded_num_tokens,                             //
                      mExpertHiddenSize,                               //
                      mExpertInterSize,                                //
                      num_experts_per_node,                            //
                      mK,                                              //
                      alpha_scale_ptr_array,                           //
                      false,                                           //
                      nullptr,                                         //
                      /*use_deepseek_fp8_block_scale=*/false,          //
                      stream,                                          //
                      mParallelismConfig,                              //
                      mEnableAlltoall,                                 //
                      tactic,                                          //
                      mMinLatencyMode,                                 //
                      num_active_experts_per_node,                     //
                      active_expert_global_ids);                       //
  }
  mInterface->is_profiler = false;

  sync_check_cuda_error(stream);
}

// ==================== Variable batched GEMM specializations ==================================
template class CutlassMoeFCRunner<float, float>;

#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_bfloat16, uint8_t>;
template class CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>;
#endif

template class CutlassMoeFCRunner<half, half>;
template class CutlassMoeFCRunner<half, uint8_t>;
template class CutlassMoeFCRunner<half, cutlass::uint4b_t>;
#ifdef ENABLE_FP8
// template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, half>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>;
#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>;
#endif
#endif
#ifdef ENABLE_FP4
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, half>;
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, half, half>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half, half>;
#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>;
#endif
#endif

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
