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

#include "contrib_ops/cuda/llm/moe_gemm/moe_expert_maps.h"

#include <array>
#include <cassert>
#include <cmath>

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/cub.cuh"
#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/common/env_utils.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"

using namespace onnxruntime::llm::common;

namespace onnxruntime::llm::kernels::cutlass_kernels {

// Ranker used to group the (token, top-k slot) pairs by selected expert.
//
// cub::BlockRadixRank keeps a private set of packed digit counters per thread, so its
// shared memory and its per-thread scan both grow with 2^LOG2_NUM_EXPERTS. At 256
// experts (LOG2_NUM_EXPERTS == 9) and a decode-sized block of 32 threads that is 512
// bins spread over a single warp, which costs ~16KB of shared memory, 255 registers per
// thread and enough spilling that the kernel is dominated by counter bookkeeping rather
// than by the handful of assignments it actually ranks. The match-based ranker ranks
// with warp ballots instead of per-thread counters, so its cost tracks the number of
// keys rather than the number of bins.
template <int BLOCK_SIZE, int LOG2_NUM_EXPERTS>
using MoeExpertRadixRank = cub::BlockRadixRankMatch<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;

template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
__global__ void fusedBuildExpertMapsSortFirstTokenKernel(const int* const token_selected_experts,
                                                         int* const permuted_row_to_unpermuted_row, int* const unpermuted_row_to_permuted_row,
                                                         int* const permuted_token_selected_experts,
                                                         int64_t* const expert_first_token_offset, const int64_t num_tokens, const int experts_per_token,
                                                         const int start_expert, const int end_expert, const int num_experts_per_node) {
  // Only using block wise collective so we can only have one block
  assert(gridDim.x == 1);

  assert(start_expert <= end_expert);
  assert(num_experts_per_node == (end_expert - start_expert));
  assert(num_experts_per_node <= (1 << LOG2_NUM_EXPERTS));

  const int token = blockIdx.x * BLOCK_SIZE + threadIdx.x;

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
    const int expert = is_valid_token ? token_selected_experts[token * EXPERTS_PER_TOKEN + i] : num_experts_per_node;

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
  using BlockRadixRank = MoeExpertRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS>;
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
      const int unpermuted_row = i * num_tokens + token;
      const int permuted_row = local_token_permuted_indices[i];
      permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
      unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
      permuted_token_selected_experts[permuted_row] = local_token_selected_experts[i];
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
bool fusedBuildExpertMapsSortFirstTokenDispatch(const int* token_selected_experts, int* permuted_row_to_unpermuted_row,
                                                int* unpermuted_row_to_permuted_row, int* permuted_token_selected_experts,
                                                int64_t* expert_first_token_offset, const int64_t num_tokens,
                                                const int num_experts_per_node, const int experts_per_token, const int start_expert, const int end_expert,
                                                cudaStream_t stream) {
  ORT_ENFORCE(num_experts_per_node == (end_expert - start_expert),
              "num_experts_per_node must be equal to end_expert - start_expert");
  const int threads = BLOCK_SIZE;
  const int blocks = (num_tokens + threads - 1) / threads;
  ORT_ENFORCE(blocks == 1, "Current implementation requires single block");

  using BlockRadixRank = MoeExpertRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS>;
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
                                     unpermuted_row_to_permuted_row, permuted_token_selected_experts,
                                     expert_first_token_offset, num_tokens, experts_per_token, start_expert, end_expert,
                                     num_experts_per_node));

  return true;
}

template <int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenBlockSize(const int* token_selected_experts, int* permuted_row_to_unpermuted_row,
                                                 int* unpermuted_row_to_permuted_row, int* permuted_token_selected_experts,
                                                 int64_t* expert_first_token_offset, const int64_t num_tokens,
                                                 const int num_experts_per_node, const int experts_per_token, const int start_expert, const int end_expert,
                                                 cudaStream_t stream) {
  const int block_size = num_tokens;
  if (num_tokens > 256) {
    ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("Number of tokens ", num_tokens, " is greater than 256, which is not supported for fused moe prologues"));
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
              permuted_token_selected_experts, expert_first_token_offset, num_tokens, num_experts_per_node,
              experts_per_token, start_expert, end_expert, stream);
}

template <int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenBlockSize(const int* token_selected_experts, int* permuted_row_to_unpermuted_row,
                                                 int* unpermuted_row_to_permuted_row, int* permuted_token_selected_experts,
                                                 int64_t* expert_first_token_offset, const int64_t num_tokens,
                                                 const int num_experts_per_node, const int experts_per_token, const int start_expert, const int end_expert,
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
    case 10: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<10, LOG2_NUM_EXPERTS>;
      break;
    }
    default: {
      ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("Top-K value ", experts_per_token, " does not have supported fused moe prologues"));
      return false;
    }
  }
  return func(token_selected_experts, permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row,
              permuted_token_selected_experts, expert_first_token_offset, num_tokens, num_experts_per_node,
              experts_per_token, start_expert, end_expert, stream);
}

bool fusedBuildExpertMapsSortFirstToken(const int* token_selected_experts, int* permuted_row_to_unpermuted_row,
                                        int* unpermuted_row_to_permuted_row, int* permuted_token_selected_experts,
                                        int64_t* expert_first_token_offset, const int64_t num_tokens,
                                        const int num_experts_per_node, const int experts_per_token, const int start_expert, const int end_expert,
                                        cudaStream_t stream) {
  // We need enough bits to represent [0, num_experts_per_node+1] (inclusive) i.e. num_experts_per_node + 2 values
  // This is floor(log2(num_experts_per_node+1)) + 1
  int expert_log = static_cast<int>(log2(num_experts_per_node + 1)) + 1;
  if (expert_log <= 10) {
    auto funcs = std::array{&fusedBuildExpertMapsSortFirstTokenBlockSize<1>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<2>, &fusedBuildExpertMapsSortFirstTokenBlockSize<3>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<4>, &fusedBuildExpertMapsSortFirstTokenBlockSize<5>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<6>, &fusedBuildExpertMapsSortFirstTokenBlockSize<7>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<8>, &fusedBuildExpertMapsSortFirstTokenBlockSize<9>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<10>};

    return funcs[expert_log - 1](token_selected_experts, permuted_row_to_unpermuted_row,
                                 unpermuted_row_to_permuted_row, permuted_token_selected_experts,
                                 expert_first_token_offset, num_tokens, num_experts_per_node, experts_per_token,
                                 start_expert, end_expert, stream);
  }
  ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("Experts per node ", num_experts_per_node, " does not have supported fused moe prologues"));
  return false;
}

int64_t computeNumTokensPerBlock(const int64_t num_tokens, const int64_t num_experts_per_node) {
  for (int64_t num_tokens_per_block = 32; num_tokens_per_block <= 1024; num_tokens_per_block *= 2) {
    const int64_t num_blocks_per_seq = onnxruntime::llm::common::ceilDiv(num_tokens, num_tokens_per_block);
    if (num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block) {
      return num_tokens_per_block;
    }
  }
  return 1024;
}

template <int kNumTokensPerBlock>
__global__ void blockExpertPrefixSumKernel(const int* token_selected_experts, int* blocked_expert_counts,
                                           int* blocked_row_to_unpermuted_row, const int64_t num_tokens, const int64_t num_experts_per_token,
                                           const int start_expert_id) {
  using BlockScan = cub::BlockScan<int, kNumTokensPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // target_expert_id and expert_id are offset by start_expert_id
  const int target_expert_id = blockIdx.x;
  const int block_id = blockIdx.y;
  const int num_blocks_per_seq = gridDim.y;
  const int token_id = block_id * kNumTokensPerBlock + threadIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  int expanded_token_id = -1;
  if (token_id < num_tokens) {
    for (int i = 0; i < num_experts_per_token; i++) {
      // TODO(enweiz): Fix uncoalesced access with shared memory.
      const int expert_id = token_selected_experts[token_id * num_experts_per_token + i] - start_expert_id;
      if (expert_id == target_expert_id) {
        expanded_token_id = i * num_tokens + token_id;
        break;
      }
    }
  }

  const int has_matched = expanded_token_id >= 0 ? 1 : 0;
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

void blockExpertPrefixSum(const int* token_selected_experts, int* blocked_expert_counts,
                          int* blocked_row_to_unpermuted_row, const int64_t num_tokens, const int64_t num_experts_per_node,
                          const int64_t num_experts_per_token, const int64_t num_tokens_per_block, const int64_t num_blocks_per_seq,
                          const int start_expert_id, cudaStream_t stream) {
  const dim3 blocks(num_experts_per_node, num_blocks_per_seq);
  const dim3 threads(num_tokens_per_block);

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
__global__ void globalExpertPrefixSumLargeKernel(const int* blocked_expert_counts, int* blocked_expert_counts_cumsum,
                                                 int64_t* expert_first_token_offset, const int64_t num_experts_per_node, const int64_t num_blocks_per_seq,
                                                 const int64_t num_elem_per_thread) {
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
__global__ void globalExpertPrefixSumKernel(const int* blocked_expert_counts, int* blocked_expert_counts_cumsum,
                                            int64_t* expert_first_token_offset, const int64_t num_experts_per_node, const int64_t num_blocks_per_seq) {
  using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  const int cnt = threadIdx.x < num_experts_per_node * num_blocks_per_seq ? blocked_expert_counts[threadIdx.x] : 0;
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

void globalExpertPrefixSum(const int* blocked_expert_counts, int* blocked_expert_counts_cumsum,
                           int64_t* expert_first_token_offset, const int64_t num_experts_per_node, const int64_t num_tokens_per_block,
                           const int64_t num_blocks_per_seq, cudaStream_t stream) {
  const int64_t num_elements = num_experts_per_node * num_blocks_per_seq;

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
    const int64_t num_elem_per_thread = onnxruntime::llm::common::ceilDiv(num_elements, 1024);
    cudaLaunchKernelEx(&config, func, blocked_expert_counts, blocked_expert_counts_cumsum,
                       expert_first_token_offset, num_experts_per_node, num_blocks_per_seq, num_elem_per_thread);
  }
}

__global__ void mergeExpertPrefixSumKernel(const int* blocked_expert_counts, const int* blocked_expert_counts_cumsum,
                                           const int* blocked_row_to_unpermuted_row, int* permuted_token_selected_experts, int* permuted_row_to_unpermuted_row,
                                           int* unpermuted_row_to_permuted_row, const int num_tokens) {
  const int target_expert_id = blockIdx.x;
  const int block_id = blockIdx.y;
  const int num_blocks_per_seq = gridDim.y;
  const int token_id = block_id * blockDim.x + threadIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  const int cnt = blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id];
  const int offset = blocked_expert_counts_cumsum[target_expert_id * num_blocks_per_seq + block_id];
  if (threadIdx.x < cnt) {
    const int unpermuted_row = blocked_row_to_unpermuted_row[target_expert_id * num_tokens + token_id];
    const int permuted_row = offset + threadIdx.x;
    permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
    permuted_token_selected_experts[permuted_row] = target_expert_id;
    unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void mergeExpertPrefixSum(const int* blocked_expert_counts, const int* blocked_expert_counts_cumsum,
                          const int* blocked_row_to_unpermuted_row, int* permuted_token_selected_experts, int* permuted_row_to_unpermuted_row,
                          int* unpermuted_row_to_permuted_row, const int64_t num_tokens, const int64_t num_experts_per_node,
                          const int64_t num_tokens_per_block, const int64_t num_blocks_per_seq, cudaStream_t stream) {
  const dim3 blocks(num_experts_per_node, num_blocks_per_seq);
  const dim3 threads(num_tokens_per_block);

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
                     unpermuted_row_to_permuted_row, static_cast<int>(num_tokens));
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

void threeStepBuildExpertMapsSortFirstToken(const int* token_selected_experts, int* permuted_token_selected_experts,
                                            int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
                                            int* blocked_expert_counts, int* blocked_expert_counts_cumsum, int* blocked_row_to_unpermuted_row,
                                            const int64_t num_tokens, const int64_t num_experts_per_node, const int64_t num_experts_per_token,
                                            const int start_expert_id, cudaStream_t stream) {
  const int64_t num_tokens_per_block = computeNumTokensPerBlock(num_tokens, num_experts_per_node);
  const int64_t num_blocks_per_seq = onnxruntime::llm::common::ceilDiv(num_tokens, num_tokens_per_block);

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

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
