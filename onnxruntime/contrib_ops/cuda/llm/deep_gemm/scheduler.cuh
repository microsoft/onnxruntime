/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/MIT
 *
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVRTC_JIT_COMPILATION
#include <cstdint>
#endif

#include "utils.cuh"

namespace deep_gemm {

enum class GemmType {
  Normal,
  GroupedContiguous,
  GroupedMasked,
  GroupedWithOffset,
  StridedBatched
};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
#endif

template <uint32_t kNumTMAMulticast, uint32_t kNumNBlocks, uint32_t kNumNBlocksPerGroup>
__device__ __forceinline__ void get_swizzled_block_idx(
    const uint32_t num_m_blocks, int block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
  DG_STATIC_ASSERT(kNumNBlocksPerGroup % kNumTMAMulticast == 0, "Invalid group size");

  // Swizzle for better L2 usages
  auto num_blocks_per_group = num_m_blocks * kNumNBlocksPerGroup;
  auto group_idx = block_idx / num_blocks_per_group;
  auto first_n_block_idx = group_idx * kNumNBlocksPerGroup;
  auto num_n_blocks_in_group = min(kNumNBlocksPerGroup, kNumNBlocks - first_n_block_idx);
  auto in_group_idx = block_idx % num_blocks_per_group;
  m_block_idx = in_group_idx / num_n_blocks_in_group;
  n_block_idx = first_n_block_idx + in_group_idx % num_n_blocks_in_group;
}

struct NormalSchedulerInput {
  uint32_t shape_m;
  int* grouped_layout;  // no use
};

struct NormalSchedulerInputSwapAB {
  uint32_t shape_n;
  int* grouped_layout;  // no use
};

struct GroupedContiguousSchedulerInput {
  uint32_t shape_m;
  int* grouped_layout;
};

struct GroupedMaskedSchedulerInput {
  uint32_t shape_m;
  int* grouped_layout;
};

struct GroupedWithOffsetSchedulerInput {
  uint32_t shape_m;
  int64_t* problem_m_offsets;
};

struct GroupedWithOffsetSchedulerInputSwapAB {
  uint32_t shape_m;
  int64_t* problem_n_offsets;
};

struct StridedBatchedSchedulerInput {
  uint32_t shape_m;
  uint64_t ld_a;
  uint64_t stride_a;
  uint64_t ld_b;
  uint64_t stride_b;
  uint64_t ld_d;
  uint64_t stride_d;
};

struct StridedBatchedSchedulerInputSwapAB {
  uint32_t shape_n;
  uint64_t ld_a;
  uint64_t stride_a;
  uint64_t ld_b;
  uint64_t stride_b;
  uint64_t ld_d;
  uint64_t stride_d;
};

template <uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumGroups, uint32_t kNumTMAMulticast,
          uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N), uint32_t kNumNBlocksPerGroup = 16>
struct NormalScheduler {
  static constexpr GemmType gemm_type = GemmType::Normal;

  int current_iter = -1;
  uint32_t num_aligned_m_blocks;
  uint32_t num_blocks;

  using Input = NormalSchedulerInput;
  Input input;

  NormalScheduler() {}

  __device__ __forceinline__ NormalScheduler(Input& input) {
    num_aligned_m_blocks = ceil_div(input.shape_m, BLOCK_M);
    num_blocks = num_aligned_m_blocks * kNumNBlocks;
  }

  __device__ __forceinline__ uint32_t get_global_m_idx(uint32_t const& block_idx) {
    return block_idx * BLOCK_M;
  }

  __device__ __forceinline__ uint32_t get_global_n_idx(
      uint32_t const shape_dim, uint32_t const block_size, uint32_t const& block_idx, uint32_t const& m_block_idx = 0) {
    return block_idx * block_size;
  }

  __device__ __forceinline__ uint32_t get_global_scales_a_idx(uint32_t const& block_idx) {
    return block_idx;
  }

  __device__ __forceinline__ uint32_t get_global_scales_b_idx(
      uint32_t const shape_dim, uint32_t const block_size, uint32_t const& block_idx, uint32_t const& m_block_idx = 0) {
    return block_idx * block_size;
  }

  __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    ++current_iter;
    auto const next_block_idx = current_iter * gridDim.x + blockIdx.x;
    if (next_block_idx >= num_blocks) {
      return false;
    }
    get_swizzled_block_idx<kNumTMAMulticast, kNumNBlocks, kNumNBlocksPerGroup>(
        num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx);
    return true;
  }
};

template <uint32_t SHAPE_M, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumGroups, uint32_t kNumTMAMulticast,
          uint32_t kNumMBlocks = ceil_div(SHAPE_M, BLOCK_M), uint32_t kNumMBlocksPerGroup = 16>
struct NormalSchedulerSwapAB {
  static constexpr GemmType gemm_type = GemmType::Normal;

  int current_iter = -1;
  uint32_t num_aligned_n_blocks;
  uint32_t num_blocks;

  using Input = NormalSchedulerInputSwapAB;
  Input input;

  NormalSchedulerSwapAB() {}

  __device__ __forceinline__ NormalSchedulerSwapAB(Input& input) {
    num_aligned_n_blocks = ceil_div(input.shape_n, BLOCK_N);
    num_blocks = num_aligned_n_blocks * kNumMBlocks;
  }

  // weight
  __device__ __forceinline__ uint32_t get_global_m_idx(
      const uint32_t shape_dim, const uint32_t block_size, uint32_t const& block_idx, uint32_t const& n_block_idx = 0) {
    return block_idx * block_size;
  }

  // act
  __device__ __forceinline__ uint32_t get_global_n_idx(uint32_t const& block_idx) {
    return block_idx * BLOCK_N;
  }

  // act scales
  __device__ __forceinline__ uint32_t get_global_scales_b_idx(uint32_t const& block_idx) {
    return block_idx;
  }

  // weight scales
  __device__ __forceinline__ uint32_t get_global_scales_a_idx(
      const uint32_t shape_dim, const uint32_t block_size, uint32_t const& block_idx, uint32_t const& n_block_idx = 0) {
    return block_idx * block_size;
  }

  __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    ++current_iter;
    auto const next_block_idx = current_iter * gridDim.x + blockIdx.x;
    if (next_block_idx >= num_blocks) {
      return false;
    }

    get_swizzled_block_idx<kNumTMAMulticast, kNumMBlocks, kNumMBlocksPerGroup>(
        num_aligned_n_blocks, next_block_idx, n_block_idx, m_block_idx);
    return true;
  }
};

template <uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumGroups, uint32_t kNumTMAMulticast,
          uint32_t kNumNBlocks, uint32_t kNumNBlocksPerGroup>
struct GroupedContiguousScheduler {
  static constexpr GemmType gemm_type = GemmType::GroupedContiguous;

  int current_iter = -1;
  uint32_t num_aligned_m_blocks;
  int* grouped_layout;
  uint32_t num_blocks;
  uint32_t shape_m;

  using Input = GroupedContiguousSchedulerInput;
  Input input;

  GroupedContiguousScheduler() {}

  __device__ __forceinline__ GroupedContiguousScheduler(Input& input) {
    num_aligned_m_blocks = ceil_div(input.shape_m, BLOCK_M);
    num_blocks = num_aligned_m_blocks * kNumNBlocks;
    this->shape_m = input.shape_m;
    this->grouped_layout = input.grouped_layout;
  }

  __device__ __forceinline__ uint32_t get_global_m_idx(uint32_t const& block_idx) {
    return block_idx * BLOCK_M;
  }

  __device__ __forceinline__ uint32_t get_global_n_idx(
      uint32_t const shape_dim, uint32_t const block_size, uint32_t const& block_idx, uint32_t const& m_block_idx = 0) {
    return __ldg(grouped_layout + m_block_idx * BLOCK_M) * shape_dim + block_idx * block_size;
  }

  __device__ __forceinline__ uint32_t get_global_scales_a_idx(uint32_t const& block_idx) {
    return block_idx;
  }

  __device__ __forceinline__ uint32_t get_global_scales_b_idx(
      uint32_t const shape_dim, uint32_t const block_size, uint32_t const& block_idx, uint32_t const& m_block_idx = 0) {
    return __ldg(grouped_layout + m_block_idx * BLOCK_M) * shape_dim + block_idx * block_size;
  }

  __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    ++current_iter;
    auto const next_block_idx = current_iter * gridDim.x + blockIdx.x;
    if (next_block_idx >= num_blocks) {
      return false;
    }
    get_swizzled_block_idx<kNumTMAMulticast, kNumNBlocks, kNumNBlocksPerGroup>(
        num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx);
    return true;
  }
};

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups,
          uint32_t kNumTMAMulticast, uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N), uint32_t kNumNBlocksPerGroup = 16>
struct GroupedMaskedScheduler {
  static constexpr GemmType gemm_type = GemmType::GroupedMasked;

  int current_iter = -1;
  uint32_t num_blocks;
  uint32_t num_aligned_m_blocks;
  uint32_t curr_group_idx;
  uint32_t curr_cumsum;
  uint32_t shape_m;
  int* grouped_layout;

  using Input = GroupedMaskedSchedulerInput;
  Input input;

  GroupedMaskedScheduler() {}

  __device__ __forceinline__ GroupedMaskedScheduler(Input& input) {
    num_aligned_m_blocks = ceil_div(input.shape_m, BLOCK_M);
    num_blocks = num_aligned_m_blocks * kNumNBlocks;
    this->shape_m = input.shape_m;
    this->grouped_layout = input.grouped_layout;
    curr_group_idx = 0;
    curr_cumsum = 0;
  }

  __device__ __forceinline__ uint32_t get_global_m_idx(uint32_t const& block_idx) {
    return curr_group_idx * shape_m + block_idx * BLOCK_M;
  }

  __device__ __forceinline__ uint32_t get_global_n_idx(
      uint32_t const shape_dim, uint32_t const block_size, uint32_t const& block_idx, uint32_t const& m_block_idx = 0) {
    return curr_group_idx * shape_dim + block_idx * block_size;
  }

  __device__ __forceinline__ uint32_t get_global_scales_a_idx(uint32_t const& block_idx) {
    return curr_group_idx * ceil_div(SHAPE_K, BLOCK_K) + block_idx;
  }

  __device__ __forceinline__ uint32_t get_global_scales_b_idx(
      uint32_t const shape_dim, uint32_t const block_size, uint32_t const& block_idx, uint32_t const& m_block_idx = 0) {
    return curr_group_idx * shape_dim + block_idx * block_size;
  }

  __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    ++current_iter;
    auto const next_block_idx = current_iter * gridDim.x + blockIdx.x;
    uint32_t num_m_blocks;
    while (true) {
      // End of the task
      if (curr_group_idx == kNumGroups)
        return false;

      // Within current group
      num_m_blocks = ceil_div(static_cast<uint32_t>(__ldg(grouped_layout + curr_group_idx)), BLOCK_M);
      auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
      if (next_block_idx < current_m_block_cumsum * kNumNBlocks)
        break;

      // Move to check the next group
      curr_group_idx++;
      curr_cumsum = current_m_block_cumsum;
    }

    get_swizzled_block_idx<kNumTMAMulticast, kNumNBlocks, kNumNBlocksPerGroup>(
        num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks, m_block_idx, n_block_idx);
    return true;
  }
};

// Need to keep the same as the one in tests/unittest/_torch/thop/deep_gemm_tests.py
template <typename T_offset, typename T_index>
__host__ __device__ __forceinline__ T_offset compute_padded_offset(T_offset offset, T_index problem_idx) {
  // This formulation ensures that padded_offset[i + 1] - padded_offset[i] >= offset[i + 1] - offset[i].
  constexpr T_offset alignment = 32;
  return (offset + problem_idx * (alignment - 1)) / alignment * alignment;
}

template <uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumGroups, uint32_t kNumTMAMulticast,
          uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N), uint32_t kNumNBlocksPerGroup = 16>
struct GroupedWithOffsetScheduler {
  static constexpr GemmType gemm_type = GemmType::GroupedWithOffset;

  int current_iter = -1;
  uint32_t curr_group_idx;
  uint32_t curr_cumsum;
  int64_t m_offset;
  int64_t m_padded_4_offset;
  int64_t m_boundary;
  int64_t* problem_m_offsets;

  using Input = GroupedWithOffsetSchedulerInput;
  Input input;

  GroupedWithOffsetScheduler() {}

  __device__ __forceinline__ GroupedWithOffsetScheduler(Input& input) {
    this->problem_m_offsets = input.problem_m_offsets;
    curr_group_idx = 0;
    curr_cumsum = 0;
  }

  __device__ __forceinline__ uint32_t get_global_m_idx(uint32_t const& block_idx) {
    return m_offset + block_idx * BLOCK_M;
  }

  __device__ __forceinline__ uint32_t get_global_n_idx(
      uint32_t const shape_dim, uint32_t const block_size, uint32_t const& block_idx, uint32_t const& m_block_idx = 0) {
    return curr_group_idx * shape_dim + block_idx * block_size;
  }

  __device__ __forceinline__ uint32_t get_global_scales_a_idx(uint32_t const& block_idx) {
    return m_padded_4_offset + block_idx * BLOCK_M;
  }

  __device__ __forceinline__ uint32_t get_global_scales_b_idx(
      uint32_t const shape_dim, uint32_t const block_size, uint32_t const& block_idx, uint32_t const& m_block_idx = 0) {
    return curr_group_idx * shape_dim + block_idx * block_size;
  }

  __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    ++current_iter;
    auto const next_block_idx = current_iter * gridDim.x + blockIdx.x;
    uint32_t num_m_blocks;
    while (true) {
      // End of the task
      if (curr_group_idx == kNumGroups)
        return false;
      m_offset = __ldg(problem_m_offsets + curr_group_idx);
      m_boundary = __ldg(problem_m_offsets + curr_group_idx + 1);
      m_padded_4_offset = compute_padded_offset(m_offset, curr_group_idx);
      auto m = m_boundary - m_offset;
      // Within current group
      num_m_blocks = ceil_div(m, static_cast<int64_t>(BLOCK_M));
      auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
      if (next_block_idx < current_m_block_cumsum * kNumNBlocks)
        break;

      // Move to check the next group
      curr_group_idx++;
      curr_cumsum = current_m_block_cumsum;
    }

    get_swizzled_block_idx<kNumTMAMulticast, kNumNBlocks, kNumNBlocksPerGroup>(
        num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks, m_block_idx, n_block_idx);
    return true;
  }
};

template <uint32_t SHAPE_M, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumGroups, uint32_t kNumTMAMulticast,
          uint32_t kNumMBlocks = ceil_div(SHAPE_M, BLOCK_M), uint32_t kNumMBlocksPerGroup = 16>
struct GroupedWithOffsetSchedulerSwapAB {
  static constexpr GemmType gemm_type = GemmType::GroupedWithOffset;

  int current_iter = -1;
  uint32_t curr_group_idx;
  uint32_t curr_cumsum;
  int64_t n_offset;
  int64_t n_padded_4_offset;
  int64_t n_boundary;
  int64_t* problem_n_offsets;

  using Input = GroupedWithOffsetSchedulerInputSwapAB;
  Input input;

  GroupedWithOffsetSchedulerSwapAB() {}

  __device__ __forceinline__ GroupedWithOffsetSchedulerSwapAB(Input& input) {
    this->problem_n_offsets = input.problem_n_offsets;
    curr_group_idx = 0;
    curr_cumsum = 0;
  }

  // weight
  __device__ __forceinline__ uint32_t get_global_m_idx(
      const uint32_t shape_dim, const uint32_t block_size, uint32_t const& block_idx, uint32_t const& n_block_idx = 0) {
    return curr_group_idx * shape_dim + block_idx * block_size;
  }

  // act
  __device__ __forceinline__ uint32_t get_global_n_idx(uint32_t const& block_idx) {
    return n_offset + block_idx * BLOCK_N;
  }

  // act scales
  __device__ __forceinline__ uint32_t get_global_scales_b_idx(uint32_t const& block_idx) {
    return n_padded_4_offset + block_idx * BLOCK_N;
  }

  // weight scales
  __device__ __forceinline__ uint32_t get_global_scales_a_idx(
      const uint32_t shape_dim, const uint32_t block_size, uint32_t const& block_idx, uint32_t const& n_block_idx = 0) {
    return curr_group_idx * shape_dim + block_idx * block_size;
  }

  __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    ++current_iter;
    auto const next_block_idx = current_iter * gridDim.x + blockIdx.x;
    uint32_t num_n_blocks;
    while (true) {
      // End of the task
      if (curr_group_idx == kNumGroups)
        return false;
      n_offset = __ldg(problem_n_offsets + curr_group_idx);
      n_boundary = __ldg(problem_n_offsets + curr_group_idx + 1);
      n_padded_4_offset = compute_padded_offset(n_offset, curr_group_idx);
      auto n = n_boundary - n_offset;
      // Within current group
      num_n_blocks = ceil_div(n, static_cast<int64_t>(BLOCK_N));
      auto current_n_block_cumsum = curr_cumsum + num_n_blocks;
      if (next_block_idx < current_n_block_cumsum * kNumMBlocks)
        break;

      // Move to check the next group
      curr_group_idx++;
      curr_cumsum = current_n_block_cumsum;
    }

    get_swizzled_block_idx<kNumTMAMulticast, kNumMBlocks, kNumMBlocksPerGroup>(
        num_n_blocks, next_block_idx - curr_cumsum * kNumMBlocks, n_block_idx, m_block_idx);
    return true;
  }
};

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups,
          uint32_t kNumTMAMulticast, uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N), uint32_t kNumNBlocksPerGroup = 16>
struct StridedBatchedScheduler {
  static constexpr GemmType gemm_type = GemmType::StridedBatched;

  int current_iter = -1;
  uint32_t curr_group_idx;
  uint32_t curr_cumsum;
  int64_t m_offset;
  int64_t m_boundary;

  using Input = StridedBatchedSchedulerInput;
  Input input;

  StridedBatchedScheduler() {}

  __device__ __forceinline__ StridedBatchedScheduler(Input& input) {
    this->input = input;
    curr_group_idx = 0;
    curr_cumsum = 0;
  }

  __device__ __forceinline__ uint32_t get_global_m_idx(uint32_t const& block_idx) {
    // Assuming stride_a % ld_a == 0 && stride_a >= ld_a
    return input.stride_a / input.ld_a * curr_group_idx + block_idx * BLOCK_M;
  }

  __device__ __forceinline__ uint32_t get_global_n_idx(
      uint32_t const shape_dim, uint32_t const block_size, uint32_t const& block_idx, uint32_t const& m_block_idx = 0) {
    // Assuming stride_b % ld_b == 0 && stride_b >= ld_b
    return input.stride_b / input.ld_b * curr_group_idx + block_idx * block_size;
  }

  __device__ __forceinline__ uint32_t get_global_scales_a_idx(uint32_t const& block_idx) {
    return curr_group_idx * ceil_div(SHAPE_K, BLOCK_K) + block_idx;
  }

  __device__ __forceinline__ uint32_t get_global_scales_b_idx(
      uint32_t const shape_dim, uint32_t const block_size, uint32_t const& block_idx, uint32_t const& m_block_idx = 0) {
    return curr_group_idx * shape_dim + block_idx * block_size;
  }

  __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    ++current_iter;
    auto const next_block_idx = current_iter * gridDim.x + blockIdx.x;
    uint32_t num_m_blocks;
    while (true) {
      // End of the task
      if (curr_group_idx == kNumGroups)
        return false;
      m_offset = curr_group_idx * input.shape_m;
      m_boundary = (curr_group_idx + 1) * input.shape_m;
      // Within current group
      num_m_blocks = ceil_div(input.shape_m, BLOCK_M);
      auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
      if (next_block_idx < current_m_block_cumsum * kNumNBlocks)
        break;

      // Move to check the next group
      curr_group_idx++;
      curr_cumsum = current_m_block_cumsum;
    }

    get_swizzled_block_idx<kNumTMAMulticast, kNumNBlocks, kNumNBlocksPerGroup>(
        num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks, m_block_idx, n_block_idx);
    return true;
  }
};

template <uint32_t SHAPE_M, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups,
          uint32_t kNumTMAMulticast, uint32_t kNumMBlocks = ceil_div(SHAPE_M, BLOCK_M), uint32_t kNumMBlocksPerGroup = 16>
struct StridedBatchedSchedulerSwapAB {
  static constexpr GemmType gemm_type = GemmType::StridedBatched;

  int current_iter = -1;
  uint32_t curr_group_idx;
  uint32_t curr_cumsum;
  int64_t n_offset;
  int64_t n_boundary;

  using Input = StridedBatchedSchedulerInputSwapAB;
  Input input;

  StridedBatchedSchedulerSwapAB() {}

  __device__ __forceinline__ StridedBatchedSchedulerSwapAB(Input& input) {
    this->input = input;
    curr_group_idx = 0;
    curr_cumsum = 0;
  }

  // weight
  __device__ __forceinline__ uint32_t get_global_m_idx(
      const uint32_t shape_dim, const uint32_t block_size, uint32_t const& block_idx, uint32_t const& n_block_idx = 0) {
    // Assuming stride_a % ld_a == 0 && stride_a >= ld_a
    return input.stride_a / input.ld_a * curr_group_idx + block_idx * block_size;
  }

  // act
  __device__ __forceinline__ uint32_t get_global_n_idx(uint32_t const& block_idx) {
    // Assuming stride_b % ld_b == 0 && stride_b >= ld_b
    return input.stride_b / input.ld_b * curr_group_idx + block_idx * BLOCK_N;
  }

  // act scales
  __device__ __forceinline__ uint32_t get_global_scales_b_idx(uint32_t const& block_idx) {
    return curr_group_idx * ceil_div(SHAPE_K, BLOCK_K) + block_idx;
  }

  // weight scales
  __device__ __forceinline__ uint32_t get_global_scales_a_idx(
      const uint32_t shape_dim, const uint32_t block_size, uint32_t const& block_idx, uint32_t const& n_block_idx = 0) {
    return curr_group_idx * shape_dim + block_idx * block_size;
  }

  __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    ++current_iter;
    auto const next_block_idx = current_iter * gridDim.x + blockIdx.x;
    uint32_t num_n_blocks;
    while (true) {
      // End of the task
      if (curr_group_idx == kNumGroups)
        return false;
      n_offset = curr_group_idx * input.shape_n;
      n_boundary = (curr_group_idx + 1) * input.shape_n;
      // Within current group
      num_n_blocks = ceil_div(input.shape_n, BLOCK_N);
      auto current_n_block_cumsum = curr_cumsum + num_n_blocks;
      if (next_block_idx < current_n_block_cumsum * kNumMBlocks)
        break;

      // Move to check the next group
      curr_group_idx++;
      curr_cumsum = current_n_block_cumsum;
    }

    // Note: Here, m and n roles are swapped
    get_swizzled_block_idx<kNumTMAMulticast, kNumMBlocks, kNumMBlocksPerGroup>(
        num_n_blocks, next_block_idx - curr_cumsum * kNumMBlocks, n_block_idx, m_block_idx);
    return true;
  }
};

template <GemmType GT, uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumTMAMulticast, uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N),
          uint32_t kNumNBlocksPerGroup = 16>
struct SchedulerSelector {
  static constexpr auto select_type() {
    if constexpr (GT == GemmType::Normal)
      return NormalScheduler<SHAPE_N, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kNumNBlocks,
                             kNumNBlocksPerGroup>();
    if constexpr (GT == GemmType::GroupedContiguous)
      return GroupedContiguousScheduler<SHAPE_N, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kNumNBlocks,
                                        kNumNBlocksPerGroup>();
    if constexpr (GT == GemmType::GroupedMasked)
      return GroupedMaskedScheduler<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kNumGroups, kNumTMAMulticast,
                                    kNumNBlocks, kNumNBlocksPerGroup>();
    if constexpr (GT == GemmType::GroupedWithOffset)
      return GroupedWithOffsetScheduler<SHAPE_N, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kNumNBlocks,
                                        kNumNBlocksPerGroup>();
    if constexpr (GT == GemmType::StridedBatched)
      return StridedBatchedScheduler<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kNumGroups, kNumTMAMulticast,
                                     kNumNBlocks, kNumNBlocksPerGroup>();
  }

  using type = decltype(select_type());
};

template <GemmType GT, uint32_t SHAPE_M, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumTMAMulticast, uint32_t kNumMBlocks = ceil_div(SHAPE_M, BLOCK_M),
          uint32_t kNumMBlocksPerGroup = 16>
struct SchedulerSelectorSwapAB {
  static constexpr auto select_type() {
    static_assert(GT == GemmType::GroupedWithOffset || GT == GemmType::Normal,
                  "Only GroupedWithOffset and Normal are supported for SwapAB");
    if constexpr (GT == GemmType::Normal)
      return NormalSchedulerSwapAB<SHAPE_M, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kNumMBlocks,
                                   kNumMBlocksPerGroup>();
    if constexpr (GT == GemmType::GroupedWithOffset)
      return GroupedWithOffsetSchedulerSwapAB<SHAPE_M, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast,
                                              kNumMBlocks, kNumMBlocksPerGroup>();
  }

  using type = decltype(select_type());
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

}  // namespace deep_gemm
