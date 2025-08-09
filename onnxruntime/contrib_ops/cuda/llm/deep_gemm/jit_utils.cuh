/*
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
#include <climits>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <nvrtc.h>
#include <string>
#include <tuple>
#include <vector>

#include "scheduler.cuh"

// Helper function to check NVRTC errors
#define CHECK_NVRTC(call)                                                       \
  do {                                                                          \
    nvrtcResult result = call;                                                  \
    if (result != NVRTC_SUCCESS) {                                              \
      std::cerr << "NVRTC error: " << nvrtcGetErrorString(result) << std::endl; \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

// Helper function to check CUDA driver errors
#define CHECK_CUDA(call)                                        \
  do {                                                          \
    CUresult result = call;                                     \
    if (result != CUDA_SUCCESS) {                               \
      const char* error_string;                                 \
      cuGetErrorString(result, &error_string);                  \
      std::cerr << "CUDA error: " << error_string << std::endl; \
      exit(1);                                                  \
    }                                                           \
  } while (0)

namespace deep_gemm::jit {

using GemmConfig = std::tuple<int, int, int, int, int>;  // block_m, block_n, num_stages, num_tma_multicast, best_smem_size

std::string gemm_type_to_string(deep_gemm::GemmType gemm_type);

int div_up(int a, int b);
int get_smem_size(int num_stages, int k, int block_m, int block_n, int block_k, bool swap_ab);
bool is_tma_multicast_legal(int n, int block_n, int num_tma_multicast, int num_sms);
GemmConfig get_best_gemm_config(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_groups,
                                int num_device_sms, bool is_grouped_contiguous, bool swap_ab);
}  // namespace deep_gemm::jit

namespace deep_gemm::jit {

std::string gemm_type_to_string(deep_gemm::GemmType gemm_type) {
  switch (gemm_type) {
    case deep_gemm::GemmType::Normal:
      return std::string("Normal");
    case deep_gemm::GemmType::GroupedContiguous:
      return std::string("GroupedContiguous");
    case deep_gemm::GemmType::GroupedMasked:
      return std::string("GroupedMasked");
    case deep_gemm::GemmType::GroupedWithOffset:
      return std::string("GroupedWithOffset");
    case deep_gemm::GemmType::StridedBatched:
      return std::string("StridedBatched");
    // Add other GEMM types as needed
    default:
      return std::string("Unknown");
  }
}

int div_up(int a, int b) {
  return (a + b - 1) / b;
}

int get_smem_size(int num_stages, int k, int block_m, int block_n, int block_k = 128, bool swap_ab = false) {
  if (!swap_ab) {
    int smem_d = block_m * block_n * 2;
    int smem_a_per_stage = block_m * block_k;
    int smem_scales_a_per_stage = block_m * 4;
    int smem_b_per_stage = block_n * block_k;
    int smem_scales_b = div_up(k, block_k) * 4;
    int smem_barrier = num_stages * 8 * 2;

    int smem_size = 0;
    smem_size += smem_d;
    smem_size += num_stages * smem_a_per_stage;
    smem_size += num_stages * smem_scales_a_per_stage;
    smem_size += num_stages * smem_b_per_stage;
    smem_size += div_up(smem_scales_b * (block_k % block_n == 0 ? 1 : 2), 8) * 8;
    smem_size += smem_barrier;

    return smem_size;
  } else {
    int smem_d = block_n * block_m * 2;
    int smem_a_per_stage = block_m * block_k;              // weight
    int smem_scales_a_per_stage = div_up(k, block_k) * 4;  // weight scales
    int smem_b_per_stage = block_n * block_k;              // act
    int smem_scales_b = div_up(block_n * 4, 128) * 128;    // act scales,tma 128B alignment
    int smem_barrier = num_stages * 8 * 2;

    int smem_size = 0;
    smem_size += smem_d;
    smem_size += num_stages * smem_a_per_stage;
    smem_size += num_stages * smem_scales_b;
    smem_size += num_stages * smem_b_per_stage;
    smem_size += div_up(smem_scales_a_per_stage, 8) * 8;
    smem_size += smem_barrier;

    return smem_size;
  }
}

bool is_tma_multicast_legal(int n, int block_n, int num_tma_multicast, int num_sms) {
  if (num_tma_multicast == 1) {
    return true;
  }
  return (n % (block_n * num_tma_multicast) == 0) && num_sms % num_tma_multicast == 0;
}

GemmConfig get_best_gemm_config(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_groups,
                                int num_device_sms, bool is_grouped_contiguous = false, bool swap_ab = false) {
  // Choose candidate block sizes
  std::vector<int> block_ms;
  block_ms.push_back((!is_grouped_contiguous && shape_m <= 64) ? 64 : 128);

  // Candidate block sizes for N dimension
  std::vector<int> block_ns;
  for (int i = 16; i <= 128; i += 8) {
    block_ns.push_back(i);
  }

  // Lambda functions for calculating waves and utilization
  auto fix_wave_saturate = [num_device_sms](int x) -> int { return x == 0 ? num_device_sms : x; };

  auto get_num_waves = [shape_m, shape_n, num_groups, num_device_sms](int block_m, int block_n) -> int { return div_up(div_up(shape_m, block_m) * div_up(shape_n, block_n) * num_groups, num_device_sms); };

  auto get_last_wave_util = [shape_m, shape_n, num_groups, num_device_sms, &fix_wave_saturate](int block_m, int block_n) -> int { return fix_wave_saturate((div_up(shape_m, block_m) * div_up(shape_n, block_n) * num_groups) % num_device_sms); };

  // Find best block sizes
  int best_block_m = 0;
  int best_block_n = 0;
  for (int block_m : block_ms) {
    for (int block_n : block_ns) {
      bool success = false;
      int num_waves = get_num_waves(block_m, block_n);
      int best_num_waves = best_block_m == 0 ? INT_MAX : get_num_waves(best_block_m, best_block_n);

      if (best_block_m == 0 || best_block_n == 0) {
        success = true;
      } else if (num_waves < best_num_waves) {
        success = true;
      } else if (num_waves == best_num_waves) {
        // Check last wave utilization
        int util = get_last_wave_util(block_m, block_n);
        int best_util = get_last_wave_util(best_block_m, best_block_n);
        success = util > best_util || (util == best_util && (block_m > best_block_m || (block_m == best_block_m && block_n < best_block_n)));
      }

      if (success) {
        best_block_m = block_m;
        best_block_n = block_n;
      }
    }
  }

  // Find best number of stages
  int best_num_stages = 0;
  int best_smem_size = 0;
  constexpr int sm90_capacity = 232448;

  std::vector<int> stage_candidates;
  if (128 % best_block_n != 0) {
    stage_candidates = {6, 5, 4};
  } else {
    stage_candidates = {8, 7, 6, 5, 4};
  }

  for (int num_stages : stage_candidates) {
    int smem_size = get_smem_size(num_stages, shape_k, best_block_m, best_block_n, 128, swap_ab);
    if (smem_size <= sm90_capacity) {
      best_num_stages = num_stages;
      best_smem_size = smem_size;
      break;
    }
  }

  // Determine TMA multicast settings
  int best_num_tma_multicast = 1;

  if (!swap_ab) {
    if (shape_m >= 1024 && is_tma_multicast_legal(shape_n, best_block_n, 2, num_device_sms) && num_groups == 1) {
      best_num_tma_multicast = 2;
    }
  } else {
    if (shape_n >= 1024 && is_tma_multicast_legal(shape_m, best_block_m, 2, num_device_sms) && num_groups == 1) {
      best_num_tma_multicast = 2;
    }
  }

  return std::make_tuple(best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size);
}
}  // namespace deep_gemm::jit
