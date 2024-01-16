/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// The CUDA kernel is modified from GroupNorm plugin of TensorRT 8.5
// Modifications: heuristic channels per block; support epsilon; support skip and bias; update coding style.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/diffusion/group_norm_common_base.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

int NextSize(int x) {
  for (size_t i = 0; i < kNumOfSizes; ++i) {
    if (x <= kSizes[i]) {
      return kSizes[i];
    }
  }

  return x;
}

int32_t GetThreadsPerBlock(int32_t channels_per_block, int32_t channels_per_thread) {
  return NextSize(channels_per_block) / channels_per_thread;
}

int32_t FindMaxDivisor(int32_t n, int32_t max_allowed_divisor) {
  int32_t max_divisor = -1;
  for (int32_t i = 1; i <= std::sqrt(n); i++) {
    if (n % i == 0) {
      int32_t divisor1 = n / i;
      int32_t divisor2 = i;

      if (divisor1 > max_divisor && divisor1 < max_allowed_divisor) {
        max_divisor = divisor1;
      }
      if (divisor2 > max_divisor && divisor2 < max_allowed_divisor) {
        max_divisor = divisor2;
      }
    }
  }
  return max_divisor;
}

// Find proper channels per block based on a cost function: The cost is number of channels corresponding to
// extra threads allocated but no channels assigned to them to work on. If cost is zero, every thread has
// work to do so it is ideal case.
int FindChannelsPerBlock(int num_channels, int channels_per_group) {
  int min_cost = -1;
  int best_candidate = -1;
  for (size_t i = kNumOfSizes; i > 0; --i) {
    if (kSizes[i - 1] < channels_per_group) {
      break;
    }

    int channels_per_block = kSizes[i - 1] / channels_per_group * channels_per_group;
    int blocks = (num_channels + channels_per_block - 1) / channels_per_block;
    int cost = blocks * kSizes[i - 1] - num_channels;
    if (cost == 0) {
      return channels_per_block;
    }

    if (min_cost == -1 || cost < min_cost) {
      min_cost = cost;
      best_candidate = channels_per_block;
    }
  }

  return best_candidate;
}

int GetChannelsPerBlock(int num_channels, int num_groups) {
  int32_t channels_per_group = num_channels / num_groups;
  int32_t channels_per_block = channels_per_group;
  if (channels_per_group < kMaxSize / 2) {
    channels_per_block = FindChannelsPerBlock(num_channels, channels_per_group);
  }
  return channels_per_block;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
