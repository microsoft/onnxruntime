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
#pragma once
#include "core/providers/cuda/cuda_common.h"
using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// TODO: Similar to SkipLayerNorm kernel, read/write up to 8 channels at same time.
constexpr static int32_t CHANNELS_PER_THREAD = 2;

constexpr static int kSizes[] = {128, 256, 320, 384, 512};
constexpr static size_t kNumOfSizes = sizeof(kSizes) / sizeof(kSizes[0]);
constexpr static int kMaxSize = kSizes[kNumOfSizes - 1];

int32_t GetThreadsPerBlock(int32_t channels_per_block, int32_t channels_per_thread);

static inline int32_t DivUp(int32_t m, int32_t n) {
  return (m + n - 1) / n;
}

int32_t FindMaxDivisor(int32_t n, int32_t max_allowed_divisor);

int GetChannelsPerBlock(int num_channels, int num_groups);

template <typename T>
struct GroupNormNHWCParams {
  // The output buffer. Shape is (n, h, w, c).
  T* dst;

  // Optional output of element-wise add result of src, skip and bias. Shape is (n, h, w, c).
  T* add_out;

  // The input buffer. Shape is (n, h, w, c).
  T const* src;

  // Optional input buffer for skip tensor. Shape is (n, h, w, c) or (n, 1, 1, c) or (n, c).
  T const* skip;

  // Optional input buffer for bias tensor. Shape is (c).
  T const* bias;

  // The gamma scaling factor.
  float const* gamma;

  // The beta term to add in GN.
  float const* beta;

  // The temporary buffer to do the global parallel reduction. Shape is (n, 2, g), where g is number of groups.
  float* group_sum_buffer;

  // The number of instances in the batch.
  int32_t n;

  // The height and width of each activation map.
  int32_t h;
  int32_t w;

  // Number of channels.
  int32_t c;

  // Number of groups.
  int32_t groups;

  // Do we apply the SiLU activation function?
  bool use_silu;

  // Precomputed values and parameters to control the execution of the kernels.

  // Number of activations per instance (h * w)
  int32_t hw;

  // Number of activations per block
  int32_t hw_per_block;

  // Number of channels per block in the C dimension.
  int32_t channels_per_block;

  // Number of channels per group in the C dimension.
  int32_t channels_per_group;

  // The precomputed stride between instances.
  int32_t hwc;
  // The inverse of hw*channels_per_group to compute mean of a group.
  float inv_hw_channels_per_group;
  // The precomputed number of groups per block.
  int32_t groups_per_block;

  // Number of threads per block
  int32_t threads_per_block;

  // Epsilon to get stable variance in normalization.
  float epsilon;

  // Whether skip need broadcast. True if shape of skip is (N, C) or (N, 1, 1, C); False otherwise.
  bool broadcast_skip;

  // For SkipGroupNorm, it points to the intermediate result of adding skip and bias.
  T* skip_workspace;

  GroupNormNHWCParams(T* output,
                      T* add_out,
                      const T* input,
                      const T* skip,
                      const T* bias,
                      const float* gamma,
                      const float* beta,
                      float* workspace,
                      float epsilon,
                      int batch_size,
                      int num_channels,
                      int height,
                      int width,
                      int num_groups,
                      bool use_silu,
                      bool broadcast_skip,
                      int channels_per_block) {
    int32_t channels_per_group_in = num_channels / num_groups;
    // channels_per_block is computed in PrePack.
    // If the gamma is not initializer, channels_per_block might be zero after PrePack. In that happens, compute it here.
    if (channels_per_block < channels_per_group_in) {
      channels_per_block = GetChannelsPerBlock(num_channels, num_groups);
    }

    this->use_silu = use_silu;
    this->dst = output;
    this->add_out = add_out;
    this->src = input;
    this->skip = skip;
    this->bias = bias;
    this->gamma = gamma;
    this->beta = beta;
    this->group_sum_buffer = workspace;
    this->n = batch_size;
    this->h = height;
    this->w = width;
    this->c = num_channels;
    this->groups = num_groups;
    this->hw = this->h * this->w;

    // This will allocate as many blocks as possible to partition HW.
    // For Stable Diffusion, latent hw is 4K ~ 16K. This will allocate 1024 blocks, and each handles 4~16 hw.
    // TODO: tune this logic to find proper blocks when hw is small.
    constexpr int32_t max_blocks_per_hw = 1024;
    const int32_t blocks_per_hw = FindMaxDivisor(this->hw, max_blocks_per_hw);
    this->hw_per_block = DivUp(this->hw, blocks_per_hw);

    this->channels_per_block = channels_per_block;
    this->channels_per_group = channels_per_group_in;
    this->hwc = this->hw * this->c;
    this->inv_hw_channels_per_group = 1.F / (float)(this->hw * this->channels_per_group);
    this->groups_per_block = channels_per_block / this->channels_per_group;
    this->epsilon = epsilon;
    this->broadcast_skip = broadcast_skip;

    // Workspace for SkipGroupNorm to store intermediate results of src+skip+bias.
    this->skip_workspace = (this->add_out != nullptr) ? this->add_out : this->dst;

    this->threads_per_block = GetThreadsPerBlock(channels_per_block, CHANNELS_PER_THREAD);
  }
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
