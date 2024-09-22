/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

// Modifications: Update interface and implmentation to be thread-safe
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/fused_multihead_attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kMinSequenceLengthFlashAttention = 385;

class MHARunner {
 public:
  MHARunner(int num_heads, int head_size, bool causal, float scale)
      : num_heads_(num_heads),
        head_size_(head_size),
        scale_(scale == 0.0f ? 1.f / sqrtf(static_cast<float>(head_size)) : scale),
        is_causal_(causal) {
  }

  virtual ~MHARunner() = default;

  virtual int NormalizeSequenceLength(int max_seq_len) const = 0;

  virtual bool IsValid(int normalized_sequence_length) const = 0;

  virtual void Run(int batch_size,
                   int normalized_sequence_length,
                   const void* input,
                   const void* cu_seqlens,
                   void* output,
                   cudaStream_t stream) const = 0;

 protected:
  int num_heads_;
  int head_size_;
  float scale_;
  bool is_causal_;
};

class FusedMHARunnerFP16v2 : public MHARunner {
 public:
  FusedMHARunnerFP16v2(int num_heads,
                       int head_size,
                       int sm,
                       bool causal,
                       bool enable_flash_attention,
                       float scale);

  ~FusedMHARunnerFP16v2() = default;  // for impl_

  static bool IsSupported(int sm, int head_size, int sequence_length, bool enable_flash_attention, bool causal);

  static std::unique_ptr<MHARunner> Create(int num_heads,
                                           int head_size,
                                           int sm,
                                           bool causal,
                                           bool enable_flash_attention,
                                           float scale);

  bool IsValid(int normalized_sequence_length) const override;

  int NormalizeSequenceLength(int max_seq_len) const override;

  void Run(int batch_size,
           int normalized_sequence_length,
           const void* input,
           const void* cu_seqlens,
           void* output,
           cudaStream_t stream) const override;

 private:
  bool enable_flash_attention_;
  class FmhaImpl;
  std::unique_ptr<FmhaImpl> impl_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
