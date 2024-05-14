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

#pragma once

#include <memory>
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/fused_multihead_attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kMinSequenceLengthFlashAttention = 385;

// Multi-Head Attention runner
class MHARunner {
 public:
  MHARunner(const int numHeads, const int headSize, const int wordSize, bool causal_mask, const float scale)
      : mS(0),
        mB(0),
        mOmatSize(0),
        mNumMats(0),
        mNumHeads(numHeads),
        mHeadSize(headSize),
        mWordSize(wordSize),
        mLdQKV(0),
        mStrideQKV(0),
        mLdOut(0),
        mStrideOut(0),
        mScale(scale == 0.0f ? 1.f / sqrtf(static_cast<float>(headSize))
                             : scale),
        mHasCausalMask(causal_mask) {
  }

  virtual ~MHARunner() = default;

  virtual void setup(const int S, const int B) {
    ORT_ENFORCE(S > 0);
    ORT_ENFORCE(B > 0);

    mB = B;
    mS = S;

    mLdQKV = 3 * B * mNumHeads * mHeadSize;
    mStrideQKV = 3 * mHeadSize;

    mLdOut = B * mNumHeads * mHeadSize;
    mStrideOut = mHeadSize;
    mOmatSize = S * S;
    mNumMats = B * mNumHeads;
  }

  virtual void run(const void* input, const void* cu_seqlens, void* output, cudaStream_t stream) = 0;

  virtual size_t getWorkspaceSize() const = 0;

  virtual bool isValid(int s) const = 0;

  virtual int getSFromMaxSeqLen(const int max_seq_len) const = 0;

 protected:
  int mS;
  int mB;
  int mOmatSize;
  int mNumMats;
  int mNumHeads;
  int mHeadSize;
  int mWordSize;
  int mLdQKV;
  int mStrideQKV;
  int mLdOut;
  int mStrideOut;

  float mScale;
  bool mHasCausalMask;
};

class FusedMHARunnerFP16v2 : public MHARunner {
 public:
  FusedMHARunnerFP16v2(const int numHeads,
                       const int headSize,
                       const int sm,
                       bool causal_mask,
                       bool enable_flash_attention,
                       const float scale);
  ~FusedMHARunnerFP16v2() = default;  // for pimpl

  virtual void setup(const int S, const int B) override;

  static bool is_supported(int sm, int head_size, int sequence_length, bool enable_flash_attention, bool causal);

  void run(const void* input, const void* cu_seqlens, void* output, cudaStream_t stream) override;

  size_t getWorkspaceSize() const override;

  bool isValid(int s) const override;

  int getSFromMaxSeqLen(const int max_seq_len) const override;

  static std::unique_ptr<MHARunner> Create(const int numHeads,
                                           const int headSize,
                                           const int sm,
                                           bool causal_mask,
                                           bool enable_flash_attention,
                                           const float scale);

 private:
  int mSm;
  bool mEnableFlashAttention;
  class mhaImpl;
  std::unique_ptr<mhaImpl> pimpl;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
