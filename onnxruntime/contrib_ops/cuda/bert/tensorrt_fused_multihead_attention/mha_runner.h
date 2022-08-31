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

#pragma once

#include <string>
#include <vector>
#include <cuda.h>
#include <cublas_v2.h>
//#include "zeroPadding2d.h"
#include "fused_multihead_attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Multi Head Attention runner
class MHARunner {
 public:
  MHARunner(const int32_t numHeads, const int32_t headSize, const int wordSize)
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
        mRsqrtHeadSize(1.F / sqrtf(headSize)) {
  }

  virtual ~MHARunner() = default;

  virtual void setup(const int32_t S, const int32_t B) {
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

  virtual void run(const void* qkvPtr, const void* maskPtr, const void* seqLens,
                   void* output, void* workspace, cudaStream_t stream) = 0;

  virtual size_t getWorkspaceSize() const = 0;

  virtual bool isValid(int32_t s) const = 0;

 protected:
  int32_t mS;
  int32_t mB;
  int32_t mOmatSize;
  int32_t mNumMats;
  int32_t mNumHeads;
  int32_t mHeadSize;
  int32_t mWordSize;
  int32_t mLdQKV;
  int32_t mStrideQKV;
  int32_t mLdOut;
  int32_t mStrideOut;

  float mRsqrtHeadSize;
};

std::pair<int, int> tuneBatchedGemm(const int32_t B, const int32_t S, const int32_t numHeads, const int32_t headSize);

template <typename T>
int32_t computeScaledSoftmax(cudaStream_t stream, const int32_t ld, const int32_t B, const int32_t N,
                             const float rsqrtHeadSize, const T* input, T* output);

template <typename T>
int32_t computeMaskedScaledSoftmax(cudaStream_t stream, const int32_t ld, const int32_t B, const int32_t N,
                                   const float rsqrtHeadSize, const int* maskIdx, const T* input, T* output);

class FusedMHARunnerFP16v2 : public MHARunner {
 public:
  FusedMHARunnerFP16v2(const int32_t numHeads, const int32_t headSize, const int32_t sm);
  ~FusedMHARunnerFP16v2() = default;  // for pimpl

  virtual void setup(const int32_t S, const int32_t B) override;

  void run(const void* qkvPtr, const void* maskPtr, const void* seqLens,
           void* output, void* workspace, cudaStream_t stream) override;

  size_t getWorkspaceSize() const override;

  bool isValid(int32_t s) const override;

 private:
  int32_t mSm;
  class mhaImpl;
  std::unique_ptr<mhaImpl> pimpl;
};
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
