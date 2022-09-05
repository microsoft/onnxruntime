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

// Modify from https://github.com/NVIDIA/TensorRT/commit/09dce45f919be40d27b2b650e716debb3740d410
// The kernels for sm70 are copied from FasterTransformer
#pragma once

#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/fused_multihead_attention.h"
#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct Fused_multihead_attention_params_v2 {
  // The QKV matrices.
  void* qkv_ptr{};
  // The mask to implement drop-out.
  void* packed_mask_ptr{};
  // The O matrix (output).
  void* o_ptr{};

  // The stride between rows of the Q, K and V matrices.
  int64_t qkv_stride_in_bytes{};
  // The stride between matrices of packed mask.
  int64_t packed_mask_stride_in_bytes{};
  // The stride between rows of O.
  int64_t o_stride_in_bytes;

  // The dimensions.
  int32_t b{};
  int32_t h{};
  int32_t s{};
  int32_t d{};
  // The scaling factors for the kernel.
  uint32_t scale_bmm1{};
  uint32_t scale_softmax{};
  uint32_t scale_bmm2{};

  // Do we use Niall's trick to avoid I2F/F2I in the INT8 kernel.
  // See https://confluence.nvidia.com/pages/viewpage.action?pageId=302779721 for details.
  bool enable_i2f_trick{};

  // array of length b+1 holding prefix sum of actual sequence lengths
  int32_t* cu_seqlens{};

  // use C/32 Format.
  bool interleaved{};
  bool ignore_b1opt{};
  bool force_unroll{};
  bool use_int8_scale_max{};

  // The number of heads computed by one iteration of the wave.
  int32_t heads_per_wave{};
  // Buffers to perform a global sync and a critical section.
  int32_t* counters{};
  int32_t* max_barriers{};
  int32_t* sum_barriers{};
  int32_t* locks{};
  // Scratch buffers to finalize softmax.
  float* max_scratch_ptr{};
  float* sum_scratch_ptr{};
  // Scratch buffer to finalize the output (not needed for FP16).
  int* o_scratch_ptr{};

  void clear() {
    *this = Fused_multihead_attention_params_v2();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
extern unsigned char fused_multihead_attention_v2_fp16_64_64_kernel_sm70_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_96_64_kernel_sm70_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_128_64_kernel_sm70_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_256_64_kernel_sm70_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_384_64_kernel_sm70_cubin[];

extern unsigned char fused_multihead_attention_v2_fp16_128_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_128_64_kernel_sm86_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_256_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_256_64_kernel_sm86_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_384_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_384_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_384_64_kernel_sm86_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_64_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_64_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_64_64_kernel_sm86_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_96_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_96_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_96_64_kernel_sm86_cubin[];

extern unsigned char cubin_fmha_v2_fp16_512_64_sm80_cu_cubin[];

extern uint32_t fused_multihead_attention_v2_fp16_64_64_kernel_sm70_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_96_64_kernel_sm70_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_128_64_kernel_sm70_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_256_64_kernel_sm70_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_384_64_kernel_sm70_cubin_len;

extern uint32_t fused_multihead_attention_v2_fp16_128_64_kernel_sm75_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_128_64_kernel_sm86_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_256_64_kernel_sm75_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_256_64_kernel_sm86_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_384_64_kernel_sm75_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_384_64_kernel_sm80_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_384_64_kernel_sm86_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_64_64_kernel_sm75_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_64_64_kernel_sm80_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_64_64_kernel_sm86_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_96_64_kernel_sm75_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_96_64_kernel_sm80_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_96_64_kernel_sm86_cubin_len;

extern uint32_t cubin_fmha_v2_fp16_512_64_sm80_cu_cubin_len;

static const struct FusedMultiHeadAttentionKernelMetaInfoV2 {
  Data_type mDataType;
  uint32_t mS;
  uint32_t mD;
  uint32_t mSM;
  const unsigned char* mCubin;
  uint32_t mCubinSize;
  const char* mFuncName;
  uint32_t mSharedMemBytes;
  uint32_t mThreadsPerCTA;
  uint32_t mUnrollStep;
  bool mInterleaved;
} sMhaKernelMetaInfosV2[] = {
    // Volta
    {DATA_TYPE_FP16, 64, 64, kSM_70, fused_multihead_attention_v2_fp16_64_64_kernel_sm70_cubin,
     fused_multihead_attention_v2_fp16_64_64_kernel_sm70_cubin_len,
     "fused_multihead_attention_v2_fp16_64_64_kernel_sm70", 32768, 128, 0, false},
    {DATA_TYPE_FP16, 64, 64, kSM_70, fused_multihead_attention_v2_fp16_64_64_kernel_sm70_cubin,
     fused_multihead_attention_v2_fp16_64_64_kernel_sm70_cubin_len,
     "fused_multihead_attention_v2_fp16_64_64_kernel_sm70_noloop", 36864, 128, 32, false},
    {DATA_TYPE_FP16, 96, 64, kSM_70, fused_multihead_attention_v2_fp16_96_64_kernel_sm70_cubin,
     fused_multihead_attention_v2_fp16_96_64_kernel_sm70_cubin_len,
     "fused_multihead_attention_v2_fp16_96_64_kernel_sm70", 32768, 128, 0, false},
    {DATA_TYPE_FP16, 96, 64, kSM_70, fused_multihead_attention_v2_fp16_96_64_kernel_sm70_cubin,
     fused_multihead_attention_v2_fp16_96_64_kernel_sm70_cubin_len,
     "fused_multihead_attention_v2_fp16_96_64_kernel_sm70_noloop", 36864, 128, 32, false},
    {DATA_TYPE_FP16, 128, 64, kSM_70, fused_multihead_attention_v2_fp16_128_64_kernel_sm70_cubin,
     fused_multihead_attention_v2_fp16_128_64_kernel_sm70_cubin_len,
     "fused_multihead_attention_v2_fp16_128_64_kernel_sm70", 36864, 128, 0, false},
    {DATA_TYPE_FP16, 128, 64, kSM_70, fused_multihead_attention_v2_fp16_128_64_kernel_sm70_cubin,
     fused_multihead_attention_v2_fp16_128_64_kernel_sm70_cubin_len,
     "fused_multihead_attention_v2_fp16_128_64_kernel_sm70_noloop", 36864, 128, 32, false},
    {DATA_TYPE_FP16, 256, 64, kSM_70, fused_multihead_attention_v2_fp16_256_64_kernel_sm70_cubin,
     fused_multihead_attention_v2_fp16_256_64_kernel_sm70_cubin_len,
     "fused_multihead_attention_v2_fp16_256_64_kernel_sm70", 69632, 256, 0, false},
    {DATA_TYPE_FP16, 256, 64, kSM_70, fused_multihead_attention_v2_fp16_256_64_kernel_sm70_cubin,
     fused_multihead_attention_v2_fp16_256_64_kernel_sm70_cubin_len,
     "fused_multihead_attention_v2_fp16_256_64_kernel_sm70_noloop", 69632, 256, 32, false},
    {DATA_TYPE_FP16, 384, 64, kSM_70, fused_multihead_attention_v2_fp16_384_64_kernel_sm70_cubin,
     fused_multihead_attention_v2_fp16_384_64_kernel_sm70_cubin_len,
     "fused_multihead_attention_v2_fp16_384_64_kernel_sm70", 69632, 256, 0, false},
    {DATA_TYPE_FP16, 384, 64, kSM_70, fused_multihead_attention_v2_fp16_384_64_kernel_sm70_cubin,
     fused_multihead_attention_v2_fp16_384_64_kernel_sm70_cubin_len,
     "fused_multihead_attention_v2_fp16_384_64_kernel_sm70_noloop", 69632, 256, 32, false},

    // Turing
    {DATA_TYPE_FP16, 64, 64, kSM_75, fused_multihead_attention_v2_fp16_64_64_kernel_sm75_cubin,
     fused_multihead_attention_v2_fp16_64_64_kernel_sm75_cubin_len,
     "fused_multihead_attention_v2_fp16_64_64_kernel_sm75", 24576, 128, 0, false},
    {DATA_TYPE_FP16, 96, 64, kSM_75, fused_multihead_attention_v2_fp16_96_64_kernel_sm75_cubin,
     fused_multihead_attention_v2_fp16_96_64_kernel_sm75_cubin_len,
     "fused_multihead_attention_v2_fp16_96_64_kernel_sm75", 24576, 128, 0, false},
    {DATA_TYPE_FP16, 128, 64, kSM_75, fused_multihead_attention_v2_fp16_128_64_kernel_sm75_cubin,
     fused_multihead_attention_v2_fp16_128_64_kernel_sm75_cubin_len,
     "fused_multihead_attention_v2_fp16_128_64_kernel_sm75_noloop", 20480, 128, 32, false},
    {DATA_TYPE_FP16, 128, 64, kSM_75, fused_multihead_attention_v2_fp16_128_64_kernel_sm75_cubin,
     fused_multihead_attention_v2_fp16_128_64_kernel_sm75_cubin_len,
     "fused_multihead_attention_v2_fp16_128_64_kernel_sm75", 32768, 128, 0, false},
    {DATA_TYPE_FP16, 256, 64, kSM_75, fused_multihead_attention_v2_fp16_256_64_kernel_sm75_cubin,
     fused_multihead_attention_v2_fp16_256_64_kernel_sm75_cubin_len,
     "fused_multihead_attention_v2_fp16_256_64_kernel_sm75_noloop", 36864, 128, 32, false},
    {DATA_TYPE_FP16, 256, 64, kSM_75, fused_multihead_attention_v2_fp16_256_64_kernel_sm75_cubin,
     fused_multihead_attention_v2_fp16_256_64_kernel_sm75_cubin_len,
     "fused_multihead_attention_v2_fp16_256_64_kernel_sm75", 36864, 128, 0, false},
    {DATA_TYPE_FP16, 384, 64, kSM_75, fused_multihead_attention_v2_fp16_384_64_kernel_sm75_cubin,
     fused_multihead_attention_v2_fp16_384_64_kernel_sm75_cubin_len,
     "fused_multihead_attention_v2_fp16_384_64_kernel_sm75_noloop", 53248, 256, 32, false},
    {DATA_TYPE_FP16, 384, 64, kSM_75, fused_multihead_attention_v2_fp16_384_64_kernel_sm75_cubin,
     fused_multihead_attention_v2_fp16_384_64_kernel_sm75_cubin_len,
     "fused_multihead_attention_v2_fp16_384_64_kernel_sm75", 53248, 256, 0, false},

#if CUDA_VERSION >= 11000

    // Ampere
    {DATA_TYPE_FP16, 64, 64, kSM_80, fused_multihead_attention_v2_fp16_64_64_kernel_sm80_cubin,
     fused_multihead_attention_v2_fp16_64_64_kernel_sm80_cubin_len,
     "fused_multihead_attention_v2_fp16_64_64_kernel_sm80", 32768, 128, 0, false},
    {DATA_TYPE_FP16, 96, 64, kSM_80, fused_multihead_attention_v2_fp16_96_64_kernel_sm80_cubin,
     fused_multihead_attention_v2_fp16_96_64_kernel_sm80_cubin_len,
     "fused_multihead_attention_v2_fp16_96_64_kernel_sm80", 49152, 128, 0, false},
    {DATA_TYPE_FP16, 128, 64, kSM_80, fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin,
     fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin_len,
     "fused_multihead_attention_v2_fp16_128_64_kernel_sm80_noloop", 40960, 128, 32, false},
    {DATA_TYPE_FP16, 128, 64, kSM_80, fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin,
     fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin_len,
     "fused_multihead_attention_v2_fp16_128_64_kernel_sm80", 65536, 128, 0, false},
    {DATA_TYPE_FP16, 256, 64, kSM_80, fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin,
     fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin_len,
     "fused_multihead_attention_v2_fp16_256_64_kernel_sm80_noloop", 73728, 128, 32, false},
    {DATA_TYPE_FP16, 256, 64, kSM_80, fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin,
     fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin_len,
     "fused_multihead_attention_v2_fp16_256_64_kernel_sm80", 73728, 128, 0, false},
    {DATA_TYPE_FP16, 384, 64, kSM_80, fused_multihead_attention_v2_fp16_384_64_kernel_sm80_cubin,
     fused_multihead_attention_v2_fp16_384_64_kernel_sm80_cubin_len,
     "fused_multihead_attention_v2_fp16_384_64_kernel_sm80_noloop", 114688, 256, 48, false},
    {DATA_TYPE_FP16, 384, 64, kSM_80, fused_multihead_attention_v2_fp16_384_64_kernel_sm80_cubin,
     fused_multihead_attention_v2_fp16_384_64_kernel_sm80_cubin_len,
     "fused_multihead_attention_v2_fp16_384_64_kernel_sm80", 114688, 256, 0, false},
    {DATA_TYPE_FP16, 512, 64, kSM_80, cubin_fmha_v2_fp16_512_64_sm80_cu_cubin,
     cubin_fmha_v2_fp16_512_64_sm80_cu_cubin_len, "fmha_v2_fp16_512_64_sm80_kernel", 73728, 256, 0, false},
    {DATA_TYPE_FP16, 512, 64, kSM_80, cubin_fmha_v2_fp16_512_64_sm80_cu_cubin,
     cubin_fmha_v2_fp16_512_64_sm80_cu_cubin_len, "fmha_v2_fp16_512_64_sm80_kernel_nl", 73728, 256, 32, false},

    // GA10x
    // Note: For GA10X keep only kernels whose sharedMemBytes < 100KiB
    {DATA_TYPE_FP16, 64, 64, kSM_86, fused_multihead_attention_v2_fp16_64_64_kernel_sm86_cubin,
     fused_multihead_attention_v2_fp16_64_64_kernel_sm86_cubin_len,
     "fused_multihead_attention_v2_fp16_64_64_kernel_sm80", 32768, 128, 0, false},
    {DATA_TYPE_FP16, 96, 64, kSM_86, fused_multihead_attention_v2_fp16_96_64_kernel_sm86_cubin,
     fused_multihead_attention_v2_fp16_96_64_kernel_sm86_cubin_len,
     "fused_multihead_attention_v2_fp16_96_64_kernel_sm80", 49152, 128, 0, false},
    {DATA_TYPE_FP16, 128, 64, kSM_86, fused_multihead_attention_v2_fp16_128_64_kernel_sm86_cubin,
     fused_multihead_attention_v2_fp16_128_64_kernel_sm86_cubin_len,
     "fused_multihead_attention_v2_fp16_128_64_kernel_sm80_noloop", 40960, 128, 32, false},
    {DATA_TYPE_FP16, 128, 64, kSM_86, fused_multihead_attention_v2_fp16_128_64_kernel_sm86_cubin,
     fused_multihead_attention_v2_fp16_128_64_kernel_sm86_cubin_len,
     "fused_multihead_attention_v2_fp16_128_64_kernel_sm80", 65536, 128, 0, false},
    {DATA_TYPE_FP16, 256, 64, kSM_86, fused_multihead_attention_v2_fp16_256_64_kernel_sm86_cubin,
     fused_multihead_attention_v2_fp16_256_64_kernel_sm86_cubin_len,
     "fused_multihead_attention_v2_fp16_256_64_kernel_sm80_noloop", 73728, 128, 32, false},
    {DATA_TYPE_FP16, 256, 64, kSM_86, fused_multihead_attention_v2_fp16_256_64_kernel_sm86_cubin,
     fused_multihead_attention_v2_fp16_256_64_kernel_sm86_cubin_len,
     "fused_multihead_attention_v2_fp16_256_64_kernel_sm80", 73728, 128, 0, false},
    {DATA_TYPE_FP16, 384, 64, kSM_86, fused_multihead_attention_v2_fp16_384_64_kernel_sm86_cubin,
     fused_multihead_attention_v2_fp16_384_64_kernel_sm86_cubin_len,
     "fused_multihead_attention_v2_fp16_384_64_kernel_sm80_noloop", 65536, 256, 48, false},
    {DATA_TYPE_FP16, 384, 64, kSM_86, fused_multihead_attention_v2_fp16_384_64_kernel_sm86_cubin,
     fused_multihead_attention_v2_fp16_384_64_kernel_sm86_cubin_len,
     "fused_multihead_attention_v2_fp16_384_64_kernel_sm80", 65536, 256, 0, false},
#endif
};

class FusedMultiHeadAttentionXMMAKernelV2
    : public TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoV2,
                                                Fused_multihead_attention_params_v2> {
 public:
  FusedMultiHeadAttentionXMMAKernelV2(
      const FusedMultiHeadAttentionKernelMetaInfoV2* pMetaStart, uint32_t nMetaCount, Data_type type, uint32_t sm)
      : TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoV2,
                                           Fused_multihead_attention_params_v2>(pMetaStart, nMetaCount, type, sm) {
  }

  inline uint64_t hashID(uint32_t s, uint32_t headsize, bool interleaved, bool unroll) const {
    // we only have 30 bits room for head size
    ORT_ENFORCE(headsize <= 0x3FFFFFFF);
    return static_cast<uint64_t>(s) << 32 | (headsize << 2) | (interleaved ? 2U : 0U) | (unroll ? 1U : 0U);
  }

  virtual uint64_t hashID(const KernelMeta& kernelMeta) const {
    return hashID(kernelMeta.mS, kernelMeta.mD, kernelMeta.mInterleaved, kernelMeta.mUnrollStep);
  }

  virtual void run(Fused_multihead_attention_params_v2& params, cudaStream_t ss) const {
    if (params.interleaved) {
      ORT_ENFORCE(mDataType == DATA_TYPE_INT8);
    }

    bool forceUnroll = params.force_unroll;
    if (!forceUnroll && !params.ignore_b1opt && mSM >= kSM_75) {
      const struct {
        uint32_t mSM;
        Data_type mDataType;
        int32_t mS;
        int32_t mMaxBatch;
      } unrollList[] = {
        {kSM_75, DATA_TYPE_FP16, 256, 1},
        {kSM_75, DATA_TYPE_FP16, 384, 1},
#if CUDA_VERSION >= 11000
        {kSM_80, DATA_TYPE_FP16, 128, 4},
        {kSM_80, DATA_TYPE_FP16, 256, 4},
        {kSM_80, DATA_TYPE_FP16, 384, 4},

        {kSM_86, DATA_TYPE_FP16, 128, 4},
        {kSM_86, DATA_TYPE_FP16, 256, 4},
#endif
      };
      for (uint32_t i = 0u; i < sizeof(unrollList) / sizeof(unrollList[0]); ++i) {
        if (mSM == unrollList[i].mSM &&
            mDataType == unrollList[i].mDataType &&
            params.s == unrollList[i].mS &&
            params.b <= unrollList[i].mMaxBatch) {
          forceUnroll = true;
          break;
        }
      }
    }

    const auto findIter = mFunctions.find(hashID(params.s, params.d, params.interleaved, forceUnroll));
    // Provide debug information if the kernel is missing in the pool.
    std::stringstream configss;
    configss << "s: " << params.s << " d: " << params.d << " interleaved?: "
             << params.interleaved << " forceUnroll?: " << forceUnroll;
    ORT_ENFORCE(findIter != mFunctions.end(), configss.str().c_str());

    const auto& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
    const CUfunction func = findIter->second.mDeviceFunction;

    void* kernelParams[] = {&params, nullptr};
    if (!forceUnroll) {
      cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                                        kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                 mDriver);
    } else {
      int32_t unroll = kernelMeta.mS / kernelMeta.mUnrollStep;
      ORT_ENFORCE(kernelMeta.mS == kernelMeta.mUnrollStep * unroll);
      cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, unroll, kernelMeta.mThreadsPerCTA, 1, 1,
                                        kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                 mDriver);
    }
  }
};

using FusedMHAKernelFactoryV2 = TFusedMHAKernelFactory<FusedMultiHeadAttentionXMMAKernelV2>;

inline const FusedMultiHeadAttentionXMMAKernelV2* getXMMAKernelsV2(Data_type type, uint32_t sm) {
  return FusedMHAKernelFactoryV2::Get().getXMMAKernels(
      sMhaKernelMetaInfosV2, sizeof(sMhaKernelMetaInfosV2) / sizeof(sMhaKernelMetaInfosV2[0]), type, sm);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
