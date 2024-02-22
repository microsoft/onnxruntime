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

#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/fused_multihead_attention_common.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/fused_multihead_attention_v2.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/flash_attention/sharedCubinLoader.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Do not modify this, it is integrated from src/fused_multihead_attention_demo_bert_params.h in fmha_v2.
////////////////////////////////////////////////////////////////////////////////////////////////////
struct Gmem_params {
  // The matrix.
  void* ptr;

  // The stride between rows of the Q, K and V matrices.
  int64_t stride_in_bytes;

  // The number of heads
  int32_t h;

  // Hidden dim per head
  int32_t d;

  // array of length b+1 holding prefix sum of actual sequence lenghts.
  int32_t* cu_seqlens;
};

struct Fused_multihead_attention_params_mhca {
  // The QKV matrices.
  void* qkv_ptr;
  // The mask to implement drop-out.
  void* packed_mask_ptr;
  // The O matrix (output).
  void* o_ptr;

  // The stride between rows of the Q, K and V matrices.
  int64_t qkv_stride_in_bytes;
  // The stride between matrices of packed mask.
  int64_t packed_mask_stride_in_bytes;
  // The stride between rows of O.
  int64_t o_stride_in_bytes;

  // The dimensions.
  int32_t b, h, s, d;
  // The scaling factors for the kernel.
  uint32_t scale_bmm1, scale_softmax, scale_bmm2;

  // Do we use Niall's trick to avoid I2F/F2I in the INT8 kernel.
  // See https://confluence.nvidia.com/pages/viewpage.action?pageId=302779721 for details.
  bool enable_i2f_trick;

  // array of length b+1 holding prefix sum of actual sequence lenghts
  int32_t* cu_seqlens;

  // use C/32 Format.
  bool interleaved = false;
  bool ignore_b1opt = false;
  bool force_unroll = false;
  bool use_int8_scale_max = false;

  // Sequence length of Q
  int32_t s_q;
  int32_t d_padded;

  Gmem_params gmem_q_params;
  Gmem_params gmem_kv_params;

  void clear() {
    qkv_ptr = nullptr;
    packed_mask_ptr = nullptr;
    o_ptr = nullptr;

    qkv_stride_in_bytes = 0;
    packed_mask_stride_in_bytes = 0;
    o_stride_in_bytes = 0;

    b = 0;
    h = 0;
    s = 0;
    d = 0;
    // The scaling factors for the kernel.
    scale_bmm1 = 0;
    scale_softmax = 0;
    scale_bmm2 = 0;

    enable_i2f_trick = false;

    cu_seqlens = nullptr;
    interleaved = false;
    ignore_b1opt = false;
    force_unroll = false;
    use_int8_scale_max = false;

    s_q = 0;
    d_padded = 0;
  }
};

extern unsigned char cubin_fmha_mhca_fp16_128_64_sm75_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_64_sm86_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_64_sm89_cu_cubin[];

extern unsigned char cubin_fmha_mhca_fp16_128_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_128_sm89_cu_cubin[];

extern unsigned char cubin_fmha_mhca_fp16_128_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_256_sm86_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_256_sm89_cu_cubin[];

extern uint32_t cubin_fmha_mhca_fp16_128_64_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_64_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_64_sm89_cu_cubin_len;

extern uint32_t cubin_fmha_mhca_fp16_128_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_128_sm89_cu_cubin_len;

extern uint32_t cubin_fmha_mhca_fp16_128_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_256_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_256_sm89_cu_cubin_len;

static const struct FusedMultiHeadCrossAttentionKernelMetaInfoV2 {
  Data_type mDataType;
  int32_t mS;
  int32_t mD;
  int32_t mSM;
  unsigned char const* mCubin;
  uint32_t mCubinSize;
  char const* mFuncName;
  int32_t mSharedMemBytes;
  int32_t mThreadsPerCTA;
  int32_t mUnrollStep;
  bool mInterleaved;
} sMhaKernelMetaInfos[] = {
    {DATA_TYPE_FP16, 128, 64, kSM_75, cubin_fmha_mhca_fp16_128_64_sm75_cu_cubin, cubin_fmha_mhca_fp16_128_64_sm75_cu_cubin_len, "fmha_mhca_fp16_128_64_sm75_kernel", 40960, 128, 0, false},
    {DATA_TYPE_FP16, 128, 64, kSM_75, cubin_fmha_mhca_fp16_128_64_sm75_cu_cubin, cubin_fmha_mhca_fp16_128_64_sm75_cu_cubin_len, "fmha_mhca_fp16_128_64_sm75_kernel_nl", 36864, 128, 32, false},

    {DATA_TYPE_FP16, 128, 64, kSM_80, cubin_fmha_mhca_fp16_128_64_sm80_cu_cubin, cubin_fmha_mhca_fp16_128_64_sm80_cu_cubin_len, "fmha_mhca_fp16_128_64_sm80_kernel", 49152, 128, 0, false},
    {DATA_TYPE_FP16, 128, 64, kSM_80, cubin_fmha_mhca_fp16_128_64_sm80_cu_cubin, cubin_fmha_mhca_fp16_128_64_sm80_cu_cubin_len, "fmha_mhca_fp16_128_64_sm80_kernel_nl", 49152, 128, 64, false},
    {DATA_TYPE_FP16, 128, 128, kSM_80, cubin_fmha_mhca_fp16_128_128_sm80_cu_cubin, cubin_fmha_mhca_fp16_128_128_sm80_cu_cubin_len, "fmha_mhca_fp16_128_128_sm80_kernel", 98304, 128, 0, false},
    {DATA_TYPE_FP16, 128, 128, kSM_80, cubin_fmha_mhca_fp16_128_128_sm80_cu_cubin, cubin_fmha_mhca_fp16_128_128_sm80_cu_cubin_len, "fmha_mhca_fp16_128_128_sm80_kernel_nl", 81920, 128, 32, false},
    {DATA_TYPE_FP16, 128, 256, kSM_80, cubin_fmha_mhca_fp16_128_256_sm80_cu_cubin, cubin_fmha_mhca_fp16_128_256_sm80_cu_cubin_len, "fmha_mhca_fp16_128_256_sm80_kernel", 163840, 256, 0, false},
    {DATA_TYPE_FP16, 128, 256, kSM_80, cubin_fmha_mhca_fp16_128_256_sm80_cu_cubin, cubin_fmha_mhca_fp16_128_256_sm80_cu_cubin_len, "fmha_mhca_fp16_128_256_sm80_kernel_nl", 147456, 256, 16, false},

    {DATA_TYPE_FP16, 128, 64, kSM_86, cubin_fmha_mhca_fp16_128_64_sm86_cu_cubin, cubin_fmha_mhca_fp16_128_64_sm86_cu_cubin_len, "fmha_mhca_fp16_128_64_sm86_kernel", 49152, 128, 0, false},
    {DATA_TYPE_FP16, 128, 64, kSM_86, cubin_fmha_mhca_fp16_128_64_sm86_cu_cubin, cubin_fmha_mhca_fp16_128_64_sm86_cu_cubin_len, "fmha_mhca_fp16_128_64_sm86_kernel_nl", 49152, 128, 64, false},
    {DATA_TYPE_FP16, 128, 128, kSM_86, cubin_fmha_mhca_fp16_128_128_sm86_cu_cubin, cubin_fmha_mhca_fp16_128_128_sm86_cu_cubin_len, "fmha_mhca_fp16_128_128_sm86_kernel", 98304, 128, 0, false},
    {DATA_TYPE_FP16, 128, 128, kSM_86, cubin_fmha_mhca_fp16_128_128_sm86_cu_cubin, cubin_fmha_mhca_fp16_128_128_sm86_cu_cubin_len, "fmha_mhca_fp16_128_128_sm86_kernel_nl", 98304, 128, 64, false},
    {DATA_TYPE_FP16, 128, 256, kSM_86, cubin_fmha_mhca_fp16_128_256_sm86_cu_cubin, cubin_fmha_mhca_fp16_128_256_sm86_cu_cubin_len, "fmha_mhca_fp16_128_256_sm86_kernel", 163840, 256, 0, false},
    {DATA_TYPE_FP16, 128, 256, kSM_86, cubin_fmha_mhca_fp16_128_256_sm86_cu_cubin, cubin_fmha_mhca_fp16_128_256_sm86_cu_cubin_len, "fmha_mhca_fp16_128_256_sm86_kernel_nl", 81920, 256, 16, false},

    {DATA_TYPE_FP16, 128, 64, kSM_89, cubin_fmha_mhca_fp16_128_64_sm89_cu_cubin, cubin_fmha_mhca_fp16_128_64_sm89_cu_cubin_len, "fmha_mhca_fp16_128_64_sm89_kernel", 49152, 128, 0, false},
    {DATA_TYPE_FP16, 128, 64, kSM_89, cubin_fmha_mhca_fp16_128_64_sm89_cu_cubin, cubin_fmha_mhca_fp16_128_64_sm89_cu_cubin_len, "fmha_mhca_fp16_128_64_sm89_kernel_nl", 49152, 128, 64, false},
    {DATA_TYPE_FP16, 128, 128, kSM_89, cubin_fmha_mhca_fp16_128_128_sm89_cu_cubin, cubin_fmha_mhca_fp16_128_128_sm89_cu_cubin_len, "fmha_mhca_fp16_128_128_sm89_kernel", 98304, 128, 0, false},
    {DATA_TYPE_FP16, 128, 128, kSM_89, cubin_fmha_mhca_fp16_128_128_sm89_cu_cubin, cubin_fmha_mhca_fp16_128_128_sm89_cu_cubin_len, "fmha_mhca_fp16_128_128_sm89_kernel_nl", 81920, 128, 32, false},
    {DATA_TYPE_FP16, 128, 256, kSM_89, cubin_fmha_mhca_fp16_128_256_sm89_cu_cubin, cubin_fmha_mhca_fp16_128_256_sm89_cu_cubin_len, "fmha_mhca_fp16_128_256_sm89_kernel", 163840, 256, 0, false},
    {DATA_TYPE_FP16, 128, 256, kSM_89, cubin_fmha_mhca_fp16_128_256_sm89_cu_cubin, cubin_fmha_mhca_fp16_128_256_sm89_cu_cubin_len, "fmha_mhca_fp16_128_256_sm89_kernel_nl", 81920, 256, 16, false}};

static Fused_multihead_attention_params_mhca getMHCAParams(
    // sizes
    int32_t b, int32_t s_q, int32_t s_kv, int32_t h, int32_t d,
    // device pointers
    void const* q_packed_d, void const* kv_packed_d, void* cu_seqlens_q_d, void* cu_seqlens_kv_d, void* o_packed_d) {
  Fused_multihead_attention_params_mhca params{};

  int32_t const d_padded = d <= 64 ? 64 : static_cast<int>(std::pow(2, std::ceil(std::log(d) / std::log(2))));

  // Set the pointers.
  params.o_ptr = o_packed_d;
  params.o_stride_in_bytes = static_cast<int64_t>(h) * d * sizeof(half);

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.s_q = s_q;
  params.s = s_kv;
  params.d = d;
  params.d_padded = d_padded;

  const float scale_bmm1 = 1.f / sqrtf(static_cast<float>(d));
  constexpr float scale_softmax = 1.f;  // Seems to be only required for int8
  constexpr float scale_bmm2 = 1.f;

  // Set the different scale values.
  set_alpha_fp16(params.scale_bmm1, scale_bmm1);
  set_alpha_fp16(params.scale_softmax, scale_softmax);
  set_alpha_fp16(params.scale_bmm2, scale_bmm2);

  // Set the pointers.
  params.gmem_q_params.ptr = const_cast<void*>(q_packed_d);
  params.gmem_q_params.stride_in_bytes = static_cast<int64_t>(h) * d * static_cast<int64_t>(sizeof(half));
  params.gmem_q_params.h = h;
  params.gmem_q_params.d = d;
  params.gmem_q_params.cu_seqlens = static_cast<int32_t*>(cu_seqlens_q_d);

  params.gmem_kv_params.ptr = const_cast<void*>(kv_packed_d);
  params.gmem_kv_params.stride_in_bytes = static_cast<int64_t>(h) * 2 * d * static_cast<int64_t>(sizeof(half));
  params.gmem_kv_params.h = h;
  params.gmem_kv_params.d = d;
  params.gmem_kv_params.cu_seqlens = static_cast<int32_t*>(cu_seqlens_kv_d);

  // Set flags
  params.force_unroll = true;

  return params;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
class FusedMultiHeadCrossAttentionKernel
    : public TSharedCubinKernel<FusedMultiHeadCrossAttentionKernelMetaInfoV2, Fused_multihead_attention_params_mhca> {
 public:
  FusedMultiHeadCrossAttentionKernel(FusedMultiHeadCrossAttentionKernelMetaInfoV2 const* pMetaStart,
                                     int32_t nMetaCount, Data_type type, int32_t sm)
      : TSharedCubinKernel<FusedMultiHeadCrossAttentionKernelMetaInfoV2, Fused_multihead_attention_params_mhca>(
            pMetaStart, nMetaCount, type, sm) {
  }

  uint64_t hashID(int32_t headsize, bool interleaved, bool unroll) const {
    // we only have 30 bits room for head size
    ORT_ENFORCE(headsize <= 0x3FFFFFFF);
    return (headsize << 2) | (interleaved ? 2U : 0U) | (unroll ? 1U : 0U);
  }

  uint64_t hashID(Fused_multihead_attention_params_mhca const& param) const {
    return hashID(param.d_padded, param.interleaved, param.force_unroll);
  }

  uint64_t hashID(KernelMeta const& kernelMeta) const {
    return hashID(kernelMeta.mD, kernelMeta.mInterleaved, kernelMeta.mUnrollStep > 0);
  }

  void dumpHashId(Fused_multihead_attention_params_mhca const& param, std::ostringstream& message) const override {
    message << "\t d_padded: " << param.d_padded << "\n"
            << "\t interleaved: " << param.interleaved << "\n"
            << "\t force_unroll: " << param.force_unroll << "\n";
  }

  int32_t getSForUnroll(Fused_multihead_attention_params_mhca const& param) const override {
    return param.s_q;
  }
};

using FusedMHACrossKernelFactory = TSharedCubinKernelFactory<FusedMultiHeadCrossAttentionKernel>;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Below are public interface

inline bool has_fused_cross_attention_kernel(int sm, int head_size, int kv_sequence_length) {
  constexpr int min_head_size = 0;
  const int max_head_size = (sm == 75 ? 64 : 256);
  return (sm == 75 || sm == 80 || sm == 86 || sm == 89) &&
         (head_size > min_head_size) && (head_size <= max_head_size) &&
         (kv_sequence_length <= 128);  // TODO: shall we remove this constraint on kv_sequence_length?
}

inline FusedMultiHeadCrossAttentionKernel const* get_fused_cross_attention_kernels(int32_t sm) {
  return FusedMHACrossKernelFactory::Get().getCubinKernels(
      sMhaKernelMetaInfos, sizeof(sMhaKernelMetaInfos) / sizeof(sMhaKernelMetaInfos[0]), DATA_TYPE_FP16, sm);
}

inline void run_fused_cross_attention(
    void const* devQ,                                   // Q in device
    void const* devKV,                                  // KV in device
    void* cuSeqlensQ,                                   // cumulated sequence length of Q in device
    void* cuSeqlensKV,                                  // cumulated sequence length of KV in device
    void* devOutput,                                    // output in device
    FusedMultiHeadCrossAttentionKernel const* kernels,  // kernels
    int32_t b = 2,                                      // batch size
    int32_t h = 8,                                      // number of heads
    int32_t d = 64,                                     // head size
    int32_t seqQ = 4096,                                // sequence length of Q
    int32_t seqKV = 77,                                 // sequence lenth of KV
    cudaStream_t stream = 0) {                          // cuda stream
  Fused_multihead_attention_params_mhca params = getMHCAParams(
      b, seqQ, seqKV, h, d, devQ, devKV, cuSeqlensQ, cuSeqlensKV, devOutput);

  kernels->run(params, stream);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
