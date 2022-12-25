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

extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_40_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_64_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin[];

extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_40_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_64_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin_len;

constexpr int32_t S{0};

static const struct FusedMultiHeadFlashAttentionKernelMetaInfoV2
{
    Data_type mDataType;
    int32_t mS;
    int32_t mQStep;
    int32_t mKVStep;
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
{ DATA_TYPE_FP16, S, 64, 64, 16, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_16_sm75_kernel", 6144, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 64, 16, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_16_sm75_kernel_nl", 6144, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 64, 32, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_32_sm75_kernel", 12288, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 64, 32, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_32_sm75_kernel_nl", 12288, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 64, 40, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_40_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_40_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_40_sm75_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 64, 40, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_40_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_40_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_40_sm75_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 64, 64, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_64_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_64_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_64_sm75_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 64, 64, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_64_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_64_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_64_sm75_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm75_kernel", 32768, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm75_kernel_nl", 32768, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm75_kernel", 32768, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm75_kernel_nl", 32768, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm75_kernel", 65536, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm75_kernel_nl", 65536, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm75_kernel", 65536, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm75_kernel_nl", 65536, 128, 64, false },

{ DATA_TYPE_FP16, S, 64, 32, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_16_sm80_kernel", 8192, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_16_sm80_kernel_nl", 8192, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_16_sm80_kernel", 12288, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_16_sm80_kernel_nl", 12288, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_32_sm80_kernel", 12288, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_32_sm80_kernel_nl", 12288, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_32_sm80_kernel", 20480, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_32_sm80_kernel_nl", 20480, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm80_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm80_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_40_sm80_kernel", 40960, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_40_sm80_kernel_nl", 40960, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm80_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm80_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_64_sm80_kernel", 40960, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_64_sm80_kernel_nl", 40960, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm80_kernel", 49152, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm80_kernel_nl", 49152, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 32, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_80_sm80_kernel", 81920, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 32, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_80_sm80_kernel_nl", 81920, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm80_kernel", 49152, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm80_kernel_nl", 49152, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 32, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_128_sm80_kernel", 81920, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 32, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_128_sm80_kernel_nl", 81920, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm80_kernel", 98304, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm80_kernel_nl", 98304, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm80_kernel", 98304, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm80_kernel_nl", 98304, 128, 64, false },

{ DATA_TYPE_FP16, S, 64, 32, 16, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_16_sm86_kernel", 8192, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 16, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_16_sm86_kernel_nl", 8192, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 16, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_16_sm86_kernel", 12288, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 16, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_16_sm86_kernel_nl", 12288, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 32, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_32_sm86_kernel", 12288, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 32, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_32_sm86_kernel_nl", 12288, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 32, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_32_sm86_kernel", 20480, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 32, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_32_sm86_kernel_nl", 20480, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 40, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm86_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 40, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm86_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 40, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_40_sm86_kernel", 40960, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 40, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_40_sm86_kernel_nl", 40960, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 64, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm86_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 64, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm86_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 64, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_64_sm86_kernel", 40960, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 64, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_64_sm86_kernel_nl", 40960, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm86_kernel", 49152, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm86_kernel_nl", 49152, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 32, 80, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_80_sm86_kernel", 81920, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 32, 80, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_80_sm86_kernel_nl", 81920, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm86_kernel", 49152, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm86_kernel_nl", 49152, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 32, 128, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_128_sm86_kernel", 81920, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 32, 128, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_128_sm86_kernel_nl", 81920, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm86_kernel", 98304, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm86_kernel_nl", 98304, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm86_kernel", 98304, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm86_kernel_nl", 98304, 128, 64, false },

{ DATA_TYPE_FP16, S, 64, 32, 16, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_16_sm89_kernel", 8192, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 16, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_16_sm89_kernel_nl", 8192, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 16, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_16_sm89_kernel", 12288, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 16, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_16_sm89_kernel_nl", 12288, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 32, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_32_sm89_kernel", 12288, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 32, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_32_sm89_kernel_nl", 12288, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 32, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_32_sm89_kernel", 20480, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 32, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_32_sm89_kernel_nl", 20480, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 40, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm89_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 40, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm89_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 40, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_40_sm89_kernel", 40960, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 40, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_40_sm89_kernel_nl", 40960, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 64, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm89_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 64, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm89_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 64, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_64_sm89_kernel", 40960, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 64, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_64_sm89_kernel_nl", 40960, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm89_kernel", 49152, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm89_kernel_nl", 49152, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 32, 80, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_80_sm89_kernel", 81920, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 32, 80, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_80_sm89_kernel_nl", 81920, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm89_kernel", 49152, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm89_kernel_nl", 49152, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 32, 128, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_128_sm89_kernel", 81920, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 32, 128, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_128_sm89_kernel_nl", 81920, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm89_kernel", 98304, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm89_kernel_nl", 98304, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm89_kernel", 98304, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm89_kernel_nl", 98304, 128, 64, false }
};

static Fused_multihead_attention_params_v2 getMHFAParams(
    int32_t b, int32_t s, int32_t h, int32_t d,
    void const* qkv_packed_d, void* cu_seqlens_d, void* o_packed_d)
{
    Fused_multihead_attention_params_v2 params{};

    // Set the pointers.
    params.qkv_ptr = const_cast<void*>(qkv_packed_d);
    params.qkv_stride_in_bytes = h * 3 * d * sizeof(half);
    params.o_ptr = o_packed_d;
    params.o_stride_in_bytes = h * d * sizeof(half);;
    params.cu_seqlens = static_cast<int*>(cu_seqlens_d);

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s = s;
    params.d = d;

    // Set the different scale values.
    const float scale_bmm1 = 1.f / sqrtf(static_cast<float>(d));
    constexpr float scale_softmax = 1.f;
    constexpr float scale_bmm2 = 1.f;
    set_alpha_fp16(params.scale_bmm1, scale_bmm1);
    set_alpha_fp16(params.scale_softmax, scale_softmax);
    set_alpha_fp16(params.scale_bmm2, scale_bmm2);

    // Set flags
    params.interleaved = false;
    params.ignore_b1opt = false;
    params.force_unroll = true;
    params.use_int8_scale_max = false;

    return params;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
class FusedMultiHeadFlashAttentionKernel
    : public TSharedCubinKernel<FusedMultiHeadFlashAttentionKernelMetaInfoV2, Fused_multihead_attention_params_v2>
{
public:
    FusedMultiHeadFlashAttentionKernel(FusedMultiHeadFlashAttentionKernelMetaInfoV2 const* pMetaStart,
        int32_t nMetaCount, Data_type type, int32_t sm)
        : TSharedCubinKernel<FusedMultiHeadFlashAttentionKernelMetaInfoV2, Fused_multihead_attention_params_v2>(
            pMetaStart, nMetaCount, type, sm)
    {
    }

    uint64_t hashID(int32_t headsize, int32_t qStep, int32_t kvStep, bool interleaved, bool unroll) const
    {
        // we only have 30 bits room for head size
        ORT_ENFORCE(headsize <= 0x3FFFFFFF);
        ORT_ENFORCE(qStep <= 0xFFFF);
        ORT_ENFORCE(kvStep <= 0xFFFF);
        return static_cast<uint64_t>(qStep << 16 | kvStep) << 32 | (headsize << 2) | (interleaved ? 2U : 0U)
            | (unroll ? 1U : 0U);
    }

    uint64_t hashID(Fused_multihead_attention_params_v2 const& param) const
    {
        bool const isSmallBS = param.b * param.h < 64;
        bool const isSM75 = mSM == 75;
        int32_t qStep{64};
        int32_t kvStep{16};
        switch (param.d)
        {
        case 16:
        case 32:
        case 40:
        case 64:
            qStep = isSM75 ? 64 : (isSmallBS ? 64 : 128);
            kvStep = isSM75 ? 64 : (isSmallBS ? 32 : 16);
            break;
        case 80:
        case 128:
            qStep = isSM75 ? 64 : (isSmallBS ? 64 : 128);
            kvStep = isSM75 ? 32 : (isSmallBS ? 32 : 32);
            break;
        default: break;
        }
        return hashID(param.d, qStep, kvStep, param.interleaved, param.force_unroll);
    }

    uint64_t hashID(KernelMeta const& kernelMeta) const
    {
        return hashID(
            kernelMeta.mD, kernelMeta.mQStep, kernelMeta.mKVStep, kernelMeta.mInterleaved, kernelMeta.mUnrollStep > 0);
    }
};

using FusedMHAFlashKernelFactory = TSharedCubinKernelFactory<FusedMultiHeadFlashAttentionKernel>;

inline FusedMultiHeadFlashAttentionKernel const* getFMHAFlashCubinKernels(Data_type type, int32_t sm)
{
    return FusedMHAFlashKernelFactory::Get().getCubinKernels(
        sMhaKernelMetaInfos, sizeof(sMhaKernelMetaInfos) / sizeof(sMhaKernelMetaInfos[0]), type, sm);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
