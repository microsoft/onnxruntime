/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "fused_multihead_attention.h"
#include "fused_multihead_attention_common.h"
#include <assert.h>
#include <stdint.h>

namespace fastertransformer
{
struct Fused_multihead_attention_params_v2
{
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

#if defined(STORE_P)
    // The pointer to the P matrix (for debugging).
    void* p_ptr;
    // The stride between rows of the P matrix (for debugging).
    int64_t p_stride_in_bytes;
#endif // defined(STORE_P)

#if defined(STORE_S)
    // The pointer to the S matrix (for debugging).
    void* s_ptr;
    // The stride between rows of the S matrix (for debugging).
    int64_t s_stride_in_bytes;
#endif // defined(STORE_S)

    // The dimensions.
    int b, h, s, d;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    // Do we use Niall's trick to avoid I2F/F2I in the INT8 kernel.
    // See https://confluence.nvidia.com/pages/viewpage.action?pageId=302779721 for details.
    bool enable_i2f_trick;

    // array of length b+1 holding prefix sum of actual sequence lenghts
    int* cu_seqlens;

    // use C/32 Format.
    bool interleaved = false;
    bool ignore_b1opt = false;
    bool force_unroll = false;
    bool use_int8_scale_max = false;

    void clear()
    {
        qkv_ptr = nullptr;
        packed_mask_ptr = nullptr;
        o_ptr = nullptr;

        qkv_stride_in_bytes = 0;
        packed_mask_stride_in_bytes = 0;
        o_stride_in_bytes = 0;
#if defined(STORE_P)
        p_ptr = nullptr;
        p_stride_in_bytes = 0
#endif // defined(STORE_P)

#if defined(STORE_S)
            s_ptr
            = nullptr;
        s_stride_in_bytes = 0;
#endif // defined(STORE_S)

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
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////
extern unsigned char fused_multihead_attention_v2_fp16_128_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_256_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_384_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_384_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_64_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_64_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_96_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_96_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_64_64_kernel_sm70_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_96_64_kernel_sm70_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_128_64_kernel_sm70_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_256_64_kernel_sm70_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_384_64_kernel_sm70_cubin[];

extern unsigned char fused_multihead_attention_v2_int8_64_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_64_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_128_64_kernel_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_192_64_kernel_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_256_64_kernel_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_384_64_kernel_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin[];

extern unsigned int fused_multihead_attention_v2_fp16_128_64_kernel_sm75_cubin_len;
extern unsigned int fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin_len;
extern unsigned int fused_multihead_attention_v2_fp16_256_64_kernel_sm75_cubin_len;
extern unsigned int fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin_len;
extern unsigned int fused_multihead_attention_v2_fp16_384_64_kernel_sm75_cubin_len;
extern unsigned int fused_multihead_attention_v2_fp16_384_64_kernel_sm80_cubin_len;
extern unsigned int fused_multihead_attention_v2_fp16_64_64_kernel_sm75_cubin_len;
extern unsigned int fused_multihead_attention_v2_fp16_64_64_kernel_sm80_cubin_len;
extern unsigned int fused_multihead_attention_v2_fp16_96_64_kernel_sm75_cubin_len;
extern unsigned int fused_multihead_attention_v2_fp16_96_64_kernel_sm80_cubin_len;
extern unsigned int fused_multihead_attention_v2_fp16_96_64_kernel_sm70_cubin_len;
extern unsigned int fused_multihead_attention_v2_fp16_64_64_kernel_sm70_cubin_len;
extern unsigned int fused_multihead_attention_v2_fp16_128_64_kernel_sm70_cubin_len;
extern unsigned int fused_multihead_attention_v2_fp16_256_64_kernel_sm70_cubin_len;
extern unsigned int fused_multihead_attention_v2_fp16_384_64_kernel_sm70_cubin_len;
extern unsigned int fused_multihead_attention_v2_int8_64_64_kernel_sm75_cubin_len;
extern unsigned int fused_multihead_attention_v2_int8_64_64_kernel_sm80_cubin_len;;
extern unsigned int fused_multihead_attention_v2_int8_128_64_kernel_cubin_len;
extern unsigned int fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin_len;
extern unsigned int fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin_len;
extern unsigned int fused_multihead_attention_v2_int8_192_64_kernel_cubin_len;
extern unsigned int fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin_len;
extern unsigned int fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin_len;
extern unsigned int fused_multihead_attention_v2_int8_256_64_kernel_cubin_len;
extern unsigned int fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin_len;
extern unsigned int fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin_len;
extern unsigned int fused_multihead_attention_v2_int8_384_64_kernel_cubin_len;
extern unsigned int fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin_len;
extern unsigned int fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin_len;

static const struct FusedMultiHeadAttentionKernelMetaInfoV2
{
    Data_type mDataType;
    unsigned int mS;
    unsigned int mD;
    unsigned int mSM;
    const unsigned char* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    unsigned int mSharedMemBytes;
    unsigned int mThreadsPerCTA;
    unsigned int mUnrollStep;
    bool mInterleaved;
} sMhaKernelMetaInfosV2[] = {
    // Xavier
    {DATA_TYPE_INT8, 128, 64, kSM_72, fused_multihead_attention_v2_int8_128_64_kernel_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm72_interleaved", 24576, 128, 0, true},
    {DATA_TYPE_INT8, 128, 64, kSM_72, fused_multihead_attention_v2_int8_128_64_kernel_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm72", 32768, 128, 0, false},
    {DATA_TYPE_INT8, 192, 64, kSM_72, fused_multihead_attention_v2_int8_192_64_kernel_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm72_interleaved", 28672, 128, 0, true},
    {DATA_TYPE_INT8, 192, 64, kSM_72, fused_multihead_attention_v2_int8_192_64_kernel_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm72", 45056, 128, 0, false},
    {DATA_TYPE_INT8, 256, 64, kSM_72, fused_multihead_attention_v2_int8_256_64_kernel_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm72_interleaved", 36864, 128, 0, true},
    {DATA_TYPE_INT8, 256, 64, kSM_72, fused_multihead_attention_v2_int8_256_64_kernel_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm72", 57344, 128, 0, false},
    {DATA_TYPE_INT8, 384, 64, kSM_72, fused_multihead_attention_v2_int8_384_64_kernel_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm72_interleaved", 51200, 128, 0, true},
    {DATA_TYPE_INT8, 384, 64, kSM_72, fused_multihead_attention_v2_int8_384_64_kernel_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm72", 77824, 128, 0, false},
    //Volta
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

    {DATA_TYPE_INT8, 64, 64, kSM_75, fused_multihead_attention_v2_int8_64_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_64_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_64_64_kernel_sm75", 20480, 128, 0, false},
    {DATA_TYPE_INT8, 128, 64, kSM_75, fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm75_interleaved_noloop", 18432, 128, 16, true},
    {DATA_TYPE_INT8, 128, 64, kSM_75, fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm75_noloop", 18432, 128, 16, false},
    {DATA_TYPE_INT8, 128, 64, kSM_75, fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm75_interleaved", 24576, 128, 0, true},
    {DATA_TYPE_INT8, 128, 64, kSM_75, fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm75", 24576, 128, 0, false},
    {DATA_TYPE_INT8, 192, 64, kSM_75, fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm75_interleaved_noloop", 28672, 128, 64, true},
    {DATA_TYPE_INT8, 192, 64, kSM_75, fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm75_noloop", 28672, 128, 64, false},
    {DATA_TYPE_INT8, 192, 64, kSM_75, fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm75_interleaved", 28672, 128, 0, true},
    {DATA_TYPE_INT8, 192, 64, kSM_75, fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm75", 28672, 128, 0, false},
    {DATA_TYPE_INT8, 256, 64, kSM_75, fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm75_interleaved_noloop", 34816, 128, 32, true},
    {DATA_TYPE_INT8, 256, 64, kSM_75, fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm75_noloop", 34816, 128, 32, false},
    {DATA_TYPE_INT8, 256, 64, kSM_75, fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm75_interleaved", 34816, 128, 0, true},
    {DATA_TYPE_INT8, 256, 64, kSM_75, fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm75", 34816, 128, 0, false},
    {DATA_TYPE_INT8, 384, 64, kSM_75, fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm75_interleaved_noloop", 51200, 128, 32, true},
    {DATA_TYPE_INT8, 384, 64, kSM_75, fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm75_noloop", 51200, 128, 32, false},
    {DATA_TYPE_INT8, 384, 64, kSM_75, fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm75_interleaved", 51200, 128, 0, true},
    {DATA_TYPE_INT8, 384, 64, kSM_75, fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm75", 51200, 128, 0, false},

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
    {DATA_TYPE_INT8, 64, 64, kSM_80, fused_multihead_attention_v2_int8_64_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_64_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_64_64_kernel_sm80", 24576, 128, 0, false},
    {DATA_TYPE_INT8, 128, 64, kSM_80, fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80_interleaved_noloop", 20480, 128, 16, true},
    {DATA_TYPE_INT8, 128, 64, kSM_80, fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80_noloop", 20480, 128, 16, false},
    {DATA_TYPE_INT8, 128, 64, kSM_80, fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80_interleaved", 24576, 128, 0, true},
    {DATA_TYPE_INT8, 128, 64, kSM_80, fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80", 32768, 128, 0, false},
    {DATA_TYPE_INT8, 192, 64, kSM_80, fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm80_interleaved_noloop", 28672, 128, 32, true},
    {DATA_TYPE_INT8, 192, 64, kSM_80, fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm80_noloop", 28672, 128, 32, false},
    {DATA_TYPE_INT8, 192, 64, kSM_80, fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm80_interleaved", 32768, 128, 0, true},
    {DATA_TYPE_INT8, 192, 64, kSM_80, fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm80", 32768, 128, 0, false},
    {DATA_TYPE_INT8, 256, 64, kSM_80, fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm80_interleaved_noloop", 36864, 128, 32, true},
    {DATA_TYPE_INT8, 256, 64, kSM_80, fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm80_noloop", 36864, 128, 32, false},
    {DATA_TYPE_INT8, 256, 64, kSM_80, fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm80_interleaved", 36864, 128, 0, true},
    {DATA_TYPE_INT8, 256, 64, kSM_80, fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm80", 36864, 128, 0, false},
    {DATA_TYPE_INT8, 384, 64, kSM_80, fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm80_interleaved_noloop", 53248, 128, 32, true},
    {DATA_TYPE_INT8, 384, 64, kSM_80, fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm80_noloop", 53248, 128, 32, false},
    {DATA_TYPE_INT8, 384, 64, kSM_80, fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm80_interleaved", 51200, 128, 0, true},
    {DATA_TYPE_INT8, 384, 64, kSM_80, fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm80", 53248, 128, 0, false},

    // GA10x
    // Note: For GA10X keep only kernels whose sharedMemBytes < 100KiB
    {DATA_TYPE_FP16, 64, 64, kSM_86, fused_multihead_attention_v2_fp16_64_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_fp16_64_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_fp16_64_64_kernel_sm80", 32768, 128, 0, false},
    {DATA_TYPE_FP16, 96, 64, kSM_86, fused_multihead_attention_v2_fp16_96_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_fp16_96_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_fp16_96_64_kernel_sm80", 49152, 128, 0, false},
    {DATA_TYPE_FP16, 128, 64, kSM_86, fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_fp16_128_64_kernel_sm80_noloop", 40960, 128, 32, false},
    {DATA_TYPE_FP16, 128, 64, kSM_86, fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_fp16_128_64_kernel_sm80", 65536, 128, 0, false},
    {DATA_TYPE_FP16, 256, 64, kSM_86, fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_fp16_256_64_kernel_sm80_noloop", 73728, 128, 32, false},
    {DATA_TYPE_FP16, 256, 64, kSM_86, fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_fp16_256_64_kernel_sm80", 73728, 128, 0, false},

    {DATA_TYPE_INT8, 128, 64, kSM_86, fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80_interleaved_noloop", 20480, 128, 16, true},
    {DATA_TYPE_INT8, 128, 64, kSM_86, fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80_noloop", 20480, 128, 16, false},
    {DATA_TYPE_INT8, 128, 64, kSM_86, fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80_interleaved", 24576, 128, 0, true},
    {DATA_TYPE_INT8, 128, 64, kSM_86, fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80", 32768, 128, 0, false},
    {DATA_TYPE_INT8, 192, 64, kSM_86, fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm80_interleaved_noloop", 28672, 128, 32, true},
    {DATA_TYPE_INT8, 192, 64, kSM_86, fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm80_noloop", 28672, 128, 32, false},
    {DATA_TYPE_INT8, 192, 64, kSM_86, fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm80_interleaved", 32768, 128, 0, true},
    {DATA_TYPE_INT8, 192, 64, kSM_86, fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm80", 32768, 128, 0, false},
    {DATA_TYPE_INT8, 256, 64, kSM_86, fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm80_interleaved_noloop", 36864, 128, 32, true},
    {DATA_TYPE_INT8, 256, 64, kSM_86, fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm80_noloop", 36864, 128, 32, false},
    {DATA_TYPE_INT8, 256, 64, kSM_86, fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm80_interleaved", 36864, 128, 0, true},
    {DATA_TYPE_INT8, 256, 64, kSM_86, fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm80", 36864, 128, 0, false},
    {DATA_TYPE_INT8, 384, 64, kSM_86, fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm80_interleaved_noloop", 53248, 128, 32, true},
    {DATA_TYPE_INT8, 384, 64, kSM_86, fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm80_noloop", 53248, 128, 32, false},
    {DATA_TYPE_INT8, 384, 64, kSM_86, fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm80_interleaved", 51200, 128, 0, true},
    {DATA_TYPE_INT8, 384, 64, kSM_86, fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm80", 53248, 128, 0, false},
#endif
};

class FusedMultiHeadAttentionXMMAKernelV2
    : public TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoV2,
          Fused_multihead_attention_params_v2>
{
public:
    FusedMultiHeadAttentionXMMAKernelV2(const FusedMultiHeadAttentionKernelMetaInfoV2* pMetaStart,
        unsigned int nMetaCount, Data_type type, unsigned int sm)
        : TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoV2,
              Fused_multihead_attention_params_v2>(pMetaStart, nMetaCount, type, sm)
    {
    }

    inline uint64_t hashID(unsigned int s, bool interleaved, bool unroll) const
    {
        return (uint64_t) s << 32 | (interleaved ? 2ull : 0ull) | (unroll ? 1ull : 0ull);
    }

    virtual uint64_t hashID(const KernelMeta& kernelMeta) const
    {
        assert(kernelMeta.mD == 64);
        return hashID(kernelMeta.mS, kernelMeta.mInterleaved, kernelMeta.mUnrollStep);
    }

    virtual void run(Fused_multihead_attention_params_v2& params, cudaStream_t ss) const
    {
        assert(params.d == 64);
        if (params.interleaved)
        {
            assert(mDataType == fastertransformer::DATA_TYPE_INT8);
        }

        bool forceUnroll = params.force_unroll;
        if (!forceUnroll && !params.ignore_b1opt && mSM >= kSM_75)
        {
            const struct
            {
                unsigned int mSM;
                Data_type mDataType;
                int mS;
                int mMaxBatch;
            } unrollList[]
                = { {kSM_75, fastertransformer::DATA_TYPE_FP16, 256, 1},
                      {kSM_75, fastertransformer::DATA_TYPE_FP16, 384, 1},
                      {kSM_75, fastertransformer::DATA_TYPE_INT8, 128, 1},
                      {kSM_75, fastertransformer::DATA_TYPE_INT8, 192, 2},
                      {kSM_75, fastertransformer::DATA_TYPE_INT8, 256, 1},
                      {kSM_75, fastertransformer::DATA_TYPE_INT8, 384, 1},
#if CUDA_VERSION >= 11000
                      {kSM_80, fastertransformer::DATA_TYPE_FP16, 128, 4},
                      {kSM_80, fastertransformer::DATA_TYPE_FP16, 256, 4},
                      {kSM_80, fastertransformer::DATA_TYPE_FP16, 384, 4},
                      {kSM_80, fastertransformer::DATA_TYPE_INT8, 128, 4},
                      {kSM_80, fastertransformer::DATA_TYPE_INT8, 192, 16},
                      {kSM_80, fastertransformer::DATA_TYPE_INT8, 256, 8},
                      {kSM_80, fastertransformer::DATA_TYPE_INT8, 384, 8},

                      {kSM_86, fastertransformer::DATA_TYPE_FP16, 128, 4},
                      {kSM_86, fastertransformer::DATA_TYPE_FP16, 256, 4},
                      {kSM_86, fastertransformer::DATA_TYPE_INT8, 128, 4},
                      {kSM_86, fastertransformer::DATA_TYPE_INT8, 192, 16},
                      {kSM_86, fastertransformer::DATA_TYPE_INT8, 256, 8},
                      {kSM_86, fastertransformer::DATA_TYPE_INT8, 384, 8},
#endif
                  };
            for (unsigned int i = 0u; i < sizeof(unrollList) / sizeof(unrollList[0]); ++i)
            {
                if (mSM == unrollList[i].mSM && mDataType == unrollList[i].mDataType && params.s == unrollList[i].mS
                    && params.b <= unrollList[i].mMaxBatch)
                {
                    forceUnroll = true;
                    break;
                }
            }
        }

        const auto findIter = mFunctions.find(hashID(params.s, params.interleaved, forceUnroll));
        // ASSERT(findIter != mFunctions.end());

        const auto& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;

        void* kernelParams[] = {&params, nullptr};
        if (!forceUnroll)
        {
            cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                           kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                mDriver);
        }
        else
        {
            int unroll = kernelMeta.mS / kernelMeta.mUnrollStep;
            assert(kernelMeta.mS == kernelMeta.mUnrollStep * unroll);
            cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, unroll, kernelMeta.mThreadsPerCTA, 1, 1,
                           kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                mDriver);
        }
    }
};

using FusedMHAKernelFactoryV2 = TFusedMHAKernelFactory<FusedMultiHeadAttentionXMMAKernelV2>;

inline const FusedMultiHeadAttentionXMMAKernelV2* getXMMAKernelsV2(Data_type type, unsigned int sm)
{
    return FusedMHAKernelFactoryV2::Get().getXMMAKernels(
        sMhaKernelMetaInfosV2, sizeof(sMhaKernelMetaInfosV2) / sizeof(sMhaKernelMetaInfosV2[0]), type, sm);
}

} // namespace fastertransformer
