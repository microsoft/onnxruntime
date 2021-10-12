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

#include "qkvToContext.h"
//#include "common.cuh"

#include <cassert>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>


namespace fastertransformer {

#if (defined __clang__)
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Werror=strict-aliasing"
#endif

static inline void set_alpha(uint32_t&, float, Data_type)
{
    assert(true);
    // if (dtype == DATA_TYPE_FP16)
    // {
    //     half2 h2 = __float2half2_rn(norm);
    //     alpha = reinterpret_cast<const uint32_t&>(h2);
    // }
    // else if (dtype == DATA_TYPE_FP32)
    // {
    //     alpha = reinterpret_cast<const uint32_t&>(norm);
    // }
    // else if (dtype == DATA_TYPE_INT32)
    // {
    //     int32_t inorm = static_cast<int32_t>(norm);
    //     alpha = reinterpret_cast<const uint32_t&>(inorm);
    // }
    // else
    // {
    //     assert(false);
    // }
}

#if (defined __clang__)
#   pragma clang diagnostic pop
#endif

class FusedMHARunnerFP16v2::mhaImpl
{
public:
    mhaImpl(FusedMHARunnerFP16v2* interface)
        : interface(interface)
        , sm(interface->mSm)
        , xmmaKernel(getXMMAKernelsV2(DATA_TYPE_FP16, sm))
    {
        assert((sm == kSM_72 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86) && "Unsupported architecture");
        params.clear();
    }

    ~mhaImpl() {}

    size_t getPackedMaskSizeInBytes() const
    {
        // check that we initialized
        assert(xmmas_m > 0);
        assert(threads_per_cta > 0);
        assert(interface->mB > 0);
        return interface->mB * xmmas_m * threads_per_cta * sizeof(uint32_t);
    }

    void setup(const int S, const int B)
    {
        // TODO these implementation details might be better centralized into the XMMA code, since they are needed in
        // several places (also outside of this plugin)
        size_t warps_m = 2, warps_n = 2, warps_k = 1;
        if (sm == 70)
        {
            if (S == 64 || S == 96)
            {
                warps_m = 2;
                warps_n = 2;
            }
            else if (S == 128)
            {
                warps_m = 1;
                warps_n = 4;
            }
            else if (S == 256 || S == 384)
            {
                warps_m = 1;
                warps_n = 8;
            }
            else
            {
                assert(false && "Unsupporte seqlen");
            }
        }
        else
        {
            if (S == 64 || S == 96 || S == 128)
            {
                warps_m = 2;
                warps_n = 2;
            }
            else if (S == 256)
            {
                warps_m = 1;
                warps_n = 4;
            }
            else if (S == 384)
            {
                warps_m = 1;
                warps_n = 8;
            }
            else
            {
                assert(false && "Unsupporte seqlen");
            }
        }
        // The number of threads per CTA.
        threads_per_cta = warps_m * warps_n * warps_k * 32;
        // The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension.
        xmmas_m = (S + 16 * warps_m - 1) / (16 * warps_m);
        // The number of xmmas in the N dimension.
        xmmas_n = (S + 16 * warps_n - 1) / (16 * warps_n);

        const float scale_bmm1 = interface->mRsqrtHeadSize;
        const float scale_softmax = 1.f; // Seems to be only required for int8
        const float scale_bmm2 = 1.f;

        Data_type scale_type = DATA_TYPE_FP16;
        set_alpha(params.scale_bmm1, scale_bmm1, scale_type);
        set_alpha(params.scale_softmax, scale_softmax, scale_type);
        set_alpha(params.scale_bmm2, scale_bmm2, scale_type);

        params.b = B;
        params.h = interface->mNumHeads;
        params.s = S;
        params.d = interface->mHeadSize;

        // mLdQKV = 3 * B * mNumHeads * mHeadSize;
        // mLdOut = B * mNumHeads * mHeadSize;

        params.qkv_stride_in_bytes = 3 * interface->mNumHeads * interface->mHeadSize * sizeof(half);
        params.packed_mask_stride_in_bytes = xmmas_m * threads_per_cta * sizeof(uint32_t);
        params.o_stride_in_bytes = interface->mNumHeads * interface->mHeadSize * sizeof(half);
    }

    void run(const void* qkvPtr, const void* maskPtr, const void* cuSeqlenPtr, void* output, void* workspace, cudaStream_t stream)
    {
        params.qkv_ptr = const_cast<void*>(qkvPtr);

        params.packed_mask_ptr = const_cast<void*>(maskPtr);

        params.o_ptr = output;

        params.cu_seqlens = static_cast<int*>(const_cast<void*>(cuSeqlenPtr));
        xmmaKernel->run(params, stream);
        //check_cuda_error(cudaPeekAtLastError());
    }

    bool isValid(int s) const
    {
        return xmmaKernel->isValid(s);
    }

    int getSFromMaxSeqLen(const int max_seq_len)
    {
        int S = 1024;
        if (max_seq_len <= 64)
        {
          S = 64;
        }
        else if (max_seq_len <= 96)
        {
          S = 96;
        }
        else if (max_seq_len <= 128)
        {
          S = 128;
        }
        else if (max_seq_len <= 256)
        {
          S = 256;
        }
        else if (max_seq_len <= 384)
        {
          S = 384;
        }
        else if (max_seq_len <= 512)
        {
          S = 512;
        }
        return S;
    }

private:
    FusedMHARunnerFP16v2* interface;
    Fused_multihead_attention_params_v2 params;
    int sm;
    const FusedMultiHeadAttentionXMMAKernelV2* xmmaKernel;
    size_t xmmas_m;
    size_t xmmas_n;
    size_t threads_per_cta;
};

FusedMHARunnerFP16v2::FusedMHARunnerFP16v2(const int numHeads, const int headSize, const int sm, const float q_scaling)
    : MHARunner(numHeads, headSize, 2, q_scaling)
    , mSm(sm)
    , pimpl(new mhaImpl(this))
{
}

void FusedMHARunnerFP16v2::setup(const int S, const int B)
{
    MHARunner::setup(S, B);
    pimpl->setup(S, B);
}

size_t FusedMHARunnerFP16v2::getWorkspaceSize() const
{
    return 0;
}

void FusedMHARunnerFP16v2::run(const void* input, const void* mask, void* workspace, void* output, cudaStream_t stream)
{
    assert(false && "not implemented");
}

void FusedMHARunnerFP16v2::run(const void* input, const void* mask, const void* seqlen, void* workspace, void* output, cudaStream_t stream)
{
    pimpl->run(input, mask, seqlen, output, workspace, stream);
}

bool FusedMHARunnerFP16v2::isValid(int s) const
{
    return pimpl->isValid(s);
}

void FusedMHARunnerFP16v2::setScaleList(const float scaleQkv, const float dqProbs, const float scaleCtx)
{
}

int FusedMHARunnerFP16v2::getSFromMaxSeqLen(const int max_seq_len)
{
    return pimpl->getSFromMaxSeqLen(max_seq_len);
}


// Int8 starts here: TODO refactor the duplicate stuff

class FusedMHARunnerInt8v2::mhaImpl
{

public:
    mhaImpl(FusedMHARunnerInt8v2* interface)
        : interface(interface)
        , sm(interface->mSm)
        , xmmaKernel(getXMMAKernelsV2(DATA_TYPE_INT8, sm))
    {
        assert((sm == kSM_72 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86) && "Unsupported architecture");
        params.clear();
    }

    ~mhaImpl() {}

    size_t getPackedMaskSizeInBytes() const
    {
        assert(xmmas_m > 0);
        assert(threads_per_cta > 0);
        assert(interface->mB > 0);
        return interface->mB * xmmas_m * threads_per_cta * sizeof(uint32_t);
    }

    void setup(const int S, const int B)
    {
        size_t warps_m = 1;
        size_t warps_n = 1; 
        size_t warps_k = 1;
        if ((sm == 75 || sm == 80) && S == 64)
        {
            warps_m = 2;
            warps_n = 2;
        }
	else if (S == 128)
        {
            warps_m = 2;
            warps_n = 2;
        }
        else if (S == 192 || S == 256)
        {
            warps_m = 1;
            warps_n = 4;
        }
        else if (S == 384)
        {
            warps_m = 1;
            warps_n = 8;
        }

        else
        {
            assert(false && "Unsupported seqlen.");
        }
        // The number of threads per CTA.
        threads_per_cta = warps_m * warps_n * warps_k * 32;
        // The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension.
        xmmas_m = (S + 16 * warps_m - 1) / (16 * warps_m);
        // The number of xmmas in the N dimension.
        xmmas_n = (S + 16 * warps_n - 1) / (16 * warps_n);

        params.b = B;
        params.h = interface->mNumHeads;
        params.s = S;
        params.d = interface->mHeadSize;
        params.use_int8_scale_max = true;
        params.packed_mask_stride_in_bytes = xmmas_m * threads_per_cta * sizeof(uint32_t);
        params.qkv_stride_in_bytes = 3 * interface->mNumHeads * interface->mHeadSize * sizeof(int8_t);
        params.o_stride_in_bytes = interface->mNumHeads * interface->mHeadSize * sizeof(int8_t);

        float scaleQkv = interface->mScaleQkv;
        float scaleCtx = interface->mScaleCtx;

        float scaleBmm1 = scaleQkv * scaleQkv * (1.f / sqrtf(interface->mHeadSize));
        float scaleBmm2 = interface->mDqProbs * scaleQkv / scaleCtx;
        float scaleSoftmax = 1.f / interface->mDqProbs;

        // params.scale_bmm1 = reinterpret_cast<const uint32_t&>(scaleBmm1);
        // params.scale_bmm2 = reinterpret_cast<const uint32_t&>(scaleBmm2);
        // params.scale_softmax = reinterpret_cast<const uint32_t&>(scaleSoftmax);

        params.enable_i2f_trick
            = -double(1 << 22) * double(scaleBmm2) <= -128.f && double(1 << 22) * double(scaleBmm2) >= 127.f;
        
    }

    void run(const void* qkvPtr, const void* maskPtr, const void* cuSeqlenPtr, void* output, void* workspace, cudaStream_t stream)
    {
        params.qkv_ptr = const_cast<void*>(qkvPtr);

        params.o_ptr = output;

        params.cu_seqlens = static_cast<int*>(const_cast<void*>(cuSeqlenPtr));

        xmmaKernel->run(params, stream);
    }

    bool isValid(int s) const
    {
        return xmmaKernel->isValid(s);
    }

    int getSFromMaxSeqLen(const int max_seq_len)
    {
        int S = 1024;
	if (sm == 75 || sm == 80)
        {
          if (max_seq_len <= 64)
          {
            S = 64;
          }
	  else if (max_seq_len <= 128)
          {
            S = 128;
          }
          else if (max_seq_len <= 192)
          {
            S = 192;
          }
          else if (max_seq_len <= 256)
          {
            S = 256;
          }
          else if (max_seq_len <= 384)
          {
            S = 384;
          }
          else if (max_seq_len <= 512)
          {
            S = 512;
          }
	}
	else
        {
	  if (max_seq_len <= 128)
          {
            S = 128;
          }
          else if (max_seq_len <= 192)
          {
            S = 192;
          }
          else if (max_seq_len <= 256)
          {
            S = 256;
          }
          else if (max_seq_len <= 384)
          {
            S = 384;
          }
          else if (max_seq_len <= 512)
          {
            S = 512;
          }
        }
        return S;
    }


private:
    FusedMHARunnerInt8v2* interface;
    Fused_multihead_attention_params_v2 params;
    int sm;
    const FusedMultiHeadAttentionXMMAKernelV2* xmmaKernel;
    size_t xmmas_m;
    size_t xmmas_n;
    size_t threads_per_cta;
};

FusedMHARunnerInt8v2::FusedMHARunnerInt8v2(const int numHeads, const int headSize, const int sm)
    : MHARunner(numHeads, headSize, 1)
    , mSm(sm)
    , pimpl(new mhaImpl(this))
{
}

void FusedMHARunnerInt8v2::setScaleList(const float scaleQkv, const float dqProbs, const float scaleCtx)
{
    mDqProbs = dqProbs;
    mScaleQkv = scaleQkv;
    mScaleCtx = scaleCtx;
}

void FusedMHARunnerInt8v2::setup(const int S, const int B)
{
    pimpl->setup(S, B);
}

size_t FusedMHARunnerInt8v2::getWorkspaceSize() const
{
    return 0;
}

void FusedMHARunnerInt8v2::run(const void* input, const void* mask, void* workspace, void* output, cudaStream_t stream)
{
    assert(false && "Not implemented");
}

void FusedMHARunnerInt8v2::run(const void* input, const void* mask, const void* seqlen, void* workspace, void* output, cudaStream_t stream)
{
    pimpl->run(input, mask, seqlen, output, workspace, stream);
}

bool FusedMHARunnerInt8v2::isValid(int s) const
{
    return pimpl->isValid(s);
}

int FusedMHARunnerInt8v2::getSFromMaxSeqLen(const int max_seq_len)
{
    return pimpl->getSFromMaxSeqLen(max_seq_len);
}

} // namespace fastertransformer
