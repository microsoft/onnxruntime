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
#include "cudaDriverWrapper.h"
#include "cuda_runtime_api.h"
#include "fused_multihead_attention_common.h"
#include <assert.h>
#include <memory>
#include <mutex>
#include <set>
#include <stdint.h>
#include <unordered_map>
#include <vector>
namespace fastertransformer
{
static inline size_t get_size_in_bytes(size_t n, Data_type dtype)
{
    switch (dtype)
    {
    case DATA_TYPE_E8M10: return n * 4;
    case DATA_TYPE_FP32: return n * 4;
    case DATA_TYPE_FP16: return n * 2;
    case DATA_TYPE_INT32: return n * 4;
    case DATA_TYPE_INT8: return n;
    case DATA_TYPE_INT4: return n / 2;
    case DATA_TYPE_BOOL: return n / 8;
    case DATA_TYPE_E8M7: return n * 2;
    default: assert(false); return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params
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
};

////////////////////////////////////////////////////////////////////////////////////////////////////
extern unsigned char fused_multihead_attention_fp16_64_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_fp16_96_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_fp16_64_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_fp16_96_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_int8_128_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_int8_384_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_int8_384_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_int8_128_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o[];

extern unsigned int fused_multihead_attention_fp16_64_64_kernel_sm75_cu_o_len;
extern unsigned int fused_multihead_attention_fp16_96_64_kernel_sm75_cu_o_len;
extern unsigned int fused_multihead_attention_fp16_64_64_kernel_sm80_cu_o_len;
extern unsigned int fused_multihead_attention_fp16_96_64_kernel_sm80_cu_o_len;
extern unsigned int fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o_len;
extern unsigned int fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o_len;
extern unsigned int fused_multihead_attention_int8_128_64_kernel_sm75_cu_o_len;
extern unsigned int fused_multihead_attention_int8_384_64_kernel_sm75_cu_o_len;
extern unsigned int fused_multihead_attention_int8_384_64_kernel_sm80_cu_o_len;
extern unsigned int fused_multihead_attention_int8_128_64_kernel_sm80_cu_o_len;
extern unsigned int fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o_len;
extern unsigned int fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o_len;

static const struct FusedMultiHeadAttentionKernelMetaInfoV1
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
} sMhaKernelMetaInfos[] = {
    // Turing
    {DATA_TYPE_FP16, 64, 64, kSM_75, fused_multihead_attention_fp16_64_64_kernel_sm75_cu_o,
        fused_multihead_attention_fp16_64_64_kernel_sm75_cu_o_len,
        "fused_multihead_attention_v2_fp16_64_64_kernel_sm75", 24576, 128},
    {DATA_TYPE_FP16, 96, 64, kSM_75, fused_multihead_attention_fp16_96_64_kernel_sm75_cu_o,
        fused_multihead_attention_fp16_96_64_kernel_sm75_cu_o_len,
        "fused_multihead_attention_v2_fp16_96_64_kernel_sm75", 24576, 128},
    {DATA_TYPE_FP16, 128, 64, kSM_75, fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o,
        fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o_len, "fused_multihead_attention_fp16_128_64_kernel_sm75",
        32768, 128},
    {DATA_TYPE_FP16, 384, 64, kSM_75, fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o,
        fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o_len, "fused_multihead_attention_fp16_384_64_kernel_sm75",
        57344, 256},
    {DATA_TYPE_INT8, 128, 64, kSM_75, fused_multihead_attention_int8_128_64_kernel_sm75_cu_o,
        fused_multihead_attention_int8_128_64_kernel_sm75_cu_o_len, "fused_multihead_attention_int8_128_64_kernel_sm75",
        16384, 128},
    {DATA_TYPE_INT8, 384, 64, kSM_75, fused_multihead_attention_int8_384_64_kernel_sm75_cu_o,
        fused_multihead_attention_int8_384_64_kernel_sm75_cu_o_len, "fused_multihead_attention_int8_384_64_kernel_sm75",
        53284, 256},
#if CUDA_VERSION >= 11000
    // Ampere
    {DATA_TYPE_FP16, 64, 64, kSM_80, fused_multihead_attention_fp16_64_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_64_64_kernel_sm80_cu_o_len,
        "fused_multihead_attention_v2_fp16_64_64_kernel_sm80", 32768, 128},
    {DATA_TYPE_FP16, 96, 64, kSM_80, fused_multihead_attention_fp16_96_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_96_64_kernel_sm80_cu_o_len,
        "fused_multihead_attention_v2_fp16_96_64_kernel_sm80", 49152, 128},
    {DATA_TYPE_FP16, 128, 64, kSM_80, fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o_len, "fused_multihead_attention_fp16_128_64_kernel_sm80",
        49152, 128},
    {DATA_TYPE_FP16, 384, 64, kSM_80, fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o_len, "fused_multihead_attention_fp16_384_64_kernel_sm80",
        114688, 256},
    {DATA_TYPE_INT8, 128, 64, kSM_80, fused_multihead_attention_int8_128_64_kernel_sm80_cu_o,
        fused_multihead_attention_int8_128_64_kernel_sm80_cu_o_len, "fused_multihead_attention_int8_128_64_kernel_sm80",
        24576, 128},
    {DATA_TYPE_INT8, 384, 64, kSM_80, fused_multihead_attention_int8_384_64_kernel_sm80_cu_o,
        fused_multihead_attention_int8_384_64_kernel_sm80_cu_o_len, "fused_multihead_attention_int8_384_64_kernel_sm80",
        57344, 256},

    // GA10x
    // Note: For GA10X keep only kernels whose sharedMemBytes < 100KiB
    {DATA_TYPE_FP16, 64, 64, kSM_86, fused_multihead_attention_fp16_64_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_64_64_kernel_sm80_cu_o_len,
        "fused_multihead_attention_v2_fp16_64_64_kernel_sm80", 32768, 128},
    {DATA_TYPE_FP16, 96, 64, kSM_86, fused_multihead_attention_fp16_96_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_96_64_kernel_sm80_cu_o_len,
        "fused_multihead_attention_v2_fp16_96_64_kernel_sm80", 49152, 128},
    {DATA_TYPE_FP16, 128, 64, kSM_86, fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o_len, "fused_multihead_attention_fp16_128_64_kernel_sm80",
        49152, 128},
    {DATA_TYPE_INT8, 128, 64, kSM_86, fused_multihead_attention_int8_128_64_kernel_sm80_cu_o,
        fused_multihead_attention_int8_128_64_kernel_sm80_cu_o_len, "fused_multihead_attention_int8_128_64_kernel_sm80",
        24576, 128},
    {DATA_TYPE_INT8, 384, 64, kSM_86, fused_multihead_attention_int8_384_64_kernel_sm80_cu_o,
        fused_multihead_attention_int8_384_64_kernel_sm80_cu_o_len, "fused_multihead_attention_int8_384_64_kernel_sm80",
        57344, 256},

#endif
};

template <typename TKernelMeta, typename TKernelParam>
class TFusedMultiHeadAttentionXMMAKernel
{
public:
    using KernelMeta = TKernelMeta;
    using KernelParam = TKernelParam;
    inline uint64_t hashID(unsigned int s, unsigned int d) const
    {
        return (uint64_t) s << 32 | d;
    }
    virtual uint64_t hashID(const KernelMeta& kernelMeta) const
    {
        return hashID(kernelMeta.mS, kernelMeta.mD);
    }

    TFusedMultiHeadAttentionXMMAKernel(
        const TKernelMeta* pMetaStart, unsigned int nMetaCount, Data_type type, unsigned int sm)
        : mDataType(type)
        , mKernelMeta(pMetaStart)
        , mKernelMetaCount(nMetaCount)
        , mSM(sm)
    {
    }

    void loadXMMAKernels()
    {
        if (!mFunctions.empty())
        {
            return;
        }

        for (unsigned int i = 0; i < mKernelMetaCount; ++i)
        {
            const auto& kernelMeta = mKernelMeta[i];
            if (kernelMeta.mSM == mSM && kernelMeta.mDataType == mDataType)
            {
                CUmodule hmod{0};
                auto findModuleIter = mModules.find(kernelMeta.mCubin);
                if (findModuleIter != mModules.end())
                {
                    hmod = findModuleIter->second;
                }
                else
                {
                    cuErrCheck(mDriver.cuModuleLoadData(&hmod, kernelMeta.mCubin), mDriver);
                    mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
                }

                FusedMultiHeadAttentionKernelInfo funcInfo;
                funcInfo.mMetaInfoIndex = i;
                cuErrCheck(mDriver.cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName), mDriver);
                if (kernelMeta.mSharedMemBytes >= 48 * 1024)
                {
                    cuErrCheck(mDriver.cuFuncSetAttribute(funcInfo.mDeviceFunction,
                                   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelMeta.mSharedMemBytes),
                        mDriver);
                }
                mFunctions.insert(std::make_pair(hashID(kernelMeta), funcInfo));
                int s = static_cast<int>(kernelMeta.mS);
                if (mValidSequences.find(s) == mValidSequences.end())
                    mValidSequences.insert(s);
            }
        }
    }

    bool isValid(int s) const
    {
        return (mValidSequences.find(s) != mValidSequences.end());
    }

    virtual void run(TKernelParam& params, cudaStream_t ss) const
    {
        const auto findIter = mFunctions.find(hashID(params.s, params.d));
        // ASSERT(findIter != mFunctions.end()); //TODO check the ASSERT

        const auto& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;

        void* kernelParams[] = {&params, nullptr};
        cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                       kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
            mDriver);
    }

    virtual ~TFusedMultiHeadAttentionXMMAKernel() = default;

protected:
    // nvinfer1::CUDADriverWrapper mDriver;
    CUDADriverWrapper mDriver;

    Data_type mDataType;
    const TKernelMeta* mKernelMeta;
    unsigned int mKernelMetaCount;
    unsigned int mSM;
    std::unordered_map<const unsigned char*, CUmodule> mModules;
    struct FusedMultiHeadAttentionKernelInfo
    {
        unsigned int mMetaInfoIndex;
        CUfunction mDeviceFunction;
    };
    std::unordered_map<uint64_t, FusedMultiHeadAttentionKernelInfo> mFunctions;
    std::set<int> mValidSequences;
};

template <typename TFusedMHAKernelList>
class TFusedMHAKernelFactory
{
public:
    const TFusedMHAKernelList* getXMMAKernels(const typename TFusedMHAKernelList::KernelMeta* pKernelList,
        unsigned int nbKernels, Data_type type, unsigned int sm)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        const auto id = hashID(type, sm);
        const auto findIter = mKernels.find(id);
        if (findIter == mKernels.end())
        {
            TFusedMHAKernelList* newKernel = new TFusedMHAKernelList{pKernelList, nbKernels, type, sm};
            newKernel->loadXMMAKernels();
            mKernels.insert(std::make_pair(id, std::unique_ptr<TFusedMHAKernelList>(newKernel)));
            return newKernel;
        }
        return findIter->second.get();
    }

    static TFusedMHAKernelFactory<TFusedMHAKernelList>& Get()
    {
        static TFusedMHAKernelFactory<TFusedMHAKernelList> s_factory;
        return s_factory;
    }

private:
    TFusedMHAKernelFactory() = default;

    inline uint64_t hashID(Data_type type, unsigned int sm) const
    {
        return (uint64_t) type << 32 | sm;
    }

    std::unordered_map<uint64_t, const std::unique_ptr<TFusedMHAKernelList>> mKernels;
};

using FusedMultiHeadAttentionXMMAKernel
    = TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoV1, Fused_multihead_attention_params>;
using FusedMHAKernelFactory = TFusedMHAKernelFactory<FusedMultiHeadAttentionXMMAKernel>;

inline const FusedMultiHeadAttentionXMMAKernel* getXMMAKernels(Data_type type, unsigned int sm)
{
    return FusedMHAKernelFactory::Get().getXMMAKernels(
        sMhaKernelMetaInfos, sizeof(sMhaKernelMetaInfos) / sizeof(sMhaKernelMetaInfos[0]), type, sm);
}

} // namespace fastertransformer
