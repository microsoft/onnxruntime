/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "contrib_ops/cuda/bert/tensorrt_llm_fmha/common/assert.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>

namespace tensorrt_llm
{
namespace kernels
{

// Internal for K and V cache indexing
enum class KVIdxType : int32_t
{
    K_IDX = 0,
    V_IDX = 1
};

struct KVBlockArray
{
    // Struct operates on paged kv cache providing
    // functions for accessing blocks of in K and V caches
    // and elements inside these blocks

    // Max number of blocks per sequence
    int32_t mMaxBlocksPerSeq;
    // Current number of sequences
    int32_t mMaxSeqs;
    // Number of tokens. It must be power of 2.
    int32_t mTokensPerBlock;
    // Exponent of number of tokens with base 2.
    // E.g. for mTokensPerBlock 64, mTokensPerBlockLog2 equals to 6
    int32_t mTokensPerBlockLog2;
    // Table maps logical block idx to the data pointer of k/v cache block pool
    // Shape [B, W, 2, M], where 2 is table for K and V,
    // B is current number of sequences
    // W is beam width
    // M is Max number of blocks per sequence
    // int64_t reinterpred to void* pointing to the KV cache data
    int64_t* data;

    KVBlockArray() {}

    KVBlockArray(int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock, int32_t sizePerToken)
        : mMaxSeqs(batchSize)
        , mMaxBlocksPerSeq(maxBlocksPerSeq)
        , mTokensPerBlock(tokensPerBlock)
        , data(nullptr)
    {
        const float tokensPerBlockSeqLog2 = static_cast<float>(log2(mTokensPerBlock));
        TLLM_CHECK_WITH_INFO(
            ceil(tokensPerBlockSeqLog2) == floor(tokensPerBlockSeqLog2), "tokensPerBlock must be power of 2");
        // NOTE: pointer offset arithmetic offset is performed on int32_t (see this.getRowPtr).
        // If needed, we could do it on uint32_t or even uint64_t, but that might have performance implications
        TLLM_CHECK_WITH_INFO(static_cast<int64_t>(mMaxSeqs - 1) * mMaxBlocksPerSeq * 2 + maxBlocksPerSeq
                <= std::numeric_limits<int32_t>::max(),
            "kv cache is too large for gpt_attention_plugin");
        mTokensPerBlockLog2 = static_cast<int>(tokensPerBlockSeqLog2);
    }

    __host__ __device__ inline void** getRowPtr(KVIdxType kvIdx, int32_t seqIdx)
    {
        // Returns pointer to array of pointers to K or V cache for one specific sequence seqIdx.
        // seqIdx is in range [0; B]
        return reinterpret_cast<void**>(
            data + seqIdx * mMaxBlocksPerSeq * 2 + static_cast<int32_t>(kvIdx) * mMaxBlocksPerSeq);
    }

    __host__ __device__ inline void* getBlockPtr(void** pointer, int32_t tokenIdx)
    {
        return pointer[tokenIdx >> mTokensPerBlockLog2];
    }

    __host__ __device__ inline void* getBlockPtr(int32_t seqIdx, int32_t tokenIdx, KVIdxType kvIdx)
    {
        return getBlockPtr(getRowPtr(kvIdx, seqIdx), tokenIdx);
    }

    __host__ __device__ inline void* getKBlockPtr(int32_t seqIdx, int32_t tokenIdx)
    {
        return getBlockPtr(seqIdx, tokenIdx, KVIdxType::K_IDX);
    }

    __host__ __device__ inline void* getVBlockPtr(int32_t seqIdx, int32_t tokenIdx)
    {
        return getBlockPtr(seqIdx, tokenIdx, KVIdxType::V_IDX);
    }

    __host__ __device__ inline int32_t getLocalIdx(int32_t globalIdx)
    {
        return globalIdx & ((1 << mTokensPerBlockLog2) - 1);
    }

    __host__ __device__ inline int32_t getKVLocalIdx(
        int32_t globalTokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx)
    {
        // For K or V, the hidden dimension per head is *not* decomposed. The layout of each block of K or V is:
        // [numHeads, tokensPerBlock, hiddenSizePerHead].
        // This member function computes the corresponding linear index.
        // NOTE: we have remapped K layout as the same of V.
        return headIdx * mTokensPerBlock * dimsPerHead + getLocalIdx(globalTokenIdx) * dimsPerHead + channelIdx;
    }
};

struct KVLinearBuffer
{
    // Struct operates on contiguous kv cache providing
    // functions for accessing specific elements in K and V caches

    // Current number of sequences
    int32_t mMaxSeqs;
    // Max sequence length
    int32_t mMaxSeqLen;
    // Bytes per sequence (H*D*M_S*sizeof(DataType))
    int32_t mBytesPerSeq;
    // Pointer to the of K/V cache data
    // Shape [B, 2, S*H*D], where 2 is for K and V,
    // B is current number of sequences and
    // H is number of heads
    // S is maximum sequence length
    // D is dimension per head
    // K shape is [B, 1, H, S, D]
    // V shape is [B, 1, H, S, D]
    // NOTE: we have remapped K layout as the same of V.
    int8_t* data;

    KVLinearBuffer() {}

    KVLinearBuffer(int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock, int32_t sizePerToken)
        : mMaxSeqs(batchSize)
        , mMaxSeqLen(tokensPerBlock)
        , mBytesPerSeq(tokensPerBlock * sizePerToken)
        , data(nullptr)
    {
        // NOTE: pointer offset arithmetic offset is performed on int32_t (see this.getRowPtr).
        // If needed, we could do it on uint32_t or even uint64_t, but that might have performance implications
        TLLM_CHECK_WITH_INFO(
            static_cast<int64_t>(mMaxSeqs - 1) * mBytesPerSeq * 2 + mBytesPerSeq <= std::numeric_limits<int32_t>::max(),
            "kv cache is too large for gpt_attention_plugin");
    }

    __host__ __device__ inline void** getRowPtr(KVIdxType kvIdx, int32_t seqIdx)
    {
        return reinterpret_cast<void**>(data + seqIdx * mBytesPerSeq * 2 + static_cast<int32_t>(kvIdx) * mBytesPerSeq);
    }

    __host__ __device__ inline void* getBlockPtr(void** pointer, int32_t tokenIdx)
    {
        return reinterpret_cast<void*>(pointer);
    }

    __host__ __device__ inline void* getKBlockPtr(int32_t seqIdx, int32_t /*tokenIdx*/)
    {
        return reinterpret_cast<void*>(getRowPtr(KVIdxType::K_IDX, seqIdx));
    }

    __host__ __device__ inline void* getVBlockPtr(int32_t seqIdx, int32_t /*tokenIdx*/)
    {
        return reinterpret_cast<void*>(getRowPtr(KVIdxType::V_IDX, seqIdx));
    }

    __host__ __device__ inline int32_t getKVLocalIdx(
        int32_t tokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx)
    {
        return headIdx * mMaxSeqLen * dimsPerHead + tokenIdx * dimsPerHead + channelIdx;
    }
};

} // namespace kernels
} // namespace tensorrt_llm
