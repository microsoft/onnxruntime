/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#if USE_TENSORRT_LLM_FMHA

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "contrib_ops/cuda/bert/tensorrt_llm_fmha/fused_multihead_attention_common.h"
#include "contrib_ops/cuda/bert/tensorrt_llm_fmha/common/cudaUtils.h"
#include "contrib_ops/cuda/bert/tensorrt_llm_fmha/tmaDescriptor.h"

//namespace onnxruntime{
namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

class MHARunner
{
public:
    MHARunner(const Data_type dataType, const int numHeads, const int headSize, const float qScaling);

    MHARunner() = default;

    virtual ~MHARunner() = default;

    virtual void setup(const int b, const int s, const int sliding_window_size, const int total_seqlen,
        const bool has_alibi = false, const bool scale_alibi = false, const int tp_size = 1, const int tp_rank = 0)
        = 0;

    virtual void setup_paged_kv(const int b, const int s_q, const int s_kv, const int blocks_per_context_sequence,
        const int tokens_per_kv_block, const int sliding_window_size, const int total_seqlen,
        const bool has_alibi = false, const bool scale_alibi = false, const int tp_size = 1, const int tp_rank = 0)
        = 0;

    static bool fmha_supported(const int headSize, const int sm);

    virtual bool fmha_supported() = 0;

    virtual void setup_flags(const bool force_fp32_acc, const bool is_s_padded, const bool causal_mask,
        const int num_kv_heads /* MQA or GQA */)
        = 0;

    virtual void run(const void* input, const void* cu_seqlens, void* output, cudaStream_t stream) = 0;

    virtual void run_paged_kv(const void* q_input, void* paged_kv_tma_desc, const void* paged_kv_block_ptrs_on_host,
        const KVBlockArray paged_kv_cache, const void* cu_q_seqlens, const void* cu_kv_seqlens, void* output,
        cudaStream_t stream)
        = 0;

    virtual bool isValid(int s) const = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Workflow of fmha runner:
// 1. check if FMHA kernels are supported statically.
// 2. construct FMHA runner object.
// 3. setup_flags (used by all kernels).
// 4. setup runtime parameters (used by this specific case).
// 5. run the kernel (with all needed device pointers).

class FusedMHARunnerV2 : public MHARunner
{
public:
    FusedMHARunnerV2(const Data_type dataType, const int numHeads, const int headSize, const float qScaling);

    ~FusedMHARunnerV2(); // for pimpl

    void setup(const int b, const int s, const int sliding_window_size, const int total_seqlen,
        const bool has_alibi = false, const bool scale_alibi = false, const int tp_size = 1,
        const int tp_rank = 0) override;

    void setup_paged_kv(const int b, const int s_q, const int s_kv, const int blocks_per_context_sequence,
        const int tokens_per_kv_block, const int sliding_window_size, const int total_seqlen,
        const bool has_alibi = false, const bool scale_alibi = false, const int tp_size = 1,
        const int tp_rank = 0) override;

    bool fmha_supported() override;

    void run(const void* input, const void* cu_seqlens, void* output, cudaStream_t stream) override;
    void run_paged_kv(const void* q_input, void* paged_kv_tma_desc, const void* paged_kv_block_ptrs_on_host,
        const KVBlockArray paged_kv_cache, const void* cu_q_seqlens, const void* cu_kv_seqlens, void* output,
        cudaStream_t stream) override;

    void setup_flags(const bool force_fp32_acc, const bool is_s_padded, const bool causal_mask,
        const int num_kv_heads /* MQA or GQA */) override;

    bool isValid(int s) const override;

    static std::unique_ptr<MHARunner> Create(const Data_type dataType, const int numHeads, const int headSize, const float qScaling);

private:
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
};

} // namespace kernels
} // namespace tensorrt_llm
//}
#endif
