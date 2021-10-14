/*
* Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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
#include <assert.h>
#include <array>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include "contrib_ops/cuda/fastertransformer/utils/arguments.h"
#include "contrib_ops/cuda/fastertransformer/cuda/cuda_kernels.h"
#include <float.h>
#include <type_traits>

namespace fastertransformer{

#define DO_SPLIT_SMALL_TOP_K_SOFTMAX
static const int SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE = 256;
static const int SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS = 128;
static const int MAX_K = 4;

constexpr float HALF_FLT_MAX = 65504.F;

template<typename T, int MAX_K>
struct TopK
{
    int p[MAX_K];
    T u[MAX_K];

    __device__ __forceinline__ void insert(T elem, int elem_id)
    {
        if (elem > u[MAX_K-1] || (p[MAX_K-1] == -1) || ((elem == u[MAX_K-1]) && (elem_id < p[MAX_K-1])))
        //if (elem > u[MAX_K-1] || ((elem == u[MAX_K-1]) && (elem_id < p[MAX_K-1])))
        {
            u[MAX_K-1] = elem;
            p[MAX_K-1] = elem_id;
        }

        for(int k = MAX_K - 2; k >= 0; --k)
        {
            if ((u[k+1] > u[k]) || (p[k] == -1) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
            //if ((u[k+1] > u[k]) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
            {
                T u2 = u[k];
                int p2 = p[k];
                u[k] = u[k+1];
                p[k] = p[k+1];
                u[k+1] = u2;
                p[k+1] = p2;
            }
        }
    }

    __device__ __forceinline__ void init()
    {
        const bool IS_FP16 = std::is_same<T, half>::value;
        const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;

        for(int i = 0; i < MAX_K; i++)
        {
            p[i] = -1;
            u[i] = -MAX_T_VAL;
        }
    }
};

template<typename T, int MAX_K>
__device__ __forceinline__ TopK<T, MAX_K> reduce_topk_op(const TopK<T, MAX_K>& a, const TopK<T, MAX_K>& b)
{
    TopK<T, MAX_K> res = a;
    for(int i = 0; i < MAX_K; ++i)
        res.insert(b.u[i], b.p[i]);
    return res;
}

template<typename T>
struct TopK_2
{
    int p = -1;
    T u = -((std::is_same<T, half>::value)? HALF_FLT_MAX : FLT_MAX);

    __device__ __forceinline__ void insert(T elem, int elem_id)
    {
        if(elem > u)
        {
            u = elem;
            p = elem_id;
        }
    }

    __device__ __forceinline__ void init()
    {
        u = -((std::is_same<T, half>::value)? HALF_FLT_MAX : FLT_MAX);
        p = -1;
    }
};

template<typename T>
__device__ __forceinline__ TopK_2<T> reduce_topk_op_2(const TopK_2<T>& a, const TopK_2<T>& b)
{
    return a.u > b.u ? a : b;
}

template <typename T>
void topK_kernelLauncher(T* log_probs,
                        int* topk_tmp_id_buf,
                        T* topk_tmp_val_buf,
                        int* ids,
                        DecodingBeamsearchArguments args,
                        cudaStream_t stream);

template <typename T>
void topK_kernelLauncher(void* workspace,
                         size_t& workspace_size,
                         T* log_probs,
                         int* ids,
                         const bool* finished,
                         DecodingBeamsearchArguments args,
                         cudaStream_t stream);

template <typename T>
void topK_softMax(const T* log_probs, 
                  const T* bias, 
                  const bool* finished, 
                  float* cum_log_probs, 
                  int* ids, 
                  void * tmp_storage,
                  DecodingBeamsearchArguments args,
                  cudaStream_t stream);

/* *************************** end of BeamSearch kernel *********************************** */

/* ********************************** Sampling kernel *********************************** */
void ker_curand_setupLauncher(curandState_t* state,
    DecodingSamplingArguments args,
    cudaStream_t stream);


template <typename T>
void topK_sampling_kernel_kernelLauncher_v2(void* workspace,
                                            size_t& workspace_size,
                                            T* log_probs,
                                            int* ids,
                                            int* sequence_length,
                                            bool* finished_buf,
                                            curandState_t* curandstate,
                                            DecodingSamplingArguments args,
                                            cudaStream_t stream,
                                            const int batch_size);

template <typename T>
void topK_sampling_kernel_kernelLauncher(void* workspace,
                                        size_t& workspace_size,
                                        T* log_probs,
                                        int* ids,
                                        int* sequence_length,
                                        bool* finished_buf,
                                        int random_num,
                                        DecodingSamplingArguments args,
                                        cudaStream_t stream,
                                        const int batch_size);

template<typename T>
void topP_sampling_kernel_kernelLauncher(void* workspace,
                                         size_t& workspace_size,
                                         const T* log_probs,
                                         const int* id_vals,
                                         const int* offset_buf,
                                         bool* finished_buf,
                                         int step,
                                         DecodingSamplingArguments& args,
                                         int* output_ids, 
                                         int* sequence_length,
                                         const int n,
                                         cudaStream_t stream,
                                         const int batch_size);

template<typename T>
void topP_sampling_kernel_kernelLauncher_v2(void* workspace,
                                         size_t& workspace_size,
                                         const T* log_probs,
                                         const int* id_vals,
                                         int* offset_buf,
                                         int* begin_offset_buf,
                                         bool* finished_buf,
                                         curandState_t* curandstate,
                                         DecodingSamplingArguments& args,
                                         int* output_ids, 
                                         int* sequence_length,
                                         const int n,
                                         cudaStream_t stream,
                                         const int batch_size);

template<typename T>
void beam_topK_kernelLauncher(const T* log_probs, 
                              int* topk_tmp_id_buf,
                              T* topk_tmp_val_buf,
                              DecodingSamplingArguments args,
                              cudaStream_t stream);

template<typename T>
void topK_topP_sampling_kernel_kernelLauncher(void* workspace,
                                              size_t& workspace_size,
                                              int* output_ids,
                                              const T* logits,
                                              const int random_num,
                                              DecodingSamplingArguments& args,
                                              cudaStream_t stream,
                                              const int batch_size);

template<typename T>
void topK_topP_sampling_kernel_kernelLauncher_v2(void* workspace,
                                                 size_t& workspace_size,
                                                 int* output_ids,
                                                 const T* logits,
                                                 bool* finished_buf,
                                                 curandState_t* curandstate,
                                                 DecodingSamplingArguments& args,
                                                 cudaStream_t stream,
                                                 const int batch_size);

/* *************************** end of Sampling kernel *********************************** */

}//namespace fastertransformer
