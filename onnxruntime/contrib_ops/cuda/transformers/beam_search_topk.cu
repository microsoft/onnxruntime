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

#include "beam_search_topk.h"

#include <cub/cub.cuh>

#include "reduce_kernel_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__global__ void BeamSearchOnlineTopKStage1Kernel(
  const T* input,
  int K,
  int vocab_size,
  int vocab_part_size,
  T* output_values,
  int32_t* output_token) {
    TopK<T, MAX_K> top_k;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    for (int i = 0; i < MAX_K; ++i) {
        top_k.p[i] = -1;
        top_k.u[i] = -MAX_T_VAL;
    }

    int batch_beam = blockIdx.x;
    int voc_part_id = blockIdx.y;

    int token_id_base = voc_part_id * vocab_part_size;
    const T* input_block = input + batch_beam * vocab_size;
    // voc_part_size
    for(int i = threadIdx.x + token_id_base; i < vocab_part_size + token_id_base; i+= blockDim.x) {
        if( i < vocab_size) {
            top_k.insert(input_block[i], i);
            if(batch_beam == 1)
                printf("(%d, %d, %f)\n", voc_part_id, i, (float)(input_block[i]));
        }
    }

    /*
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        for(int i = 0; i < MAX_K; i++)
            printf("(%d, %f, %d),", i, (float)(top_k.u[i]), top_k.p[i]);
        printf("\n\n");
    }*/

    // reduce in thread block
    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopK<T, MAX_K> top_k_block = BlockReduce(temp_storage).Reduce(top_k, reduce_topk_op<T, MAX_K>);
    __syncthreads();

    output_values += batch_beam * K;
    output_token += batch_beam * K;
    if(threadIdx.x == 0) {
        for(int i = 0; i < K; i++) {
            output_values[i] = top_k_block.u[i];
            output_token[i] = top_k_block.p[i];
        }
        // for(int i = 0; i < K; i++)
        //     printf("(%d, %d, %f, %d)\n", batch_beam, i, (float)(output_values[i]), output_token[i]);
    }

  }

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__global__ void BeamSearchOnlineTopKStage2Kernel(
  const T* input_values,
  const int32_t* input_indices,
  int32_t K,
  int32_t vocab_size,
  int32_t parts_per_beam,
  T* output_values,
  int32_t* output_indices) {

    const int vector_id = blockIdx.x;
    const int thread_id = threadIdx.x;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    extern __shared__ char buf_s_[];  // intermediate result
    T* buf_value = reinterpret_cast<T*>(buf_s_);
    int32_t* buf_indices = reinterpret_cast<int32_t*>(buf_s_ + MAX_K * parts_per_beam * sizeof(int32_t));

    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    input_values += vector_id * K * parts_per_beam;
    input_indices += vector_id * K * parts_per_beam;

    TopK<T, MAX_K> partial;
    for (int i = 0; i < MAX_K; ++i) {
        partial.p[i] = -1;
        partial.u[i] = -MAX_T_VAL;
    }

    // load and unpack into registers through smem
    for (int idx = thread_id; idx < K * parts_per_beam; idx += THREADBLOCK_SIZE) {
        buf_value[idx] = input_values[idx];
        buf_indices[idx] = input_indices[idx];
    }
    __syncthreads();

    if (thread_id < parts_per_beam) {
        T* b_v = buf_value + thread_id * K;
        int32_t* b_i = buf_indices + thread_id * K;
        for (int i = 0; i < K; i++) {
            partial.p[i] = b_v[i];
            partial.u[i] = b_i[i];
        }
    }
    __syncthreads();

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (thread_id == 0) {
        output_values += vector_id * K;
        output_indices += vector_id * K;

        for (int i = 0; i < K; ++i) {
            if (i < K) {
                output_values[i] = total.p[i];
                output_indices[i] = total.u[i];
            }
        }
    }

  }

template<typename T, int MAX_K>
void LaunchBeamSearchOnlineTopKStage2Kernel(
  const T* topk_values_tmp,
  const int32_t* topk_indices_tmp,
  int32_t batch_size,
  int32_t num_beams,
  int32_t vocab_size,
  int32_t parts_per_beam,
  int32_t K,
  T* output_values,
  int32_t* output_indices,
  cudaStream_t stream) {

  ORT_ENFORCE(parts_per_beam <= 128, "Parts per beam should not be greater than 128");

  int smem_stage2_size = parts_per_beam * MAX_K * 2 * sizeof(int32_t);
  
  if(parts_per_beam <=32) {
    BeamSearchOnlineTopKStage2Kernel<T, MAX_K, 32><<<batch_size * num_beams, 32, smem_stage2_size, stream>>>(
        topk_values_tmp, topk_indices_tmp, K, vocab_size, parts_per_beam, output_values, output_indices);
    return;
  }

  if(parts_per_beam <=64) {
    BeamSearchOnlineTopKStage2Kernel<T, MAX_K, 64><<<batch_size * num_beams, 64, smem_stage2_size, stream>>>(
        topk_values_tmp, topk_indices_tmp, K, vocab_size, parts_per_beam, output_values, output_indices);
    return;
  }

    BeamSearchOnlineTopKStage2Kernel<T, MAX_K, 128><<<batch_size * num_beams, 128, smem_stage2_size, stream>>>(
        topk_values_tmp, topk_indices_tmp, K, vocab_size, parts_per_beam, output_values, output_indices);
    return;
  }

template<typename T, int MAX_K>
void TopKLauncherMaxK(
  const T* input,
  int batch_size,
  int num_beams,
  int vocab_size,
  int K,
  T* output_values,
  int32_t* output_indices,
  T* output_values_tmp,
  int32_t* output_indices_tmp,
  cudaStream_t stream)
{
    constexpr int THREAD_BLOCK_SIZE = (MAX_K < 16) ? (MAX_K < 8) ? 256 : 128 : 64;

    // int voc_parts = 4;
    // if (batch_size * num_beams < 256) {
    //     // volta has 80 SMs, so we aim for three waves
    //     voc_parts = (240 + batch_size * num_beams - 1) / (batch_size * num_beams);
    //     voc_parts = std::min(128, voc_parts);  // we implment up to 128
    // }

    int voc_parts = 1;
    dim3 grid(batch_size * num_beams, voc_parts);
    cudaFuncSetAttribute(BeamSearchOnlineTopKStage1Kernel<T, MAX_K, THREAD_BLOCK_SIZE>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxL1);
    BeamSearchOnlineTopKStage1Kernel<T, MAX_K, THREAD_BLOCK_SIZE>
        <<<grid, THREAD_BLOCK_SIZE, 0, stream>>>(input, K, vocab_size, (vocab_size + voc_parts - 1) / voc_parts, output_values_tmp, output_indices_tmp);

    LaunchBeamSearchOnlineTopKStage2Kernel<T, MAX_K>(
        output_values_tmp,
        output_indices_tmp,
        batch_size,
        num_beams,
        vocab_size,
        voc_parts,
        K,
        output_values,
        output_indices,
        stream);
}

template<typename T>
void LaunchTopK(
    const T* input,
    int batch_size,
    int num_beams,
    int vocab_size,
    int K,
    T* output_values,
    int32_t* output_indices,
    T* output_values_tmp,
    int32_t* output_indices_tmp,
    cudaStream_t stream) {
        ORT_ENFORCE(K <= 64, "Online TopK doesn't support K > 64");
        if(K <= 4) {
            return TopKLauncherMaxK<T, 4>(input, batch_size, num_beams, vocab_size, K, output_values, output_indices, output_values_tmp, output_indices_tmp, stream);
        } else if(K <= 8) {
            return TopKLauncherMaxK<T, 8>(input, batch_size, num_beams, vocab_size, K, output_values, output_indices, output_values_tmp, output_indices_tmp, stream);
        } else if(K <= 16) {
            return TopKLauncherMaxK<T, 16>(input, batch_size, num_beams, vocab_size, K, output_values, output_indices, output_values_tmp, output_indices_tmp, stream);
        } else if(K <= 32) {
            return TopKLauncherMaxK<T, 32>(input, batch_size, num_beams, vocab_size, K, output_values, output_indices, output_values_tmp, output_indices_tmp, stream);
        } else {
            return TopKLauncherMaxK<T, 64>(input, batch_size, num_beams, vocab_size, K, output_values, output_indices, output_values_tmp, output_indices_tmp, stream);
        }
    }

template void LaunchTopK(const float* input,
                         int batch_size,
                         int num_beams,
                         int vocab_size,
                         int K,
                         float* output_values,
                         int32_t* output_indices,
                         float* output_values_tmp,
                         int32_t* output_indices_tmp,
                         cudaStream_t stream);

template void LaunchTopK(const half* input,
                         int batch_size,
                         int num_beams,
                         int vocab_size,
                         int K,
                         half* output_values,
                         int32_t* output_indices,
                         half* output_values_tmp,
                         int32_t* output_indices_tmp,
                         cudaStream_t stream);

}
}
}
