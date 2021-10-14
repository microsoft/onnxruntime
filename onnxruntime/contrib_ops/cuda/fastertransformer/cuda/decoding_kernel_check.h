/*
Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once
#include "cuda_kernels.h"
#include "contrib_ops/cuda/fastertransformer/utils/common.h"
#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>

namespace fastertransformer{

void init_kernel_check(bool* d_finished, int* d_sequence_length, int* d_word_ids, float* d_cum_log_probs, const int sentence_id, const int batch_size, 
  const int beam_width, cudaStream_t stream);

void update_logits_kernel_check(float* logits, const float* bias, const int end_id, const bool* finished, const int m, const int n, cudaStream_t stream);

void broadcast_kernel_check(float* log_probs, float* cum_log_probs, const int batch_size, const int beam_width,
  const int vocab_size, cudaStream_t stream);

void topK_kernel_check(const float* log_probs, int* ids, const int batch_size, const int beam_width, const int vocab_size,
  cudaStream_t stream);

void update_kernel_check(float* log_probs, float* cum_log_probs, int* ids, bool* finished, int* parent_ids, int* sequence_length, 
  int* word_ids, int* output_ids,
  const int batch_size, const int beam_width,
  const int vocab_size, cudaStream_t stream,
  const int end_id, int* finished_count);

template <typename T>
void update_KV_cache_kernel_check(T** key_cache, T** value_cache, const int* beam_ids, const int batch_size, const int beam_width,
  const int head_num, const int size_per_head, const int step, const int cache_size, const int decoder_layers, cudaStream_t stream){
    
    const int hidden_dim = head_num * size_per_head;

    printf("[INFO] decoding update KV cache check for step %d. \n", step);
    const int src_id = step & 0x1;
    const int tgt_id = 1 - src_id;
    const int max_seq_len = cache_size / (batch_size * beam_width * hidden_dim);

    // CPU input
    T *h_key_cache_src = new T[cache_size * decoder_layers];
    T *h_value_cache_src = new T[cache_size * decoder_layers];
    int * h_beam_ids = new int[batch_size * beam_width];

    // CPU output
    T *h_key_cache_tgt_after_update_cpu = new T[cache_size * decoder_layers];
    T *h_value_cache_tgt_after_update_cpu = new T[cache_size * decoder_layers];

    // GPU output
    T *h_key_cache_tgt_after_update = new T[cache_size * decoder_layers];
    T *h_value_cache_tgt_after_update = new T[cache_size * decoder_layers];

    check_cuda_error(cudaMemcpy(h_key_cache_src, key_cache[src_id], sizeof(T) * cache_size * decoder_layers, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_value_cache_src, value_cache[src_id], sizeof(T) * cache_size * decoder_layers, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_beam_ids, beam_ids, sizeof(int) * batch_size * beam_width, cudaMemcpyDeviceToHost));

    // compute on GPU and copy the result to CPU
    // we use sequence major cache format here
    update_KV_cache_kernelLauncher<T>(key_cache, value_cache, beam_ids, nullptr, batch_size, beam_width, head_num, size_per_head, step, -1, cache_size, decoder_layers, stream);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    check_cuda_error(cudaMemcpy(h_key_cache_tgt_after_update, key_cache[tgt_id], sizeof(T) * cache_size * decoder_layers, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_value_cache_tgt_after_update, value_cache[tgt_id], sizeof(T) * cache_size * decoder_layers, cudaMemcpyDeviceToHost));

    // compute on CPU
    for(int i = 0; i < decoder_layers * max_seq_len; i++){
        for(int j = 0; j < batch_size * beam_width; j++){
            for(int k = 0; k < hidden_dim; k++){
                h_key_cache_tgt_after_update_cpu[i * batch_size * beam_width * hidden_dim + j * hidden_dim + k] = h_key_cache_src[i * batch_size * beam_width * hidden_dim +  h_beam_ids[j] * hidden_dim + k ];
                h_value_cache_tgt_after_update_cpu[i * batch_size * beam_width * hidden_dim + j * hidden_dim + k] = h_value_cache_src[i * batch_size * beam_width * hidden_dim +  h_beam_ids[j] * hidden_dim + k ];
            }
        }
    }

    // check key cache
    for(int i = 0; i < cache_size * decoder_layers; i++){
        float diff = (float)(h_key_cache_tgt_after_update_cpu[i] - h_key_cache_tgt_after_update[i]);
        if(diff < 0) diff = diff * -1;
        if(diff > 1e-5){
            printf("[ERROR] update key cache fail on %d with | %f - %f | = %f. \n", i, (float)h_key_cache_tgt_after_update_cpu[i], (float)h_key_cache_tgt_after_update[i], diff);
            exit(-1);
        }
    }

    // check value cache
    for(int i = 0; i < cache_size * decoder_layers; i++){
        float diff = (float)(h_value_cache_tgt_after_update_cpu[i] - h_value_cache_tgt_after_update[i]);
        if(diff < 0) diff = diff * -1;
        if(diff > 1e-5){
            printf("[ERROR] update value cache fail on %d with | %f - %f | = %f. \n", i, (float)h_value_cache_tgt_after_update_cpu[i], (float)h_value_cache_tgt_after_update[i], diff);
            exit(-1);
        }
    }

    delete [] h_key_cache_src;
    delete [] h_value_cache_src;
    delete [] h_beam_ids;

    delete [] h_key_cache_tgt_after_update_cpu;
    delete [] h_value_cache_tgt_after_update_cpu;

    delete [] h_key_cache_tgt_after_update;
    delete [] h_value_cache_tgt_after_update;
    printf("[INFO] decoding update KV cache check for step %d finish. \n", step);
}

} // end of namespace fastertransformer
