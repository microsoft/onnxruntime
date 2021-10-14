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

#include "contrib_ops/cuda/fastertransformer/utils/common.h"

#include "cuda_kernels.h"
#include "cub/cub.cuh"
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>
#include <vector>
#include <type_traits>

namespace fastertransformer
{
  /* ********************************** common kernel *********************************** */

  template <typename T>
  __global__ void init_kernel(bool* finished, 
                              int* sequence_length, 
                              int* word_ids, 
                              T* cum_log_probs, 
                              const int sentence_id, 
                              const int beam_width,
                              const int batch_size)
  {
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : 1e20f;
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * beam_width; index += blockDim.x * gridDim.x)
    {
      finished[index] = false;
      sequence_length[index] = 0;
      word_ids[index] = sentence_id;
      cum_log_probs[index] = (index % beam_width == 0) ? (T)0.0f: -MAX_T_VAL;
    }
  }

  template <typename T>
  void init_kernelLauncher(bool* finished, 
            int* sequence_length, 
            int* word_ids, 
            T* cum_log_probs, 
            const int sentence_id, 
            const int batch_size, 
            const int beam_width, 
            cudaStream_t stream)
  {
    //dim3 grid((int)ceil(batch_size * beam_width * 1.0 / 256));
    dim3 grid(GRID_COUNT(batch_size * beam_width, 256));
    dim3 block(256);
    
    init_kernel<T><<<grid, block, 0, stream>>>(finished,
                                               sequence_length,
                                               word_ids,
                                               cum_log_probs,
                                               sentence_id,
                                               beam_width,
                                               batch_size);
  }

  __global__ void sampling_init_kernel(bool* finished, 
                                       int* sequence_length, 
                                       int* word_ids, 
                                       const int start_id,
                                       const int batch_size)
  {
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size; index += blockDim.x * gridDim.x)
    {
      finished[index] = false;
      sequence_length[index] = 0;
      word_ids[index] = start_id;
    }
  }

  void sampling_init_kernelLauncher(bool* finished, 
                                    int* sequence_length, 
                                    int* word_ids, 
                                    const int start_id, 
                                    const int batch_size, 
                                    cudaStream_t stream)
  {
    //dim3 grid((int)ceil(batch_size * 1.0 / 256));
    dim3 grid(GRID_COUNT(batch_size, 256));
    dim3 block(256);

    
    sampling_init_kernel<<<grid, block, 0, stream>>>(finished,
                                                     sequence_length,
                                                     word_ids,
                                                     start_id,
                                                     batch_size);
  }

  template <typename T>
  __global__ void embedding_lookup_sine_position_encoding_kernel(T* from_tensor,
                                                                const T* embedding_table, 
                                                                const T* position_encoding,
                                                                const int* word_ids,
                                                                const int batch_size,
                                                                const int hidden_units)
  {
      // 1. lookup from embedding table
      // 2. multiply hidden_dim**0.5
      // 3. add the position encoding
      T scale = (T)sqrtf(float(hidden_units));
      for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * hidden_units; index += blockDim.x * gridDim.x)
      {
        const int row_index = index / hidden_units; 
        const int col_index = index % hidden_units; 
        from_tensor[index] = embedding_table[word_ids[row_index] * hidden_units + col_index] * scale + position_encoding[col_index];
      }
  }

  template <typename T>
  void embedding_lookup_sine_position_encoding_kernel_launcher(T* from_tensor,
                                                              const T* embedding_table, 
                                                              const T* position_encoding,
                                                              const int* word_ids,
                                                              const int batch_size,
                                                              const int hidden_units, 
                                                              cudaStream_t stream)
  {
      dim3 grid(min(batch_size, 65536));
      dim3 block(min(hidden_units, 1024));
      embedding_lookup_sine_position_encoding_kernel<T><<<grid, block, 0, stream>>>(from_tensor,
                                                                                  embedding_table,
                                                                                  position_encoding,
                                                                                  word_ids,
                                                                                  batch_size, 
                                                                                  hidden_units);
  }

  // TODO Add half2 implementation
  template <typename T>
  __global__ void embedding_position_lookups_kernel(T* from_tensor,
                                                    const T* embedding_table,
                                                    const T* pos_table,
                                                    const int* word_ids,
                                                    const int local_batch_size,
                                                    const int batch_size,
                                                    const int hidden_units,
                                                    int step,
                                                    int ite,
                                                    int max_input_len,
                                                    const int* start_lengths)
  {
      int timestep = step - 1;
      // if the input is padded in the batch, indices of the word_id and the pos_table also should be shifted forward by the length of the padding.
      int len_padding = max_input_len - start_lengths[local_batch_size * ite + blockIdx.x];
      int idx_word_id = (step == max_input_len) ? timestep - len_padding : timestep;
      int idx_pos_table = timestep - len_padding;

      // printf("batch id: %d, len_padding: %d, max_input_len: %d\n", local_batch_size * ite + blockIdx.x, len_padding, max_input_len);

      int *word_ids_buf = (int*)word_ids + idx_word_id * batch_size + local_batch_size * ite;
      for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < local_batch_size * hidden_units; index += blockDim.x * gridDim.x)
      {
          const int row_index = index / hidden_units;
          const int col_index = index % hidden_units;

          from_tensor[index] = embedding_table[word_ids_buf[row_index] * hidden_units + col_index]
          + pos_table[idx_pos_table * hidden_units + col_index];
      }
  }


  template <typename T>
  void embedding_position_lookups_kernel_launcher(T* from_tensor,
                                                  const T* embedding_table, 
                                                  const T* pos_table, 
                                                  const int* word_ids,
                                                  const int local_batch_size,
                                                  const int batch_size,
                                                  const int hidden_units, 
                                                  int step, 
                                                  int ite,
                                                  int max_input_len,
                                                  const int* start_lengths,
                                                  cudaStream_t stream)
  {
      dim3 grid(min(local_batch_size, 65536));
      dim3 block(min(hidden_units, 1024));
      embedding_position_lookups_kernel<T><<<grid, block, 0, stream>>>(from_tensor,
                                                                       embedding_table,
                                                                       pos_table,
                                                                       word_ids,
                                                                       local_batch_size,
                                                                       batch_size,
                                                                       hidden_units,
                                                                       step, 
                                                                       ite,
                                                                       max_input_len,
                                                                       start_lengths);
  }

  template <typename T> __launch_bounds__(1024, 1)
  __global__ void start_id_embedding_position_lookups_kernel(T* from_tensor,
                                                             int* output_ids,
                                                             const T* embedding_table,
                                                             const T* pos_table,
                                                             const int* word_ids,
                                                             const int start_step,
                                                             const int length,
                                                             const int max_length,
                                                             const int batch_size,
                                                             const int hidden_units)
  {
      for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * length * hidden_units; index += blockDim.x * gridDim.x)
      {
          // transpose the word_ids [batch, length] (part of [batch, max_length]) to output_ids [length, batch]
          if(index < batch_size * max_length)
          {
            const int seq_id = index % max_length;
            const int batch_id = index / max_length;
            if(seq_id < length)
              output_ids[seq_id * batch_size + batch_id] = word_ids[index];
            // output_ids[index] = word_ids[index];
          }
        
          // embedding lookup from word ids [batch, length] (part of [batch, max_length]) and [vocab, hidden] to generate embedding [batch, length, hidden]
          const int word_index = index / hidden_units;
          const int word_index_row = word_index / length;
          const int word_index_col = word_index % length;
          const int real_word_index = word_index_row * max_length + word_index_col;
          const int step = start_step + word_index % length;
          const int col_index = index % hidden_units;
          from_tensor[index] = embedding_table[word_ids[real_word_index] * hidden_units + col_index] 
                              + pos_table[(step - 1) * hidden_units + col_index];
      }
  }


  template <typename T>
  void start_id_embedding_position_lookups_kernel_launcher(T* from_tensor,
                                                           int *output_ids,
                                                           const T* embedding_table, 
                                                           const T* pos_table, 
                                                           const int* word_ids,
                                                           const int start_step,
                                                           const int length,
                                                           const int max_length,
                                                           const int batch_size,
                                                           const int hidden_units, 
                                                           cudaStream_t stream)
  {
      dim3 grid(min(batch_size * length, 65536));
      dim3 block(min(hidden_units, 1024));
      start_id_embedding_position_lookups_kernel<T><<<grid, block, 0, stream>>>(from_tensor,
                                                                                output_ids,
                                                                                embedding_table,
                                                                                pos_table,
                                                                                word_ids,
                                                                                start_step,
                                                                                length,
                                                                                max_length,
                                                                                batch_size,
                                                                                hidden_units);
  }

  // TODO Add half2 implementation
  template <typename T>
  __global__ void apply_temperature_penalty_kernel(T* logits,
                                                   const T temperature_inverse,
                                                   const int m,
                                                   const int vocab_size,
                                                   const int vocab_size_padd)
  {
      const bool IS_FP16 = std::is_same<T, half>::value;
      const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;
      for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < m * vocab_size_padd; index += blockDim.x * gridDim.x)
      {
          if(index % vocab_size_padd < vocab_size) logits[index] = logits[index] * temperature_inverse;
          else logits[index] = -MAX_T_VAL;
      }
  }

  template <typename T>
  void apply_temperature_penalty_kernelLauncher(T* logits,
                                                const T temperature,
                                                const int m,
                                                const int vocab_size,
                                                const int vocab_size_padd,
                                                cudaStream_t stream) {
      dim3 grid(min(m, 65536));
      dim3 block(min(vocab_size_padd, 1024));
      const T temperature_inverse = (T)(1.f / (float) temperature);
      apply_temperature_penalty_kernel<T><<<grid, block, 0, stream>>>(logits,
                                                                      temperature_inverse,
                                                                      m,
                                                                      vocab_size,
                                                                      vocab_size_padd);
  }

  template <typename T>
  __global__ void apply_repetition_penalty_kernel(T* logits,
                                                  const float penalty,
                                                  int* start_ids,
                                                  int* output_ids,
                                                  const int batch_size,
                                                  const int local_batch_size,
                                                  const int vocab_size,
                                                  const int vocab_size_padd,
                                                  const int* start_lengths,
                                                  const int max_input_len,
                                                  const int step,
                                                  const int ite) {

    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < local_batch_size * step; index += blockDim.x * gridDim.x) {
      int tid = index / local_batch_size;
      int lid = index % local_batch_size;
      int bid = lid + ite * local_batch_size;

      bool is_mask = (tid >= start_lengths[bid] && tid < max_input_len);

      if (is_mask) continue;  // padding has nothing to do with repetition penalty.

      int vid;
      if (tid < start_lengths[bid]) {  // get tokens from context input
        int idx = bid * max_input_len + tid; // start_ids shape: (batch_size, max_input_len)
        vid = start_ids[idx];
      } else {  // get tokens from previous output
        int idx = batch_size * tid + local_batch_size * ite + lid;  // output_ids shape: (input_len + output_len, batch_size)
        vid = output_ids[idx];
      }

      if(vid >= vocab_size) continue;

      int idx_out = lid * vocab_size_padd + vid;  // logits shape: (local_batch_size, vocab_size_padd)
      logits[idx_out] = logits[idx_out] < T(0) ? float(logits[idx_out]) * penalty : float(logits[idx_out]) / penalty;
    }
  }

  template <typename T>
  void apply_repetition_penalty_kernelLauncher(T* logits,
                                               const float penalty,
                                               int* start_ids,
                                               int* output_ids,
                                               const int batch_size,
                                               const int local_batch_size,
                                               const int vocab_size,
                                               const int vocab_size_padd,
                                               const int* start_lengths,
                                               const int max_input_len,
                                               const int step,
                                               const int ite,
                                               cudaStream_t stream) {
    
    dim3 block(512);
    //dim3 grid((int)(ceil(local_batch_size * step / 512.)));
    dim3 grid(GRID_COUNT(local_batch_size * step, 512));
    apply_repetition_penalty_kernel<T><<<grid, block, 0, stream>>>(logits,
                                                                   penalty,
                                                                   start_ids,
                                                                   output_ids,
                                                                   batch_size,
                                                                   local_batch_size,
                                                                   vocab_size,
                                                                   vocab_size_padd,
                                                                   start_lengths,
                                                                   max_input_len,
                                                                   step,
                                                                   ite);
  }

  __global__ void set_start_ids_kernel(int* out_ids,
                                       const int* in_ids, 
                                       const int max_start_len, 
                                       const int step, 
                                       const int ite,
                                       const int batch_size,
                                       const int local_batch_size,
                                       const int end_id)
  {
      const int id = blockIdx.x * blockDim.x + threadIdx.x;
      if(id < local_batch_size)
      {
        int in_id = in_ids[(ite * local_batch_size + id) * max_start_len + step];
        if(in_id != end_id)
          out_ids[step * batch_size + ite * local_batch_size + id] = in_ids[(ite * local_batch_size + id) * max_start_len + step];
      }
  }

  void set_start_ids_kernelLauncher(int* out_ids,
                                    const int* in_ids,
                                    const int max_start_len,
                                    const int step,
                                    const int ite,
                                    const int batch_size,
                                    const int local_batch_size,
                                    const int end_id,
                                    cudaStream_t stream)
  {
      //dim3 grid((int)(ceil(local_batch_size / 512.)));
      dim3 grid(GRID_COUNT(local_batch_size, 512));

      set_start_ids_kernel<<<grid, 512, 0, stream>>>(out_ids,
                                                     in_ids,
                                                     max_start_len,
                                                     step,
                                                     ite,
                                                     batch_size,
                                                     local_batch_size,
                                                     end_id);
  }

  template <typename T>
  __global__ void kernel_padding_kernel(T *padded_kernel, const T *kernel,
                                      const int row_dim, const int col_dim, const int padded_col_dim)
  {
    for(int id = threadIdx.x + blockIdx.x * blockDim.x; id < row_dim * padded_col_dim; id += blockDim.x * gridDim.x)
    {
      int row_id = id / padded_col_dim;
      int col_id = id % padded_col_dim;
      if(col_id < col_dim)
      {
        padded_kernel[id] = kernel[row_id * col_dim + col_id];
      }
      else
      {
        padded_kernel[id] = (T)(0.0f);
      }
    }
  }

  template <typename T>
  void kernel_padding_kernelLauncher(T *padded_kernel, const T *kernel,
                                     const int row_dim, const int col_dim, const int padded_col_dim, cudaStream_t stream)
  {
    // pad 0 into the kernel from shape [row_dim, col_dim] to [row_dim, padded_col_dim]
    dim3 block(512);
    dim3 grid(min(65536, GRID_COUNT(row_dim * padded_col_dim, 512))); //(int)(ceil(row_dim * padded_col_dim / 512.)) ));
    kernel_padding_kernel<<<grid, block, 0, stream>>>(padded_kernel, kernel, row_dim, col_dim, padded_col_dim);
  }

  template <typename T1, typename T2>
  __global__ void bias_padding_kernel(T1 *padded_bias, const T2 *bias,
                                      const int col_dim, const int padded_col_dim)
  {
      const int index = blockIdx.x * blockDim.x + threadIdx.x;
      if(index < col_dim)
      {
        padded_bias[index] = (T1)bias[index];
      }
      else if(index >= col_dim && index < padded_col_dim)
      {
        padded_bias[index] = (T1)(std::is_same<T1, half>::value ? -60000 : -1e20f);
      }
  }

  template <typename T1, typename T2>
  void bias_padding_kernelLauncher(T1 *padded_bias, const T2 *bias,
                                   const int col_dim, const int padded_col_dim, cudaStream_t stream)
  {
    // pad -max into the bias from shape [col_dim] to [padded_col_dim]
    dim3 block(512);
    dim3 grid(GRID_COUNT(padded_col_dim, 512));
    //dim3 grid( (int)(ceil(padded_col_dim / 512.)) );

    assert(grid.x < 65536);
    bias_padding_kernel<<<grid, block, 0, stream>>>(padded_bias, bias, col_dim, padded_col_dim);
  }

  /* *************************** end of common kernel *********************************** */

  /* ********************************** BeamSearch kernel *********************************** */

  template<typename T>
  __global__
  void broadcast_kernel(T* log_probs, 
                        T* cum_log_probs, 
                        const int vocab_size, 
                        const int N)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = tid / vocab_size;

    if(tid < N)
      log_probs[tid] += cum_log_probs[bid];
}

  void broadcast_kernelLauncher(float* log_probs, 
                                float* cum_log_probs, 
                                const int batch_size, 
                                const int beam_width, 
                                const int vocab_size, 
                                cudaStream_t stream)
  {
    
    int N = batch_size * beam_width * vocab_size;
    dim3 block(1024);
    dim3 grid((N - 1) / block.x + 1);
  
    broadcast_kernel<float><<<grid, block, 0, stream>>>(log_probs, cum_log_probs, vocab_size, N);
  }

  template <typename T>
  __global__
  void update_kernel(T* log_probs, T* cum_log_probs, 
                    bool* finished, 
                    int* parent_ids, int* sequence_length, 
                    int* word_ids, int* output_ids, 
                    const int batch_size, const int beam_width, 
                    const int vocab_size, const int end_id, 
                    int* finished_count)
  {
    int tid = threadIdx.x;
    sequence_length[tid] = finished[tid] ? sequence_length[tid] : sequence_length[tid] + 1;

    int beam_id = word_ids[tid] / vocab_size;
    int word_id = word_ids[tid] % vocab_size;

    cum_log_probs[tid] = log_probs[word_ids[tid]];
    sequence_length[tid] = sequence_length[beam_id];
    finished[tid] = word_id == end_id ? 1 : 0;
    parent_ids[tid] = beam_id;
    word_ids[tid] = word_id;
    output_ids[tid] = word_id;
  }

  void update_kernelLauncher(float* log_probs, float* cum_log_probs, 
    bool* finished, 
    int* parent_ids, int* sequence_length,
    int* word_ids, int* output_ids, 
    const int batch_size, const int beam_width, 
    const int vocab_size, cudaStream_t stream, 
    const int end_id, int* finished_count)
  { 
    dim3 grid(1);
    dim3 block(batch_size * beam_width);

    assert(block.x <= 1024);

    update_kernel<float><<<grid, block, 0, stream>>>(log_probs, cum_log_probs,
                                              finished, parent_ids, sequence_length,
                                              word_ids, output_ids, batch_size, 
                                              beam_width, vocab_size, end_id, 
                                              finished_count);
  }

  template <typename T>
  __global__
  void update_kernel_v2(bool* finished, int* parent_ids, 
                        int* sequence_length, 
                        int* word_ids, int* output_ids, 
                        const int vocab_size, const int end_id, 
                        const int batch_size, const int beam_width,
                        int* finished_count)
  {
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * beam_width; index += blockDim.x * gridDim.x)
    {
      sequence_length[index] = finished[index] ? sequence_length[index] : sequence_length[index] + 1;
  
      int beam_id = word_ids[index] / vocab_size;
      int word_id = word_ids[index] % vocab_size;
  
      sequence_length[index] = sequence_length[beam_id];
      finished[index] = word_id == end_id ? 1 : 0;
      parent_ids[index] = beam_id;
      word_ids[index] = word_id;
      output_ids[index] = word_id;
    }
  }

  void update_kernelLauncher_v2(bool* finished, int* parent_ids, 
                                int* sequence_length, int* word_ids, 
                                int* output_ids, 
                                int* finished_count,
                                DecodingBeamsearchArguments args,
                                cudaStream_t stream)
  {
    //dim3 grid((int)ceil(args.batch_size_ * args.beam_width_ * 1.0 / 256));
    dim3 grid(GRID_COUNT(args.batch_size_ * args.beam_width_, 256));
    dim3 block(256);

    update_kernel_v2<float><<<grid, block, 0, stream>>>(finished, parent_ids, 
                                                        sequence_length, word_ids, 
                                                        output_ids, args.vocab_size_padded_, 
                                                        args.end_id_, 
                                                        static_cast<int>(args.batch_size_), args.beam_width_,
                                                        finished_count);
  }

  template <typename T>
  __global__ void update_KV_cache_kernel(const T* __restrict key_src_cache, 
                                        T* key_tgt_cache,
                                        const T* __restrict value_src_cache, 
                                        T* value_tgt_cache,
                                        const int* beam_ids,
                                        const bool* finished, 
                                        const int batch_size, 
                                        const int beam_width, 
                                        const int hidden_dim, 
                                        const int cache_size, 
                                        const int step, 
                                        const int decoder_layers)
  {
    int layer_id = blockIdx.x / batch_size / beam_width / step;
    int batch_id = (blockIdx.x % (batch_size * beam_width * step)) / (beam_width * step);
    int beam_id = (blockIdx.x % (beam_width * step)) / step;
    if(finished[batch_id * beam_width + beam_id]) return;
    int step_id = blockIdx.x % step;

    int hidden_id = step_id * batch_size * beam_width * hidden_dim + 
      beam_ids[batch_id * beam_width + beam_id] * hidden_dim;

    int tgt_hidden_id = step_id * batch_size * beam_width * hidden_dim + 
      batch_id * beam_width * hidden_dim + beam_id * hidden_dim;

    const T* key_src_ptr = key_src_cache + layer_id * cache_size;
    T* key_tgt_ptr = key_tgt_cache + layer_id * cache_size;
    const T* value_src_ptr = value_src_cache + layer_id * cache_size;
    T* value_tgt_ptr = value_tgt_cache + layer_id * cache_size;


    for(int tid = threadIdx.x; tid < hidden_dim; tid += blockDim.x)
    {
      key_tgt_ptr[tgt_hidden_id + tid] = key_src_ptr[hidden_id + tid];
      value_tgt_ptr[tgt_hidden_id + tid] = value_src_ptr[hidden_id + tid];
    }
    
  }

  template <>
  __global__ void update_KV_cache_kernel(const half* __restrict key_src_cache, 
                                        half* key_tgt_cache,
                                        const half* __restrict value_src_cache, 
                                        half* value_tgt_cache,
                                        const int* beam_ids, 
                                        const bool* finished,
                                        const int batch_size, 
                                        const int beam_width, 
                                        const int hidden_dim, 
                                        const int cache_size, 
                                        const int step, 
                                        const int decoder_layers)
  {
    int layer_id = blockIdx.x / batch_size / beam_width / step;
    int batch_id = (blockIdx.x % (batch_size * beam_width * step)) / (beam_width * step);
    int beam_id = (blockIdx.x % (beam_width * step)) / step;
    if(finished[batch_id * beam_width + beam_id]) return;
    int step_id = blockIdx.x % step;

    int hidden_id = (step_id * batch_size * beam_width * hidden_dim + 
      beam_ids[batch_id * beam_width + beam_id] * hidden_dim) / 2;

    int tgt_hidden_id = (step_id * batch_size * beam_width * hidden_dim + 
      batch_id * beam_width * hidden_dim + beam_id * hidden_dim) / 2;

    const half2* key_src_ptr = (const half2*)key_src_cache + layer_id * cache_size / 2;
    half2* key_tgt_ptr = (half2*)key_tgt_cache + layer_id * cache_size / 2;
    const half2* value_src_ptr = (const half2*)value_src_cache + layer_id * cache_size / 2;
    half2* value_tgt_ptr = (half2*)value_tgt_cache + layer_id * cache_size / 2;
    
    for(int tid = threadIdx.x; tid < hidden_dim / 2; tid += blockDim.x)
    {
      key_tgt_ptr[tgt_hidden_id + tid] = key_src_ptr[hidden_id + tid];
      value_tgt_ptr[tgt_hidden_id + tid] = value_src_ptr[hidden_id + tid];
    }
    
  }

  template <typename T>
  __global__ void update_KV_batch_major_cache_kernel(const T* __restrict key_src_cache, 
                                        T* key_tgt_cache,
                                        const T* __restrict value_src_cache, 
                                        T* value_tgt_cache,
                                        const int* beam_ids,
                                        const bool* finished, 
                                        const int batch_size, 
                                        const int beam_width,
                                        const int size_per_head, 
                                        const int cache_size, 
                                        const int step,
                                        const int max_seq_len,
                                        const int decoder_layers)
  {
    int layer_id = blockIdx.z;
    int head_id = blockIdx.y;
    int bb_id = blockIdx.x;
    int batch_id = bb_id / beam_width;
    int beam_id = bb_id % beam_width;

    if(finished[batch_id * beam_width + beam_id]) return;

    const int hidden_dim = size_per_head * gridDim.y;

    int src_offset = layer_id * cache_size + 
                      (beam_ids[batch_id * beam_width + beam_id] * hidden_dim + 
                                                         head_id * size_per_head) * max_seq_len;
    int tgt_offset = layer_id * cache_size + 
                      ((batch_id * beam_width + beam_id) * hidden_dim + 
                                                         head_id * size_per_head) * max_seq_len;

    // for better memory access always do 16 byte loads.
    // [B, H, Dh/x, L, x]  and [B, H, L, Dh/x, x] (i.e. [B, H, L, Dh])
    auto key_src_ptr = reinterpret_cast<const uint4*>(key_src_cache + src_offset);
    auto value_src_ptr = reinterpret_cast<const uint4*>(value_src_cache + src_offset);
    auto key_tgt_ptr = reinterpret_cast<uint4*>(key_tgt_cache + tgt_offset);
    auto value_tgt_ptr = reinterpret_cast<uint4*>(value_tgt_cache + tgt_offset);
    constexpr int x = (sizeof(T) == 4)? 4 : 8;

    // step starts from 1
    #if 0
    constexpr int WARP_SIZE = 32;
    const int num_warps = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    for (int dhx = warp_id; dhx < size_per_head/x; dhx += num_warps)
    {
      for (int tid = lane_id; tid < step; tid += WARP_SIZE)
      {
        key_tgt_ptr[dhx * max_seq_len + tid] = key_src_ptr[dhx * max_seq_len + tid];
      }
    }
    #else
    // seems to be a bit faster
    for (int tid = threadIdx.x; tid < max_seq_len * size_per_head/x; tid += blockDim.x)
    {
      // could consider fast int division here
      if (tid % max_seq_len < step)
      {
        key_tgt_ptr[tid] = key_src_ptr[tid];
      }
    }
    #endif

    for (int tid = threadIdx.x; tid < step * size_per_head/x; tid += blockDim.x)
    {
      value_tgt_ptr[tid] = value_src_ptr[tid];
    }
  }

  template <typename T>
  void update_KV_cache_kernelLauncher(T** key_cache, 
                                      T** value_cache, 
                                      const int* beam_ids, 
                                      const bool* finished,
                                      const int batch_size, 
                                      const int beam_width,
                                      const int head_num, 
                                      const int size_per_head,
                                      const int step,
                                      const int decoder_max_seq_len,
                                      const int cache_size, 
                                      const int decoder_layers, 
                                      cudaStream_t stream)
  {
    int src_id = step & 0x1;
    int tgt_id = 1 - src_id;

    if (decoder_max_seq_len < 0)
    {
      int hidden_dim = head_num * size_per_head;
      dim3 grid(decoder_layers * batch_size * beam_width * step);
      dim3 block(min(1024, hidden_dim));
      block.x = block.x / (4 / sizeof(T));
  
      update_KV_cache_kernel<<<grid, block, 0, stream>>>(
        key_cache[src_id], key_cache[tgt_id],
        value_cache[src_id], value_cache[tgt_id],
        beam_ids, finished,
        batch_size, beam_width, hidden_dim, cache_size, step, decoder_layers);
    }
    else
    {
      dim3 grid(batch_size * beam_width, head_num, decoder_layers);
      constexpr int block_sz = 128;

      update_KV_batch_major_cache_kernel<<<grid, block_sz, 0, stream>>>(
        key_cache[src_id], key_cache[tgt_id],
        value_cache[src_id], value_cache[tgt_id],
        beam_ids, finished,
        batch_size, beam_width, size_per_head, cache_size, step, 
        decoder_max_seq_len, decoder_layers);
    }

  }

  template <typename T>
  __global__
  void apply_logit_penalties_kernel(int step,
      int vocab_size, 
      int beam_width,
      T* log_probs, 
      int* current_ids,
      int* previous_ids,
      int* parent_ids,
      int  end_id,
      float inv_temp,
      float len_penalty,
      float repeat_penalty,
      int* vocab_mask) {

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bbid = blockIdx.y;
    int bbsize = gridDim.y;
    int batchid = bbid / beam_width;
    // int beamid = bbid % beam_width;

    for (int i = tid + bid*blockDim.x; i < vocab_size; i +=  blockDim.x*gridDim.x) {
      log_probs[i+bbid*vocab_size] *= inv_temp;
    }
    if (tid == 0 && bid == 0) {
      // apply repetition penalty (this can apply the penalty multiple times to a repeated word).
      int prev_id = current_ids[bbid];
      if (log_probs[prev_id+bbid*vocab_size] > T(0)) {
        log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) / repeat_penalty;
      } else {
        log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) * repeat_penalty;
      }
      if (step > 1) {
        int parent_beamid = parent_ids[bbsize*(step-2) + bbid];
        for (int i = step-2; i > 0; --i) {
          prev_id = previous_ids[bbsize*i+batchid*beam_width+parent_beamid];
          if (log_probs[prev_id+bbid*vocab_size] > T(0)) {
            log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) / repeat_penalty;
          } else {
            log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) * repeat_penalty;
          }
          //if (i > 0) parent_beamid = parent_ids[bbsize*(i-1)+parent_beamid];
          parent_beamid = parent_ids[bbsize*(i-1)+parent_beamid];
        }
      }
      prev_id = previous_ids[batchid*beam_width];
      if (log_probs[prev_id+bbid*vocab_size] > T(0)) {
        log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) / repeat_penalty;
      } else {
        log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) * repeat_penalty;
      }
      // apply length penalty
      if (log_probs[end_id+bbid*vocab_size] > T(0))  {
        log_probs[end_id+bbid*vocab_size] = float(log_probs[end_id+bbid*vocab_size]) / len_penalty;
      } else {
        log_probs[end_id+bbid*vocab_size] = float(log_probs[end_id+bbid*vocab_size]) * len_penalty;
      }
    }
  }

  template <typename T>
  void apply_logit_penalties(int step, 
                            T* log_probs, 
                            int* current_ids,
                            int* previous_ids, 
                            int* parent_ids,
                            GptArguments args,
                            cudaStream_t stream) {

    int vocab_size = args.vocab_size_padded_;
    int beam_width = 1;
    int batch_size = static_cast<int>(args.batch_size_);
    dim3 block(256);
    dim3 grid((vocab_size + block.x - 1)/block.x, beam_width*batch_size);
    apply_logit_penalties_kernel<T><<<grid, block, 0, stream>>> (step, 
        vocab_size, 
        beam_width, 
        log_probs, 
        current_ids,
        previous_ids, 
        parent_ids,
        args.end_id_, 
        1.f/args.temperature_, 
        args.len_penalty,
        args.repetition_penalty_, 
        args.vocab_mask);
  }

  extern __shared__ char transposeTileBuf_g[];

  template <typename data_type>
  __global__ void transpose_kernel(data_type * __restrict__ out, const data_type *__restrict__ in, int height, int width, int tH, int tW, int stride)
  // int tH, int tW should be template parameters for the best performance, we do not do that sine the task is tiny.
  // batch  stride (blockIdx.z dimension) for fully packed tensor ==  height * width
  {
      data_type *tile = (data_type *)transposeTileBuf_g;

      int tidx = threadIdx.x % tW;
      int tidy = threadIdx.x / tW;

      int xIndex = blockIdx.x * tW + tidx;
      int yIndex = blockIdx.y * tH + tidy;
      int indexIn = xIndex + yIndex * width;

      if ((xIndex < width) && (yIndex < height))
      {
          tile[tidy * tW + tidx] = in[blockIdx.z * stride + indexIn];
      }

      tidx = threadIdx.x % tH;
      tidy = threadIdx.x / tH;

      xIndex = blockIdx.y * tH + tidx;
      yIndex = blockIdx.x * tW + tidy;
      int indexOut = xIndex + yIndex * height;

      __syncthreads();

      if ((xIndex < height) &&  (yIndex < width))
      {
          out[blockIdx.z * stride + indexOut] = tile[tidx * tW + tidy];
      }
  }

  template <typename data_type>
  void transpose(data_type *out, const data_type *in, int batch, int height, int width, int stride, cudaStream_t stream)
  {
      int tW, tH;

      if ((width <= 1) || (height <= 1) )
      {
          assert(0);
      }

      if (height <= width)
      {
          tH = std::min((height / 2) * 2, 16);
          tW = std::min(256 / tH, width);
      }
      else
      {
          tW = std::min((width / 2) * 2, 16);
          tH = std::min(256 / tW, height);
      }
      assert(tW <= width);
      assert(tH <= height);

      dim3 grid((width + tW - 1) / tW, (height + tH - 1) / tH, batch);
      transpose_kernel<data_type><<<grid, tW * tH, tH * tW * sizeof(data_type), stream>>>(out, in, height, width, tH, tW, stride);
  }

  // TODO Add half2 implementation
  template <typename DataType_>
  __global__ void transpose_axis_01_kernel(DataType_ *out, DataType_ *in, const int dim0, const int dim1, const int dim2)
  {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < dim0 * dim1 * dim2)
    {
      const int input_dim2_index = index % dim2;
      index = (index - input_dim2_index) / dim2;
      const int input_dim1_index = index % dim1;
      index = (index - input_dim1_index) / dim1;
      const int input_dim0_index = index % dim0;

      out[input_dim1_index * dim0 * dim2 + 
          input_dim0_index * dim2 + 
          input_dim2_index] = in[input_dim0_index * dim1 * dim2 + 
                                 input_dim1_index * dim2 + 
                                 input_dim2_index];
    }
  }

  template <typename DataType_>
  void transpose_axis_01_kernelLauncher(DataType_ *out, DataType_ *in, const int dim0, 
                                        const int dim1, const int dim2, cudaStream_t stream)
  {
    dim3 block(512);
    dim3 grid(GRID_COUNT(dim0 * dim1 * dim2, 512));
    //dim3 grid((int)(ceil(dim0 * dim1 * dim2 / 512.)));

    transpose_axis_01_kernel<<<grid, block, 0, stream>>>(out, in, dim0, dim1, dim2);
  }

  /* *************************** end of BeamSearch kernel *********************************** */

  /* ********************************** Sampling kernel *********************************** */
  __global__ void topp_initialization_kernel(bool* finished,
                                             int* sequence_length, 
                                             int* word_ids,
                                             int* topp_id_val_buf,
                                             int* topp_offset_buf,
                                             const int batch_size, 
                                             const int n,
                                             const int start_id)
  {
      int tid = threadIdx.x;
      int bid = blockIdx.x;
  
      if(bid == 0)
      {
          for(int i = tid; i < batch_size + 1; i+= blockDim.x)
          {
              topp_offset_buf[i] = i * n;
          }
          
          for(int i = tid; i < batch_size; i+= blockDim.x)
          {
              if(finished != nullptr) finished[i] = false;
              if(sequence_length != nullptr) sequence_length[i] = 0;
              if(word_ids != nullptr) word_ids[i] = start_id; 
          }
      }
  
      int index = tid + bid * blockDim.x;
      while(index < batch_size * n)
      {
          topp_id_val_buf[index] = index % n;
          index += blockDim.x * gridDim.x;
      }
  }

  __global__ void topp_initialization_kernel_v2(bool* finished,
                                            int* sequence_length, 
                                            int* word_ids,
                                            int* topp_id_val_buf,
                                            int* topp_offset_buf,
                                            int* begin_topp_offset_buf_,
                                            const int batch_size, 
                                            const int n,
                                            const int start_id)
  {
      int tid = threadIdx.x;
      int bid = blockIdx.x;
  
      if(bid == 0)
      {
          for(int i = tid; i < batch_size + 1; i+= blockDim.x)
          {
              topp_offset_buf[i] = i * n;
              begin_topp_offset_buf_[i] = topp_offset_buf[i];
          }
          
          for(int i = tid; i < batch_size; i+= blockDim.x)
          {
              if(finished != nullptr) finished[i] = false;
              if(sequence_length != nullptr) sequence_length[i] = 0;
              if(word_ids != nullptr) word_ids[i] = start_id; 
          }
      }
  
      int index = tid + bid * blockDim.x;
      while(index < batch_size * n)
      {
          topp_id_val_buf[index] = index % n;
          index += blockDim.x * gridDim.x;
      }
  }



  void topp_initialization_kernelLauncher(bool* finished,
                                          int* sequence_length, 
                                          int* word_ids,
                                          int* topp_id_val_buf,
                                          int* topp_offset_buf,
                                          const int n,
                                          DecodingSamplingArguments args,
                                          cudaStream_t stream)
  {
      // n: the coloumn number of logits_buffer for top_p sampling
      topp_initialization_kernel<<<32, 512, 0, stream>>>(finished,
                                                         sequence_length,
                                                         word_ids,
                                                         topp_id_val_buf,
                                                         topp_offset_buf,
                                                         static_cast<int>(args.batch_size_), 
                                                         n,
                                                         args.start_id_);
  }

  void topp_initialization_kernelLauncher_v2(bool* finished,
    int* sequence_length, 
    int* word_ids,
    int* topp_id_val_buf,
    int* topp_offset_buf,
    int* begin_topp_offset_buf_,
    const int n,
    DecodingSamplingArguments args,
    cudaStream_t stream)
{
    // n: the coloumn number of logits_buffer for top_p sampling
    topp_initialization_kernel_v2<<<32, 512, 0, stream>>>(finished,
                        sequence_length,
                        word_ids,
                        topp_id_val_buf,
                        topp_offset_buf,
                        begin_topp_offset_buf_,
                        static_cast<int>(args.batch_size_), 
                        n,
                        args.start_id_);
}

  template <typename T>
  size_t get_topp_sort_temp_storage_size(const T* log_probs,
                                         const int* id_vals,
                                         T* sorted_log_probs,
                                         int* sorted_id_vals, 
                                         int* topp_offset_buf,
                                         const int batch_size,
                                         const int vocab_size)
  {
      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;
      
      cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, 
                                                      temp_storage_bytes,
                                                      log_probs, 
                                                      sorted_log_probs,
                                                      id_vals, 
                                                      sorted_id_vals, 
                                                      vocab_size * batch_size,
                                                      batch_size, 
                                                      topp_offset_buf, topp_offset_buf + 1);
      return temp_storage_bytes;
  }
  /* *************************** end of Sampling kernel *********************************** */

  // TODO Remove the gather_tree_kernel of th_op/utils.cu
  // modified from TensorFlow's implementation of tf.contrib.seq2seq.gather_tree
  __global__ void gather_tree_kernel(const int batch_size, const int max_time, const int beam_width, const int end_token,
                                    const int* step_ids, const int* parent_ids, int* max_sequence_lengths, int* beams) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size * beam_width; i += gridDim.x * blockDim.x) {
      const int batch = i / beam_width;
      const int beam = i % beam_width;

      const int max_seq_len_b = min(max_time, __ldg(max_sequence_lengths + batch));
      if (max_seq_len_b <= 0) {
        continue;
      }

  #define GET_IX(time_ix, beam_ix) (batch_size * beam_width * (time_ix) + beam_width * batch + (beam_ix))

      const int initial_beam_ix = GET_IX(max_seq_len_b - 1, beam);
      beams[initial_beam_ix] = __ldg(step_ids + initial_beam_ix);
      int parent = __ldg(parent_ids + initial_beam_ix) % beam_width;
      bool found_bad = false;
      for (int level = max_seq_len_b - 2; level >= 0; --level) {
        const int level_beam_ix = GET_IX(level, beam);
        const int level_parent_ix = GET_IX(level, parent);
        if (parent < 0 || parent > beam_width) {
          beams[level_beam_ix] = -1;
          parent = -1;
          found_bad = true;
        } else {
          beams[level_beam_ix] = __ldg(step_ids + level_parent_ix);
          parent = __ldg(parent_ids + level_parent_ix) % beam_width;
        }
      }
  // Not necessary when using a BeamSearchDecoder, but necessary
  // when a user feeds in possibly broken trajectory (i.e., non-eos
  // entries in a beam following eos entries).
      if (!found_bad) {
        bool finished = false;
        for (int time = 0; time < max_seq_len_b; ++time) {
          const int level_beam_ix = GET_IX(time, beam);
          if (finished) {
            beams[level_beam_ix] = end_token;
          } else if (beams[level_beam_ix] == end_token) {
            finished = true;
          }
        }
      }
  #undef GET_IX
    }
  }


  void gather_tree_kernel_launcher(int max_time, int batch_size, int beam_width,
                                  int* step_ids, int* parent_ids, int* max_sequence_lengths,
                                  int end_token, int* beams, cudaStream_t stream) {
    int batchbeam = batch_size * beam_width;
    dim3 grid(1), block(batchbeam);
    // though decoder do not support > 1024 for now
    if (batchbeam > 1024) {
      grid.x = GRID_COUNT(batch_size * beam_width, 1024); //ceil(batch_size * beam_width / 1024.);
      block.x = 1024;
    }
    gather_tree_kernel<<<grid, block, 0, stream>>>(batch_size, max_time, beam_width, end_token,
                                                  step_ids, parent_ids, max_sequence_lengths, beams);
  }

  /* ********************************** Instantiation *********************************** */
  template 
  void embedding_lookup_sine_position_encoding_kernel_launcher(float* from_tensor,
                                                               const float* embedding_table,
                                                               const float* position_encoding,
                                                               const int* word_ids,
                                                               const int batch_size,
                                                               const int hidden_units,
                                                               cudaStream_t stream);

  template 
  void embedding_lookup_sine_position_encoding_kernel_launcher(half* from_tensor,
                                                               const half* embedding_table,
                                                               const half* position_encoding,
                                                               const int* word_ids,
                                                               const int batch_size,
                                                               const int hidden_units,
                                                               cudaStream_t stream);

  template 
  void embedding_position_lookups_kernel_launcher(float* from_tensor,
                                                  const float* embedding_table,
                                                  const float* pos_table,
                                                  const int* word_ids,
                                                  const int local_batch_size,
                                                  const int batch_size,
                                                  const int hidden_units,
                                                  int step,
                                                  int ite,
                                                  int max_input_len,
                                                  const int* start_lengths,
                                                  cudaStream_t stream);

  template 
  void embedding_position_lookups_kernel_launcher(half* from_tensor,
                                                  const half* embedding_table,
                                                  const half* pos_table,
                                                  const int* word_ids,
                                                  const int local_batch_size,
                                                  const int batch_size,
                                                  const int hidden_units,
                                                  int step,
                                                  int ite,
                                                  int max_input_len,
                                                  const int* start_lengths,
                                                  cudaStream_t stream);

  template
  void start_id_embedding_position_lookups_kernel_launcher(float* from_tensor,
                                                           int* output_ids,
                                                           const float* embedding_table,
                                                           const float* pos_table, 
                                                           const int* word_ids,
                                                           const int start_step,
                                                           const int length,
                                                           const int max_length,
                                                           const int batch_size,
                                                           const int hidden_units, 
                                                           cudaStream_t stream);

  template
  void start_id_embedding_position_lookups_kernel_launcher(half* from_tensor,
                                                           int* output_ids,
                                                           const half* embedding_table,
                                                           const half* pos_table, 
                                                           const int* word_ids,
                                                           const int start_step,
                                                           const int length,
                                                           const int max_length,
                                                           const int batch_size,
                                                           const int hidden_units, 
                                                           cudaStream_t stream);

  template void apply_temperature_penalty_kernelLauncher(float* logits,
                                                         const float temperature,
                                                         const int m,
                                                         const int vocab_size,
                                                         const int vocab_size_padd,
                                                         cudaStream_t stream);

  template void apply_temperature_penalty_kernelLauncher(half* logits,
                                                         const half temperature,
                                                         const int m,
                                                         const int vocab_size,
                                                         const int vocab_size_padd,
                                                         cudaStream_t stream);


  template void apply_repetition_penalty_kernelLauncher(float* logits,
                                                        const float penalty,
                                                        int* start_ids,
                                                        int* output_ids,
                                                        const int batch_size,
                                                        const int local_batch_size,
                                                        const int vocab_size,
                                                        const int vocab_size_padd,
                                                        const int* start_lengths,
                                                        const int max_input_len,
                                                        const int step,
                                                        const int ite,
                                                        cudaStream_t stream);

template void apply_repetition_penalty_kernelLauncher(half* logits,
                                                      const float penalty,
                                                      int* start_ids,
                                                      int* output_ids,
                                                      const int batch_size,
                                                      const int local_batch_size,
                                                      const int vocab_size,
                                                      const int vocab_size_padd,
                                                      const int* start_lengths,
                                                      const int max_input_len,
                                                      const int step,
                                                      const int ite,
                                                      cudaStream_t stream);


  template void kernel_padding_kernelLauncher(float *padded_kernel, const float *kernel,
                                           const int row_dim, const int col_dim,
                                           const int padded_col_dim, cudaStream_t stream);

  template void kernel_padding_kernelLauncher(half *padded_kernel, const half *kernel,
                                           const int row_dim, const int col_dim,
                                           const int padded_col_dim, cudaStream_t stream);
  
  template void bias_padding_kernelLauncher(float *padded_bias, const float *bias, const int col_dim,
                                            const int padded_col_dim, cudaStream_t stream);

  template void bias_padding_kernelLauncher(float *padded_bias, const half *bias, const int col_dim,
                                            const int padded_col_dim, cudaStream_t stream);

  template void bias_padding_kernelLauncher(half *padded_bias, const half *bias, const int col_dim,
                                            const int padded_col_dim, cudaStream_t stream);

  template void update_KV_cache_kernelLauncher(float** key_cache,
                                               float** value_cache,
                                               const int* beam_ids,
                                               const bool* finished,
                                               const int batch_size,
                                               const int beam_width,
                                               const int head_num, 
                                               const int size_per_head,
                                               const int step,
                                               const int decoder_max_seq_len,
                                               const int cache_size,
                                               const int decoder_layers,
                                               cudaStream_t stream);
  
  template void update_KV_cache_kernelLauncher(half** key_cache,
                                               half** value_cache,
                                               const int* beam_ids,
                                               const bool* finished,
                                               const int batch_size,
                                               const int beam_width,
                                               const int head_num, 
                                               const int size_per_head,
                                               const int step,
                                               const int decoder_max_seq_len,
                                               const int cache_size,
                                               const int decoder_layers,
                                               cudaStream_t stream);

  template void apply_logit_penalties(int step,
                                      float* log_probs,
                                      int* current_ids,
                                      int* previous_ids,
                                      int* parent_ids,
                                      GptArguments args,
                                      cudaStream_t stream);

  template void apply_logit_penalties(int step,
                                      half* log_probs,
                                      int* current_ids,
                                      int* previous_ids,
                                      int* parent_ids,
                                      GptArguments args,
                                      cudaStream_t stream);

  template size_t get_topp_sort_temp_storage_size(const float* log_probs,
                                                  const int* id_vals,
                                                  float* sorted_log_probs,
                                                  int* sorted_id_vals,
                                                  int* topp_offset_buf,
                                                  const int batch_size,
                                                  const int vocab_size);

  template size_t get_topp_sort_temp_storage_size(const half* log_probs,
                                                  const int* id_vals,
                                                  half* sorted_log_probs,
                                                  int* sorted_id_vals,
                                                  int* topp_offset_buf,
                                                  const int batch_size,
                                                  const int vocab_size);

  template void transpose(float *out,
                          const float *in,
                          int batch,int height,
                          int width,int stride,
                          cudaStream_t stream);
  template void transpose(half *out,
                          const half *in,
                          int batch,int height,
                          int width,int stride,
                          cudaStream_t stream);

  template void transpose_axis_01_kernelLauncher(float *out,
                                                 float *in,
                                                 const int dim0,
                                                 const int dim1,
                                                 const int dim2,
                                                 cudaStream_t stream);

  template void transpose_axis_01_kernelLauncher(half *out,
                                                 half *in,
                                                 const int dim0,
                                                 const int dim1,
                                                 const int dim2,
                                                 cudaStream_t stream);
 
  template void init_kernelLauncher(bool* finished,
                                    int* sequence_length,
                                    int* word_ids,
                                    float* cum_log_probs,
                                    const int sentence_id,
                                    const int batch_size,
                                    const int beam_width,
                                    cudaStream_t stream);

  template void init_kernelLauncher(bool* finished,
                                   int* sequence_length,
                                   int* word_ids,
                                   half* cum_log_probs,
                                   const int sentence_id,
                                   const int batch_size,
                                   const int beam_width,
                                   cudaStream_t stream);

  /* *************************** end of Instantiation *********************************** */

} // end of name space fastertransformer
