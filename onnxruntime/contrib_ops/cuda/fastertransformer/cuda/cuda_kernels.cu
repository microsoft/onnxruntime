/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>

namespace fastertransformer{

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}

template <typename T>
  __inline__ __device__
T warpReduceMax(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceMax(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();
  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)-1e20f;
  val = warpReduceMax<T>(val);

  return val;
}

template <typename T>
__global__ void update_logits_kernel(float* logits, const T* tmp_logits, const T* bias, const int end_id, const bool* finished, const int n)
{
  int bid = blockIdx.x;
  bool finish = finished[bid];
  int offset = bid * n;

  float max_val = -1 * FLT_MAX;
  __shared__ float s_max_val;
  __shared__ float s_sum_val;

  if(finish)
  {
    for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
    {
      logits[offset + tid] = (tid == end_id) ? 0 : -FLT_MAX;
    }
  }
  else
  {
    for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
    {
      if(finish)
        logits[offset + tid] = (tid == end_id) ? FLT_MAX : -1 * FLT_MAX;
      else
        logits[offset + tid] = (float)(tmp_logits[offset + tid] + bias[tid]);
      max_val = max(max_val, logits[offset + tid]);
    }

    max_val = blockReduceMax<float>((float)max_val);
    if(threadIdx.x == 0)
      s_max_val = max_val;
    __syncthreads();

    float sum_val = 0.0f;
    for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
    {
      logits[offset + tid] = __expf((float)logits[offset + tid] - s_max_val);
      sum_val += (float)logits[offset + tid];
    }

    sum_val = blockReduceSum<float>(sum_val);
    if(threadIdx.x == 0)
      s_sum_val = sum_val;
    __syncthreads();

    for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
    {
      logits[offset + tid] = logf((float)logits[offset + tid] / s_sum_val);
    }
  }
}

template <typename T>
__global__ void update_logits_kernel_without_softmax(T* logits, const T* bias, const int end_id, const bool* finished, const int n)
{
  int bid = blockIdx.x;
  bool finish = finished != nullptr ? finished[bid] : false;
  int offset = bid * n;
  
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;
  for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
  {
    if(finish)
    {
      logits[offset + tid] = (tid == end_id) ? MAX_T_VAL : -MAX_T_VAL;
    }
    else
    {
      logits[offset + tid] += bias[tid];
    }
  }
}

template <typename T>
__global__ void softmax_kernel(T* logits, const T* bias,
                               const int end_id, const bool* finished,
                               const int n_padded, const int n)
{
  int bid = blockIdx.x;
  bool finish = (finished != nullptr) ? finished[bid] : false;
  int offset = bid * n_padded;

  float max_val = -1 * FLT_MAX;
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;
  __shared__ float s_max_val;
  __shared__ float s_sum_val;

  for(int tid = threadIdx.x; tid < n_padded; tid += blockDim.x)
  {
    if(tid < n)
    {
      if(finish)
        logits[offset + tid] = (tid == end_id) ? MAX_T_VAL : -MAX_T_VAL;
      else
      {
        T bias_val = (bias != nullptr) ? bias[tid] : (T)0.0f;
        logits[offset + tid] += bias_val;
      }
    }
    else
    {
      logits[offset + tid] = -MAX_T_VAL;
    }
    max_val = max(max_val, (float)logits[offset + tid]);
  }

  max_val = blockReduceMax<float>((float)max_val);
  if(threadIdx.x == 0)
    s_max_val = max_val;
  __syncthreads();

  float sum_val = 0.0f;
  for(int tid = threadIdx.x; tid < n_padded; tid += blockDim.x)
  {
    logits[offset + tid] = __expf((float)logits[offset + tid] - s_max_val);
    sum_val += (float)logits[offset + tid];
  }

  sum_val = blockReduceSum<float>(sum_val);
  if(threadIdx.x == 0)
    s_sum_val = sum_val;
  __syncthreads();

  for(int tid = threadIdx.x; tid < n_padded; tid += blockDim.x)
  {
    logits[offset + tid] = ((float)logits[offset + tid] / s_sum_val);
  }
}

template<typename T>
__global__ void remove_sequence_length_padding(const T* src, T* tgt,
                                              const int* tmp_mask_offset,
                                              int* mask_offset,
                                              const int n)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  mask_offset[bid] = tmp_mask_offset[bid];
  const int src_seq_id = bid + mask_offset[bid];
  const int tgt_seq_id = bid;


  for(int i = tid; i < n; i += blockDim.x)
  {
    tgt[tgt_seq_id * n + i] = src[src_seq_id * n + i];
  }
}

template<typename T>
void remove_sequence_length_padding_kernelLauncher(const T* src, T* tgt, 
                                                  const int* tmp_mask_offset, 
                                                  int* mask_offset,
                                                  const int m, const int n, cudaStream_t stream)
{
  // src: [batch_size*max_seq_len, hidden_dim]
  // tgt: [valid_word_num, hidden_dim]
  remove_sequence_length_padding<<<m, 256, 0, stream>>>(src, tgt, tmp_mask_offset, mask_offset, n);
}

template<typename T>
__global__ void rebuild_sequence_length_padding(const T* src, T* tgt,
                                            const int* mask_offset,
                                            const int n)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int tgt_seq_id = bid + mask_offset[bid];
  const int src_seq_id = bid;

  for(int i = tid; i < n; i += blockDim.x)
  {
    tgt[tgt_seq_id * n + i] = src[src_seq_id * n + i];
  }
}

template<typename T>
void rebuild_sequence_length_padding_kernelLauncher(const T* src, T* tgt, 
                                                  const int* mask_offset, const int m, 
                                                  const int n, cudaStream_t stream)
{
  // src: [valid_word_num, hidden_dim]
  // tgt: [batch_size*max_seq_len, hidden_dim]
  rebuild_sequence_length_padding<<<m, 256, 0, stream>>>(src, tgt, mask_offset, n);
}

__global__ void build_sequence_length_padding_offset(const int* sequence_length, 
  const int batch_size, const int max_seq_len, int* valid_word_num, int* tmp_mask_offset)
{
  // do cumulated sum
  int total_seq_len = 0;
  int cum_offset = 0;
  int index = 0;
  for(int i = 0; i < batch_size; i++) 
  {
    const int seq_len = sequence_length[i];
    for(int j = 0; j < seq_len; j++)
    {
      tmp_mask_offset[index] = cum_offset;
      index++;
    }
    cum_offset += max_seq_len - seq_len;
    total_seq_len += seq_len;
  }
  valid_word_num[0] = total_seq_len;
}

void build_sequence_length_padding_offset_kernelLauncher(const int* sequence_length, 
  const int batch_size, const int max_seq_len, int* valid_word_num, int* tmp_mask_offset,
  cudaStream_t stream)
{
  build_sequence_length_padding_offset<<<1, 1, 0, stream>>>(sequence_length, 
    batch_size, max_seq_len, valid_word_num, tmp_mask_offset);
}

template void rebuild_sequence_length_padding_kernelLauncher(const float* src, float* tgt, 
  const int* mask_offset, const int m, 
  const int n, cudaStream_t stream);


template void rebuild_sequence_length_padding_kernelLauncher(const half* src, half* tgt, 
  const int* mask_offset, const int m, 
  const int n, cudaStream_t stream);

template void remove_sequence_length_padding_kernelLauncher(const float* src, float* tgt, 
  const int* tmp_mask_offset, 
  int* mask_offset, const int m, 
  const int n, cudaStream_t stream);

template void remove_sequence_length_padding_kernelLauncher(const half* src, half* tgt, 
  const int* tmp_mask_offset, 
  int* mask_offset, const int m, 
  const int n, cudaStream_t stream);

///template <typename T>
///__global__ void cuda_random_uniform_kernel(T* buffer, const int size)
///{
///  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
///  curandState_t local_state;
///  curand_init((T)1337.f, idx, 0, &local_state);
///  for(int index = idx; index < size; index += blockDim.x * gridDim.x)
///  {
///    buffer[index] = (T)(curand_uniform(&local_state) * 0.2f - 0.1f);
///  }
///}

///template <typename T>
///void cuda_random_uniform_kernelLauncher(T *buffer, const int size)
///{
///  cuda_random_uniform_kernel<<<256, 256>>>(buffer, size);
///}

///template void cuda_random_uniform_kernelLauncher(float *buffer, const int size);
///template void cuda_random_uniform_kernelLauncher(half *buffer, const int size);

template <typename T>
void update_logits(float* logits, const T* tmp_logits, const T* bias, const int end_id, const bool* finished, 
  const int m, const int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));
  /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
  update_logits_kernel<<<grid, block, 0, stream>>>(logits, tmp_logits, bias, end_id, finished, n);
}

template void update_logits(float* logits, const float* tmp_logits, const float* bias, const int end_id,
  const bool* finished, const int m, const int n, cudaStream_t stream);

template void update_logits(float* logits, const half* tmp_logits, const half* bias, const int end_id,
  const bool* finished, const int m, const int n, cudaStream_t stream);

template<typename T>
void update_logits_without_softmax(T* logits, const T* bias, const int end_id, const bool* finished, 
  const int m, const int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));
  /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
  update_logits_kernel_without_softmax<<<grid, block, 0, stream>>>(logits, bias, end_id, finished, n);
}

template void update_logits_without_softmax(float* logits, const float* bias, const int end_id, const bool* finished, 
  const int m, const int n, cudaStream_t stream);

template void update_logits_without_softmax(half* logits, const half* bias, const int end_id, const bool* finished, 
  const int m, const int n, cudaStream_t stream);
  
template<typename T>
void softmax_kernelLauncher(T* logits, const T* bias, const int end_id, const bool* finished,
                            const int m, const int n_padded, const int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));
  /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
  softmax_kernel<<<grid, block, 0, stream>>>(logits, bias, end_id, finished, n_padded, n);
}

template void softmax_kernelLauncher(float* logits, const float* bias, const int end_id, const bool* finished,
                                     const int m, const int n_padded, const int n, cudaStream_t stream);

template void softmax_kernelLauncher(half* logits, const half* bias, const int end_id, const bool* finished,
                                     const int m, const int n_padded, const int n, cudaStream_t stream);

/* *********************************** Debug tools *********************************** */

template <typename T>
__global__
void print_abs_mean_kernel(const T* buf, uint size)
{
  float sum;
  for(int i = 0; i < size; i++)
  {
    sum += abs((float)buf[i]);
    // printf("[INFO] buf[%d] %f \n", i, buf[i]);
  }
  printf("mean: %f \n", (float) sum / (float) size);
  printf("sum: %f \n", sum);
}

template <typename T>
__global__
void print_kernel(const T* buf, uint size)
{
  for(int i = 0; i < size; i++)
  {
    printf("%f ", (float(buf[i])));
  }
  printf("\n");
}

template <>
__global__
void print_kernel(const int* buf, uint size)
{
  for(int i = 0; i < size; i++)
  {
    printf("%d ", buf[i]);
  }
  printf("\n");
}

template <typename T>
void print_first_k(const T* buf, uint size, cudaStream_t stream)
{
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
  print_kernel<<<1, 1, 0, stream>>>(buf, size);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
}

template <typename T>
void print_abs_mean(const T* buf, uint size, cudaStream_t stream)
{
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
  print_abs_mean_kernel<<<1, 1, 0, stream>>>(buf, size);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
}

template void print_first_k(const float*, uint size, cudaStream_t);
template void print_first_k(const half*, uint size, cudaStream_t);
template void print_first_k(const int*, uint size, cudaStream_t);
template void print_first_k(const bool*, uint size, cudaStream_t);

template void print_abs_mean(const float* buf, uint size, cudaStream_t stream);
template void print_abs_mean(const half* buf, uint size, cudaStream_t stream);
template void print_abs_mean(const int* buf, uint size, cudaStream_t stream);

/* **************************** end of Debug tools *********************************** */

// TODO remove in v4.1
/* *************************** depreciated kernels *********************************** */

template <typename T>
__global__
void topK_kernel(const T* log_probs, int* ids, const int batch_size, const int N, const int K)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val, max_val;
  __shared__ float s_max_val;
  for(int ite = 0; ite < batch_size; ++ite)
  {
    bool choosed = false;
    val = (tid < N ) ? (float)log_probs[ite * N + tid] : -1e20f;

    for(int kids = 0; kids < K; ++kids)
    {
      max_val = blockReduceMax<float>(val);

      if(threadIdx.x == 0)
        s_max_val = max_val;
      __syncthreads();

      if(s_max_val == val && !choosed && tid < N) 
      {
        ids[ite * gridDim.x * K + blockIdx.x * K + kids] = tid + ite * N;
        val = -1e20f;
        choosed = true;
      }
    }
  }
}

template <typename T>
__global__
void topK_kernel_2nd(const T* log_probs, int* ids, const int batch_size, const int N, const int K, const int id_offset)
{
  int tid = threadIdx.x;
  float val, max_val;
  __shared__ float s_max_val;
  __shared__ int beam_index;
  __shared__ int ids_before_sort[16];

  for(int ite = 0; ite < batch_size; ++ite)
  {
    bool choosed = false;
    const int id = (tid < N) ? ids[ite * N + tid] : -1;
    val = (tid < N) ? (float)log_probs[id] : -1e20f;

    __syncthreads();

    if(tid == 0) beam_index = 0;
    if(tid < 16) ids_before_sort[tid] = -1;
    
    __syncthreads();
    while(beam_index < K){
      int begin_beam_index = beam_index;
      max_val = blockReduceMax<float>(val);
      if(threadIdx.x == 0){
        s_max_val = max_val;
      }
      __syncthreads();
      if(s_max_val == val && !choosed && id != -1)
      {
        int id_offset_ = atomicAdd(&beam_index, 1);
        ids_before_sort[id_offset_] = id;
        val = -1e20f;
        choosed = true;
      }
      __syncthreads();

      // simply sort the ids
      if(threadIdx.x == 0 && beam_index - begin_beam_index > 1){
        for(int i = begin_beam_index; i < beam_index; i++){
          for(int j = i; j < beam_index; j++){
            if(ids_before_sort[j] < ids_before_sort[i]){
              int tmpid = ids_before_sort[j];
              ids_before_sort[j] = ids_before_sort[i];
              ids_before_sort[i] = tmpid;
            }
          }
        }
      }
    }
    __syncthreads();
    if(tid < K) ids[ite * K + tid] = ids_before_sort[tid];
    __syncthreads();
  }
}

void topK(const float* log_probs, int* ids, const int batch_size, const int beam_width, const int vocab_size,
  cudaStream_t stream)
{
  int N = beam_width * vocab_size;
  dim3 block(1024);
  dim3 grid((N - 1) / block.x + 1);
  /* First round topK, for each batch, get grid.x * K values */
  topK_kernel<float><<<grid, block, 0, stream>>>(log_probs, ids, batch_size, N, beam_width);
  /*Second round, for each batch, get the final TopK values out from grid.x * K values. */
  topK_kernel_2nd<float><<<1, block, 0, stream>>>(log_probs, ids, batch_size, beam_width * grid.x, beam_width, N);
}

template <typename T>
__global__ void embedding_lookup_kernel(const T* embedding_table, const int* word_ids,
    const int hidden_units, T* from_tensor)
{
  int write_pos = threadIdx.x + blockIdx.x * hidden_units;
  from_tensor[write_pos] = embedding_table[word_ids[blockIdx.x] * hidden_units + threadIdx.x];
}

template <typename T>
void embedding_lookup(const T* embedding_table, const int* word_ids, T* from_tensor,
  const int batch_size, const int beam_width, const int hidden_units, cudaStream_t stream)
{
  dim3 grid(batch_size * beam_width);
  dim3 block(hidden_units);
  assert(hidden_units <= 1024);
  embedding_lookup_kernel<<<grid, block, 0, stream>>>(embedding_table, word_ids, hidden_units, from_tensor);
}

template<typename T>
__global__
void sine_position_encoder_kernel(T* output, int step, int n){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  float half_n = (float)n / 2.;

  // input = input * hidden_dim**0.5
  output[bid * n + tid] = output[bid * n + tid] * (T)sqrtf(float(n));

  float log_timescale_increment = __logf(10000) / (half_n - 1.f);
  float inv_timescales = __expf( (tid % (int)half_n) * -1 * log_timescale_increment );
  float scaled_time = inv_timescales * step;
  
  T encoding_val = (tid < half_n) ? (T) __sinf(scaled_time) : (T) __cosf(scaled_time);
  output[bid * n + tid] = output[bid * n + tid]  + encoding_val;
}

template<typename T>
void sine_position_encoder(
  T* output,
  int step,
  int m, int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  sine_position_encoder_kernel<T><<<grid, block, 0, stream>>>(output, step, n);
}

template void embedding_lookup(const float* embedding_table, const int* word_ids, float* from_tensor,
  const int batch_size, const int beam_width, const int hidden_units, cudaStream_t stream);

template void embedding_lookup(const half* embedding_table, const int* word_ids, half* from_tensor,
  const int batch_size, const int beam_width, const int hidden_units, cudaStream_t stream);

template void sine_position_encoder(
  float* output,
  int step,
  int m, int n,
  cudaStream_t stream);

template void sine_position_encoder(
  half* output,
  int step,
  int m, int n,
  cudaStream_t stream);

/* *************************** end of depreciated kernels *********************************** */

}//namespace 
