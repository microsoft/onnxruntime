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
/**
 * Open sourced multi-head attention
 **/
#include <type_traits>
#include <stdint.h>

#include "contrib_ops/cuda/fastertransformer/open_decoder.h"
#include "cub/cub.cuh"
#include "contrib_ops/cuda/fastertransformer/utils/nvtx_utils.h"
#include "masked_multihead_attention.h"

namespace fastertransformer{

const int WARP_SIZE = 32;
const bool ATTENION_OPT = true;
const int ATTENTION_BLOCK_SIZE = 256;

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int HALF_ELEMENTS_PER_WARP_LOAD>
using Copy_half_t =
    typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 32, half,
        typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 64, int,
            typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 128, int2, int4
            >::type
        >::type
    >::type;

template <typename T, int ELEMENTS_PER_WARP_LOAD>
using Copy_t = Copy_half_t<sizeof(T) / sizeof(half) * ELEMENTS_PER_WARP_LOAD>;

///////////////////////////////////////////////////////////////////////////////////////////////////

/**
  masked multi-head attention
*/
#define FINAL_MASK 0xffffffff
template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}
/* Calculate the sum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32]; 
  // __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; 
  int wid = threadIdx.x >> 5;  

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
                              
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
//  __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)-1e20f;
  val = warpReduceMax(val);

  return val;
}

template <int size_per_head, int block_sz, typename T>
__global__ 
void masked_attention_kernel_opt(
  T* __restrict key_buf, T* __restrict value_buf,
  T* __restrict query_buf, const T* __restrict self_Q_bias, 
  T* __restrict key_cache, const T* __restrict self_K_bias, 
  T* __restrict value_cache, const T* __restrict self_V_bias,
  T* __restrict context_buf, const bool* finished, 
  int batch_size, int head_num, const int step, const T scalar)
{
  if(finished != nullptr && finished[blockIdx.x / head_num] == true) return;
  typedef Copy_t<T, size_per_head> copy_t;
  const int elems_per_thread = size_per_head / WARP_SIZE;

  union Access_t
  {
    copy_t v;
    T x[elems_per_thread]; // supported size 1,2,4
  };
  typedef struct Float_n_t
  {
    T x[elems_per_thread]; // supported size 1,2,4
  } float_n_t;

  __shared__ float_n_t sq[block_sz];

  extern __shared__ float logits[]; // use to store the logits from [0~step]

  const int tid = threadIdx.x;
  const int warp_num = block_sz / WARP_SIZE;
  const int bid = blockIdx.x;
  const int head_id = blockIdx.x % head_num;
  const int warp_id = tid / WARP_SIZE; // warp_id in block
  const int lane_id = tid % WARP_SIZE; // lane_id in warp

  typedef cub::BlockReduce<float, block_sz> MaxValBlockReduce;
  typedef cub::BlockReduce<float, block_sz> BlockReduce;
  __shared__ typename MaxValBlockReduce::TempStorage max_val_block_temp_storage;
  __shared__ typename BlockReduce::TempStorage block_temp_storage;
  __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage[warp_num];

  int qkv_id = bid * size_per_head;
  int qkv_bias_id = head_id * size_per_head;

  query_buf = &query_buf[qkv_id];
  key_buf = &key_buf[qkv_id];
  value_buf = &value_buf[qkv_id];
  self_K_bias = &self_K_bias[qkv_bias_id];
  key_cache = &key_cache[qkv_id];
  self_Q_bias = &self_Q_bias[qkv_bias_id];
  self_V_bias = &self_V_bias[qkv_bias_id];
  value_cache = &value_cache[qkv_id];
  context_buf = &context_buf[qkv_id];

  Access_t bias_r, query_buf_r;
  Access_t key_val_r, key_buf_r;
  Access_t value_val_r, value_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((copy_t *)query_buf + lane_id);
  key_buf_r.v = *((copy_t *)key_buf + lane_id);
  bias_r.v = *((copy_t *)self_Q_bias + lane_id);
  float qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i)
  {
    qb_r[i] =  (float)query_buf_r.x[i] + (float)bias_r.x[i];
  }

  //offset for each step
  int offset = batch_size * head_num * size_per_head;
  bias_r.v = *((copy_t *) self_K_bias + lane_id);
  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&key_cache[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = (float)key_buf_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&key_cache[ite * offset] + lane_id) = key_val_r.v;
    }
    float val = 0.f;
    for (int i = 0; i < elems_per_thread; i++)
    {
      val = val +  (float)key_val_r.x[i] * qb_r[i] * (float)scalar;
    }
    float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
    if (lane_id == 0)
    {
      logits[ite] = qk; 
    }
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;

  float local_i = -1e20f;
  for(int i = tid; i < step; i += blockDim.x)
    local_i = max(local_i, logits[i]);

  float max_val = MaxValBlockReduce(max_val_block_temp_storage).Reduce(local_i, cub::Max());
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();


  float local_o = 0.0f;
  for(int i = tid; i < step; i += blockDim.x)
  {
    logits[i] = __expf(logits[i] - s_max_val);
    local_o += logits[i];
  }
  float val = BlockReduce(block_temp_storage).Sum(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  float s_sum_inverse = __fdividef(1.0f, s_sum);
  for(int i = tid; i < step; i += blockDim.x)
  {
    logits[i] = logits[i] * s_sum_inverse;
  }
  __syncthreads(); 

  // This optimization introduces discrepancy because of different order in FP32 summation
  float sum_r[elems_per_thread] = {0.f};
  bias_r.v = *((copy_t *) self_V_bias + lane_id);
  value_buf_r.v = *((copy_t *)value_buf + lane_id);

  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    value_val_r.v = *((copy_t *)&value_cache[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        value_val_r.x[i] = (float)value_buf_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&value_cache[ite * offset] + lane_id) = value_val_r.v;
    }
    for (int i = 0; i < elems_per_thread; ++i)
    {
      sum_r[i] += (float)value_val_r.x[i] * logits[ite]; 
    }
  }
  for (int i = 0; i < elems_per_thread; i++)
  {
    sq[warp_id * WARP_SIZE + lane_id].x[i] = sum_r[i];
  }
  __syncthreads();
  if (warp_id == 0)
  {
    #pragma unroll
    for (int j = 1; j < warp_num; j++)
    {
      for (int i = 0; i < elems_per_thread; ++i)
      {
        sum_r[i] = sum_r[i] + (float)sq[j * WARP_SIZE + tid].x[i];
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++)
  {
    value_val_r.x[i] = sum_r[i];
  }
  if (warp_id == 0)
  {
    *((copy_t *)context_buf + lane_id) = value_val_r.v;
  }
}

template <typename T>
__global__ 
void masked_attention_kernel(
  T* key_buf, T* value_buf,
  T* query_buf, const T* self_Q_bias, 
  T* key_cache, const T* self_K_bias, T* value_cache, const T* self_V_bias,
  T* context_buf, const bool* finished,
  int batch_size, int head_num, int size_per_head, const int step, const T scalar)
{
  if(finished != nullptr && finished[blockIdx.x / head_num] == true) return;
  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  int qkv_bias_id = head_id * size_per_head + tid;

  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id] + self_Q_bias[qkv_bias_id];
  __syncthreads();

  //offset for each step
  int offset = batch_size * head_num * size_per_head;
  for(int ite = 0; ite < step; ++ite)
  {
    T key = tid < size_per_head ? key_cache[ite * offset + qkv_id] : (T)0.0f;
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1 && tid < size_per_head)
    {
      key = key_buf[qkv_id] + self_K_bias[qkv_bias_id];
      key_cache[ite * offset + qkv_id] = key; 
    }
    
    T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
    T qk = blockReduceSum(val);
    if(threadIdx.x == 0)
      logits[ite] = qk;
    __syncthreads(); //try to remove
  }
  __syncthreads(); //try to remove

  __shared__ float s_max_val, s_sum;
  float local_i = tid < step ? (float)logits[tid] : -1e20f; 
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < step ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  if(tid < step)
    logits[tid] = local_o / s_sum;
  __syncthreads();

  if(tid < size_per_head)
  {
    T sum = (T)0.0f;
    for(int ite = 0; ite < step; ++ite)
    {
      T value = value_cache[ite * offset + qkv_id];
      //for the last step, we should update K + bias_K to the cache
      if(ite == step - 1)
      {
        value = value_buf[qkv_id] + self_V_bias[qkv_bias_id];
        value_cache[ite * offset + qkv_id] = value;
      }
      sum += value * logits[ite];
    }
    context_buf[qkv_id] = sum;
  }
}

template <typename T>
void masked_attention_dispatch(
  T* key_buf, T* value_buf,
  T* query_buf, const T* self_Q_bias, 
  T* key_cache, const T* self_K_bias, T* value_cache, const T* self_V_bias,
  T* context_buf, const bool* finished, int max_batch_size, int inference_batch_size,
  int head_num, int size_per_head, const int step, const int max_seq_len, cudaStream_t stream)
{
  if (max_seq_len < 0) {
    const int block_sz = ATTENTION_BLOCK_SIZE;
    T scalar = (T)(1.f / sqrtf(size_per_head * 1.0f));
  
    dim3 grid(inference_batch_size * head_num);
  
    int cond = size_per_head * ((ATTENION_OPT)? 1:0);
    switch (cond)
    {
      case 32:
        masked_attention_kernel_opt<32, block_sz, T><<<grid, block_sz, sizeof(float)*step, stream>>>(
          key_buf, value_buf,
          query_buf, self_Q_bias,  key_cache, self_K_bias, value_cache, self_V_bias, context_buf, finished,
          max_batch_size, head_num, step, scalar);
        break;
      case 64:
          masked_attention_kernel_opt<64, block_sz, T><<<grid, block_sz, sizeof(float)*step, stream>>>(
            key_buf, value_buf,
            query_buf, self_Q_bias,
            key_cache, self_K_bias,
            value_cache, self_V_bias,
            context_buf, 
            finished,
            max_batch_size, head_num, step, scalar);
        break;
      case 128:
          masked_attention_kernel_opt<128, block_sz, T><<<grid, block_sz, sizeof(float)*step, stream>>>(
            key_buf, value_buf,
            query_buf, self_Q_bias,  key_cache, self_K_bias, value_cache, self_V_bias, context_buf, finished,
            max_batch_size, head_num, step, scalar);
        break;
      default:
        // default path
        int block_size = 128;
        
        //suppose size_per_head <= 128
        if(step <= 64)
          block_size = 64;
        else if(step <= 128 && step > size_per_head)
          block_size = 128;
        else if(step > 128 && step <= 256)
          block_size = 256;
        else if(step > 256 && step <= 512)
          block_size = 512;
        else
          block_size = 1024;
        
        if((int)block_size < size_per_head)
          block_size = size_per_head;
          
        assert(block_size <= 1024);
        dim3 block(block_size);
        T scalar = 1 / sqrtf(size_per_head * 1.0f);
  
        
        int shared_size = sizeof(T) * (size_per_head + step);
        masked_attention_kernel<T><<<grid, block, shared_size, stream>>>(
          key_buf, value_buf,
          query_buf, self_Q_bias,
          key_cache, self_K_bias,
          value_cache, self_V_bias,
          context_buf, finished, max_batch_size,
          head_num, size_per_head, step, scalar);
    }
  }
  else {
    assert(step > 0);
    assert(size_per_head == 32 || size_per_head == 64 || size_per_head == 128);
    using DataType = typename std::conditional<sizeof(T) == 4, float, uint16_t>::type;
    // Prepare the parameters.
    Masked_multihead_attention_params<DataType> params;
    memset(&params, 0, sizeof(params));
    params.q_bias = reinterpret_cast<const DataType *>(self_Q_bias);
    params.k_bias = reinterpret_cast<const DataType *>(self_K_bias);
    params.v_bias = reinterpret_cast<const DataType *>(self_V_bias);
  
    // Set the output buffer.
    params.out = reinterpret_cast<DataType *>(context_buf);
  
    // Set the input buffers.
    params.q = reinterpret_cast<const DataType *>(query_buf);
    params.k = reinterpret_cast<const DataType *>(key_buf);
    params.v = reinterpret_cast<const DataType *>(value_buf);
    params.stride = 0;
    params.finished = const_cast<bool*>(finished);
  
    params.k_cache = reinterpret_cast<DataType *>(key_cache);
    params.v_cache = reinterpret_cast<DataType *>(value_cache);
    params.batch_size = inference_batch_size;
    params.seq_length = max_seq_len;
    params.timestep = step-1;
    params.num_heads = head_num;
    params.hidden_size_per_head = size_per_head;
    params.inv_sqrt_dh = 1.F / sqrtf((float) params.hidden_size_per_head);

    params.is_mask = false;

    masked_multihead_attention(params, stream);
  }
}

template void masked_attention_dispatch(
  float* key_buf, 
  float* value_buf,
  float* query_buf, 
  const float* self_Q_bias, 
  float* key_cache, 
  const float* self_K_bias, 
  float* value_cache, 
  const float* self_V_bias,
  float* context_buf, 
  const bool* finished, 
  int max_batch_size, 
  int inference_batch_size, 
  int head_num, 
  int size_per_head, 
  const int step,
  const int max_seq_size,
  cudaStream_t stream);

template void masked_attention_dispatch(
  half* key_buf, 
  half* value_buf,
  half* query_buf, 
  const half* self_Q_bias, 
  half* key_cache, 
  const half* self_K_bias, 
  half* value_cache, 
  const half* self_V_bias,
  half* context_buf, 
  const bool* finished, 
  int max_batch_size, 
  int inference_batch_size, 
  int head_num, 
  int size_per_head, 
  const int step,
  const int max_seq_size,
  cudaStream_t stream);

template <int size_per_head, int block_sz, typename T>
__global__ 
void fusedQKV_masked_attention_kernel_opt(
  const T* __restrict qkv_buf, const T* __restrict qkv_bias,
  T* __restrict key_cache,
  T* __restrict value_cache,
  T* __restrict context_buf, const bool* finished, int batch_size, int head_num, const int step, const T scalar)
{
  if(finished != nullptr && finished[blockIdx.x / head_num] == true) return;
  typedef Copy_t<T, size_per_head> copy_t;
  const int elems_per_thread = size_per_head / WARP_SIZE;

  union Access_t
  {
    copy_t v;
    T x[elems_per_thread]; // supported size 1,2,4
  };
  typedef struct Float_n_t
  {
    T x[elems_per_thread]; // supported size 1,2,4
  } float_n_t;

  __shared__ float_n_t sq[block_sz];

  extern __shared__ float logits[]; // use to store the logits from [0~step]

  const int tid = threadIdx.x;
  const int warp_num = block_sz / WARP_SIZE;
  const int bid = blockIdx.x;
  const int head_id = blockIdx.x % head_num;
  const int warp_id = tid / WARP_SIZE; // warp_id in block
  const int lane_id = tid % WARP_SIZE; // lane_id in warp
  const int batch_id = bid / head_num;
  const int hidden_units = head_num * size_per_head;

  typedef cub::BlockReduce<float, block_sz> MaxValBlockReduce;
  typedef cub::BlockReduce<float, block_sz> BlockReduce;
  __shared__ typename MaxValBlockReduce::TempStorage max_val_block_temp_storage;
  __shared__ typename BlockReduce::TempStorage block_temp_storage;
  __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage[warp_num];

  int qkv_id = batch_id * 3 * hidden_units + head_id * size_per_head;
  int qkv_bias_id = head_id * size_per_head;
  int cache_qkv_id = bid * size_per_head;
  
  const T* query_buf = qkv_buf + qkv_id;
  const T* key_buf = qkv_buf + hidden_units + qkv_id;
  const T* value_buf = qkv_buf + 2 * hidden_units + qkv_id;
  const T* self_Q_bias = qkv_bias + qkv_bias_id;
  const T* self_K_bias = qkv_bias + hidden_units + qkv_bias_id;
  const T* self_V_bias = qkv_bias + 2 * hidden_units + qkv_bias_id;
  value_cache = value_cache + cache_qkv_id;
  key_cache = key_cache + cache_qkv_id;
  context_buf = context_buf + cache_qkv_id;

  Access_t bias_r, query_buf_r;
  Access_t key_val_r, key_buf_r;
  Access_t value_val_r, value_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((copy_t *)query_buf + lane_id);
  key_buf_r.v = *((copy_t *)key_buf + lane_id);
  bias_r.v = *((copy_t *)self_Q_bias + lane_id);
  float qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i)
  {
    qb_r[i] =  (float)query_buf_r.x[i] + (float)bias_r.x[i];
  }

  //offset for each step
  int offset = batch_size * hidden_units;
  bias_r.v = *((copy_t *) self_K_bias + lane_id);
  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&key_cache[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = (float)key_buf_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&key_cache[ite * offset] + lane_id) = key_val_r.v;
    }
    float val = 0.f;
    for (int i = 0; i < elems_per_thread; i++)
    {
      val = val +  (float)key_val_r.x[i] * qb_r[i] * (float)scalar;
    }
    float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
    if (lane_id == 0)
    {
      logits[ite] = qk; 
    }
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;

  float local_i = -1e20f;
  for(int i = tid; i < step; i += blockDim.x)
    local_i = max(local_i, logits[i]);

  float max_val = MaxValBlockReduce(max_val_block_temp_storage).Reduce(local_i, cub::Max());
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();


  float local_o = 0.0f;
  for(int i = tid; i < step; i += blockDim.x)
  {
    logits[i] = __expf(logits[i] - s_max_val);
    local_o += logits[i];
  }
  float val = BlockReduce(block_temp_storage).Sum(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  float s_sum_inverse = __fdividef(1.0f, s_sum);
  for(int i = tid; i < step; i += blockDim.x)
  {
    logits[i] = logits[i] * s_sum_inverse;
  }
  __syncthreads(); 

  // This optimization introduces discrepancy because of different order in FP32 summation
  float sum_r[elems_per_thread] = {0.f};
  bias_r.v = *((copy_t *) self_V_bias + lane_id);
  value_buf_r.v = *((copy_t *)value_buf + lane_id);

  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    value_val_r.v = *((copy_t *)&value_cache[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        value_val_r.x[i] = (float)value_buf_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&value_cache[ite * offset] + lane_id) = value_val_r.v;
    }
    for (int i = 0; i < elems_per_thread; ++i)
    {
      sum_r[i] += (float)value_val_r.x[i] * logits[ite]; 
    }
  }
  for (int i = 0; i < elems_per_thread; i++)
  {
    sq[warp_id * WARP_SIZE + lane_id].x[i] = sum_r[i];
  }
  __syncthreads();
  if (warp_id == 0)
  {
    #pragma unroll
    for (int j = 1; j < warp_num; j++)
    {
      for (int i = 0; i < elems_per_thread; ++i)
      {
        sum_r[i] = sum_r[i] + (float)sq[j * WARP_SIZE + tid].x[i];
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++)
  {
    value_val_r.x[i] = sum_r[i];
  }
  if (warp_id == 0)
  {
    *((copy_t *)context_buf + lane_id) = value_val_r.v;
  }
}

template <typename T>
void fusedQKV_masked_attention_dispatch(
  const T* qkv_buf, const T* qkv_bias,
  T* key_cache, T* value_cache,
  T* context_buf, const bool* finished, int max_batch_size, int inference_batch_size, 
  int head_num, int size_per_head, const int step, const int max_seq_len, cudaStream_t stream)
{
  if (max_seq_len < 0) {
    const int block_sz = ATTENTION_BLOCK_SIZE;
    T scalar = (T)(1.f / sqrtf(size_per_head * 1.0f));
  
    dim3 grid(inference_batch_size * head_num);
  
    int cond = size_per_head * ((ATTENION_OPT)? 1:0);
    switch (cond)
    {
      case 32:
        fusedQKV_masked_attention_kernel_opt<32, block_sz, T><<<grid, block_sz, sizeof(float)*step, stream>>>(
          qkv_buf, qkv_bias,
          key_cache, value_cache,
          context_buf,
          finished,
          max_batch_size, head_num, step, scalar);
        break;
      case 64:
        fusedQKV_masked_attention_kernel_opt<64, block_sz, T><<<grid, block_sz, sizeof(float)*step, stream>>>(
          qkv_buf, qkv_bias,
          key_cache,
          value_cache,
          context_buf,
          finished,
          max_batch_size, head_num, step, scalar);
        break;
      case 128:
        fusedQKV_masked_attention_kernel_opt<128, block_sz, T><<<grid, block_sz, sizeof(float)*step, stream>>>(
          qkv_buf, qkv_bias,
          key_cache,
          value_cache,
          context_buf,
          finished,
          max_batch_size, head_num, step, scalar);
        break;
      default:
        assert(false);
    }
  }
  else {
    using DataType = typename std::conditional<sizeof(T) == 4, float, uint16_t>::type;
    // Prepare the parameters.
    Masked_multihead_attention_params<DataType> params;
    memset(&params, 0, sizeof(params));
    int hidden_units = head_num * size_per_head;
    params.q_bias = reinterpret_cast<const DataType *>(qkv_bias);
    params.k_bias = reinterpret_cast<const DataType *>(qkv_bias) + hidden_units;
    params.v_bias = reinterpret_cast<const DataType *>(qkv_bias) + 2 * hidden_units;
  
    // Set the output buffer.
    params.out = reinterpret_cast<DataType *>(context_buf);
  
    // Set the input buffers.
    params.q = reinterpret_cast<const DataType *>(qkv_buf);
    params.k = reinterpret_cast<const DataType *>(qkv_buf) + hidden_units;
    params.v = reinterpret_cast<const DataType *>(qkv_buf) + 2 * hidden_units;
    params.stride = 3 * hidden_units;
    params.finished = const_cast<bool*>(finished);
  
    params.k_cache = reinterpret_cast<DataType *>(key_cache);
    params.v_cache = reinterpret_cast<DataType *>(value_cache);
    params.batch_size = inference_batch_size;
    params.seq_length = max_seq_len;
    params.timestep = step-1;
    params.num_heads = head_num;
    params.hidden_size_per_head = size_per_head;
    params.inv_sqrt_dh = 1.F / sqrtf((float) params.hidden_size_per_head);

    params.is_mask = false;

    masked_multihead_attention(params, stream);
  }
}

template void fusedQKV_masked_attention_dispatch(
  const float* qkv_buf, 
  const float* qkv_bias,
  float* key_cache, 
  float* value_cache,
  float* context_buf, 
  const bool* finished, 
  int max_batch_size, 
  int inference_batch_size, 
  int head_num, 
  int size_per_head, 
  const int step, 
  const int max_seq_len,
  cudaStream_t stream);
  
template void fusedQKV_masked_attention_dispatch(
  const half* qkv_buf, 
  const half* qkv_bias,
  half* key_cache, 
  half* value_cache,
  half* context_buf, 
  const bool* finished, 
  int max_batch_size, 
  int inference_batch_size, 
  int head_num, 
  int size_per_head,
  const int step, 
  const int max_seq_len,
  cudaStream_t stream);

template <typename T>
void fusedQKV_masked_attention_kernelLauncher(
    const T* qkv_buf,
    const T* qkv_bias,
    T* k_cache,
    T* v_cache,
    T* output,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    const int max_seq_len,
    cudaStream_t stream)
{
  fusedQKV_masked_attention_dispatch(qkv_buf,
                                      qkv_bias,
                                      k_cache,
                                      v_cache,
                                      output,
                                      nullptr,
                                      batch_size,
                                      batch_size,
                                      head_num,
                                      size_per_head,
                                      seq_len,
                                      max_seq_len,
                                      stream);
}

template <typename T>
void fusedQKV_masked_attention_dispatch_v2(
  const T* qkv_buf, const T* qkv_bias,
  T* key_cache, T* value_cache,
  T* context_buf, const bool* finished, int max_batch_size, int inference_batch_size, 
  int head_num, int size_per_head, const int step, const int max_seq_len, 
  const int max_input_len, const int* input_lengths, cudaStream_t stream)
{
  using DataType = typename std::conditional<sizeof(T) == 4, float, uint16_t>::type;
  // Prepare the parameters.
  Masked_multihead_attention_params<DataType> params;
  memset(&params, 0, sizeof(params));
  int hidden_units = head_num * size_per_head;
  params.q_bias = reinterpret_cast<const DataType *>(qkv_bias);
  params.k_bias = reinterpret_cast<const DataType *>(qkv_bias) + hidden_units;
  params.v_bias = reinterpret_cast<const DataType *>(qkv_bias) + 2 * hidden_units;

  // Set the output buffer.
  params.out = reinterpret_cast<DataType *>(context_buf);

  // Set the input buffers.
  params.q = reinterpret_cast<const DataType *>(qkv_buf);
  params.k = reinterpret_cast<const DataType *>(qkv_buf) + hidden_units;
  params.v = reinterpret_cast<const DataType *>(qkv_buf) + 2 * hidden_units;
  params.stride = 3 * hidden_units;
  params.finished = const_cast<bool*>(finished);

  params.k_cache = reinterpret_cast<DataType *>(key_cache);
  params.v_cache = reinterpret_cast<DataType *>(value_cache);
  params.batch_size = inference_batch_size;
  params.seq_length = max_seq_len;
  params.timestep = step-1;
  params.num_heads = head_num;
  params.hidden_size_per_head = size_per_head;
  params.inv_sqrt_dh = 1.F / sqrtf((float) params.hidden_size_per_head);

  params.is_mask = true;
  params.input_lengths = input_lengths;
  params.max_input_len = max_input_len;

  masked_multihead_attention(params, stream);
}

template void fusedQKV_masked_attention_dispatch_v2(
  const float* qkv_buf, 
  const float* qkv_bias,
  float* key_cache, 
  float* value_cache,
  float* context_buf, 
  const bool* finished, 
  int max_batch_size, 
  int inference_batch_size, 
  int head_num, 
  int size_per_head, 
  const int step, 
  const int max_seq_len,
  const int max_input_len, 
  const int* input_lengths,
  cudaStream_t stream);
  
template void fusedQKV_masked_attention_dispatch_v2(
  const half* qkv_buf, 
  const half* qkv_bias,
  half* key_cache, 
  half* value_cache,
  half* context_buf, 
  const bool* finished, 
  int max_batch_size, 
  int inference_batch_size, 
  int head_num, 
  int size_per_head,
  const int step, 
  const int max_seq_len,
  const int max_input_len, 
  const int* input_lengths,
  cudaStream_t stream);

template <typename T>
void fusedQKV_masked_attention_kernelLauncher_v2(
    const T* qkv_buf,
    const T* qkv_bias,
    T* k_cache,
    T* v_cache,
    T* output,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    const int max_seq_len,
    const int max_input_len,
    const int* input_lengths,
    cudaStream_t stream)
{
  fusedQKV_masked_attention_dispatch_v2(qkv_buf,
                                        qkv_bias,
                                        k_cache,
                                        v_cache,
                                        output,
                                        nullptr,
                                        batch_size,
                                        batch_size,
                                        head_num,
                                        size_per_head,
                                        seq_len,
                                        max_seq_len,
                                        max_input_len,
                                        input_lengths,                                      
                                        stream);
}

template<typename T>
__global__ void transpose_4d(T* dst, T* src,
                            const int dim0,
                            const int dim1,
                            const int dim2,
                            const int dim3,
                            const int dim0_leading_dim,
                            const int ite)
{
  // transpose from [dim0, dim1, dim2, dim3] to [dim2, X, dim1, dim3]
  // where the dimension of X is dim0_leading_dim, and offset is ite * dim0
  for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < dim0 * dim1 * dim2 * dim3; i+= blockDim.x * gridDim.x)
  {
    int index = i;
    const int d3 = index % dim3;
    index = (index - d3) / dim3;
    const int d2 = index % dim2;
    index = (index - d2) / dim2;
    const int d1 = index % dim1;
    index = (index - d1) / dim1;
    const int d0 = index % dim0;
    index = (index - d0) / dim0;
    dst[d2 * dim0_leading_dim * dim1 * dim3 + (d0 + dim0 * ite) * dim1 * dim3 + d1 * dim3 + d3] = src[i];
  }
}

template<>
__global__ void transpose_4d(half* dst, half* src,
                            const int dim0,
                            const int dim1,
                            const int dim2,
                            const int dim3,
                            const int dim0_leading_dim,
                            const int ite)
{
  half2 *dst_ptr = (half2 *) dst;
  half2 *src_ptr = (half2 *) src;
  const int half_dim3 = dim3 / 2;
  // transpose from [dim0, dim1, dim2, half_dim3] to [dim2, dim0, dim1, half_dim3]
  // where the dimension of X is dim0_leading_dim, and offset is ite * dim0
  for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < dim0 * dim1 * dim2 * half_dim3; i+= blockDim.x * gridDim.x)
  {
    int index = i;
    const int d3 = index % half_dim3;
    index = (index - d3) / half_dim3;
    const int d2 = index % dim2;
    index = (index - d2) / dim2;
    const int d1 = index % dim1;
    index = (index - d1) / dim1;
    const int d0 = index % dim0;
    index = (index - d0) / dim0;
    dst_ptr[d2 * dim0_leading_dim * dim1 * half_dim3 + (d0 + dim0 * ite)  * dim1 * half_dim3 + d1 * half_dim3 + d3] = src_ptr[i];
  }
}

template<typename T>
void transpose_4d_kernelLauncher(T* dst, T* src,
                                const int local_batch_size,
                                const int seq_len,
                                const int size_per_head,
                                const int local_hidden_units,
                                const int local_head_num,
                                const int batch_size,
                                const int ite,
                                cudaStream_t stream)
{
  transpose_4d<<<local_batch_size * seq_len * local_hidden_units / 512, 512 / (4 / (sizeof(T))), 0, stream>>>(
    dst, src, 
    local_batch_size, local_head_num, 
    seq_len, size_per_head, batch_size, ite);
}    

template void transpose_4d_kernelLauncher(
  float* dst, 
  float* src,
  const int local_batch_size,
  const int seq_len,
  const int size_per_head,
  const int local_hidden_units,
  const int local_head_num,
  const int batch_size,
  const int ite,
  cudaStream_t stream);

template void transpose_4d_kernelLauncher(
  half* dst, 
  half* src,
  const int local_batch_size,
  const int seq_len,
  const int size_per_head,
  const int local_hidden_units,
  const int local_head_num,
  const int batch_size,
  const int ite,
  cudaStream_t stream);

#define NEW_TRANSPOSE_BATCH_MAJOR 1

template<typename T>
__global__ void transpose_4d_batch_major_k_cache(T* k_dst, const T* k_src,
                              const int head_num,
                              const int size_per_head,
                              const int seq_len,
                              const int max_seq_len)
{
  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  constexpr int X_ELEMS = (sizeof(T) == 4)? 4 : 8;

  auto key_src = reinterpret_cast<const uint4*>(k_src + batch_id * head_num * size_per_head * seq_len + head_id * size_per_head * seq_len);
  auto key_dst = reinterpret_cast<uint4*>(k_dst + batch_id * head_num * size_per_head * max_seq_len + head_id * size_per_head * max_seq_len);
  
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size_per_head_div_x = size_per_head / X_ELEMS;
  if (out_idx >= head_num * size_per_head_div_x * max_seq_len) return;
  
  int idx = out_idx;
  const int k_seq_len_id = idx % max_seq_len;
  idx = (idx - k_seq_len_id) / max_seq_len;
  const int k_head_size_id = idx % size_per_head_div_x;

  if (k_seq_len_id < seq_len)
    key_dst[out_idx] = key_src[k_seq_len_id * size_per_head_div_x + k_head_size_id];
}

template<typename T>
__global__ void transpose_4d_batch_major_v_cache(T* v_dst, const T* v_src,
                              const int head_num,
                              const int size_per_head,
                              const int seq_len,
                              const int max_seq_len)
{
  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;

  // 16 byte loads will handle "x" dimension
  auto val_src = reinterpret_cast<const uint4*>(v_src + batch_id * head_num * size_per_head * seq_len + head_id * size_per_head * seq_len);
  auto val_dst = reinterpret_cast<uint4*>(v_dst + batch_id * head_num * size_per_head * max_seq_len + head_id * size_per_head * max_seq_len);

  // idx is over output dimension L * size_per_head / x for values
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  constexpr int X_ELEMS = (sizeof(T) == 4)? 4 : 8;
  const int size_per_head_div_x = size_per_head / X_ELEMS;

  if (idx >= size_per_head_div_x * seq_len) return;

  val_dst[idx] = val_src[idx];
}

template<typename T>
__global__ void transpose_4d_batch_major(T* k_dst, T* v_dst,
                              const T* k_src, const T* v_src,
                              const int head_num,
                              const int size_per_head,
                              const int seq_len,
                              const int max_seq_len)
{
    const int hidden_dim = head_num * size_per_head;
    const int x = (sizeof(T) == 4)? 4 : 8;
    const int size_per_head_split = size_per_head / x;
    const int batch_id = blockIdx.x;
    const int seq_id = blockIdx.y;

    for(int id = threadIdx.x; id < head_num * size_per_head_split * x; id += blockDim.x)
    {
        int tmp_id = id;
        int x_id = tmp_id % x;
        tmp_id = (tmp_id - x_id) / x;
        int size_id = tmp_id % size_per_head_split;
        tmp_id = (tmp_id - size_id) / size_per_head_split;
        int head_id = tmp_id % head_num;

        // key: [B, head_num, L, size_per_head / x, x] -> [B, head_num, size_per_head / x, L, x]
        k_dst[batch_id * hidden_dim * max_seq_len + head_id * size_per_head * max_seq_len + size_id * max_seq_len * x + seq_id * x + x_id] =
          k_src[batch_id * hidden_dim * seq_len + head_id * size_per_head * seq_len + seq_id * size_per_head + size_id * x + x_id];

        // value: [B, head_num, L, size_per_head / x, x] -> [B, head_num, L, size_per_head/x, x]
        v_dst[batch_id * hidden_dim * max_seq_len + head_id * size_per_head * max_seq_len + seq_id * size_per_head + size_id * x + x_id] =
          v_src[batch_id * hidden_dim * seq_len + head_id * size_per_head * seq_len + seq_id * size_per_head + size_id * x + x_id];
    }
}

template<typename T>
void transpose_4d_batch_major_kernelLauncher(T* k_dst, T* v_dst,
                                  const T* k_src, const T* v_src,
                                  const int local_batch_size,
                                  const int seq_len,
                                  const int max_seq_len,
                                  const int size_per_head,
                                  const int local_head_num,
                                  cudaStream_t stream)
{
  constexpr int block_sz = 128;
#if NEW_TRANSPOSE_BATCH_MAJOR == 1
  constexpr int x = (sizeof(T) == 4)? 4 : 8;
  int size = max_seq_len * size_per_head / x; 
  dim3 grid((size + block_sz - 1) / block_sz, local_batch_size, local_head_num);
  dim3 grid_v((seq_len * size_per_head / x + block_sz - 1) / block_sz, local_batch_size, local_head_num);

  transpose_4d_batch_major_k_cache<<<grid, block_sz, 0, stream>>>(
    k_dst, k_src,
    local_head_num,
    size_per_head,
    seq_len,
    max_seq_len
  );

  transpose_4d_batch_major_v_cache<<<grid_v, block_sz, 0, stream>>>(
    v_dst, v_src,
    local_head_num,
    size_per_head,
    seq_len,
    max_seq_len
  );
#else
  dim3 grid(local_batch_size, seq_len);

  transpose_4d_batch_major<<<grid, block_sz, 0, stream>>>(
    k_dst, v_dst,
    k_src, v_src,
    local_head_num,
    size_per_head,
    seq_len,
    max_seq_len
  );
#endif
}

template void transpose_4d_batch_major_kernelLauncher(float* k_dst, float* v_dst,
  const float* k_src, const float* v_src,
  const int local_batch_size,
  const int seq_len,
  const int max_seq_len,
  const int size_per_head,
  const int local_head_num,
  cudaStream_t stream);

template void transpose_4d_batch_major_kernelLauncher(half* k_dst, half* v_dst,
  const half* k_src, const half* v_src,
  const int local_batch_size,
  const int seq_len,
  const int max_seq_len,
  const int size_per_head,
  const int local_head_num,
  cudaStream_t stream);

template<typename T>
__global__
void add_QKV_bias_generalized_2(const T* __restrict QKV,
                                const T* __restrict bias,
                                T* q_buf_, T* k_buf_, T* v_buf_,
                                const int batch_size, const int seq_len,
                                const int head_num, const int size_per_head,
                                const int word_per_block)
{
  // QKV: [batch x sequence length, hidden * 3]
  const T* data_ptr;
  T* buf_ptr;

  int n = head_num * size_per_head;
  const int blocks_per_word = n / blockDim.x;
  const int blocks_per_buffer = gridDim.x / 3;
  const int qkv_id = blockIdx.x / blocks_per_buffer;
  const int block_id_in_buffer = blockIdx.x % blocks_per_buffer;
  const int offset = block_id_in_buffer * blockDim.x + threadIdx.x;
  const int bias_id = offset % n;
  T* buf_ptrs[3] = {q_buf_, k_buf_, v_buf_};
  
  const int bid = blockIdx.x;
  
  for(int index = threadIdx.x; index < n; index += blockDim.x)
  {
    buf_ptrs[index / n][bid * n + index % n] = QKV[bid * 3 * n + index] + __ldg(&bias[index]);
  }
}

template <typename T, int size_per_head, int block_sz>
__global__ 
void cross_attention_kernel_opt(
  T* __restrict query_buf, const T* __restrict Q_bias, 
  T* __restrict key_cache, const T* __restrict K_bias, 
  T* __restrict value_cache, const T* __restrict V_bias,
  const int* length_per_sample, T* __restrict context_buf,
  const bool* finished,
  int batch_size, int head_num, const int step, const int seq_len, const float scalar)
{  
  if(finished != nullptr && finished[blockIdx.x / head_num] == true) return;
  typedef Copy_t<T, size_per_head> copy_t;
  const int elems_per_thread = size_per_head / WARP_SIZE;
  union Access_t
  {
    copy_t v;
    T x[elems_per_thread]; // supported size 1,2,4
  };
  typedef struct Float_n_t
  {
    float x[elems_per_thread]; // supported size 1,2,4
  } float_n_t;

  __shared__ float_n_t sq[block_sz];
  extern __shared__ float logits[]; // use to store the logits from [0~step]

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int warp_num = block_sz / WARP_SIZE;

  typedef cub::BlockReduce<float, block_sz> MaxValBlockReduce;
  typedef cub::BlockReduce<float, block_sz> BlockReduce;
  __shared__ typename MaxValBlockReduce::TempStorage max_val_block_temp_storage;
  __shared__ typename BlockReduce::TempStorage block_temp_storage;

  __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage[warp_num];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x / head_num;
  const int head_id = blockIdx.x % head_num;
  
  int length = __ldg(&length_per_sample[bid]);

  const int lane_id = tid % WARP_SIZE;

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head;
  int qkv_bias_id = head_id * size_per_head;

  int key_value_id = bid * (seq_len * head_num * size_per_head) + 
  + head_id * size_per_head;

  query_buf = &query_buf[qkv_id];
  K_bias = &K_bias[qkv_bias_id];
  key_cache = &key_cache[key_value_id];
  Q_bias = &Q_bias[qkv_bias_id];
  V_bias = &V_bias[qkv_bias_id];
  value_cache = &value_cache[key_value_id];
  context_buf = &context_buf[qkv_id];

  Access_t bias_r, key_val_r, query_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((copy_t *)query_buf + lane_id);
  bias_r.v = *((copy_t *)Q_bias + lane_id);
  float qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i)
  {
    qb_r[i] =  (float)query_buf_r.x[i] + (float)bias_r.x[i];
  }

  //offset for each step
  int offset =  head_num * size_per_head;

  bias_r.v = *((copy_t *) K_bias + lane_id);
  for(int ite = warp_id; ite < length; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&key_cache[ite * offset] + lane_id);

    //For the first step, we should add bias to key memory cache.
    //The KV memory cache only need to be updated at the first step.
    if (step == 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = (float)key_val_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&key_cache[ite * offset] + lane_id) = key_val_r.v;
    }
    float val = 0.f;
    for (int i = 0; i < elems_per_thread; i++)
    {
      val = val +  (float)key_val_r.x[i] * qb_r[i] * scalar;
    }
    float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
    if (lane_id == 0)
    {
      logits[ite] = qk; 
    }
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;
  float local_i = -1e20f;
  for(int i = tid; i < length; i += blockDim.x)
    local_i = max(local_i, logits[i]);

  float max_val = MaxValBlockReduce(max_val_block_temp_storage).Reduce(local_i, cub::Max());
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  float local_o = 0.0f;
  for(int i = tid; i < length; i += blockDim.x)
  {
    logits[i] = __expf(logits[i] - s_max_val);
    local_o += logits[i];
  }
  float val = BlockReduce(block_temp_storage).Sum(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  float s_sum_inverse = __fdividef(1.0f, s_sum);
  for(int i = tid; i < length; i += blockDim.x)
  {
    logits[i] = logits[i] * s_sum_inverse;
  }
  __syncthreads(); 

  // This optimization introduces discrepancy because of different order in FP32 summation
  float sum_r[elems_per_thread] = {0.f};
  bias_r.v = *((copy_t *) V_bias + lane_id);
  for(int ite = warp_id; ite < length; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&value_cache[ite * offset] + lane_id);

    //For the first step, we should add bias to key memory cache.
    if(step == 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = (float)key_val_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&value_cache[ite * offset] + lane_id) = key_val_r.v;
    }
    for (int i = 0; i < elems_per_thread; ++i)
    {
      sum_r[i] += (float)key_val_r.x[i] * logits[ite]; 
    }
  }
  for (int i = 0; i < elems_per_thread; i++)
  {
    sq[warp_id * WARP_SIZE + lane_id].x[i] = sum_r[i];
  }
  __syncthreads();
  if (threadIdx.x < WARP_SIZE)
  {
    #pragma unroll
    for (int j = 1; j < warp_num; j++)
    {
      for (int i = 0; i < elems_per_thread; ++i)
      {
        sum_r[i] = sum_r[i] + (float)sq[j * WARP_SIZE + threadIdx.x].x[i];
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++)
  {
    key_val_r.x[i] = sum_r[i];
  }
  if (threadIdx.x  < WARP_SIZE)
  {
    *((copy_t *)context_buf + lane_id) = key_val_r.v;
  }
}

template<typename T>
__global__
void cross_attention_kernel(
  T* query_buf, const T* Q_bias,
  T* key_cache, const T* K_bias,
  T* value_cache, const T* V_bias,
  const int* length_per_sample, T* context_buf, 
  const bool* finished,
  int batch_size, int head_num, int size_per_head, int step, const int seq_len, const T scalar)
{
  if(finished != nullptr && finished[blockIdx.x / head_num] == true) return;
  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int length = __ldg(&length_per_sample[bid]);

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  int qkv_bias_id = head_id * size_per_head + tid;

  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id] + Q_bias[qkv_bias_id];
  __syncthreads();

  for(int ite = 0; ite < length; ++ite)
  {
    int key_id = bid * (seq_len * head_num * size_per_head) + ite * (head_num * size_per_head)
      + head_id * size_per_head + tid;

    T key = tid < size_per_head ? key_cache[key_id] : (T)(0.0f);

    //For the first step, we should add bias to key memory cache.
    //The KV memory cache only need to be updated at the first step.
    if(step == 1 && tid < size_per_head)
    {
      key += K_bias[head_id * size_per_head + tid];
      key_cache[key_id] = key;
    }

    T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
    T qk = blockReduceSum(val);
    if(threadIdx.x == 0)
      logits[ite] = qk;
    __syncthreads(); //try to remove
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;

  float local_i = tid < length ? (float)logits[tid] : -1e20f; 
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < length ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();
  if(tid < length)
    logits[tid] = local_o / s_sum;
  __syncthreads();

  if(tid < size_per_head)
  {
    T sum = (T)0.0f;
    for(int ite = 0; ite < length; ++ite)
    {
      int value_id = bid * seq_len * head_num * size_per_head + ite * head_num * size_per_head 
        + head_id * size_per_head + tid;

      T value = value_cache[value_id];

      //for the first step, we should add bias to key memory cache
      if(step == 1)
      {
        value += V_bias[head_id * size_per_head + tid];
        value_cache[value_id] = value;
      }  
      sum += value * logits[ite];
    }
    context_buf[bid * head_num * size_per_head + head_id * size_per_head + tid] = sum;
  }
}

template <typename T>
void cross_attention_dispatch(T* query_buf, const T* Q_bias, 
  T* key_cache, const T* K_bias, T* value_cache, const T* V_bias, const int* length,
  T* context_buf, const bool* finished,
  int batch_size, int head_num, int size_per_head, int step, int seq_len, cudaStream_t stream)
  {
    const int block_sz = ATTENTION_BLOCK_SIZE;
    float scalar = 1.f / sqrtf(size_per_head * 1.0f);

    dim3 grid(batch_size * head_num);

    int cond = size_per_head * ((ATTENION_OPT)? 1:0);
    switch (cond)
    {
      case 32:
        cross_attention_kernel_opt<T, 32, block_sz><<<grid, block_sz, sizeof(float)*seq_len, stream>>>(
          query_buf, Q_bias, key_cache, K_bias, value_cache, V_bias, length, context_buf, finished,
          batch_size, head_num, step, seq_len, scalar);
        break;
      case 64:
        cross_attention_kernel_opt<T, 64, block_sz><<<grid, block_sz, sizeof(float)*seq_len, stream>>>(
          query_buf, Q_bias, key_cache, K_bias, value_cache, V_bias, length, context_buf, finished,
          batch_size, head_num, step, seq_len, scalar);
        break;
      case 128:
        cross_attention_kernel_opt<T, 128, block_sz><<<grid, block_sz, sizeof(float)*seq_len, stream>>>(
          query_buf, Q_bias, key_cache, K_bias, value_cache, V_bias, length, context_buf, finished,
          batch_size, head_num, step, seq_len, scalar);
        break;
      default:
        // default path

        int block_size = 128;

        if(seq_len <= 64)
          block_size = 64;
        else if(seq_len <= 128 && seq_len > size_per_head)
          block_size = 128;
        else if(seq_len > 128 && seq_len <= 256)
          block_size = 256;
        else if(seq_len > 256 && seq_len <= 512)
          block_size = 512;
        else
          block_size = 1024;

        if(block_size < size_per_head)
          block_size = size_per_head;

        assert(block_size <= 1024);
        dim3 block(block_size);
        
        int shared_size = sizeof(T) * (size_per_head + seq_len);
        cross_attention_kernel<T><<<grid, block, shared_size, stream>>>(
          query_buf, Q_bias, 
          key_cache, K_bias,
          value_cache, V_bias,
          length, context_buf, finished,
          batch_size,
          head_num, size_per_head, step, seq_len, scalar);
    }
  }

template void cross_attention_dispatch(
  float* query_buf, 
  const float* Q_bias, 
  float* key_cache, 
  const float* K_bias, 
  float* value_cache, 
  const float* V_bias, 
  const int* length,
  float* context_buf, 
  const bool* finished,
  int batch_size, 
  int head_num, 
  int size_per_head, 
  int step, 
  int seq_len, 
  cudaStream_t stream);

template void cross_attention_dispatch(
  half* query_buf, 
  const half* Q_bias, 
  half* key_cache, 
  const half* K_bias, 
  half* value_cache, 
  const half* V_bias, 
  const int* length,
  half* context_buf, 
  const bool* finished,
  int batch_size, 
  int head_num, 
  int size_per_head, 
  int step, 
  int seq_len, 
  cudaStream_t stream);

  template void fusedQKV_masked_attention_kernelLauncher(
    const float* qkv_buf,
    const float* qkv_bias,
    float* k_cache,
    float* v_cache,
    float* output,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    const int max_seq_len,
    cudaStream_t stream);
  
  template void fusedQKV_masked_attention_kernelLauncher(
    const half* qkv_buf,
    const half* qkv_bias,
    half* k_cache,
    half* v_cache,
    half* output,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    const int max_seq_len,
    cudaStream_t stream);

  template void fusedQKV_masked_attention_kernelLauncher_v2(
    const float* qkv_buf,
    const float* qkv_bias,
    float* k_cache,
    float* v_cache,
    float* output,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    const int max_seq_len,
    const int max_input_len,
    const int* input_lengths,
    cudaStream_t stream);
  
  template void fusedQKV_masked_attention_kernelLauncher_v2(
    const half* qkv_buf,
    const half* qkv_bias,
    half* k_cache,
    half* v_cache,
    half* output,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    const int max_seq_len,
    const int max_input_len,
    const int* input_lengths,
    cudaStream_t stream);

}//namespace fastertransformer