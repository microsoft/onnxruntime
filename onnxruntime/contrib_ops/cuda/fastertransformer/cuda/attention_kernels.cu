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

#include "contrib_ops/cuda/fastertransformer/cuda/attention_kernels.cuh"

namespace fastertransformer 
{

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
    #pragma unroll
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
    #pragma unroll
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


    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}

__inline__ __device__
int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
  return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

template<typename T>
__global__
void add_QKV_bias(T* Q, const T* bias_Q, T* K, const T* bias_K, T* V, const T* bias_V, T* q_buf_, T* k_buf_, T* v_buf_, 
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int word_per_block)
{

  T* data_ptr;
  T* buf_ptr;
  const T* bias_ptr;
  
  int m = batch_size * seq_len;
  int n = head_num * size_per_head;

  int qkv_id = blockIdx.x * word_per_block / m;
  int row_offset = (blockIdx.x * word_per_block % m) * n;

  if(qkv_id == 0)
  {
    data_ptr = Q + row_offset;
    buf_ptr = q_buf_;
    bias_ptr = bias_Q;
  }
  else if(qkv_id == 1)
  {
    data_ptr = K + row_offset;
    buf_ptr = k_buf_;
    bias_ptr = bias_K;
  }
  else
  {
    data_ptr = V + row_offset;
    buf_ptr = v_buf_;
    bias_ptr = bias_V;
  }

  int batch_id = (blockIdx.x * word_per_block % m) / seq_len;
  int head_id = threadIdx.x / size_per_head;
  int id_in_head = threadIdx.x % size_per_head;
  int word_start_id = (blockIdx.x * word_per_block) % seq_len;

  T bias = __ldg(&bias_ptr[threadIdx.x]);

  for(int i = word_start_id; i < word_start_id + word_per_block; ++i)
  {
    T tmp = data_ptr[threadIdx.x] + bias;

    int target_id = batch_id * (seq_len * head_num * size_per_head) + head_id * seq_len * size_per_head + 
      i * size_per_head + id_in_head;

    buf_ptr[target_id] = tmp;
    data_ptr += n;
  }
}

template <>
__global__
void add_QKV_bias(half* Q, const half* bias_Q, half* K, const half* bias_K, half* V, const half* bias_V, 
  half* q_buf_, half* k_buf_, half* v_buf_, 
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int word_per_block)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = tid / (head_num * seq_len * size_per_head);
  int seq_id = (tid % (head_num * seq_len * size_per_head)) / (head_num * size_per_head);
  int head_id = (tid % (head_num * size_per_head)) / size_per_head;
  int id = tid % size_per_head;
  int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);

  int bias_id = threadIdx.x;

  half2* src_ptr = (half2*)Q;
  half2* dst_ptr = (half2*)q_buf_;
  const half2* bias_ptr = (const half2*)bias_Q;

  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));

  src_ptr = (half2*)K;
  dst_ptr = (half2*)k_buf_;
  bias_ptr = (const half2*)bias_K;
  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));

  src_ptr = (half2*)V;
  dst_ptr = (half2*)v_buf_;
  bias_ptr = (const half2*)bias_V;
  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));
}


template<typename T>
__global__
void add_QKV_bias_generalized(const T* __restrict Q,
                              const T* __restrict bias_Q,
                              const T* __restrict K,
                              const T* __restrict bias_K,
                              const T* __restrict V,
                              const T* __restrict bias_V,
                              T* q_buf_, T* k_buf_, T* v_buf_,
                              const int batch_size, const int seq_len,
                              const int head_num, const int size_per_head,
                              const int word_per_block)
{

  const T* data_ptr;
  T* buf_ptr;
  T bias;

  int n = head_num * size_per_head;
  const int blocks_per_word = n / blockDim.x;
  const int blocks_per_buffer = gridDim.x / 3;
  const int qkv_id = blockIdx.x / blocks_per_buffer;
  const int block_id_in_buffer = blockIdx.x % blocks_per_buffer;
  const int offset = block_id_in_buffer * blockDim.x + threadIdx.x;
  const int bias_id = offset % n;

  if(qkv_id == 0)
  {
    data_ptr = Q + offset;
    buf_ptr = q_buf_;
    bias = __ldg(&bias_Q[bias_id]);
  }
  else if(qkv_id == 1)
  {
    data_ptr = K + offset;
    buf_ptr = k_buf_;
    bias = __ldg(&bias_K[bias_id]);
  }
  else
  {
    data_ptr = V + offset;
    buf_ptr = v_buf_;
    bias = __ldg(&bias_V[bias_id]);
  }

  const int head_id = bias_id / size_per_head;
  const int size_id = bias_id % size_per_head;

  for(int i = 0; i < word_per_block; i++)
  {
    const int block_lane = i * blocks_per_buffer;
    const int batch_id = (block_id_in_buffer + block_lane) / seq_len / blocks_per_word;
    const int word_id = ((block_id_in_buffer + block_lane) / blocks_per_word) % seq_len;

    int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head +
      word_id * size_per_head + size_id;
    buf_ptr[target_id] = __ldg(&data_ptr[block_lane * blockDim.x]) + bias;
  }
}

template <typename T>
void add_QKV_bias_transpose_kernelLauncher(
  T* q_buf,
  T* k_buf,
  T* v_buf,
  T* Q,
  const T* bias_Q,
  T* K,
  const T* bias_K,
  T* V,
  const T* bias_V,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const int size_per_head,
  cudaStream_t stream)
{
  const int k = head_num * size_per_head;
  dim3 grid, block;
  if(k <= 1024)
  {
    if(sizeof(T) == 4)
    {
      const int m = batch_size * seq_len;
      const int word_per_block = 1;
      assert(k <= 1024);
      assert(m / word_per_block * 3 <= 65536);

      dim3 grid(m / word_per_block * 3);
      dim3 block(k);
      add_QKV_bias<T><<<grid, block, 0, stream>>>(Q, bias_Q, K, bias_K, V, bias_V, q_buf, k_buf, v_buf,
        batch_size, seq_len, head_num, size_per_head, word_per_block);
    }
    else
    {
      const int word_per_block = 1;
      grid.x = batch_size * seq_len / word_per_block;
      block.x = head_num * size_per_head * word_per_block / 2;

      assert(block.x <= 1024);

      add_QKV_bias<T><<<grid, block, 0, stream>>>(Q, bias_Q, K, bias_K, V, bias_V, q_buf, k_buf, 
      v_buf, batch_size, seq_len, head_num, size_per_head / 2, word_per_block);
    }
  }
  else
  {
    // k > 1024, so split into many block
    if(sizeof(T) == 4)
    {
      const int m = batch_size * seq_len;
      const int word_per_block = 4;
      dim3 block;
      if(k % 512 == 0)
        block.x = 512;
      else if(k % 384 == 0)
        block.x = 384;
      else if(k % 256 == 0)
        block.x = 256;
      else if(k % 128 == 0)
        block.x = 128;
      else
        printf("[ERROR] no supported k %d \n", k);
      assert(k % block.x == 0);
      dim3 grid(m * k / block.x / word_per_block * 3);
      assert(grid.x <= 65536 && grid.x > 0);
      add_QKV_bias_generalized<T><<<grid, block, 0, stream>>>(Q, bias_Q, K, bias_K, V, bias_V, q_buf, k_buf, v_buf,
        batch_size, seq_len, head_num, size_per_head, word_per_block);

    }
    else
    {
      const int m = batch_size * seq_len;
      const int word_per_block = 4;
      const int half_k = k / 2;
      dim3 block;
      if(half_k % 512 == 0)
        block.x = 512;
      else if(half_k % 384 == 0)
        block.x = 384;
      else if(half_k % 256 == 0)
        block.x = 256;
      else if(half_k % 128 == 0)
        block.x = 128;
      else if(half_k % 64 == 0)
        block.x = 64;
      else
        printf("[ERROR] no supported half_k %d \n", half_k);
      assert(half_k % block.x == 0);
      dim3 grid(m * half_k / block.x / word_per_block * 3);
      assert(grid.x <= 65536 && grid.x > 0);
      add_QKV_bias_generalized<half2><<<grid, block, 0, stream>>>((const half2*)Q, (const half2*)bias_Q,
                                                                  (const half2*)K, (const half2*)bias_K,
                                                                  (const half2*)V, (const half2*)bias_V,
                                                                  (half2*)q_buf, (half2*)k_buf, (half2*)v_buf,
                                                                  batch_size, seq_len, head_num, 
                                                                  size_per_head / 2, word_per_block);
    }
  }
}

template <typename T>
__global__
void add_fusedQKV_bias_transpose_kernel(
  T* q_buf,
  T* k_buf,
  T* v_buf,
  const T* __restrict QKV,
  const T* __restrict qkv_bias,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const int size_per_head)
{
  // QKV: [m, 3, n]
  // qkv_bias: [3, n]
  // q_buf, k_buf, v_buf: [batch, head_num, seq_len, size_per_head]

  T* qkv_ptr[3] = {q_buf, k_buf, v_buf};
  const int n = head_num * size_per_head;
  for(int index = blockDim.x * blockIdx.x + threadIdx.x; index < batch_size * seq_len * 3 * n; index += gridDim.x * blockDim.x)
  {
    int bias_id = index % (3 * n);
    T val = __ldg(&QKV[index]) + __ldg(&qkv_bias[bias_id]);

    int tmp_index = index;
    const int target_batch_id = tmp_index / (seq_len * 3 * n);
    tmp_index -= target_batch_id * seq_len * 3 * n;
    const int seq_id = tmp_index / (3 * n);
    tmp_index -= seq_id * 3 * n;
    const int qkv_id = tmp_index / n;
    tmp_index -= qkv_id * n;
    const int head_id = tmp_index / size_per_head;
    const int size_id = tmp_index - head_id * size_per_head;

    qkv_ptr[qkv_id][
      target_batch_id * head_num * seq_len * size_per_head +
      head_id * seq_len * size_per_head +
      seq_id * size_per_head +
      size_id] = val;
  }
}

template <typename T>
void add_fusedQKV_bias_transpose_kernelLauncher(
  T* q_buf,
  T* k_buf,
  T* v_buf,
  T* QKV,
  const T* qkv_bias,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const int size_per_head,
  cudaStream_t stream)
{
  const int m = batch_size * seq_len;
  const int n = head_num * size_per_head;
  dim3 block(384);
  dim3 grid((int)(ceil(1.0 * m * n / 384)));
  add_fusedQKV_bias_transpose_kernel<<<grid, block, 0, stream>>>(
    q_buf, k_buf, v_buf, QKV, qkv_bias,
    batch_size, seq_len, head_num, size_per_head);
}

template <typename T>
__global__
void softmax_kernel(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, 
  const T scalar)
{
    int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * seq_len * seq_len;
    int mask_offset = batch_id * seq_len * seq_len;

    __shared__ float s_sum, s_max;

    for(int i = 0; i < seq_len; ++i)
    {
      float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
      float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
      
      mask_val = (1.0f - mask_val) * -10000.0f;

      float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scalar + mask_val): -1e20f;

      float max_val = blockReduceMax<float>(tmp);

      if(threadIdx.x == 0)
        s_max = max_val;
      __syncthreads();

      qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

      float sum_val = blockReduceSum<float>(qk);

      if(threadIdx.x == 0)
      {
        s_sum = sum_val + 1e-6f;
      }
      __syncthreads();

      if(threadIdx.x < seq_len)
        qk_buf_[threadIdx.x + qk_offset] = (T)(qk / s_sum);

      qk_offset += seq_len;
      mask_offset += seq_len;
    }
}


template <typename T>
__global__
void softmax_kernel_v2(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, 
  const int seq_len, const float scalar)
{
    int batch_id = blockIdx.x / head_num / seq_len;
    int seq_id = blockIdx.x % seq_len;
    int qk_offset = blockIdx.x * seq_len;
    int mask_offset = batch_id * seq_len * seq_len + seq_id * seq_len;

    __shared__ float s_sum, s_max;

    float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
      
    mask_val = (1.0f - mask_val) * -10000.0f;

    float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scalar + mask_val) : -1e20f;
    float max_val = blockReduceMax<float>(tmp);
    if(threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    float qk_tmp = threadIdx.x < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);

    if(threadIdx.x == 0)
    {
      s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

//grid = (seq_len/word_per_thread, batch_size, head_num)
//block.x = max(32, (seq_len + 31)/32*32)
template <typename T>
__global__
void softmax_kernel_v3(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, const T scalar)
{
    
  bool qual = threadIdx.x < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    float tmp = -1e20f;
    int qk_offset;
    __shared__ float s_mean, s_max;
    if (qual){
      qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + seq_id) *seq_len + threadIdx.x;
      int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len + threadIdx.x;

      float qk = static_cast<float>(qk_buf_[qk_offset]);
      float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

      mask_val = (1.0f - mask_val) * -10000.0f;

      tmp = qk * static_cast<float>(scalar) + mask_val;
    }

    float max_val = blockReduceMax<float>(tmp);
    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();
    
    float qk_tmp = qual ? __expf(tmp - s_max) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);
    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();
    
    if(qual)
      qk_buf_[qk_offset] = (T)(qk_tmp * s_mean);
  }
}  


//grid = (seq_len/word_per_thread, batch_size, head_num)
//block.x = max(32, (seq_len/2 + 31)/32*32)
//seq_len % 2 == 0
template <>
__global__
void softmax_kernel_v3(half* qk_buf_, const half* attr_mask, 
                      const int batch_size, const int head_num, 
                      const int seq_len, const half scalar)
{
  int threadIdx2 = threadIdx.x << 1;
  bool qual = threadIdx2 < seq_len;
  half2* qk_buf_half2Ptr = (half2*) qk_buf_;
  const half2* attr_mask_half2Ptr = (const half2*) attr_mask;
  __shared__ float s_mean, s_max;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    int qk_offset;
    half2 tmp = __float2half2_rn(0.0f);

    float max_val = -1e20f;
    half2 qk;
    if (qual){ 
      qk_offset = ((((blockIdx.y*head_num + blockIdx.z)*seq_len + seq_id) *seq_len) >> 1) + threadIdx.x;
      int mask_offset = (((blockIdx.y * seq_len + seq_id) * seq_len) >> 1) + threadIdx.x;

      qk = qk_buf_half2Ptr[qk_offset];
      half2 mask_val = __ldg(&attr_mask_half2Ptr[mask_offset]);
      half2 mask_val_tmp = __hmul2(__hsub2(__float2half2_rn(1.0f), mask_val), __float2half2_rn(-10000.0f));
      tmp = __hadd2(__hmul2(__half2half2(scalar), qk), mask_val_tmp);
      max_val = fmax((float)tmp.x, (float)tmp.y);
    }
    
    max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();
    
    if (qual){
      tmp = h2exp(__hsub2(tmp, __float2half2_rn(s_max)));
    }
    float sum_val = blockDim.x <= 32 ? warpReduceSum((float)(tmp.x + tmp.y)) : blockReduceSum<float>((float)(tmp.x + tmp.y));

    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(qual){
      qk = __hmul2(tmp, __float2half2_rn(s_mean));
      qk_buf_half2Ptr[qk_offset] = qk;
    }
  }
}

template <typename T>
__global__
void softmax_kernel_v3_LE32(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, const T scalar)
{
  bool qual = threadIdx.x < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    int qk_offset;
    __shared__ float s_mean, s_max;
    float tmp = -1e20f;
    if (qual){
      qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + seq_id) *seq_len + threadIdx.x;
      int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len + threadIdx.x;

      float qk = static_cast<float>(qk_buf_[qk_offset]);
      float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

      mask_val = (1.0f - mask_val) * -10000.0f;

      tmp = static_cast<float>(qk) * static_cast<float>(scalar) + mask_val;
    }
    float max_val = warpReduceMax<float>(tmp);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    tmp = qual ? __expf(tmp - s_max) : 0.0f;
    float sum_val = warpReduceSum<float>(tmp);

    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(qual)
      qk_buf_[qk_offset] = (T)(tmp * s_mean);
  }
}

template<typename T>
void attn_softmax_kernelLauncher(
  T* buffer,
  const T* attr_mask,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const T scalar,
  cudaStream_t stream)
{
  dim3 grid, block;
  //deal with odd seq_len
  if (seq_len % 2 != 0){
    if(seq_len <= 32)
      block.x = 32;
    else if(seq_len > 32 && seq_len <= 64)
      block.x = 64;
    else if(seq_len > 64 && seq_len <= 128)
      block.x = 128;
    else if(seq_len > 128 && seq_len <= 256)
      block.x = 256;
    else if(seq_len > 256 && seq_len <= 512)
      block.x = 512;
    else
      block.x = 1024;

    if(batch_size * head_num <= 120)
    {
      grid.x = batch_size * head_num * seq_len;
      softmax_kernel_v2<T><<<grid, block, 0, stream>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
    }
    else
    {
      grid.x = batch_size * head_num;
      softmax_kernel<T><<<grid, block, 0, stream>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
    }
  }
  //deal with even seq_len 
  else{
    grid.x = seq_len;
    if (batch_size * head_num > 360)
      grid.x = (seq_len + 31) / 32; // ceil(float(seq_len)/32.0f);
    grid.y = batch_size;
    grid.z = head_num;
    if (seq_len <= 32){
      block.x = 32;
      softmax_kernel_v3_LE32<T><<<grid, block, 0, stream>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
    }
    else{
      if (sizeof(T) == 2){
        block.x = (seq_len/2 + 31)/32*32;
        softmax_kernel_v3<<<grid, block, 0, stream>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
      }
      else{
        block.x = (seq_len + 31)/32*32;
        softmax_kernel_v3<T><<<grid, block, 0, stream>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
      }
    }
    grid.x = grid.y = grid.z = 1;
  }
}

template<typename T>
__global__
void transpose(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
  int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = blockIdx.x % seq_len;
  int head_id = (blockIdx.x % (head_num * seq_len))/ seq_len;
  dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
    + head_id * size_per_head + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template<>
  __global__
void transpose(half* src, half* dst,
    const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int batch_id = tid / (head_num * seq_len * size_per_head);
  int head_id = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
  int seq_id = (tid % (seq_len * size_per_head)) / size_per_head;
  int id = tid % size_per_head;

  int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);
  half2* src_ptr = (half2*)src;
  half2* dst_ptr = (half2*)dst;

  dst_ptr[target_id] = src_ptr[tid];
}

template <typename T> 
void transpose_kernelLauncher(
  T* dst,
  T* src,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const int size_per_head,
  cudaStream_t stream)
{
  dim3 grid, block;
  if(sizeof(T) == 2)
  {
    const int seq_per_block = 4;
    grid.x = batch_size * head_num * seq_len / seq_per_block;
    block.x = seq_per_block * size_per_head / 2;

    assert(grid.x * seq_per_block == batch_size * head_num * seq_len);

    transpose<T><<<grid, block, 0, stream>>>(src, dst, 
        batch_size, seq_len, head_num, size_per_head / 2);
  }
  else
  {
    const int seq_per_block = 1;
    grid.x = batch_size * head_num * seq_len / seq_per_block;
    block.x = seq_per_block * size_per_head;
    transpose<T><<<grid, block, 0, stream>>>(src, dst, 
      batch_size, seq_len, head_num, size_per_head);
  }
}

template void add_QKV_bias_transpose_kernelLauncher(
    float* q_buf,
    float* k_buf,
    float* v_buf,
    float* Q,
    const float* bias_Q,
    float* K,
    const float* bias_K,
    float* V,
    const float* bias_V,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    cudaStream_t stream);
  
template void add_QKV_bias_transpose_kernelLauncher(
    half* q_buf,
    half* k_buf,
    half* v_buf,
    half* Q,
    const half* bias_Q,
    half* K,
    const half* bias_K,
    half* V,
    const half* bias_V,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    cudaStream_t stream);

template void add_fusedQKV_bias_transpose_kernelLauncher(
    float* q_buf,
    float* k_buf,
    float* v_buf,
    float* QKV,
    const float* qkv_bias,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    cudaStream_t stream);

template void add_fusedQKV_bias_transpose_kernelLauncher(
    half* q_buf,
    half* k_buf,
    half* v_buf,
    half* QKV,
    const half* qkv_bias,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    cudaStream_t stream);

template void attn_softmax_kernelLauncher(
    float* buffer,
    const float* attr_mask,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const float scalar,
    cudaStream_t stream);
    
template void attn_softmax_kernelLauncher(
    half* buffer,
    const half* attr_mask,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const half scalar,
    cudaStream_t stream);
      
template void transpose_kernelLauncher(
    float* dst,
    float* src,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    cudaStream_t stream);
    
template void transpose_kernelLauncher(
    half* dst,
    half* src,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    cudaStream_t stream);
      
} // namespace fastertransformer