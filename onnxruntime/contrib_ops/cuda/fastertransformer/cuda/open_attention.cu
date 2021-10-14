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
/**
* Open sourced multi-head attention
**/

#include "contrib_ops/cuda/fastertransformer/utils/allocator.h"
#include "contrib_ops/cuda/fastertransformer/cuda/multi_head_attention.h"
#include "contrib_ops/cuda/fastertransformer/cuda/open_attention.h"
#include "contrib_ops/cuda/fastertransformer/cuda/attention_kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace fastertransformer{
namespace cuda{

/**
* Multi-head attetion open sourced
*/
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

//build a mapping for fullData to removePaddingData
//grid((valid_word_num+63)/64)
//block(64)
__global__ void mappingRemovePaddingData(int *mapping, const int* sequence_id_offset, const int valid_word_num){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < valid_word_num)
    mapping[idx + __ldg(sequence_id_offset + idx)] = idx;
}

void mappingRemovePaddingData_kernelLauncher(const int batch_size, const int seq_len, 
                                             const int valid_word_num, int *mapping, 
                                             const int* sequence_id_offset, cudaStream_t stream)
{
  cudaMemsetAsync(mapping, -1, batch_size * seq_len * sizeof(int), stream);
  mappingRemovePaddingData<<<dim3((valid_word_num + 63)/64), dim3(64), 0, stream>>>(mapping, sequence_id_offset, valid_word_num);
}

//add_QK_bias_transform for batch int8 cublasLtMatmul & per axis quantization for weight
//1.add QK bias
//2.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = batch_size * seq_len, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C
//only for int32 input & int8 output
//seq_len, size_per_head must be a multiple of 32
//grid.x = batch_size * seq_len * 2;
//block.x = head_num * size_per_head / 4;
//using char4
template <typename T>
__global__
void add_QK_bias_transform(int8_t *q_buf_, int8_t *k_buf_, const int32_t* Q, const T* bias_Q, 
                           const int32_t* K, const T* bias_K, const int m, const int batch_size, 
                           const int seq_len, const int head_num, const int size_per_head, int stride, 
                           const float * q_weight_amax, const float *q_input_deQFactor_div127_ptr, const float * k_weight_amax, 
                           const float *k_input_deQFactor_div127_ptr, const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                           bool use_ORDER_COL32_2R_4R4)
{
  const int32_t* data_ptr;
  char4* buf_ptr4;
  const T* bias_ptr;
  const float* weight_amax;
  int qk_id = blockIdx.x / m;

  data_ptr = qk_id == 0 ? Q : K;
  buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
  bias_ptr = qk_id == 0 ? bias_Q : bias_K;
  const float input_deQFactor_div127 = qk_id == 0 ? __ldg(q_input_deQFactor_div127_ptr) : __ldg(k_input_deQFactor_div127_ptr);
  weight_amax = qk_id == 0 ? q_weight_amax : k_weight_amax;
  const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

  int threadIdx4 = threadIdx.x << 2;
  int batch_id = (blockIdx.x % m) / seq_len;
  int head_id = threadIdx4 / size_per_head;
  int id_in_head = threadIdx4 % size_per_head;
  int word_id = blockIdx.x % seq_len;

  int data_id = (((threadIdx4 >> 5) << 5)*m + ((blockIdx.x%m) << 5) + (threadIdx4&31));

  float scale;
  float tmp;
  char4 tmp4;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.x = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4)* input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.y = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.z = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.w = float_to_int8_rn(tmp*output_scale);


  //row_id, col_id of sub-matrix (m = seq_len, n = size_per_head), column-major

  int row_id = word_id;
  int col_id = id_in_head;
  //new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len)
  int new_col = col_id >> 5;
  int new_row;  
  if (use_ORDER_COL32_2R_4R4)
  {
    int row_in_tile = row_id & 31;
    int col_in_tile = col_id & 31; 
    new_row = (qk_id != 1) ?
              //COL32
              ((row_id << 5) + (col_id&31))
            :
               //COL32_2R_4R4
               (
               ((row_id >> 5) << 10) +
               //(((row%8)/2*4+row/8)*2+row%2)*32+col
               (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
               )
               ;
  }
  else
  {
    new_row = (qk_id != 1) ?
              //COL32
              ((row_id << 5) + (col_id&31))
            :
              //COL4
              ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
              ////row_id%2 is even row, otherwise odd row
              ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
              (
              ((((row_id >> 3) << 3) + ((row_id&1) << 2) + ((col_id&31) >> 3)) << 5) +
              ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
              ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
              (((((col_id&7) >= 4)?4:0) + ((row_id&7) >> 1)) << 2) +
              ////col_id%4 is the id of 4 cols
              (col_id&3)
              )
              ;
  }

  buf_ptr4[(((batch_id*head_num + head_id) * stride + (new_col << 5)*seq_len + new_row) >> 2)] = tmp4;
}

template <typename T>
void add_QK_bias_transform_kernelLauncher(int8_t *q_buf, int8_t *k_buf, const int32_t* Q, const T* bias_Q, 
                                          const int32_t* K, const T* bias_K, const int batch_size, 
                                          const int seq_len, const int head_num, const int size_per_head, 
                                          const float * q_weight_amax, const float *q_input_deQFactor_div127_ptr, 
                                          const float * k_weight_amax, const float *k_input_deQFactor_div127_ptr, 
                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream)
{
  add_QK_bias_transform<<<dim3(batch_size*seq_len*2), dim3((head_num * size_per_head)/4), 0, stream>>>(
    q_buf, k_buf, Q, bias_Q, K, bias_K, 
    batch_size * seq_len, batch_size, seq_len, head_num, size_per_head, seq_len*size_per_head, 
    q_weight_amax, q_input_deQFactor_div127_ptr, k_weight_amax, k_input_deQFactor_div127_ptr, 
    q_output_scale_ptr, k_output_scale_ptr, use_ORDER_COL32_2R_4R4);
}

template
void add_QK_bias_transform_kernelLauncher(int8_t *q_buf, int8_t *k_buf, const int32_t* Q, const float* bias_Q, 
                                          const int32_t* K, const float* bias_K, const int batch_size, 
                                          const int seq_len, const int head_num, const int size_per_head, 
                                          const float * q_weight_amax, const float *q_input_deQFactor_div127_ptr, 
                                          const float * k_weight_amax, const float *k_input_deQFactor_div127_ptr, 
                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);

template
void add_QK_bias_transform_kernelLauncher(int8_t *q_buf, int8_t *k_buf, const int32_t* Q, const half* bias_Q, 
                                          const int32_t* K, const half* bias_K, const int batch_size, 
                                          const int seq_len, const int head_num, const int size_per_head, 
                                          const float * q_weight_amax, const float *q_input_deQFactor_div127_ptr, 
                                          const float * k_weight_amax, const float *k_input_deQFactor_div127_ptr, 
                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);

//add_QK_bias_padding_transform for batch int8 cublasLtMatmul & per tensor quantization for weight
//1.add QK bias
//2.padding seq_len in k_buf_ to a multiple of 32 named seq_len_padded
//3.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = batch_size * seq_len, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len_padded, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C
//only for int8 IO
//size_per_head must be a multiple of 32
//grid.x = batch_size * seq_len * 2;
//block.x = head_num * size_per_head / 4;
//using char4
template <typename T>
__global__
void add_QK_bias_transform_varlen(int8_t *q_buf_, int8_t *k_buf_, const int8_t* Q, const T* bias_Q, 
                           const int8_t* K, const T* bias_K, const int m, const int batch_size, 
                           const int seq_len, const int head_num, const int size_per_head, 
                           const int seq_len_padded, const int stride_q, const int stride_k,
                           const float *q_input_deQFactor_ptr, const float *k_input_deQFactor_ptr, 
                           const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                           bool use_ORDER_COL32_2R_4R4)
{
  const char4* data_ptr;
  char4* buf_ptr4;
  const T* bias_ptr;
  int qk_id = blockIdx.x / m;

  data_ptr = qk_id == 0 ? (const char4*)Q : (const char4*)K;
  buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
  bias_ptr = qk_id == 0 ? bias_Q : bias_K;
  const float input_deQFactor = qk_id == 0 ? __ldg(q_input_deQFactor_ptr) : __ldg(k_input_deQFactor_ptr);
  const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

  int threadIdx4 = threadIdx.x << 2;
  int batch_id = (blockIdx.x % m) / seq_len;
  int head_id = threadIdx4 / size_per_head;
  int id_in_head = threadIdx4 % size_per_head;
  int word_id = blockIdx.x % seq_len;

  int data_id = (((threadIdx4 >> 5) << 5)*m + ((blockIdx.x%m) << 5) + (threadIdx4&31)) >> 2;

  float scale;
  float tmp;
  char4 tmp4 = __ldg(data_ptr+data_id);
  scale = static_cast<float>(tmp4.x) * input_deQFactor;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.x = float_to_int8_rn(tmp*output_scale);

  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(tmp4.y) * input_deQFactor;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.y = float_to_int8_rn(tmp*output_scale);

  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(tmp4.z) * input_deQFactor;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.z = float_to_int8_rn(tmp*output_scale);

  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(tmp4.w) * input_deQFactor;;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.w = float_to_int8_rn(tmp*output_scale);


  //row_id, col_id of sub-matrix (m = seq_len/seq_len_padded, n = size_per_head), column-major

  int row_id = word_id;
  int col_id = id_in_head;
  //new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len / COL32_ * seq_len_padded)
  int new_col = col_id >> 5;
  int new_row;  
  if (use_ORDER_COL32_2R_4R4)
  {
    int row_in_tile = row_id & 31;
    int col_in_tile = col_id & 31; 
    new_row = (qk_id != 1) ?
              //COL32
              ((row_id << 5) + (col_id&31))
            :
               //COL32_2R_4R4
               (
               ((row_id >> 5) << 10) +
               //(((row%8)/2*4+row/8)*2+row%2)*32+col
               (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
               )
               ;
  }
  else
  {
    new_row = (qk_id != 1) ?
              //COL32
              ((row_id << 5) + (col_id&31))
            :
              //COL4
              ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
              ////row_id%2 is even row, otherwise odd row
              ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
              (
              ((((row_id >> 3) << 3) + ((row_id&1) << 2) + ((col_id&31) >> 3)) << 5) +
              ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
              ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
              (((((col_id&7) >= 4)?4:0) + ((row_id&7) >> 1)) << 2) +
              ////col_id%4 is the id of 4 cols
              (col_id&3)
              )
              ;
  }

  const int act_seq_len = (qk_id == 0) ? seq_len : seq_len_padded;
  const int stride = (qk_id == 0) ? stride_q : stride_k;
  buf_ptr4[(((batch_id*head_num + head_id) * stride + (new_col << 5)*act_seq_len + new_row) >> 2)] = tmp4;
}   

//add_QK_bias_transform for batch int8 cublasLtMatmul & per axis quantization for weight
//1.add QK bias
//2.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = batch_size * seq_len, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C
//only for int8 IO
//seq_len, size_per_head must be a multiple of 32
//grid.x = batch_size * seq_len * 2;
//block.x = head_num * size_per_head / 4;
//using char4
template <typename T>
__global__
void add_QK_bias_transform(int8_t *q_buf_, int8_t *k_buf_, const int8_t* Q, const T* bias_Q, 
                           const int8_t* K, const T* bias_K, const int m, const int batch_size, 
                           const int seq_len, const int head_num, const int size_per_head, int stride, 
                           const float *q_input_deQFactor_ptr, const float *k_input_deQFactor_ptr, const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                           bool use_ORDER_COL32_2R_4R4)
{
  const char4* data_ptr;
  char4* buf_ptr4;
  const T* bias_ptr;
  int qk_id = blockIdx.x / m;

  data_ptr = qk_id == 0 ? (const char4*)Q : (const char4*)K;
  buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
  bias_ptr = qk_id == 0 ? bias_Q : bias_K;
  const float input_deQFactor = qk_id == 0 ? __ldg(q_input_deQFactor_ptr) : __ldg(k_input_deQFactor_ptr);
  const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

  int threadIdx4 = threadIdx.x << 2;
  int batch_id = (blockIdx.x % m) / seq_len;
  int head_id = threadIdx4 / size_per_head;
  int id_in_head = threadIdx4 % size_per_head;
  int word_id = blockIdx.x % seq_len;

  int data_id = (((threadIdx4 >> 5) << 5)*m + ((blockIdx.x%m) << 5) + (threadIdx4&31)) >> 2;

  float scale;
  float tmp;
  char4 tmp4 = __ldg(data_ptr+data_id);
  scale = static_cast<float>(tmp4.x) * input_deQFactor;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.x = float_to_int8_rn(tmp*output_scale);

  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(tmp4.y) * input_deQFactor;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.y = float_to_int8_rn(tmp*output_scale);

  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(tmp4.z) * input_deQFactor;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.z = float_to_int8_rn(tmp*output_scale);

  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(tmp4.w) * input_deQFactor;;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.w = float_to_int8_rn(tmp*output_scale);


  //row_id, col_id of sub-matrix (m = seq_len, n = size_per_head), column-major

  int row_id = word_id;
  int col_id = id_in_head;
  //new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len)
  int new_col = col_id >> 5;
  int new_row;  
  if (use_ORDER_COL32_2R_4R4)
  {
    int row_in_tile = row_id & 31;
    int col_in_tile = col_id & 31; 
    new_row = (qk_id != 1) ?
              //COL32
              ((row_id << 5) + (col_id&31))
            :
               //COL32_2R_4R4
               (
               ((row_id >> 5) << 10) +
               //(((row%8)/2*4+row/8)*2+row%2)*32+col
               (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
               )
               ;
  }
  else
  {
    new_row = (qk_id != 1) ?
              //COL32
              ((row_id << 5) + (col_id&31))
            :
              //COL4
              ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
              ////row_id%2 is even row, otherwise odd row
              ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
              (
              ((((row_id >> 3) << 3) + ((row_id&1) << 2) + ((col_id&31) >> 3)) << 5) +
              ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
              ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
              (((((col_id&7) >= 4)?4:0) + ((row_id&7) >> 1)) << 2) +
              ////col_id%4 is the id of 4 cols
              (col_id&3)
              )
              ;
  }

  buf_ptr4[(((batch_id*head_num + head_id) * stride + (new_col << 5)*seq_len + new_row) >> 2)] = tmp4;
}   


template <typename T>
void add_QK_bias_transform_kernelLauncher(int8_t *q_buf, int8_t *k_buf, const int8_t* Q, const T* bias_Q, 
                                          const int8_t* K, const T* bias_K, const int batch_size, 
                                          const int seq_len, const int head_num, const int size_per_head, 
                                          const float *q_input_deQFactor_ptr, const float *k_input_deQFactor_ptr, 
                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream) 
{
  assert(size_per_head % 32 == 0);
  if (seq_len % 32 == 0)
  {
    add_QK_bias_transform_varlen<<<dim3(batch_size*seq_len*2), dim3((head_num*size_per_head)/4), 0, stream>>>(
      q_buf, k_buf, Q, bias_Q, K, bias_K, 
      batch_size * seq_len, batch_size, seq_len, head_num, size_per_head, 
      seq_len, seq_len*size_per_head, seq_len*size_per_head,
      q_input_deQFactor_ptr, k_input_deQFactor_ptr, q_output_scale_ptr, k_output_scale_ptr,
      use_ORDER_COL32_2R_4R4);
  }
  else
  {
    int seq_len_padded = (seq_len + 31)/32*32;
    //The padding words will not be considered in softmax, so we don't need memset for k_buf_ 
    //cudaMemsetAsync(k_buf, 0, batch_size * head_num * seq_len_padded * size_per_head * sizeof(int8_t), stream);
    add_QK_bias_transform_varlen<<<dim3(batch_size*seq_len*2), dim3((head_num*size_per_head)/4), 0, stream>>>(
      q_buf, k_buf, Q, bias_Q, K, bias_K, 
      batch_size * seq_len, batch_size, seq_len, head_num, size_per_head, 
      seq_len_padded, seq_len*size_per_head, seq_len_padded*size_per_head,
      q_input_deQFactor_ptr, k_input_deQFactor_ptr, q_output_scale_ptr, k_output_scale_ptr,
      use_ORDER_COL32_2R_4R4);
  }
}

template
void add_QK_bias_transform_kernelLauncher(int8_t *q_buf, int8_t *k_buf, const int8_t* Q, const float* bias_Q, 
                                          const int8_t* K, const float* bias_K, const int batch_size, 
                                          const int seq_len, const int head_num, const int size_per_head, 
                                          const float *q_input_deQFactor_ptr, const float *k_input_deQFactor_ptr, 
                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);
                                          
template
void add_QK_bias_transform_kernelLauncher(int8_t *q_buf, int8_t *k_buf, const int8_t* Q, const half* bias_Q, 
                                          const int8_t* K, const half* bias_K, const int batch_size, 
                                          const int seq_len, const int head_num, const int size_per_head, 
                                          const float *q_input_deQFactor_ptr, const float *k_input_deQFactor_ptr, 
                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);

//add_QK_bias_transform & rebuild padding for batch int8 cublasLtMatmul & per axis quantization for weight
//1.add QK bias
//2.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = valid_word_num, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4
//only for int32 input & int8 output
//seq_len, size_per_head must be a multiple of 32
//grid.x = valid_word_num * 2;
//block.x = head_num * size_per_head / 4;
//using char4
template <typename T>
__global__
void add_QK_bias_transform_rebuild_padding(int8_t *q_buf_, int8_t *k_buf_, const int32_t* Q, const T* bias_Q, 
                                           const int32_t* K, const T* bias_K, const int* sequence_id_offset, 
                                           const int valid_word_num, const int m, const int batch_size, const int seq_len, 
                                           const int head_num, const int size_per_head, int stride, const float * q_weight_amax, 
                                           const float *q_input_deQFactor_div127_ptr, const float * k_weight_amax, 
                                           const float *k_input_deQFactor_div127_ptr, const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                           bool use_ORDER_COL32_2R_4R4)
{
  const int32_t* data_ptr;
  char4* buf_ptr4;
  const T* bias_ptr;
  const float* weight_amax;
  int qk_id = blockIdx.x / valid_word_num;

  data_ptr = qk_id == 0 ? Q : K;
  buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
  bias_ptr = qk_id == 0 ? bias_Q : bias_K;
  
  int threadIdx4 = threadIdx.x << 2;
  int m_full_idx = blockIdx.x % valid_word_num;
  m_full_idx = (valid_word_num != m) ? (m_full_idx + __ldg(sequence_id_offset+m_full_idx)) : m_full_idx;
  int batch_id = m_full_idx / seq_len;
  int head_id = threadIdx4 / size_per_head;
  int id_in_head = threadIdx4 % size_per_head;
  int word_id = m_full_idx % seq_len;
  
  const float input_deQFactor_div127 = qk_id == 0 ? __ldg(q_input_deQFactor_div127_ptr) : __ldg(k_input_deQFactor_div127_ptr);
  weight_amax = qk_id == 0 ? q_weight_amax : k_weight_amax;
  const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

  int data_id = (((threadIdx4 >> 5) << 5)*valid_word_num + ((blockIdx.x%valid_word_num) << 5) + (threadIdx4&31));
    
  float scale;
  float tmp;
  char4 tmp4;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.x = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4)* input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.y = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.z = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.w = float_to_int8_rn(tmp*output_scale);

  //row_id, col_id of sub-matrix (m = seq_len, n = size_per_head), column-major
  int row_id = word_id;
  int col_id = id_in_head;
  //new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len)
  int new_col = col_id >> 5;
  int new_row; 
  if (use_ORDER_COL32_2R_4R4)
  {
    int row_in_tile = row_id & 31;
    int col_in_tile = col_id & 31; 
    new_row = (qk_id != 1) ?
              //COL32
              ((row_id << 5) + (col_id&31))
            :
              //COL32_2R_4R4
              (
              ((row_id >> 5) << 10) +
              //(((row%8)/2*4+row/8)*2+row%2)*32+col
              (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
              )
              ;
  }
  else
  {
    new_row = (qk_id != 1) ?
              //COL32
              ((row_id << 5) + (col_id&31))
            :
              //COL4
              ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
              ////row_id%2 is even row, otherwise odd row
              ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
              (
              ((((row_id >> 3) << 3) + ((row_id&1) << 2) + ((col_id&31) >> 3)) << 5) +
              ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
              ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
              (((((col_id&7) >= 4)?4:0) + ((row_id&7) >> 1)) << 2) +
              ////col_id%4 is the id of 4 cols
              (col_id&3)
              )
              ;
  }

  buf_ptr4[(((batch_id*head_num + head_id) * stride + (new_col << 5)*seq_len + new_row) >> 2)] = tmp4;
}

template <typename T>
void add_QK_bias_transform_rebuild_padding_kernelLauncher(int8_t *q_buf, int8_t *k_buf, 
                                                          const int32_t* Q, const T* bias_Q, 
                                                          const int32_t* K, const T* bias_K, 
                                                          const int* sequence_id_offset, const int valid_word_num, 
                                                          const int batch_size, const int seq_len, 
                                                          const int head_num, const int size_per_head, 
                                                          const float * q_weight_amax, 
                                                          const float *q_input_deQFactor_div127_ptr, 
                                                          const float * k_weight_amax, 
                                                          const float *k_input_deQFactor_div127_ptr, 
                                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream)
{
  add_QK_bias_transform_rebuild_padding<<<dim3(valid_word_num*2), dim3((head_num*size_per_head)/4), 0, stream>>>(
    q_buf, k_buf, Q, bias_Q, K, bias_K, 
    sequence_id_offset, valid_word_num, 
    batch_size*seq_len, batch_size, seq_len, 
    head_num, size_per_head, seq_len*size_per_head, 
    q_weight_amax, q_input_deQFactor_div127_ptr, 
    k_weight_amax, k_input_deQFactor_div127_ptr, 
    q_output_scale_ptr, k_output_scale_ptr,
    use_ORDER_COL32_2R_4R4);
}  

template
void add_QK_bias_transform_rebuild_padding_kernelLauncher(int8_t *q_buf, int8_t *k_buf, 
                                                          const int32_t* Q, const float* bias_Q, 
                                                          const int32_t* K, const float* bias_K, 
                                                          const int* sequence_id_offset, const int valid_word_num, 
                                                          const int batch_size, const int seq_len, 
                                                          const int head_num, const int size_per_head, 
                                                          const float * q_weight_amax, 
                                                          const float *q_input_deQFactor_div127_ptr, 
                                                          const float * k_weight_amax, 
                                                          const float *k_input_deQFactor_div127_ptr, 
                                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);
                                                          
template
void add_QK_bias_transform_rebuild_padding_kernelLauncher(int8_t *q_buf, int8_t *k_buf, 
                                                          const int32_t* Q, const half* bias_Q, 
                                                          const int32_t* K, const half* bias_K, 
                                                          const int* sequence_id_offset, const int valid_word_num, 
                                                          const int batch_size, const int seq_len, 
                                                          const int head_num, const int size_per_head, 
                                                          const float * q_weight_amax, 
                                                          const float *q_input_deQFactor_div127_ptr, 
                                                          const float * k_weight_amax, 
                                                          const float *k_input_deQFactor_div127_ptr, 
                                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);  

//add_QK_bias_transform & rebuild padding for batch int8 cublasLtMatmul & per tensor quantization for weight
//1.add QK bias
//2.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = valid_word_num, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4
//only for int8 IO
//seq_len, size_per_head must be a multiple of 32
//grid.x = valid_word_num * 2;
//block.x = head_num * size_per_head / 4;
//using char4
template <typename T>
__global__
void add_QK_bias_transform_rebuild_padding(int8_t *q_buf_, int8_t *k_buf_, const int8_t* Q, const T* bias_Q, 
                                           const int8_t* K, const T* bias_K, const int* sequence_id_offset, 
                                           const int valid_word_num, const int m, const int batch_size, const int seq_len, 
                                           const int head_num, const int size_per_head, int stride,  
                                           const float *q_deQFactor_ptr,  const float *k_deQFactor_ptr, 
                                           const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                           bool use_ORDER_COL32_2R_4R4)
{
  const char4* data_ptr;
  char4* buf_ptr4;
  const T* bias_ptr;
  int qk_id = blockIdx.x / valid_word_num;

  data_ptr = qk_id == 0 ? (const char4*)Q : (const char4*)K;
  buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
  bias_ptr = qk_id == 0 ? bias_Q : bias_K;
  
  int threadIdx4 = threadIdx.x << 2;
  int m_full_idx = blockIdx.x % valid_word_num;
  m_full_idx = (valid_word_num != m) ? (m_full_idx + __ldg(sequence_id_offset+m_full_idx)) : m_full_idx;
  int batch_id = m_full_idx / seq_len;
  int head_id = threadIdx4 / size_per_head;
  int id_in_head = threadIdx4 % size_per_head;
  int word_id = m_full_idx % seq_len;
  
  const float deQFactor = qk_id == 0 ? __ldg(q_deQFactor_ptr) : __ldg(k_deQFactor_ptr);
  const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

  int data_id = (((threadIdx4 >> 5) << 5)*valid_word_num + ((blockIdx.x%valid_word_num) << 5) + (threadIdx4&31)) >> 2;
    
  float scale;
  float tmp;
  char4 tmp4;
  
  tmp4 = __ldg(data_ptr+data_id);
  
  scale = static_cast<float>(tmp4.x) * deQFactor;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.x = float_to_int8_rn(tmp*output_scale);

  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(tmp4.y) * deQFactor;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.y = float_to_int8_rn(tmp*output_scale);

  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(tmp4.z) * deQFactor;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.z = float_to_int8_rn(tmp*output_scale);

  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(tmp4.w) * deQFactor;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.w = float_to_int8_rn(tmp*output_scale);

  //row_id, col_id of sub-matrix (m = seq_len, n = size_per_head), column-major
  int row_id = word_id;
  int col_id = id_in_head;
  //new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len)
  int new_col = col_id >> 5;
  int new_row; 
  if (use_ORDER_COL32_2R_4R4)
  {
    int row_in_tile = row_id & 31;
    int col_in_tile = col_id & 31; 
    new_row = (qk_id != 1) ?
              //COL32
              ((row_id << 5) + (col_id&31))
            :
              //COL32_2R_4R4
              (
              ((row_id >> 5) << 10) +
              //(((row%8)/2*4+row/8)*2+row%2)*32+col
              (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
              )
              ;
  }
  else
  {
    new_row = (qk_id != 1) ?
              //COL32
              ((row_id << 5) + (col_id&31))
            :
              //COL4
              ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
              ////row_id%2 is even row, otherwise odd row
              ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
              (
              ((((row_id >> 3) << 3) + ((row_id&1) << 2) + ((col_id&31) >> 3)) << 5) +
              ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
              ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
              (((((col_id&7) >= 4)?4:0) + ((row_id&7) >> 1)) << 2) +
              ////col_id%4 is the id of 4 cols
              (col_id&3)
              )
              ;
  }

  buf_ptr4[(((batch_id*head_num + head_id) * stride + (new_col << 5)*seq_len + new_row) >> 2)] = tmp4;
}


//add_QK_bias_transform & rebuild padding for batch int8 cublasLtMatmul & per tensor quantization for weight
//1.add QK bias
//2.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = valid_word_num, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  seq_len_padded = (seq_len + 31)/32*32;
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len_padded, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4
//only for int8 IO
//seq_len, size_per_head must be a multiple of 32
//grid.x = valid_word_num * 2;
//block.x = head_num * size_per_head / 4;
//using char4
template <typename T>
__global__
void add_QK_bias_transform_rebuild_padding_varlen(int8_t *q_buf_, int8_t *k_buf_, const int8_t* Q, const T* bias_Q,
                                                  const int8_t* K, const T* bias_K, const int* sequence_id_offset,
                                                  const int valid_word_num, const int m, const int batch_size, 
                                                  const int seq_len, const int seq_len_padded, const int head_num,  
                                                  const int size_per_head, int stride_q, int stride_k,
                                                  const float *q_deQFactor_ptr,  const float *k_deQFactor_ptr,
                                                  const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                                  bool use_ORDER_COL32_2R_4R4)
{
  const char4* data_ptr;
  char4* buf_ptr4;
  const T* bias_ptr;
  int qk_id = blockIdx.x / valid_word_num;

  data_ptr = qk_id == 0 ? (const char4*)Q : (const char4*)K;
  buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
  bias_ptr = qk_id == 0 ? bias_Q : bias_K;

  int threadIdx4 = threadIdx.x << 2;
  int m_full_idx = blockIdx.x % valid_word_num;
  m_full_idx = (valid_word_num != m) ? (m_full_idx + __ldg(sequence_id_offset+m_full_idx)) : m_full_idx;
  int batch_id = m_full_idx / seq_len;
  int head_id = threadIdx4 / size_per_head;
  int id_in_head = threadIdx4 % size_per_head;
  int word_id = m_full_idx % seq_len;

  const float deQFactor = qk_id == 0 ? __ldg(q_deQFactor_ptr) : __ldg(k_deQFactor_ptr);
  const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

  int data_id = (((threadIdx4 >> 5) << 5)*valid_word_num + ((blockIdx.x%valid_word_num) << 5) + (threadIdx4&31)) >> 2;

  float scale;
  float tmp;
  char4 tmp4;

  tmp4 = __ldg(data_ptr+data_id);

  scale = static_cast<float>(tmp4.x) * deQFactor;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.x = float_to_int8_rn(tmp*output_scale);

  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(tmp4.y) * deQFactor;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.y = float_to_int8_rn(tmp*output_scale);

  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(tmp4.z) * deQFactor;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.z = float_to_int8_rn(tmp*output_scale);

  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(tmp4.w) * deQFactor;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.w = float_to_int8_rn(tmp*output_scale);

  //row_id, col_id of sub-matrix (m = seq_len or seq_len_padded, n = size_per_head), column-major
  int row_id = word_id;
  int col_id = id_in_head;
  //new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len) or (COL32_ * seq_len_padded)
  int new_col = col_id >> 5;
  int new_row;
  if (use_ORDER_COL32_2R_4R4)
  {
    int row_in_tile = row_id & 31;
    int col_in_tile = col_id & 31;
    new_row = (qk_id != 1) ?
              //COL32
              ((row_id << 5) + (col_id&31))
            :
              //COL32_2R_4R4
              (
              ((row_id >> 5) << 10) +
              //(((row%8)/2*4+row/8)*2+row%2)*32+col
              (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
              )
              ;
  }
  else
  {
    new_row = (qk_id != 1) ?
              //COL32
              ((row_id << 5) + (col_id&31))
            :
              //COL4
              ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
              ////row_id%2 is even row, otherwise odd row
              ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
              (
              ((((row_id >> 3) << 3) + ((row_id&1) << 2) + ((col_id&31) >> 3)) << 5) +
              ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
              ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
              (((((col_id&7) >= 4)?4:0) + ((row_id&7) >> 1)) << 2) +
              ////col_id%4 is the id of 4 cols
              (col_id&3)
              )
              ;
  }

  const int stride = (qk_id != 1) ? stride_q : stride_k;
  const int len = (qk_id != 1) ? seq_len : seq_len_padded;
  buf_ptr4[(((batch_id*head_num + head_id) * stride + (new_col << 5)*len + new_row) >> 2)] = tmp4;
}

template <typename T>
void add_QK_bias_transform_rebuild_padding_kernelLauncher(int8_t *q_buf, int8_t *k_buf, const int8_t* Q, const T* bias_Q, 
                                                          const int8_t* K, const T* bias_K, const int* sequence_id_offset, 
                                                          const int valid_word_num, 
                                                          const int batch_size, const int seq_len, 
                                                          const int head_num, const int size_per_head,  
                                                          const float *q_deQFactor_ptr,  const float *k_deQFactor_ptr, 
                                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream)
{
  int seq_len_padded = (seq_len + 31)/32*32;
  add_QK_bias_transform_rebuild_padding_varlen<<<dim3(valid_word_num*2), dim3((head_num*size_per_head)/4), 0, stream>>>(
    q_buf, k_buf, Q, bias_Q, K, bias_K, 
    sequence_id_offset, valid_word_num, 
    batch_size * seq_len, batch_size, 
    seq_len, seq_len_padded, head_num, size_per_head, 
    seq_len*size_per_head, seq_len_padded*size_per_head,
    q_deQFactor_ptr, k_deQFactor_ptr, 
    q_output_scale_ptr, k_output_scale_ptr, 
    use_ORDER_COL32_2R_4R4);
}

template
void add_QK_bias_transform_rebuild_padding_kernelLauncher(int8_t *q_buf, int8_t *k_buf, 
                                                          const int8_t* Q, const float* bias_Q, 
                                                          const int8_t* K, const float* bias_K, 
                                                          const int* sequence_id_offset, const int valid_word_num, 
                                                          const int batch_size, const int seq_len, 
                                                          const int head_num, const int size_per_head,  
                                                          const float *q_deQFactor_ptr,  const float *k_deQFactor_ptr, 
                                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);
                                                          
template
void add_QK_bias_transform_rebuild_padding_kernelLauncher(int8_t *q_buf, int8_t *k_buf, 
                                                          const int8_t* Q, const half* bias_Q, 
                                                          const int8_t* K, const half* bias_K, 
                                                          const int* sequence_id_offset, const int valid_word_num, 
                                                          const int batch_size, const int seq_len, 
                                                          const int head_num, const int size_per_head,  
                                                          const float *q_deQFactor_ptr,  const float *k_deQFactor_ptr, 
                                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);

//input matrix a matrix of m = batch_size*seq_len , n = head_num*size_per_head, CUBLASLT_ORDER_COL32
//output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len , CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4
//only for int32_t Input int8_t Output
//seq_len, size_per_head must be a multiple of 32
//grid = (size_per_head/32, seq_len/32, batch_size*head_num)
//block = (8, 32);
//using char4
//per axis quantization for weight
template <typename T>
__global__
void add_V_bias_transform(int8_t *v_buf_, const int32_t *V, const T *V_bias, const int batch_size, const int seq_len, 
                          const int head_num, const int size_per_head, int stride, const float* weight_amax, 
                          const float *input_deQFactor_div127_ptr, const float *out_scale_ptr, bool use_ORDER_COL32_2R_4R4)
{
  const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
  const float out_scale = __ldg(out_scale_ptr);
  __shared__ int8_t shm[32][33];
  const int32_t* data_ptr = V;
  char4* buf_ptr4 = (char4*) v_buf_;
  const T* bias_ptr = V_bias;

  int threadIdx4 = threadIdx.x << 2;

  //for src of (seq_len, size_per_head)
  int batch_id = blockIdx.z/head_num;
  int head_id = blockIdx.z%head_num;
  int word_id = (blockIdx.y << 5) + threadIdx.y;
  int id_in_size = (blockIdx.x << 5) + threadIdx4;

  //for V layout (batch_size*seq_len, head_num*size_per_head)
  int col = head_id*size_per_head + id_in_size;
  int row = batch_id*seq_len + word_id;
  int inIdx = (((col >> 5) << 5)*batch_size*seq_len + ((row << 5) + (col&31)));
  //for shm row-major
  int sh_col = threadIdx4;
  int sh_row = threadIdx.y;
  
  float tmp;
  float scale;

  //const half2* bias_ptr2 = (const half2*)bias_ptr;
  //half2 tmp2;

  //tmp2 = __ldg(&bias_ptr2[col >> 1]);
  
  scale = __ldg(data_ptr + inIdx) * __ldg(weight_amax + col) * input_deQFactor_div127;
  tmp = scale + static_cast<float>(__ldg(bias_ptr + col));//(tmp2.x);
  shm[sh_row][sh_col] = float_to_int8_rn(tmp*out_scale);
  
  scale = __ldg(data_ptr + inIdx + 1) * __ldg(weight_amax + col + 1) * input_deQFactor_div127;
  tmp = scale + static_cast<float>(__ldg(bias_ptr+col+1));//(tmp2.y);
  shm[sh_row][sh_col+1] = float_to_int8_rn(tmp*out_scale);
  
  //tmp2 = __ldg(&bias_ptr2[(col >> 1) + 1]);

  scale = __ldg(data_ptr+inIdx+2) * __ldg(weight_amax+col+2) * input_deQFactor_div127;
  tmp = scale + static_cast<float>(__ldg(bias_ptr+col+2));//(tmp2.x);
  shm[sh_row][sh_col+2] = float_to_int8_rn(tmp*out_scale);
  
  scale = __ldg(data_ptr+inIdx + 3) * __ldg(weight_amax+col+3) * input_deQFactor_div127;
  tmp = scale + static_cast<float>(__ldg(bias_ptr+col+3));//(tmp2.y);
  shm[sh_row][sh_col+3] = float_to_int8_rn(tmp*out_scale);

  __syncthreads();

  //for dst of (size_per_head, seq_len)
  word_id = (blockIdx.y << 5) + threadIdx4;
  id_in_size = (blockIdx.x << 5) + threadIdx.y;
  col = (word_id >> 5);

  if (use_ORDER_COL32_2R_4R4)
  {
    int row_in_tile = id_in_size & 31;
    int col_in_tile = word_id & 31; 
    row = (
          //COL32_2R_4R4
          ((id_in_size >> 5) << 10) +
          //(((row%8)/2*4+row/8)*2+row%2)*32+col
          (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
          );
  }
  else
  { 
    row = (
          //COL4
          ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
          ////id_in_size%2 is even row, otherwise odd row
          ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
          ((((id_in_size >> 3) << 3) + ((id_in_size&1) << 2) + ((word_id&31) >> 3)) << 5) +
          ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
          ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
          (((((word_id&7) >= 4)?4:0) + ((id_in_size&7) >> 1)) << 2) +
          ////word_id%4 is the id of 4 cols
          (word_id&3)
          );
  }

        
  char4 dataTmp;
  dataTmp.x = shm[sh_col][sh_row];
  dataTmp.y = shm[sh_col+1][sh_row];
  dataTmp.z = shm[sh_col+2][sh_row];
  dataTmp.w = shm[sh_col+3][sh_row];
  buf_ptr4[(blockIdx.z*stride + (col << 5)*size_per_head + row) >> 2] = dataTmp;
}

template <>
__global__
void add_V_bias_transform(int8_t *v_buf_, const int32_t *V, const half *V_bias, const int batch_size, const int seq_len, 
                          const int head_num, const int size_per_head, int stride, const float* weight_amax, 
                          const float *input_deQFactor_div127_ptr, const float *out_scale_ptr, bool use_ORDER_COL32_2R_4R4)
{
  const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
  const float out_scale = __ldg(out_scale_ptr);
  __shared__ int8_t shm[32][33];
  const int32_t* data_ptr = V;
  char4* buf_ptr4 = (char4*) v_buf_;

  int threadIdx4 = threadIdx.x << 2;

  //for src of (seq_len, size_per_head)
  int batch_id = blockIdx.z/head_num;
  int head_id = blockIdx.z%head_num;
  
  int blockIdy32 = (blockIdx.y << 5);
  int blockIdx32 = (blockIdx.x << 5);
  int word_id = blockIdy32 + threadIdx.y;
  int id_in_size = blockIdx32 + threadIdx4;

  //for V layout (batch_size*seq_len, head_num*size_per_head)
  int col = head_id*size_per_head + id_in_size;
  int row = batch_id*seq_len + word_id;
  int inIdx = ((col & 0xffffffe0)*batch_size*seq_len + ((row << 5) + (col&31)));
  //for shm row-major
  int sh_col = threadIdx4;
  int sh_row = threadIdx.y;
  
  int col_2 = col >> 1;
  float scale;

  const half2* bias_ptr2 = (const half2*)V_bias;
  half2 tmp2;

  tmp2 = __ldg(bias_ptr2+col_2);
  
  scale = __ldg(data_ptr+inIdx) * __ldg(weight_amax+col) * input_deQFactor_div127;
  scale = scale + static_cast<float>(tmp2.x);
  shm[sh_row][sh_col] = float_to_int8_rn(scale*out_scale);
  
  scale = __ldg(data_ptr+inIdx+1) * __ldg(weight_amax+col+1) * input_deQFactor_div127;
  scale = scale + static_cast<float>(tmp2.y);
  shm[sh_row][sh_col+1] = float_to_int8_rn(scale*out_scale);
  
  tmp2 = __ldg(bias_ptr2 + col_2 + 1);

  scale = __ldg(data_ptr + inIdx + 2) * __ldg(weight_amax + col + 2) * input_deQFactor_div127;
  scale = scale + static_cast<float>(tmp2.x);
  shm[sh_row][sh_col+2] = float_to_int8_rn(scale*out_scale);
  
  scale = __ldg(data_ptr + inIdx + 3) * __ldg(weight_amax + col + 3) * input_deQFactor_div127;
  scale = scale + static_cast<float>(tmp2.y);
  shm[sh_row][sh_col+3] = float_to_int8_rn(scale*out_scale);

  __syncthreads();

  //for dst of (size_per_head, seq_len)
  word_id = blockIdy32 + threadIdx4;
  id_in_size = blockIdx32 + threadIdx.y;
  col = (word_id >> 5);

  if (use_ORDER_COL32_2R_4R4)
  {
    int row_in_tile = id_in_size & 31;
    int col_in_tile = word_id & 31; 
    row = (
          //COL32_2R_4R4
          ((id_in_size >> 5) << 10) +
          //(((row%8)/2*4+row/8)*2+row%2)*32+col
          (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
          );
  }
  else
  { 
    row = (
          //COL4
          ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
          ////id_in_size%2 is even row, otherwise odd row
          ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
          (((id_in_size & 0xfffffff8) + ((id_in_size&1) << 2) + ((word_id&31) >> 3)) << 5) +
          ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
          ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
          (((((word_id&7) >= 4)?4:0) + ((id_in_size&7) >> 1)) << 2) +
          ////word_id%4 is the id of 4 cols
          (word_id&3)
          );
  }
        
  char4 dataTmp;
  dataTmp.x = shm[sh_col][sh_row];
  dataTmp.y = shm[sh_col+1][sh_row];
  dataTmp.z = shm[sh_col+2][sh_row];
  dataTmp.w = shm[sh_col+3][sh_row];
  buf_ptr4[(blockIdx.z*stride + (col << 5)*size_per_head + row) >> 2] = dataTmp;
}

template <typename T>
void add_V_bias_transform_kernelLauncher(int8_t *v_buf, const int32_t *V, const T *V_bias, 
                                         const int batch_size, const int seq_len, 
                                         const int head_num, const int size_per_head, 
                                         const float* weight_amax, 
                                         const float *input_deQFactor_div127_ptr, const float *out_scale_ptr, 
                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream)
{
  add_V_bias_transform<<<dim3(size_per_head/32, seq_len/32, batch_size*head_num), dim3(8, 32), 0, stream>>>(v_buf, V, V_bias, batch_size, seq_len, head_num, size_per_head, seq_len*size_per_head, weight_amax, input_deQFactor_div127_ptr, out_scale_ptr, use_ORDER_COL32_2R_4R4);
}

template 
void add_V_bias_transform_kernelLauncher(int8_t *v_buf, const int32_t *V, const float *V_bias, 
                                         const int batch_size, const int seq_len, 
                                         const int head_num, const int size_per_head, 
                                         const float* weight_amax, 
                                         const float *input_deQFactor_div127_ptr, const float *out_scale_ptr, 
                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);

template 
void add_V_bias_transform_kernelLauncher(int8_t *v_buf, const int32_t *V, const half *V_bias, 
                                         const int batch_size, const int seq_len, 
                                         const int head_num, const int size_per_head, 
                                         const float* weight_amax, 
                                         const float *input_deQFactor_div127_ptr, const float *out_scale_ptr, 
                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);

//input matrix a matrix of m = batch_size*seq_len , n = head_num*size_per_head, CUBLASLT_ORDER_COL32
//seq_len_padded = (seq_len+31)/32*32
//output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len_padded , CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4
//only for int8_t IO
//size_per_head must be a multiple of 32
//grid = (size_per_head/32, seq_len_padded/32, batch_size*head_num)
//block = (8, 32);
//using char4
//per tensor quantization for weight
template <typename T>
__global__
void add_V_bias_transform_varlen(int8_t *v_buf_, const int8_t *V, const T *V_bias, const int batch_size, const int seq_len, 
                          const int head_num, const int size_per_head, const int seq_len_padded, int stride,
                          const float *input_deQFactor_ptr, const float *out_scale_ptr, bool use_ORDER_COL32_2R_4R4)
{
  const float input_deQFactor = __ldg(input_deQFactor_ptr);
  const float out_scale = __ldg(out_scale_ptr);
  __shared__ int8_t shm[32][33];
  const char4* data_ptr = (const char4*)V;
  char4* buf_ptr4 = (char4*) v_buf_;
  const T* bias_ptr = V_bias;

  int threadIdx4 = threadIdx.x << 2;

  //for src of (seq_len, size_per_head)
  int batch_id = blockIdx.z/head_num;
  int head_id = blockIdx.z%head_num;
  int word_id = (blockIdx.y << 5) + threadIdx.y;
  int id_in_size = (blockIdx.x << 5) + threadIdx4;

  int col, row;
  //for shm row-major
  int sh_col = threadIdx4;
  int sh_row = threadIdx.y;
  char4 dataTmp;
  if (word_id < seq_len)
  {
    //for V layout (batch_size*seq_len, head_num*size_per_head)
    col = head_id*size_per_head + id_in_size;
    row = batch_id*seq_len + word_id;
    int inIdx = (((col >> 5) << 5)*batch_size*seq_len + ((row << 5) + (col&31))) >> 2;
  
    float tmp;
    float scale;
  
    dataTmp = __ldg(data_ptr + inIdx);
  
    scale = dataTmp.x * input_deQFactor;
    tmp = scale + static_cast<float>(__ldg(bias_ptr + col));//(tmp2.x);
    shm[sh_row][sh_col] = float_to_int8_rn(tmp*out_scale);
  
    scale = dataTmp.y * input_deQFactor;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+1));//(tmp2.y);
    shm[sh_row][sh_col+1] = float_to_int8_rn(tmp*out_scale);

    scale = dataTmp.z * input_deQFactor;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+2));//(tmp2.x);
    shm[sh_row][sh_col+2] = float_to_int8_rn(tmp*out_scale);
  
    scale = dataTmp.w * input_deQFactor;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+3));//(tmp2.y);
    shm[sh_row][sh_col+3] = float_to_int8_rn(tmp*out_scale);
  }
  else
  {
    shm[sh_row][sh_col] = shm[sh_row][sh_col+1] = shm[sh_row][sh_col+2] = shm[sh_row][sh_col+3] = 0;
  }

  __syncthreads();

  //for dst of (size_per_head, seq_len_padded)
  word_id = (blockIdx.y << 5) + threadIdx4;
  id_in_size = (blockIdx.x << 5) + threadIdx.y;
  col = (word_id >> 5);

  if (use_ORDER_COL32_2R_4R4)
  {
    int row_in_tile = id_in_size & 31;
    int col_in_tile = word_id & 31; 
    row = (
          //COL32_2R_4R4
          ((id_in_size >> 5) << 10) +
          //(((row%8)/2*4+row/8)*2+row%2)*32+col
          (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
          );
  }
  else
  { 
    row = (
          //COL4
          ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
          ////id_in_size%2 is even row, otherwise odd row
          ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
          ((((id_in_size >> 3) << 3) + ((id_in_size&1) << 2) + ((word_id&31) >> 3)) << 5) +
          ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
          ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
          (((((word_id&7) >= 4)?4:0) + ((id_in_size&7) >> 1)) << 2) +
          ////word_id%4 is the id of 4 cols
          (word_id&3)
          );
  }

  dataTmp.x = shm[sh_col][sh_row];
  dataTmp.y = shm[sh_col+1][sh_row];
  dataTmp.z = shm[sh_col+2][sh_row];
  dataTmp.w = shm[sh_col+3][sh_row];
  buf_ptr4[(blockIdx.z*stride + (col << 5)*size_per_head + row) >> 2] = dataTmp;
}

template <typename T>
void add_V_bias_transform_kernelLauncher(int8_t *v_buf, const int8_t *V, const T *V_bias, const int batch_size, 
                                         const int seq_len, const int head_num, const int size_per_head,
                                         const float *input_deQFactor_ptr, const float *out_scale_ptr, 
                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream)
{
  assert(size_per_head % 32 == 0);
  if (seq_len % 32 == 0)
  {
    add_V_bias_transform_varlen<<<dim3(size_per_head/32, seq_len/32, batch_size*head_num), dim3(8, 32), 0, stream>>>(
      v_buf, V, V_bias, 
      batch_size, seq_len, head_num, size_per_head, 
      seq_len, seq_len*size_per_head,
      input_deQFactor_ptr, out_scale_ptr, use_ORDER_COL32_2R_4R4);
  }
  else
  {
    const int seq_len_padded = (seq_len + 31)/32*32;
    add_V_bias_transform_varlen<<<dim3(size_per_head/32, seq_len_padded/32, batch_size*head_num), dim3(8, 32), 0, stream>>>(
      v_buf, V, V_bias, 
      batch_size, seq_len, head_num, size_per_head, 
      seq_len_padded, seq_len_padded*size_per_head,
      input_deQFactor_ptr, out_scale_ptr, use_ORDER_COL32_2R_4R4);
  }
}                

template
void add_V_bias_transform_kernelLauncher(int8_t *v_buf, const int8_t *V, const float *V_bias, const int batch_size, 
                                         const int seq_len, const int head_num, const int size_per_head,
                                         const float *input_deQFactor_ptr, const float *out_scale_ptr, 
                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream); 

template
void add_V_bias_transform_kernelLauncher(int8_t *v_buf, const int8_t *V, const half *V_bias, const int batch_size, 
                                         const int seq_len, const int head_num, const int size_per_head,
                                         const float *input_deQFactor_ptr, const float *out_scale_ptr, 
                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);                                          

//add bias into V & rebuild padding 
//input matrix a matrix of m = valid_word_num, n = head_num*size_per_head, CUBLASLT_ORDER_COL32
//output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len , CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4
//only for int32_t Input int8_t Output
//seq_len, size_per_head must be a multiple of 32
//grid = (size_per_head/32, seq_len/32, batch_size*head_num)
//block = (8, 32);
//using char4
//per axis quantization for weight
template <typename T>
__global__
void add_V_bias_transform_rebuild_padding(int8_t *v_buf_, const int32_t *V, const T *V_bias, const int* sequence_id_map, const int valid_word_num, 
                                          const int batch_size, const int seq_len, const int head_num, const int size_per_head, int stride, 
                                          const float* weight_amax, const float *input_deQFactor_div127_ptr, const float *out_scale_ptr, bool use_ORDER_COL32_2R_4R4)
{
  __shared__ int8_t shm[32][33];
  const int32_t* data_ptr = V;
  char4* buf_ptr4 = (char4*) v_buf_;
  const T* bias_ptr = V_bias;

  int threadIdx4 = threadIdx.x << 2;

  //for src of (seq_len, size_per_head)
  int batch_id = blockIdx.z/head_num;
  int head_id = blockIdx.z%head_num;
  int word_id = (blockIdx.y << 5) + threadIdx.y;
  int id_in_size = (blockIdx.x << 5) + threadIdx4;

  //for shm row-major
  int sh_col = threadIdx4;
  int sh_row = threadIdx.y;
  
  //for V layout (batch_size*seq_len, head_num*size_per_head)
  int col;
  int row = __ldg(sequence_id_map + batch_id*seq_len + word_id);
  
  if (row != -1){
    col = head_id*size_per_head + id_in_size;  
    int inIdx = ((col & 0xffffffe0)*valid_word_num + ((row << 5) + (col&31)));
  
    float tmp;
    float scale;
  
    const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
    const float out_scale = __ldg(out_scale_ptr);
  
    scale = __ldg(data_ptr + inIdx) * __ldg(weight_amax + col) * input_deQFactor_div127;
    tmp = scale + static_cast<float>(__ldg(bias_ptr + col));
    shm[sh_row][sh_col] = float_to_int8_rn(tmp*out_scale);
  
    scale = __ldg(data_ptr + inIdx + 1) * __ldg(weight_amax + col + 1) * input_deQFactor_div127;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+1));
    shm[sh_row][sh_col+1] = float_to_int8_rn(tmp*out_scale);

    scale = __ldg(data_ptr+inIdx+2) * __ldg(weight_amax+col+2) * input_deQFactor_div127;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+2));
    shm[sh_row][sh_col+2] = float_to_int8_rn(tmp*out_scale);
  
    scale = __ldg(data_ptr+inIdx + 3) * __ldg(weight_amax+col+3) * input_deQFactor_div127;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+3));
    shm[sh_row][sh_col+3] = float_to_int8_rn(tmp*out_scale);
  }
  else{
    shm[sh_row][sh_col] = shm[sh_row][sh_col + 1] = shm[sh_row][sh_col + 2] = shm[sh_row][sh_col + 3] = 0;
  }
  __syncthreads();

  char4 dataTmp;
  dataTmp.x = shm[sh_col][sh_row];
  dataTmp.y = shm[sh_col+1][sh_row];
  dataTmp.z = shm[sh_col+2][sh_row];
  dataTmp.w = shm[sh_col+3][sh_row];

  //for dst of (size_per_head, seq_len)
  word_id = (blockIdx.y << 5) + threadIdx4;
  id_in_size = (blockIdx.x << 5) + threadIdx.y;
  col = (word_id >> 5);
  
  if (use_ORDER_COL32_2R_4R4)
  {
    int row_in_tile = id_in_size & 31;
    int col_in_tile = word_id & 31; 
    row = (
          //COL32_2R_4R4
          ((id_in_size >> 5) << 10) +
          //(((row%8)/2*4+row/8)*2+row%2)*32+col
          (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
          ); 
  }
  else
  {
    row = (
          //COL4
          ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
          ////id_in_size%2 is even row, otherwise odd row
          ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
          (((id_in_size & 0xfffffff8) + ((id_in_size&1) << 2) + ((word_id&31) >> 3)) << 5) +
          ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
          ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
          (((((word_id&7) >= 4)?4:0) + ((id_in_size&7) >> 1)) << 2) +
          ////word_id%4 is the id of 4 cols
          (word_id&3)
          );
  }
        
  buf_ptr4[(blockIdx.z*stride + (col << 5)*size_per_head + row) >> 2] = dataTmp;
}

template <>
__global__
void add_V_bias_transform_rebuild_padding(int8_t *v_buf_, const int32_t *V, const half *V_bias, const int* sequence_id_map, const int valid_word_num, 
                                          const int batch_size, const int seq_len, const int head_num, const int size_per_head, int stride, 
                                          const float* weight_amax, const float *input_deQFactor_div127_ptr, const float *out_scale_ptr, bool use_ORDER_COL32_2R_4R4)
{
  __shared__ int8_t shm[32][33];
  const int32_t* data_ptr = V;
  char4* buf_ptr4 = (char4*) v_buf_;

  int threadIdx4 = threadIdx.x << 2;

  //for src of (seq_len, size_per_head)
  int batch_id = blockIdx.z/head_num;
  int head_id = blockIdx.z%head_num;
  
  int blockIdy32 = (blockIdx.y << 5);
  int blockIdx32 = (blockIdx.x << 5);
  int word_id = blockIdy32 + threadIdx.y;
  int id_in_size = blockIdx32 + threadIdx4;

  
  //for shm row-major
  int sh_col = threadIdx4;
  int sh_row = threadIdx.y;
  
  //for V layout (batch_size*seq_len, head_num*size_per_head)
  int col;
  int row = __ldg(sequence_id_map + batch_id*seq_len + word_id);
  
  if (row >= 0){
    const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
    const float out_scale = __ldg(out_scale_ptr);
    col = head_id*size_per_head + id_in_size;
    int inIdx = ((col & 0xffffffe0)*valid_word_num + ((row << 5) + (col&31)));
    int col_2 = col >> 1;
    float scale;

    const half2* bias_ptr2 = (const half2*)V_bias;
    half2 tmp2;

    tmp2 = __ldg(bias_ptr2+col_2);
  
    scale = __ldg(data_ptr+inIdx) * __ldg(weight_amax+col) * input_deQFactor_div127;
    scale = scale + static_cast<float>(tmp2.x);
    shm[sh_row][sh_col] = float_to_int8_rn(scale*out_scale);
  
    scale = __ldg(data_ptr+inIdx+1) * __ldg(weight_amax+col+1) * input_deQFactor_div127;
    scale = scale + static_cast<float>(tmp2.y);
    shm[sh_row][sh_col+1] = float_to_int8_rn(scale*out_scale);
  
    tmp2 = __ldg(bias_ptr2 + col_2 + 1);

    scale = __ldg(data_ptr + inIdx + 2) * __ldg(weight_amax + col + 2) * input_deQFactor_div127;
    scale = scale + static_cast<float>(tmp2.x);
    shm[sh_row][sh_col+2] = float_to_int8_rn(scale*out_scale);
  
    scale = __ldg(data_ptr + inIdx + 3) * __ldg(weight_amax + col + 3) * input_deQFactor_div127;
    scale = scale + static_cast<float>(tmp2.y);
    shm[sh_row][sh_col+3] = float_to_int8_rn(scale*out_scale);
  }
  else{
    shm[sh_row][sh_col] = shm[sh_row][sh_col + 1] = shm[sh_row][sh_col + 2] = shm[sh_row][sh_col + 3] = 0;
  }
  __syncthreads();

  char4 dataTmp;
  dataTmp.x = shm[sh_col][sh_row];
  dataTmp.y = shm[sh_col+1][sh_row];
  dataTmp.z = shm[sh_col+2][sh_row];
  dataTmp.w = shm[sh_col+3][sh_row];

  //for dst of (size_per_head, seq_len)
  word_id = blockIdy32 + threadIdx4;
  id_in_size = blockIdx32 + threadIdx.y;
  col = (word_id >> 5);
  
  if (use_ORDER_COL32_2R_4R4)
  {
    int row_in_tile = id_in_size & 31;
    int col_in_tile = word_id & 31; 
    row = (
          //COL32_2R_4R4
          ((id_in_size >> 5) << 10) +
          //(((row%8)/2*4+row/8)*2+row%2)*32+col
          (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
          ); 
  }
  else
  {
    row = (
          //COL4
          ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
          ////id_in_size%2 is even row, otherwise odd row
          ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
          (((id_in_size & 0xfffffff8) + ((id_in_size&1) << 2) + ((word_id&31) >> 3)) << 5) +
          ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
          ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
          (((((word_id&7) >= 4)?4:0) + ((id_in_size&7) >> 1)) << 2) +
          ////word_id%4 is the id of 4 cols
          (word_id&3)
          );
  }
        
  buf_ptr4[(blockIdx.z*stride + (col << 5)*size_per_head + row) >> 2] = dataTmp;
}

template <typename T>
void add_V_bias_transform_rebuild_padding_kernelLauncher(int8_t *v_buf, const int32_t *V, const T *V_bias, 
                                                         const int* sequence_id_map, const int valid_word_num, 
                                                         const int batch_size, const int seq_len, 
                                                         const int head_num, const int size_per_head, 
                                                         const float* weight_amax, 
                                                         const float *input_deQFactor_div127_ptr, 
                                                         const float *out_scale_ptr, 
                                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream)
{
  add_V_bias_transform_rebuild_padding<<<dim3(size_per_head/32, seq_len/32, batch_size*head_num), dim3(8, 32), 0, stream>>>(
    v_buf, V, V_bias, 
    sequence_id_map, valid_word_num, 
    batch_size, seq_len, 
    head_num, size_per_head, 
    seq_len*size_per_head, 
    weight_amax, input_deQFactor_div127_ptr,
    out_scale_ptr, use_ORDER_COL32_2R_4R4);
}     

template 
void add_V_bias_transform_rebuild_padding_kernelLauncher(int8_t *v_buf, const int32_t *V, const float *V_bias, 
                                                         const int* sequence_id_map, const int valid_word_num, 
                                                         const int batch_size, const int seq_len, 
                                                         const int head_num, const int size_per_head, 
                                                         const float* weight_amax, 
                                                         const float *input_deQFactor_div127_ptr, 
                                                         const float *out_scale_ptr, 
                                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);
                                                         
template 
void add_V_bias_transform_rebuild_padding_kernelLauncher(int8_t *v_buf, const int32_t *V, const half *V_bias, 
                                                         const int* sequence_id_map, const int valid_word_num, 
                                                         const int batch_size, const int seq_len, 
                                                         const int head_num, const int size_per_head, 
                                                         const float* weight_amax, 
                                                         const float *input_deQFactor_div127_ptr, 
                                                         const float *out_scale_ptr, 
                                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);

//add bias into V & rebuild padding 
//input matrix a matrix of m = valid_word_num, n = head_num*size_per_head, CUBLASLT_ORDER_COL32
//output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len , CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4
//only for int8_t IO
//seq_len, size_per_head must be a multiple of 32
//grid = (size_per_head/32, seq_len/32, batch_size*head_num)
//block = (8, 32);
//using char4
//per tensor quantization for weight
template <typename T>
__global__
void add_V_bias_transform_rebuild_padding(int8_t *v_buf_, const int8_t *V, const T *V_bias, const int* sequence_id_map, const int valid_word_num, 
                                          const int batch_size, const int seq_len, const int head_num, const int size_per_head, int stride, 
                                          const float *deQFactor_ptr, const float *out_scale_ptr, bool use_ORDER_COL32_2R_4R4)
{
  __shared__ int8_t shm[32][33];
  const char4* data_ptr = (const char4*)V;
  char4* buf_ptr4 = (char4*) v_buf_;
  const T* bias_ptr = V_bias;

  int threadIdx4 = threadIdx.x << 2;

  //for src of (seq_len, size_per_head)
  int batch_id = blockIdx.z/head_num;
  int head_id = blockIdx.z%head_num;
  int word_id = (blockIdx.y << 5) + threadIdx.y;
  int id_in_size = (blockIdx.x << 5) + threadIdx4;

  //for shm row-major
  int sh_col = threadIdx4;
  int sh_row = threadIdx.y;
  
  //for V layout (batch_size*seq_len, head_num*size_per_head)
  int col;
  int row = __ldg(sequence_id_map + batch_id*seq_len + word_id);
  
  if (row != -1){
    col = head_id*size_per_head + id_in_size;  
    int inIdx = ((col & 0xffffffe0)*valid_word_num + ((row << 5) + (col&31))) >> 2;
  
    float tmp;
    float scale;
  
    const float deQFactor = __ldg(deQFactor_ptr);
    const float out_scale = __ldg(out_scale_ptr);
  
    char4 dataTmp = __ldg(data_ptr + inIdx);
  
    scale = dataTmp.x * deQFactor;
    tmp = scale + static_cast<float>(__ldg(bias_ptr + col));
    shm[sh_row][sh_col] = float_to_int8_rn(tmp*out_scale);
  
    scale = dataTmp.y * deQFactor;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+1));
    shm[sh_row][sh_col+1] = float_to_int8_rn(tmp*out_scale);

    scale = dataTmp.z * deQFactor;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+2));
    shm[sh_row][sh_col+2] = float_to_int8_rn(tmp*out_scale);
  
    scale = dataTmp.w * deQFactor;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+3));
    shm[sh_row][sh_col+3] = float_to_int8_rn(tmp*out_scale);
  }
  else{
    shm[sh_row][sh_col] = shm[sh_row][sh_col + 1] = shm[sh_row][sh_col + 2] = shm[sh_row][sh_col + 3] = 0;
  }
  __syncthreads();

  char4 dataTmp;  
  dataTmp.x = shm[sh_col][sh_row];
  dataTmp.y = shm[sh_col+1][sh_row];
  dataTmp.z = shm[sh_col+2][sh_row];
  dataTmp.w = shm[sh_col+3][sh_row];

  //for dst of (size_per_head, seq_len)
  word_id = (blockIdx.y << 5) + threadIdx4;
  id_in_size = (blockIdx.x << 5) + threadIdx.y;
  col = (word_id >> 5);
  
  if (use_ORDER_COL32_2R_4R4)
  {
    int row_in_tile = id_in_size & 31;
    int col_in_tile = word_id & 31; 
    row = (
          //COL32_2R_4R4
          ((id_in_size >> 5) << 10) +
          //(((row%8)/2*4+row/8)*2+row%2)*32+col
          (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
          ); 
  }
  else
  {
    row = (
          //COL4
          ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
          ////id_in_size%2 is even row, otherwise odd row
          ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
          (((id_in_size & 0xfffffff8) + ((id_in_size&1) << 2) + ((word_id&31) >> 3)) << 5) +
          ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
          ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
          (((((word_id&7) >= 4)?4:0) + ((id_in_size&7) >> 1)) << 2) +
          ////word_id%4 is the id of 4 cols
          (word_id&3)
          );
  }
        
  buf_ptr4[(blockIdx.z*stride + (col << 5)*size_per_head + row) >> 2] = dataTmp;
}

//add bias into V & rebuild padding
//input matrix a matrix of m = valid_word_num, n = head_num*size_per_head, CUBLASLT_ORDER_COL32
//output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len_padded , CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4
//only for int8_t IO
//seq_len, size_per_head must be a multiple of 32
//grid = (size_per_head/32, seq_len_padded/32, batch_size*head_num)
//block = (8, 32);
//using char4
//per tensor quantization for weight
template <typename T>
__global__
void add_V_bias_transform_rebuild_padding_varlen(int8_t *v_buf_, const int8_t *V, const T *V_bias, const int* sequence_id_map, const int valid_word_num,
                                                 const int batch_size, const int seq_len, const int seq_len_padded, 
                                                 const int head_num, const int size_per_head, int stride,
                                                 const float *deQFactor_ptr, const float *out_scale_ptr, bool use_ORDER_COL32_2R_4R4)
{
  __shared__ int8_t shm[32][33];
  const char4* data_ptr = (const char4*)V;
  char4* buf_ptr4 = (char4*) v_buf_;
  const T* bias_ptr = V_bias;

  int threadIdx4 = threadIdx.x << 2;

  //for src of (seq_len, size_per_head)
  int batch_id = blockIdx.z/head_num;
  int head_id = blockIdx.z%head_num;
  int word_id = (blockIdx.y << 5) + threadIdx.y;
  int id_in_size = (blockIdx.x << 5) + threadIdx4;

  //for shm row-major
  int sh_col = threadIdx4;
  int sh_row = threadIdx.y;

  //for V layout (batch_size*seq_len, head_num*size_per_head)
  int col;
  int row = word_id < seq_len ? __ldg(sequence_id_map + batch_id*seq_len + word_id) : -1;

  if (row != -1){
    col = head_id*size_per_head + id_in_size;
    int inIdx = ((col & 0xffffffe0)*valid_word_num + ((row << 5) + (col&31))) >> 2;

    float tmp;
    float scale;

    const float deQFactor = __ldg(deQFactor_ptr);
    const float out_scale = __ldg(out_scale_ptr);

    char4 dataTmp = __ldg(data_ptr + inIdx);

    scale = dataTmp.x * deQFactor;
    tmp = scale + static_cast<float>(__ldg(bias_ptr + col));
    shm[sh_row][sh_col] = float_to_int8_rn(tmp*out_scale);

    scale = dataTmp.y * deQFactor;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+1));
    shm[sh_row][sh_col+1] = float_to_int8_rn(tmp*out_scale);

    scale = dataTmp.z * deQFactor;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+2));
    shm[sh_row][sh_col+2] = float_to_int8_rn(tmp*out_scale);

    scale = dataTmp.w * deQFactor;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+3));
    shm[sh_row][sh_col+3] = float_to_int8_rn(tmp*out_scale);
  }
  else{
    shm[sh_row][sh_col] = shm[sh_row][sh_col + 1] = shm[sh_row][sh_col + 2] = shm[sh_row][sh_col + 3] = 0;
  }
  __syncthreads();

  char4 dataTmp;
  dataTmp.x = shm[sh_col][sh_row];
  dataTmp.y = shm[sh_col+1][sh_row];
  dataTmp.z = shm[sh_col+2][sh_row];
  dataTmp.w = shm[sh_col+3][sh_row];

  //for dst of (size_per_head, seq_len_padded)
  word_id = (blockIdx.y << 5) + threadIdx4;
  id_in_size = (blockIdx.x << 5) + threadIdx.y;
  col = (word_id >> 5);

  if (use_ORDER_COL32_2R_4R4)
  {
    int row_in_tile = id_in_size & 31;
    int col_in_tile = word_id & 31;
    row = (
          //COL32_2R_4R4
          ((id_in_size >> 5) << 10) +
          //(((row%8)/2*4+row/8)*2+row%2)*32+col
          (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
          );
  }
  else
  {
    row = (
          //COL4
          ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
          ////id_in_size%2 is even row, otherwise odd row
          ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
          (((id_in_size & 0xfffffff8) + ((id_in_size&1) << 2) + ((word_id&31) >> 3)) << 5) +
          ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
          ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
          (((((word_id&7) >= 4)?4:0) + ((id_in_size&7) >> 1)) << 2) +
          ////word_id%4 is the id of 4 cols
          (word_id&3)
          );
  }

  buf_ptr4[(blockIdx.z*stride + (col << 5)*size_per_head + row) >> 2] = dataTmp;
}


template <typename T>
void add_V_bias_transform_rebuild_padding_kernelLauncher(int8_t *v_buf, const int8_t *V, const T *V_bias, 
                                                         const int* sequence_id_map, const int valid_word_num, 
                                                         const int batch_size, const int seq_len, 
                                                         const int head_num, const int size_per_head,
                                                         const float *deQFactor_ptr, const float *out_scale_ptr, 
                                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream)
{
  int seq_len_padded = (seq_len + 31)/32*32;
  add_V_bias_transform_rebuild_padding_varlen<<<dim3(size_per_head/32, seq_len_padded/32, batch_size*head_num), dim3(8, 32), 0, stream>>>(
    v_buf, V, V_bias, sequence_id_map, valid_word_num, 
    batch_size, seq_len, seq_len_padded, head_num, size_per_head, seq_len_padded*size_per_head, 
    deQFactor_ptr, out_scale_ptr, use_ORDER_COL32_2R_4R4);
}           

template
void add_V_bias_transform_rebuild_padding_kernelLauncher(int8_t *v_buf, const int8_t *V, const float *V_bias, 
                                                         const int* sequence_id_map, const int valid_word_num, 
                                                         const int batch_size, const int seq_len, 
                                                         const int head_num, const int size_per_head,
                                                         const float *deQFactor_ptr, const float *out_scale_ptr, 
                                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);
                                                         
template
void add_V_bias_transform_rebuild_padding_kernelLauncher(int8_t *v_buf, const int8_t *V, const half *V_bias, 
                                                         const int* sequence_id_map, const int valid_word_num, 
                                                         const int batch_size, const int seq_len, 
                                                         const int head_num, const int size_per_head,
                                                         const float *deQFactor_ptr, const float *out_scale_ptr, 
                                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream); 

__global__
void trt_add_QKV_bias(half2* Q, const half2* bias_Q, half2* K, const half2* bias_K, half2* V, const half2* bias_V, 
  half2* q_buf_, half2* k_buf_, half2* v_buf_, 
  const int valid_word_num, const int head_num, const int size_per_head)
{
  // Add bias, and then transpose from 
  // [3, valid_word_num, head, size] -> [valid_word_num, head, 3, size]
  
  // const int seq_id = blockIdx.x % valid_word_num;
  // const int qkv_id = (blockIdx.x - seq_id) / valid_word_num;
  const int seq_id = blockIdx.x;
  const int size_id = threadIdx.x % size_per_head;
  const int head_id = (threadIdx.x - size_id) / size_per_head;

  const int target_offset = blockIdx.x * head_num * 3 * size_per_head + head_id * 3 * size_per_head;

  q_buf_[ target_offset + 
          0 * size_per_head +
          size_id] = Q[ seq_id * blockDim.x + threadIdx.x] + bias_Q[threadIdx.x];

  q_buf_[ target_offset + 
          1 * size_per_head +
          size_id] = K[ seq_id * blockDim.x + threadIdx.x] + bias_K[threadIdx.x];

  q_buf_[ target_offset + 
          2 * size_per_head +
          size_id] = V[ seq_id * blockDim.x + threadIdx.x] + bias_V[threadIdx.x];
}

template<OperationType OpType_>
void OpenMultiHeadAttention<OpType_>::trt_add_QKV_bias_kernelLauncher(
  const DataType_* bias_Q,
  const DataType_* bias_K,
  const DataType_* bias_V)
{
  dim3 grid;
  dim3 block;

  grid.x = param_.valid_word_num;
  block.x = head_num_ * size_per_head_ / 2;

  assert(block.x <= 1024);

  trt_add_QKV_bias<<<grid, block, 0, param_.stream>>>((half2*)query_buf_, (const half2*)bias_Q, 
                                                      (half2*)key_buf_, (const half2*)bias_K, 
                                                      (half2*)value_buf_, (const half2*)bias_V, 
                                                      (half2*)q_buf_, (half2*)k_buf_, (half2*)v_buf_,
                                                      param_.valid_word_num, 
                                                      head_num_, size_per_head_ / 2);
}

// add bias and then transform from 
// 3 * ([valid_word_num, head*size] + CUBLASLT_ORDER_COL32) -> [valid_word_num, head, 3, size] + row-major
// input is INT32 && per axis quantization for weight
// output is INT8 && per tensor quantization
// grid((head*size + 31)/32, (valid_word_num + 31)/32, 3)
// block(8, 32)
// size should be a multiple of 4
//using char4 as output, int4 as input
template <typename T>
__global__
void trt_add_QKV_bias_COL32_int32IInt8O(char4* output, const int4* QKV,
                                        const T* bias_Q, const T* bias_K, const T* bias_V, 
                                        const float *input_deQFactor_div127_ptr,
                                        const float *q_weight_amax,  
                                        const float *k_weight_amax,
                                        const float *v_weight_amax,
                                        const float qkv_output_scale, const int valid_word_num, const int head_num, 
                                        const int size_per_head, const int head_num_x_size_per_head)
{
  const int qkv_id = blockIdx.z;
  const int seq_id = (blockIdx.y << 5) + threadIdx.y;
  const int threadIdx4 = threadIdx.x << 2;
  int hidden_id = (blockIdx.x << 5) + threadIdx4;
  const int size_id = hidden_id % size_per_head;
  const int head_id = hidden_id / size_per_head;
  
  const bool qual = (seq_id < valid_word_num) && (hidden_id < head_num_x_size_per_head);
  if (qual)
  {
    const float* weight_amax = qkv_id == 0 ? q_weight_amax : (qkv_id == 1 ? k_weight_amax : v_weight_amax);
    const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr); 
  
    const T* bias_ptr = (qkv_id == 0) ? bias_Q : ((qkv_id == 1) ? bias_K : bias_V);
  
    const int input_id = (qkv_id * valid_word_num * head_num_x_size_per_head + ((hidden_id & 0xffffffe0)*valid_word_num + (seq_id << 5) + (hidden_id&31))) >> 2;
    
    char4 tmp;
    const int4 tmp_int4 = __ldg(QKV+input_id);
    
    tmp.x = float_to_int8_rn((static_cast<float>(tmp_int4.x) * __ldg(weight_amax+hidden_id) * input_deQFactor_div127 + static_cast<float>(__ldg(bias_ptr + hidden_id))) * qkv_output_scale);
    
    hidden_id += 1;
    tmp.y = float_to_int8_rn((static_cast<float>(tmp_int4.y) * __ldg(weight_amax+hidden_id) * input_deQFactor_div127 + static_cast<float>(__ldg(bias_ptr + hidden_id))) * qkv_output_scale);
    
    hidden_id += 1;
    tmp.z = float_to_int8_rn((static_cast<float>(tmp_int4.z) * __ldg(weight_amax+hidden_id) * input_deQFactor_div127 + static_cast<float>(__ldg(bias_ptr + hidden_id))) * qkv_output_scale);
    
    hidden_id += 1;
    tmp.w = float_to_int8_rn((static_cast<float>(tmp_int4.w) * __ldg(weight_amax+hidden_id) * input_deQFactor_div127 + static_cast<float>(__ldg(bias_ptr + hidden_id))) * qkv_output_scale);

    //const int output_id = (seq_id * 3 * head_num_x_size_per_head + head_id * 3 * size_per_head + qkv_id * size_per_head + size_id) >> 2;
    const int output_id = ((seq_id * head_num_x_size_per_head + head_id * size_per_head) * 3 + qkv_id * size_per_head + size_id) >> 2;    
    
    output[output_id] = tmp;
  }
}

template<OperationType OpType_>
void OpenMultiHeadAttention<OpType_>::trt_add_QKV_bias_COL32_int32Iint8O_kernelLauncher(
  int8_t* output,
  const int32_t* Q,
  const DataType_* bias_Q,
  const DataType_* bias_K,
  const DataType_* bias_V,
  const float *input_deQFactor_div127_ptr,
  const float * q_weight_amax,  
  const float * k_weight_amax,
  const float * v_weight_amax,
  const float qkv_output_scale)
{
  int head_num_x_size_per_head = head_num_*size_per_head_;
  dim3 grid((head_num_x_size_per_head + 31)/32, (param_.valid_word_num + 31)/32, 3);
  dim3 block(8, 32);

  assert(size_per_head_ % 4 == 0);

  trt_add_QKV_bias_COL32_int32IInt8O<<<grid, block, 0, param_.stream>>>((char4*)output, (const int4*)Q,
                                                                   bias_Q, bias_K, bias_V,
                                                                   input_deQFactor_div127_ptr,
                                                                   q_weight_amax, 
                                                                   k_weight_amax,
                                                                   v_weight_amax,
                                                                   qkv_output_scale, param_.valid_word_num, 
                                                                   head_num_, size_per_head_, head_num_x_size_per_head);
}

// Add bias, and then transform from 
// 3 * ([valid_word_num, head*size] + CUBLASLT_ORDER_COL32) -> [valid_word_num, head, 3, size] + row-major
// grid((head*size + 31)/32, (valid_word_num + 31)/32, 3)
// block(8, 32)
// size should be a multiple of 4
template <typename T>
__global__
void trt_add_QKV_bias_COL32_int8IO(char4* output, const char4* QKV,  
                                   const T* bias_Q, const T* bias_K, const T* bias_V, 
                                   const float *q_input_deQFactor_ptr, const float *k_input_deQFactor_ptr, 
                                   const float *v_input_deQFactor_ptr, const float qkv_output_scale,
                                   const int valid_word_num, const int head_num, const int size_per_head,
                                   const int head_num_x_size_per_head)
{
  const int qkv_id = blockIdx.z;
  const int seq_id = (blockIdx.y << 5) + threadIdx.y;
  const int threadIdx4 = threadIdx.x << 2;
  const int hidden_id = (blockIdx.x << 5) + threadIdx4;
  const int size_id = hidden_id % size_per_head;
  const int head_id = hidden_id / size_per_head;
  
  const bool qual = (seq_id < valid_word_num) && (hidden_id < head_num_x_size_per_head);
  if (qual)
  {
    const float *input_deQFactor_ptr = (qkv_id == 0) ? q_input_deQFactor_ptr : ((qkv_id == 1) ? k_input_deQFactor_ptr : v_input_deQFactor_ptr);
    const float input_deQFactor = __ldg(input_deQFactor_ptr);

    const T* bias_ptr = (qkv_id == 0) ? bias_Q : ((qkv_id == 1) ? bias_K : bias_V);
  
    const int input_id = (qkv_id * valid_word_num * head_num_x_size_per_head + ((hidden_id & 0xffffffe0)*valid_word_num + (seq_id << 5) + (hidden_id&31))) >> 2;
    
    char4 tmp = __ldg(QKV+input_id);
    
    tmp.x = float_to_int8_rn((static_cast<float>(tmp.x) * input_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id))) * qkv_output_scale);
    
    tmp.y = float_to_int8_rn((static_cast<float>(tmp.y) * input_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id + 1))) * qkv_output_scale);
    
    tmp.z = float_to_int8_rn((static_cast<float>(tmp.z) * input_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id + 2))) * qkv_output_scale);
    
    tmp.w = float_to_int8_rn((static_cast<float>(tmp.w) * input_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id + 3))) * qkv_output_scale);

    //const int output_id = (seq_id * 3 * head_num_x_size_per_head + head_id * 3 * size_per_head + qkv_id * size_per_head + size_id) >> 2;
    const int output_id = ((seq_id * head_num_x_size_per_head + head_id * size_per_head) * 3 + qkv_id * size_per_head + size_id) >> 2;    
    
    output[output_id] = tmp;
  }
}

template<OperationType OpType_>
void OpenMultiHeadAttention<OpType_>::trt_add_QKV_bias_COL32_int8IO_kernelLauncher(
  int8_t* output,
  const int8_t* Q,
  const DataType_* bias_Q,
  const DataType_* bias_K,
  const DataType_* bias_V,
  const float *q_input_deQFactor_ptr, 
  const float *k_input_deQFactor_ptr, 
  const float *v_input_deQFactor_ptr, 
  const float qkv_output_scale)
{
  int head_num_x_size_per_head = head_num_*size_per_head_;
  dim3 grid((head_num_x_size_per_head + 31)/32, (param_.valid_word_num + 31)/32, 3);
  dim3 block(8, 32);

  assert(size_per_head_ % 4 == 0);

  trt_add_QKV_bias_COL32_int8IO<<<grid, block, 0, param_.stream>>>((char4*)output, (const char4*)Q,
                                                                   bias_Q, bias_K, bias_V,
                                                                   q_input_deQFactor_ptr, k_input_deQFactor_ptr, v_input_deQFactor_ptr,
                                                                   qkv_output_scale, param_.valid_word_num, 
                                                                   head_num_, size_per_head_, head_num_x_size_per_head);
}

template<OperationType OpType_>
void OpenMultiHeadAttention<OpType_>::int8_fused_multiHeadAttr_kernelLauncher(const void* Q, 
                                                                              const float *q_deQFactor_ptr, const float *k_deQFactor_ptr, const float *v_deQFactor_ptr, 
                                                                              const float mScaleQkv, const int S)
{

  if (int8_mode_ == 1)
  {
    trt_add_QKV_bias_COL32_int32Iint8O_kernelLauncher((int8_t*)q_buf_, 
                                                      (const int32_t*)Q,
                                                      param_.self_attention.query_weight.bias,
                                                      param_.self_attention.key_weight.bias,
                                                      param_.self_attention.value_weight.bias,
                                                      param_.amaxList+2, query_weight_amax_list, 
                                                      key_weight_amax_list, value_weight_amax_list, 
                                                      1.0f/mScaleQkv);
  }
  else if (int8_mode_ == 2)
  {
    trt_add_QKV_bias_COL32_int8IO_kernelLauncher((int8_t*)q_buf_,
                                                 (const int8_t*)Q,
                                                 param_.self_attention.query_weight.bias,
                                                 param_.self_attention.key_weight.bias,
                                                 param_.self_attention.value_weight.bias,
                                                 q_deQFactor_ptr, k_deQFactor_ptr, v_deQFactor_ptr,
                                                 1.0f/mScaleQkv
                                                );
  }

  const int B = param_.trt_seqlen_size - 1;
  dispatcher_int8->setup(S, B); 
  dispatcher_int8->run((int8_t*)q_buf_, nullptr, param_.trt_seqlen_offset, trt_attn_workspace_, (int8_t*)transpose_dst_int_buf_, param_.stream);   
    
  //transpose_dst_int_buf_ is [batch*seqlen, hidden_dim] row-major
  rowMajorToCOL32_kernelLauncher((int8_t*)(param_.attr_out), (const int8_t*)transpose_dst_int_buf_, param_.valid_word_num, head_num_*size_per_head_, param_.stream);     
}


template<OperationType OpType_>
void OpenMultiHeadAttention<OpType_>::fused_multiHeadAttr_kernelLauncher(const int S)
{

  trt_add_QKV_bias_kernelLauncher(param_.self_attention.query_weight.bias,
                                  param_.self_attention.key_weight.bias,
                                  param_.self_attention.value_weight.bias);


  const int B = param_.trt_seqlen_size - 1;
  dispatcher_fp16->setup(S, B);
  dispatcher_fp16->run(q_buf_, nullptr, param_.trt_seqlen_offset, trt_attn_workspace_, param_.attr_out, param_.stream);
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
void add_QKV_bias_rebuild_padding(T* Q, const T* bias_Q, T* K, const T* bias_K, T* V, const T* bias_V, T* q_buf_, T* k_buf_, T* v_buf_, 
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int* mask_offset)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int bdim = blockDim.x;

  const int tgt_batch_id = (bid + mask_offset[bid]) / seq_len;
  const int tgt_seq_id = (bid + mask_offset[bid]) % seq_len;
  const int tgt_head_id = tid / size_per_head;
  const int tgt_hidden_id = tid % size_per_head;

  const int src_id = bid * bdim + tid;
  const int tgt_id = tgt_batch_id * head_num * seq_len * size_per_head + \
                    tgt_head_id * seq_len * size_per_head + \
                    tgt_seq_id * size_per_head + \
                    tgt_hidden_id;
  
  q_buf_[tgt_id] = Q[src_id] + bias_Q[tid];
  k_buf_[tgt_id] = K[src_id] + bias_K[tid];
  v_buf_[tgt_id] = V[src_id] + bias_V[tid];
}

template<typename T>
void add_QKV_bias_rebuild_padding_kernelLauncher(T* Q, const T* bias_Q, T* K, const T* bias_K, T* V, const T* bias_V, T* q_buf, T* k_buf, T* v_buf, 
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int valid_word_num, 
  const int* mask_offset, cudaStream_t stream)
{
  const int k = head_num*size_per_head;
    
  if(std::is_same<T, float>::value)
  {
    add_QKV_bias_rebuild_padding<<<valid_word_num, k, 0, stream>>>(Q, bias_Q, K, bias_K, 
      V, bias_V, q_buf, k_buf, v_buf, 
      batch_size, seq_len, head_num, size_per_head, mask_offset);                                                 
  }
  else
  {
    add_QKV_bias_rebuild_padding<<<valid_word_num, k / 2, 0, stream>>>((half2*)Q, (const half2*)bias_Q, 
      (half2*)K, (const half2*)bias_K, (half2*)V, (const half2*)bias_V, 
      (half2*)q_buf, (half2*)k_buf, (half2*)v_buf,
       batch_size, seq_len, head_num, size_per_head / 2, mask_offset);
  }  
}

template
void add_QKV_bias_rebuild_padding_kernelLauncher(float* Q, const float* bias_Q, float* K, const float* bias_K, float* V, const float* bias_V, float* q_buf, float* k_buf, float* v_buf, const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int valid_word_num, const int* mask_offset, cudaStream_t stream);

template
void add_QKV_bias_rebuild_padding_kernelLauncher(half* Q, const half* bias_Q, half* K, const half* bias_K, half* V, const half* bias_V, half* q_buf, half* k_buf, half* v_buf, const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int valid_word_num, const int* mask_offset, cudaStream_t stream);

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

//grid = (seq_len/word_per_thread, batch_size, head_num)
//block.x = max(32, (seq_len + 31)/32*32)
//for seq_len not larger than 32
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

//input are a series of sub-matrixes of m = seq_len, n = seq_len, CUBLASLT_ORDER_COL32
//grid = (seq_len, batch_size, head_num)
//block.x = max(32, (seq_len/4 + 31)/32*32)
//for int32_t I; int8 O;
template <typename T>
__global__
void softmax_COL32(int8_t* output, const int32_t* input, const T* attr_mask, const int batch_size, 
                   const int head_num, const int seq_len, const float scalar1a, const float *scalar1b, 
                   const float *scalar1c, const float *amax_ptr, const int head_num_x_seq_len, const int seq_len_x_seq_len)
{
  const float amax = __ldg(amax_ptr);
  const float scalar1 = scalar1a * __ldg(scalar1b) * __ldg(scalar1c);
  int mask_id;
  int threadIdx4 = threadIdx.x << 2;

  char4* buf4Ptr = (char4 *)output;

  bool qual = threadIdx4 < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    char4 tmp4 = {0, 0, 0, 0};
    int inIdx = (blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len) +
                (threadIdx4 & 0xffffffe0) * seq_len +
                (seq_id << 5) + (threadIdx4 & 31);
                
    //set softmax of padding word to 0
    float mask_in_seq = static_cast<float>(__ldg(attr_mask+(blockIdx.y*seq_len_x_seq_len + seq_id)));
    if (mask_in_seq < 0.1f){
      if (qual)
        buf4Ptr[inIdx >> 2] = tmp4;
      continue;
    }  

    float4 floatTmp4 = {0.0f, 0.0f, 0.0f, 0.0f};    

    if (qual){
      floatTmp4.x = static_cast<float>(__ldg(input + inIdx)) * scalar1;
      floatTmp4.y = static_cast<float>(__ldg(input+inIdx+1)) * scalar1;
      floatTmp4.z = static_cast<float>(__ldg(input+inIdx+2)) * scalar1;
      floatTmp4.w = static_cast<float>(__ldg(input+inIdx+3)) * scalar1;
    }

    float mask_val, max_val;
    max_val = -1e20f;

    __shared__ float s_max, s_sum;

    if (qual){
      mask_id = threadIdx4 + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len;
      //for x
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id))) * -10000.0f;
      floatTmp4.x = floatTmp4.x + mask_val;
      max_val = fmaxf(max_val, floatTmp4.x);

      //for y
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id+1))) * -10000.0f;
      floatTmp4.y = floatTmp4.y + mask_val;
      max_val = fmaxf(max_val, floatTmp4.y);

      //for z
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id+2))) * -10000.0f;
      floatTmp4.z = floatTmp4.z + mask_val;
      max_val = fmaxf(max_val, floatTmp4.z);

      //for w
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id+3))) * -10000.0f;
      floatTmp4.w = floatTmp4.w + mask_val;
      max_val = fmaxf(max_val, floatTmp4.w);
    }

    max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    float sum_val = 0.0f;

    if (qual){
      floatTmp4.x = __expf(floatTmp4.x - s_max);
      sum_val += floatTmp4.x;
      floatTmp4.y = __expf(floatTmp4.y - s_max);
      sum_val += floatTmp4.y;
      floatTmp4.z = __expf(floatTmp4.z - s_max);
      sum_val += floatTmp4.z;
      floatTmp4.w = __expf(floatTmp4.w - s_max);
      sum_val += floatTmp4.w;
    }
    
    sum_val = blockDim.x <= 32 ? warpReduceSum(sum_val) : blockReduceSum<float>(sum_val);

    if (threadIdx.x == 0){
      s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
      s_sum = __fdividef(s_sum, amax);
    }
    __syncthreads();

    if (qual){

      tmp4.x = float_to_int8_rn(floatTmp4.x*s_sum);
      tmp4.y = float_to_int8_rn(floatTmp4.y*s_sum);
      tmp4.z = float_to_int8_rn(floatTmp4.z*s_sum);
      tmp4.w = float_to_int8_rn(floatTmp4.w*s_sum);

      buf4Ptr[inIdx >> 2] = tmp4;
    }
  }
}

//input are a series of sub-matrixes of m = seq_len, n = seq_len_padded, CUBLASLT_ORDER_COL32
//seq_len_padded = (seq_len+31)/32*32
//grid = (seq_len, batch_size, head_num)
//block.x = max(32, (seq_len_padded/4 + 31)/32*32)
//for int8_t IO;
template <typename T>
__global__
void softmax_COL32_varlen(int8_t* output, const int8_t* input, const T* attr_mask, const int batch_size, 
                   const int head_num, const int seq_len, const int seq_len_padded, 
                   const float scalar1a, const float *scalar1b, const float *amax_ptr, 
                   const int seq_len_x_seq_len, const int seq_len_x_seq_len_padded)
{
  const float amax = __ldg(amax_ptr);
  const float scalar1 = scalar1a * __ldg(scalar1b);
  int mask_id;
  int threadIdx4 = threadIdx.x << 2;

  char4* buf4Ptr = (char4 *)output;
  const char4* inBuf4Ptr = (const char4*)input;

  const bool qual = threadIdx4 < seq_len;
  const bool qual_padded = threadIdx4 < seq_len_padded;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
  
    char4 tmp4 = {0, 0, 0, 0};
    int inIdx = ((blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len_padded) +
                (threadIdx4 & 0xffffffe0) * seq_len +
                (seq_id << 5) + (threadIdx4 & 31)) >> 2;
  
    //set softmax of padding word in rows to 0
    const float mask_in_seq = static_cast<float>(__ldg(attr_mask+(blockIdx.y*seq_len_x_seq_len + seq_id)));
    if (mask_in_seq < 0.1f){
      if (qual_padded)
        buf4Ptr[inIdx] = tmp4;
      continue;
    }
    
    //set softmax of padding word in cols to 0
    float4 floatTmp4 = {0.0f, 0.0f, 0.0f, 0.0f};
    if (qual){
      tmp4 = __ldg(inBuf4Ptr + inIdx);
      floatTmp4.x = static_cast<float>(tmp4.x) * scalar1;
      floatTmp4.y = static_cast<float>(tmp4.y) * scalar1;
      floatTmp4.z = static_cast<float>(tmp4.z) * scalar1;
      floatTmp4.w = static_cast<float>(tmp4.w) * scalar1;
    }

    float mask_val, max_val;
    max_val = -1e20f;

    __shared__ float s_max, s_sum;

    if (qual){
      mask_id = threadIdx4 + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len;
      //for x
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id))) * -10000.0f;
      floatTmp4.x = floatTmp4.x + mask_val;
      max_val = fmaxf(max_val, floatTmp4.x);

      //for y
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id+1))) * -10000.0f;
      floatTmp4.y = floatTmp4.y + mask_val;
      max_val = fmaxf(max_val, floatTmp4.y);

      //for z
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id+2))) * -10000.0f;
      floatTmp4.z = floatTmp4.z + mask_val;
      max_val = fmaxf(max_val, floatTmp4.z);

      //for w
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id+3))) * -10000.0f;
      floatTmp4.w = floatTmp4.w + mask_val;
      max_val = fmaxf(max_val, floatTmp4.w);
    }

    max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    float sum_val = 0.0f;

    if (qual){
      floatTmp4.x = __expf(floatTmp4.x - s_max);
      sum_val += floatTmp4.x;
      floatTmp4.y = __expf(floatTmp4.y - s_max);
      sum_val += floatTmp4.y;
      floatTmp4.z = __expf(floatTmp4.z - s_max);
      sum_val += floatTmp4.z;
      floatTmp4.w = __expf(floatTmp4.w - s_max);
      sum_val += floatTmp4.w;
    }
    
    sum_val = blockDim.x <= 32 ? warpReduceSum(sum_val) : blockReduceSum<float>(sum_val);

    if (threadIdx.x == 0){
      s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
      s_sum = __fdividef(s_sum, amax);
    }
    __syncthreads();

    if (qual_padded){

      tmp4.x = qual ? float_to_int8_rn(floatTmp4.x*s_sum) : static_cast<int8_t>(0);
      tmp4.y = qual ? float_to_int8_rn(floatTmp4.y*s_sum) : static_cast<int8_t>(0);
      tmp4.z = qual ? float_to_int8_rn(floatTmp4.z*s_sum) : static_cast<int8_t>(0);
      tmp4.w = qual ? float_to_int8_rn(floatTmp4.w*s_sum) : static_cast<int8_t>(0);

      buf4Ptr[inIdx] = tmp4;
    }
  }
}

//input are a series of sub-matrixes of m = seq_len, n = seq_len_padded, CUBLASLT_ORDER_COL32
//seq_len_padded = (seq_len+31)/32*32
//grid = (seq_len, batch_size, head_num)
//block.x = max(32, (seq_len_padded + 31)/32*32)
//for int8_t IO, I/O with int8_t element;
template <typename T>
__global__
void softmax_COL32_perElement_varlen(int8_t* output, const int8_t* input, const T* attr_mask, const int batch_size,
                                     const int head_num, const int seq_len, const int seq_len_padded,
                                     const float scalar1a, const float *scalar1b, const float *amax_ptr,
                                     const int seq_len_x_seq_len, const int seq_len_x_seq_len_padded)
{
  const float amax = __ldg(amax_ptr);
  const float scalar1 = scalar1a * __ldg(scalar1b);
  int mask_id;
  const int tidx = threadIdx.x;

  const bool qual = tidx < seq_len;
  const bool qual_padded = tidx < seq_len_padded;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){

    int8_t tmp = 0;
    int inIdx = ((blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len_padded) +
                (tidx & 0xffffffe0) * seq_len +
                (seq_id << 5) + (tidx & 31));

    //set softmax of padding word in rows to 0
    const float mask_in_seq = static_cast<float>(__ldg(attr_mask+(blockIdx.y*seq_len_x_seq_len + seq_id)));
    if (mask_in_seq < 0.1f){
      if (qual_padded)
        output[inIdx] = tmp;
      continue;
    }

    //set softmax of padding word in cols to 0
    float floatTmp = qual ? (static_cast<float>(__ldg(input + inIdx)) * scalar1) : 0.0f;

    float mask_val, max_val;
    max_val = -1e20f;

    __shared__ float s_max, s_sum;

    if (qual){
      mask_id = tidx + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len;
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id))) * -10000.0f;
      floatTmp = floatTmp + mask_val;
    }

    max_val = blockDim.x <= 32 ? warpReduceMax(floatTmp) : blockReduceMax<float>(floatTmp);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    float sum_val = 0.0f;

    floatTmp = qual ? __expf(floatTmp - s_max) : floatTmp;

    sum_val = blockDim.x <= 32 ? warpReduceSum(floatTmp) : blockReduceSum<float>(floatTmp);

    if (threadIdx.x == 0){
      s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
      s_sum = __fdividef(s_sum, amax);
    }
    __syncthreads();

    if (qual_padded){
      tmp = qual ? float_to_int8_rn(floatTmp*s_sum) : static_cast<int8_t>(0);
      output[inIdx] = tmp;
    }
  }
}


//input are a series of sub-matrixes of m = seq_len, n = seq_len, CUBLASLT_ORDER_COL32
//grid = (seq_len, batch_size, head_num)
//block.x = (seq_len + 31)/32
//for int32_t I; int8 O;
//for seq_len <= 32
template <typename T>
__global__
void softmax_COL32_LE32(int8_t* output, const int32_t* input, const T* attr_mask, const int batch_size, 
                        const int head_num, const int seq_len, const float scalar1a, const float *scalar1b, 
                        const float *scalar1c, const float *amax_ptr, const int head_num_x_seq_len, const int seq_len_x_seq_len)
{
  const float amax = __ldg(amax_ptr);
  const float scalar1 = scalar1a * __ldg(scalar1b) * __ldg(scalar1c);
  int mask_id;
  int threadIdxx = threadIdx.x;
  bool qual = threadIdxx < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    int inIdx = (blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len) +
                (threadIdxx & 0xffffffe0) * seq_len +
                (seq_id << 5) + (threadIdxx & 31);
  
    //set softmax of padding word to 0
    float mask_in_seq = static_cast<float>(__ldg(attr_mask+(blockIdx.y*seq_len_x_seq_len + seq_id)));
    if (mask_in_seq < 0.1f){
      if (qual)
        output[inIdx] = 0;
      continue;
    }

    float floatTmp = qual ? static_cast<float>(__ldg(input + inIdx)) * scalar1 : 0.0f;

    float mask_val, max_val;

    __shared__ float s_max, s_sum;

    mask_id = qual ? threadIdxx + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len : 0;
    mask_val = qual ? (1.0f - static_cast<float>(__ldg(attr_mask+mask_id))) * -10000.0f : 0.0f;
    floatTmp = qual ? floatTmp + mask_val : 0.0f;
    max_val = qual ? floatTmp : -1e20f;

    max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    floatTmp = qual ? __expf(floatTmp - s_max) : 0.0f;
    
    float sum_val = blockDim.x <= 32 ? warpReduceSum(floatTmp) : blockReduceSum<float>(floatTmp);

    if (threadIdx.x == 0){
      s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
      s_sum = __fdividef(s_sum, amax);
    }
    __syncthreads();

    
    if (qual){
      output[inIdx] = float_to_int8_rn(floatTmp*s_sum);
    }
  }
}

//input are a series of sub-matrixes of m = seq_len, n = seq_len_padded, CUBLASLT_ORDER_COL32
//seq_len_padded = (seq_len+31)/32*32
//attr_mask is [batch_size, seq_len, seq_len]
//grid = (seq_len, batch_size, head_num)
//block.x = seq_len_padded
//for int8_t IO;
//for seq_len_padded == 32
template <typename T>
__global__
void softmax_COL32_LE32_varlen(int8_t* output, const int8_t* input, const T* attr_mask, const int batch_size, 
                        const int head_num, const int seq_len, const int seq_len_padded,
                        const float scalar1a, const float *scalar1b, const float *amax_ptr, 
                        const int seq_len_x_seq_len, const int seq_len_x_seq_len_padded)
{
  const float amax = __ldg(amax_ptr);
  const float scalar1 = scalar1a * __ldg(scalar1b);
  int mask_id;
  int threadIdxx = threadIdx.x;
  const bool qual = threadIdxx < seq_len;
  const bool qual_padded = threadIdxx < seq_len_padded;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    int inIdx = (blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len_padded) +
                (threadIdxx & 0xffffffe0) * seq_len +
                (seq_id << 5) + (threadIdxx & 31);

    //set softmax of padding word in rows to 0
    float mask_in_seq = static_cast<float>(__ldg(attr_mask+(blockIdx.y*seq_len_x_seq_len + seq_id)));
    if (mask_in_seq < 0.1f){
      if (qual_padded)
        output[inIdx] = 0;
      continue;
    }

    float mask_val, max_val;
    __shared__ float s_max, s_sum;

    //set softmax of padding word in cols to 0
    float floatTmp = qual ? static_cast<float>(__ldg(input + inIdx)) * scalar1 : 0.0f;
    mask_id = qual ? threadIdxx + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len : 0;
    mask_val = qual ? (1.0f - static_cast<float>(__ldg(attr_mask+mask_id))) * -10000.0f : 0.0f;
    floatTmp = qual ? floatTmp + mask_val : 0.0f;
    max_val = qual ? floatTmp : -1e20f;

    max_val = warpReduceMax(max_val);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    floatTmp = qual ? __expf(floatTmp - s_max) : 0.0f;
    
    float sum_val = blockDim.x <= 32 ? warpReduceSum(floatTmp) : blockReduceSum<float>(floatTmp);

    if (threadIdx.x == 0){
      s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
      s_sum = __fdividef(s_sum, amax);
    }
    __syncthreads();

    
    if (qual_padded){
      output[inIdx] = qual ? float_to_int8_rn(floatTmp*s_sum) : static_cast<int8_t>(0);
    }
  }
}

//input are a series of sub-matrixes of m = seq_len, n = seq_len, CUBLASLT_ORDER_COL32
//grid = (seq_len, batch_size, head_num)
//block.x = max(32, (seq_len/2 + 31)/32*32)
//for int32_t I; int8 O;
//for seq_len in (32, 64]
template <typename T>
__global__
void softmax_COL32_LE64(int8_t* output, const int32_t* input, const T* attr_mask, const int batch_size, 
                        const int head_num, const int seq_len, const float scalar1a, const float *scalar1b, 
                        const float *scalar1c, const float *amax_ptr, const int head_num_x_seq_len, const int seq_len_x_seq_len)
{
  const float amax = __ldg(amax_ptr);
  const float scalar1 = scalar1a * __ldg(scalar1b) * __ldg(scalar1c);
  int mask_id;
  int threadIdx2 = threadIdx.x << 1;

  char2* buf2Ptr = (char2 *)output;

  bool qual = threadIdx2 < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    char2 tmp2 = {0, 0};
    int inIdx = (blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len) +
                (threadIdx2 & 0xffffffe0) * seq_len +
                (seq_id << 5) + (threadIdx2 & 31);

    //set softmax of padding word to 0
    float mask_in_seq = static_cast<float>(__ldg(attr_mask+(blockIdx.y*seq_len_x_seq_len + seq_id)));
    if (mask_in_seq < 0.1f){
      if (qual)
        buf2Ptr[inIdx >> 1] = tmp2;
      continue;
    }

    float2 floatTmp2 = {0.0f, 0.0f};
    if (qual){
      floatTmp2.x = static_cast<float>(__ldg(input + inIdx)) * scalar1;
      floatTmp2.y = static_cast<float>(__ldg(input + inIdx + 1)) * scalar1;
    }

    float mask_val, max_val;
    max_val = -1e20f;

    __shared__ float s_max, s_sum;

    if (qual){
      mask_id = threadIdx2 + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len;
      //for x
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id))) * -10000.0f;
      floatTmp2.x = floatTmp2.x + mask_val;

      //for y
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id+1))) * -10000.0f;
      floatTmp2.y = floatTmp2.y + mask_val;
            
      max_val = fmaxf(floatTmp2.x, floatTmp2.y);
    }

    max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    float sum_val = 0.0f;

    if (qual){
      floatTmp2.x = __expf(floatTmp2.x - s_max);
      sum_val += floatTmp2.x;
      floatTmp2.y = __expf(floatTmp2.y - s_max);
      sum_val += floatTmp2.y;
    }
    
    sum_val = blockDim.x <= 32 ? warpReduceSum(sum_val) : blockReduceSum<float>(sum_val);

    if (threadIdx.x == 0){
      s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
      s_sum = __fdividef(s_sum, amax);
    }
    __syncthreads();

    if (qual){
      tmp2.x = float_to_int8_rn(floatTmp2.x*s_sum);
      tmp2.y = float_to_int8_rn(floatTmp2.y*s_sum);
      buf2Ptr[inIdx >> 1] = tmp2;
    }
  }
}

//input are a series of sub-matrixes of m = seq_len, n = seq_len_padded, CUBLASLT_ORDER_COL32
//seq_len_padded = (seq_len+31)/32*32
//grid = (seq_len, batch_size, head_num)
//block.x = 32
//for int8_t IO
//for seq_len in (32, 64]
template <typename T>
__global__
void softmax_COL32_LE64_varlen(int8_t* output, const int8_t* input, const T* attr_mask, const int batch_size, 
                        const int head_num, const int seq_len, const int seq_len_padded, 
                        const float scalar1a, const float *scalar1b, const float *amax_ptr, 
                        const int seq_len_x_seq_len, const int seq_len_x_seq_len_padded)
{
  const float amax = __ldg(amax_ptr);
  const float scalar1 = scalar1a * __ldg(scalar1b);
  int mask_id;
  int threadIdx2 = threadIdx.x << 1;

  char2* buf2Ptr = (char2 *)output;
  const char2* inBuf2Ptr = (const char2 *)input;

  const bool qual = threadIdx2 < seq_len;
  const bool qual_padded = threadIdx2 < seq_len_padded;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    char2 tmp2 = {0, 0};
    int inIdx = ((blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len_padded) +
                (threadIdx2 & 0xffffffe0) * seq_len +
                (seq_id << 5) + (threadIdx2 & 31)) >> 1;

    //set softmax of padding word in rows to 0
    float mask_in_seq = static_cast<float>(__ldg(attr_mask+(blockIdx.y*seq_len_x_seq_len + seq_id)));
    if (mask_in_seq < 0.1f){
      if (qual_padded)
        buf2Ptr[inIdx] = tmp2;
      continue;
    }

    //set softmax of padding word in cols to 0
    float2 floatTmp2 = {0.0f, 0.0f};
    if (qual){
      tmp2 = __ldg(inBuf2Ptr + inIdx);
      floatTmp2.x = static_cast<float>(tmp2.x) * scalar1;
      floatTmp2.y = static_cast<float>(tmp2.y) * scalar1;
    }

    float mask_val, max_val;
    max_val = -1e20f;

    __shared__ float s_max, s_sum;

    if (qual){
      mask_id = threadIdx2 + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len;
      //for x
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id))) * -10000.0f;
      floatTmp2.x = floatTmp2.x + mask_val;

      //for y
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id+1))) * -10000.0f;
      floatTmp2.y = floatTmp2.y + mask_val;
            
      max_val = fmaxf(floatTmp2.x, floatTmp2.y);
    }

    max_val = warpReduceMax(max_val);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    float sum_val = 0.0f;

    if (qual){
      floatTmp2.x = __expf(floatTmp2.x - s_max);
      sum_val += floatTmp2.x;
      floatTmp2.y = __expf(floatTmp2.y - s_max);
      sum_val += floatTmp2.y;
    }
    
    sum_val = warpReduceSum(sum_val);

    if (threadIdx.x == 0){
      s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
      s_sum = __fdividef(s_sum, amax);
    }
    __syncthreads();

    if (qual_padded){
      tmp2.x = qual ? float_to_int8_rn(floatTmp2.x*s_sum) : static_cast<int8_t>(0);
      tmp2.y = qual ? float_to_int8_rn(floatTmp2.y*s_sum) : static_cast<int8_t>(0);
      buf2Ptr[inIdx] = tmp2;
    }
  }
}


template <typename T>
void softmax_COL32_kernelLauncher(int8_t* output, const int32_t* input, const T* attr_mask, 
                                  const int batch_size, const int head_num, const int seq_len, 
                                  const float scalar1a, const float *scalar1b, const float *scalar1c, 
                                  const float *amax_ptr, cudaStream_t stream)
{
  dim3 grid, block;
  grid.x = seq_len;
  grid.y = batch_size;
  grid.z = head_num;

  if (seq_len <= 32){
    if (batch_size * head_num > 960)
      grid.x = (seq_len + 31) / 32; //ceil(float(seq_len)/32.0f);
    block.x = (seq_len + 31)/32*32;
    softmax_COL32_LE32<<<grid, block, 0, stream>>>(output, input, attr_mask, batch_size, head_num, 
                                                   seq_len, scalar1a, scalar1b, scalar1c, 
                                                   amax_ptr, seq_len*head_num, seq_len*seq_len);
  }
  else if (seq_len <= 64){
    assert(seq_len % 2 == 0);
    block.x = (seq_len/2 + 31)/32*32;
    if (batch_size * head_num > 960)
      grid.x = (seq_len + 31) / 32; //ceil(float(seq_len)/32.0f);
    softmax_COL32_LE64<<<grid, block, 0, stream>>>(output, input, attr_mask, batch_size, head_num, 
                                                   seq_len, scalar1a, scalar1b, scalar1c, 
                                                   amax_ptr, seq_len*head_num, seq_len*seq_len);
  }
  else
  {
    assert(seq_len % 4 == 0);
    block.x = (seq_len/4 + 31)/32*32;
    softmax_COL32<<<grid, block, 0, stream>>>(output, input, attr_mask, batch_size, head_num, 
                                              seq_len, scalar1a, scalar1b, scalar1c, 
                                              amax_ptr, seq_len*head_num, seq_len*seq_len);
  }
}

template
void softmax_COL32_kernelLauncher(int8_t* output, const int32_t* input, const float* attr_mask, 
                                  const int batch_size, const int head_num, const int seq_len, 
                                  const float scalar1a, const float *scalar1b, const float *scalar1c, 
                                  const float *amax_ptr, cudaStream_t stream);
                                  
template
void softmax_COL32_kernelLauncher(int8_t* output, const int32_t* input, const half* attr_mask, 
                                  const int batch_size, const int head_num, const int seq_len, 
                                  const float scalar1a, const float *scalar1b, const float *scalar1c, 
                                  const float *amax_ptr, cudaStream_t stream);

template <typename T>
void softmax_COL32_kernelLauncher(int8_t* output, const int8_t* input, const T* attr_mask, 
                                  const int batch_size, const int head_num, const int seq_len, 
                                  const float scalar1a, const float *scalar1b, const float *amax_ptr, 
                                  cudaStream_t stream)
{
  dim3 grid, block;
  grid.x = seq_len;
  grid.y = batch_size;
  grid.z = head_num;
  const int seq_len_padded = (seq_len + 31)/32*32;

  if (seq_len <= 32){
    if (batch_size * head_num > 960)
      grid.x = (seq_len + 31) / 32; //ceil(float(seq_len)/32.0f);
    block.x = seq_len_padded;
    softmax_COL32_LE32_varlen<<<grid, block, 0, stream>>>(output, input, attr_mask, batch_size, head_num, 
                                                   seq_len, seq_len_padded, scalar1a, scalar1b, amax_ptr,
                                                   seq_len*seq_len, seq_len*seq_len_padded);
  }
  else if (seq_len <= 64 && (seq_len % 2 == 0)){
    block.x = 32;
    if (batch_size * head_num > 960)
      grid.x = (seq_len + 31) / 32; // ceil(float(seq_len)/32.0f);
    softmax_COL32_LE64_varlen<<<grid, block, 0, stream>>>(output, input, attr_mask, batch_size, head_num, 
                                                   seq_len, seq_len_padded, scalar1a, scalar1b, amax_ptr,
                                                   seq_len*seq_len, seq_len*seq_len_padded);
  }
  else if (seq_len > 64 && (seq_len % 4 == 0))
  {
    block.x = (seq_len_padded/4 + 31)/32*32;
    softmax_COL32_varlen<<<grid, block, 0, stream>>>(output, input, attr_mask, batch_size, head_num, 
                                              seq_len, seq_len_padded, scalar1a, scalar1b, amax_ptr,
                                              seq_len*seq_len, seq_len*seq_len_padded);
  }
  else
  {
    block.x = (seq_len_padded + 31)/32*32;
    softmax_COL32_perElement_varlen<<<grid, block, 0, stream>>>(output, input, attr_mask, batch_size, head_num,
                                              seq_len, seq_len_padded, scalar1a, scalar1b, amax_ptr,
                                              seq_len*seq_len, seq_len*seq_len_padded);
  }
}

template
void softmax_COL32_kernelLauncher(int8_t* output, const int8_t* input, const float* attr_mask, 
                                  const int batch_size, const int head_num, const int seq_len, 
                                  const float scalar1a, const float *scalar1b, 
                                  const float *amax_ptr, cudaStream_t stream);
                                  
template
void softmax_COL32_kernelLauncher(int8_t* output, const int8_t* input, const half* attr_mask, 
                                  const int batch_size, const int head_num, const int seq_len, 
                                  const float scalar1a, const float *scalar1b, 
                                  const float *amax_ptr, cudaStream_t stream);                                  


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

template<typename T>
void transpose_kernelLauncher(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head, cudaStream_t stream)
{
  dim3 grid, block;
  if (std::is_same<T, float>::value)
  {
    const int seq_per_block = 1;
    grid.x = batch_size * head_num * seq_len / seq_per_block;
    block.x = seq_per_block * size_per_head;
    transpose<<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head);
  }
  else
  {
    const int seq_per_block = 4;
    grid.x = batch_size * head_num * seq_len / seq_per_block;
    block.x = seq_per_block * size_per_head / 2;
    assert(grid.x * seq_per_block == batch_size * head_num * seq_len);
    transpose<<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head / 2);
  }
}

template
void transpose_kernelLauncher(float* src, float* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head, cudaStream_t stream);

template
void transpose_kernelLauncher(half* src, half* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head, cudaStream_t stream);

template<typename T>
__global__
void transpose_rebuild_padding(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head,
  const int* mask_offset)
{
  // TODO: optimize this kernel? 
  // do remove_sequence_length_padding
  const int tid = threadIdx.x; // batch * seq_len or valid_word_num
  const int bid = blockIdx.x; // head_num * size_per_head

  const int src_batch_id = (bid + mask_offset[bid]) / seq_len;
  const int src_seq_id = (bid + mask_offset[bid]) % seq_len;

  const int dst_seq_id = bid;

  const int head_id = tid / size_per_head;
  const int hidden_id = tid % size_per_head;
  dst[dst_seq_id * head_num * size_per_head + tid] = src[ src_batch_id * head_num * seq_len * size_per_head +
    head_id * seq_len * size_per_head + src_seq_id * size_per_head + hidden_id];
}

template<typename T>
void transpose_rebuild_padding_kernelLauncher(T* src, T* dst, const int valid_word_num,
                                              const int batch_size, const int seq_len, 
                                              const int head_num, const int size_per_head, 
                                              const int* mask_offset, cudaStream_t stream)
{
  int k = head_num * size_per_head;
  if (std::is_same<T, float>::value)
  {
    transpose_rebuild_padding<<<valid_word_num, k, 0, stream>>>(src, dst, 
            batch_size, seq_len, head_num, size_per_head, mask_offset);
  }
  else
  {
    transpose_rebuild_padding<half2><<<valid_word_num, k / 2, 0, stream>>>(
            (half2*)src, (half2*)dst, 
            batch_size, seq_len, head_num, size_per_head / 2, mask_offset);
  }  
}

template
void transpose_rebuild_padding_kernelLauncher(float* src, float* dst, const int valid_word_num,
                                              const int batch_size, const int seq_len, 
                                              const int head_num, const int size_per_head, 
                                              const int* mask_offset, cudaStream_t stream);
                                              
template
void transpose_rebuild_padding_kernelLauncher(half* src, half* dst, const int valid_word_num,
                                              const int batch_size, const int seq_len, 
                                              const int head_num, const int size_per_head, 
                                              const int* mask_offset, cudaStream_t stream);

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

template void OpenMultiHeadAttention<OperationType::FP32>::trt_add_QKV_bias_kernelLauncher(
  const float* bias_Q,
  const float* bias_K,
  const float* bias_V);

template void OpenMultiHeadAttention<OperationType::FP16>::trt_add_QKV_bias_kernelLauncher(
  const half* bias_Q,
  const half* bias_K,
  const half* bias_V);

template void OpenMultiHeadAttention<OperationType::FP32>::trt_add_QKV_bias_COL32_int8IO_kernelLauncher(
  int8_t* output,
  const int8_t* Q,
  const float* bias_Q,
  const float* bias_K,
  const float* bias_V,
  const float *q_input_deQFactor_ptr, 
  const float *k_input_deQFactor_ptr, 
  const float *v_input_deQFactor_ptr, 
  const float qkv_output_scale);

template void OpenMultiHeadAttention<OperationType::FP16>::trt_add_QKV_bias_COL32_int8IO_kernelLauncher(
  int8_t* output,
  const int8_t* Q,
  const half* bias_Q,
  const half* bias_K,
  const half* bias_V,
  const float *q_input_deQFactor_ptr, 
  const float *k_input_deQFactor_ptr, 
  const float *v_input_deQFactor_ptr, 
  const float qkv_output_scale);

template void OpenMultiHeadAttention<OperationType::FP32>::trt_add_QKV_bias_COL32_int32Iint8O_kernelLauncher(
  int8_t* output,
  const int32_t* Q,
  const float* bias_Q,
  const float* bias_K,
  const float* bias_V,
  const float *input_deQFactor_div127_ptr,
  const float * q_weight_amax,
  const float * k_weight_amax,
  const float * v_weight_amax,
  const float qkv_output_scale);

template void OpenMultiHeadAttention<OperationType::FP16>::trt_add_QKV_bias_COL32_int32Iint8O_kernelLauncher(
  int8_t* output,
  const int32_t* Q,
  const half* bias_Q,
  const half* bias_K,
  const half* bias_V,
  const float *input_deQFactor_div127_ptr,
  const float * q_weight_amax,
  const float * k_weight_amax,
  const float * v_weight_amax,
  const float qkv_output_scale);

template void OpenMultiHeadAttention<OperationType::FP32>::fused_multiHeadAttr_kernelLauncher(const int S);
template void OpenMultiHeadAttention<OperationType::FP16>::fused_multiHeadAttr_kernelLauncher(const int S);

template void OpenMultiHeadAttention<OperationType::FP32>::int8_fused_multiHeadAttr_kernelLauncher(
  const void* Q, 
  const float *q_deQFactor_ptr, const float *k_deQFactor_ptr, const float *v_deQFactor_ptr, 
  const float mScaleQkv, const int S);
template void OpenMultiHeadAttention<OperationType::FP16>::int8_fused_multiHeadAttr_kernelLauncher(
  const void* Q, 
  const float *q_deQFactor_ptr, const float *k_deQFactor_ptr, const float *v_deQFactor_ptr, 
  const float mScaleQkv, const int S);

__global__
void trt_add_QKV_bias_2(const half2* Q, const half2* bias_Q, 
                        const half2* K, const half2* bias_K, 
                        const half2* V, const half2* bias_V, 
                        half2* qkv_buf_,  
                        const int valid_word_num, 
                        const int head_num, const int size_per_head)
{
  // Add bias, and then transpose from 
  // [3, valid_word_num, head, size] -> [valid_word_num, head, 3, size]

  const int seq_id = blockIdx.x;
  const int size_id = threadIdx.x % size_per_head;
  const int head_id = (threadIdx.x - size_id) / size_per_head;

  const int target_offset = blockIdx.x * head_num * 3 * size_per_head + head_id * 3 * size_per_head;

  qkv_buf_[ target_offset + 
          0 * size_per_head +
          size_id] = Q[ seq_id * blockDim.x + threadIdx.x] + bias_Q[threadIdx.x];

  qkv_buf_[ target_offset + 
          1 * size_per_head +
          size_id] = K[ seq_id * blockDim.x + threadIdx.x] + bias_K[threadIdx.x];

  qkv_buf_[ target_offset + 
          2 * size_per_head +
          size_id] = V[ seq_id * blockDim.x + threadIdx.x] + bias_V[threadIdx.x];
}

void trt_add_QKV_bias_transpose_debug_kernelLauncher(
  const half* query_buf, const half* bias_Q,
  const half* key_buf, const half* bias_K,
  const half* value_buf, const half* bias_V,
  half* context_buf, 
  const int valid_word_num, 
  const int head_num, const int size_per_head,
  cudaStream_t stream)
{
  dim3 grid;
  dim3 block;
  
  grid.x = 3 * valid_word_num;
  block.x = head_num * size_per_head / 2;
  
  assert(block.x <= 1024);

  trt_add_QKV_bias_2<<<grid, block, 0, stream>>>( (const half2*)query_buf, (const half2*)bias_Q, 
                                                  (const half2*)key_buf, (const half2*)bias_K, 
                                                  (const half2*)value_buf, (const half2*)bias_V, 
                                                  (half2*)context_buf, 
                                                  valid_word_num, 
                                                  head_num, size_per_head / 2);
}


}//namespace cuda
}//namespace fastertransformer

