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
#include "cuda_int8_kernels.h"
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>

namespace fastertransformer{

template <typename T>
__inline__ __device__
T gelu(T x)
{
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <>
__inline__ __device__
half2 gelu(half2 val)
{
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp =  __half22float2(val);

  tmp.x = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));

}

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

//transpose matrix & transform COL32 to col-major
//input matrix is (m n) COL32
//output matrix is (n m) col-major
//grid((n+31)/32, (m+31)/32)
//block(32, 32)
template<typename T>
__global__
void transposeMatrix_COL32ToColMajor_kernel(T*dst, const T* src, const int m, const int n)
{

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  bool check = ((x < n) && (y < m));
  // COL32_col = x >> 5 ; COL32_row = (y << 5) + (x & 31);
  // COL32_idx = (COL32_col << 5) * m + COL32_row = (x & 0xffffffe0)*m + (y << 5) + (x & 31)
  if (check)
    dst[y*n+x] = __ldg(src+((x & 0xffffffe0)*m + (y << 5) + (x & 31)));
}

//transpose matrix & transform COL32 to col-major
//input matrix is (m n) COL32
//output matrix is (n m) col-major
//grid((n+31)/32, (m+31)/32)
//block(16, 32)
template<>
__global__
void transposeMatrix_COL32ToColMajor_kernel(half2*dst, const half2* src, const int m, const int n)
{

  int x = (blockIdx.x*blockDim.x + threadIdx.x) << 1;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  bool check = ((x < n) && (y < m));
  // COL32_col = x >> 5 ; COL32_row = (y << 5) + (x & 31);
  // COL32_idx = (COL32_col << 5) * m + COL32_row = (x & 0xffffffe0)*m + (y << 5) + (x & 31)
  if (check)
    dst[(y*n+x) >> 1] = __ldg(src+(((x & 0xffffffe0)*m + (y << 5) + (x & 31)) >> 1));
}

//transpose matrix & transform COL32 to col-major
//input matrix is (m n) COL32
//output matrix is (n m) col-major
template <typename T>
void transposeMatrix_COL32ToColMajor_kernelLauncher(T* dst, const T* src, const int m, const int n, cudaStream_t stream)
{
  assert(n%32 == 0);
  if (sizeof(T) == sizeof(float))
    transposeMatrix_COL32ToColMajor_kernel<T><<<dim3((n+31)/32, (m+31)/32), dim3(32, 32), 0, stream>>>(dst, src, m, n);
  else if (sizeof(T) == sizeof(half))
    transposeMatrix_COL32ToColMajor_kernel<<<dim3((n+31)/32, (m+31)/32), dim3(16, 32), 0, stream>>>((half2*)dst, (const half2*)src, m, n);
}

template void transposeMatrix_COL32ToColMajor_kernelLauncher<float>(float* dst, const float* src, const int m, const int n, cudaStream_t stream);

template void transposeMatrix_COL32ToColMajor_kernelLauncher<half>(half *dst, const half* src, const int m, const int n, cudaStream_t stream);

template void transposeMatrix_COL32ToColMajor_kernelLauncher<int8_t>(int8_t* dst, const int8_t* src, const int m, const int n, cudaStream_t stream);


//transpose matrix & transfrom col-major to COL32 & quantize
//input matrix is (m, n) col-major 
//output matrix is (n, m) COL32, using char4 to write out
//m should be a mutiple of 32
//grid((m+31)/32, (n+31)/32)
//block(8, 32)
template<typename T>
__global__
void transposeMatrix_colMajorToCOL32_quantize_kernel(char4*dst, const T* src, const int m, const int n, const float* scale_ptr)
{

  const float scale = __ldg(scale_ptr);

  int x = (blockIdx.x*blockDim.x + threadIdx.x) << 2;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  bool check = ((x < m) && (y < n));
  if (check)
  {
    char4 tmp4;
    tmp4.x = float_to_int8_rn(static_cast<float>(__ldg(src+y*m+x))*scale);
    tmp4.y = float_to_int8_rn(static_cast<float>(__ldg(src+y*m+x+1))*scale);
    tmp4.z = float_to_int8_rn(static_cast<float>(__ldg(src+y*m+x+2))*scale);  
    tmp4.w = float_to_int8_rn(static_cast<float>(__ldg(src+y*m+x+3))*scale); 
    
    // COL32_col = x >> 5 ; COL32_row = (y << 5) + (x & 31); 
    // COL32_idx = (COL32_col << 5) * n + COL32_row = (x & 0xffffffe0)*n + (y << 5) + (x & 31)
    
    dst[((x & 0xffffffe0) * n + (y << 5) + (x & 31)) >> 2] = tmp4;
  }
}

//transpose matrix & transfrom col-major to COL32 & quantize
//input matrix is (m, n) col-major 
//output matrix is (n, m) COL32, using char4 to write out
//m should be a mutiple of 32
//grid((m+31)/32, (n+31)/32)
//block(8, 32)
template <typename T>
void transposeMatrix_colMajorToCOL32_quantize_kernelLauncher(int8_t* dst, const T* src, const int m, const int n, const float* scale_ptr, cudaStream_t stream)
{
  assert(m%32 == 0);
  transposeMatrix_colMajorToCOL32_quantize_kernel<T><<<dim3((m+31)/32, (n+31)/32), dim3(8, 32), 0, stream>>>((char4 *)dst, src, m, n, scale_ptr);
}

template void transposeMatrix_colMajorToCOL32_quantize_kernelLauncher<float>(int8_t* dst, const float* src, const int m, const int n, const float* scale_ptr, cudaStream_t stream);

template void transposeMatrix_colMajorToCOL32_quantize_kernelLauncher<half>(int8_t *dst, const half* src, const int m, const int n, const float* scale_ptr, cudaStream_t stream);
                                                      
//transpose matrix & transfrom col-major to COL32
//input matrix is (m, n) col-major 
//output matrix is (n, m) COL32
//m should be a mutiple of 32
//grid((m+31)/32, (n+31)/32)
//block(32, 32)
template<typename T>
__global__
void transposeMatrix_colMajorToCOL32_kernel(T*dst, const T* src, const int m, const int n)
{

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  bool check = ((x < m) && (y < n));
  if (check)
  {
  
    // COL32_col = x >> 5 ; COL32_row = (y << 5) + (x & 31); 
    // COL32_idx = (COL32_col << 5) * n + COL32_row = (x & 0xffffffe0)*n + (y << 5) + (x & 31)
    dst[(x & 0xffffffe0) * n + (y << 5) + (x & 31)] = __ldg(src+y*m+x);   
  }
}

//transpose matrix & transfrom col-major to COL32
//input matrix is (m, n) col-major 
//output matrix is (n, m) COL32
//m should be a mutiple of 32
//grid((m+31)/32, (n+31)/32)
//block(16, 32)
template<>
__global__
void transposeMatrix_colMajorToCOL32_kernel(half2*dst, const half2* src, const int m, const int n)
{

  int x = (blockIdx.x*blockDim.x + threadIdx.x) << 1;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  bool check = ((x < m) && (y < n));
  if (check)
  {
  
    // COL32_col = x >> 5 ; COL32_row = (y << 5) + (x & 31); 
    // COL32_idx = (COL32_col << 5) * n + COL32_row = (x & 0xffffffe0)*n + (y << 5) + (x & 31)
    dst[((x & 0xffffffe0) * n + (y << 5) + (x & 31)) >> 1] = __ldg(src+((y*m+x) >> 1));   
  }
}

//transpose matrix & transfrom col-major to COL32
//input matrix is (m, n) col-major 
//output matrix is (n, m) COL32, using char4 to write out
//m should be a mutiple of 32
//grid((m+31)/32, (n+31)/32)
//block(8, 32)
template <typename T>
void transposeMatrix_colMajorToCOL32_kernelLauncher(T* dst, const T* src, const int m, const int n, cudaStream_t stream)
{
  assert(m%32 == 0);
  if (sizeof(T) == sizeof(float))
    transposeMatrix_colMajorToCOL32_kernel<T><<<dim3((m+31)/32, (n+31)/32), dim3(32, 32), 0, stream>>>(dst, src, m, n);
  else if (sizeof(T) == sizeof(half))
    transposeMatrix_colMajorToCOL32_kernel<<<dim3((m+31)/32, (n+31)/32), dim3(16, 32), 0, stream>>>((half2*)dst, (const half2*)src, m, n);
}

template void transposeMatrix_colMajorToCOL32_kernelLauncher<float>(float* dst, const float* src, const int m, const int n, cudaStream_t stream);

template void transposeMatrix_colMajorToCOL32_kernelLauncher<half>(half *dst, const half* src, const int m, const int n, cudaStream_t stream);

//transfrom row-major to COL32
//input matrix is (m, n) row-major 
//output matrix is (m, n) COL32
//n should be a mutiple of 32
//grid((n+31)/32, (m+31)/32)
//block(8, 32)
__global__
void rowMajorToCOL32_kernel(char4*dst, const char4* src, const int m, const int n)
{

  int n_id = (blockIdx.x*blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y*blockDim.y + threadIdx.y;

  bool check = ((m_id < m) && (n_id < n));
  if (check)
  {
  
    // COL32_col = n_id >> 5 ; COL32_row = (m_id << 5) + (n_id & 31); 
    // COL32_idx = (COL32_col << 5) * m + COL32_row = (n_id & 0xffffffe0)*m + (m_id << 5) + (n_id & 31)
    dst[((n_id & 0xffffffe0)*m + (m_id << 5) + (n_id & 31)) >> 2] = __ldg(src+((m_id*n+n_id) >> 2));   
  }
}

//transfrom row-major to COL32
//input matrix is (m, n) row-major 
//output matrix is (m, n) COL32
//n should be a mutiple of 32
//grid((n+31)/32, (m+31)/32)
//block(8, 32)
void rowMajorToCOL32_kernelLauncher(int8_t* dst, const int8_t* src, const int m, const int n, cudaStream_t stream)
{
  assert(n%32 == 0);
  rowMajorToCOL32_kernel<<<dim3((n+31)/32, (m+31)/32), dim3(8, 32), 0, stream>>>((char4*)dst, (const char4*)src, m, n);
}


//add bias to matrix of m * n, CUBLASLT_ORDER_COL32
//grid, thread = (m), (n/4)
//using char4 as output
//for per-channel-quantization weight
__global__
void add_bias_act_COL32_int32I_int8O(int8_t *out, const int32_t* input, const float* bias, const int m, const int n, 
                                     const float* weight_amax, const float *input_deQFactor_div127_ptr, const float *out_scale_ptr)
{

  const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
  const float out_scale = __ldg(out_scale_ptr);
 
  int col_start = threadIdx.x << 2;
  char4 *outTmpPtr = (char4 *)out;
  char4 tmp;
  int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start&31)) >> 2;
  float val;
  
  const int4 input4 = __ldg(((const int4*)input)+outIdx);
  const float4 weight4 = __ldg(((const float4*)weight_amax)+threadIdx.x);
  const float4 bias4 = __ldg(((const float4*)bias)+threadIdx.x);
  
  val = static_cast<float>(input4.x)*weight4.x*input_deQFactor_div127 + bias4.x;
  val = gelu(val);
  tmp.x = float_to_int8_rn(val*out_scale);
 
  val = static_cast<float>(input4.y)*weight4.y*input_deQFactor_div127 + bias4.y;
  val = gelu(val);
  tmp.y = float_to_int8_rn(val*out_scale);
  
  col_start = col_start + 1;
  val = static_cast<float>(input4.z)*weight4.z*input_deQFactor_div127 + bias4.z;
  val = gelu(val);
  tmp.z = float_to_int8_rn(val*out_scale);

  col_start = col_start + 1;
  val = static_cast<float>(input4.w)*weight4.w*input_deQFactor_div127 + bias4.w;
  val = gelu(val);
  tmp.w = float_to_int8_rn(val*out_scale);

  outTmpPtr[outIdx] = tmp;
}


__global__
void add_bias_act_COL32_int32I_int8O(char4 *out, const int4* input, const half2* bias, const int m, const int n, 
                                     const float4* weight_amax, const float *input_deQFactor_div127_ptr, const float *out_scale_ptr)
{
  const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
  const float out_scale = __ldg(out_scale_ptr);
  int col_start = threadIdx.x << 2;
  int threadIdx2 = threadIdx.x << 1;
  char4 tmp;
  int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start&31)) >> 2;
  float val;
  
  const int4 input4 = __ldg(input+outIdx);
  const float4 weight4 = __ldg(weight_amax+threadIdx.x);
  const half2 biasTmp = __ldg(bias+threadIdx2);
  const half2 biasTmp2 = __ldg(bias+threadIdx2+1);

  val = static_cast<float>(input4.x)*weight4.x*input_deQFactor_div127 + static_cast<float>(biasTmp.x);
  val = gelu(val);
  tmp.x = float_to_int8_rn(out_scale * val);

  val = static_cast<float>(input4.y)*weight4.y*input_deQFactor_div127 + static_cast<float>(biasTmp.y);
  val = gelu(val);
  tmp.y = float_to_int8_rn(out_scale * val);
  
  val = static_cast<float>(input4.z)*weight4.z*input_deQFactor_div127 + static_cast<float>(biasTmp2.x);
  val = gelu(val);
  tmp.z = float_to_int8_rn(out_scale * val);

  val = static_cast<float>(input4.w)*weight4.w*input_deQFactor_div127 + static_cast<float>(biasTmp2.y);
  val = gelu(val);
  tmp.w = float_to_int8_rn(out_scale * val);

  out[outIdx] = tmp;
}

template <typename T>
void add_bias_act_COL32_int32I_int8O_kernelLauncher(int8_t *out, const int32_t* input, const T* bias, const int m, const int n, 
                                                    cudaStream_t stream, const float* weight_amax, const float* input_deQFactor_div127_ptr, const float* out_scale_ptr){
  dim3 grid(m);
  dim3 block(n/4);
  assert(block.x <= 1024);
  if (sizeof(T) == sizeof(half))
    add_bias_act_COL32_int32I_int8O<<<grid, block, 0, stream>>>((char4*)out, (const int4*)input, (const half2*)bias, m, n, (const float4*)weight_amax, input_deQFactor_div127_ptr, out_scale_ptr);
  else
    add_bias_act_COL32_int32I_int8O<<<grid, block, 0, stream>>>(out, input, (const float*)bias, m, n, weight_amax, input_deQFactor_div127_ptr, out_scale_ptr);
}

template void add_bias_act_COL32_int32I_int8O_kernelLauncher<float>(int8_t *out, const int32_t* input, const float* bias, const int m, const int n, cudaStream_t stream, const float* weight_amax, const float *input_deQFactor_div127_ptr, const float *out_scale_ptr);

template void add_bias_act_COL32_int32I_int8O_kernelLauncher<half>(int8_t *out, const int32_t* input, const half* bias, const int m, const int n, cudaStream_t stream, const float* weight_amax, const float *input_deQFactor_div127_ptr, const float *out_scale_ptr);


//add bias to matrix of m * n, CUBLASLT_ORDER_COL32
//grid, thread = (m), (n/4)
//using char4
//for per-tensor-quantization weight
template <typename T>
__global__
void add_bias_act_COL32_int8IO(int8_t *out, const int8_t* input, const T* bias, const int m, const int n, 
                               const float *input_deQFactor_ptr, const float *out_scale_ptr)
{

  const float input_deQFactor = __ldg(input_deQFactor_ptr);
  const float out_scale = __ldg(out_scale_ptr);
 
  int col_start = threadIdx.x << 2;
  char4 *outTmpPtr = (char4 *)out;
  char4 *inputTmpPtr = (char4*)input;
  char4 tmp;
  int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start&31)) >> 2;
  float val;
  tmp = __ldg(inputTmpPtr+outIdx);
  val = static_cast<float>(tmp.x)*input_deQFactor + static_cast<float>(__ldg(bias+col_start));
  val = gelu(val);
  tmp.x = float_to_int8_rn(val*out_scale);
 
  col_start = col_start + 1;
  val = static_cast<float>(tmp.y)*input_deQFactor + static_cast<float>(__ldg(bias+col_start));
  val = gelu(val);
  tmp.y = float_to_int8_rn(val*out_scale);
  
  col_start = col_start + 1;
  val = static_cast<float>(tmp.z)*input_deQFactor + static_cast<float>(__ldg(bias+col_start));
  val = gelu(val);
  tmp.z = float_to_int8_rn(val*out_scale);

  col_start = col_start + 1;
  val = static_cast<float>(tmp.w)*input_deQFactor + static_cast<float>(__ldg(bias+col_start));
  val = gelu(val);
  tmp.w = float_to_int8_rn(val*out_scale);

  outTmpPtr[outIdx] = tmp;
}

template <typename T>
void add_bias_act_COL32_int8IO_kernelLauncher(int8_t *out, const int8_t* input, const T* bias, const int m, const int n, 
                                              cudaStream_t stream, const float* input_deQFactor_ptr, const float* out_scale_ptr){
  dim3 grid(m);
  dim3 block(n/4);
  assert(block.x <= 1024);
  
  add_bias_act_COL32_int8IO<<<grid, block, 0, stream>>>(out, input, bias, m, n, input_deQFactor_ptr, out_scale_ptr);
}

template void add_bias_act_COL32_int8IO_kernelLauncher<float>(int8_t *out, const int8_t* input, const float* bias, const int m, const int n, cudaStream_t stream, const float *input_deQFactor_ptr, const float *out_scale_ptr);

template void add_bias_act_COL32_int8IO_kernelLauncher<half>(int8_t *out, const int8_t* input, const half* bias, const int m, const int n, cudaStream_t stream, const float *input_deQFactor_ptr, const float *out_scale_ptr);

//input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n)
template <typename T>
__global__
void add_bias_input_layernorm_COL32_int8I_DataTypeO(T* output, const int8_t* input1, const int8_t* input2, const T* bias, const T* gamma, 
                                                    const T* beta, int m, int n, 
                                                    const float *input1_deQFactor_ptr, 
                                                    const float *input2_deQFactor_ptr)
{
  const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  int col_start = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out;
  int idx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31));
  
  local_out = static_cast<float>(__ldg(input2+idx))*input2_deQFactor + static_cast<float>(__ldg(input1+idx))*input1_deQFactor + static_cast<float>(__ldg(bias+col_start));

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out = local_out - s_mean;

  variance = blockReduceSum<float>(local_out * local_out);
  
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  local_out = (local_out * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  
  output[idx] = local_out;
}

//input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/2)
template <>
__global__
void add_bias_input_layernorm_COL32_int8I_DataTypeO(half2* output, const int8_t* input1, const int8_t* input2, const half2* bias, const half2* gamma, 
                                                    const half2* beta, int m, int n, 
                                                    const float *input1_deQFactor_ptr, 
                                                    const float *input2_deQFactor_ptr)
{
  const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  int col_start = threadIdx.x << 1;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float2 local_out;
  int idx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31)) >> 1;
  
  const char2 * input1_ptr2 = (const char2*)input1;
  const char2 * input2_ptr2 = (const char2*)input2;
  char2 input_tmp1 = __ldg(input1_ptr2 + idx);
  char2 input_tmp2 = __ldg(input2_ptr2 + idx);

  half2 bias_tmp = __ldg(bias+threadIdx.x);
  
  local_out.x = static_cast<float>(input_tmp1.x)*input1_deQFactor + static_cast<float>(input_tmp2.x)*input2_deQFactor + static_cast<float>(bias_tmp.x);
  
  local_out.y = static_cast<float>(input_tmp1.y)*input1_deQFactor + static_cast<float>(input_tmp2.y)*input2_deQFactor + static_cast<float>(bias_tmp.y);

  mean = blockReduceSum<float>(local_out.x + local_out.y);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out.x = local_out.x - s_mean;
  
  local_out.y = local_out.y - s_mean;

  variance = blockReduceSum<float>(local_out.x * local_out.x + local_out.y * local_out.y);
  
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  half2 gamma_tmp = __ldg(gamma+threadIdx.x);
  half2 beta_tmp = __ldg(beta+threadIdx.x);
  
  local_out.x = (local_out.x * s_variance) * static_cast<float>(gamma_tmp.x) + static_cast<float>(beta_tmp.x);
  local_out.y = (local_out.y * s_variance) * static_cast<float>(gamma_tmp.y) + static_cast<float>(beta_tmp.y);
  
  bias_tmp.x = half(local_out.x);
  bias_tmp.y = half(local_out.y);
  
  output[idx] = bias_tmp;
}

template<typename T>
void add_bias_input_layernorm_COL32_int8I_DataTypeO_kernelLauncher(T *output, const int8_t *input1,
                                                                   const int8_t *input2, const T *bias,
                                                                   const T *gamma, const T *beta,
                                                                   int m, int n,
                                                                   cudaStream_t stream, 
                                                                   const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr)
{
  dim3 grid(m);
  dim3 block(n);
  if (sizeof(T) == sizeof(half)){
    assert(n/2 <= 1024 && n%2 == 0);
    block.x = n/2;
    add_bias_input_layernorm_COL32_int8I_DataTypeO<<<grid, block, 0, stream>>>((half2*)output, input1, input2, (const half2*)bias, (const half2*)gamma, 
                                                                               (const half2*)beta, m, n, input1_deQFactor_ptr, 
                                                                               input2_deQFactor_ptr);
  }
  else{
    assert(n <= 1024);
    add_bias_input_layernorm_COL32_int8I_DataTypeO<T><<<grid, block, 0, stream>>>(output, input1, input2, bias, gamma, beta, 
                                                                                  m, n, input1_deQFactor_ptr,
                                                                                  input2_deQFactor_ptr);
  }
}

template void add_bias_input_layernorm_COL32_int8I_DataTypeO_kernelLauncher<float>(float* output, const int8_t* input1, const int8_t* input2, const float* bias, const float* gamma, const float* beta, int m, int n, cudaStream_t stream, const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr);

template void add_bias_input_layernorm_COL32_int8I_DataTypeO_kernelLauncher<half>(half* output, const int8_t* input1, const int8_t* input2, const half* bias, const half* gamma, const half* beta, int m, int n, cudaStream_t stream, const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr);

//input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/4)
//using char4
template <typename T>
__global__
void add_bias_input_layernorm_COL32_int8IO(int8_t* output, const int8_t* input1, const int8_t* input2, const T* bias, const T* gamma, 
                                           const T* beta, int m, int n, 
                                           const float *input1_deQFactor_ptr, 
                                           const float *input2_deQFactor_ptr, const float *output_scale_ptr)
{
  const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  const float output_scale = __ldg(output_scale_ptr);
  int col_start = threadIdx.x << 2;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out[4];
  int outIdx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31)) >> 2;
  char4 *outTmpPtr = (char4*)output;
  char4 *input1TmpPtr = (char4*)input1;
  char4 *input2TmpPtr = (char4*)input2;
  char4 input1Tmp = __ldg(input1TmpPtr+outIdx);
  char4 input2Tmp = __ldg(input2TmpPtr+outIdx);
  
  
  int col_start_tmp = col_start;
  local_out[0] = static_cast<float>(input2Tmp.x)*input2_deQFactor + static_cast<float>(input1Tmp.x)*input1_deQFactor + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[1] = static_cast<float>(input2Tmp.y)*input2_deQFactor + static_cast<float>(input1Tmp.y)*input1_deQFactor + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[2] = static_cast<float>(input2Tmp.z)*input2_deQFactor + static_cast<float>(input1Tmp.z)*input1_deQFactor + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[3] = static_cast<float>(input2Tmp.w)*input2_deQFactor + static_cast<float>(input1Tmp.w)*input1_deQFactor + static_cast<float>(__ldg(bias+col_start_tmp));


  mean = blockReduceSum<float>(local_out[0] + local_out[1] + local_out[2] + local_out[3]);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out[0] = local_out[0] - s_mean;
  local_out[1] = local_out[1] - s_mean;
  local_out[2] = local_out[2] - s_mean;
  local_out[3] = local_out[3] - s_mean;
  variance = blockReduceSum<float>(local_out[0] * local_out[0] +
                                   local_out[1] * local_out[1] +
                                   local_out[2] * local_out[2] +
                                   local_out[3] * local_out[3]
                                  );
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

  col_start = col_start+1;
  local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);
  
  col_start = col_start+1;
  local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);
  
  col_start = col_start+1;
  local_out[3] = (local_out[3] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);
  
  outTmpPtr[outIdx] = input2Tmp;
}

template <>
__global__
void add_bias_input_layernorm_COL32_int8IO(int8_t* output, const int8_t* input1, const int8_t* input2, const half2* bias, const half2* gamma, 
                                           const half2* beta, int m, int n, const float *input1_deQFactor_ptr, 
                                           const float *input2_deQFactor_ptr, const float *output_scale_ptr)
{
  const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  const float output_scale = __ldg(output_scale_ptr);
  int col_start = threadIdx.x << 2;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out[4];
  int outIdx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31)) >> 2;
  char4 *outTmpPtr = (char4*)output;
  char4 *input1TmpPtr = (char4*)input1;
  char4 *input2TmpPtr = (char4*)input2;
  char4 input1Tmp = __ldg(input1TmpPtr + outIdx);
  char4 input2Tmp = __ldg(input2TmpPtr + outIdx);
  
  int col_start_tmp = col_start;
  half2 biasTmp = __ldg(bias + (col_start_tmp >> 1));  
  local_out[0] = static_cast<float>(input2Tmp.x)*input2_deQFactor + static_cast<float>(input1Tmp.x)*input1_deQFactor + static_cast<float>(biasTmp.x);
  col_start_tmp = col_start_tmp + 1;
  local_out[1] = static_cast<float>(input2Tmp.y)*input2_deQFactor + static_cast<float>(input1Tmp.y)*input1_deQFactor + static_cast<float>(biasTmp.y);
  
  col_start_tmp = col_start_tmp + 1;
  biasTmp = __ldg(bias + (col_start_tmp >> 1));
  local_out[2] = static_cast<float>(input2Tmp.z)*input2_deQFactor + static_cast<float>(input1Tmp.z)*input1_deQFactor + static_cast<float>(biasTmp.x);
  col_start_tmp = col_start_tmp + 1;
  local_out[3] = static_cast<float>(input2Tmp.w)*input2_deQFactor + static_cast<float>(input1Tmp.w)*input1_deQFactor + static_cast<float>(biasTmp.y);


  mean = blockReduceSum<float>(local_out[0] + local_out[1] + local_out[2] + local_out[3]);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out[0] = local_out[0] - s_mean;
  local_out[1] = local_out[1] - s_mean;
  local_out[2] = local_out[2] - s_mean;
  local_out[3] = local_out[3] - s_mean;
  variance = blockReduceSum<float>(local_out[0] * local_out[0] +
                                   local_out[1] * local_out[1] +
                                   local_out[2] * local_out[2] +
                                   local_out[3] * local_out[3]
                                  );
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  col_start_tmp = col_start >> 1;
  biasTmp = __ldg(gamma+col_start_tmp);
  half2 betaTmp = __ldg(beta+col_start_tmp); 
  
  local_out[0] = (local_out[0] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
  input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

  col_start = col_start+1;
  local_out[1] = (local_out[1] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
  input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);
  
  col_start = col_start+1;
  col_start_tmp = col_start >> 1;
  biasTmp = __ldg(gamma+col_start_tmp);
  betaTmp = __ldg(beta+col_start_tmp);
  local_out[2] = (local_out[2] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
  input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);
  
  col_start = col_start+1;
  local_out[3] = (local_out[3] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
  input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);
  
  outTmpPtr[outIdx] = input2Tmp;
}

template<typename T>
void add_bias_input_layernorm_COL32_int8IO_kernelLauncher(int8_t* output, const int8_t* input1, const int8_t* input2, const T* bias, const T* gamma, 
                                                          const T* beta, int m, int n, cudaStream_t stream, 
                                                          const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr, const float *output_scale_ptr)
{
  dim3 grid(m);
  dim3 block(n/4);
  assert(n <= 1024);
  if (sizeof(T) == sizeof(half)){
    add_bias_input_layernorm_COL32_int8IO<<<grid, block, 0, stream>>>(output, input1, input2, (const half2*)bias, (const half2*)gamma, 
                                                                      (const half2*)beta, m, n, input1_deQFactor_ptr, 
                                                                      input2_deQFactor_ptr, output_scale_ptr);
  }
  else{
    add_bias_input_layernorm_COL32_int8IO<T><<<grid, block, 0, stream>>>(output, input1, input2, bias, gamma, beta, 
                                                                         m, n, input1_deQFactor_ptr, 
                                                                         input2_deQFactor_ptr, output_scale_ptr);
  }
}

template void add_bias_input_layernorm_COL32_int8IO_kernelLauncher<float>(int8_t* output, const int8_t* input1, const int8_t* input2, const float* bias, const float* gamma, const float* beta, int m, int n, cudaStream_t stream, const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr, const float *output_scale_ptr);

template void add_bias_input_layernorm_COL32_int8IO_kernelLauncher<half>(int8_t* output, const int8_t* input1, const int8_t* input2, const half* bias, const half* gamma, const half* beta, int m, int n, cudaStream_t stream, const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr, const float *output_scale_ptr);



//input1/input2/output matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n)
//for per_channel_quantization for weight
__global__
void add_bias_input_layernorm_COL32_int32I_DataTypeO(float* output, const int32_t* input1, const float* input2, const float* bias, const float* gamma, 
                                                     const float* beta, int m, int n, const float* weight_amax, const float *input1_amax_ptr)
{
  const float input1_amax = __ldg(input1_amax_ptr);
  int col_start = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out;
  int outIdx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31));

  float tmp = static_cast<float>(__ldg(input1 + outIdx)) * __ldg(weight_amax + col_start) * input1_amax * 0.000062f; //(1/127/127);
  float inputTmp = __ldg(input2 + outIdx);

  local_out = tmp + inputTmp + __ldg(bias + col_start);

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = __fdividef(mean, n);
  __syncthreads();

  local_out = local_out - s_mean;

  variance = blockReduceSum<float>(local_out * local_out);
  if(threadIdx.x == 0){
    s_variance = __fdividef(variance, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();

  local_out = (local_out * s_variance) * __ldg(gamma + col_start) + __ldg(beta + col_start);

  output[outIdx] = local_out;
}

__global__
void add_bias_input_layernorm_COL32_int32I_DataTypeO(half2* output, const int2* input1, const half2* input2, const half2* bias, const half2* gamma, 
                                                     const half2* beta, int m, int n, const float2* weight_amax, const float *input1_amax_ptr)
{
  int col_start = threadIdx.x << 1;

  const float input1_amax = __ldg(input1_amax_ptr);

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float2 local_out;
  int outIdx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31)) >> 1;

  const int2 input1Tmp = __ldg(input1 + outIdx);
  const float2 weightTmp = __ldg(weight_amax + threadIdx.x);

  float2 addTmp2;
  addTmp2.x = static_cast<float>(input1Tmp.x) * weightTmp.x * input1_amax * 0.000062f; //(1/127/127);
  addTmp2.y = static_cast<float>(input1Tmp.y) * weightTmp.y * input1_amax * 0.000062f; //(1/127/127);
  
  const half2 inputTmp = __ldg(input2 + outIdx);
  const half2 biasTmp = __ldg(bias + threadIdx.x);

  local_out = __half22float2(__hadd2(inputTmp, biasTmp));
  local_out.x = local_out.x + addTmp2.x;
  local_out.y = local_out.y + addTmp2.y;

  mean = blockReduceSum<float>(local_out.x + local_out.y);
  if(threadIdx.x == 0)
    s_mean = __fdividef(mean, n);
  __syncthreads();

  local_out.x = local_out.x - s_mean;
  local_out.y = local_out.y - s_mean;

  variance = blockReduceSum<float>(local_out.x*local_out.x + local_out.y*local_out.y);
  if(threadIdx.x == 0){
    s_variance = __fdividef(variance, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();

  float2 outputTmp;
  const half2 gammaTmp = __ldg(gamma + threadIdx.x);
  const half2 betaTmp = __ldg(beta + threadIdx.x);

  outputTmp.x = (local_out.x * s_variance) * static_cast<float>(gammaTmp.x) + static_cast<float>(betaTmp.x);
  outputTmp.y = (local_out.y * s_variance) * static_cast<float>(gammaTmp.y) + static_cast<float>(betaTmp.y);

  output[outIdx] =  __float22half2_rn(outputTmp);
}


template <typename T>
void add_bias_input_layernorm_COL32_int32I_DataTypeO_kernelLauncher(T* output, const int32_t* input1, const T* input2, const T* bias, const T* gamma, 
                                                                    const T* beta, int m, int n, cudaStream_t stream, const float* weight_amax, 
                                                                    const float* input1_amax_ptr){

  dim3 grid(m);
  dim3 block(n);
  if (sizeof(T) == sizeof(half)){
    block.x /= 2;
    assert(block.x <= 1024);
    add_bias_input_layernorm_COL32_int32I_DataTypeO<<<grid, block, 0, stream>>>((half2 *)output, (const int2*)input1, (const half2 *)input2, (const half2 *)bias, (const half2 *)gamma, 
                                                                                (const half2 *)beta, m, n, (const float2*)weight_amax, input1_amax_ptr);
  }
  else{
    assert(block.x <= 1024);
    add_bias_input_layernorm_COL32_int32I_DataTypeO<<<grid, block, 0, stream>>>((float *)output, input1, (const float*)input2, (const float*)bias, (const float*)gamma, 
                                                                                (const float*)beta, m, n, weight_amax, input1_amax_ptr);
  }
}

template void add_bias_input_layernorm_COL32_int32I_DataTypeO_kernelLauncher<float>(float* output, const int32_t* input1, const float* input2, const float* bias, const float* gamma, const float* beta, int m, int n, cudaStream_t stream, const float* weight_amax, const float *input1_amax_ptr);

template void add_bias_input_layernorm_COL32_int32I_DataTypeO_kernelLauncher<half>(half* output, const int32_t* input1, const half* input2, const half* bias, const half* gamma, const half* beta, int m, int n, cudaStream_t stream, const float* weight_amax, const float *input1_amax_ptr);

//src is the result of batch MM, whose size is batch_size*head_num*(seq_len, size_per_head), CUBLASLT_ORDER_COL32
//dst is of m = batch_size*seq_len, k(n) = head_num*size_per_head, CUBLASLT_ORDER_COL32
//grid(seq_len, batch_size)
//block(size_per_head/4, head_num)
//assume size_per_head is multiples of 32
__global__
void transpose_COL32_kernel(char4* dst, const int4* src, const int batch_size, const int seq_len, const int head_num, 
                            const int size_per_head, const float *v_buf_addBias_deQFactor, const float* qk_afterSM_deQFactor, const float* out_scale_ptr, 
                            const int batch_size_x_seq_len, const int seq_len_x_size_per_head)
{
  const float scale = __ldg(v_buf_addBias_deQFactor) * __ldg(qk_afterSM_deQFactor) * __ldg(out_scale_ptr);
  int threadIdx4 = threadIdx.x << 2;
  int batch_id = blockIdx.y;
  int seq_id = blockIdx.x;
  int head_id = threadIdx.y;
  //get the (row, col) output layout of m*k
  //m = batch_size*seq_len
  //k = head_num*size_per_head
  int mk_row = batch_id*seq_len + seq_id;
  int mk_col = head_id*size_per_head + threadIdx4;
  //get the (row, col) layout of COL32; leading dimension = 32*m = 32*batch_size*seq_len
  int COL32_row = (mk_row << 5) + (mk_col&31);
  //int COL32_col = mk_col >> 5;
  int outIdx = ((mk_col & 0xffffffe0)*batch_size_x_seq_len + COL32_row) >> 2;

  //get the (row, col) input layout of m'*k'
  //m' = seq_len
  //k' = size_per_head
  mk_row = seq_id;
  mk_col = threadIdx4;
  //get the (row, col) layout of COL32; leading dimension = 32*m' = 32*seq_len
  COL32_row = (mk_row << 5) + (mk_col&31);
  //COL32_col = mk_col >> 5;

  int inIdx = ((batch_id*head_num + head_id)*seq_len_x_size_per_head + (mk_col & 0xffffffe0)*seq_len + COL32_row) >> 2;
  char4 tmp;
  
  int4 srcTmp4 = __ldg(src + inIdx);
  tmp.x = float_to_int8_rn(srcTmp4.x*scale);
  tmp.y = float_to_int8_rn(srcTmp4.y*scale);
  tmp.z = float_to_int8_rn(srcTmp4.z*scale);
  tmp.w = float_to_int8_rn(srcTmp4.w*scale);
  dst[outIdx] = tmp;
}

void transpose_COL32_kernelLauncher(int8_t* dst, const int* src, const int batch_size, const int seq_len, const int head_num, 
                                    const int size_per_head, const float *v_buf_addBias_deQFactor, const float* qk_afterSM_deQFactor, 
                                    const float* out_scale_ptr, cudaStream_t stream){
  assert(size_per_head%32==0);
  transpose_COL32_kernel<<<dim3(seq_len, batch_size), dim3(size_per_head/4, head_num), 0, stream>>>((char4*)dst, (const int4*)src, batch_size, seq_len, head_num, size_per_head, v_buf_addBias_deQFactor, qk_afterSM_deQFactor, out_scale_ptr, batch_size*seq_len, seq_len*size_per_head);
}

//src is the result of batch MM, whose size is batch_size*head_num*(seq_len, size_per_head), CUBLASLT_ORDER_COL32
//dst is of m = batch_size*seq_len, k(n) = head_num*size_per_head, CUBLASLT_ORDER_COL32
//grid(seq_len, batch_size)
//block(size_per_head/4, head_num)
//assume size_per_head is multiples of 32
__global__
void transpose_COL32_kernel(int8_t* dst, const int8_t* src, const int batch_size, const int seq_len, const int head_num, 
                            const int size_per_head, const float *bmm2_deQFactor, const float* out_scale_ptr, 
                            const int batch_size_x_seq_len, const int seq_len_x_size_per_head)
{
  int threadIdx4 = threadIdx.x << 2;
  int batch_id = blockIdx.y;
  int seq_id = blockIdx.x;
  int head_id = threadIdx.y;
  //get the (row, col) output layout of m*k
  //m = batch_size*seq_len
  //k = head_num*size_per_head
  int mk_row = batch_id*seq_len + seq_id;
  int mk_col = head_id*size_per_head + threadIdx4;
  //get the (row, col) layout of COL32; leading dimension = 32*m = 32*batch_size*seq_len
  int COL32_row = (mk_row << 5) + (mk_col&31);
  int COL32_col = mk_col >> 5;
  int outIdx = ((COL32_col << 5)*batch_size_x_seq_len + COL32_row) >> 2;

  //get the (row, col) input layout of m'*k'
  //m' = seq_len
  //k' = size_per_head
  mk_row = seq_id;
  mk_col = threadIdx4;
  //get the (row, col) layout of COL32; leading dimension = 32*m' = 32*seq_len
  COL32_row = (mk_row << 5) + (mk_col&31);
  COL32_col = mk_col >> 5;

  int inIdx = ((batch_id*head_num + head_id)*seq_len_x_size_per_head + (COL32_col << 5 )*seq_len + COL32_row) >> 2;
  const char4* src_ptr4 = (const char4*)src;
  char4 *dst_ptr4 = (char4 *)dst;
  dst_ptr4[outIdx] = __ldg(src_ptr4 + inIdx);
}

void transpose_COL32_kernelLauncher(int8_t* dst, const int8_t* src, const int batch_size, const int seq_len, const int head_num, 
                                    const int size_per_head, const float *bmm2_deQFactor, 
                                    const float* out_scale_ptr, cudaStream_t stream){
  assert(size_per_head%32==0);
  transpose_COL32_kernel<<<dim3(seq_len, batch_size), dim3(size_per_head/4, head_num), 0, stream>>>(dst, src, batch_size, seq_len, head_num, size_per_head, bmm2_deQFactor, out_scale_ptr, batch_size*seq_len, seq_len*size_per_head);
}

//src is the result of batch MM, whose size is batch_size*head_num*(seq_len, size_per_head), CUBLASLT_ORDER_COL32
//dst is of m = valid_word_num, k(n) = head_num*size_per_head, CUBLASLT_ORDER_COL32
//grid(seq_len, batch_size)
//block(size_per_head/4, head_num)
//assume size_per_head is multiples of 32
__global__
void transpose_COL32_rebuild_padding_kernel(int8_t* dst, const int32_t* src, const int* sequence_id_map, const int valid_word_num, const int batch_size, 
                                            const int seq_len, const int head_num, const int size_per_head, const float *v_buf_addBias_deQFactor, 
                                            const float* qk_afterSM_deQFactor, const float* out_scale_ptr, const int seq_len_x_size_per_head)
{
  const float scale = __ldg(v_buf_addBias_deQFactor) * __ldg(qk_afterSM_deQFactor) * __ldg(out_scale_ptr);
  int threadIdx4 = threadIdx.x << 2;
  int batch_id = blockIdx.y;
  int seq_id = blockIdx.x;
  int head_id = threadIdx.y;
  //get the (row, col) output layout of m*k
  //m = valid_word_num
  //k = head_num*size_per_head
  int mk_row = __ldg(sequence_id_map + batch_id*seq_len + seq_id);
  if (mk_row >= 0){
    int mk_col = head_id*size_per_head + threadIdx4;
    //get the (row, col) layout of COL32; leading dimension = 32*m = 32*valid_word_num
    int COL32_row = (mk_row << 5) + (mk_col&31);
    int COL32_col = mk_col >> 5;
    int outIdx = ((COL32_col << 5)*valid_word_num + COL32_row) >> 2;

    //get the (row, col) input layout of m'*k'
    //m' = seq_len
    //k' = size_per_head
    mk_row = seq_id;
    mk_col = threadIdx4;
    //get the (row, col) layout of COL32; leading dimension = 32*m' = 32*seq_len
    COL32_row = (mk_row << 5) + (mk_col&31);
    COL32_col = mk_col >> 5;

    int inIdx = (batch_id*head_num + head_id)*seq_len_x_size_per_head + (COL32_col << 5 )*seq_len + COL32_row;
    char4 tmp;
    tmp.x = float_to_int8_rn(__ldg(src+inIdx)*scale);
    tmp.y = float_to_int8_rn(__ldg(src+inIdx+1)*scale);
    tmp.z = float_to_int8_rn(__ldg(src+inIdx+2)*scale);
    tmp.w = float_to_int8_rn(__ldg(src+inIdx+3)*scale);
    char4 *dst_ptr4 = (char4 *)dst;
    dst_ptr4[outIdx] = tmp;
  }
}

void transpose_COL32_rebuild_padding_kernelLauncher(int8_t* dst, const int* src, const int* sequence_id_map, const int valid_word_num, const int batch_size, 
                                                    const int seq_len, const int head_num, const int size_per_head, const float *v_buf_addBias_deQFactor, 
                                                    const float* qk_afterSM_deQFactor, const float* out_scale_ptr, cudaStream_t stream){
  assert(size_per_head%32==0);
  transpose_COL32_rebuild_padding_kernel<<<dim3(seq_len, batch_size), dim3(size_per_head/4, head_num), 0, stream>>>(dst, src, sequence_id_map, valid_word_num, batch_size, 
                                                                                                                    seq_len, head_num, size_per_head, v_buf_addBias_deQFactor, 
                                                                                                                    qk_afterSM_deQFactor, out_scale_ptr, seq_len*size_per_head);
}

//src is the result of batch MM, whose size is batch_size*head_num*(seq_len, size_per_head), CUBLASLT_ORDER_COL32
//dst is of m = valid_word_num, k(n) = head_num*size_per_head, CUBLASLT_ORDER_COL32
//grid(seq_len, batch_size)
//block(size_per_head/4, head_num)
//assume size_per_head is multiples of 32
__global__
void transpose_COL32_rebuild_padding_kernel(int8_t* dst, const int8_t* src, const int* sequence_id_map, const int valid_word_num, const int batch_size, 
                                            const int seq_len, const int head_num, const int size_per_head, const float *bmm2_deQFactor, const float* out_scale_ptr, const int seq_len_x_size_per_head)
{
  int threadIdx4 = threadIdx.x << 2;
  int batch_id = blockIdx.y;
  int seq_id = blockIdx.x;
  int head_id = threadIdx.y;
  //get the (row, col) output layout of m*k
  //m = valid_word_num
  //k = head_num*size_per_head
  int mk_row = __ldg(sequence_id_map + batch_id*seq_len + seq_id);
  if (mk_row >= 0){
    int mk_col = head_id*size_per_head + threadIdx4;
    //get the (row, col) layout of COL32; leading dimension = 32*m = 32*valid_word_num
    int COL32_row = (mk_row << 5) + (mk_col&31);
    int COL32_col = mk_col >> 5;
    int outIdx = ((COL32_col << 5)*valid_word_num + COL32_row) >> 2;

    //get the (row, col) input layout of m'*k'
    //m' = seq_len
    //k' = size_per_head
    mk_row = seq_id;
    mk_col = threadIdx4;
    //get the (row, col) layout of COL32; leading dimension = 32*m' = 32*seq_len
    COL32_row = (mk_row << 5) + (mk_col&31);
    COL32_col = mk_col >> 5;

    int inIdx = ((batch_id*head_num + head_id)*seq_len_x_size_per_head + (COL32_col << 5 )*seq_len + COL32_row) >> 2;
    
    const char4* src_ptr4 = (const char4*)src;
    
    char4 *dst_ptr4 = (char4 *)dst;
    dst_ptr4[outIdx] = __ldg(src_ptr4 + inIdx);
  }
}

void transpose_COL32_rebuild_padding_kernelLauncher(int8_t* dst, const int8_t* src, const int* sequence_id_map, const int valid_word_num, const int batch_size, 
                                                    const int seq_len, const int head_num, const int size_per_head, const float *bmm2_deQFactor,
                                                    const float* out_scale_ptr, cudaStream_t stream){
  assert(size_per_head%32==0);
  transpose_COL32_rebuild_padding_kernel<<<dim3(seq_len, batch_size), dim3(size_per_head/4, head_num), 0, stream>>>(dst, src, sequence_id_map, valid_word_num, batch_size, 
                                                                                                                    seq_len, head_num, size_per_head, bmm2_deQFactor, 
                                                                                                                    out_scale_ptr, seq_len*size_per_head);
}


__global__
void quantized_kernel(char4 *dst, const float4* src, const int size_div_4, const float* scale_ptr)
{
  int tid = (blockIdx.x*blockDim.x + threadIdx.x);
  if (tid < size_div_4){
    const float scale = __ldg(scale_ptr);
    char4 tmp;
    const float4 floatTmp = __ldg(src + tid);
    tmp.x = float_to_int8_rn(floatTmp.x*scale);
    tmp.y = float_to_int8_rn(floatTmp.y*scale);
    tmp.z = float_to_int8_rn(floatTmp.z*scale);
    tmp.w = float_to_int8_rn(floatTmp.w*scale);
    dst[tid] = tmp;
  }
}

__global__
void quantized_kernel(char4 *dst, const half2* src, const int size_div_4, const float* scale_ptr)
{
  int tid = (blockIdx.x*blockDim.x + threadIdx.x);
  if (tid < size_div_4){
    const float scale = __ldg(scale_ptr);
    char4 tmp;
    int src_id = tid << 1;
    
    const half2 half2Tmp = __ldg(src + src_id);
    tmp.x = float_to_int8_rn(static_cast<float>(half2Tmp.x)*scale);
    tmp.y = float_to_int8_rn(static_cast<float>(half2Tmp.y)*scale);
    
    const half2 half2Tmp2 = __ldg(src + src_id + 1);
    tmp.z = float_to_int8_rn(static_cast<float>(half2Tmp2.x)*scale);
    tmp.w = float_to_int8_rn(static_cast<float>(half2Tmp2.y)*scale);
    dst[tid] = tmp;
  }
}

template <typename T>
void quantized_kernelLauncher(int8_t* dst, const T * src, const int size, const float* scale_ptr, cudaStream_t stream)
{
   assert(size % 4 == 0);
   dim3 grid((size+255)/256);
   dim3 block(64);
   if (sizeof(T) == sizeof(float))
     quantized_kernel<<<grid, block, 0, stream>>>((char4*)dst, (const float4*)src, size/4, scale_ptr);
   else if (sizeof(T) == sizeof(half))
     quantized_kernel<<<grid, block, 0, stream>>>((char4*)dst, (const half2*)src, size/4, scale_ptr);
}

template void quantized_kernelLauncher<float>(int8_t* dst, const float * src, const int size, const float* scale_ptr, cudaStream_t stream);

template void quantized_kernelLauncher<half>(int8_t* dst, const half * src, const int size, const float* scale_ptr, cudaStream_t stream);

template <typename T>
__global__
void dequantized_kernel(T *dst, const int8_t* src, const int size, const float *scale_ptr)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < size){
    float tmp = float(src[tid]);
    dst[tid] = T(float(tmp) *  __ldg(scale_ptr));
  }
}

template <typename T>
void dequantized_kernelLauncher(T* dst, const int8_t * src, const int size, const float *scale_ptr, cudaStream_t stream)
{
   dim3 grid((size+255)/256);
   dim3 block(256);
   dequantized_kernel<T><<<grid, block, 0, stream>>>(dst, src, size, scale_ptr);
}
template void dequantized_kernelLauncher<float>(float* dst, const int8_t * src, const int size, const float *scale_ptr, cudaStream_t stream);

template void dequantized_kernelLauncher<half>(half* dst, const int8_t * src, const int size, const float *scale_ptr, cudaStream_t stream);

template void dequantized_kernelLauncher<int32_t>(int32_t* dst, const int8_t * src, const int size, const float *scale_ptr, cudaStream_t stream);


//layout should be COL32
template<typename T>
__global__ void rebuild_sequence_length_padding_COL32(const T* src, T* tgt,
                                                      const int* mask_offset, const int m,
                                                      const int n, const int tgt_m)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int tgt_seq_id = bid + __ldg(mask_offset + bid);
  const int src_seq_id = bid;
  const int src_col32_lda = m << 5;
  const int tgt_col32_lda = tgt_m << 5;
  const int src_row_tmp = src_seq_id << 5;
  const int tgt_row_tmp = tgt_seq_id << 5;
  for(int i = tid; i < n; i += blockDim.x)
  {
    int col = i >> 5;
    int src_row = src_row_tmp + (i & 31);
    int tgt_row = tgt_row_tmp + (i & 31);

    tgt[col*tgt_col32_lda + tgt_row] = __ldg(src + col*src_col32_lda + src_row);
  }
}

//for half input
//layout should be COL32
template<>
__global__ void rebuild_sequence_length_padding_COL32(const half2* src, half2* tgt,
                                                      const int* mask_offset, const int m,
                                                      const int n, const int tgt_m)
{
  const int tid2 = threadIdx.x << 1;
  const int bid = blockIdx.x;
  const int tgt_seq_id = bid + __ldg(mask_offset + bid);
  const int src_seq_id = bid;
  const int src_col32_lda = m << 5;
  const int tgt_col32_lda = tgt_m << 5;
  const int src_row_tmp = src_seq_id << 5;
  const int tgt_row_tmp = tgt_seq_id << 5;
  for(int i = tid2; i < n; i += 2*blockDim.x)
  {
    int col = i >> 5;
    int src_row = src_row_tmp + (i & 31);
    int tgt_row = tgt_row_tmp + (i & 31);

    tgt[(col*tgt_col32_lda + tgt_row) >> 1] = __ldg(src + ((col*src_col32_lda + src_row) >> 1));
  }
}

//for int8 input
//layout should be COL32
template<>
__global__ void rebuild_sequence_length_padding_COL32(const char4* src, char4* tgt,
                                                      const int* mask_offset, const int m,
                                                      const int n, const int tgt_m)
{
  const int tid4 = threadIdx.x << 2;
  const int bid = blockIdx.x;
  const int tgt_seq_id = bid + __ldg(mask_offset + bid);
  const int src_seq_id = bid;
  const int src_col32_lda = m << 5;
  const int tgt_col32_lda = tgt_m << 5;
  const int src_row_tmp = src_seq_id << 5;
  const int tgt_row_tmp = tgt_seq_id << 5;
  for(int i = tid4; i < n; i += 4*blockDim.x)
  {
    int col = i >> 5;
    int src_row = src_row_tmp + (i & 31);
    int tgt_row = tgt_row_tmp + (i & 31);

    tgt[(col*tgt_col32_lda + tgt_row) >> 2] = __ldg(src + ((col*src_col32_lda + src_row) >> 2));
  }
}

template<typename T>
void rebuild_sequence_length_padding_COL32_kernelLauncher(const T* src, T* tgt,
                                                          const int* mask_offset, const int m,
                                                          const int n, const int tgt_m, cudaStream_t stream)
{
  // src: [valid_word_num, hidden_dim]
  // tgt: [batch_size*max_seq_len, hidden_dim]
  dim3 block(256);
  if (sizeof(T) == sizeof(half))
  {
    if (n/2 < 256)
      block.x = n/2;
    rebuild_sequence_length_padding_COL32<<<m, block, 0, stream>>>((const half2*)src, (half2*)tgt, mask_offset, m, n, tgt_m);
  }
  else if (sizeof(T) == sizeof(int8_t))
  {
    if (n/4 < 256)
      block.x = n/4;
    rebuild_sequence_length_padding_COL32<<<m, block, 0, stream>>>((const char4*)src, (char4*)tgt, mask_offset, m, n, tgt_m);
  }
  else  
    rebuild_sequence_length_padding_COL32<<<m, block, 0, stream>>>(src, tgt, mask_offset, m, n, tgt_m);
}

template 
void rebuild_sequence_length_padding_COL32_kernelLauncher(const int8_t* src, int8_t* tgt,
                                                          const int* mask_offset, const int m,
                                                          const int n, const int tgt_m, cudaStream_t stream);
                                                          
template 
void rebuild_sequence_length_padding_COL32_kernelLauncher(const half* src, half* tgt,
                                                          const int* mask_offset, const int m,
                                                          const int n, const int tgt_m, cudaStream_t stream);
                                                          
template 
void rebuild_sequence_length_padding_COL32_kernelLauncher(const float* src, float* tgt,
                                                          const int* mask_offset, const int m,
                                                          const int n, const int tgt_m, cudaStream_t stream);

}//namespace 




