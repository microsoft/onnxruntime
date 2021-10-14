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
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

namespace fastertransformer
{

/* ********************************** common kernel *********************************** */

void transpose_COL32_kernelLauncher(int8_t *dst, const int *src,
                                    const int batch_size, const int seq_len,
                                    const int head_num, const int size_per_head,
                                    const float *v_buf_addBias_deQFactor, const float *qk_afterSM_deQFactor,
                                    const float *out_scale_ptr, cudaStream_t stream);
                                    
void transpose_COL32_kernelLauncher(int8_t *dst, const int8_t *src,
                                    const int batch_size, const int seq_len,
                                    const int head_num, const int size_per_head,
                                    const float *bmm2_deQFactor, 
                                    const float *out_scale_ptr, cudaStream_t stream);                                    

void transpose_COL32_rebuild_padding_kernelLauncher(int8_t *dst, const int *src,
                                                    const int *sequence_id_map, const int valid_word_num,
                                                    const int batch_size, const int seq_len,
                                                    const int head_num, const int size_per_head,
                                                    const float *v_buf_addBias_deQFactor,
                                                    const float *qk_afterSM_deQFactor, const float *out_scale_ptr,
                                                    cudaStream_t stream);

void transpose_COL32_rebuild_padding_kernelLauncher(int8_t* dst, const int8_t* src, 
                                                    const int* sequence_id_map, const int valid_word_num, 
                                                    const int batch_size, const int seq_len, 
                                                    const int head_num, const int size_per_head, 
                                                    const float *bmm2_deQFactor,
                                                    const float* out_scale_ptr, cudaStream_t stream);

template <typename T>
void add_bias_input_layernorm_COL32_int32I_DataTypeO_kernelLauncher(T *output, const int32_t *input1,
                                                                    const T *input2, const T *bias,
                                                                    const T *gamma, const T *beta,
                                                                    int m, int n,
                                                                    cudaStream_t stream, const float *weight_amax,
                                                                    const float *input1_amax_ptr);

template <typename T>
void add_bias_input_layernorm_COL32_int8I_DataTypeO_kernelLauncher(T *output, const int8_t *input1,
                                                                   const int8_t *input2, const T *bias,
                                                                   const T *gamma, const T *beta,
                                                                   int m, int n,
                                                                   cudaStream_t stream, 
                                                                   const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr);                                                                 
                                                                 
template <typename T>
void add_bias_input_layernorm_COL32_int8IO_kernelLauncher(int8_t *output, const int8_t *input1,
                                                          const int8_t *input2, const T *bias,
                                                          const T *gamma, const T *beta,
                                                          int m, int n,
                                                          cudaStream_t stream,
                                                          const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr,
                                                          const float *output_scale_ptr);                                                                 

template <typename T>
void add_bias_act_COL32_int32I_int8O_kernelLauncher(int8_t *out, const int32_t *input,
                                                    const T *bias, const int m,
                                                    const int n, cudaStream_t stream,
                                                    const float *weight_amax, const float *input_deQFactor_div127_ptr,
                                                    const float *out_scale_ptr);
                                                    
template <typename T>
void add_bias_act_COL32_int8IO_kernelLauncher(int8_t *out, const int8_t* input, 
                                              const T* bias, const int m, const int n, 
                                              cudaStream_t stream, const float* input_deQFactor_ptr, const float* out_scale_ptr);

template <typename T>
void transposeMatrix_COL32ToColMajor_kernelLauncher(T* dst, const T* src, 
                                                    const int m, const int n, 
                                                    cudaStream_t stream);
                                         
template <typename T>
void transposeMatrix_colMajorToCOL32_kernelLauncher(T* dst, const T* src, 
                                                    const int m, const int n, 
                                                    cudaStream_t stream);
                                                   
template <typename T>
void transposeMatrix_colMajorToCOL32_quantize_kernelLauncher(int8_t* dst, const T* src, 
                                                             const int m, const int n, 
                                                             const float* scale_ptr, cudaStream_t stream);

template <typename T>
void quantized_kernelLauncher(int8_t *dst, const T *src,
                              const int size, const float *scale_ptr,
                              cudaStream_t stream);

template <typename T>
void dequantized_kernelLauncher(T *dst, const int8_t *src,
                                const int size, const float *scale_ptr,
                                cudaStream_t stream);

template<typename T>
void rebuild_sequence_length_padding_COL32_kernelLauncher(const T* src, T* tgt,
                                                          const int* mask_offset, const int m,
                                                          const int n, const int tgt_m, cudaStream_t stream);

void rowMajorToCOL32_kernelLauncher(int8_t* dst, const int8_t* src, 
                                    const int m, const int n, cudaStream_t stream);

} //namespace fastertransformer
