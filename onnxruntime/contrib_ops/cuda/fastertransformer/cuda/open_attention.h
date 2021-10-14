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

#pragma once

#include "contrib_ops/cuda/fastertransformer/utils/allocator.h"
#include "contrib_ops/cuda/fastertransformer/cuda/multi_head_attention.h"
#include "contrib_ops/cuda/fastertransformer/cuda/attention_kernels.cuh"
#include "contrib_ops/cuda/fastertransformer/cuda/transformer_kernels.cuh"
#include "contrib_ops/cuda/fastertransformer/cuda/cuda_kernels.h"
#include "contrib_ops/cuda/fastertransformer/cuda/cuda_int8_kernels.h"
#include "contrib_ops/cuda/fastertransformer/gemm_test/encoder_gemm_func.h"
#include "contrib_ops/cuda/fastertransformer/gemm_test/encoder_igemm_func.h"
#include "contrib_ops/cuda/fastertransformer/utils/functions.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "contrib_ops/cuda/fastertransformer/trt_fused_multihead_attention/qkvToContext.h"

namespace fastertransformer{
namespace cuda{

void trt_add_QKV_bias_transpose_debug_kernelLauncher(
  const half* query_buf, const half* bias_Q,
  const half* key_buf, const half* bias_K,
  const half* value_buf, const half* bias_V,
  half* context_buf, 
  const int valid_word_num, 
  const int head_num, const int size_per_head,
  cudaStream_t stream); // Used to debug the trt_add_QKV_bias kernel

template <typename T>
void add_QK_bias_transform_kernelLauncher(int8_t *q_buf, int8_t *k_buf, const int32_t* Q, const T* bias_Q, 
                                          const int32_t* K, const T* bias_K, const int batch_size, 
                                          const int seq_len, const int head_num, const int size_per_head, 
                                          const float * q_weight_amax, const float *q_input_deQFactor_div127_ptr, 
                                          const float * k_weight_amax, const float *k_input_deQFactor_div127_ptr, 
                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);
                                          
template <typename T>
void add_QK_bias_transform_kernelLauncher(int8_t *q_buf, int8_t *k_buf, const int8_t* Q, const T* bias_Q, 
                                          const int8_t* K, const T* bias_K, const int batch_size, 
                                          const int seq_len, const int head_num, const int size_per_head, 
                                          const float *q_input_deQFactor_ptr, const float *k_input_deQFactor_ptr, 
                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);                                          

template <typename T>
void add_V_bias_transform_kernelLauncher(int8_t *v_buf, const int32_t *V, const T *V_bias, 
                                         const int batch_size, const int seq_len, 
                                         const int head_num, const int size_per_head, 
                                         const float* weight_amax, 
                                         const float *input_deQFactor_div127_ptr, const float *out_scale_ptr, 
                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);
                                         
template <typename T>
void add_V_bias_transform_kernelLauncher(int8_t *v_buf, const int8_t *V, const T *V_bias, const int batch_size, 
                                         const int seq_len, const int head_num, const int size_per_head,
                                         const float *input_deQFactor_ptr, const float *out_scale_ptr, 
                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);
                                         
void mappingRemovePaddingData_kernelLauncher(const int batch_size, const int seq_len, 
                                             const int valid_word_num, int *mapping, 
                                             const int* sequence_id_offset, cudaStream_t stream);
                                             
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
                                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);
                                                          
template <typename T>
void add_QK_bias_transform_rebuild_padding_kernelLauncher(int8_t *q_buf, int8_t *k_buf, const int8_t* Q, const T* bias_Q, 
                                                          const int8_t* K, const T* bias_K, const int* sequence_id_offset, 
                                                          const int valid_word_num, 
                                                          const int batch_size, const int seq_len, 
                                                          const int head_num, const int size_per_head,  
                                                          const float *q_deQFactor_ptr,  const float *k_deQFactor_ptr, 
                                                          const float *q_output_scale_ptr, const float *k_output_scale_ptr,
                                                          bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);
                                                          
template <typename T>
void add_V_bias_transform_rebuild_padding_kernelLauncher(int8_t *v_buf, const int32_t *V, const T *V_bias, 
                                                         const int* sequence_id_map, const int valid_word_num, 
                                                         const int batch_size, const int seq_len, 
                                                         const int head_num, const int size_per_head, 
                                                         const float* weight_amax, 
                                                         const float *input_deQFactor_div127_ptr, 
                                                         const float *out_scale_ptr, 
                                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);
                                                         
template <typename T>
void add_V_bias_transform_rebuild_padding_kernelLauncher(int8_t *v_buf, const int8_t *V, const T *V_bias, 
                                                         const int* sequence_id_map, const int valid_word_num, 
                                                         const int batch_size, const int seq_len, 
                                                         const int head_num, const int size_per_head,
                                                         const float *deQFactor_ptr, const float *out_scale_ptr, 
                                                         bool use_ORDER_COL32_2R_4R4, cudaStream_t stream);  

template <typename T>
void softmax_COL32_kernelLauncher(int8_t* qk_buf, const int32_t* qk_int_buf, const T* attr_mask, 
                                  const int batch_size, const int head_num, const int seq_len, 
                                  const float scalar1a, const float *scalar1b, const float *scalar1c, 
                                  const float *amax_ptr, cudaStream_t stream);

template <typename T>
void softmax_COL32_kernelLauncher(int8_t* qk_buf, const int8_t* qk_int_buf, const T* attr_mask, 
                                  const int batch_size, const int head_num, const int seq_len, 
                                  const float scalar1a, const float *scalar1b, const float *amax_ptr, 
                                  cudaStream_t stream);

template<typename T>
void add_QKV_bias_rebuild_padding_kernelLauncher(T* Q, const T* bias_Q, T* K, const T* bias_K, 
                                                 T* V, const T* bias_V, T* q_buf, T* k_buf, T* v_buf, 
                                                 const int batch_size, const int seq_len, 
                                                 const int head_num, const int size_per_head, const int valid_word_num, 
                                                 const int* mask_offset, cudaStream_t stream);
                                                 
template<typename T>
void transpose_kernelLauncher(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head, cudaStream_t stream); 

template<typename T>
void transpose_rebuild_padding_kernelLauncher(T* src, T* dst, const int valid_word_num,
                                              const int batch_size, const int seq_len, 
                                              const int head_num, const int size_per_head, 
                                              const int* mask_offset, cudaStream_t stream);

template<OperationType OpType_>
class OpenMultiHeadAttentionTraits;

template<>
class OpenMultiHeadAttentionTraits<OperationType::FP32>
{
 public:
  typedef float DataType;
  static cudaDataType_t const computeType = CUDA_R_32F;
  static cudaDataType_t const scaleType = CUDA_R_32F;
  static cudaDataType_t const AType = CUDA_R_32F;
  static cudaDataType_t const BType = CUDA_R_32F;
  static cudaDataType_t const CType = CUDA_R_32F;
  //others
};

template<>
class OpenMultiHeadAttentionTraits<OperationType::FP16>
{
 public:
  typedef half DataType;
  static cudaDataType_t const computeType = CUDA_R_16F;
  static cudaDataType_t const scaleType = CUDA_R_16F;
  static cudaDataType_t const AType = CUDA_R_16F;
  static cudaDataType_t const BType = CUDA_R_16F;
  static cudaDataType_t const CType = CUDA_R_16F;
  //others
};

/**
 * Multi-head attetion open sourced
 */
template<OperationType OpType_>
class OpenMultiHeadAttention: IMultiHeadAttention<OpType_>
{
 private:
  typedef OpenMultiHeadAttentionTraits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  IAllocator* allocator_ = NULL;
  MultiHeadInitParam<DataType_> param_;

  //algo for batch matrix multiplication in unfused mha
  int cublasBmmAlgo_[2];
  std::map<std::string, cublasLtMatmulAlgo_info> cublasAlgoMap_;
  std::map<std::string, int> parameterMap_;
  bool is_fuse_QKV_;

  DataType_* buf_ = NULL;
  DataType_* query_buf_;
  DataType_* key_buf_;
  DataType_* value_buf_;
  DataType_* q_buf_;
  DataType_* k_buf_;
  DataType_* v_buf_;
  DataType_* qk_buf_;
  DataType_* transpose_dst_;
  
  DataType_** qkv_kernel_;
  DataType_** qkv_input_;
  DataType_** qkv_buf_;

  void* cublas_workspace_;

  void* trt_attn_workspace_;

  const float *query_weight_amax_list, *key_weight_amax_list, *value_weight_amax_list;

  int sm_;
  int batch_size_;
  int from_seq_len_;
  int to_seq_len_;
  int head_num_;
  int size_per_head_;
  float q_scaling_;
  //int8_mode == 0 -- not use int8
  //int8_mode == 1 -- use int8; without quantized residual; when (batch*seqLen >= 512) or (seqLen % 32 !=0 ), using trt fused mha
  //int8_mode == 2 -- use int8; with quantized residual; with trt fused mha
  //int8_mode == 3 -- use int8; with quantized residual; without trt fused mha
  int int8_mode_ = 0;
  int* sequence_id_map_;
  int* Q_int_buf_;
  int* K_int_buf_;
  int* V_int_buf_;
  int* qk_int_buf_;
  int* transpose_dst_int_buf_;

  bool allow_gemm_test_ = false;
  bool use_ORDER_COL32_2R_4R4_ = false;
  std::unique_ptr<MHARunner> dispatcher_fp16, dispatcher_int8;

 public:

  void getCublasBmmAlgoFromMap()
  {
    int batchCount, m, n, k, dataType;
    if (std::is_same<half, DataType_>::value)
      dataType = HALF_DATATYPE;
    else
      dataType = FLOAT_DATATYPE;
    //bmm1    
    batchCount = batch_size_*head_num_;
    m = from_seq_len_; 
    n = from_seq_len_; 
    k = size_per_head_; 
    char mark[256];
    sprintf(mark, "%d_%d_%d_%d_%d", batchCount, n, m, k, dataType);
    if (cublasAlgoMap_.find(mark) != cublasAlgoMap_.end())
    {
      cublasBmmAlgo_[0] = cublasAlgoMap_[mark].algoId;
    }
    else
    {
      cublasBmmAlgo_[0] = dataType == FLOAT_DATATYPE ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    //bmm2
    batchCount = batch_size_*head_num_;
    m = from_seq_len_;
    n = size_per_head_;
    k = from_seq_len_;
    sprintf(mark, "%d_%d_%d_%d_%d", batchCount, n, m, k, dataType);
    if (cublasAlgoMap_.find(mark) != cublasAlgoMap_.end())
    {
      cublasBmmAlgo_[1] = cublasAlgoMap_[mark].algoId;
    }
    else
    {
      cublasBmmAlgo_[1] = dataType == FLOAT_DATATYPE ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
  }

  void judgeFusedQKV()
  {
    is_fuse_QKV_ = false;
    int m, n, k, dataType;
    if (std::is_same<half, DataType_>::value)
      dataType = HALF_DATATYPE;
    else
      dataType = FLOAT_DATATYPE;

    m = batch_size_*from_seq_len_;
    n = head_num_*size_per_head_;
    k = head_num_*size_per_head_;
    char mark[256], mark2[256];
    sprintf(mark, "1_%d_%d_%d_%d", n, m, k, dataType);
    sprintf(mark2, "3_%d_%d_%d_%d", n, m, k, dataType);
    if (
        cublasAlgoMap_.find(mark) != cublasAlgoMap_.end() && 
        cublasAlgoMap_.find(mark2) != cublasAlgoMap_.end() &&
        3*cublasAlgoMap_[mark].exec_time > cublasAlgoMap_[mark2].exec_time
       )
    {
        is_fuse_QKV_ = true;
    }
  }

  //free buffer for OpenMultiHeadAttention
  void freeBuffer()
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    if (buf_ != NULL)
    {
      if (allocator_ == NULL)
      {
        printf("[ERROR][OpenMultiHeadAttention][freeBuffer] allocator_ is NULL!\n");
        exit(-1);
      }
      allocator_->free(buf_);
      buf_ = NULL;
    }
  }

  size_t get_workspace_size()
  {
    size_t size = 0;
    
    const int buf_size = batch_size_ * head_num_ * from_seq_len_ * size_per_head_;
    const int qk_buf_size = batch_size_ * head_num_ * from_seq_len_ * from_seq_len_;
    const int seq_len_padded = (from_seq_len_ + 31)/32*32;
    const int padded_buf_size = batch_size_ * head_num_ * seq_len_padded * size_per_head_;
    const int padded_qk_buf_size = batch_size_ * head_num_ * seq_len_padded * seq_len_padded;

    if(int8_mode_ != 0)
    {
             //query_buf_(Q_int_buf_) key_buf_(K_int_buf_) value_buf_(V_int_buf_) qk_int_buf_ transpose_dst_(transpose_dst_int_buf_)
      size = sizeof(int) * (4*buf_size + padded_qk_buf_size) +
             //int8 q_buf_ k_buf_ v_buf_ qk_buf_
             sizeof(int8_t) * (3*padded_buf_size + padded_qk_buf_size) +
             //sequence_id_map 
             (batch_size_*from_seq_len_)*sizeof(int) +
             //trt_attn_workspace_
             (dispatcher_int8.get() ? dispatcher_int8->getWorkspaceSize() : 0);

    }
    else
    {
      size = sizeof(DataType_) * (buf_size * 7 + qk_buf_size) + sizeof(DataType_*) * 9 + 
                (dispatcher_fp16.get() ? dispatcher_fp16->getWorkspaceSize() : 0);
    }
    return size;
  }

  //allocate buffer for OpenMultiHeadAttention
  //read config again if hasChangedConfig == true
  void allocateBuffer(IAllocator* allocator, void* cublas_workspace, int batch_size, int from_seq_len, int to_seq_len,
                      int head_num, int size_per_head, bool hasChangedConfig, bool use_trt_kernel)
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    if (allocator == NULL)
    {
      printf("[ERROR][OpenMultiHeadAttention][allocateBuffer] allocator == NULL!\n");
      exit(-1);
    }
    
    try
    {
      //only allocate new buffer when buf_ is empty
      //if buf_ is not empty, use previous allocated one
      //this can ensure consistency between (allocator_, batch_size_, ...) and buf_
      if (buf_ != NULL){
        printf("[ERROR][OpenMultiHeadAttention][allocateBuffer] previous buffer is not freed, use previous one. To allocate new buffer, please use freeBuffer() to free previous buffer first.\n");
        exit(-1);
      }
      else
      {
        allocator_ = allocator;
        batch_size_ = batch_size;
        from_seq_len_ = from_seq_len;
        to_seq_len_ = to_seq_len;
        head_num_ = head_num;
        size_per_head_ = size_per_head;
        cublas_workspace_ = cublas_workspace;

        const int buf_size = batch_size_ * head_num_ * from_seq_len_ * size_per_head_;
        const int qk_buf_size = batch_size_ * head_num_ * from_seq_len_ * from_seq_len_;

        if (int8_mode_ != 0)
        {
          if ((int8_mode_ == 1 && (batch_size_*from_seq_len_ >= 512 || (from_seq_len_ % 32 != 0))) || int8_mode_ == 2)
          {
            if (use_trt_kernel && (sm_ == kSM_86 || sm_ == kSM_80 || sm_ == kSM_75 || sm_ == kSM_72) && size_per_head_ == 64)
            {
              //try
              {
                dispatcher_int8.reset(new FusedMHARunnerInt8v2(head_num_, size_per_head_, sm_));    
              }
            }
          }
          const int seq_len_padded = (from_seq_len_ + 31)/32*32;
          const int padded_buf_size = batch_size_ * head_num_ * seq_len_padded * size_per_head_;
          const int padded_qk_buf_size = batch_size_ * head_num_ * seq_len_padded * seq_len_padded;
          
          buf_ = (DataType_*) allocator_->malloc(get_workspace_size(), false);
          if (buf_ == NULL)
            throw std::runtime_error(std::string("Allocator failed to allocate internal buffer."));
          Q_int_buf_ = (int *)(buf_);
          K_int_buf_ = Q_int_buf_ + buf_size;
          V_int_buf_ = K_int_buf_ + buf_size;
          transpose_dst_int_buf_ = V_int_buf_ + buf_size;
          qk_int_buf_ = transpose_dst_int_buf_ + buf_size;
          q_buf_ = (DataType_*)(qk_int_buf_ + padded_qk_buf_size);
          //the actual size is calculated with int8_t datatype 
          k_buf_ = (DataType_*)((int8_t*)q_buf_ + padded_buf_size);
          v_buf_ = (DataType_*)((int8_t*)k_buf_ + padded_buf_size);
          qk_buf_ = (DataType_*)((int8_t*)v_buf_ + padded_buf_size);
          sequence_id_map_ = (int*)((int8_t*)qk_buf_ + padded_qk_buf_size);
          trt_attn_workspace_ = (void*)(sequence_id_map_ + (batch_size_*from_seq_len_));
        }
        else
        {
        if (use_trt_kernel && (sm_ == kSM_70 || sm_ == kSM_86 || sm_ == kSM_80 || sm_ == kSM_75 || sm_ == kSM_72) && size_per_head_ == 64)
            dispatcher_fp16.reset(new FusedMHARunnerFP16v2(head_num_, size_per_head_, sm_, q_scaling_));
          buf_ = (DataType_*) allocator_->malloc(get_workspace_size(), false);
          if (buf_ == NULL)
            throw std::runtime_error(std::string("Allocator failed to allocate internal buffer."));
          query_buf_ = buf_;
          key_buf_ = buf_ + buf_size;
          value_buf_ = buf_ + 2 * buf_size;
          q_buf_ = buf_ + 3 * buf_size;
          k_buf_ = buf_ + 4 * buf_size;
          v_buf_ = buf_ + 5 * buf_size;
          qk_buf_ = buf_ + 6 * buf_size;
          transpose_dst_ = qk_buf_ + qk_buf_size;
          qkv_kernel_ = (DataType_**)(transpose_dst_ + buf_size);
          qkv_input_ = qkv_kernel_ + 3;
          qkv_buf_ = qkv_input_ + 3;
          trt_attn_workspace_ = (void*)(qkv_buf_ + 3);
        }
      }

      //no gemm test in OpenMultiHeadAttention 
      //if config changes, read config again
      if (hasChangedConfig)
      {
        int isConfigExist = -1;
        if (int8_mode_ != 0)
        {
          isConfigExist = access(IGEMM_CONFIG, 0);
        }
        else
        {
          isConfigExist = access(GEMM_CONFIG, 0);
        }
        if (isConfigExist == -1)
          printf("[WARNING][OpenMultiHeadAttention] %s is not found; using default GEMM algo\n", int8_mode_ != 0 ? IGEMM_CONFIG : GEMM_CONFIG);
        else
        {
          readAlgoFromConfig(int8_mode_, cublasAlgoMap_, parameterMap_, false);
        }
      }

      if (int8_mode_ == 0)
      {
        getCublasBmmAlgoFromMap();
        judgeFusedQKV();
      }
    }
    catch(std::runtime_error& error)
    {
      throw error;
    }
  }

  //Ctor
  OpenMultiHeadAttention(int int8_mode=0, bool allow_gemm_test=false, bool use_ORDER_COL32_2R_4R4=false, int sm = 75, float q_scaling=1.0) : 
    int8_mode_(int8_mode), allow_gemm_test_(allow_gemm_test), use_ORDER_COL32_2R_4R4_(use_ORDER_COL32_2R_4R4), sm_(sm), q_scaling_(q_scaling)
   {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    //sm_ = getSMVersion();

    try
    {
      int isConfigExist = -1;
      if (int8_mode_ != 0)
        isConfigExist = access(IGEMM_CONFIG, 0);
      else
        isConfigExist = access(GEMM_CONFIG, 0);

      if (isConfigExist == -1)
      {
        if (!allow_gemm_test_)
        {
          printf("[WARNING][OpenMultiHeadAttention] %s is not found; using default GEMM algo\n", int8_mode_ != 0 ? IGEMM_CONFIG : GEMM_CONFIG);
        }
      }
      else
      {
        readAlgoFromConfig(int8_mode_, cublasAlgoMap_, parameterMap_, false);
      }
    }
    catch(std::runtime_error& error)
    {
      throw error;
    }
  }

  OpenMultiHeadAttention(const OpenMultiHeadAttention *attention)
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    sm_ = attention->sm_;
    int8_mode_ = attention->int8_mode_;
    allow_gemm_test_ = attention->allow_gemm_test_;
    q_scaling_ = attention->q_scaling_;

    for(int i = 0; i < 2; i++) cublasBmmAlgo_[i] = attention->cublasBmmAlgo_[i];
    cublasAlgoMap_ = attention->cublasAlgoMap_;
    parameterMap_ = attention->parameterMap_;
    is_fuse_QKV_ = attention->is_fuse_QKV_;
    use_ORDER_COL32_2R_4R4_ = attention->use_ORDER_COL32_2R_4R4_;
  }

  void multiHeadAttr_nofuse_kernelLauncher(
      cudaStream_t stream,
      cublasHandle_t cublas_handle,
      cublasLtHandle_t cublaslt_handle,
      DataType_* Q,
      const DataType_* bias_Q,
      DataType_* K,
      const DataType_* bias_K,
      DataType_* V,
      const DataType_* bias_V,
      const DataType_* attr_mask,
      DataType_* dst,
      const int batch_size,
      const int seq_len,
      const int head_num,
      const int size_per_head,
      const int int8_mode_,
      const DataType_ scalar)
  {
    const int k = head_num * size_per_head;

    if (int8_mode_ != 0)
    {
      //var for int8
      const float*Qbias_amax_ptr, *Kbias_amax_ptr, *Vbias_amax_ptr, *bmm1_amax_ptr, *Softmax_amax_ptr, *bmm2_amax_ptr, *in_amax_ptr, *Q_aftergemm_amax_ptr, *K_aftergemm_amax_ptr, *V_aftergemm_amax_ptr;
      Qbias_amax_ptr = param_.amaxList + 8;
      Kbias_amax_ptr = param_.amaxList + 16;
      Vbias_amax_ptr = param_.amaxList + 24;
      Softmax_amax_ptr = param_.amaxList + 32;
      bmm2_amax_ptr = param_.amaxList + 36;
      Q_aftergemm_amax_ptr = param_.amaxList + 4;
      K_aftergemm_amax_ptr = param_.amaxList + 12;
      V_aftergemm_amax_ptr = param_.amaxList + 20;
      bmm1_amax_ptr = param_.amaxList + 28;
      in_amax_ptr = param_.amaxList;
      
	  if (size_per_head % 32 != 0)
      {
        printf("[ERROR][FT][multiHeadAttr_nofuse_kernelLauncher] int8 unfused mha kernel only works when size_per_head %% 32 == 0.\n");
        exit(-1);
      }
      if ((seq_len % 32 != 0) && int8_mode_ == 1)
      {
        printf("[ERROR][FT][multiHeadAttr_nofuse_kernelLauncher] int8 mode1 unfused mha kernel only works when seq_len %% 32 == 0.\n");
        exit(-1);
	    }
      const int seq_len_padded = (seq_len + 31)/32*32;

      if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
      {
        if (int8_mode_ == 1)
        {
          add_QK_bias_transform_kernelLauncher((int8_t*)q_buf_, (int8_t*)k_buf_, 
                                               (const int32_t*) Q, bias_Q, (const int32_t*) K, bias_K, 
                                               batch_size, seq_len, head_num, size_per_head, 
                                               query_weight_amax_list, in_amax_ptr+2, 
                                               key_weight_amax_list, in_amax_ptr+2, 
                                               Qbias_amax_ptr+3, Kbias_amax_ptr+3,
                                               use_ORDER_COL32_2R_4R4_, stream);
          add_V_bias_transform_kernelLauncher((int8_t*)v_buf_, 
                                              (const int32_t *)V, bias_V, 
                                              batch_size, seq_len, head_num, size_per_head, 
                                              value_weight_amax_list, in_amax_ptr+2, Vbias_amax_ptr+3, 
                                              use_ORDER_COL32_2R_4R4_, stream);
        }
        else if (int8_mode_ == 2 || int8_mode_ == 3)
        {
          add_QK_bias_transform_kernelLauncher((int8_t*)q_buf_, (int8_t*)k_buf_, 
                                               (const int8_t*) Q, bias_Q, (const int8_t*)K, bias_K, 
                                               batch_size, seq_len, head_num, size_per_head, 
                                               Q_aftergemm_amax_ptr+1, K_aftergemm_amax_ptr+1, 
                                               Qbias_amax_ptr+3, Kbias_amax_ptr+3,
                                               use_ORDER_COL32_2R_4R4_, stream);
          add_V_bias_transform_kernelLauncher((int8_t*)v_buf_, (const int8_t *)V, bias_V, 
                                               batch_size, seq_len, head_num, size_per_head,
                                               V_aftergemm_amax_ptr+1, Vbias_amax_ptr+3, 
                                               use_ORDER_COL32_2R_4R4_, stream);
        }
      }
      else{
        mappingRemovePaddingData_kernelLauncher(batch_size, seq_len, 
                                                param_.valid_word_num, sequence_id_map_, 
                                                param_.sequence_id_offset, stream);
        // if we use remove padding, then initialize the q_buf_, k_buf_ and v_buf_ to prevent bugs. v_buf_ will be properly initiaized in add_V_bias_transform_rebuild_padding_kernelLauncher()
        cudaMemsetAsync((int8_t*)q_buf_, 0, 2 * batch_size_ * seq_len_padded * head_num * size_per_head * sizeof(int8_t), param_.stream);
        if (int8_mode_ == 1)
        {
          
          add_QK_bias_transform_rebuild_padding_kernelLauncher((int8_t*)q_buf_, (int8_t*)k_buf_, 
                                                               (const int32_t*)Q, bias_Q, 
                                                               (const int32_t*)K, bias_K, 
                                                               param_.sequence_id_offset, param_.valid_word_num, 
                                                               batch_size, seq_len, 
                                                               head_num, size_per_head, 
                                                               query_weight_amax_list, in_amax_ptr+2, 
                                                               key_weight_amax_list, in_amax_ptr+2, 
                                                               Qbias_amax_ptr+3, Kbias_amax_ptr+3,
                                                               use_ORDER_COL32_2R_4R4_, stream);

          add_V_bias_transform_rebuild_padding_kernelLauncher((int8_t*)v_buf_, (const int32_t *)V, bias_V,
                                                              sequence_id_map_, param_.valid_word_num, 
                                                              batch_size, seq_len, head_num, size_per_head, 
                                                              value_weight_amax_list, in_amax_ptr+2, Vbias_amax_ptr+3, 
                                                              use_ORDER_COL32_2R_4R4_, stream);      
        }
        else if (int8_mode_ == 2 || int8_mode_ == 3)
        {
          add_QK_bias_transform_rebuild_padding_kernelLauncher((int8_t*)q_buf_, (int8_t*)k_buf_, 
                                                               (const int8_t*)Q, bias_Q, 
                                                               (const int8_t*)K, bias_K,
                                                               param_.sequence_id_offset, param_.valid_word_num,
                                                               batch_size, seq_len, head_num, size_per_head, 
                                                               Q_aftergemm_amax_ptr+1, K_aftergemm_amax_ptr+1,
                                                               Qbias_amax_ptr+3, Kbias_amax_ptr+3,
                                                               use_ORDER_COL32_2R_4R4_, stream);

          add_V_bias_transform_rebuild_padding_kernelLauncher((int8_t*)v_buf_, (const int8_t *)V, bias_V, 
                                                              sequence_id_map_, param_.valid_word_num, 
                                                              batch_size, seq_len, head_num, size_per_head,
                                                              V_aftergemm_amax_ptr+1, Vbias_amax_ptr+3, 
                                                              use_ORDER_COL32_2R_4R4_, stream);
        }
      }
  
      int batchCount = batch_size * head_num;
    
      if (int8_mode_ == 1)
      {     
        cublasLtMM_withAlgo(qk_int_buf_, batchCount, seq_len, seq_len, size_per_head, 
                            size_per_head*seq_len, size_per_head*seq_len, seq_len*seq_len, 
                            (int8_t*)q_buf_, (int8_t*)k_buf_, cublaslt_handle, stream, cublasAlgoMap_, use_ORDER_COL32_2R_4R4_);

        softmax_COL32_kernelLauncher((int8_t*)qk_buf_, qk_int_buf_, attr_mask, 
                                     batch_size, head_num, seq_len, 
                                     float(scalar), Qbias_amax_ptr + 1, Kbias_amax_ptr + 1, 
                                     Softmax_amax_ptr, stream);
      
        cublasLtMM_withAlgo(transpose_dst_int_buf_, batchCount, seq_len, size_per_head, seq_len, 
                            seq_len*seq_len, size_per_head*seq_len, size_per_head*seq_len, (int8_t*)qk_buf_, 
                            (int8_t*)v_buf_, cublaslt_handle, stream, cublasAlgoMap_, use_ORDER_COL32_2R_4R4_);

        if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
        {
          transpose_COL32_kernelLauncher((int8_t*)dst, (const int*)transpose_dst_int_buf_, batch_size, seq_len, head_num, 
                                         size_per_head, Vbias_amax_ptr+1, Softmax_amax_ptr+1, bmm2_amax_ptr+3, stream);
        }
        else
        {
          transpose_COL32_rebuild_padding_kernelLauncher((int8_t*)dst, (const int*)transpose_dst_int_buf_, sequence_id_map_, 
                                                         param_.valid_word_num, batch_size, seq_len, head_num, size_per_head, 
                                                         Vbias_amax_ptr+1, Softmax_amax_ptr+1, bmm2_amax_ptr+3, stream);     
        }    
      }
      else if (int8_mode_ == 2 || int8_mode_ == 3)
      {
        cublasLtMM_withAlgo_int8IO((int8_t*)qk_int_buf_, batchCount, seq_len, seq_len_padded, size_per_head, 
                                    size_per_head*seq_len, size_per_head*seq_len_padded, seq_len*seq_len_padded, 
                                    param_.int8O_gemm_deQ_scale_list[3],
                                    (int8_t*)q_buf_, (int8_t*)k_buf_, cublaslt_handle, stream, cublasAlgoMap_, use_ORDER_COL32_2R_4R4_);
       
        softmax_COL32_kernelLauncher((int8_t*)qk_buf_, (int8_t*)qk_int_buf_, attr_mask, 
                                     batch_size, head_num, seq_len, 
                                     float(scalar), bmm1_amax_ptr + 1, Softmax_amax_ptr, 
                                     stream); 
      
        cublasLtMM_withAlgo_int8IO((int8_t*)transpose_dst_int_buf_, batchCount, seq_len, size_per_head, seq_len_padded, 
                                    seq_len*seq_len_padded, size_per_head*seq_len_padded, size_per_head*seq_len, param_.int8O_gemm_deQ_scale_list[4], (int8_t*)qk_buf_, 
                                    (int8_t*)v_buf_, cublaslt_handle, stream, cublasAlgoMap_, use_ORDER_COL32_2R_4R4_);
        if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
        {
          transpose_COL32_kernelLauncher((int8_t*)dst, (const int8_t*)transpose_dst_int_buf_, batch_size, seq_len, head_num, 
                                          size_per_head, bmm2_amax_ptr+1, bmm2_amax_ptr+3, stream);
        }
        else
        {
          transpose_COL32_rebuild_padding_kernelLauncher((int8_t*)dst, (const int8_t*)transpose_dst_int_buf_, sequence_id_map_, 
                                                          param_.valid_word_num, batch_size, seq_len, head_num, size_per_head, 
                                                          bmm2_amax_ptr+1, bmm2_amax_ptr+3, stream);
        }
      }
    }
    //FP32/FP16
    else
    {
      if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
      {
        add_QKV_bias_transpose_kernelLauncher(q_buf_, k_buf_, v_buf_,
          Q, bias_Q,
          K, bias_K,
          V, bias_V,
          batch_size_, seq_len,
          head_num,
          size_per_head, stream);
      }
      else
      {
        // if we use remove padding, then initialize the q_buf_, k_buf_ and v_buf_ to prevent bugs.
        cudaMemsetAsync(q_buf_, 0, 3 * batch_size_ * seq_len * head_num * size_per_head * sizeof(DataType_), param_.stream);

        add_QKV_bias_rebuild_padding_kernelLauncher(Q, bias_Q, K, bias_K, V, bias_V, q_buf_, k_buf_, v_buf_, 
          batch_size, seq_len, head_num, size_per_head, param_.valid_word_num, param_.sequence_id_offset, stream);
      }

      DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;
    
      check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        seq_len, seq_len, size_per_head,
        &alpha,
        k_buf_, AType_, size_per_head, seq_len * size_per_head,
        q_buf_, BType_, size_per_head, seq_len * size_per_head,
        &beta,
        qk_buf_, CType_, seq_len, seq_len * seq_len,
        batch_size * head_num,
        computeType_,
        static_cast<cublasGemmAlgo_t>(cublasBmmAlgo_[0])));
    
      attn_softmax_kernelLauncher(qk_buf_, attr_mask, batch_size, seq_len, head_num, scalar, stream);

      check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_per_head, seq_len, seq_len,
        &alpha,
        v_buf_, AType_, size_per_head, seq_len * size_per_head,
        qk_buf_, BType_, seq_len, seq_len * seq_len,
        &beta,
        transpose_dst_, CType_, size_per_head, seq_len * size_per_head,
        batch_size * head_num,
        computeType_,
        static_cast<cublasGemmAlgo_t>(cublasBmmAlgo_[1])));
      
      if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
      {
        transpose_kernelLauncher(transpose_dst_, dst, batch_size, seq_len, head_num, size_per_head, stream);
      }
      else
      {
        transpose_rebuild_padding_kernelLauncher(transpose_dst_, dst, param_.valid_word_num,
                                                 batch_size, seq_len, head_num, size_per_head, 
                                                 param_.sequence_id_offset, stream);
      }
    }  
  }

  void forward()
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    try
    { 
      forward(param_.from_tensor, param_.to_tensor);
    }
    catch(std::runtime_error& error)
    {
      throw error;
    }
  }

  void forward(const DataType_* from_tensor, const DataType_* to_tensor)
  {
    if(param_.sequence_id_offset != nullptr && param_.valid_word_num != batch_size_ * from_seq_len_)
    {
      is_fuse_QKV_ = false;
    }

    if(is_fuse_QKV_ == true && int8_mode_ == 0)
    {
      // For tensorrt, we cannot get the pointer of from tensor until enqueue
      const DataType_* hA[] {param_.self_attention.query_weight.kernel,
                             param_.self_attention.key_weight.kernel,
                             param_.self_attention.value_weight.kernel,
                             from_tensor, to_tensor, to_tensor,
                             query_buf_, key_buf_, value_buf_};
      cudaMemcpyAsync((void*)qkv_kernel_, hA, sizeof(DataType_*) * 9, cudaMemcpyHostToDevice, param_.stream);
    }

#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    const int m = param_.sequence_id_offset == nullptr ? batch_size_ * from_seq_len_ : param_.valid_word_num;
    const int k = head_num_ * size_per_head_;
    const int n = k;
    const DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

    try
    { 
      if (int8_mode_ != 0){
        //K_int_buf_ V_int_buf_ should point to correct buffer according to param_.valid_word_num
        if (int8_mode_ == 1) {
          K_int_buf_ = (int*)Q_int_buf_ + param_.valid_word_num * head_num_ * size_per_head_;
          V_int_buf_ = (int*)K_int_buf_ + param_.valid_word_num * head_num_ * size_per_head_;
        } else if (int8_mode_ == 2 || int8_mode_ == 3){
          K_int_buf_ = (int*)((int8_t*)Q_int_buf_ + param_.valid_word_num * head_num_ * size_per_head_);
          V_int_buf_ = (int*)((int8_t*)K_int_buf_ + param_.valid_word_num * head_num_ * size_per_head_);
        }

        int fusedINT8QKV = 0;
        const int8_t* Q_weight = (const int8_t*)(param_.self_attention.query_weight.kernel);
        const int8_t* K_weight = (const int8_t*)(param_.self_attention.key_weight.kernel);
        const int8_t* V_weight = (const int8_t*)(param_.self_attention.value_weight.kernel);
        //for QKV weight are DataType_ & continue
        if ((param_.self_attention.query_weight.kernel + n*k == param_.self_attention.key_weight.kernel) &&
            (param_.self_attention.key_weight.kernel + n*k == param_.self_attention.value_weight.kernel))
          fusedINT8QKV = 1;
          //for QVK weight are int8 & continue
        else if ((Q_weight + n*k == K_weight) && (K_weight + n*k == V_weight))
          fusedINT8QKV = 2;
        
        if (int8_mode_ == 1)
        {
          if (fusedINT8QKV == 0){
            cublasLtMM_withAlgo(Q_int_buf_, 1, m, n, k, 0, 0, 0, 
                                param_.int8_from_tensor, Q_weight, 
                                param_.cublaslt_handle, param_.stream, 
                                cublasAlgoMap_, use_ORDER_COL32_2R_4R4_);
            cublasLtMM_withAlgo(K_int_buf_, 1, m, n, k, 0, 0, 0, 
                                param_.int8_from_tensor, K_weight, 
                                param_.cublaslt_handle, param_.stream, 
                                cublasAlgoMap_, use_ORDER_COL32_2R_4R4_);
            cublasLtMM_withAlgo(V_int_buf_, 1, m, n, k, 0, 0, 0, 
                                param_.int8_from_tensor, V_weight, 
                                param_.cublaslt_handle, param_.stream, 
                                cublasAlgoMap_, use_ORDER_COL32_2R_4R4_);
          }
          else{
            int strideFactor = (fusedINT8QKV == 1) ? (sizeof(DataType_)/sizeof(int8_t)) : 1; 
            cublasLtMM_withAlgo(Q_int_buf_, 3, m, n, k, 0, n*k*strideFactor, 
                                n*m, param_.int8_from_tensor, Q_weight, 
                                param_.cublaslt_handle, param_.stream, cublasAlgoMap_, use_ORDER_COL32_2R_4R4_);
          }
        }
        else if (int8_mode_ == 2 || int8_mode_ == 3)
        {
          if (fusedINT8QKV == 0){
            cublasLtMM_withAlgo_int8IO((int8_t*)Q_int_buf_, 1, m, n, k, 0, 0, 0, 
                                       param_.int8O_gemm_deQ_scale_list[0],  
                                       param_.int8_from_tensor, Q_weight, 
                                       param_.cublaslt_handle, param_.stream, 
                                       cublasAlgoMap_, use_ORDER_COL32_2R_4R4_);
            cublasLtMM_withAlgo_int8IO((int8_t*)K_int_buf_, 1, m, n, k, 0, 0, 0, 
                                       param_.int8O_gemm_deQ_scale_list[1],
                                       param_.int8_from_tensor, K_weight, 
                                       param_.cublaslt_handle, param_.stream, 
                                       cublasAlgoMap_, use_ORDER_COL32_2R_4R4_);
            cublasLtMM_withAlgo_int8IO((int8_t*)V_int_buf_, 1, m, n, k, 0, 0, 0,
                                       param_.int8O_gemm_deQ_scale_list[2],            
                                       param_.int8_from_tensor, V_weight, 
                                       param_.cublaslt_handle, param_.stream, 
                                       cublasAlgoMap_, use_ORDER_COL32_2R_4R4_);
          }
          else{
            int strideFactor = (fusedINT8QKV == 1) ? (sizeof(DataType_)/sizeof(int8_t)) : 1; 
            cublasLtMM_withAlgo_int8IO((int8_t*)Q_int_buf_, 3, m, n, k, 0, n*k*strideFactor, n*m, 
                                       param_.int8O_gemm_deQ_scale_list[0],
                                       param_.int8_from_tensor, Q_weight, 
                                       param_.cublaslt_handle, param_.stream, cublasAlgoMap_, use_ORDER_COL32_2R_4R4_);
          }  
        }

        int S;
        if(dispatcher_int8.get())
          S = dispatcher_int8->getSFromMaxSeqLen(from_seq_len_);
        if(dispatcher_int8.get() && dispatcher_int8->isValid(S) && param_.trt_seqlen_offset != nullptr)
        {
          // This function is only used when we satisfy the following conditions:
          // 1. INT8
          // 2. GPU SM >= 75
          int8_fused_multiHeadAttr_kernelLauncher((const void*)Q_int_buf_,
                                                  param_.amaxList + 4 + 1, 
                                                  param_.amaxList + 12 + 1, 
                                                  param_.amaxList + 20 + 1,
                                                  param_.trt_fused_mha_amax_list[0]/127.0f,
                                                  S
                                                 );
        }
        else
        {

          DataType_ scalar = 1 / (sqrtf(size_per_head_ * 1.0f) * q_scaling_);
          multiHeadAttr_nofuse_kernelLauncher(
                param_.stream,
                param_.cublas_handle,
                param_.cublaslt_handle,
                (DataType_*)Q_int_buf_,
                param_.self_attention.query_weight.bias,
                (DataType_*)(K_int_buf_),
                param_.self_attention.key_weight.bias,
                (DataType_*)(V_int_buf_),
                param_.self_attention.value_weight.bias,
                param_.attr_mask,
                param_.attr_out,
                batch_size_,
                from_seq_len_,
                head_num_,
                size_per_head_,
                int8_mode_,
                scalar);
        }
      }
      else{      
        if(is_fuse_QKV_ == true)
        {
          int algoId = getAlgoIdFromMap(cublasAlgoMap_, 3, n, m, k, AType_ == CUDA_R_16F ? HALF_DATATYPE : FLOAT_DATATYPE);
          check_cuda_error(cublasGemmBatchedEx(param_.cublas_handle, 
                           CUBLAS_OP_N, CUBLAS_OP_N, 
                           n, m, k, 
                           &alpha, 
                           (const void* const*) qkv_kernel_, AType_, n,
                           (const void* const*) qkv_input_, BType_, k,
                           &beta, 
                           (void* const*)qkv_buf_, CType_, n,
                           3,
                           computeType_, 
                           static_cast<cublasGemmAlgo_t>(algoId)));
        }
        else
        {
          cublasMM_cublasLtMM_wrapper(param_.cublaslt_handle, param_.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                           n, m, k, &alpha,
                           param_.self_attention.query_weight.kernel, AType_, n,
                           from_tensor, BType_, k,
                           &beta, (DataType_ *)query_buf_, CType_, n,
                           param_.stream, cublasAlgoMap_, sm_, cublas_workspace_);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          cublasMM_cublasLtMM_wrapper(param_.cublaslt_handle, param_.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                           n, m, k, &alpha,
                           param_.self_attention.key_weight.kernel, AType_, n,
                           to_tensor, BType_, k,
                           &beta, (DataType_ *)key_buf_, CType_, n,
                           param_.stream, cublasAlgoMap_, sm_, cublas_workspace_); 

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif

          cublasMM_cublasLtMM_wrapper(param_.cublaslt_handle, param_.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                           n, m, k, &alpha,
                           param_.self_attention.value_weight.kernel, AType_, n,
                           to_tensor, BType_, k,
                           &beta, (DataType_ *)value_buf_, CType_, n,
                           param_.stream, cublasAlgoMap_, sm_, cublas_workspace_); 
        }
     
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        int S;
        if(dispatcher_fp16.get())
          S = dispatcher_fp16->getSFromMaxSeqLen(from_seq_len_);
        if(dispatcher_fp16.get() && OpType_==OperationType::FP16 && dispatcher_fp16->isValid(S) && param_.trt_seqlen_offset != nullptr)
        {
          // This function is only used when we satisfy the following conditions:
          // 1. FP16
          // 2. GPU SM >= 72
          //  3. Temporally add seqlen <= 384 limitation because the current fused mha cannot handle seqlen > 384.
          fused_multiHeadAttr_kernelLauncher(S);
        }
        else
        {
          DataType_ scalar = 1 / (sqrtf(size_per_head_ * 1.0f) * q_scaling_);

          multiHeadAttr_nofuse_kernelLauncher(
            param_.stream,
            param_.cublas_handle,
            param_.cublaslt_handle,
            query_buf_,
            param_.self_attention.query_weight.bias,
            key_buf_,
            param_.self_attention.key_weight.bias,
            value_buf_,
            param_.self_attention.value_weight.bias,
            param_.attr_mask,
            param_.attr_out,
            batch_size_,
            from_seq_len_,
            head_num_,
            size_per_head_,
            int8_mode_,
            scalar);
        }
      }
    }
    catch(std::runtime_error& error)
    {
      throw error;
    }
  }

  void fused_multiHeadAttr_kernelLauncher(const int S);
  
  void int8_fused_multiHeadAttr_kernelLauncher(const void* Q, 
                                               const float *q_deQFactor_ptr, const float *k_deQFactor_ptr, const float *v_deQFactor_ptr, 
                                               const float mScaleQkv, const int S);

  void trt_add_QKV_bias_kernelLauncher(
      const DataType_* bias_Q,
      const DataType_* bias_K,
      const DataType_* bias_V);
      
  void trt_add_QKV_bias_COL32_int8IO_kernelLauncher(
      int8_t* output,
      const int8_t* Q,
      const DataType_* bias_Q,
      const DataType_* bias_K,
      const DataType_* bias_V,
      const float *q_input_deQFactor_ptr, 
      const float *k_input_deQFactor_ptr, 
      const float *v_input_deQFactor_ptr, 
      const float qkv_output_scale); 

  void trt_add_QKV_bias_COL32_int32Iint8O_kernelLauncher(
      int8_t* output,
      const int32_t* Q,
      const DataType_* bias_Q,
      const DataType_* bias_K,
      const DataType_* bias_V,
      const float *input_deQFactor_div127_ptr,
      const float * q_weight_amax,
      const float * k_weight_amax,
      const float * v_weight_amax,
      const float qkv_output_scale);

  void initialize(MultiHeadInitParam<DataType_> param)
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    param_ = param;
    if (int8_mode_ != 0){
      int hidden_dim = head_num_ * size_per_head_;
      query_weight_amax_list = param_.amaxList + ACTIVATION_AMAX_NUM;
      key_weight_amax_list = query_weight_amax_list + hidden_dim;
      value_weight_amax_list = key_weight_amax_list + hidden_dim;
      if (dispatcher_int8.get())
      {
        dispatcher_int8.get()->setScaleList(param_.trt_fused_mha_amax_list[0]/127.0f, param_.trt_fused_mha_amax_list[1]/127.0f, param_.trt_fused_mha_amax_list[2]/127.0f);
      }
    } 
  }

  ~OpenMultiHeadAttention() override
  {
    if (buf_ != NULL)
    {
      if (allocator_ == NULL)
      {
        printf("[ERROR][OpenMultiHeadAttention][~OpenMultiHeadAttention] allocator_ is NULL!\n");
        exit(-1);
      }
      allocator_->free(buf_);
      buf_ = NULL;
    }
  }
};
                                       
}//namespace cuda
}//namespace fastertransformer
