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
 * Decoder transformer
 **/

#pragma once

#include "contrib_ops/cuda/fastertransformer/utils/common.h"
#include "contrib_ops/cuda/fastertransformer/utils/functions.h"
#include "contrib_ops/cuda/fastertransformer/utils/allocator.h"
#include "contrib_ops/cuda/fastertransformer/utils/arguments.h"
#include "contrib_ops/cuda/fastertransformer/cuda/cuda_kernels.h"
#include "contrib_ops/cuda/fastertransformer/open_decoder.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include "contrib_ops/cuda/fastertransformer/utils/nvtx_utils.h"

namespace fastertransformer
{

template <OperationType OpType_>
class DecodingGpt
{
private:
    typedef DecoderTransformerTraits<OpType_> Traits_;
    typedef typename Traits_::DataType DataType_;
    const IAllocator &allocator_;
    struct GptArguments args_;
    TensorParallelParam t_parallel_param_;
    LayerParallelParam l_parallel_param_;

    const cudaDataType_t computeType_ = Traits_::computeType;
    const cudaDataType_t AType_ = Traits_::AType;
    const cudaDataType_t BType_ = Traits_::BType;
    const cudaDataType_t CType_ = Traits_::CType;
    std::map<std::string, cublasLtMatmulAlgo_info> cublasAlgoMap_;

    DataType_ *embedding_kernel_padded_;

    OpenDecoder<OpType_> *decoder_;
    DataType_ **K_cache_;
    DataType_ **V_cache_;
    DataType_ *from_tensor_[2];
    DataType_ *decoder_buf_;
    DataType_ *decoder_normed_result_buf_;
    DataType_ *logits_buf_;
    void *buf_;
    
    void *topk_workspace_ = nullptr;
    size_t topk_workspace_size_ = 0;
    void *topp_workspace_ = nullptr;
    size_t topp_workspace_size_ = 0;
    void *topk_topp_workspace_ = nullptr;
    size_t topk_topp_workspace_size_ = 0;
    void *cublas_workspace_ = nullptr;
    int *topp_id_vals_buf_;
    int *topp_offset_buf_;
    curandState_t *curandstate_buf_;
    int *begin_topp_offset_buf_;

    size_t nccl_buf_size_;
    DataType_ *nccl_logits_buf_;

    bool *finished_buf_;
    bool *h_finished_buf_;
    
public:
    DecodingGpt(const IAllocator &allocator, const int batch_size,
                 const int seq_len,
                 const int head_num, const int size_per_head,
                 const int vocab_size, const int decoder_layers,
                 const int start_id, const int end_id,
                 const int candidate_num = 1,
                 const float probability_threshold = 0.0,
                 const float temperature = 1.0,
                 const int tensor_para_size = 1,
                 const int layer_para_size = 1,
                 const bool is_fuse_QKV = true,
                 const float repetition_penalty = 1.0) : allocator_(allocator)
    {
#ifndef NDEBUG
        PRINT_FUNC_NAME_();
#endif
        assert(temperature != 0.0);
        assert(repetition_penalty > 0.0);
        assert(candidate_num > 0 || probability_threshold > 0.0);
        assert(decoder_layers % layer_para_size == 0);

        args_.batch_size_ = batch_size;
        args_.seq_len_ = seq_len;
        args_.head_num_ = head_num;
        args_.size_per_head_ = size_per_head;
        args_.hidden_units_ = head_num * size_per_head;
        args_.decoder_layers_ = decoder_layers;
        args_.vocab_size_ = vocab_size;
        args_.start_id_ = start_id;
        args_.end_id_ = end_id;
        args_.candidate_num_ = candidate_num;
        args_.probability_threshold_ = probability_threshold;
        args_.temperature_ = temperature;
        args_.repetition_penalty_ = repetition_penalty;
        
        K_cache_ = new DataType_ *[1];
        V_cache_ = new DataType_ *[1];

        decoder_ = new OpenDecoder<OpType_>(args_.head_num_, size_per_head, 0 /* memory_hidden_units */, is_fuse_QKV);
        decoder_->set_max_batch_size(args_.batch_size_);

        args_.vocab_size_padded_ = div_up(args_.vocab_size_, 64) * 64;

        size_t from_tensor_size = args_.batch_size_ * args_.hidden_units_;                    // type T
        size_t decoder_workspace_size = (size_t)decoder_->getWorkspaceSize();                                             // type T
        size_t decoder_normed_result_buffer_size = args_.batch_size_ * args_.hidden_units_;   // type T
        // cache costs lots of memory, so we only store part of them when we use multi-gpu for inference
        size_t cache_size = args_.batch_size_ * args_.seq_len_ * args_.hidden_units_ / tensor_para_size;         // type T
        size_t logits_buf_size = args_.batch_size_ * args_.vocab_size_padded_; // type T

        size_t topp_id_vals_buf_size = args_.batch_size_ * args_.vocab_size_padded_; // type int
        size_t topp_offset_buf_size = args_.batch_size_ + 1;
        size_t begin_topp_offset_buf_size = topp_offset_buf_size;
        size_t curandState_size = args_.batch_size_;
        size_t finished_buf_size = args_.batch_size_;

        const int MEM_C = 128;
        size_t embedding_kernel_transposed_padded_size = args_.hidden_units_ * args_.vocab_size_padded_;
        embedding_kernel_transposed_padded_size = div_up(embedding_kernel_transposed_padded_size, MEM_C) * MEM_C;

        // prevent memory misalinged address
        logits_buf_size = (size_t)(ceil(logits_buf_size / 4.)) * 4;
        
        topp_id_vals_buf_size = (size_t)(ceil(topp_id_vals_buf_size / 4.)) * 4;
        topp_offset_buf_size = (size_t)(ceil(topp_offset_buf_size / 4.)) * 4;
        begin_topp_offset_buf_size = topp_offset_buf_size;
        curandState_size = (size_t)(ceil(curandState_size / 32.)) * 32;
        finished_buf_size = (size_t)(ceil(finished_buf_size / 32.)) * 32;

        topP_sampling_kernel_kernelLauncher_v2(topp_workspace_,
                                               topp_workspace_size_,
                                               logits_buf_,
                                               topp_id_vals_buf_,
                                               topp_offset_buf_,
                                               begin_topp_offset_buf_,
                                               nullptr,
                                               curandstate_buf_,
                                               args_,
                                               nullptr,
                                               nullptr,
                                               args_.vocab_size_padded_,
                                               0,
                                               args_.batch_size_);

        topK_sampling_kernel_kernelLauncher_v2(topk_workspace_,
                                               topk_workspace_size_,
                                               logits_buf_,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               curandstate_buf_,
                                               args_,
                                               0,
                                               args_.batch_size_);

        topK_topP_sampling_kernel_kernelLauncher_v2(topk_topp_workspace_,
                                              topk_topp_workspace_size_,
                                              nullptr,
                                              logits_buf_,
                                              nullptr,
                                              curandstate_buf_,
                                              args_,
                                              0,
                                              args_.batch_size_);

        size_t datatype_buf_size = from_tensor_size * 2 + decoder_workspace_size +
                                cache_size * 2 * (args_.decoder_layers_ / layer_para_size) + decoder_normed_result_buffer_size;

        nccl_buf_size_ = args_.batch_size_ * args_.vocab_size_padded_;
        nccl_buf_size_ = (size_t)(ceil(nccl_buf_size_ / 4.)) * 4;

        buf_ = reinterpret_cast<void *>(allocator_.malloc(
            ((sizeof(DataType_) == sizeof(half)) ? CUBLAS_WORKSPACE_SIZE : 0) + 
            sizeof(DataType_) * embedding_kernel_transposed_padded_size +
            sizeof(DataType_) * (datatype_buf_size + logits_buf_size) + 
            sizeof(int) * (topp_id_vals_buf_size + topp_offset_buf_size + begin_topp_offset_buf_size) +
            topp_workspace_size_ + topk_workspace_size_ + topk_topp_workspace_size_ + sizeof(DataType_) * nccl_buf_size_ +
            finished_buf_size + curandState_size * sizeof(curandState_t)));

        if (sizeof(DataType_) == sizeof(half))
        {
          cublas_workspace_ = buf_;
          embedding_kernel_padded_ = (DataType_ *)((char*)cublas_workspace_ + CUBLAS_WORKSPACE_SIZE);
        }
        else
        {
          cublas_workspace_ = nullptr;
          embedding_kernel_padded_ = (DataType_ *)buf_;
        }
        from_tensor_[0] = (DataType_ *)(embedding_kernel_padded_ + embedding_kernel_transposed_padded_size);
        from_tensor_[1] = (DataType_ *)(from_tensor_[0] + from_tensor_size);

        K_cache_[0] = from_tensor_[1] + from_tensor_size + 0 * cache_size * args_.decoder_layers_ / layer_para_size;
        V_cache_[0] = from_tensor_[1] + from_tensor_size + 1 * cache_size * args_.decoder_layers_ / layer_para_size;

        decoder_buf_ = V_cache_[0] + cache_size * args_.decoder_layers_ / layer_para_size;
        decoder_normed_result_buf_ = (decoder_buf_ + decoder_workspace_size);
        logits_buf_ = decoder_normed_result_buf_ + decoder_normed_result_buffer_size;
        topp_id_vals_buf_ = (int *)((DataType_*)logits_buf_ + logits_buf_size);
        begin_topp_offset_buf_ = (int *)(topp_id_vals_buf_ + topp_id_vals_buf_size);
        topp_offset_buf_ = (int *)((int*)begin_topp_offset_buf_ + begin_topp_offset_buf_size);
        topp_workspace_ = (void *)((int*)topp_offset_buf_ + topp_offset_buf_size);
        topk_workspace_ = (void *)((char*)topp_workspace_ + topp_workspace_size_);
        topk_topp_workspace_ = (void *)((char*)topk_workspace_ + topk_workspace_size_);
        nccl_logits_buf_ = (DataType_ *)((char*)topk_topp_workspace_ + topk_topp_workspace_size_);
        curandstate_buf_ = (curandState_t*)(nccl_logits_buf_ + nccl_buf_size_);
        finished_buf_ = (bool*)(curandstate_buf_ + curandState_size);
        h_finished_buf_ = new bool[args_.batch_size_];

        cudaMemset(embedding_kernel_padded_, 0, embedding_kernel_transposed_padded_size * sizeof(DataType_));

        int isConfigExist = access("decoding_gemm_config.in", 0);
        if (isConfigExist == -1)
            printf("[WARNING] decoding_gemm_config.in is not found\n");
        else
        {
            readAlgoFromConfig(cublasAlgoMap_, 1);
            // check that the gemm_config setting is runnable
            for (auto iter = cublasAlgoMap_.begin() ; iter != cublasAlgoMap_.end() ; iter++)
            {
                int algoId = iter->second.algoId;
                int stages = iter->second.stages;
                //only check for cublas
                if (stages != -1)
                    continue;
                if (Traits_::OpType == OperationType::FP32)
                {
                    if (algoId > CUBLAS_GEMM_ALGO23 || algoId < CUBLAS_GEMM_DEFAULT)
                    {
                        // the algorithm is not for FP32
                        printf("[ERROR] cuBLAS Algorithm %d is not used in FP32. \n", algoId);
                        exit(-1);
                    }
                }
                else
                {
                    if (algoId > CUBLAS_GEMM_ALGO15_TENSOR_OP || algoId < CUBLAS_GEMM_DEFAULT_TENSOR_OP)
                    {
                        // the algorithm is not for FP16
                        printf("[ERROR] cuBLAS Algorithm %d is not used in FP16. \n", algoId);
                        exit(-1);
                    }
                }
            }
        }
    }

    void set_tensor_parallel_param(const TensorParallelParam param)
    {
        t_parallel_param_ = param;
        decoder_->set_tensor_parallel_param(param);
    }

    void set_layer_parallel_param(const LayerParallelParam param)
    {
        l_parallel_param_ = param;
        decoder_->set_layer_parallel_param(param);
    }

    void forward_context(const DecoderInitParam<DataType_> *decoder_param,
                         const DecodingInitParam<DataType_> decoding_params)
    {
#ifndef NDEBUG
        PRINT_FUNC_NAME_();
#endif
        const int input_len = decoding_params.request_input_len;
        const int max_len = (decoding_params.request_output_len > 0 && input_len + decoding_params.request_output_len <= args_.seq_len_) ?
                            input_len + decoding_params.request_output_len :
                            args_.seq_len_;
        const int request_batch_size = decoding_params.request_batch_size;
        cudaMemsetAsync(decoding_params.output_ids, 0, sizeof(int) * request_batch_size * max_len, decoding_params.stream);
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        // const int input_len = decoding_params.request_input_len;
        const int max_input_len = decoding_params.max_input_len;

        // d_start_ids: [batch * seqlen]
        if(input_len == 1)
        {
            cudaMemcpyAsync(decoding_params.output_ids, decoding_params.d_start_ids, 
                            sizeof(int) * request_batch_size, cudaMemcpyDeviceToDevice, decoding_params.stream);
            return;
        }
        const int local_batch_size = ceil(request_batch_size * 1.0 / l_parallel_param_.world_size);
        const int m = local_batch_size * input_len;
        const int h_1 = args_.hidden_units_;

        DataType_* from_tensor[2];
        DataType_* decoder_output;
        DataType_* decoder_workspace;
        void *buf = reinterpret_cast<void *>(allocator_.malloc(
            decoder_->getContextWorkspaceSize(input_len, local_batch_size) + 
            (m * h_1 + 2 * request_batch_size * input_len * h_1) * sizeof(DataType_)
        ));
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        from_tensor[0] = (DataType_*) buf;
        from_tensor[1] = from_tensor[0] + request_batch_size * input_len * h_1;
        decoder_output = from_tensor[1] + request_batch_size * input_len * h_1;
        decoder_workspace = decoder_output + m * h_1;

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        if(l_parallel_param_.rank == 0)
        {
            PUSH_RANGE("Before Transformer/Embedding")
            start_id_embedding_position_lookups_kernel_launcher(from_tensor[0],
                                                                decoding_params.output_ids,
                                                                decoding_params.embedding_table,
                                                                decoding_params.position_encoding_table,
                                                                decoding_params.d_start_ids,
                                                                1,
                                                                input_len,
                                                                max_input_len,
                                                                request_batch_size,
                                                                args_.hidden_units_, 
                                                                decoding_params.stream);
            POP_RANGE
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        }

        int ite_num = (int)(ceil(request_batch_size * 1.0 / local_batch_size));
        for(int ite = 0; ite < ite_num; ite++)
        {
            int in_id, out_id;
            for (int layer = 0; layer < args_.decoder_layers_; ++layer)
            {
                if(l_parallel_param_.is_valid(layer))
                {
                    in_id = layer & 0x1;
                    out_id = 1 - in_id;

                    if(layer == l_parallel_param_.layers_per_group * l_parallel_param_.rank && layer != 0 && l_parallel_param_.world_size > 1)
                    {
                        const int size = m * t_parallel_param_.local_hidden_units_;
                        nccl_recv(from_tensor[in_id] + ite * m * h_1 + size * t_parallel_param_.rank, size, l_parallel_param_.rank - 1, 
                                    l_parallel_param_.nccl_comm, decoding_params.stream);
                        all2all_gather(from_tensor[in_id] + ite * m * h_1, from_tensor[in_id] + ite * m * h_1, size, 
                                    t_parallel_param_, decoding_params.stream);
                    }

                    decoder_->initialize(decoder_param[layer], decoder_buf_, cublas_workspace_, false);
#ifndef NDEBUG
                    cudaDeviceSynchronize();
                    check_cuda_error(cudaGetLastError());
#endif

                    int dummy_decoder_max_seq_len = args_.seq_len_;
                    // int dummy_decoder_max_seq_len = -1;
                    size_t cache_offset;
                    if(dummy_decoder_max_seq_len == -1)
                    {
                        cache_offset = (layer - l_parallel_param_.layers_per_group * l_parallel_param_.rank) *
                                        args_.batch_size_ * args_.seq_len_ * t_parallel_param_.local_hidden_units_;
                    }
                    else
                    {
                        cache_offset = (layer - l_parallel_param_.layers_per_group * l_parallel_param_.rank) *
                                        args_.batch_size_ * args_.seq_len_ * t_parallel_param_.local_hidden_units_ +
                                        ite * local_batch_size * args_.seq_len_ * t_parallel_param_.local_hidden_units_;
                    }
                    decoder_->forward_context(decoder_workspace,
                                              from_tensor[out_id] + ite * m * h_1,
                                              K_cache_[0] + cache_offset,
                                              V_cache_[0] + cache_offset,
                                              from_tensor[in_id] + ite * m * h_1,
                                              decoding_params.d_attn_mask + ite * local_batch_size * input_len * input_len,
                                              local_batch_size,
                                              input_len,
                                              ite,
                                              dummy_decoder_max_seq_len,
                                              layer == args_.decoder_layers_ - 1);
#ifndef NDEBUG
                    cudaDeviceSynchronize();
                    check_cuda_error(cudaGetLastError());
#endif
                    if(layer == l_parallel_param_.layers_per_group * (l_parallel_param_.rank + 1) - 1 && layer != args_.decoder_layers_ - 1 && l_parallel_param_.world_size > 1)
                    {
                        const int size = m * t_parallel_param_.local_hidden_units_;
                        nccl_send(from_tensor[out_id] + ite * m * h_1 + size * t_parallel_param_.rank, size, l_parallel_param_.rank + 1,
                                    l_parallel_param_.nccl_comm, decoding_params.stream);
                    }
                }
            } // end of for loop of layer
        } // end of for loop of ite
        allocator_.free(buf);
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
    }

    void forward(const DecoderInitParam<DataType_> *decoder_param,
                 DecodingInitParam<DataType_> decoding_params)
    {
#ifndef NDEBUG
        PRINT_FUNC_NAME_();
#endif
        const int input_len = decoding_params.request_input_len;
        const int max_input_len = decoding_params.max_input_len;
        const int request_batch_size = decoding_params.request_batch_size;
        const int max_len = (decoding_params.request_output_len > 0 && input_len + decoding_params.request_output_len <= args_.seq_len_) ?
                            input_len + decoding_params.request_output_len :
                            args_.seq_len_;

        assert(request_batch_size <= args_.batch_size_);
        assert(request_batch_size % l_parallel_param_.local_batch_size == 0);
        const int m = request_batch_size;
        const int k = args_.hidden_units_;
        const DataType_* embedding_kernel_ptr = nullptr;

        cudaMemsetAsync(finished_buf_, false, sizeof(finished_buf_[0]) * request_batch_size, decoding_params.stream);
        if (args_.probability_threshold_ != 0.0)
        {
            topp_initialization_kernelLauncher_v2(nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  topp_id_vals_buf_,
                                                  topp_offset_buf_,
                                                  begin_topp_offset_buf_,
                                                  args_.candidate_num_ > 0 ? args_.candidate_num_ : args_.vocab_size_padded_,
                                                  args_,
                                                  decoding_params.stream);

#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }
        ker_curand_setupLauncher(curandstate_buf_,
                                 args_,
                                 decoding_params.stream);

        if(std::is_same<DataType_, float>::value || (std::is_same<DataType_, half>::value && args_.vocab_size_padded_ == args_.vocab_size_))
        {
            embedding_kernel_ptr = (const DataType_ *)decoding_params.embedding_kernel;
        }
        else
        {
            cudaMemcpyAsync(embedding_kernel_padded_, decoding_params.embedding_kernel, 
                            sizeof(DataType_) * args_.vocab_size_ * args_.hidden_units_, cudaMemcpyDeviceToDevice, decoding_params.stream);
            embedding_kernel_ptr = (const DataType_ *)embedding_kernel_padded_;
        }
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        bool is_generation_done = false;
        const int local_batch = l_parallel_param_.local_batch_size;
        for (size_t step = input_len; step < max_len; ++step)
        {
            const int ite_num = request_batch_size / local_batch;
            for(size_t ite = 0; ite < ite_num; ite++)
            {
                if(l_parallel_param_.rank == 0 && l_parallel_param_.world_size > 1)
                {
                    if(step != (size_t)input_len)
                    {
                        PUSH_RANGE("token/recv")
                        nccl_recv(decoding_params.output_ids + (step - 1) * m + ite * local_batch, local_batch,
                                  l_parallel_param_.world_size - 1, l_parallel_param_.nccl_comm, decoding_params.stream);
                        POP_RANGE
                    }
                }

                if(l_parallel_param_.rank < l_parallel_param_.world_size - 1 && l_parallel_param_.world_size > 1)
                {
                    if(step != (size_t)input_len)
                    {
                        nccl_broadcast(finished_buf_ + ite * local_batch, local_batch, l_parallel_param_.world_size - 1, l_parallel_param_, decoding_params.stream);
                    }
                }
                if(ite == 0)
                {
                    cudaMemcpyAsync(h_finished_buf_, finished_buf_, sizeof(bool) * request_batch_size, cudaMemcpyDeviceToHost, decoding_params.stream);
                    cudaStreamSynchronize(decoding_params.stream);
                    uint sum = 0;
                    for (uint i = 0; i < request_batch_size; i++)
                    {
                        sum += (int)h_finished_buf_[i];
                    }
                    if (sum == request_batch_size)
                    {
                        is_generation_done = true;
                        break;
                    }
                }

                if(l_parallel_param_.rank == 0)
                {
                    PUSH_RANGE("Before Transformer/Embedding")
                    embedding_position_lookups_kernel_launcher(from_tensor_[0],
                                                            decoding_params.embedding_table,
                                                            decoding_params.position_encoding_table,
                                                            decoding_params.output_ids,
                                                            local_batch,
                                                            m,
                                                            args_.hidden_units_,
                                                            step,
                                                            ite,
                                                            max_input_len,
                                                            decoding_params.d_start_lengths,
                                                            decoding_params.stream);
                    POP_RANGE
#ifndef NDEBUG
                    cudaDeviceSynchronize();
                    check_cuda_error(cudaGetLastError());
#endif
                }

                //we use two-way buffer
                int from_id, out_id;
                for (int layer = 0; layer < args_.decoder_layers_; ++layer)
                {
                    if(l_parallel_param_.is_valid(layer))
                    {
                        /*
                            For the first layer (layer-0), from_id is 0. We also stored the embedding lookup 
                            result in from_tensor_[0]
                        */
                        from_id = layer & 0x1;
                        out_id = 1 - from_id;

                        if(layer == l_parallel_param_.layers_per_group * l_parallel_param_.rank && layer != 0 && l_parallel_param_.world_size > 1)
                        {
                            const int size = local_batch * t_parallel_param_.local_hidden_units_;
                            nccl_recv(from_tensor_[from_id] + size * t_parallel_param_.rank, size, l_parallel_param_.rank - 1, 
                                      l_parallel_param_.nccl_comm, decoding_params.stream);
                            all2all_gather(from_tensor_[from_id], from_tensor_[from_id], size, 
                                           t_parallel_param_, decoding_params.stream);
                        }

                        /*
                            We use one decoder_ object to process multiple decoder layers. 

                            At the beginning of each decoder layer, we initialize the decoder object 
                            with corresponding weights and decoder_buf_.

                            The decoder_buf_ is reused.
                        */
                        decoder_->initialize(decoder_param[layer], decoder_buf_, cublas_workspace_, false);
                        
#ifndef NDEBUG
                        cudaDeviceSynchronize();
                        check_cuda_error(cudaGetLastError());
#endif
                        int dummy_decoder_max_seq_len = args_.seq_len_;
                        // int dummy_decoder_max_seq_len = -1;
                        size_t cache_offset;
                        if(dummy_decoder_max_seq_len == -1)
                        {
                            cache_offset = (layer - l_parallel_param_.layers_per_group * l_parallel_param_.rank) *
                                            args_.batch_size_ * args_.seq_len_ * t_parallel_param_.local_hidden_units_ +
                                            ite * local_batch * t_parallel_param_.local_hidden_units_;
                        }
                        else
                        {
                            cache_offset = (layer - l_parallel_param_.layers_per_group * l_parallel_param_.rank) * 
                                            args_.batch_size_ * args_.seq_len_ * t_parallel_param_.local_hidden_units_ + 
                                            ite * local_batch * args_.seq_len_ * t_parallel_param_.local_hidden_units_;
                        }
                        decoder_->forward_v2(from_tensor_[from_id], 
                                            nullptr, // memory_tensor should be nullptr
                                            K_cache_[0] + cache_offset,
                                            V_cache_[0] + cache_offset,
                                            nullptr, nullptr, // key_mem_cache_ and value_mem_cache_ should be nullptr
                                            nullptr, // memory_sequence_length should be nullptr
                                            from_tensor_[out_id], step, dummy_decoder_max_seq_len,
                                            false, 
                                            finished_buf_ + ite * local_batch,
                                            max_input_len, 
                                            decoding_params.d_start_lengths + ite * local_batch);

#ifndef NDEBUG
                        cudaDeviceSynchronize();
                        check_cuda_error(cudaGetLastError());
#endif          

                        if(layer == l_parallel_param_.layers_per_group * (l_parallel_param_.rank + 1) - 1 && layer != args_.decoder_layers_ - 1 && l_parallel_param_.world_size > 1)
                        {
                            const size_t size = local_batch * t_parallel_param_.local_hidden_units_;
                            nccl_send(from_tensor_[out_id] + size * t_parallel_param_.rank, size, l_parallel_param_.rank + 1, 
                                      l_parallel_param_.nccl_comm, decoding_params.stream);
                        }
                    }
                }

                if(l_parallel_param_.rank == l_parallel_param_.world_size - 1)
                {

                    layer_norm(from_tensor_[out_id],
                               decoding_params.layernorm.gamma,
                               decoding_params.layernorm.beta,
                               decoder_normed_result_buf_,
                               local_batch,
                               k,
                               decoding_params.stream);

#ifndef NDEBUG
                    cudaDeviceSynchronize();
                    check_cuda_error(cudaGetLastError());
#endif

                    DataType_ alpha = DataType_(1.0f);
                    DataType_ beta = DataType_(0.0f);
                    assert(args_.vocab_size_padded_ % t_parallel_param_.world_size == 0);
                    int n = args_.vocab_size_padded_ / t_parallel_param_.world_size;
                    
                    if(t_parallel_param_.world_size == 1)
                    {
                        PUSH_RANGE("After Transformer/GEMM")
                        cublasMM_cublasLtMM_wrapper_decoder(decoding_params.cublaslt_handle, 
                                                            decoding_params.cublas_handle, 
                                                            CUBLAS_OP_T, CUBLAS_OP_N,
                                                            n, local_batch, k,
                                                            &alpha,
                                                            embedding_kernel_ptr, AType_, k,
                                                            decoder_normed_result_buf_, BType_, k,
                                                            &beta,
                                                            logits_buf_, CType_, n,
                                                            decoding_params.stream, cublasAlgoMap_,
                                                            cublas_workspace_);
                        POP_RANGE
                    }
                    else
                    {
                        PUSH_RANGE("After Transformer/GEMM")
                        cublasMM_cublasLtMM_wrapper_decoder(decoding_params.cublaslt_handle, 
                                                            decoding_params.cublas_handle, 
                                                            CUBLAS_OP_T, CUBLAS_OP_N,
                                                            n, local_batch, k,
                                                            &alpha,
                                                            embedding_kernel_ptr + t_parallel_param_.rank * n * k,
                                                            AType_, k,
                                                            decoder_normed_result_buf_, BType_, k,
                                                            &beta,
                                                            nccl_logits_buf_ + t_parallel_param_.rank * local_batch * n,
                                                            CType_, n,
                                                            decoding_params.stream, cublasAlgoMap_,
                                                            cublas_workspace_);
                        POP_RANGE
                    }

#ifndef NDEBUG
                    cudaDeviceSynchronize();
                    check_cuda_error(cudaGetLastError());
#endif
                    
                    if(t_parallel_param_.world_size == 1)
                    {
                        apply_temperature_penalty_kernelLauncher(logits_buf_,
                                                                (DataType_) args_.temperature_,
                                                                local_batch,
                                                                args_.vocab_size_,
                                                                n,
                                                                decoding_params.stream);
                    }
                    else
                    {
                        if(t_parallel_param_.rank == t_parallel_param_.world_size - 1)
                        {
                            apply_temperature_penalty_kernelLauncher(nccl_logits_buf_ + t_parallel_param_.rank * local_batch * n,
                                                                    (DataType_) args_.temperature_,
                                                                    local_batch,
                                                                    args_.vocab_size_ - n * t_parallel_param_.rank,
                                                                    n,
                                                                    decoding_params.stream);
                        }
                        else
                        {
                            apply_temperature_penalty_kernelLauncher(nccl_logits_buf_ + t_parallel_param_.rank * local_batch * n,
                                                                    (DataType_) args_.temperature_,
                                                                    local_batch,
                                                                    n,
                                                                    n,
                                                                    decoding_params.stream);
                        }
                    }

#ifndef NDEBUG
                    cudaDeviceSynchronize();
                    check_cuda_error(cudaGetLastError());
#endif

                    // reduce and concat the reuslt
                    if(t_parallel_param_.world_size > 1)
                    {
                        PUSH_RANGE("After Transformer/all2all_gather")
                        all2all_gather(nccl_logits_buf_, nccl_logits_buf_, local_batch * n, 
                                       t_parallel_param_, decoding_params.stream);
                        POP_RANGE
                        
                        transpose_axis_01_kernelLauncher(logits_buf_, nccl_logits_buf_, 
                                                         t_parallel_param_.world_size, local_batch, n, decoding_params.stream);
                    }

#ifndef NDEBUG
                    cudaDeviceSynchronize();
                    check_cuda_error(cudaGetLastError());
#endif

                    n = args_.vocab_size_padded_;

                    // Apply repetition penalty.
                    if (args_.repetition_penalty_ != 1.0) {
                        PUSH_RANGE("After Transformer/Repetition_penalty")
                        apply_repetition_penalty_kernelLauncher(logits_buf_,
                                                                args_.repetition_penalty_,
                                                                decoding_params.d_start_ids,
                                                                decoding_params.output_ids,
                                                                m,
                                                                local_batch,
                                                                args_.vocab_size_,
                                                                n,
                                                                decoding_params.d_start_lengths,
                                                                max_input_len,
                                                                step,
                                                                ite,
                                                                decoding_params.stream);
                        POP_RANGE
                    }

#ifndef NDEBUG
                    cudaDeviceSynchronize();
                    check_cuda_error(cudaGetLastError());
#endif

                    // Sampling
                    if(args_.candidate_num_ > 0 && args_.probability_threshold_ == 0.0)
                    {
                        PUSH_RANGE("After Transformer/Sampling")
                        // top k sampling
                        topK_sampling_kernel_kernelLauncher_v2(topk_workspace_,
                                                               topk_workspace_size_,
                                                               logits_buf_,
                                                               decoding_params.output_ids + step * m + ite * local_batch,
                                                               nullptr,
                                                               finished_buf_ + ite * local_batch,
                                                               curandstate_buf_, // used as random number
                                                               args_,
                                                               decoding_params.stream,
                                                               local_batch);
                        POP_RANGE
                    }
                    else if(args_.candidate_num_ == 0 && args_.probability_threshold_ > 0.0f)
                    {
                        PUSH_RANGE("After Transformer/Sampling")
                        // top p sampling
                        softmax_kernelLauncher(logits_buf_,
                                               (DataType_*) nullptr,
                                               args_.end_id_,
                                               finished_buf_ + ite * local_batch,
                                               local_batch,
                                               args_.vocab_size_padded_,
                                               args_.vocab_size_,
                                               decoding_params.stream);
#ifndef NDEBUG
                        cudaDeviceSynchronize();
                        check_cuda_error(cudaGetLastError());
#endif
                        topP_sampling_kernel_kernelLauncher_v2(topp_workspace_,
                                                               topp_workspace_size_,
                                                               logits_buf_,
                                                               topp_id_vals_buf_,
                                                               topp_offset_buf_,
                                                               begin_topp_offset_buf_,
                                                               finished_buf_ + ite * local_batch,
                                                               curandstate_buf_,
                                                               args_,
                                                               decoding_params.output_ids + step * m + ite * local_batch,
                                                               nullptr,
                                                               n,
                                                               decoding_params.stream,
                                                               local_batch);

                        POP_RANGE
                    }
                    else if(args_.candidate_num_ > 0 && args_.probability_threshold_ > 0.0f)
                    {
                        PUSH_RANGE("After Transformer/Sampling")
                        topK_topP_sampling_kernel_kernelLauncher_v2(topk_topp_workspace_,
                                                                    topk_topp_workspace_size_,
                                                                    decoding_params.output_ids + step * m + ite * local_batch,
                                                                    logits_buf_,
                                                                    finished_buf_ + ite * local_batch,
                                                                    curandstate_buf_,
                                                                    args_,
                                                                    decoding_params.stream,
                                                                    local_batch);
                        POP_RANGE
                    }
#ifndef NDEBUG
                    cudaDeviceSynchronize();
                    check_cuda_error(cudaGetLastError());
#endif
                }
                if(step < (size_t)max_input_len)
                {
                    // Replace the sampled id by start ids
                    set_start_ids_kernelLauncher(decoding_params.output_ids, decoding_params.d_start_ids, max_input_len,
                                                 step, ite, request_batch_size, local_batch, args_.end_id_, decoding_params.stream);
                }

                if(l_parallel_param_.rank == l_parallel_param_.world_size - 1 && l_parallel_param_.world_size > 1)
                {
                    PUSH_RANGE("token/send")
                    nccl_send(decoding_params.output_ids + step * m + ite * local_batch, local_batch, 0, l_parallel_param_.nccl_comm, decoding_params.stream);
                    POP_RANGE
                }

#ifndef NDEBUG
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
#endif

                if(l_parallel_param_.rank == l_parallel_param_.world_size - 1 && l_parallel_param_.world_size > 1 && step < max_len - 1)
                {
                    nccl_broadcast(finished_buf_ + ite * local_batch, local_batch, l_parallel_param_.world_size - 1, l_parallel_param_, decoding_params.stream);
                }
#ifndef NDEBUG
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
#endif
            } // end for ite for loop

            if (is_generation_done) {
                break;
            }
        } // end for decoding step for loop
        if(l_parallel_param_.rank == 0 && l_parallel_param_.world_size > 1)
        {
            for(size_t ite = 0; ite < request_batch_size / local_batch; ite++)
            {
                nccl_recv(decoding_params.output_ids + (max_len - 1) * m + ite * local_batch,
                          local_batch, l_parallel_param_.world_size - 1,
                          l_parallel_param_.nccl_comm, decoding_params.stream);
            }
        }
    } // end of forward

    virtual ~DecodingGpt()
    {
        delete[] K_cache_;
        delete[] V_cache_;
        delete decoder_;
        allocator_.free(buf_);
        delete [] h_finished_buf_;
    }

    inline int get_num_layer() {return args_.decoder_layers_;}

    inline void set_local_batch_size(int local_batch)
    { 
        l_parallel_param_.local_batch_size = local_batch;
        decoder_->set_local_batch_size(local_batch);
    }
};

} //namespace fastertransformer