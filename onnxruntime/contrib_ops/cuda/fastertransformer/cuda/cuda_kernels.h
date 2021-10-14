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
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include "contrib_ops/cuda/fastertransformer/utils/arguments.h"
#include "contrib_ops/cuda/fastertransformer/cuda/topk_kernels.cuh"

namespace fastertransformer
{

typedef unsigned int uint;

#define FINAL_MASK 0xffffffff
#define CUDART_PI_F 3.141592654f

/* ********************************** common kernel *********************************** */

template <typename T>
void init_kernelLauncher(bool* finished, int* sequence_length, int* word_ids, 
                        T* cum_log_probs, const int sentence_id, 
                        const int batch_size, const int beam_width, cudaStream_t stream);

template <typename T>
void embedding_lookup_sine_position_encoding_kernel_launcher(T *from_tensor,
                                                             const T *embedding_table,
                                                             const T *position_encoding_table,
                                                             const int *word_ids,
                                                             const int batch_size,
                                                             const int hidden_units,
                                                             cudaStream_t stream);

template <typename T>
void remove_sequence_length_padding_kernelLauncher(const T *src, T *tgt,
                                                   const int *tmp_mask_offset,
                                                   int *mask_offset, const int m,
                                                   const int n, cudaStream_t stream);

template <typename T>
void rebuild_sequence_length_padding_kernelLauncher(const T *src, T *tgt,
                                                    const int *mask_offset, const int m,
                                                    const int n, cudaStream_t stream);

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
                                                cudaStream_t stream);

template <typename T>
void start_id_embedding_position_lookups_kernel_launcher(T* from_tensor,
                                                         int* output_ids,
                                                         const T* embedding_table, 
                                                         const T* pos_table, 
                                                         const int* word_ids,
                                                         const int start_step,
                                                         const int length,
                                                         const int max_length,
                                                         const int batch_size,
                                                         const int hidden_units, 
                                                         cudaStream_t stream);

template <typename T>
void apply_temperature_penalty_kernelLauncher(T* logits,
                                              const T temperature,
                                              const int m,
                                              const int vocab_size,
                                              const int vocab_size_padd,
                                              cudaStream_t stream);

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
                                              cudaStream_t stream);

void set_start_ids_kernelLauncher(int* out_ids,
                                  const int* in_ids,
                                  const int max_start_len,
                                  const int step,
                                  const int ite,
                                  const int batch_size,
                                  const int local_batch_size,
                                  const int end_id,
                                  cudaStream_t stream);

template <typename T>
void kernel_padding_kernelLauncher(T *padded_kernel, const T *kernel,
                                   const int row_dim, const int col_dim,
                                   const int padded_col_dim, cudaStream_t stream);

template <typename T1, typename T2>
void bias_padding_kernelLauncher(T1 *padded_bias, const T2 *bias,
                                 const int col_dim, const int padded_col_dim, 
                                 cudaStream_t stream);

template <typename T>
void transpose(T *out, const T *in, int batch,
               int height, int width, int stride, cudaStream_t stream);

template <typename DataType_>
void transpose_axis_01_kernelLauncher(DataType_ *out, DataType_ *in, const int dim0, 
                                    const int dim1, const int dim2, cudaStream_t stream);

void build_sequence_length_padding_offset_kernelLauncher(const int *sequence_length,
                                                         const int batch_size, const int max_seq_len,
                                                         int *valid_word_num, int *tmp_mask_offset,
                                                         cudaStream_t stream);

///template <typename T>
///void cuda_random_uniform_kernelLauncher(T *buffer, const int size);

/* *************************** end of common kernel *********************************** */

/* ********************************** BeamSearch kernel *********************************** */

void broadcast_kernelLauncher(float *log_probs, float *cum_log_probs,
                              const int batch_size, const int beam_width,
                              const int vocab_size, cudaStream_t stream);

template<typename T>
void update_logits(float* logits, const T *tmp_logits, const T* bias, const int end_ids, 
                   const bool* finished, const int m, const int n, 
                   cudaStream_t stream);

template <typename T>
void apply_logit_penalties(int step, 
                           T* log_probs, 
                           int* current_ids, 
                           int* previous_ids, 
                           int* parent_ids, 
                           GptArguments args,
                           cudaStream_t stream);

void update_kernelLauncher(float* log_probs, float* cum_log_probs,
                           bool* finished, int* parent_ids, int* sequence_length, 
                           int* word_ids, int* output_ids,
                           const int batch_size, const int beam_width,
                           const int vocab_size, cudaStream_t stream,
                           const int end_id, 
                           int* finished_count);

void update_kernelLauncher_v2(bool* finished, int* parent_ids, 
                              int* sequence_length, int* word_ids, 
                              int* output_ids, 
                              int* finished_count,
                              DecodingBeamsearchArguments args,
                              cudaStream_t stream);

template <typename T>
void update_KV_cache_kernelLauncher(T **key_cache, T **value_cache, const int *beam_ids,
                                    const bool* finished,
                                    const int batch_size, const int beam_width,
                                    const int head_num, const int size_per_head,
                                    const int step, const int decoder_max_seq_len,
                                    const int cache_size, const int decoder_layers,
                                    cudaStream_t stream);

void gather_tree_kernel_launcher(int max_time, int batch_size, int beam_width,
                                  int* step_ids, int* parent_ids, int* max_sequence_lengths,
                                  int end_token, int* beams, cudaStream_t stream);

/* *************************** end of BeamSearch kernel *********************************** */

/* ********************************** Sampling kernel *********************************** */

template <typename T>
size_t get_topp_sort_temp_storage_size(const T* log_probs,
                                       const int* id_vals,
                                       T* sorted_log_probs,
                                       int* sorted_id_vals,
                                       int* topp_offset_buf,
                                       const int batch_size,
                                       const int vocab_size);

void sampling_init_kernelLauncher(bool* finished, 
                                  int* sequence_length, 
                                  int* word_ids, 
                                  const int start_id, 
                                  const int batch_size, 
                                  cudaStream_t stream);
                                
void topp_initialization_kernelLauncher(bool* finished,
                                        int* sequence_length, 
                                        int* word_ids,
                                        int* topp_id_val_buf,
                                        int* topp_offset_buf,
                                        const int logits_buf_size,
                                        DecodingSamplingArguments args,
                                        cudaStream_t stream);
                        
void topp_initialization_kernelLauncher_v2(bool* finished,
                                        int* sequence_length, 
                                        int* word_ids,
                                        int* topp_id_val_buf,
                                        int* topp_offset_buf,
                                        int* begin_topp_offset_buf_,
                                        const int logits_buf_size,
                                        DecodingSamplingArguments args,
                                        cudaStream_t stream);

void init_topp_id_val_kernel_kernelLauncher(int *topp_id_val_buf,
                                            int *topp_offset_buf,
                                            const int batch_size,
                                            const int vocab_size,
                                            cudaStream_t stream);

template <typename T>
void update_logits_without_softmax(T* logits, const T* bias, const int end_ids, 
                                   const bool* finished, const int m, const int n, 
                                   cudaStream_t stream);

template <typename T>
void softmax_kernelLauncher(T* logits, const T* bias, const int end_ids,
                            const bool* finished, const int m, const int n_padded, const int n,
                            cudaStream_t stream);

/* *************************** end of Sampling kernel *********************************** */

/* *********************************** Debug tools *********************************** */

template <typename T>
void print_first_k(const T *buf, uint size, cudaStream_t stream);

template <typename T>
void print_abs_mean(const T *buf, uint size, cudaStream_t stream);

/* **************************** end of Debug tools *********************************** */

/* *************************** depreciated kernels *********************************** */

void topK(const float *log_probs, int *ids, const int batch_size,
          const int beam_width, const int vocab_size, cudaStream_t stream);

template <typename T>
void embedding_lookup(const T *embedding_table, const int *word_ids,
                      T *from_tensor, const int batch_size,
                      const int beam_width, const int hidden_units,
                      cudaStream_t stream);

template <typename T>
void sine_position_encoder(T *output, int step, int m, int n, cudaStream_t stream);

/* ******************** end of depreciated kernels *********************************** */

} //namespace fastertransformer
