// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/framework/allocator.h"
#endif

#include <vector>
#include "core/common/gsl.h"
#include "contrib_ops/cpu/transformers/logits_processor.h"
#include "contrib_ops/cpu/transformers/generation_shared.h"

namespace onnxruntime {
class IExecutionProvider;
namespace concurrency {
class ThreadPool;
}
}  // namespace onnxruntime

namespace onnxruntime {
namespace contrib {

enum DeviceCopyDirection {
  hostToHost = 0,
  hostToDevice = 1,
  deviceToHost = 2,
  deviceToDevice = 3
};

namespace GenerationDeviceHelper {

#ifdef USE_CUDA
using ReorderPastStateFunc = std::function<Status(
    const void* cuda_device_prop,  // cudaDeviceProp
    Tensor& past_state,
    Tensor& past_state_staging,
    Stream* stream)>;  // cublasHandle_t

using InitCacheIndirFunc = std::function<Status(
    Tensor& cache_indir,
    Stream* stream)>;
#endif

using TopkFunc = std::function<Status(
    const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
    AllocatorPtr allocator,
    Stream* stream,  // cudaStream_t
    onnxruntime::concurrency::ThreadPool* threadpool,
    Tensor& output_values,
    Tensor& output_indices)>;

// Create subgraph inputs: input_ids, position_ids and attention_mask (for GPT-2).
using CreateGptInputsFunc = std::function<Status(
    const Tensor* original_input_ids,
    const OrtValue* attn_mask_value,
    int num_beams,
    int pad_token_id,
    gsl::span<int32_t>& sequence_lengths,
    AllocatorPtr allocator,
    OrtValue& expanded_input_ids,
    OrtValue& expanded_position_ids,
    OrtValue& expanded_attention_mask)>;

using AddToFeedsFunc = std::function<Status(
    Stream* ort_stream,
    std::initializer_list<OrtValue> inputs,
    std::vector<OrtValue>& feeds,
    IAllocatorUniquePtr<char>& buffer,
    AllocatorPtr device_allocator,
    AllocatorPtr host_allocator,
    const OrtMemoryInfo& location)>;

template <typename T>
using InitBeamStateFunc = std::function<void(
    transformers::IBeamSearchState<T>* beam_state,
    gsl::span<int32_t>& sequence_lengths,
    int batch_size,
    int num_beams,
    Stream* stream)>;

using CreateBeamScorer = std::function<std::unique_ptr<transformers::IBeamScorer>(
    const transformers::IGenerationParameters& parameters,
    AllocatorPtr& allocator,
    AllocatorPtr& allocator_cpu,
    Stream* stream)>;

template <typename T>
using InitGreedyStateFunc = std::function<void(
    transformers::IGreedySearchState<T>* greedy_state,
    gsl::span<int32_t>& sequence_lengths,
    Stream* stream)>;

template <typename T>
using ProcessLogitsFunc = std::function<Status(
    const OrtValue& logits,                                 // logits output of subgraph
    transformers::IBeamSearchState<T>* beam_state,          // state
    transformers::ISequences* sequences,                    // sequences
    AllocatorPtr& allocator,                                // default allocator
    onnxruntime::concurrency::ThreadPool* thread_pool,      // thread pool (for CPU only)
    transformers::ILogitsProcessorList* logits_processors,  // logits processors
    transformers::IBeamScorer* beam_scorer,                 // beam scorer
    const transformers::IGenerationParameters* parameters,  // parameters
    int step,                                               // iteration counter
    Stream* stream,                                         // cuda stream (for CUDA only)
    const transformers::IConsoleDumper* dumper)>;           // tensor dumper

template <typename T>
using GreedySearchProcessLogitsFunc = std::function<Status(
    const OrtValue& logits,                                 // logits output of subgraph
    transformers::IGreedySearchState<T>* greedy_state,      // state
    transformers::ISamplingState<T>* sampling_state,        // sampling buffers
    transformers::ISequences* sequences,                    // sequences
    AllocatorPtr& allocator,                                // default allocator
    onnxruntime::concurrency::ThreadPool* thread_pool,      // thread pool (for CPU only)
    transformers::ILogitsProcessorList* logits_processors,  // logits processors
    const transformers::IGenerationParameters* parameters,  // parameters
    bool do_sampling,                                       // whether to do sampling
    int step,                                               // iteration counter
    Stream* ort_stream,                                     // cuda stream (for CUDA only)
    const transformers::IConsoleDumper* dumper)>;           // tensor dumper

template <typename T>
using DeviceCopyFunc = std::function<Status(
    gsl::span<T> target,
    gsl::span<const T> source,
    Stream* stream,
    int copyDirection)>;

// Update subgraph inputs given outputs of last iteration (for GPT-2).
template <typename T>
using UpdateGptFeedsFunc = std::function<Status(
    AllocatorPtr allocator,
    Stream* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    bool increase_position,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices_cpu,
    gsl::span<const int32_t> beam_indices_gpu,
    int num_beams,
    int gpt_subgraph_first_past_input_idx,
    int gpt_subgraph_first_present_output_idx,
    bool past_present_share_buffer,
    int past_sequence_len,
    int input_sequence_len,
    bool need_cache_indir)>;

// Create encoder inputs (for encoder-decoder model like T5).
using CreateEncoderInputsFunc = std::function<Status(
    const Tensor* original_encoder_input_ids,
    const OrtValue* attn_mask_value,
    const OrtValue* input_images_value,
    int pad_token_id,
    int start_token_id,
    AllocatorPtr allocator,
    OrtValue& encoder_input_ids,
    OrtValue& encoder_attention_mask,
    OrtValue& encoder_input_images,
    OrtValue& decoder_input_ids)>;

// Update decoder inputs given decoder outputs of last iteration (for encoder-decoder model like T5).
template <typename T>
using UpdateDecoderFeedsFunc = std::function<Status(
    AllocatorPtr allocator,
    Stream* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int num_present_tensors,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    gsl::span<const int32_t> beam_indices_gpu,
    int num_beams,
    int t5_decoder_first_past_input_idx,
    int t5_decoder_first_present_output_idx,
    bool use_sequence_as_input_ids,
    int current_length,
    int input_sequence_len,
    bool past_present_share_buffer,
    bool need_cache_indir,
    transformers::Sequences& sequences,
    const transformers::IConsoleDumper* dumper)>;

//------------------------------------------------
//  Modified functions for Whisper Model
//------------------------------------------------
using CreateWhisperEncoderInputsFunc = std::function<Status(
    const Tensor* original_encoder_input_features,
    const OrtValue* original_decoder_input_ids_value,
    int start_token_id,
    AllocatorPtr allocator,
    OrtValue& encoder_input_ids,
    OrtValue& decoder_input_ids)>;

template <typename T>
using ExpandBufferFunc = std::function<Status(
    Stream* stream,
    const OrtValue& input,
    int num_beams,
    AllocatorPtr allocator,
    OrtValue& expanded,
    bool only_copy_shape,
    int max_sequence_length)>;

using UpdateDecoderCrossQKFunc = std::function<Status(
    int iteration_number,
    Stream* stream,
    OrtValue* cross_qks,
    IAllocatorUniquePtr<float*>& qk_layer_pointers,
    int num_layers,
    int cross_qk_layer_head_pair_count,
    const int* cross_qk_layer_head_pairs,
    float* cross_qk_buffer_data,
    int max_length,
    AllocatorPtr allocator)>;

using FinalizeDecoderCrossQKFunc = std::function<Status(
    Stream* stream,
    int iteration_number,
    int context_decoding_len,
    int batch_size,
    int num_beams,
    int max_length,
    int cross_qk_layer_head_pair_count,
    const int* cross_qk_layer_head_pairs,
    int frames_of_k,
    const float* cross_qk_buffer_data,
    float* cross_qk_output,
    int num_return_sequences,
    const int* cache_indir_data,
    gsl::span<const int32_t> beam_indices)>;

}  // namespace GenerationDeviceHelper

// These are CPU specific device helper implementations
namespace GenerationCpuDeviceHelper {
Status TopK(
    const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
    AllocatorPtr allocator,
    Stream* stream,
    onnxruntime::concurrency::ThreadPool* threadpool,
    Tensor& output_values,
    Tensor& output_indices);

Status AddToFeeds(
    Stream* ort_stream,
    std::initializer_list<OrtValue> inputs,
    std::vector<OrtValue>& feeds,
    IAllocatorUniquePtr<char>& buffer,
    AllocatorPtr device_allocator,
    AllocatorPtr host_allocator,
    const OrtMemoryInfo& location);

template <typename T>
void InitBeamState(transformers::IBeamSearchState<T>* beam_state,
                   gsl::span<int32_t>& sequence_lengths,
                   int batch_size,
                   int num_beams,
                   Stream* stream);

template <typename T>
void InitGreedyState(transformers::IGreedySearchState<T>* greedy_state,
                     gsl::span<int32_t>& sequence_lengths,
                     Stream* ort_stream);

template <typename T>
Status ProcessLogits(const OrtValue& logits,                                 // logits output of subgraph
                     transformers::IBeamSearchState<T>* beam_state,          // state
                     transformers::ISequences* sequences,                    // sequences
                     AllocatorPtr& allocator,                                // default allocator
                     onnxruntime::concurrency::ThreadPool* thread_pool,      // thread pool (for CPU only)
                     transformers::ILogitsProcessorList* logits_processors,  // logits processors
                     transformers::IBeamScorer* beam_scorer,                 // beam scorer
                     const transformers::IGenerationParameters* parameters,  // parameters
                     int step,                                               // iteration counter
                     Stream* stream,                                         // cuda stream (for CUDA only)
                     const transformers::IConsoleDumper* dumper);            // tensor dumper

template <typename T>
Status GreedySearchProcessLogits(const OrtValue& logits,                                 // logits output of subgraph
                                 transformers::IGreedySearchState<T>* greedy_state,      // state
                                 transformers::ISamplingState<T>* sampling_state,        // sampling buffers
                                 transformers::ISequences* sequences,                    // sequences
                                 AllocatorPtr& allocator,                                // default allocator
                                 onnxruntime::concurrency::ThreadPool* thread_pool,      // thread pool (for CPU only)
                                 transformers::ILogitsProcessorList* logits_processors,  // logits processors
                                 const transformers::IGenerationParameters* parameters,  // parameters
                                 bool do_sampling,                                       // whether to do sampling
                                 int step,                                               // iteration counter
                                 Stream* stream,                                         // cuda stream (for CUDA only)
                                 const transformers::IConsoleDumper* dumper);            // tensor dumper

template <typename T>
Status DeviceCopy(gsl::span<T> target,
                  gsl::span<const T> source,
                  Stream* stream,
                  int copyDirectionn);

// ---------------------------------------------------------------
// Functions for GPT model only
// ---------------------------------------------------------------

Status CreateGptInputs(
    const Tensor* original_input_ids,
    const OrtValue* attn_mask_value,
    int num_beams,
    int pad_token_id,
    gsl::span<int32_t>& sequence_lengths,
    AllocatorPtr allocator,
    OrtValue& expanded_input_ids,
    OrtValue& expanded_position_ids,
    OrtValue& expanded_attention_mask);

template <typename T>
Status UpdateGptFeeds(
    AllocatorPtr allocator,
    Stream* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    bool increase_position,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices_cpu,
    gsl::span<const int32_t> beam_indices_gpu,
    int num_beams,
    int gpt_subgraph_first_past_input_idx,
    int gpt_subgraph_first_present_output_idx,
    bool past_present_share_buffer,
    int past_sequence_len,
    int input_sequence_len,
    bool need_cache_indir);

// ---------------------------------------------------------------
// Functions for encoder-decoder model like T5
// ---------------------------------------------------------------
template <typename T>
Status CreateEncoderInputs(
    const Tensor* original_encoder_input_ids,
    const OrtValue* attn_mask_value,
    const OrtValue* input_images_value,
    int pad_token_id,
    int start_token_id,
    AllocatorPtr allocator,
    OrtValue& encoder_input_ids,
    OrtValue& encoder_attention_mask,
    OrtValue& encoder_input_images,
    OrtValue& decoder_input_ids);

// Update decoder inputs given decoder outputs of last iteration.
template <typename T>
Status UpdateDecoderFeeds(
    AllocatorPtr allocator,
    Stream* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int num_present_tensors,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    gsl::span<const int32_t> beam_indices_gpu,
    int num_beams,
    int t5_decoder_first_past_input_idx,
    int t5_decoder_first_present_output_idx,
    bool use_sequence_as_input_ids,
    int current_length,
    int input_sequence_len,
    bool past_present_share_buffer,
    bool need_cache_indir,
    transformers::Sequences& sequences,
    const transformers::IConsoleDumper* dumper);

// ---------------------------------------------------------------
// Functions for encoder-decoder model with float input like Whisper
// ---------------------------------------------------------------
template <typename T>
Status CreateWhisperEncoderInputs(
    const Tensor* original_encoder_input_features,
    const OrtValue* original_decoder_input_ids_value,
    int start_token_id,
    AllocatorPtr allocator,
    OrtValue& encoder_input_ids,
    OrtValue& decoder_input_ids);

// ---------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------
template <typename T>
void ExpandInputs(const OrtValue& input, int num_beams, AllocatorPtr allocator, OrtValue& expanded);

template <typename T>
Status ExpandBuffer(
    Stream* stream,
    const OrtValue& input,
    int num_beams,
    AllocatorPtr allocator,
    OrtValue& expanded,
    bool only_copy_shape,
    int max_sequence_length);

Status UpdateDecoderCrossQK(
    int iteration_number,
    Stream* stream,
    OrtValue* cross_qks,
    IAllocatorUniquePtr<float*>& qk_layer_pointers,
    int num_layers,
    int cross_qk_layer_head_pair_count,
    const int* cross_qk_layer_head_pairs,
    float* cross_qk_buffer_data,
    int max_length,
    AllocatorPtr allocator);

Status FinalizeDecoderCrossQK(
    Stream* stream,
    int iteration_number,
    int context_decoding_len,
    int batch_size,
    int num_beams,
    int max_length,
    int cross_qk_layer_head_pair_count,
    const int* cross_qk_layer_head_pairs,
    int frames_of_k,
    const float* cross_qk_buffer_data,
    float* cross_qk_output,
    int num_return_sequences,
    const int* cache_indir_data,
    gsl::span<const int32_t> beam_indices);

}  // namespace GenerationCpuDeviceHelper
}  // namespace contrib
}  // namespace onnxruntime
