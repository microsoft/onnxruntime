// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/cuda_common.h"

#include "core/common/gsl.h"
#include "contrib_ops/cpu/transformers/sequences.h"
#include "contrib_ops/cpu/transformers/generation_shared.h"

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}
}  // namespace onnxruntime

namespace onnxruntime {
namespace contrib {
// These are CUDA specific device helper implementations
namespace GenerationCudaDeviceHelper {

Status ReorderPastState(
    const void* cuda_device_prop,
    Tensor& past_state,
    Tensor& past_state_staging,
    Stream* stream);

Status InitCacheIndir(
    Tensor& cache_indir,
    Stream* stream);

Status TopK(const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
            AllocatorPtr allocator,
            Stream* stream,
            onnxruntime::concurrency::ThreadPool* threadpool,
            Tensor& output_values,
            Tensor& output_indices);

Status AddToFeeds(Stream* ort_stream,
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
                   Stream* ort_stream);

std::unique_ptr<transformers::IBeamScorer> CreateBeamScorer(const transformers::IGenerationParameters& parameters, AllocatorPtr& allocator, AllocatorPtr& allocator_cpu, Stream* ort_stream);

template <typename T>
void InitGreedyState(transformers::IGreedySearchState<T>* greedy_state,
                     gsl::span<int32_t>& sequence_lengths,
                     Stream* stream);

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
                  int copyDirection);

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

template <typename T>
Status ExpandBuffer(
    Stream* stream,
    const OrtValue& input,
    int num_beams,
    AllocatorPtr allocator,
    OrtValue& expanded,
    bool only_copy_shape,
    int max_sequence_length = 0);

}  // namespace GenerationCudaDeviceHelper
}  // namespace contrib
}  // namespace onnxruntime
