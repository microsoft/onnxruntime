// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/framework/allocator.h"
#endif

#include <vector>
#include "gsl/gsl"
#include "contrib_ops/cpu/transformers/logits_processor.h"
#include "contrib_ops/cpu/transformers/beam_search_shared.h"

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

namespace BeamSearchDeviceHelper {
using TopkFunc = std::function<Status(
    const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
    AllocatorPtr allocator,
    void* stream,  // cudaStream_t
    onnxruntime::concurrency::ThreadPool* threadpool,
    std::unique_ptr<Tensor>& output_values,
    std::unique_ptr<Tensor>& output_indices)>;

// Create subgraph inputs: input_ids, position_ids and attention_mask (for GPT-2).
using CreateGptInputsFunc = std::function<Status(
    const Tensor* original_input_ids,
    int num_beams,
    int pad_token_id,
    gsl::span<int32_t>& sequence_lengths,
    AllocatorPtr allocator,
    OrtValue& expanded_input_ids,
    OrtValue& expanded_position_ids,
    OrtValue& expanded_attention_mask)>;

using AddToFeedsFunc = std::function<Status(
    const IExecutionProvider* provider,
    std::initializer_list<OrtValue> inputs,
    std::vector<OrtValue>& feeds,
    IAllocatorUniquePtr<char>& buffer)>;

template <typename T>
using InitBeamStateFunc = std::function<void(
    transformers::IBeamSearchState<T>* beam_state,
    gsl::span<int32_t>& sequence_lengths,
    int batch_size,
    int num_beams,
    void* stream)>;

template <typename T>
using ProcessLogitsFunc = std::function<Status(
    const OrtValue& logits,                                 // logits output of subgraph
    transformers::IBeamSearchState<T>* beam_state,          // state
    transformers::IBeamSearchCpuState* cpu_state,           // state in CPU
    transformers::ISequences* sequences,                    // sequences
    AllocatorPtr& allocator,                                // default allocator
    onnxruntime::concurrency::ThreadPool* thread_pool,      // thread pool (for CPU only)
    transformers::ILogitsProcessorList* logits_processors,  // logits processors
    transformers::IBeamScorer* beam_scorer,                 // beam scorer
    const transformers::IBeamSearchParameters* parameters,  // parameters
    int step,                                               // iteration counter
    void* stream,                                           // cuda stream (for CUDA only)
    const transformers::IConsoleDumper* dumper)>;           // tensor dumper

template <typename T>
using DeviceCopyFunc = std::function<Status(
    gsl::span<T> target,
    gsl::span<const T> source,
    void* stream,
    int copyDirection)>;

// Update subgraph inputs given outputs of last iteration (for GPT-2).
template <typename T>
using UpdateGptFeedsFunc = std::function<Status(
    AllocatorPtr allocator,
    void* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    int num_beams,
    const transformers::IConsoleDumper* dumper)>;

// Create encoder inputs (for encoder-decoder model like T5).
using CreateEncoderInputsFunc = std::function<Status(
    const Tensor* original_encoder_input_ids,
    int num_beams,
    int pad_token_id,
    int start_token_id,
    AllocatorPtr allocator,
    OrtValue& expanded_encoder_input_ids,
    OrtValue& expanded_encoder_attention_mask,
    OrtValue& expanded_decoder_input_ids)>;

// Update decoder inputs given decoder outputs of last iteration (for encoder-decoder model like T5).
template <typename T>
using UpdateDecoderFeedsFunc = std::function<Status(
    AllocatorPtr allocator,
    void* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int num_present_tensors,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    int num_beams,
    const transformers::IConsoleDumper* dumper)>;
}  // namespace BeamSearchDeviceHelper

// These are CPU specific device helper implementations
namespace BeamSearchCpuDeviceHelper {
Status TopK(
    const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
    AllocatorPtr allocator,
    void* stream,
    onnxruntime::concurrency::ThreadPool* threadpool,
    std::unique_ptr<Tensor>& output_values,
    std::unique_ptr<Tensor>& output_indices);

Status AddToFeeds(
    const IExecutionProvider* execution_provider,
    std::initializer_list<OrtValue> inputs,
    std::vector<OrtValue>& feeds,
    IAllocatorUniquePtr<char>& buffer);

template <typename T>
void InitBeamState(transformers::IBeamSearchState<T>* beam_state,
                   gsl::span<int32_t>& sequence_lengths,
                   int batch_size,
                   int num_beams,
                   void* stream);

template <typename T>
Status ProcessLogits(const OrtValue& logits,                                 // logits output of subgraph
                     transformers::IBeamSearchState<T>* beam_state,          // state
                     transformers::IBeamSearchCpuState* cpu_state,           // state in CPU
                     transformers::ISequences* sequences,                    // sequences
                     AllocatorPtr& allocator,                                // default allocator
                     onnxruntime::concurrency::ThreadPool* thread_pool,      // thread pool (for CPU only)
                     transformers::ILogitsProcessorList* logits_processors,  // logits processors
                     transformers::IBeamScorer* beam_scorer,                 // beam scorer
                     const transformers::IBeamSearchParameters* parameters,  // parameters
                     int step,                                               // iteration counter
                     void* stream,                                           // cuda stream (for CUDA only)
                     const transformers::IConsoleDumper* dumper);            // tensor dumper

template <typename T>
Status DeviceCopy(gsl::span<T> target,
                  gsl::span<const T> source,
                  void* stream,
                  int copyDirectionn);

// ---------------------------------------------------------------
// Functions for GPT model only
// ---------------------------------------------------------------

Status CreateGptInputs(
    const Tensor* original_input_ids,
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
    void* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    int num_beams,
    const transformers::IConsoleDumper* dumper);

// ---------------------------------------------------------------
// Functions for encoder-decoder model like T5
// ---------------------------------------------------------------
Status CreateEncoderInputs(
    const Tensor* original_encoder_input_ids,
    int num_beams,
    int pad_token_id,
    int start_token_id,
    AllocatorPtr allocator,
    OrtValue& expanded_encoder_input_ids,
    OrtValue& expanded_encoder_attention_mask,
    OrtValue& expanded_decoder_input_ids);

// Update decoder inputs given decoder outputs of last iteration.
template <typename T>
Status UpdateDecoderFeeds(
    AllocatorPtr allocator,
    void* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int num_present_tensors,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    int num_beams,
    const transformers::IConsoleDumper* dumper);

// ---------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------
template <typename T>
void ExpandInputs(const OrtValue& input, int num_beams, AllocatorPtr allocator, OrtValue& expanded);

}  // namespace BeamSearchCpuDeviceHelper
}  // namespace contrib
}  // namespace onnxruntime
