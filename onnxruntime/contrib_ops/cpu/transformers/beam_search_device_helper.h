#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/framework/allocator.h"
#endif

#include "gsl/gsl"
#include "logits_processor.h"
#include "beam_search_shared.h"

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
    void* stream,  // cudaStream_t stream,
    onnxruntime::concurrency::ThreadPool* threadpool,
    std::unique_ptr<Tensor>& output_values,
    std::unique_ptr<Tensor>& output_indices)>;

// Create subgraph inputs: input_ids, position_ids and attention_mask
using CreateInputsFunc = std::function<Status(
    const Tensor* original_input_ids,
    int num_beams,
    int pad_token_id,
    gsl::span<int32_t>& sequence_lengths,
    AllocatorPtr alloactor,
    OrtValue& expanded_input_ids,
    OrtValue& expanded_position_ids,
    OrtValue& expanded_attention_mask)>;

using AddToFeedsFunc = std::function<Status(
    const IExecutionProvider* provider,
    OrtValue& input_ids,
    OrtValue& position_ids,
    OrtValue& attention_mask,
    std::vector<OrtValue>& feeds,
    IAllocatorUniquePtr<char>& buffer)>;

template <typename T>
using InitBeamStateFunc = std::function<void(
    transformers::IBeamSearchState<T>* beam_state,
    transformers::IBeamSearchCpuState* cpu_state,
    gsl::span<int32_t>& sequence_lengths,
    int batch_size,
    int num_beams,
    gsl::span<const int32_t> input_ids_in_cpu,
    int sequence_length,
    int max_length,
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

template <typename T>
using UpdateFeedsFunc = std::function<Status(
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

Status CreateInputs(
    const Tensor* original_input_ids,
    int num_beams,
    int pad_token_id,
    gsl::span<int32_t>& sequence_lengths,
    AllocatorPtr alloactor,
    OrtValue& expanded_input_ids,
    OrtValue& expanded_position_ids,
    OrtValue& expanded_attention_mask);

Status AddToFeeds(
    const IExecutionProvider* execution_provider,
    OrtValue& input_ids,
    OrtValue& position_ids,
    OrtValue& attention_mask,
    std::vector<OrtValue>& feeds,
    IAllocatorUniquePtr<char>& buffer);

template <typename T>
void InitBeamState(transformers::IBeamSearchState<T>* beam_state,
                   transformers::IBeamSearchCpuState* cpu_state,
                   gsl::span<int32_t>& sequence_lengths,
                   int batch_size,
                   int num_beams,
                   gsl::span<const int32_t> input_ids_in_cpu,
                   int sequence_length,
                   int max_length,
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

template <typename T>
Status UpdateFeeds(
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

}  // namespace BeamSearchCpuDeviceHelper
}  // namespace contrib
}  // namespace onnxruntime