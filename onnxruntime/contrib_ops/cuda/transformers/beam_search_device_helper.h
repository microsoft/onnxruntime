#pragma once

#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/cuda_common.h"

#include "gsl/gsl"
#include "contrib_ops/cpu/transformers/beam_search_shared.h"

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}
}  // namespace onnxruntime

namespace onnxruntime {
namespace contrib {
// These are CUDA specific device helper implementations
namespace BeamSearchCudaDeviceHelper {

Status TopK(const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
            AllocatorPtr allocator,
            void* stream,
            onnxruntime::concurrency::ThreadPool* threadpool,
            std::unique_ptr<Tensor>& output_values,
            std::unique_ptr<Tensor>& output_indices);

Status AddToFeeds(const IExecutionProvider* execution_provider,
                  OrtValue& input_ids,
                  OrtValue& position_ids,
                  OrtValue& attention_mask,
                  std::vector<OrtValue>& feeds,
                  IAllocatorUniquePtr<char>& buffer);

void InitBeamState(transformers::IBeamSearchState<float>* beam_state,
                   transformers::IBeamSearchCpuState<float>* cpu_state,
                   gsl::span<int64_t>& sequence_lengths,
                   int batch_size,
                   int num_beams,
                   gsl::span<const int64_t> input_ids_in_cpu,
                   int sequence_length,
                   int max_length,
                   void* stream);

Status ProcessLogits(const OrtValue& logits,                                        // logits output of subgraph
                     transformers::IBeamSearchState<float>* beam_state,             // state in device
                     transformers::IBeamSearchCpuState<float>* cpu_state,           // state in CPU
                     transformers::ISequences* sequences,                           // sequences
                     AllocatorPtr& allocator,                                       // default allocator
                     onnxruntime::concurrency::ThreadPool* thread_pool,             // thread pool (for CPU only)
                     transformers::ILogitsProcessorList<float>* logits_processors,  // logits processors
                     transformers::IBeamScorer<float>* beam_scorer,                 // beam scorer
                     const transformers::IBeamSearchParameters* parameters,         // parameters
                     int step,                                                      // iteration counter
                     void* stream,                                                  // cuda stream (for CUDA only)
                     const transformers::IConsoleDumper* dumper);                   // tensor dumper

Status DeviceCopy(gsl::span<float> target,
                  gsl::span<const float> source,
                  void* stream,
                  int copyDirection);

Status UpdateFeeds(
    AllocatorPtr allocator,
    void* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    gsl::span<const int64_t> beam_next_tokens,
    gsl::span<const int64_t> beam_indices,
    int num_beams,
    const transformers::IConsoleDumper* dumper);

}  // namespace BeamSearchCudaDeviceHelper
}  // namespace contrib
}  // namespace onnxruntime