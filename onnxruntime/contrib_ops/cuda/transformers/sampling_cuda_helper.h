// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cpu/tensor/utils.h"
#include "contrib_ops/cpu/transformers/generation_shared.h"
#include "core/providers/cuda/math/softmax.h"

#ifdef DEBUG_GENERATION
#include <iostream>
#endif

namespace onnxruntime {
namespace contrib {
namespace SamplingCudaHelper {

template <typename T>
class TopPSamplingCuda{
  public:
    TopPSamplingCuda(AllocatorPtr& allocator,
                     cudaStream_t cuda_stream,
                     transformers::ISamplingState<T>* sampling_state,
                     transformers::IGreedySearchState<T>* greedy_state,
                     const transformers::IGenerationParameters* parameters,
                     const transformers::IConsoleDumper* dumper):
      allocator_(allocator),
      cuda_stream_(cuda_stream),
      sampling_state_(sampling_state),
      greedy_state_(greedy_state),
      parameters_(parameters),
      dumper_(dumper) {}

    Status Sample(int step, gsl::span<T>& next_token_scores);

  private:
    AllocatorPtr& allocator_;
    cudaStream_t cuda_stream_;
    transformers::ISamplingState<T>* sampling_state_;
    transformers::IGreedySearchState<T>* greedy_state_;
    const transformers::IGenerationParameters* parameters_;
    const transformers::IConsoleDumper* dumper_;
};

template <typename T>
Status TopPSamplingCuda<T>::Sample(int step, gsl::span<T>& next_token_scores) {
  typedef typename ToCudaType<T>::MappedType CudaT;

  gsl::span<int>& d_index_in = sampling_state_->d_index_in;
  gsl::span<int>& d_offset = sampling_state_->d_offset;

  BufferUniquePtr& storage_buffer = sampling_state_->storage_buffer;
  size_t& temp_storage_bytes = sampling_state_->temp_storage_bytes;
  if (step == 1) {
    temp_storage_bytes = cuda::GetTempStorageSize<CudaT>(reinterpret_cast<CudaT*>(next_token_scores.data()),
                                                         d_index_in.data(),
                                                         d_offset.data(),
                                                         parameters_->batch_size * parameters_->vocab_size,
                                                         parameters_->batch_size,
                                                         cuda_stream_);

    cuda::LaunchSetupParamsKernel(d_index_in.data(),
                                  d_offset.data(),
                                  parameters_->batch_size,
                                  parameters_->vocab_size,
                                  cuda_stream_);

#ifdef DEBUG_GENERATION
  dumper_->Print("d_offset_buffer", d_offset.data(), parameters_->batch_size + 1, 1);
#endif

    void* temp_storage = allocator_->Alloc(sampling_state_->temp_storage_bytes);
    BufferUniquePtr temp_storage_buffer(temp_storage, BufferDeleter(allocator_));
    storage_buffer = std::move(temp_storage_buffer);
  }

  gsl::span<T>& d_sorted_score = sampling_state_->d_sorted_score;
  gsl::span<int>& d_index_out = sampling_state_->d_index_out;

#ifdef DEBUG_GENERATION
    dumper_->Print("temp_storage_bytes", sampling_state_->temp_storage_bytes, true);
#endif

  cuda::LaunchSortPairsDescending<CudaT>(storage_buffer.get(),
                                         temp_storage_bytes,
                                         reinterpret_cast<CudaT*>(next_token_scores.data()),
                                         reinterpret_cast<CudaT*>(d_sorted_score.data()),
                                         d_index_in.data(),
                                         d_index_out.data(),
                                         parameters_->batch_size * parameters_->vocab_size,
                                         parameters_->batch_size,
                                         d_offset.data(),
                                         cuda_stream_);

#ifdef DEBUG_GENERATION
  dumper_->Print("d_sorted_score_buffer",
                 reinterpret_cast<T*>(d_sorted_score.data()),
                 parameters_->batch_size,
                 parameters_->vocab_size);
  dumper_->Print("d_index_buffer_in", d_index_in.data(), parameters_->batch_size, parameters_->vocab_size);
  dumper_->Print("d_index_buffer_out", d_index_out.data(), parameters_->batch_size, parameters_->vocab_size);
#endif

  gsl::span<float>& d_sorted_softmaxed_score = sampling_state_->d_sorted_softmaxed_score;
  dispatch_blockwise_softmax_forward<CudaT, float, float, false>(cuda_stream_,
                                                                 d_sorted_softmaxed_score.data(),
                                                                 reinterpret_cast<CudaT*>(d_sorted_score.data()),
                                                                 parameters_->vocab_size,
                                                                 parameters_->vocab_size,
                                                                 parameters_->vocab_size,
                                                                 parameters_->batch_size);

#ifdef DEBUG_GENERATION
  dumper_->Print("d_sorted_softmaxed_score_buffer",
                 d_sorted_softmaxed_score.data(),
                 parameters_->batch_size,
                 parameters_->vocab_size);
#endif

  cuda::LaunchFilterLogitsKernel<CudaT>(d_sorted_softmaxed_score.data(),
                                        d_index_out.data(),
                                        reinterpret_cast<CudaT*>(next_token_scores.data()),
                                        parameters_->top_p,
                                        parameters_->filter_value,
                                        parameters_->batch_size,
                                        parameters_->vocab_size,
                                        cuda_stream_);

#ifdef DEBUG_GENERATION
  dumper_->Print("next_token_scores after filtering logits",
                 reinterpret_cast<T*>(next_token_scores.data()),
                 parameters_->batch_size,
                 parameters_->vocab_size);
#endif

  // TODO(wy): Can we only do softmax at the very beginning and sort the softmaxed scores.
  gsl::span<float>& d_softmaxed_score = sampling_state_->d_softmaxed_score;
  dispatch_blockwise_softmax_forward<CudaT, float, float, false>(cuda_stream_,
                                                                 d_softmaxed_score.data(),
                                                                 reinterpret_cast<CudaT*>(next_token_scores.data()),
                                                                 parameters_->vocab_size,
                                                                 parameters_->vocab_size,
                                                                 parameters_->vocab_size,
                                                                 parameters_->batch_size);

#ifdef DEBUG_GENERATION
  dumper_->Print("d_softmaxed_score_buffer",
                 d_softmaxed_score.data(),
                 parameters_->batch_size,
                 parameters_->vocab_size);
#endif

  // Multinomial sampling
  gsl::span<float>& d_sampled = sampling_state_->d_sampled;
  gsl::span<float>& h_sampled_all = sampling_state_->h_sampled_all;
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(d_sampled.data(),
                                       h_sampled_all.data() + (step - 1) * parameters_->batch_size,
                                       sizeof(float) * parameters_->batch_size,
                                       cudaMemcpyHostToDevice,
                                       cuda_stream_));

#ifdef DEBUG_GENERATION
  dumper_->Print("d_sampled", d_sampled.data(), parameters_->batch_size, 1);
#endif

  gsl::span<int64_t>& d_indices = sampling_state_->d_indices;
  gsl::span<int>& presence_mask = sampling_state_->d_presence_mask;
  cuda::TorchMultinomialKernelLauncher(d_softmaxed_score.data(),
                                       d_sampled.data(),
                                       d_indices.data(),
                                       parameters_->batch_size,
                                       parameters_->vocab_size,
                                       presence_mask.data(),
                                       cuda_stream_);

#ifdef DEBUG_GENERATION
  dumper_->Print("d_indices", d_indices.data(), parameters_->batch_size, 1);
#endif

  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(greedy_state_->next_tokens_cpu.data(),
                                       sampling_state_->d_indices.data(),
                                       greedy_state_->next_tokens_cpu.size_bytes(),
                                       cudaMemcpyDeviceToHost,
                                       cuda_stream_));

  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(sampling_state_->h_softmaxed_score.data(),
                                       sampling_state_->d_softmaxed_score.data(),
                                       sampling_state_->h_softmaxed_score.size_bytes(),
                                       cudaMemcpyDeviceToHost,
                                       cuda_stream_));

  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(cuda_stream_));

  return Status::OK();
}

} // namespace SamplingCudaHelper
} // namespace contrib
} // namespace onnxruntime
