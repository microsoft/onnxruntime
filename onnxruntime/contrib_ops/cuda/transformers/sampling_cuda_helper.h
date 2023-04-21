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

using onnxruntime::cuda::dispatch_blockwise_softmax_forward;
using onnxruntime::cuda::ToCudaType;

namespace onnxruntime {
namespace contrib {
namespace SamplingCudaHelper {

template <typename T>
Status Sample(AllocatorPtr& allocator,
              cudaStream_t cuda_stream,
              gsl::span<T>& next_token_scores,
              transformers::ISamplingState<T>* sampling_state,
              transformers::IGreedySearchState<T>* greedy_state,
              const transformers::IGenerationParameters* parameters,
              int step,
              const transformers::IConsoleDumper* dumper) {
  ORT_UNUSED_PARAMETER(dumper);
  typedef typename ToCudaType<T>::MappedType CudaT;

  gsl::span<int>& d_index_in = sampling_state->d_index_in;
  gsl::span<int>& d_offset = sampling_state->d_offset;

  BufferUniquePtr& storage_buffer = sampling_state->storage_buffer;
  size_t& temp_storage_bytes = sampling_state->temp_storage_bytes;

  bool is_descending = parameters->custom_sampling;
  if (step == 1) {
    cuda::GetTempStorageSize<CudaT>(reinterpret_cast<CudaT*>(next_token_scores.data()),
                                    d_index_in.data(),
                                    d_offset.data(),
                                    parameters->batch_size * parameters->vocab_size,
                                    parameters->batch_size,
                                    cuda_stream,
                                    is_descending,
                                    temp_storage_bytes);

    cuda::LaunchSetupParamsKernel(d_index_in.data(),
                                  d_offset.data(),
                                  parameters->batch_size,
                                  parameters->vocab_size,
                                  cuda_stream);

#ifdef DEBUG_GENERATION
    dumper->Print("d_offset_buffer", d_offset.data(), parameters->batch_size + 1, 1);
#endif

    void* temp_storage = allocator->Alloc(sampling_state->temp_storage_bytes);
    BufferUniquePtr temp_storage_buffer(temp_storage, BufferDeleter(allocator));
    storage_buffer = std::move(temp_storage_buffer);
  }

  gsl::span<T>& d_sorted_score = sampling_state->d_sorted_score;
  gsl::span<int>& d_index_out = sampling_state->d_index_out;

#ifdef DEBUG_GENERATION
  dumper->Print("temp_storage_bytes", sampling_state->temp_storage_bytes, true);
#endif

  cuda::LaunchSortPairs<CudaT>(storage_buffer.get(),
                               temp_storage_bytes,
                               reinterpret_cast<CudaT*>(next_token_scores.data()),
                               reinterpret_cast<CudaT*>(d_sorted_score.data()),
                               d_index_in.data(),
                               d_index_out.data(),
                               parameters->batch_size * parameters->vocab_size,
                               parameters->batch_size,
                               d_offset.data(),
                               cuda_stream,
                               is_descending);

#ifdef DEBUG_GENERATION
  dumper->Print("d_sorted_score_buffer",
                reinterpret_cast<T*>(d_sorted_score.data()),
                parameters->batch_size,
                parameters->vocab_size);
  dumper->Print("d_index_buffer_in", d_index_in.data(), parameters->batch_size, parameters->vocab_size);
  dumper->Print("d_index_buffer_out", d_index_out.data(), parameters->batch_size, parameters->vocab_size);
#endif

  gsl::span<float>& d_sorted_softmaxed_score = sampling_state->d_sorted_softmaxed_score;
  ORT_RETURN_IF_ERROR((dispatch_blockwise_softmax_forward<CudaT, float, float, false>(cuda_stream,
                                                                                      d_sorted_softmaxed_score.data(),
                                                                                      reinterpret_cast<CudaT*>(d_sorted_score.data()),
                                                                                      parameters->vocab_size,
                                                                                      parameters->vocab_size,
                                                                                      parameters->vocab_size,
                                                                                      parameters->batch_size)));

#ifdef DEBUG_GENERATION
  dumper->Print("d_sorted_softmaxed_score_buffer",
                d_sorted_softmaxed_score.data(),
                parameters->batch_size,
                parameters->vocab_size);
#endif

  cuda::LaunchFilterLogitsKernel<CudaT>(d_sorted_softmaxed_score.data(),
                                        d_index_out.data(),
                                        reinterpret_cast<CudaT*>(next_token_scores.data()),
                                        parameters->top_p,
                                        parameters->filter_value,
                                        parameters->min_tokens_to_keep,
                                        parameters->batch_size,
                                        parameters->vocab_size,
                                        cuda_stream,
                                        is_descending);

#ifdef DEBUG_GENERATION
  dumper->Print("next_token_scores after filtering logits",
                reinterpret_cast<T*>(next_token_scores.data()),
                parameters->batch_size,
                parameters->vocab_size);
#endif

  gsl::span<float>& d_softmaxed_score = sampling_state->d_softmaxed_score;
  ORT_RETURN_IF_ERROR((dispatch_blockwise_softmax_forward<CudaT, float, float, false>(cuda_stream,
                                                                                      d_softmaxed_score.data(),
                                                                                      reinterpret_cast<CudaT*>(next_token_scores.data()),
                                                                                      parameters->vocab_size,
                                                                                      parameters->vocab_size,
                                                                                      parameters->vocab_size,
                                                                                      parameters->batch_size)));

#ifdef DEBUG_GENERATION
  dumper->Print("d_softmaxed_score_buffer",
                d_softmaxed_score.data(),
                parameters->batch_size,
                parameters->vocab_size);
#endif

  // Multinomial sampling
  gsl::span<float>& d_sampled = sampling_state->d_sampled;
  gsl::span<float>& h_sampled_all = sampling_state->h_sampled_all;
  size_t sample_offset = (static_cast<size_t>(step) - 1) * static_cast<size_t>(parameters->batch_size);
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(d_sampled.data(),
                                       h_sampled_all.data() + sample_offset,
                                       sizeof(float) * parameters->batch_size,
                                       cudaMemcpyHostToDevice,
                                       cuda_stream));

#ifdef DEBUG_GENERATION
  dumper->Print("d_sampled", d_sampled.data(), parameters->batch_size, 1);
#endif

  gsl::span<int32_t>& d_indices = sampling_state->d_indices;
  gsl::span<int>& presence_mask = sampling_state->d_presence_mask;
  cuda::TorchMultinomialKernelLauncher(d_softmaxed_score.data(),
                                       d_sampled.data(),
                                       d_indices.data(),
                                       parameters->batch_size,
                                       parameters->vocab_size,
                                       presence_mask.data(),
                                       cuda_stream);

#ifdef DEBUG_GENERATION
  dumper->Print("d_indices", d_indices.data(), parameters->batch_size, 1);
#endif

  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(greedy_state->next_tokens.data(),
                                       sampling_state->d_indices.data(),
                                       greedy_state->next_tokens.size_bytes(),
                                       cudaMemcpyDeviceToHost,
                                       cuda_stream));

  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(sampling_state->h_softmaxed_score.data(),
                                       sampling_state->d_softmaxed_score.data(),
                                       sampling_state->h_softmaxed_score.size_bytes(),
                                       cudaMemcpyDeviceToHost,
                                       cuda_stream));

  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(cuda_stream));

  return Status::OK();
}

}  // namespace SamplingCudaHelper
}  // namespace contrib
}  // namespace onnxruntime
