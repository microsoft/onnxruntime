// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace onnxruntime {
namespace contrib {
namespace SamplingCpuHelper {

template <typename T>
void filter_scores(std::vector<size_t>& sorted_indice,
                   gsl::span<T>& next_token_score,
                   const transformers::IGenerationParameters* parameters,
                   size_t index) {
  size_t real_index = sorted_indice[index];
  next_token_score[real_index] = (T)parameters->filter_value;
}

template <typename T>
void cumulate_and_filter_custom(gsl::span<T>& next_token_scores,
                                gsl::span<T>& cumulative_probs,
                                const transformers::IGenerationParameters* parameters,
                                std::vector<size_t>& sorted_indices) {
  for (size_t i = 0; i < static_cast<size_t>(parameters->batch_size); i++) {
    size_t offset = i * parameters->vocab_size;
    if (cumulative_probs[offset] > parameters->top_p) {
      filter_scores(sorted_indices, next_token_scores, parameters, 1 + offset);
    }
    for (size_t j = 1; j < static_cast<size_t>(parameters->vocab_size) - 1; j++) {
      cumulative_probs[j + offset] += cumulative_probs[j + offset - 1];
      if (cumulative_probs[j + offset] > parameters->top_p) {
        filter_scores(sorted_indices, next_token_scores, parameters, j + offset + 1);
      }
    }
  }
}

template <typename T>
void cumulate_and_filter(gsl::span<T>& next_token_scores,
                         gsl::span<T>& cumulative_probs,
                         const transformers::IGenerationParameters* parameters,
                         std::vector<size_t>& sorted_indices) {
  for (size_t i = 0; i < static_cast<size_t>(parameters->batch_size); i++) {
    size_t offset = i * parameters->vocab_size;
    if (cumulative_probs[offset] <= 1 - parameters->top_p) {
      filter_scores(sorted_indices, next_token_scores, parameters, offset);
    }
    for (size_t j = 1; j < static_cast<size_t>(parameters->vocab_size) - static_cast<size_t>(parameters->min_tokens_to_keep); j++) {
      cumulative_probs[j + offset] += cumulative_probs[j + offset - 1];
      if (cumulative_probs[j + offset] <= 1 - parameters->top_p) {
        filter_scores(sorted_indices, next_token_scores, parameters, j + offset);
      }
    }
  }
}

template <typename T>
Status Sample(AllocatorPtr& allocator,
              onnxruntime::concurrency::ThreadPool* thread_pool,
              gsl::span<T>& next_token_scores,
              transformers::ISamplingState<T>* sampling_state,
              transformers::IGreedySearchState<T>* greedy_state,
              const transformers::IGenerationParameters* parameters,
              const transformers::IConsoleDumper* dumper) {
  ORT_UNUSED_PARAMETER(dumper);

  gsl::span<T>& sorted_scores = sampling_state->sorted_scores;
  memcpy(sorted_scores.data(), next_token_scores.data(), next_token_scores.size_bytes());
  std::vector<size_t> sorted_indices(static_cast<size_t>(parameters->batch_size) * static_cast<size_t>(parameters->vocab_size));

  std::function<bool(T, T)> predicator;
  if (parameters->custom_sampling) {
    predicator = std::greater<T>();
  } else {
    predicator = std::less<T>();
  }

  // TODO: This could be optimized with allocated buffer and handwritten sort algorithm
  for (size_t i = 0; i < static_cast<size_t>(parameters->batch_size); i++) {
    auto indices_begin = sorted_indices.begin() + i * parameters->vocab_size;
    auto indices_end = sorted_indices.begin() + (i + 1) * parameters->vocab_size;
    std::iota(indices_begin, indices_end, 0);
    std::sort(indices_begin, indices_end,
              [&next_token_scores, &predicator](size_t i1, size_t i2) {
                return !predicator(next_token_scores[i1], next_token_scores[i2]);
              });

    std::sort(sorted_scores.begin() + i * parameters->vocab_size,
              sorted_scores.begin() + (i + 1) * parameters->vocab_size,
              predicator);
  }

  gsl::span<T>& cumulative_probs = sampling_state->cumulative_probs;

  ORT_RETURN_IF_ERROR(SoftmaxCPU<T>(parameters->batch_size,
                                    parameters->vocab_size,
                                    sorted_scores.data(),
                                    cumulative_probs.data(),
                                    false,
                                    thread_pool));

  if (parameters->custom_sampling) {
    cumulate_and_filter_custom(next_token_scores, cumulative_probs, parameters, sorted_indices);
  } else {
    cumulate_and_filter(next_token_scores, cumulative_probs, parameters, sorted_indices);
  }

  gsl::span<T>& next_token_probs = sampling_state->h_softmaxed_score;
  ORT_RETURN_IF_ERROR(SoftmaxCPU<T>(parameters->batch_size,
                                    parameters->vocab_size,
                                    next_token_scores.data(),
                                    next_token_probs.data(),
                                    false,
                                    thread_pool));

  // torch.multinomial()
  int64_t next_token_probs_dims[] = {static_cast<int64_t>(parameters->batch_size), parameters->vocab_size};
  TensorShape next_token_probs_shape(&next_token_probs_dims[0], 2);
  auto element_type = DataTypeImpl::GetType<T>();
  OrtValue next_token_probs_value;
  Tensor::InitOrtValue(element_type,
                       next_token_probs_shape,
                       next_token_probs.data(),
                       allocator->Info(),
                       next_token_probs_value);
  const Tensor& input = next_token_probs_value.Get<Tensor>();

  std::default_random_engine& generator = sampling_state->generator;

  int64_t sampled_idx_dims[] = {static_cast<int64_t>(parameters->batch_size), 1};
  TensorShape sampled_idx_shape(&sampled_idx_dims[0], 2);

  gsl::span<int64_t>& next_token_idx = greedy_state->next_tokens_cpu;

  OrtValue sampled_idx_ov;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(),
                       sampled_idx_shape,
                       next_token_idx.data(),
                       allocator->Info(),
                       sampled_idx_ov);
  Tensor* sampled_idx = sampled_idx_ov.GetMutable<Tensor>();

  // Copy the allocator because MultinomialComputeShared() uses move(allocator)
  AllocatorPtr allocatortemp = allocator;
  ORT_RETURN_IF_ERROR(MultinomialComputeShared<int64_t>(allocatortemp,
                                                        input,
                                                        parameters->batch_size,
                                                        parameters->vocab_size,
                                                        1,
                                                        generator,
                                                        *sampled_idx));
  // TODO: update presense_mask()
#ifdef DEBUG_GENERATION
    dumper->Print("sampled_idx", *sampled_idx);
#endif

  return Status::OK();
}

} // namespace SamplingCudaHelper
} // namespace contrib
} // namespace onnxruntime
