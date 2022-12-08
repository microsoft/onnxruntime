// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace onnxruntime {
namespace contrib {
namespace SamplingCpuHelper {

template <typename T>
class TopPSamplingCpu{
  public:
    TopPSamplingCpu(AllocatorPtr& allocator,
                    onnxruntime::concurrency::ThreadPool* thread_pool,
                    transformers::ISamplingState<T>* sampling_state,
                    transformers::IGreedySearchState<T>* greedy_state,
                    const transformers::IGenerationParameters* parameters,
                    const transformers::IConsoleDumper* dumper):
      allocator_(allocator),
      thread_pool_(thread_pool),
      sampling_state_(sampling_state),
      greedy_state_(greedy_state),
      parameters_(parameters),
      dumper_(dumper) {}

    Status Sample(gsl::span<T>& next_token_scores);

  private:
    void filter_scores(std::vector<size_t>& sorted_indice, gsl::span<T>& next_token_score, size_t index);

    AllocatorPtr& allocator_;
    onnxruntime::concurrency::ThreadPool* thread_pool_;
    transformers::ISamplingState<T>* sampling_state_;
    transformers::IGreedySearchState<T>* greedy_state_;
    const transformers::IGenerationParameters* parameters_;
    const transformers::IConsoleDumper* dumper_;
};

template <typename T>
void TopPSamplingCpu<T>::filter_scores(std::vector<size_t>& sorted_indice,
                                       gsl::span<T>& next_token_score,
                                       size_t index) {
  size_t real_index = sorted_indice[index];
  next_token_score[real_index] = parameters_->filter_value;
}

template <typename T>
Status TopPSamplingCpu<T>::Sample(gsl::span<T>& next_token_scores) {
  if (parameters_->top_p == 0.0f) {
      ORT_THROW("top_p shall be greater than 0");
  }

  std::vector<T>& sorted_scores = sampling_state_->sorted_scores;
  sorted_scores.assign(next_token_scores.begin(), next_token_scores.end());
  // Decending sort
  std::vector<size_t>& sorted_indices = sampling_state_->sorted_indices;

  for (size_t i = 0; i < static_cast<size_t>(parameters_->batch_size); i++) {
    auto indices_begin = sorted_indices.begin() + i * parameters_->vocab_size;
    auto indices_end = sorted_indices.begin() + (i + 1) * parameters_->vocab_size;
    std::iota(indices_begin, indices_end, 0);
    std::sort(indices_begin, indices_end,
              [&next_token_scores](size_t i1, size_t i2) {
                return next_token_scores[i1] > next_token_scores[i2];
              });

    std::sort(sorted_scores.begin() + i * parameters_->vocab_size,
              sorted_scores.end() + (i + 1) * parameters_->vocab_size,
              std::greater<T>());
  }

  std::vector<T>& cumulative_probs = sampling_state_->cumulative_probs;

  ORT_RETURN_IF_ERROR(SoftmaxCPU<T>(parameters_->batch_size,
                                    parameters_->vocab_size,
                                    sorted_scores.data(),
                                    cumulative_probs.data(),
                                    false,
                                    thread_pool_));

  for (size_t i = 0; i < static_cast<size_t>(parameters_->batch_size); i++) {
    size_t offset = i * parameters_->vocab_size;
    if (cumulative_probs[offset] > parameters_->top_p) {
      filter_scores(sorted_indices, next_token_scores, 1 + offset);
    }
    for (size_t j = 1; j < static_cast<size_t>(parameters_->vocab_size) - 1; j++) {
      cumulative_probs[j + offset] += cumulative_probs[j + offset - 1];
      if (cumulative_probs[j + offset] > parameters_->top_p) {
        filter_scores(sorted_indices, next_token_scores, j + offset + 1);
      }
    }
  }

  // TODO(wy): This softmax may not be necessary.
  gsl::span<T>& next_token_probs = sampling_state_->h_softmaxed_score;
  ORT_RETURN_IF_ERROR(SoftmaxCPU<T>(parameters_->batch_size,
                                    parameters_->vocab_size,
                                    next_token_scores.data(),
                                    next_token_probs.data(),
                                    false,
                                    thread_pool_));

  // torch.multinomial()
  int64_t next_token_probs_dims[] = {static_cast<int64_t>(parameters_->batch_size), parameters_->vocab_size};
  TensorShape next_token_probs_shape(&next_token_probs_dims[0], 2);
  auto element_type = DataTypeImpl::GetType<T>();
  OrtValue next_token_probs_value;
  Tensor::InitOrtValue(element_type,
                       next_token_probs_shape,
                       next_token_probs.data(),
                       allocator_->Info(),
                       next_token_probs_value);
  const Tensor& input = next_token_probs_value.Get<Tensor>();

  std::default_random_engine& generator = sampling_state_->generator;

  int64_t sampled_idx_dims[] = {static_cast<int64_t>(parameters_->batch_size), 1};
  TensorShape sampled_idx_shape(&sampled_idx_dims[0], 2);

  gsl::span<int64_t>& next_token_idx = greedy_state_->next_tokens_cpu;

  OrtValue sampled_idx_ov;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(),
                       sampled_idx_shape,
                       next_token_idx.data(),
                       allocator_->Info(),
                       sampled_idx_ov);
  Tensor* sampled_idx = sampled_idx_ov.GetMutable<Tensor>();

  // Copy the allocator because MultinomialComputeShared() uses move(allocator)
  AllocatorPtr allocator_temp = allocator_;
  ORT_RETURN_IF_ERROR(MultinomialComputeShared<int64_t>(allocator_temp,
                                                        input,
                                                        parameters_->batch_size,
                                                        parameters_->vocab_size,
                                                        1,
                                                        generator,
                                                        *sampled_idx));
  // TODO: update presense_mask()

#ifdef DEBUG_GENERATION
    dumper_->Print("sampled_idx", *sampled_idx);
#endif

  return Status::OK();
}

} // namespace SamplingCudaHelper
} // namespace contrib
} // namespace onnxruntime
