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
                    const transformers::IGenerationParameters* parameters):
      allocator_(allocator),
      thread_pool_(thread_pool),
      sampling_state_(sampling_state),
      greedy_state_(greedy_state),
      parameters_(parameters) {}

    Status Sample(gsl::span<T>& next_token_scores);

  private:
    void filter_scores(std::vector<size_t>& sorted_indice, gsl::span<T>& next_token_score, size_t index);

    AllocatorPtr& allocator_;
    onnxruntime::concurrency::ThreadPool* thread_pool_;
    transformers::ISamplingState<T>* sampling_state_;
    transformers::IGreedySearchState<T>* greedy_state_;
    const transformers::IGenerationParameters* parameters_;
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

  for (int i = 0; i < parameters_->batch_size; i++) {
    gsl::span<T> next_token_score = next_token_scores.subspan(i * parameters_->vocab_size,
                                                              parameters_->vocab_size);

    // Copy the vector
    std::vector<T> sorted_score(next_token_score.begin(), next_token_score.end());

    // Decending sort
    std::vector<size_t> sorted_indice(parameters_->vocab_size);
    std::iota(sorted_indice.begin(), sorted_indice.end(), 0);
    std::sort(sorted_indice.begin(),
              sorted_indice.end(),
              [&sorted_score](size_t i1, size_t i2) {
                return sorted_score[i1] > sorted_score[i2];
              });

    std::sort(sorted_score.begin(), sorted_score.end(), std::greater<T>());
    std::vector<T> cumulative_prob(parameters_->vocab_size);

    // TODO: batch
    ORT_RETURN_IF_ERROR(SoftmaxCPU<T>(1,
                                      parameters_->vocab_size,
                                      sorted_score.data(),
                                      cumulative_prob.data(),
                                      false,
                                      thread_pool_));

    if (cumulative_prob[0] > parameters_->top_p) {
      filter_scores(sorted_indice, next_token_score, 1);
    }
    for (size_t i = 1; i < static_cast<size_t>(parameters_->vocab_size) - 1; i++) {
      cumulative_prob[i] += cumulative_prob[i - 1];
      if (cumulative_prob[i] > parameters_->top_p) {
        filter_scores(sorted_indice, next_token_score, i + 1);
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
    dumper->Print("sampled_idx", *sampled_idx);
#endif

  return Status::OK();
}

} // namespace SamplingCudaHelper
} // namespace contrib
} // namespace onnxruntime
