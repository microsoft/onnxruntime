// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <random>
#include <vector>
#include "contrib_ops/cpu/transformers/generation_shared.h"
#include "contrib_ops/cpu/transformers/generate_impl_base.h"

namespace onnxruntime {
namespace contrib {

namespace transformers {

template <typename T>
struct SamplingState : public ISamplingState<T> {
  void Init(AllocatorPtr allocator,
            AllocatorPtr cpu_allocator,
            int batch_size,
            int vocab_size,
            int max_iter,
            int seed,
            bool is_cuda,
            Stream* stream) {
    int total_count = batch_size * vocab_size;

    this->h_softmaxed_score = AllocateBuffer<float>(cpu_allocator, h_softmaxed_score_buffer_, SafeInt<size_t>(total_count), stream);

    this->generator = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};

    if (is_cuda) {
      this->d_index_in = AllocateBuffer<int>(allocator, d_index_in_buffer_, SafeInt<size_t>(total_count), stream);
      this->d_index_out = AllocateBuffer<int>(allocator, d_index_out_buffer_, SafeInt<size_t>(total_count), stream);
      this->d_offset = AllocateBuffer<int>(allocator, d_offset_buffer_, SafeInt<size_t>(batch_size + 1), stream);
      this->d_sorted_score = AllocateBuffer<T>(allocator, d_sorted_score_buffer_, SafeInt<size_t>(total_count), stream);
      this->d_sorted_softmaxed_score = AllocateBuffer<float>(allocator, d_sorted_softmaxed_score_buffer_, SafeInt<size_t>(total_count), stream);
      this->d_softmaxed_score = AllocateBuffer<float>(allocator, d_softmaxed_score_buffer_, SafeInt<size_t>(total_count), stream);
      this->d_sampled = AllocateBuffer<float>(allocator, d_sampled_buffer_, SafeInt<size_t>(batch_size), stream);
      this->h_sampled_all = AllocateBuffer<float>(cpu_allocator, h_sampled_all_buffer_, SafeInt<size_t>(batch_size * max_iter), stream);
      this->d_indices = AllocateBuffer<int32_t>(allocator, d_indices_buffer_, SafeInt<size_t>(batch_size), stream);
      this->temp_storage_bytes = 0;
      // TODO: Do not allocate this buffer if there's no presence_mask
      this->d_presence_mask = AllocateBuffer<int>(allocator, d_presence_mask_buffer_, SafeInt<size_t>(total_count), stream);

      std::uniform_real_distribution<float> distribution(0.0, 1.0);
      static_cast<void>(distribution(this->generator));
      for (size_t i = 0; i < this->h_sampled_all.size(); ++i) {
        this->h_sampled_all[i] = distribution(this->generator);
      }
    } else {
      // TODO: Some buffer can be reused for CPU
      this->sorted_scores = AllocateBuffer<T>(cpu_allocator, sorted_scores_buffer_, SafeInt<size_t>(total_count), stream);
      this->cumulative_probs = AllocateBuffer<T>(cpu_allocator, cumulative_probs_buffer_, SafeInt<size_t>(total_count), stream);
    }
  }

 private:
  IAllocatorUniquePtr<void> d_index_in_buffer_;
  IAllocatorUniquePtr<void> d_index_out_buffer_;
  IAllocatorUniquePtr<void> d_offset_buffer_;
  IAllocatorUniquePtr<void> d_sorted_score_buffer_;
  IAllocatorUniquePtr<void> d_sorted_softmaxed_score_buffer_;
  IAllocatorUniquePtr<void> d_softmaxed_score_buffer_;
  IAllocatorUniquePtr<void> h_softmaxed_score_buffer_;
  IAllocatorUniquePtr<void> d_sampled_buffer_;
  IAllocatorUniquePtr<void> h_sampled_all_buffer_;
  IAllocatorUniquePtr<void> d_indices_buffer_;
  IAllocatorUniquePtr<void> d_presence_mask_buffer_;
  IAllocatorUniquePtr<void> sorted_scores_buffer_;
  IAllocatorUniquePtr<void> cumulative_probs_buffer_;
};

template <typename T>
struct GreedySearchState : public IGreedySearchState<T> {
  Sequences sequences;

  void Init(AllocatorPtr cpu_allocator,
            AllocatorPtr allocator,
            int batch_size,
            int vocab_size,
            int sequence_length,
            int max_length,
            int num_heads,
            int head_size,
            bool has_decoder_masked_self_attention,
            bool is_cuda,
            Stream* stream) {
    // below buffers are on cpu
    this->sequences_space = AllocateBuffer<int32_t>(cpu_allocator,
                                                    sequences_space_buffer_,
                                                    SafeInt<size_t>(2) * batch_size * max_length, stream);
    memset(this->sequences_space.data(), 0, this->sequences_space.size_bytes());
    this->sequences.Init(this->sequences_space, static_cast<int>(batch_size), sequence_length, max_length);

    this->sequence_lengths = AllocateBuffer<int32_t>(cpu_allocator, sequence_lengths_buffer_, batch_size, stream);
    this->eos_meet = AllocateBuffer<bool>(cpu_allocator, eos_meet_buffer_, batch_size, stream);
    memset(this->eos_meet.data(), 0, this->eos_meet.size_bytes());

    this->next_tokens = AllocateBuffer<int32_t>(cpu_allocator, next_tokens_buffer_, SafeInt<size_t>(batch_size), stream);

    // below buffers are on cpu or cuda
    size_t next_token_size = SafeInt<size_t>(batch_size) * vocab_size;
    this->next_token_scores = AllocateBuffer<T>(allocator, next_token_scores_buffer_, next_token_size, stream);
    this->next_positions = AllocateBuffer<int32_t>(allocator, next_positions_buffer_, batch_size, stream);

    if (is_cuda) {
      AllocateTempBufferForGetGreedySearchTopOne<T>(
          batch_size,
          allocator,
          this->temp_topk_buffer_,
          this->temp_topk_scores_buffer,
          this->temp_topk_tokens_buffer,
          this->topk_scores_buffer,
          this->topk_tokens_buffer,
          stream);

      // If at all we need to, we only need to re-order past state for CUDA as
      //`DecoderMaskedSelfAttention` is only supported on CUDA
      if (has_decoder_masked_self_attention) {
        TensorShape staging_for_past_state_reorder_buffer_shape = {batch_size, num_heads, max_length, head_size};

        Tensor temp(DataTypeImpl::GetType<T>(), staging_for_past_state_reorder_buffer_shape, allocator);

        this->staging_for_past_state_reorder = std::move(temp);
      }
    }
  }

  void SetSequence(gsl::span<const int32_t> input_ids_in_cpu,
                   size_t batch_beam_size,
                   int max_length,
                   int sequence_length) {
    gsl::span<int32_t> sequences_0 = this->sequences_space;
    for (size_t i = 0; i < batch_beam_size; i++) {
      for (int j = 0; j < sequence_length; j++) {
        sequences_0[SafeInt<gsl::index>(i) * max_length + j] =
            static_cast<int32_t>(input_ids_in_cpu[SafeInt<gsl::index>(i) * sequence_length + j]);
      }
    }
  }

 private:
  IAllocatorUniquePtr<void> sequences_space_buffer_;
  IAllocatorUniquePtr<void> sequence_lengths_buffer_;
  IAllocatorUniquePtr<void> next_token_scores_buffer_;
  IAllocatorUniquePtr<void> next_tokens_buffer_;
  IAllocatorUniquePtr<void> next_positions_buffer_;
  IAllocatorUniquePtr<void> eos_meet_buffer_;
  IAllocatorUniquePtr<void> temp_topk_buffer_;
  IAllocatorUniquePtr<void> staging_for_past_state_reorder_buffer_;
};

// Base class of gready search implementation that is common for both GPT-2 and Bart/T5.
template <typename T, typename ParametersT>
class GreedySearchBase : public GenerateBase {
 public:
  GreedySearchBase(OpKernelContextInternal& context,
                   const SessionState& decoder_session_state,
                   concurrency::ThreadPool* thread_pool,
                   Stream* ort_stream,
                   IConsoleDumper* cuda_dumper,
                   ParametersT& params,
                   const GenerationDeviceHelper::TopkFunc& topk_func,
                   const GenerationDeviceHelper::GreedySearchProcessLogitsFunc<T>& process_logits_func,
                   const GenerationDeviceHelper::DeviceCopyFunc<float>& device_copy_func)
      : GenerateBase(context,
                     decoder_session_state,
                     thread_pool,
                     ort_stream,
                     cuda_dumper,
                     topk_func,
                     device_copy_func),
        parameters_(&params),
        process_logits_func_(process_logits_func) {
    parameters_->ParseFromInputs(&context);
  }

  ~GreedySearchBase() override = default;

  // Initialize by validating all the inputs, and allocating the output tensors.
  Status Initialize() override;

  // Validate inputs.
  Status CheckInputs(const OpKernelContextInternal& context) override;

 protected:
  // Process logits and append next tokens to sequences.
  Status GenerateNextToken(const OrtValue& logits,
                           gsl::span<int32_t>& next_tokens,
                           GreedySearchState<T>& greedy_state,
                           ISamplingState<T>& sampling_state,
                           int counter,
                           int eos_token_id);

  // Calculate scores from logits, then apply filtering and select next token for each beam.
  Status ProcessLogits(const OrtValue& logits,  // logits output of subgraph
                       GreedySearchState<T>& greedy_state,
                       ISamplingState<T>& sampling_state,
                       AllocatorPtr& allocator,
                       int counter);

  ParametersT* parameters_;

  // Device specific functions
  GenerationDeviceHelper::GreedySearchProcessLogitsFunc<T> process_logits_func_;
};

template <typename T, typename ParametersT>
Status GreedySearchBase<T, ParametersT>::CheckInputs(const OpKernelContextInternal& context) {
  // Input shapes:
  //   input_ids          : (batch_size, sequence_length)
  //   vocab_mask         : (vocab_size) or nullptr
  //   decoder_input_ids  : (batch_size, initial_decode_sequence_length)
  ORT_RETURN_IF_ERROR(this->CheckInputsImpl(parameters_,
                                            context.Input<Tensor>(0),     // input_ids
                                            context.Input<Tensor>(4),     // vocab_mask
                                            context.Input<Tensor>(5),     // prefix_vocab_mask
                                            context.Input<Tensor>(6),     // attention_mask
                                            context.Input<Tensor>(7),     // presence_mask
                                            context.Input<Tensor>(10)));  // decoder_input_ids

  return Status::OK();
}

template <typename T, typename ParametersT>
Status GreedySearchBase<T, ParametersT>::Initialize() {
  ORT_RETURN_IF_ERROR(this->context_.GetTempSpaceAllocator(&this->temp_space_allocator_));

  ORT_RETURN_IF_ERROR(CheckScalarInput("max_length", 1, true));
  ORT_RETURN_IF_ERROR(CheckScalarInput("min_length", 2, false));

  ORT_RETURN_IF_ERROR(CheckInputs(this->context_));

  // This flag will be updated later when the scores output exists.
  parameters_->output_scores = false;

  if (!this->IsCuda()) {
    // Logits processor is used in CPU only. In CUDA, cuda kernels are used instead.
    // Initialize processors after CheckInputs so that parameters_->vocab_mask is ready.
    this->logits_processors_.Init(*parameters_);
  }

  return Status::OK();
}

template <typename T, typename ParametersT>
Status GreedySearchBase<T, ParametersT>::ProcessLogits(
    const OrtValue& logits,
    GreedySearchState<T>& greedy_state,
    ISamplingState<T>& sampling_state,
    AllocatorPtr& allocator,
    int counter) {
  bool use_sampling = std::is_same<ParametersT, SamplingParameters>::value;
  return process_logits_func_(logits, &greedy_state, &sampling_state, &(greedy_state.sequences), allocator,
                              this->thread_pool_, &this->logits_processors_, parameters_,
                              use_sampling, counter, this->ort_stream_, this->GetConsoleDumper());
}

template <typename T, typename ParametersT>
Status GreedySearchBase<T, ParametersT>::GenerateNextToken(
    const OrtValue& logits,
    gsl::span<int32_t>& next_tokens,
    GreedySearchState<T>& greedy_state,
    ISamplingState<T>& sampling_state,
    int counter,
    int eos_token_id) {
  // Process logits to get next token scores
  ORT_RETURN_IF_ERROR(ProcessLogits(logits, greedy_state, sampling_state, this->temp_space_allocator_, counter));

  next_tokens = greedy_state.next_tokens;
  gsl::span<bool>& eos_meet = greedy_state.eos_meet;
  for (size_t batch_id = 0; batch_id < next_tokens.size(); ++batch_id) {
    if (next_tokens[batch_id] == eos_token_id || eos_meet[batch_id] == true) {
      eos_meet[batch_id] = true;
      next_tokens[batch_id] = parameters_->pad_token_id;
    }
  }

  greedy_state.sequences.AppendNextTokenToSequences(next_tokens);

#ifdef DEBUG_GENERATION
  greedy_state.sequences.PrintSequences(&cpu_dumper_);
#endif

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
