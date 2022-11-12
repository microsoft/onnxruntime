// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include "contrib_ops/cpu/transformers/generation_shared.h"
#include "contrib_ops/cpu/transformers/generate_impl_base.h"

namespace onnxruntime {
namespace contrib {

namespace transformers {

template <typename T>
struct GreedySearchState : public IGreedySearchState<T> {
  Sequences sequences;

  void Init(AllocatorPtr cpu_allocator,
            AllocatorPtr allocator,
            int batch_size,
            int vocab_size,
            int sequence_length,
            int max_length,
            bool /*is_cuda*/) {
    // below buffers are on cpu
    this->sequences_space = AllocateBuffer<int32_t>(cpu_allocator,
                                                    sequences_space_buffer_,
                                                    SafeInt<size_t>(2) * batch_size * max_length);
    memset(this->sequences_space.data(), 0, this->sequences_space.size_bytes());
    this->sequences.Init(this->sequences_space, static_cast<int>(batch_size), sequence_length, max_length);

    this->sequence_lengths = AllocateBuffer<int32_t>(cpu_allocator, sequence_lengths_buffer_, batch_size);
    this->eos_meet = AllocateBuffer<bool>(cpu_allocator, eos_meet_buffer_, batch_size);
    memset(this->eos_meet.data(), 0, this->eos_meet.size_bytes());

    this->next_tokens_cpu = AllocateBuffer<int64_t>(cpu_allocator,
                                                    next_tokens_cpu_buffer_,
                                                    SafeInt<size_t>(batch_size));
    this->next_tokens = AllocateBuffer<int32_t>(cpu_allocator, next_tokens_buffer_, SafeInt<size_t>(batch_size));

    // below buffers are on cpu or cuda
    size_t next_token_size = SafeInt<size_t>(batch_size) * vocab_size;
    this->next_token_scores = AllocateBuffer<T>(allocator, next_token_scores_buffer_, next_token_size);
    this->next_positions = AllocateBuffer<int32_t>(allocator, next_positions_buffer_, batch_size);
  }

  void SetSequence(gsl::span<const int32_t> input_ids_in_cpu,
                   size_t batch_beam_size,
                   int max_length,
                   int sequence_length) {
    gsl::span<int32_t> sequences_0 = this->sequences_space;
    for (size_t i = 0; i < batch_beam_size; i++) {
      for (int j = 0; j < sequence_length; j++) {
        sequences_0[SafeInt<gsl::index>(i) * max_length + j] = \
        static_cast<int32_t>(input_ids_in_cpu[SafeInt<gsl::index>(i) * sequence_length + j]);
      }
    }
  }

 private:
  BufferUniquePtr sequences_space_buffer_;
  BufferUniquePtr sequence_lengths_buffer_;
  BufferUniquePtr next_token_scores_buffer_;
  BufferUniquePtr next_tokens_buffer_;
  BufferUniquePtr next_tokens_cpu_buffer_;
  BufferUniquePtr next_positions_buffer_;
  BufferUniquePtr eos_meet_buffer_;
};

// Base class of gready search implementation that is common for both GPT-2 and Bart/T5.
template <typename T>
class GreedySearchBase : public GenerateBase {
 public:
  GreedySearchBase(OpKernelContextInternal& context,
                   const SessionState& decoder_session_state,
                   concurrency::ThreadPool* thread_pool,
                   void* cuda_stream,
                   IConsoleDumper* cuda_dumper,
                   GreedySearchParameters& params,
                   const GenerationDeviceHelper::TopkFunc& topk_func,
                   const GenerationDeviceHelper::GreedySearchProcessLogitsFunc<T>& process_logits_func,
                   const GenerationDeviceHelper::DeviceCopyFunc<float>& device_copy_func)
    : GenerateBase(context,
                   decoder_session_state,
                   thread_pool,
                   cuda_stream,
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
                           int counter,
                           int eos_token_id);

  // Calculate scores from logits, then apply filtering and select next token for each beam.
  Status ProcessLogits(const OrtValue& logits,  // logits output of subgraph
                       GreedySearchState<T>& greedy_state,
                       AllocatorPtr& allocator,
                       int counter);

  GreedySearchParameters* parameters_;

  // Device specific functions
  GenerationDeviceHelper::GreedySearchProcessLogitsFunc<T> process_logits_func_;
};

template <typename T>
Status GreedySearchBase<T>::CheckInputs(const OpKernelContextInternal& context) {
  // Input shapes:
  //   input_ids  : (batch_size, sequence_length)
  //   vocab_mask : (vocab_size) or nullptr
  ORT_RETURN_IF_ERROR(this->CheckInputsImpl(parameters_,
                                            context.Input<Tensor>(0),   // input_ids
                                            context.Input<Tensor>(4),   // vocab_mask
                                            context.Input<Tensor>(5),   // prefix_vocab_mask
                                            nullptr));                  // attention_mask

  return Status::OK();
}

template <typename T>
Status GreedySearchBase<T>::Initialize() {
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

template <typename T>
Status GreedySearchBase<T>::ProcessLogits(
    const OrtValue& logits,
    GreedySearchState<T>& greedy_state,
    AllocatorPtr& allocator,
    int counter) {
  return process_logits_func_(logits, &greedy_state, &(greedy_state.sequences), allocator,
                              this->thread_pool_, &this->logits_processors_,
                              parameters_, counter, this->cuda_stream_, this->GetConsoleDumper());
}

template <typename T>
Status GreedySearchBase<T>::GenerateNextToken(
    const OrtValue& logits,
    gsl::span<int32_t>& next_tokens,
    GreedySearchState<T>& greedy_state,
    int counter,
    int eos_token_id) {
  // Process logits to get next token scores
  ORT_RETURN_IF_ERROR(ProcessLogits(logits, greedy_state, this->temp_space_allocator_, counter));

  next_tokens = greedy_state.next_tokens;
  for (size_t i = 0; i < next_tokens.size(); i++) {
    next_tokens[i] = gsl::narrow_cast<int32_t>(greedy_state.next_tokens_cpu[i]);
  }

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
