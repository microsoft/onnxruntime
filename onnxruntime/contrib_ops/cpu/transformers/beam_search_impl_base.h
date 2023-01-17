// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "contrib_ops/cpu/transformers/generate_impl_base.h"

namespace onnxruntime {
namespace contrib {

namespace transformers {

template <typename T>
struct BeamSearchState : public IBeamSearchState<T> {
  void Init(AllocatorPtr allocator,
            int batch_size,
            int num_beams,
            int vocab_size,
            int sequence_length,
            int max_length,
            bool output_scores,
            bool use_position) {
    size_t batch_beam_size = SafeInt<size_t>(batch_size) * num_beams;

    size_t next_token_size = SafeInt<size_t>(batch_beam_size) * vocab_size;
    this->next_token_logits = AllocateBuffer<T>(allocator, next_token_logits_buffer_, next_token_size);
    this->next_token_scores = AllocateBuffer<float>(allocator, next_token_scores_buffer_, next_token_size);

    this->next_tokens = AllocateBuffer<int32_t>(allocator, next_tokens_buffer_, SafeInt<size_t>(2) * batch_beam_size);

    this->next_indices = AllocateBuffer<int32_t>(allocator, next_indices_buffer_, SafeInt<size_t>(2) * batch_beam_size);

    this->next_scores = AllocateBuffer<float>(allocator, next_scores_buffer_, SafeInt<size_t>(2) * batch_beam_size);

    constexpr size_t max_parts_of_vocab = 128;
    size_t topk_buffer_size = SafeInt<size_t>(batch_beam_size) * (max_parts_of_vocab + 1) * num_beams * 2 * 2;
    this->topk_buffer = AllocateBuffer<float>(allocator, topk_temp_buffer_, topk_buffer_size);

    if (use_position) {
      this->next_positions = AllocateBuffer<int32_t>(allocator, next_positions_buffer_, batch_beam_size);
    }

    this->beam_scores = AllocateBuffer<float>(allocator, beam_scores_buffer_, batch_beam_size);

    if (output_scores) {
      size_t elements = SafeInt<size_t>(max_length - sequence_length) * batch_size * num_beams * vocab_size;
      this->scores = AllocateBuffer<float>(allocator, scores_buffer_, elements);
      this->remaining_scores = this->scores;
    }
  }

 private:
  BufferUniquePtr next_token_logits_buffer_;
  BufferUniquePtr next_token_scores_buffer_;
  BufferUniquePtr next_tokens_buffer_;
  BufferUniquePtr next_indices_buffer_;
  BufferUniquePtr next_scores_buffer_;
  BufferUniquePtr next_positions_buffer_;
  BufferUniquePtr beam_scores_buffer_;
  BufferUniquePtr scores_buffer_;
  BufferUniquePtr topk_temp_buffer_;
};

struct BeamSearchCpuState : public IBeamSearchCpuState {
  Sequences sequences;

  void Init(AllocatorPtr allocator, size_t batch_beam_size, int max_length, int sequence_length, bool is_cuda) {
    this->sequence_lengths = AllocateBuffer<int32_t>(allocator, sequence_lengths_buffer_, batch_beam_size);

    size_t sequences_bytes = SafeInt<size_t>(2) * batch_beam_size * max_length;
    this->sequences_space = AllocateBuffer<int32_t>(allocator, sequences_space_buffer_, sequences_bytes);
    memset(this->sequences_space.data(), 0, this->sequences_space.size_bytes());

    if (is_cuda) {
      // buffers used by CUDA operator but not by CPU operator.
      this->topk_scores = AllocateBuffer<float>(allocator, topk_scores_buffer_, 2 * batch_beam_size);
      this->topk_tokens = AllocateBuffer<int32_t>(allocator, topk_tokens_buffer_, 2 * batch_beam_size);
      this->topk_indices = AllocateBuffer<int32_t>(allocator, topk_indices_buffer_, 2 * batch_beam_size);
      this->final_beam_scores = AllocateBuffer<float>(allocator, final_beam_scores_buffer_, batch_beam_size);
    }

    this->sequences.Init(this->sequences_space, static_cast<int>(batch_beam_size), sequence_length, max_length);
  }

  // Copy expanded input_ids to sequences[0]
  void SetSequence(gsl::span<const int32_t> input_ids_in_cpu,
                   size_t batch_beam_size,
                   int max_length,
                   int sequence_length) {
    gsl::span<int32_t> sequences_0 = sequences_space;
    for (size_t i = 0; i < batch_beam_size; i++) {
      for (int j = 0; j < sequence_length; j++) {
        const size_t index = SafeInt<gsl::index>(i) * max_length + j;
        sequences_0[index] = input_ids_in_cpu[SafeInt<gsl::index>(i) * sequence_length + j];
      }
    }
  }

  // Copy unexpanded input_ids to sequences[0]
  void SetSequence(gsl::span<const int32_t> input_ids_in_cpu,
                   size_t batch_beam_size,
                   int beam_size,
                   int max_length,
                   int sequence_length) {
    gsl::span<int32_t> sequences_0 = sequences_space;
    for (size_t i = 0; i < batch_beam_size; i++) {
      for (int j = 0; j < sequence_length; j++) {
        const size_t index = SafeInt<gsl::index>(i) * max_length + j;
        sequences_0[index] = input_ids_in_cpu[SafeInt<gsl::index>(i / beam_size) * sequence_length + j];
      }
    }
  }

 private:
  BufferUniquePtr final_beam_scores_buffer_;
  BufferUniquePtr sequence_lengths_buffer_;
  BufferUniquePtr topk_scores_buffer_;
  BufferUniquePtr topk_tokens_buffer_;
  BufferUniquePtr topk_indices_buffer_;
  BufferUniquePtr sequences_space_buffer_;
};

// Base class of beam search implementation that is common for both GPT-2 and T5.
template <typename T>
class BeamSearchBase : public GenerateBase  {
 public:
  BeamSearchBase(OpKernelContextInternal& context,
                 const SessionState& decoder_session_state,
                 concurrency::ThreadPool* thread_pool,
                 Stream* ort_stream,
                 IConsoleDumper* cuda_dumper,
                 BeamSearchParameters& params,
                 const GenerationDeviceHelper::TopkFunc& topk_func,
                 const GenerationDeviceHelper::ProcessLogitsFunc<T>& process_logits_func,
                 const GenerationDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
                 const GenerationDeviceHelper::DeviceCopyFunc<int32_t>& device_copy_int32_func)
      :  GenerateBase(context,
                      decoder_session_state,
                      thread_pool,
                      ort_stream,
                      cuda_dumper,
                      topk_func,
                      device_copy_func),
        parameters_(&params),
        process_logits_func_(process_logits_func),
        device_copy_int32_func_(device_copy_int32_func) {
    parameters_->ParseFromInputs(&context);
  }

  ~BeamSearchBase() override = default;

  // Initialize by validating all the inputs, and allocating the output tensors.
  Status Initialize() override;

  // Validate inputs.
  Status CheckInputs(const OpKernelContextInternal& context) override;

 protected:
  // Process logits and append next tokens to sequences.
  Status GenerateNextToken(const OrtValue& logits,
                           gsl::span<int32_t>& beam_next_tokens,
                           gsl::span<int32_t>& beam_indices,
                           BeamSearchState<T>& beam_state,
                           BeamSearchCpuState& cpu_state,
                           int counter);

  // Calculate scores from logits, then apply filtering and select next token for each beam.
  Status ProcessLogits(const OrtValue& logits,  // logits output of subgraph
                       BeamSearchState<T>& beam_state,
                       BeamSearchCpuState& cpu_state,
                       AllocatorPtr& allocator,
                       int counter);

  BeamSearchParameters* parameters_;

  std::unique_ptr<BeamSearchScorer> beam_scorer_;

  // Device specific functions
  GenerationDeviceHelper::ProcessLogitsFunc<T> process_logits_func_;
  GenerationDeviceHelper::DeviceCopyFunc<int32_t> device_copy_int32_func_;
};

template <typename T>
Status BeamSearchBase<T>::CheckInputs(const OpKernelContextInternal& context) {
  // Input shapes:
  //   input_ids  : (batch_size, sequence_length)
  //   vocab_mask : (vocab_size) or nullptr
  ORT_RETURN_IF_ERROR(this->CheckInputsImpl(parameters_,
                                            context.Input<Tensor>(0),     // input_ids
                                            context.Input<Tensor>(7),     // vocab_mask
                                            context.Input<Tensor>(8),     // prefix_vocab_mask
                                            context.Input<Tensor>(9),    // attention_mask
                                            nullptr));                    // presence_mask

  return Status::OK();
}

template <typename T>
Status BeamSearchBase<T>::Initialize() {
  ORT_RETURN_IF_ERROR(this->context_.GetTempSpaceAllocator(&temp_space_allocator_));

  ORT_RETURN_IF_ERROR(CheckScalarInput("min_length", 1, false));
  ORT_RETURN_IF_ERROR(CheckScalarInput("max_length", 2, true));
  ORT_RETURN_IF_ERROR(CheckScalarInput("num_beams", 3, true));
  ORT_RETURN_IF_ERROR(CheckScalarInput("num_return_sequences", 4, true));
  ORT_RETURN_IF_ERROR(CheckScalarInput("length_penalty", 5, true));

  ORT_RETURN_IF(parameters_->num_return_sequences > parameters_->num_beams,
                "'num_return_sequences' has to be smaller or equal to 'num_beams'.");

  ORT_RETURN_IF_ERROR(CheckInputs(this->context_));

  // This flag will be updated later when the scores output exists.
  parameters_->output_scores = false;

  if (!IsCuda()) {
    // Logits processor is used in CPU only. In CUDA, cuda kernels are used instead.
    // Initialize processors after CheckInputs so that parameters_->vocab_mask is ready.
    logits_processors_.Init(*parameters_);
  }

  return Status::OK();
}

template <typename T>
Status BeamSearchBase<T>::ProcessLogits(
    const OrtValue& logits,
    BeamSearchState<T>& beam_state,
    BeamSearchCpuState& cpu_state,
    AllocatorPtr& allocator,
    int counter) {
  return process_logits_func_(logits, &beam_state, &cpu_state, &(cpu_state.sequences), allocator,
                              thread_pool_, &logits_processors_, beam_scorer_.get(),
                              parameters_, counter, ort_stream_, GetConsoleDumper());
}

template <typename T>
Status BeamSearchBase<T>::GenerateNextToken(
    const OrtValue& logits,
    gsl::span<int32_t>& beam_next_tokens,
    gsl::span<int32_t>& beam_indices,
    BeamSearchState<T>& beam_state,
    BeamSearchCpuState& cpu_state,
    int counter) {
  // Process logits to get next token scores
  ORT_RETURN_IF_ERROR(ProcessLogits(logits, beam_state, cpu_state, temp_space_allocator_, counter));

  gsl::span<float>& beam_scores = beam_scorer_->GetNextScores();
  // It is optional to clone beam_scores. Change it to use same buffer also works for CPU:
  //    beam_state.beam_scores = beam_scores
  // Here we make a copy to reduce the coupling with little cost (the buffer size is small).
  ORT_RETURN_IF_ERROR(device_copy_func_(beam_state.beam_scores,
                                        beam_scores,
                                        ort_stream_,
                                        DeviceCopyDirection::hostToDevice));

  beam_next_tokens = beam_scorer_->GetNextTokens();
  beam_indices = beam_scorer_->GetNextIndices();

#ifdef DEBUG_GENERATION
  cpu_dumper_.Print("beam_scores from scorer", beam_scores.data(), parameters_->batch_size, parameters_->num_beams);
  cpu_dumper_.Print("beam_next_tokens", beam_next_tokens.data(), parameters_->batch_size, parameters_->num_beams);
  cpu_dumper_.Print("beam_indices", beam_indices.data(), parameters_->batch_size, parameters_->num_beams);
#endif

  cpu_state.sequences.AppendNextTokenToSequences(beam_indices, beam_next_tokens);

#ifdef DEBUG_GENERATION
  cpu_state.sequences.PrintSequences(&cpu_dumper_);
#endif
  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
