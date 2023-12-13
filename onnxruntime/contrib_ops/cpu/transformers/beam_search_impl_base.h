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
struct BeamSearchState : IBeamSearchState<T> {
  BeamSearchState(const IGenerationParameters& parameters,
                  AllocatorPtr allocator,
                  int has_decoder_masked_attention,
                  bool use_position,
                  Stream* stream) {
    size_t batch_beam_size = SafeInt<size_t>(parameters.batch_size) * parameters.num_beams;

    size_t next_token_size = SafeInt<size_t>(batch_beam_size) * parameters.vocab_size;
    this->next_token_logits = AllocateBuffer<T>(allocator, next_token_logits_buffer_, next_token_size, stream);
    this->next_token_scores = AllocateBuffer<float>(allocator, next_token_scores_buffer_, next_token_size, stream);
    this->next_tokens = AllocateBuffer<int32_t>(allocator, next_tokens_buffer_, SafeInt<size_t>(2) * batch_beam_size, stream);
    this->next_indices = AllocateBuffer<int32_t>(allocator, next_indices_buffer_, SafeInt<size_t>(2) * batch_beam_size, stream);
    this->next_scores = AllocateBuffer<float>(allocator, next_scores_buffer_, SafeInt<size_t>(2) * batch_beam_size, stream);

    constexpr size_t max_parts_of_vocab = 128;
    size_t topk_buffer_size = SafeInt<size_t>(batch_beam_size) * (max_parts_of_vocab + 1) * parameters.num_beams * 2 * 2;
    this->topk_buffer = AllocateBuffer<float>(allocator, topk_temp_buffer_, topk_buffer_size, stream);

    if (allocator->Info().device.Type() == OrtDevice::GPU) {
      size_t sequences_elements = SafeInt<size_t>(2) * batch_beam_size * parameters.max_length;
      this->sequences_device = AllocateBuffer<int32_t>(allocator, sequences_device_buffer_, sequences_elements, stream);
    }

    if (use_position) {
      this->next_positions = AllocateBuffer<int32_t>(allocator, next_positions_buffer_, batch_beam_size, stream);
    }

    this->beam_scores = AllocateBuffer<float>(allocator, beam_scores_buffer_, batch_beam_size, stream);

    if (parameters.output_scores) {
      size_t elements = SafeInt<size_t>(parameters.max_length - parameters.sequence_length) * parameters.batch_size * parameters.num_beams * parameters.vocab_size;
      this->scores = AllocateBuffer<float>(allocator, scores_buffer_, elements, stream);
      this->remaining_scores = this->scores;
    }

    if (has_decoder_masked_attention) {
      // We need a temp staging buffer to do the past 'K' state re-ordering that is needed
      // when using DecoderMaskedSelfAttention
      TensorShape staging_for_past_state_reorder_buffer_shape = {static_cast<int64_t>(batch_beam_size), parameters.num_heads, parameters.max_length, parameters.head_size};

      Tensor temp(DataTypeImpl::GetType<T>(), staging_for_past_state_reorder_buffer_shape, allocator);

      this->staging_for_past_state_reorder = std::move(temp);
    }
  }

  void EnsurePastStateReorderStagingBuffer(AllocatorPtr allocator, int64_t sz) {
    auto current_buffer_size = this->staging_for_past_state_reorder.Shape().Size();
    if (sz > current_buffer_size) {
      TensorShape buffer_shape = {sz};
      this->staging_for_past_state_reorder = Tensor(DataTypeImpl::GetType<T>(), buffer_shape, allocator);
    }
  }

 private:
  IAllocatorUniquePtr<void> next_token_logits_buffer_;
  IAllocatorUniquePtr<void> next_token_scores_buffer_;
  IAllocatorUniquePtr<void> next_tokens_buffer_;
  IAllocatorUniquePtr<void> next_indices_buffer_;
  IAllocatorUniquePtr<void> next_scores_buffer_;
  IAllocatorUniquePtr<void> next_positions_buffer_;
  IAllocatorUniquePtr<void> beam_scores_buffer_;
  IAllocatorUniquePtr<void> scores_buffer_;
  IAllocatorUniquePtr<void> topk_temp_buffer_;
  IAllocatorUniquePtr<void> sequences_device_buffer_;
};

struct BeamSearchCpuState : IBeamSearchCpuState {
  Sequences sequences;

  BeamSearchCpuState(const IGenerationParameters& parameters, AllocatorPtr allocator, bool is_cuda, Stream* stream)
      : parameters_{parameters} {
    sequence_lengths = AllocateBuffer<int32_t>(allocator, sequence_lengths_buffer_, batch_beam_size_, stream);

    size_t sequences_bytes = SafeInt<size_t>(2) * batch_beam_size_ * parameters.max_length;
    sequences_space = AllocateBuffer<int32_t>(allocator, sequences_space_buffer_, sequences_bytes, stream, true /* fill */);
    sequences.Init(sequences_space, batch_beam_size_, parameters.sequence_length, parameters.max_length);

    if (is_cuda) {
      // buffers used by CUDA operator but not by CPU operator.
      topk_scores = AllocateBuffer<float>(allocator, topk_scores_buffer_, 2 * static_cast<size_t>(batch_beam_size_), stream);
      topk_tokens = AllocateBuffer<int32_t>(allocator, topk_tokens_buffer_, 2 * static_cast<size_t>(batch_beam_size_), stream);
      topk_indices = AllocateBuffer<int32_t>(allocator, topk_indices_buffer_, 2 * static_cast<size_t>(batch_beam_size_), stream);
      final_beam_scores = AllocateBuffer<float>(allocator, final_beam_scores_buffer_, batch_beam_size_, stream);

      size_t next_token_size = SafeInt<size_t>(batch_beam_size_) * parameters.vocab_size;
      next_token_scores = AllocateBuffer<float>(allocator, next_token_scores_buffer_, next_token_size, stream);
    }
  }

  // Copy expanded input_ids to sequences_space
  void SetExpandedSequence(gsl::span<const int32_t> input_ids_in_cpu) {
    for (int i = 0; i < batch_beam_size_; i++) {
      for (int j = 0; j < parameters_.sequence_length; j++) {
        const size_t index = SafeInt<gsl::index>(i) * parameters_.max_length + j;
        sequences_space[index] = input_ids_in_cpu[SafeInt<gsl::index>(i) * parameters_.sequence_length + j];
      }
    }
  }

  // Copy unexpanded input_ids to sequences_space (only difference from SetExpandedSequence is i is divided by parameters_.num_beams
  void SetUnexpandedSequence(gsl::span<const int32_t> input_ids_in_cpu) {
    for (int i = 0; i < batch_beam_size_; i++) {
      for (int j = 0; j < parameters_.sequence_length; j++) {
        const size_t index = SafeInt<gsl::index>(i) * parameters_.max_length + j;
        sequences_space[index] = input_ids_in_cpu[SafeInt<gsl::index>(i / parameters_.num_beams) * parameters_.sequence_length + j];
      }
    }
  }

 private:
  const IGenerationParameters& parameters_;
  const int batch_beam_size_{parameters_.batch_size * parameters_.num_beams};

  IAllocatorUniquePtr<void> final_beam_scores_buffer_;
  IAllocatorUniquePtr<void> sequence_lengths_buffer_;
  IAllocatorUniquePtr<void> topk_scores_buffer_;
  IAllocatorUniquePtr<void> topk_tokens_buffer_;
  IAllocatorUniquePtr<void> topk_indices_buffer_;
  IAllocatorUniquePtr<void> sequences_space_buffer_;
  IAllocatorUniquePtr<void> next_token_scores_buffer_;
};

// Base class of beam search implementation that is common for GPT-2, T5, and Whisper.
template <typename T>
class BeamSearchBase : public GenerateBase {
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
      : GenerateBase(context,
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

  std::unique_ptr<IBeamScorer> beam_scorer_;

  // Device specific functions
  GenerationDeviceHelper::ProcessLogitsFunc<T> process_logits_func_;
  GenerationDeviceHelper::DeviceCopyFunc<int32_t> device_copy_int32_func_;
};

template <typename T>
Status BeamSearchBase<T>::CheckInputs(const OpKernelContextInternal& context) {
  // Input shapes:
  //   input_ids          : (batch_size, sequence_length)
  //   vocab_mask         : (vocab_size) or nullptr
  //   decoder_input_ids  : (batch_size, initial_decode_sequence_length)
  ORT_RETURN_IF_ERROR(this->CheckInputsImpl(parameters_,
                                            context.Input<Tensor>(0),     // input_ids
                                            context.Input<Tensor>(7),     // vocab_mask
                                            context.Input<Tensor>(8),     // prefix_vocab_mask
                                            context.Input<Tensor>(9),     // attention_mask
                                            nullptr,                      // presence_mask
                                            context.Input<Tensor>(10)));  // decoder_input_ids

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
  std::cout << "Processing logits?" << std::endl;
  return process_logits_func_(logits, &beam_state, &(cpu_state.sequences), allocator,
                              thread_pool_, &logits_processors_, beam_scorer_.get(),
                              parameters_, counter, ort_stream_, GetConsoleDumper());
}

template <typename T>
Status BeamSearchBase<T>::GenerateNextToken(
    const OrtValue& logits,
    gsl::span<int32_t>& beam_next_tokens,
    BeamSearchState<T>& beam_state,
    BeamSearchCpuState& cpu_state,
    int counter) {
  // Process logits to get next token scores
  std::cout << "Processing Logits!" << std::endl;
  ORT_RETURN_IF_ERROR(ProcessLogits(logits, beam_state, cpu_state, temp_space_allocator_, counter));
  std::cout << "Getting next scores!" << std::endl;

  if (this->IsCuda()) {
    auto beam_scores = beam_scorer_->GetNextScores();
    // It is optional to clone beam_scores. Change it to use same buffer also works:
    //    beam_state.beam_scores = beam_scores
    // Here we make a copy to reduce the coupling with little cost (the buffer size is small).
    ORT_RETURN_IF_ERROR(device_copy_func_(beam_state.beam_scores,
                                          beam_scores,
                                          ort_stream_,
                                          DeviceCopyDirection::deviceToDevice));

    beam_next_tokens = beam_scorer_->GetNextTokens();

  std::cout << "Got next scores" << std::endl;

#ifdef DEBUG_GENERATION
    auto beam_indices = beam_scorer_->GetNextIndicesGPU();
    cuda_dumper_->Print("beam_scores from scorer", beam_state.beam_scores.data(), parameters_->batch_size, parameters_->num_beams);
    cuda_dumper_->Print("beam_next_tokens", beam_next_tokens.data(), parameters_->batch_size, parameters_->num_beams);
    cuda_dumper_->Print("beam_indices", beam_indices.data(), parameters_->batch_size, parameters_->num_beams);
#endif

    // the Cuda beam scorer does the AppendNextTokenSequences, all that's left for Cuda is a this small step
    cpu_state.sequences.AfterDeviceAppendedNextToken();

#ifdef DEBUG_GENERATION
    // CUDA equivalent of cpu_state.sequences.PrintSequences
    auto sequences_buffer = cpu_state.sequences.GetCurrentDeviceSequences();
    for (int i = 0; i < parameters_->batch_size * parameters_->num_beams; i++) {
      gsl::span<const int32_t> sequence = sequences_buffer.subspan(i * parameters_->max_length, cpu_state.sequences.GetSequenceLength());
      cuda_dumper_->Print("sequences", i, false);
      cuda_dumper_->Print(nullptr, sequence.data(), 1, static_cast<int>(sequence.size()));
    }
#endif
  } else {
    auto beam_scores = beam_scorer_->GetNextScores();
    // It is optional to clone beam_scores. Change it to use same buffer also works for CPU:
    //    beam_state.beam_scores = beam_scores
    // Here we make a copy to reduce the coupling with little cost (the buffer size is small).
    ORT_RETURN_IF_ERROR(device_copy_func_(beam_state.beam_scores,
                                          beam_scores,
                                          ort_stream_,
                                          DeviceCopyDirection::hostToDevice));

    beam_next_tokens = beam_scorer_->GetNextTokens();
    auto beam_indices = beam_scorer_->GetNextIndicesCPU();

#ifdef DEBUG_GENERATION
    cpu_dumper_.Print("beam_scores from scorer", beam_scores.data(), parameters_->batch_size, parameters_->num_beams);
    cpu_dumper_.Print("beam_next_tokens", beam_next_tokens.data(), parameters_->batch_size, parameters_->num_beams);
    cpu_dumper_.Print("beam_indices", beam_indices.data(), parameters_->batch_size, parameters_->num_beams);
#endif

    cpu_state.sequences.AppendNextTokenToSequences(beam_indices, beam_next_tokens);

#ifdef DEBUG_GENERATION
    cpu_state.sequences.PrintSequences(&cpu_dumper_);
#endif
  }

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
