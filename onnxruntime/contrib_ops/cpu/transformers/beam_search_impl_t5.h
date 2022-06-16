// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/transformers/beam_search_shared.h"  // for DEBUG_BEAM_SEARCH
#include "contrib_ops/cpu/transformers/beam_search_impl_base.h"
#include "contrib_ops/cpu/transformers/subgraph_t5_encoder.h"
#include "contrib_ops/cpu/transformers/subgraph_t5_decoder.h"

namespace onnxruntime {
namespace contrib {

namespace transformers {

// Beam search implementation for T5 model.
template <typename T>
class BeamSearchT5 : public BeamSearchBase<T> {
 public:
  BeamSearchT5(OpKernelContextInternal& context,
               const SessionState& encoder_session_state,
               const SessionState& decoder_session_state,
               T5EncoderSubgraph& encoder_subgraph,
               T5DecoderSubgraph& decoder_subgraph,
               concurrency::ThreadPool* thread_pool,
               void* cuda_stream,
               IConsoleDumper* cuda_dumper,
               BeamSearchParameters& params,
               const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
               const BeamSearchDeviceHelper::TopkFunc& topk_func,
               const BeamSearchDeviceHelper::ProcessLogitsFunc<T>& process_logits_func,
               const BeamSearchDeviceHelper::InitBeamStateFunc<T>& init_beam_state_func,
               const BeamSearchDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
               const BeamSearchDeviceHelper::DeviceCopyFunc<int32_t>& device_copy_int32_func,
               const BeamSearchDeviceHelper::CreateEncoderInputsFunc& create_encoder_inputs_func,
               const BeamSearchDeviceHelper::UpdateDecoderFeedsFunc<T>& update_decoder_feeds_func)
      : BeamSearchBase<T>(context, decoder_session_state, thread_pool,
                          cuda_stream, cuda_dumper, params,
                          topk_func, process_logits_func, device_copy_func, device_copy_int32_func),
        encoder_session_state_(encoder_session_state),
        encoder_subgraph_(encoder_subgraph),
        decoder_subgraph_(decoder_subgraph),
        add_to_feeds_func_(add_to_feeds_func),
        init_beam_state_func_(init_beam_state_func),
        create_encoder_inputs_func_(create_encoder_inputs_func),
        update_decoder_feeds_func_(update_decoder_feeds_func) {
  }

  // Execute beam search in iterations util stopping criteria is reached.
  Status Execute(const FeedsFetchesManager& encoder_feeds_fetches_manager,
                 const FeedsFetchesManager& decoder_feeds_fetches_manager);

 private:
  const SessionState& encoder_session_state_;

  T5EncoderSubgraph& encoder_subgraph_;
  T5DecoderSubgraph& decoder_subgraph_;

  // Device specific functions
  BeamSearchDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  BeamSearchDeviceHelper::InitBeamStateFunc<T> init_beam_state_func_;

  BeamSearchDeviceHelper::CreateEncoderInputsFunc create_encoder_inputs_func_;
  BeamSearchDeviceHelper::UpdateDecoderFeedsFunc<T> update_decoder_feeds_func_;
};

template <typename T>
Status BeamSearchT5<T>::Execute(const FeedsFetchesManager& encoder_feeds_fetches_manager,
                                const FeedsFetchesManager& decoder_feeds_fetches_manager) {
  auto status = Status::OK();

  const BeamSearchParameters* parameters = this->parameters_;
  ORT_ENFORCE(parameters->sequence_length == 1);

  // Allocate output tensors.
  int64_t sequences_dims[] = {parameters->batch_size, parameters->num_return_sequences, parameters->max_length};
  TensorShape sequences_shape(&sequences_dims[0], sizeof(sequences_dims) / sizeof(sequences_dims[0]));
  Tensor* output_sequences = this->context_.Output(0, sequences_shape);

  int64_t sequences_scores_dims[] = {parameters->batch_size, parameters->num_return_sequences};
  constexpr int64_t dims = sizeof(sequences_scores_dims) / sizeof(sequences_scores_dims[0]);
  TensorShape sequences_scores_shape(&sequences_scores_dims[0], dims);
  Tensor* output_sequences_scores = this->context_.Output(1, sequences_scores_shape);

  int64_t scores_dims[] = {
      static_cast<int64_t>(parameters->max_length) - static_cast<int64_t>(parameters->sequence_length),
      parameters->batch_size, parameters->num_beams, parameters->vocab_size};
  TensorShape scores_shape(&scores_dims[0], sizeof(scores_dims) / sizeof(scores_dims[0]));
  Tensor* output_scores = this->context_.Output(2, scores_shape);

  // Update the flag to indicate whether scores exists in output
  this->parameters_->output_scores = (output_scores != nullptr);

  // ------------------------------------
  // Call encoder subgraph.
  // ------------------------------------
  std::vector<OrtValue> encoder_feeds;
  std::vector<OrtValue> encoder_fetches;

  const OrtValue* encoder_input_ids_value = this->context_.GetInputOrtValue(0);
  const Tensor& encoder_input_ids = encoder_input_ids_value->Get<Tensor>();

  BeamSearchCpuState cpu_state;
  cpu_state.Init(this->cpu_allocator_,
                 static_cast<size_t>(parameters->BatchBeamSize()),
                 parameters->max_length,
                 parameters->sequence_length,
                 this->IsCuda());

  IAllocatorUniquePtr<char> buffer;
  OrtValue expanded_decoder_input_ids;  // Tensor in CPU, and it will be used to initialize sequence in cpu_state
  ORT_RETURN_IF_ERROR(this->encoder_subgraph_.CreateInitialFeeds(
      encoder_input_ids,
      this->implicit_inputs_,
      parameters->num_beams,
      parameters->pad_token_id,
      parameters->decoder_start_token_id,
      encoder_feeds,
      this->create_encoder_inputs_func_,
      this->add_to_feeds_func_,
      buffer,
      expanded_decoder_input_ids));

  ORT_RETURN_IF_ERROR(utils::ExecuteSubgraph(this->encoder_session_state_,
                                             encoder_feeds_fetches_manager,
                                             encoder_feeds,
                                             encoder_fetches,
                                             {},
                                             ExecutionMode::ORT_SEQUENTIAL,
                                             this->context_.GetTerminateFlag(),
                                             this->context_.Logger()));

#ifdef DEBUG_BEAM_SEARCH
  const IConsoleDumper* dumper = this->GetConsoleDumper();
  for (size_t i = 0; i < encoder_feeds.size(); i++) {
    dumper->Print("encoder_feeds", static_cast<int>(i), true);
    dumper->Print("", encoder_feeds[i]);
  }

  for (int i = 0; i <= T5EncoderSubgraph::kFirstPresentOutputIndex; i++) {
    dumper->Print("encoder_fetches", i, true);
    dumper->Print("", encoder_fetches[i]);
  }
#endif

  // ------------------------------------
  // Initialize resources
  // ------------------------------------

  // Copy expanded_decoder_input_ids (in CPU) to sequence. It contains decoder_start_token_id for each beam.
  cpu_state.SetSequence(expanded_decoder_input_ids.Get<Tensor>().DataAsSpan<int32_t>(),
                        static_cast<size_t>(parameters->BatchBeamSize()),
                        parameters->max_length,
                        parameters->sequence_length);

  onnxruntime::OrtStlAllocator<HypothesisScore> hypothesis_score_allocator(this->cpu_allocator_);
  onnxruntime::OrtStlAllocator<BeamHypotheses> beam_hyps_allocator(this->cpu_allocator_);
  this->beam_scorer_ = std::make_unique<BeamSearchScorer>(static_cast<size_t>(parameters->batch_size),
                                                          static_cast<size_t>(parameters->num_beams),
                                                          static_cast<size_t>(parameters->max_length),
                                                          parameters->length_penalty,
                                                          parameters->early_stopping,
                                                          static_cast<size_t>(parameters->num_return_sequences),
                                                          parameters->pad_token_id,
                                                          parameters->eos_token_id,
                                                          hypothesis_score_allocator,
                                                          beam_hyps_allocator);
  this->beam_scorer_->Initialize(this->cpu_allocator_, parameters->sequence_length);

  BeamSearchState<T> beam_state;
  constexpr bool use_position = false;
  beam_state.Init(this->temp_space_allocator_,
                  parameters->batch_size,
                  parameters->num_beams,
                  parameters->vocab_size,
                  parameters->sequence_length,
                  parameters->max_length,
                  parameters->output_scores,
                  use_position);

  init_beam_state_func_(&beam_state,
                        cpu_state.sequence_lengths,
                        parameters->batch_size,
                        parameters->num_beams,
                        this->cuda_stream_);

  // ------------------------------------------------------------------------------
  // Generate next token from logits output from encoder, and initialize decoder inputs.
  // ------------------------------------------------------------------------------
  gsl::span<int32_t> beam_next_tokens;
  gsl::span<int32_t> beam_indices;

  int iteration_counter = 0;
  std::vector<OrtValue> decoder_feeds;
  int current_length = parameters->sequence_length;
  if (current_length + 1 < parameters->max_length) {
    ++iteration_counter;
    ORT_RETURN_IF_ERROR(this->GenerateNextToken(encoder_fetches[0],
                                                beam_next_tokens,
                                                beam_indices,
                                                beam_state,
                                                cpu_state,
                                                iteration_counter));
    ++current_length;  // Increase sequence length after a new token is generated.
    ORT_RETURN_IF_ERROR(decoder_subgraph_.CreateInitialFeeds(beam_next_tokens.as_span<const int32_t>(),
                                                             this->implicit_inputs_,
                                                             encoder_feeds,
                                                             encoder_fetches,
                                                             decoder_feeds,
                                                             this->device_copy_int32_func_,
                                                             this->cuda_stream_));
  }

  // TODO(tianleiwu): allocate fetches. use ping-pong buffers for past state.
  std::vector<OrtValue> decoder_fetches;
  while (current_length < parameters->max_length) {
    iteration_counter++;
#ifdef DEBUG_BEAM_SEARCH
    auto cur_len = std::to_string(current_length);
    dumper->Print("***CurrentLength", cur_len, true);

    for (int i = 0; i <= T5DecoderSubgraph::kFirstPastInputIndex; i++) {
      dumper->Print("decoder_feeds", i, true);
      dumper->Print("", decoder_feeds[i]);
    }
#endif

    status = utils::ExecuteSubgraph(this->decoder_session_state_,
                                    decoder_feeds_fetches_manager,
                                    decoder_feeds,
                                    decoder_fetches,
                                    {},
                                    ExecutionMode::ORT_SEQUENTIAL,
                                    this->context_.GetTerminateFlag(),
                                    this->context_.Logger());

    ORT_RETURN_IF_ERROR(status);

#ifdef DEBUG_BEAM_SEARCH
    for (int i = 0; i <= T5DecoderSubgraph::kFirstPresentOutputIndex; i++) {
      dumper->Print("decoder_fetches", i, true);
      dumper->Print("", decoder_fetches[i]);
    }
#endif

    const OrtValue& logits = decoder_fetches[0];
    ORT_RETURN_IF_ERROR(this->GenerateNextToken(logits,
                                                beam_next_tokens,
                                                beam_indices,
                                                beam_state,
                                                cpu_state,
                                                iteration_counter));

    // When all batches are finished, stop earlier to avoid wasting computation.
    if (this->beam_scorer_->IsDone()) {
      break;
    }

    // Increase sequence length after a new token is generated.
    ++current_length;

    // Prepare inputs for next round of subgraph call.
    if (current_length < parameters->max_length) {
      const int num_present_outputs = 2 * parameters->num_layers;  // number of outputs with name like present_*
      ORT_RETURN_IF_ERROR(this->update_decoder_feeds_func_(
          this->temp_space_allocator_,
          this->cuda_stream_,
          decoder_fetches,
          decoder_feeds,
          num_present_outputs,
          beam_next_tokens.as_span<const int32_t>(),
          beam_indices.as_span<const int32_t>(),
          parameters->num_beams,
          this->GetConsoleDumper()));
    }
    decoder_fetches.clear();
  }

  gsl::span<const float> final_beam_scores(beam_state.beam_scores.data(), beam_state.beam_scores.size());
  if (this->IsCuda()) {
    ORT_RETURN_IF_ERROR(this->device_copy_func_(cpu_state.final_beam_scores,
                                                final_beam_scores,
                                                nullptr,
                                                DeviceCopyDirection::deviceToHost));
    final_beam_scores = gsl::make_span<const float>(cpu_state.final_beam_scores.data(),
                                                    cpu_state.final_beam_scores.size());
  }

  this->beam_scorer_->Finalize(&(cpu_state.sequences),
                               final_beam_scores,
                               output_sequences,
                               output_sequences_scores);

  // Output per token scores
  if (output_scores != nullptr) {
    gsl::span<float> target = output_scores->MutableDataAsSpan<float>();
    gsl::span<const float> source = gsl::span<const float>(beam_state.scores.data(), beam_state.scores.size());
    assert(target.length() == source.length());
    ORT_RETURN_IF_ERROR(this->device_copy_func_(target, source, nullptr, DeviceCopyDirection::deviceToDevice));
  }

  return status;
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
