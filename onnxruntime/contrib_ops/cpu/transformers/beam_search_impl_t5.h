// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "beam_search_impl_base.h"
#include "subgraph_t5_encoder.h"
#include "subgraph_t5_decoder.h"

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
               const BeamSearchDeviceHelper::CreateEncoderInputsFunc& create_encoder_inputs_func,
               const BeamSearchDeviceHelper::InitDecoderFeedsFunc<T>& init_decoder_feeds_func,
               const BeamSearchDeviceHelper::UpdateDecoderFeedsFunc<T>& update_decoder_feeds_func)
      : BeamSearchBase<T>(context, decoder_session_state, thread_pool, cuda_stream, cuda_dumper, params, topk_func, process_logits_func, device_copy_func),
        encoder_session_state_(encoder_session_state),
        encoder_subgraph_(encoder_subgraph),
        decoder_subgraph_(decoder_subgraph),
        add_to_feeds_func_(add_to_feeds_func),
        init_beam_state_func_(init_beam_state_func),
        create_encoder_inputs_func_(create_encoder_inputs_func),
        init_decoder_feeds_func_(init_decoder_feeds_func),
        update_decoder_feeds_func_(update_decoder_feeds_func) {
  }

  // Execute beam search in iterations util stopping criteria is reached.
  Status Execute(const FeedsFetchesManager& encoder_feeds_fetches_manager,
                 const FeedsFetchesManager& decoder_feeds_fetches_manager);

 private:
  // Update the input for next iteration.
  Status UpdateFeeds(
      const std::vector<OrtValue>& last_outputs,
      std::vector<OrtValue>& next_inputs,
      gsl::span<const int32_t> beam_next_tokens,
      gsl::span<const int32_t> beam_indices,
      int current_length,
      transformers::Sequences& sequences);

  const SessionState& encoder_session_state_;

  T5EncoderSubgraph& encoder_subgraph_;
  T5DecoderSubgraph& decoder_subgraph_;

  // Device specific functions
  BeamSearchDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  BeamSearchDeviceHelper::InitBeamStateFunc<T> init_beam_state_func_;

  BeamSearchDeviceHelper::CreateEncoderInputsFunc create_encoder_inputs_func_;
  BeamSearchDeviceHelper::InitDecoderFeedsFunc<T> init_decoder_feeds_func_;
  BeamSearchDeviceHelper::UpdateDecoderFeedsFunc<T> update_decoder_feeds_func_;
};

template <typename T>
Status BeamSearchT5<T>::UpdateFeeds(
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    int current_length,
    Sequences& sequences) {
  return update_feeds_func_(temp_space_allocator_, cuda_stream_, last_outputs, next_inputs,
                            beam_next_tokens, beam_indices, parameters_->num_beams, current_length, sequences, GetConsoleDumper());
}

template <typename T>
Status BeamSearchT5<T>::Execute(const FeedsFetchesManager& encoder_feeds_fetches_manager,
                                const FeedsFetchesManager& decoder_feeds_fetches_manager) {
  auto status = Status::OK();

  // Allocate output tensors.
  int64_t sequences_dims[] = {parameters_->batch_size, parameters_->num_return_sequences, parameters_->max_length};
  TensorShape sequences_shape(&sequences_dims[0], sizeof(sequences_dims) / sizeof(sequences_dims[0]));
  Tensor* output_sequences = context_.Output(0, sequences_shape);

  int64_t sequences_scores_dims[] = {parameters_->batch_size, parameters_->num_return_sequences};
  TensorShape sequences_scores_shape(&sequences_scores_dims[0], sizeof(sequences_scores_dims) / sizeof(sequences_scores_dims[0]));
  Tensor* output_sequences_scores = context_.Output(1, sequences_scores_shape);

  int64_t scores_dims[] = {
      static_cast<int64_t>(parameters_->max_length) - static_cast<int64_t>(parameters_->sequence_length),
      parameters_->batch_size, parameters_->num_beams, parameters_->vocab_size};
  TensorShape scores_shape(&scores_dims[0], sizeof(scores_dims) / sizeof(scores_dims[0]));
  Tensor* output_scores = context_.Output(2, scores_shape);

  // Update the flag to indicate whether scores exists in output
  parameters_->output_scores = (output_scores != nullptr);


  // ------------------------------------
  // Call encoder subgraph.
  // ------------------------------------
  std::vector<OrtValue> encoder_feeds;
  std::vector<OrtValue> encoder_fetches;

  const OrtValue* encoder_input_ids_value = context_.GetInputOrtValue(0);
  const Tensor& encoder_input_ids = encoder_input_ids_value->Get<Tensor>();

  BeamSearchCpuState cpu_state;
  cpu_state.Init(cpu_allocator_, static_cast<size_t>(parameters_->BatchBeamSize()), parameters_->max_length, parameters_->sequence_length, IsCuda());

  IAllocatorUniquePtr<char> buffer;
  ORT_RETURN_IF_ERROR(encoder_subgraph_.CreateInitialFeeds(
      encoder_input_ids,
      implicit_inputs_,
      parameters_->num_beams,
      parameters_->pad_token_id,
      parameters_->decoder_start_token_id,
      cpu_state.sequence_lengths,
      encoder_feeds,
      create_encoder_inputs_func_,
      add_to_feeds_func_,
      buffer));

  ORT_RETURN_IF_ERROR(utils::ExecuteSubgraph(encoder_session_state_, encoder_feeds_fetches_manager, encoder_feeds, encoder_fetches, {},
                                             ExecutionMode::ORT_SEQUENTIAL, context_.GetTerminateFlag(), context_.Logger()));


  // ------------------------------------
  // Initialize resources
  // ------------------------------------
  onnxruntime::OrtStlAllocator<HypothesisScore> hypothesis_score_allocator(cpu_allocator_);
  onnxruntime::OrtStlAllocator<BeamHypotheses> beam_hyps_allocator(cpu_allocator_);
  beam_scorer_ = std::make_unique<BeamSearchScorer>(static_cast<size_t>(parameters_->batch_size),
                                                    static_cast<size_t>(parameters_->num_beams),
                                                    static_cast<size_t>(parameters_->max_length),
                                                    parameters_->length_penalty,
                                                    parameters_->early_stopping,
                                                    static_cast<size_t>(parameters_->num_return_sequences),
                                                    parameters_->pad_token_id,
                                                    parameters_->eos_token_id,
                                                    hypothesis_score_allocator,
                                                    beam_hyps_allocator);
  beam_scorer_->Initialize(cpu_allocator_, parameters_->sequence_length);

  BeamSearchState<T> beam_state;
  beam_state.Init(temp_space_allocator_,
                  parameters_->batch_size,
                  parameters_->num_beams,
                  parameters_->vocab_size,
                  parameters_->sequence_length,
                  parameters_->max_length,
                  parameters_->output_scores);

  // TODO: decoder input IDs is int64.
  gsl::span<const int64_t> decoder_input_ids = decoder_feeds[0].Get<Tensor>().DataAsSpan<int64_t>();
  init_beam_state_func_(&beam_state,
                        &cpu_state,
                        cpu_state.sequence_lengths,
                        parameters_->batch_size,
                        parameters_->num_beams,
                        decoder_input_ids,
                        // parameters_->decoder_start_token_id,
                        parameters_->sequence_length,
                        parameters_->max_length,
                        cuda_stream_);
  cpu_state.SetSequence(static_cast<size_t>(parameters_->BatchBeamSize()), parameters_->max_length, parameters_->sequence_length)

  gsl::span<int32_t> beam_next_tokens;
  gsl::span<int32_t> beam_indices;
  int iteration_counter = 1;
  ORT_RETURN_IF_ERROR(GenerateNextToken(encoder_fetches[0], beam_next_tokens, beam_indices, beam_state, cpu_state, iteration_counter));

  std::vector<OrtValue> decoder_feeds;
  ORT_RETURN_IF_ERROR(decoder_subgraph_.CreateInitialFeeds(beam_next_tokens.as_span<const int32_t>(),
                                                           implicit_inputs_,
                                                           encoder_feeds,
                                                           encoder_fetches,
                                                           decoder_feeds));


  // TODO: allocate fetches. use ping-pong buffers for past state.
  std::vector<OrtValue> decoder_fetches;

  int current_length = parameters_->sequence_length;
  while (current_length < parameters_->max_length) {
    iteration_counter++;
#ifdef DEBUG_BEAM_SEARCH
    auto cur_len = std::to_string(current_length);
    dumper->Print("***CurrentLength", cur_len, true);
#endif

    status = utils::ExecuteSubgraph(decoder_session_state_, decoder_feeds_fetches_manager, decoder_feeds, decoder_fetches, {},
                                    ExecutionMode::ORT_SEQUENTIAL, context_.GetTerminateFlag(), context_.Logger());

    ORT_RETURN_IF_ERROR(status);

    const OrtValue& logits = decoder_fetches[0];
    ORT_RETURN_IF_ERROR(GenerateNextToken(logits, beam_next_tokens, beam_indices, beam_state, cpu_state, iteration_counter));

    // When all batches are finished, stop earlier to avoid wasting computation.
    if (beam_scorer_->IsDone()) {
      break;
    }

    // Increase sequence length after a new token is generated.
    ++current_length;

    // Prepare inputs for next round of subgraph call.
    if (current_length < parameters_->max_length) {
      ORT_RETURN_IF_ERROR(update_decoder_feeds_func_(
          temp_space_allocator_,
          cuda_stream_,
          decoder_fetches,
          decoder_feeds,
          current_length,
          beam_next_tokens.as_span<const int32_t>(),
          beam_indices.as_span<const int32_t>(),
          parameters_->num_beams,
          GetConsoleDumper()));
    }
    decoder_fetches.clear();
  }

  gsl::span<const float> final_beam_scores(beam_state.beam_scores.data(), beam_state.beam_scores.size());
  if (IsCuda()) {
    ORT_RETURN_IF_ERROR(device_copy_func_(cpu_state.final_beam_scores, final_beam_scores, nullptr, DeviceCopyDirection::deviceToHost));
    final_beam_scores = gsl::make_span<const float>(cpu_state.final_beam_scores.data(), cpu_state.final_beam_scores.size());
  }

  beam_scorer_->Finalize(&(cpu_state.sequences),
                         final_beam_scores,
                         output_sequences,
                         output_sequences_scores);

  // Output per token scores
  if (output_scores != nullptr) {
    gsl::span<float> target = output_scores->MutableDataAsSpan<float>();
    gsl::span<const float> source = gsl::span<const float>(beam_state.scores.data(), beam_state.scores.size());
    assert(target.length() == source.length());
    ORT_RETURN_IF_ERROR(device_copy_func_(target, source, nullptr, DeviceCopyDirection::deviceToDevice));
  }

  return status;
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
