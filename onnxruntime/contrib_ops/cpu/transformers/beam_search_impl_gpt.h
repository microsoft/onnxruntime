// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/transformers/beam_search_impl_base.h"

namespace onnxruntime {
namespace contrib {

namespace transformers {

// Beam search implementation for GPT-2 model.
template <typename T>
class BeamSearchGpt : public BeamSearchBase<T> {
 public:
  BeamSearchGpt(OpKernelContextInternal& context,
                const SessionState& decoder_session_state,
                GptSubgraph& gpt_subgraph,
                concurrency::ThreadPool* thread_pool,
                void* cuda_stream,
                IConsoleDumper* cuda_dumper,
                BeamSearchParameters& params,
                const BeamSearchDeviceHelper::CreateGptInputsFunc& create_inputs_func,
                const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
                const BeamSearchDeviceHelper::TopkFunc& topk_func,
                const BeamSearchDeviceHelper::ProcessLogitsFunc<T>& process_logits_func,
                const BeamSearchDeviceHelper::InitBeamStateFunc<T>& init_beam_state_func,
                const BeamSearchDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
                const BeamSearchDeviceHelper::DeviceCopyFunc<int32_t>& device_copy_int32_func,
                const BeamSearchDeviceHelper::UpdateGptFeedsFunc<T>& update_feeds_func)
      : BeamSearchBase<T>(context, decoder_session_state, thread_pool,
                          cuda_stream, cuda_dumper, params,
                          topk_func, process_logits_func, device_copy_func, device_copy_int32_func),
        gpt_subgraph_(gpt_subgraph),
        create_inputs_func_(create_inputs_func),
        add_to_feeds_func_(add_to_feeds_func),
        init_beam_state_func_(init_beam_state_func),
        update_feeds_func_(update_feeds_func) {
  }

  // Execute beam search in iterations util stopping criteria is reached.
  // In each iteration, GPT subgraph is called, and next token for each sequence is generated.
  Status Execute(const FeedsFetchesManager& feeds_fetches_manager);

 private:
  // Prepare the inputs for first inference of subgraph
  Status CreateInitialFeeds(gsl::span<int32_t>& sequence_lengths,
                            OrtValue& expanded_input_ids,
                            std::vector<OrtValue>& feeds,
                            IAllocatorUniquePtr<char>& buffer);

  // Update the input for next iteration.
  Status UpdateFeeds(
      const std::vector<OrtValue>& last_outputs,
      std::vector<OrtValue>& next_inputs,
      int current_length,
      OrtValue& position_ids,
      gsl::span<const int32_t> beam_next_tokens,
      gsl::span<const int32_t> beam_indices);

  GptSubgraph& gpt_subgraph_;

  // Device specific functions
  BeamSearchDeviceHelper::CreateGptInputsFunc create_inputs_func_;
  BeamSearchDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  BeamSearchDeviceHelper::InitBeamStateFunc<T> init_beam_state_func_;
  BeamSearchDeviceHelper::UpdateGptFeedsFunc<T> update_feeds_func_;
};

template <typename T>
Status BeamSearchGpt<T>::CreateInitialFeeds(gsl::span<int32_t>& sequence_lengths,
                                            OrtValue& expanded_input_ids,
                                            std::vector<OrtValue>& feeds,
                                            IAllocatorUniquePtr<char>& buffer) {
  const OrtValue* input_ids_value = this->context_.GetInputOrtValue(0);
  const Tensor& input_ids = input_ids_value->Get<Tensor>();
  return gpt_subgraph_.CreateInitialFeeds(input_ids,
                                          this->implicit_inputs_,
                                          this->parameters_->num_beams,
                                          this->parameters_->pad_token_id,
                                          sequence_lengths,
                                          expanded_input_ids,
                                          feeds,
                                          this->create_inputs_func_,
                                          this->add_to_feeds_func_,
                                          buffer);
}

template <typename T>
Status BeamSearchGpt<T>::UpdateFeeds(
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices) {
  return update_feeds_func_(this->temp_space_allocator_,
                            this->cuda_stream_,
                            last_outputs,
                            next_inputs,
                            current_length,
                            position_ids,
                            beam_next_tokens,
                            beam_indices,
                            this->parameters_->num_beams,
                            this->GetConsoleDumper());
}

template <typename T>
Status BeamSearchGpt<T>::Execute(const FeedsFetchesManager& feeds_fetches_manager) {
  auto status = Status::OK();
  const BeamSearchParameters* parameters = this->parameters_;
  int64_t sequences_dims[] = {parameters->batch_size, parameters->num_return_sequences, parameters->max_length};
  TensorShape sequences_shape(&sequences_dims[0], sizeof(sequences_dims) / sizeof(sequences_dims[0]));
  Tensor* output_sequences = this->context_.Output(0, sequences_shape);

  int64_t sequences_scores_dims[] = {parameters->batch_size, parameters->num_return_sequences};
  TensorShape sequences_scores_shape(&sequences_scores_dims[0], 2);
  Tensor* output_sequences_scores = this->context_.Output(1, sequences_scores_shape);

  int64_t scores_dims[] = {
      static_cast<int64_t>(parameters->max_length) - static_cast<int64_t>(parameters->sequence_length),
      parameters->batch_size, parameters->num_beams, parameters->vocab_size};
  TensorShape scores_shape(&scores_dims[0], sizeof(scores_dims) / sizeof(scores_dims[0]));
  Tensor* output_scores = this->context_.Output(2, scores_shape);

  // Update the flag to indicate whether scores exists in output
  this->parameters_->output_scores = (output_scores != nullptr);

  std::vector<OrtValue> feeds;
  // TODO(tianleiwu): allocate fetches. use ping-pong buffers for past state.
  std::vector<OrtValue> fetches;

  // Initialize resources
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

  BeamSearchCpuState cpu_state;
  cpu_state.Init(this->cpu_allocator_,
                 static_cast<size_t>(parameters->BatchBeamSize()),
                 parameters->max_length,
                 parameters->sequence_length,
                 this->IsCuda());

  // buffer in GPU for input_ids, position_ids and attention_mask
  IAllocatorUniquePtr<char> buffer;
  OrtValue expanded_input_ids_in_cpu;
  ORT_RETURN_IF_ERROR(CreateInitialFeeds(cpu_state.sequence_lengths, expanded_input_ids_in_cpu, feeds, buffer));

  BeamSearchState<T> beam_state;
  constexpr bool use_position = true;
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

  gsl::span<const int32_t> input_ids = expanded_input_ids_in_cpu.Get<Tensor>().DataAsSpan<int32_t>();
  cpu_state.SetSequence(input_ids,
                        static_cast<size_t>(parameters->BatchBeamSize()),
                        parameters->max_length,
                        parameters->sequence_length);

#ifdef DEBUG_BEAM_SEARCH
  const IConsoleDumper* dumper = this->GetConsoleDumper();
  dumper->Print("input_ids", feeds[0]);
  dumper->Print("position_ids", feeds[1]);
  dumper->Print("attention_mask", feeds[2]);
#endif

  // Position ids for all iterations except the first. It uses memory buffer owned by next_positions.
  OrtValue position_ids;
  int64_t dims[] = {parameters->BatchBeamSize(), 1};
  TensorShape shape(&dims[0], 2);
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(),
                       shape,
                       beam_state.next_positions.data(),
                       this->temp_space_allocator_->Info(),
                       position_ids);

  int current_length = parameters->sequence_length;
  int iteration_counter = 0;
  while (current_length < parameters->max_length) {
    iteration_counter++;
#ifdef DEBUG_BEAM_SEARCH
    auto cur_len = std::to_string(current_length);
    dumper->Print("***CurrentLength", cur_len, true);
#endif

    status = utils::ExecuteSubgraph(this->decoder_session_state_,
                                    feeds_fetches_manager,
                                    feeds,
                                    fetches,
                                    {},
                                    ExecutionMode::ORT_SEQUENTIAL,
                                    this->context_.GetTerminateFlag(),
                                    this->context_.Logger());

    ORT_RETURN_IF_ERROR(status);

    const OrtValue& logits = fetches[0];
    gsl::span<int32_t> beam_next_tokens;
    gsl::span<int32_t> beam_indices;
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
      ORT_RETURN_IF_ERROR(UpdateFeeds(fetches, feeds, current_length,
                                      position_ids,
                                      beam_next_tokens.as_span<const int32_t>(),
                                      beam_indices.as_span<const int32_t>()));
    }
    fetches.clear();
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
