// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "beam_search_impl_base.h"

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
                const BeamSearchDeviceHelper::UpdateGptFeedsFunc<T>& update_feeds_func)
      : BeamSearchBase<T>(context, decoder_session_state, thread_pool, cuda_stream, cuda_dumper, params, topk_func, process_logits_func, device_copy_func),
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
  Status CreateInitialFeeds(gsl::span<int32_t>& sequence_lengths, OrtValue& expanded_input_ids, std::vector<OrtValue>& feeds, IAllocatorUniquePtr<char>& buffer);

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
Status BeamSearchGpt<T>::CreateInitialFeeds(gsl::span<int32_t>& sequence_lengths, OrtValue& expanded_input_ids, std::vector<OrtValue>& feeds, IAllocatorUniquePtr<char>& buffer) {
  const OrtValue* input_ids_value = context_.GetInputOrtValue(0);
  const Tensor& input_ids = input_ids_value->Get<Tensor>();
  return gpt_subgraph_.CreateInitialFeeds(input_ids, implicit_inputs_, parameters_->num_beams, parameters_->pad_token_id, sequence_lengths, expanded_input_ids, feeds, create_inputs_func_, add_to_feeds_func_, buffer);
}

template <typename T>
Status BeamSearchGpt<T>::UpdateFeeds(
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices) {
  return update_feeds_func_(temp_space_allocator_, cuda_stream_, last_outputs, next_inputs, current_length, position_ids,
                            beam_next_tokens, beam_indices, parameters_->num_beams, GetConsoleDumper());
}

template <typename T>
Status BeamSearchGpt<T>::Execute(const FeedsFetchesManager& feeds_fetches_manager) {
  auto status = Status::OK();
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

  std::vector<OrtValue> feeds;
  // TODO: allocate fetches. use ping-pong buffers for past state.
  std::vector<OrtValue> fetches;

  // Initialize resources
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

  BeamSearchCpuState cpu_state;
  cpu_state.Init(cpu_allocator_, static_cast<size_t>(parameters_->BatchBeamSize()), parameters_->max_length, IsCuda());

  // buffer in GPU for input_ids, position_ids and attention_mask
  // size_t buffer_bytes = SafeInt<size_t>(sizeof(int32_t) + sizeof(int32_t) + sizeof(int32_t)) * parameters_->batch_size * parameters_->num_beams * parameters_->sequence_length;
  // IAllocatorUniquePtr<char> buffer = gpt_subgraph_.GetProvider()->GetScratchBuffer<char>(buffer_bytes);
  IAllocatorUniquePtr<char> buffer;
  OrtValue expanded_input_ids_in_cpu;
  ORT_RETURN_IF_ERROR(CreateInitialFeeds(cpu_state.sequence_lengths, expanded_input_ids_in_cpu, feeds, buffer));

  BeamSearchState<T> beam_state;
  beam_state.Init(temp_space_allocator_,
                  parameters_->batch_size,
                  parameters_->num_beams,
                  parameters_->vocab_size,
                  parameters_->sequence_length,
                  parameters_->max_length,
                  parameters_->output_scores);

  cpu_state.sequences.Init(cpu_state.sequences_space,
                           parameters_->BatchBeamSize(),
                           parameters_->sequence_length,
                           parameters_->max_length);

  gsl::span<const int32_t> input_ids = expanded_input_ids_in_cpu.Get<Tensor>().DataAsSpan<int32_t>();
  init_beam_state_func_(&beam_state,
                        &cpu_state,
                        cpu_state.sequence_lengths,
                        parameters_->batch_size,
                        parameters_->num_beams,
                        input_ids,
                        parameters_->sequence_length,
                        parameters_->max_length,
                        cuda_stream_);

#ifdef DEBUG_BEAM_SEARCH
  const IConsoleDumper* dumper = GetConsoleDumper();
  dumper->Print("input_ids", feeds[0]);
  dumper->Print("position_ids", feeds[1]);
  dumper->Print("attention_mask", feeds[2]);
#endif

  // position ids for all iterations except the first. It uses memory buffer owned by next_positions.
  OrtValue position_ids;
  int64_t dims[] = {parameters_->BatchBeamSize(), 1};
  TensorShape shape(&dims[0], 2);
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), shape, beam_state.next_positions.data(), temp_space_allocator_->Info(), position_ids);

  int current_length = parameters_->sequence_length;
  int iteration_counter = 0;
  while (current_length < parameters_->max_length) {
    iteration_counter++;
#ifdef DEBUG_BEAM_SEARCH
    auto cur_len = std::to_string(current_length);
    dumper->Print("***CurrentLength", cur_len, true);
#endif

    status = utils::ExecuteSubgraph(decoder_session_state_, feeds_fetches_manager, feeds, fetches, {},
                                    ExecutionMode::ORT_SEQUENTIAL, context_.GetTerminateFlag(), context_.Logger());

    ORT_RETURN_IF_ERROR(status);

    const OrtValue& logits = fetches[0];
    gsl::span<int32_t> beam_next_tokens;
    gsl::span<int32_t> beam_indices;
    ORT_RETURN_IF_ERROR(GenerateNextToken(logits, beam_next_tokens, beam_indices, beam_state, cpu_state, iteration_counter));

    // When all batches are finished, stop earlier to avoid wasting computation.
    if (beam_scorer_->IsDone()) {
      break;
    }

    // Increase sequence length after a new token is generated.
    ++current_length;

    // Prepare inputs for next round of subgraph call.
    if (current_length < parameters_->max_length) {
      ORT_RETURN_IF_ERROR(UpdateFeeds(fetches, feeds, current_length,
                                      position_ids,
                                      beam_next_tokens.as_span<const int32_t>(),
                                      beam_indices.as_span<const int32_t>()));
    }
    fetches.clear();
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
