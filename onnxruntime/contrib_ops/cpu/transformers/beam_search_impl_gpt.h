// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/transformers/beam_search_impl_base.h"

#include "core/common/span_utils.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace contrib {

namespace transformers {

namespace beam_search_gpt_impl_detail {
// Environment variable to enable or disable pre-allocation of past/present buffers.
// The present buffer will be used to make up the present state fetches for the GPT2 execution.
// The past buffer will be used to form the past state feeds for the GPT2 execution.
// This will also optimize the selection of the next run's past state from the previous run's present state
// for the selected beams.
constexpr const char* kDisableBeamSearchPreallocatedFetches = "ORT_DISABLE_BEAM_SEARCH_PREALLOCATED_FETCHES";
}  // namespace beam_search_gpt_impl_detail

// Beam search implementation for GPT-2 model.
template <typename T>
class BeamSearchGpt : public BeamSearchBase<T> {
 public:
  BeamSearchGpt(OpKernelContextInternal& context,
                const SessionState* init_run_decoder_session_state,
                GptSubgraph* init_run_gpt_subgraph,
                const SessionState& decoder_session_state,
                GptSubgraph& gpt_subgraph,
                concurrency::ThreadPool* thread_pool,
                Stream* ort_stream,
                int max_threads_per_block,
                IConsoleDumper* cuda_dumper,
                BeamSearchParameters& params,
                const GenerationDeviceHelper::CreateGptInputsFunc& create_inputs_func,
                const GenerationDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
                const GenerationDeviceHelper::TopkFunc& topk_func,
                const GenerationDeviceHelper::ProcessLogitsFunc<T>& process_logits_func,
                const GenerationDeviceHelper::InitBeamStateFunc<T>& init_beam_state_func,
                const GenerationDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
                const GenerationDeviceHelper::DeviceCopyFunc<int32_t>& device_copy_int32_func,
                const GenerationDeviceHelper::UpdateGptFeedsFunc<T>& update_feeds_func)
      : BeamSearchBase<T>(context, decoder_session_state, thread_pool,
                          ort_stream, cuda_dumper, params,
                          topk_func, process_logits_func, device_copy_func, device_copy_int32_func),
        init_run_decoder_session_state_(init_run_decoder_session_state),
        init_run_gpt_subgraph_(init_run_gpt_subgraph),
        gpt_subgraph_(gpt_subgraph),
        create_inputs_func_(create_inputs_func),
        add_to_feeds_func_(add_to_feeds_func),
        init_beam_state_func_(init_beam_state_func),
        update_feeds_func_(update_feeds_func),
        max_threads_per_block_(max_threads_per_block) {
  }

  // Execute beam search in iterations util stopping criteria is reached.
  // In each iteration, GPT subgraph is called, and next token for each sequence is generated.
  Status Execute(const FeedsFetchesManager* init_run_feeds_fetches_manager,
                 const FeedsFetchesManager& feeds_fetches_manager);

  // Using pre-allocated past and present buffers is only supported on CUDA for now.
  // TODO(hasesh): Support it on CPU as well.
  bool use_preallocated_past_and_present_buffers_ = this->IsCuda() &&
                                                    !ParseEnvironmentVariableWithDefault<bool>(
                                                        beam_search_gpt_impl_detail::kDisableBeamSearchPreallocatedFetches, false);

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
      bool increase_position,
      gsl::span<const int32_t> beam_next_tokens,
      gsl::span<const int32_t> beam_indices,
      IBeamSearchState<T>* beam_state);

  const SessionState* init_run_decoder_session_state_ = nullptr;
  GptSubgraph* init_run_gpt_subgraph_ = nullptr;
  GptSubgraph& gpt_subgraph_;

  // Device specific functions
  GenerationDeviceHelper::CreateGptInputsFunc create_inputs_func_;
  GenerationDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  GenerationDeviceHelper::InitBeamStateFunc<T> init_beam_state_func_;
  GenerationDeviceHelper::UpdateGptFeedsFunc<T> update_feeds_func_;

  // Device specific parameters
  int max_threads_per_block_ = 0;
};

template <typename T>
Status BeamSearchGpt<T>::CreateInitialFeeds(gsl::span<int32_t>& sequence_lengths,
                                            OrtValue& expanded_input_ids,
                                            std::vector<OrtValue>& feeds,
                                            IAllocatorUniquePtr<char>& buffer) {
  const OrtValue* input_ids_value = this->context_.GetInputOrtValue(0);
  const Tensor& input_ids = input_ids_value->Get<Tensor>();
  const OrtValue* attn_mask_value = this->context_.GetInputOrtValue(9);

  if (init_run_gpt_subgraph_ != nullptr) {
    return init_run_gpt_subgraph_->CreateInitialFeeds(input_ids,
                                                      this->implicit_inputs_,
                                                      this->parameters_->num_beams,
                                                      this->parameters_->pad_token_id,
                                                      sequence_lengths,
                                                      expanded_input_ids,
                                                      attn_mask_value,
                                                      feeds,
                                                      this->create_inputs_func_,
                                                      this->add_to_feeds_func_,
                                                      buffer,
                                                      this->ort_stream_);
  }

  return gpt_subgraph_.CreateInitialFeeds(input_ids,
                                          this->implicit_inputs_,
                                          this->parameters_->num_beams,
                                          this->parameters_->pad_token_id,
                                          sequence_lengths,
                                          expanded_input_ids,
                                          attn_mask_value,
                                          feeds,
                                          this->create_inputs_func_,
                                          this->add_to_feeds_func_,
                                          buffer,
                                          this->ort_stream_);
}

template <typename T>
Status BeamSearchGpt<T>::UpdateFeeds(
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    bool increase_position,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    IBeamSearchState<T>* beam_state) {
  return update_feeds_func_(this->temp_space_allocator_,
                            this->ort_stream_,
                            last_outputs,
                            next_inputs,
                            current_length,
                            position_ids,
                            increase_position,
                            beam_next_tokens,
                            beam_indices,
                            this->parameters_->num_beams,
                            gpt_subgraph_.GetFirstPastInputIndex(),
                            gpt_subgraph_.GetFirstPresentOutputIndex(),
                            false,
                            -1,
                            use_preallocated_past_and_present_buffers_,
                            beam_state,
                            max_threads_per_block_);
}

template <typename T>
Status BeamSearchGpt<T>::Execute(const FeedsFetchesManager* init_run_feeds_fetches_manager,
                                 const FeedsFetchesManager& feeds_fetches_manager) {
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
  // TODO(tianleiwu): allocate logits ?
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
                  gpt_subgraph_.num_layers,
                  gpt_subgraph_.num_heads,
                  gpt_subgraph_.head_size,
                  parameters->output_scores,
                  use_position,
                  use_preallocated_past_and_present_buffers_);

  init_beam_state_func_(&beam_state,
                        cpu_state.sequence_lengths,
                        parameters->batch_size,
                        parameters->num_beams,
                        this->ort_stream_);

  // Allocate present state fetches.
  // We will use the same buffer always - it is pre-allocated to hold max_sequence_length of present state values.
  if (use_preallocated_past_and_present_buffers_) {
    // init run and other run subgraphs are expected to have same past/present info. So, use any one of them below.
    fetches.reserve(static_cast<int64_t>(gpt_subgraph_.GetFirstPresentOutputIndex()) + gpt_subgraph_.num_layers);
    fetches.resize(gpt_subgraph_.GetFirstPresentOutputIndex(), OrtValue());
    for (int64_t layer = 0; layer < static_cast<int64_t>(gpt_subgraph_.num_layers); layer++) {
      OrtValue present_tensor;
      gsl::span<T> present_state_buffer = beam_state.GetPresentStateBuffer();

      gsl::span<T> current_layer_present_buffer = present_state_buffer.subspan(static_cast<int>(layer) * 2 * parameters->batch_size *
                                                                                   parameters->num_beams * gpt_subgraph_.num_heads *
                                                                                   parameters->max_length * gpt_subgraph_.head_size,

                                                                               2 * parameters->batch_size *
                                                                                   parameters->num_beams * gpt_subgraph_.num_heads *
                                                                                   parameters->sequence_length *
                                                                                   gpt_subgraph_.head_size);

      int64_t dims[] = {2, parameters->batch_size * parameters->num_beams,
                        gpt_subgraph_.num_heads, parameters->sequence_length,
                        gpt_subgraph_.head_size};

      TensorShape shape(&dims[0], 5);

      Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), shape, current_layer_present_buffer.data(),
                           this->temp_space_allocator_->Info(), present_tensor);

      fetches.push_back(present_tensor);
    }
  }

  gsl::span<const int32_t> input_ids = expanded_input_ids_in_cpu.Get<Tensor>().DataAsSpan<int32_t>();
  cpu_state.SetSequence(input_ids,
                        static_cast<size_t>(parameters->BatchBeamSize()),
                        parameters->max_length,
                        parameters->sequence_length);

#ifdef DEBUG_GENERATION
  const IConsoleDumper* dumper = this->GetConsoleDumper();
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
#ifdef DEBUG_GENERATION
    auto cur_len = std::to_string(current_length);
    dumper->Print("***CurrentLength", cur_len, true);
    dumper->Print("iteration", iteration_counter, true);

    dumper->Print("input_ids", feeds[0]);
    dumper->Print("position_ids", feeds[1]);
    dumper->Print("attention_mask", feeds[2]);
    for (size_t i = 3; i < feeds.size(); i++) {
      dumper->Print("past", static_cast<int>(i) - 3, true);
      dumper->Print("", feeds[i]);
    }
#endif

    // For the first iteration use the init_run_decoder subgraph (if present)
    if (iteration_counter++ == 0 &&
        init_run_decoder_session_state_ != nullptr) {
#ifdef DEBUG_NODE_INPUTS_OUTPUTS
      const_cast<SessionState*>(this->init_run_decoder_session_state_)->IncrementGraphExecutionCounter();
#endif
      status = utils::ExecuteSubgraph(*init_run_decoder_session_state_,
                                      *init_run_feeds_fetches_manager,
                                      feeds,
                                      fetches,
                                      {},
                                      ExecutionMode::ORT_SEQUENTIAL,
                                      this->context_.GetTerminateFlag(),
                                      this->context_.Logger(),
                                      this->ort_stream_);
    } else {
#ifdef DEBUG_NODE_INPUTS_OUTPUTS
      const_cast<SessionState&>(this->decoder_session_state_).IncrementGraphExecutionCounter();
#endif
      status = utils::ExecuteSubgraph(this->decoder_session_state_,
                                      feeds_fetches_manager,
                                      feeds,
                                      fetches,
                                      {},
                                      ExecutionMode::ORT_SEQUENTIAL,
                                      this->context_.GetTerminateFlag(),
                                      this->context_.Logger(),
                                      this->ort_stream_);
    }

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
      // For the first iteration, position_ids is initialized as sequence lengths. We can add it to feeds directly.
      // For the remaining iterations, we need increase position_ids first, then add it to feeds.
      bool increase_position = (iteration_counter > 1);
      ORT_RETURN_IF_ERROR(UpdateFeeds(fetches, feeds, current_length,
                                      position_ids, increase_position,
                                      ReinterpretAsSpan<const int32_t>(beam_next_tokens),
                                      ReinterpretAsSpan<const int32_t>(beam_indices),
                                      &beam_state));
    }

    if (use_preallocated_past_and_present_buffers_) {
      // Clear fetches before present state(s)
      for (size_t i = 0; i < static_cast<size_t>(gpt_subgraph_.GetFirstPresentOutputIndex()); ++i) {
        fetches[i] = OrtValue();
      }

      // Re-adjust shapes of present state fetches for next iteration
      for (size_t i = 0; i < static_cast<size_t>(gpt_subgraph_.num_layers); i++) {
        auto fetch_offset = static_cast<size_t>(gpt_subgraph_.GetFirstPresentOutputIndex());

        OrtValue fetch = fetches[fetch_offset + i];
        auto* fetch_tensor = fetch.GetMutable<Tensor>();

        OrtValue adjusted_fetch;

        // Adjusted fetch will have same buffer as the old fetch but its shape in the sequence
        // dim will be incremented by 1.
        TensorShape adjusted_shape = fetch_tensor->Shape();
        adjusted_shape[3] += 1;

        Tensor::InitOrtValue(fetch_tensor->DataType(), adjusted_shape, fetch_tensor->MutableData<T>(),
                             fetch_tensor->Location(), adjusted_fetch);

        fetches[fetch_offset + i] = adjusted_fetch;
      }

    } else {
      fetches.clear();
    }
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
    assert(target.size() == source.size());
    ORT_RETURN_IF_ERROR(this->device_copy_func_(target, source, nullptr, DeviceCopyDirection::deviceToDevice));
  }

  return status;
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
