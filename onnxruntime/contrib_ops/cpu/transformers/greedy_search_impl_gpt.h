// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <algorithm>
#include <vector>

#include "core/common/span_utils.h"
#include "contrib_ops/cpu/transformers/greedy_search_impl_base.h"

namespace onnxruntime {
namespace contrib {

namespace transformers {

namespace gpt_details {
// Some common helpers that can be shared around
std::pair<Status, std::unique_ptr<GptSubgraph>> CreateGptSubgraphAndUpdateParameters(
    const Node& node,
    const SessionState& session_state,
    const std::string& attribute_name,
    const SessionState& subgraph_session_state,
    /*out*/ BeamSearchParameters& parameters);
}  // namespace gpt_details

// Greedy search implementation for GPT-2 model.
template <typename T>
class GreedySearchGpt : public GreedySearchBase<T> {
 public:
  GreedySearchGpt(OpKernelContextInternal& context,
                  const SessionState* init_run_decoder_session_state,
                  GptSubgraph* init_run_gpt_subgraph,
                  const SessionState& decoder_session_state,
                  GptSubgraph& gpt_subgraph,
                  concurrency::ThreadPool* thread_pool,
                  Stream* ort_stream,
                  IConsoleDumper* cuda_dumper,
                  GreedySearchParameters& params,
                  const GenerationDeviceHelper::CreateGptInputsFunc& create_inputs_func,
                  const GenerationDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
                  const GenerationDeviceHelper::TopkFunc& topk_func,
                  const GenerationDeviceHelper::GreedySearchProcessLogitsFunc<T>& process_logits_func,
                  const GenerationDeviceHelper::InitGreedyStateFunc<T>& init_greedy_state_func,
                  const GenerationDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
                  const GenerationDeviceHelper::UpdateGptFeedsFunc<T>& update_feeds_func)
      : GreedySearchBase<T>(context,
                            decoder_session_state,
                            thread_pool,
                            ort_stream,
                            cuda_dumper,
                            params,
                            topk_func,
                            process_logits_func,
                            device_copy_func),
        init_run_decoder_session_state_(init_run_decoder_session_state),
        init_run_gpt_subgraph_(init_run_gpt_subgraph),
        gpt_subgraph_(gpt_subgraph),
        create_inputs_func_(create_inputs_func),
        add_to_feeds_func_(add_to_feeds_func),
        init_greedy_state_func_(init_greedy_state_func),
        update_feeds_func_(update_feeds_func) {
  }

  // Execute beam search in iterations util stopping criteria is reached.
  // In each iteration, GPT subgraph is called, and next token for each sequence is generated.
  Status Execute(const FeedsFetchesManager* init_run_feeds_fetches_manager,
                 const FeedsFetchesManager& feeds_fetches_manager);

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
      gsl::span<const int32_t> next_tokens,
      int past_sequence_length);

  const SessionState* init_run_decoder_session_state_ = nullptr;
  GptSubgraph* init_run_gpt_subgraph_ = nullptr;
  GptSubgraph& gpt_subgraph_;

  // Device specific functions
  GenerationDeviceHelper::CreateGptInputsFunc create_inputs_func_;
  GenerationDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  GenerationDeviceHelper::InitGreedyStateFunc<T> init_greedy_state_func_;
  GenerationDeviceHelper::UpdateGptFeedsFunc<T> update_feeds_func_;
};

template <typename T>
Status GreedySearchGpt<T>::CreateInitialFeeds(gsl::span<int32_t>& sequence_lengths,
                                              OrtValue& expanded_input_ids,
                                              std::vector<OrtValue>& feeds,
                                              IAllocatorUniquePtr<char>& buffer) {
  const OrtValue* input_ids_value = this->context_.GetInputOrtValue(0);
  const Tensor& input_ids = input_ids_value->Get<Tensor>();
  const OrtValue* attn_mask_value = this->context_.GetInputOrtValue(6);

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
                                                      this->ort_stream_,
                                                      this->parameters_->max_length);
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
                                          this->ort_stream_,
                                          this->parameters_->max_length);
}

template <typename T>
Status GreedySearchGpt<T>::UpdateFeeds(
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    bool increase_position,
    gsl::span<const int32_t> next_tokens,
    int past_sequence_length) {
  gsl::span<const int32_t> place_holder;
  return update_feeds_func_(this->temp_space_allocator_,
                            this->ort_stream_,
                            last_outputs,
                            next_inputs,
                            current_length,
                            position_ids,
                            increase_position,
                            next_tokens,
                            place_holder,
                            this->parameters_->num_beams,
                            gpt_subgraph_.GetFirstPastInputIndex(),
                            gpt_subgraph_.GetFirstPresentOutputIndex(),
                            gpt_subgraph_.past_present_share_buffer_,
                            past_sequence_length
                            );
}

template <typename T>
Status GreedySearchGpt<T>::Execute(const FeedsFetchesManager* init_run_feeds_fetches_manager,
                                   const FeedsFetchesManager& feeds_fetches_manager) {
  auto status = Status::OK();
  const GreedySearchParameters* parameters = this->parameters_;

  // Allocate output tensors.
  int64_t sequences_dims[] = {parameters->batch_size, parameters->max_length};
  TensorShape sequences_shape(&sequences_dims[0], sizeof(sequences_dims) / sizeof(sequences_dims[0]));
  Tensor* output_sequences = this->context_.Output(0, sequences_shape);

  std::vector<OrtValue> feeds;
  std::vector<OrtValue> fetches;

  GreedySearchState<T> greedy_state;
  greedy_state.Init(this->cpu_allocator_,
                    this->temp_space_allocator_,
                    static_cast<int>(parameters->BatchBeamSize()),
                    static_cast<int>(parameters->vocab_size),
                    static_cast<int>(parameters->sequence_length),
                    parameters->max_length,
                    this->IsCuda());

  IAllocatorUniquePtr<char> buffer;
  OrtValue expanded_input_ids_in_cpu;
  ORT_RETURN_IF_ERROR(CreateInitialFeeds(greedy_state.sequence_lengths, expanded_input_ids_in_cpu, feeds, buffer));

  if (gpt_subgraph_.past_present_share_buffer_) { // Reuse past and present
    fetches.reserve((int64_t)gpt_subgraph_.GetFirstPresentOutputIndex() + gpt_subgraph_.num_layers);
    fetches.resize(gpt_subgraph_.GetFirstPresentOutputIndex(), OrtValue());
    for (int layer = 0; layer < gpt_subgraph_.num_layers; layer++) {
      int feed_idx = gpt_subgraph_.GetFirstPastInputIndex() + layer;
      OrtValue& past_tensor_value = feeds[feed_idx];
      Tensor* past_tensor = past_tensor_value.GetMutable<Tensor>();
      OrtValue present_tensor_value;
      Tensor::InitOrtValue(past_tensor->DataType(), past_tensor->Shape(), past_tensor->MutableData<T>(),
                           past_tensor->Location(), present_tensor_value);
      fetches.push_back(present_tensor_value);
    }
  }

  init_greedy_state_func_(&greedy_state,
                          greedy_state.sequence_lengths,
                          this->ort_stream_);

  gsl::span<const int32_t> input_ids = expanded_input_ids_in_cpu.Get<Tensor>().DataAsSpan<int32_t>();
  greedy_state.SetSequence(input_ids,
                           static_cast<size_t>(parameters->BatchBeamSize()),
                           parameters->max_length,
                           parameters->sequence_length);

#ifdef DEBUG_GENERATION
  const IConsoleDumper* dumper = this->GetConsoleDumper();
#endif

  // position ids for all iterations except the first. It uses memory buffer owned by next_positions.
  OrtValue position_ids;
  int64_t dims[] = {parameters->BatchBeamSize(), 1};
  TensorShape shape(&dims[0], 2);
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(),
                       shape,
                       greedy_state.next_positions.data(),
                       this->temp_space_allocator_->Info(),
                       position_ids);

  int current_length = parameters->sequence_length;
  int iteration_counter = 0;
  while (current_length < parameters->max_length) {
#ifdef DEBUG_GENERATION
    auto cur_len = std::to_string(current_length);
    dumper->Print("***CurrentLength", cur_len, true);
    dumper->Print("input_ids", feeds[0]);
    dumper->Print("position_ids", feeds[1]);
    dumper->Print("attention_mask", feeds[2]);
    dumper->Print("past", feeds[3]);
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
    gsl::span<int32_t> next_tokens;

    ORT_RETURN_IF_ERROR(this->GenerateNextToken(logits,
                                                next_tokens,
                                                greedy_state,
                                                iteration_counter,
                                                parameters->eos_token_id));

    // When all batches are finished, stop earlier to avoid wasting computation.
    gsl::span<bool>& eos_meet = greedy_state.eos_meet;
    size_t batch_id = 0;
    while (batch_id < eos_meet.size()) {
      if (eos_meet[batch_id] == false) {
        break;
      }
      ++batch_id;
    }
    if (batch_id == eos_meet.size()) {
      break;
    }

    // Increase sequence length after a new token is generated.
    ++current_length;

    // Prepare inputs for next round of subgraph call.
    if (current_length < parameters->max_length) {
      bool increase_position = (iteration_counter > 1);

      ORT_RETURN_IF_ERROR(UpdateFeeds(fetches, feeds, current_length,
                                      position_ids, increase_position,
                                      ReinterpretAsSpan<const int32_t>(next_tokens),
                                      current_length - 1));
    }
    if (gpt_subgraph_.past_present_share_buffer_) {
      // clear fetched values before presents[]
      for (int idx = 0; idx < gpt_subgraph_.GetFirstPresentOutputIndex(); idx++) {
        fetches[idx] = OrtValue();
      }
    } else {
      fetches.clear();
    }
  }

  // Copy the sequences to output
  gsl::span<int32_t> output = output_sequences->MutableDataAsSpan<int32_t>();
  for (int batch_id = 0; batch_id < parameters->batch_size; ++batch_id) {
    auto batch_output = output.subspan(
        static_cast<size_t>(batch_id) * parameters->max_length,
        parameters->max_length);
    gsl::span<const int32_t> sequence_source = greedy_state.sequences.GetSequence(batch_id);
    gsl::copy(sequence_source, batch_output);
  }

  return status;
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
