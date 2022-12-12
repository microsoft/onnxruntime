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

// Beam search implementation for GPT-2 model.
template <typename T>
class GreedySearchGpt : public GreedySearchBase<T> {
 public:
  GreedySearchGpt(OpKernelContextInternal& context,
                  const SessionState& decoder_session_state,
                  GptSubgraph& gpt_subgraph,
                  concurrency::ThreadPool* thread_pool,
                  void* cuda_stream,
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
                            cuda_stream,
                            cuda_dumper,
                            params,
                            topk_func,
                            process_logits_func,
                            device_copy_func),
        gpt_subgraph_(gpt_subgraph),
        create_inputs_func_(create_inputs_func),
        add_to_feeds_func_(add_to_feeds_func),
        init_greedy_state_func_(init_greedy_state_func),
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
      bool increase_position,
      gsl::span<const int32_t> next_tokens);

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
                                          buffer);
}

template <typename T>
Status GreedySearchGpt<T>::UpdateFeeds(
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    bool increase_position,
    gsl::span<const int32_t> next_tokens) {
  gsl::span<const int32_t> place_holder;
  return update_feeds_func_(this->temp_space_allocator_,
                            this->cuda_stream_,
                            last_outputs,
                            next_inputs,
                            current_length,
                            position_ids,
                            increase_position,
                            next_tokens,
                            place_holder,
                            this->parameters_->num_beams,
                            gpt_subgraph_.GetFirstPastInputIndex(),
                            gpt_subgraph_.GetFirstPresentOutputIndex());
}

template <typename T>
Status GreedySearchGpt<T>::Execute(const FeedsFetchesManager& feeds_fetches_manager) {
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

  init_greedy_state_func_(&greedy_state,
                          greedy_state.sequence_lengths,
                          this->cuda_stream_);

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
    iteration_counter++;
#ifdef DEBUG_GENERATION
    auto cur_len = std::to_string(current_length);
    dumper->Print("***CurrentLength", cur_len, true);
    dumper->Print("input_ids", feeds[0]);
    dumper->Print("position_ids", feeds[1]);
    dumper->Print("attention_mask", feeds[2]);
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
                                      ReinterpretAsSpan<const int32_t>(next_tokens)));
    }
    fetches.clear();
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
