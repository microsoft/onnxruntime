// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/transformers/beam_search_impl_base.h"

#include "core/common/span_utils.h"

namespace onnxruntime {
namespace contrib {

namespace transformers {

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
                IConsoleDumper* cuda_dumper,
                BeamSearchParameters& params,
                const GenerationDeviceHelper::CreateGptInputsFunc& create_inputs_func,
                const GenerationDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
                const GenerationDeviceHelper::TopkFunc& topk_func,
                const GenerationDeviceHelper::ProcessLogitsFunc<T>& process_logits_func,
                const GenerationDeviceHelper::InitBeamStateFunc<T>& init_beam_state_func,
                const GenerationDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
                const GenerationDeviceHelper::DeviceCopyFunc<int32_t>& device_copy_int32_func,
                const GenerationDeviceHelper::UpdateGptFeedsFunc<T>& update_feeds_func,
                const GenerationDeviceHelper::CreateBeamScorer& create_beam_scorer_func)
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
        create_beam_scorer_func_(create_beam_scorer_func) {}

#ifdef USE_CUDA
  Status InitializeCuda(
      const GenerationDeviceHelper::ReorderPastStateFunc& reorder_past_state_func,
      const void* cuda_device_prop,
      int cuda_device_arch) {
    reorder_past_state_func_ = reorder_past_state_func;
    cuda_device_prop_ = cuda_device_prop;
    cuda_device_arch_ = cuda_device_arch;
    if (gpt_subgraph_.has_decoder_masked_attention_) {
      ORT_RETURN_IF_NOT(cuda_device_arch_ >= 530,
                        "Decoder masked self attention can only be used on "
                        "GPU cards of compute capability 5.3 or higher. "
                        "This card has compute capability ",
                        cuda_device_arch_);
    }
    return Status::OK();
  }
#endif

  // Execute beam search in iterations util stopping criteria is reached.
  // In each iteration, GPT subgraph is called, and next token for each sequence is generated.
  Status Execute(const FeedsFetchesManager* init_run_feeds_fetches_manager,
                 const FeedsFetchesManager& feeds_fetches_manager);

 private:
  // Prepare the inputs for first inference of subgraph
  Status CreateInitialFeeds(gsl::span<int32_t>& sequence_lengths,
                            OrtValue& expanded_input_ids,
                            std::vector<OrtValue>& feeds,
                            IAllocatorUniquePtr<char>& buffer,
                            bool need_cache_indir);

  // Update the input for next iteration.
  Status UpdateFeeds(
      const std::vector<OrtValue>& last_outputs,
      std::vector<OrtValue>& next_inputs,
      int current_length,
      OrtValue& position_ids,
      bool increase_position,
      gsl::span<const int32_t> beam_next_tokens,
      gsl::span<const int32_t> beam_indices_cpu,
      gsl::span<const int32_t> beam_indices_gpu,
      int past_sequence_length,
      int input_sequence_len,
      bool need_cache_indir);

  const SessionState* init_run_decoder_session_state_ = nullptr;
  GptSubgraph* init_run_gpt_subgraph_ = nullptr;
  GptSubgraph& gpt_subgraph_;

  // Device specific functions
  GenerationDeviceHelper::CreateGptInputsFunc create_inputs_func_;
  GenerationDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  GenerationDeviceHelper::InitBeamStateFunc<T> init_beam_state_func_;
#ifdef USE_CUDA
  GenerationDeviceHelper::ReorderPastStateFunc reorder_past_state_func_;
#endif
  GenerationDeviceHelper::UpdateGptFeedsFunc<T> update_feeds_func_;
  GenerationDeviceHelper::CreateBeamScorer create_beam_scorer_func_;

  const void* cuda_device_prop_ = nullptr;
  int cuda_device_arch_ = 0;
};

template <typename T>
Status BeamSearchGpt<T>::CreateInitialFeeds(gsl::span<int32_t>& sequence_lengths,
                                            OrtValue& expanded_input_ids,
                                            std::vector<OrtValue>& feeds,
                                            IAllocatorUniquePtr<char>& buffer,
                                            bool need_cache_indir) {
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
                                                      this->ort_stream_,
                                                      this->parameters_->max_length,
                                                      need_cache_indir);
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
                                          this->parameters_->max_length,
                                          need_cache_indir);
}

template <typename T>
Status BeamSearchGpt<T>::UpdateFeeds(
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    bool increase_position,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices_cpu,
    gsl::span<const int32_t> beam_indices_gpu,
    int past_sequence_length,
    int input_sequence_len,
    bool need_cache_indir) {
  return update_feeds_func_(this->temp_space_allocator_,
                            this->ort_stream_,
                            last_outputs,
                            next_inputs,
                            current_length,
                            position_ids,
                            increase_position,
                            beam_next_tokens,
                            beam_indices_cpu,
                            beam_indices_gpu,
                            this->parameters_->num_beams,
                            gpt_subgraph_.GetFirstPastInputIndex(),
                            gpt_subgraph_.GetFirstPresentOutputIndex(),
                            gpt_subgraph_.past_present_share_buffer_,
                            past_sequence_length,
                            input_sequence_len,
                            need_cache_indir);
}

template <typename T>
Status BeamSearchGpt<T>::Execute(const FeedsFetchesManager* init_run_feeds_fetches_manager,
                                 const FeedsFetchesManager& feeds_fetches_manager) {
  auto status = Status::OK();
  const BeamSearchParameters* parameters = this->parameters_;
  TensorShape sequences_shape{parameters->batch_size, parameters->num_return_sequences, parameters->max_length};
  Tensor* output_sequences = this->context_.Output(0, sequences_shape);

  TensorShape sequences_scores_shape{parameters->batch_size, parameters->num_return_sequences};
  Tensor* output_sequences_scores = this->context_.Output(1, sequences_scores_shape);

  TensorShape scores_shape{
      static_cast<int64_t>(parameters->max_length) - static_cast<int64_t>(parameters->sequence_length),
      parameters->batch_size, parameters->num_beams, parameters->vocab_size};
  Tensor* output_scores = this->context_.Output(2, scores_shape);

  // Update the flag to indicate whether scores exists in output
  this->parameters_->output_scores = (output_scores != nullptr);

  std::vector<OrtValue> feeds;
  // TODO(tianleiwu): allocate fetches. use ping-pong buffers for past state.
  std::vector<OrtValue> fetches;

  // Initialize resources
  this->beam_scorer_ = create_beam_scorer_func_
                           ? create_beam_scorer_func_(*parameters, this->temp_space_allocator_, this->cpu_allocator_, this->ort_stream_)
                           : std::make_unique<BeamSearchScorer>(*parameters, this->cpu_allocator_);

  BeamSearchCpuState cpu_state{*parameters,
                               this->cpu_allocator_,
                               this->IsCuda(),
                               this->ort_stream_};

  // buffer in GPU for input_ids, position_ids and attention_mask
  IAllocatorUniquePtr<char> buffer;
  OrtValue expanded_input_ids_in_cpu;
  ORT_RETURN_IF_ERROR(CreateInitialFeeds(cpu_state.sequence_lengths, expanded_input_ids_in_cpu, feeds, buffer,
                                         gpt_subgraph_.has_decoder_masked_attention_));

  if (gpt_subgraph_.past_present_share_buffer_) {  // Reuse past and present
    fetches.reserve(static_cast<size_t>(gpt_subgraph_.GetFirstPresentOutputIndex()) + gpt_subgraph_.num_layers);
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

  BeamSearchState<T> beam_state{*parameters,
                                this->temp_space_allocator_,
                                gpt_subgraph_.has_decoder_masked_attention_,
                                true /* use_position */,
                                this->ort_stream_};

  init_beam_state_func_(&beam_state,
                        cpu_state.sequence_lengths,
                        parameters->batch_size,
                        parameters->num_beams,
                        this->ort_stream_);

  cpu_state.SetExpandedSequence(expanded_input_ids_in_cpu.Get<Tensor>().DataAsSpan<int32_t>());

  // beam_state.sequences_device is the GPU version of cpu_state.sequences_space,
  // this copies it over to the GPU after setting it up on the CPU
  if (this->IsCuda()) {
    cpu_state.sequences.InitDevice(beam_state.sequences_device);
    ORT_RETURN_IF_ERROR(this->device_copy_int32_func_(beam_state.sequences_device.subspan(0, beam_state.sequences_device.size() / 2),
                                                      cpu_state.sequences_space.subspan(0, cpu_state.sequences_space.size() / 2),
                                                      nullptr,
                                                      DeviceCopyDirection::hostToDevice));
  }

#ifdef DEBUG_GENERATION
  const IConsoleDumper* dumper = this->GetConsoleDumper();
#endif
  // Position ids for all iterations except the first. It uses memory buffer owned by next_positions.
  OrtValue position_ids;
  TensorShape shape{parameters->BatchBeamSize(), 1};
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
    ORT_RETURN_IF_ERROR(this->GenerateNextToken(logits,
                                                beam_next_tokens,
                                                beam_state,
                                                cpu_state,
                                                iteration_counter));

    // When all batches are finished, stop earlier to avoid wasting computation.
    if (this->beam_scorer_->IsDone())
      break;

    // Increase sequence length after a new token is generated.
    ++current_length;

#ifdef USE_CUDA
    // Reorder past state after first run if the GPT subgraph (the one used after the first iteration)
    // contains DecoderMaskedSelfAttention nodes
    if (iteration_counter == 1 && gpt_subgraph_.has_decoder_masked_attention_) {
      size_t offset = static_cast<size_t>(gpt_subgraph_.GetFirstPresentOutputIndex());
      // We will use the same staging buffer while transposing all the layers' past state
      // and this is okay because we use the same stream to do the staging copy and the transpose
      // operations.
      // If we ever do them in different streams, we must use different staging buffers to avoid data
      // races.
      for (size_t i = 0; i < static_cast<size_t>(gpt_subgraph_.num_layers); ++i) {
        ORT_RETURN_IF_ERROR(reorder_past_state_func_(cuda_device_prop_,
                                                     *fetches[offset + i].GetMutable<Tensor>(),
                                                     beam_state.staging_for_past_state_reorder,
                                                     this->ort_stream_));
      }
    }
#endif

    // Prepare inputs for next round of subgraph call.
    if (current_length < parameters->max_length) {
      gsl::span<const int32_t> place_holder;
      // For the first iteration, position_ids is initialized as sequence lengths. We can add it to feeds directly.
      // For the remaining iterations, we need increase position_ids first, then add it to feeds.
      bool increase_position = (iteration_counter > 1);
      ORT_RETURN_IF_ERROR(UpdateFeeds(fetches, feeds, current_length,
                                      position_ids, increase_position,
                                      ReinterpretAsSpan<const int32_t>(beam_next_tokens),
                                      gpt_subgraph_.has_decoder_masked_attention_
                                          ? place_holder
                                          : ReinterpretAsSpan<const int32_t>(this->beam_scorer_->GetNextIndicesCPU()),
                                      gpt_subgraph_.has_decoder_masked_attention_
                                          ? ReinterpretAsSpan<const int32_t>(this->beam_scorer_->GetNextIndicesGPU())
                                          : place_holder,
                                      current_length - 1,
                                      parameters->sequence_length,
                                      gpt_subgraph_.has_decoder_masked_attention_));
    }

    if (this->beam_scorer_->IsDoneLater())
      break;

    if (gpt_subgraph_.past_present_share_buffer_) {
      // clear fetched values before presents[]
      for (int idx = 0; idx < gpt_subgraph_.GetFirstPresentOutputIndex(); idx++) {
        fetches[idx] = OrtValue();
      }
    } else {
      fetches.clear();
    }
  }

  gsl::span<const float> final_beam_scores = beam_state.beam_scores;
  this->beam_scorer_->Finalize(cpu_state.sequences,
                               final_beam_scores,
                               output_sequences,
                               output_sequences_scores);

  // Output per token scores
  if (output_scores) {
    gsl::span<float> target = output_scores->MutableDataAsSpan<float>();
    gsl::span<const float> source = beam_state.scores;
    assert(target.size() == source.size());
    ORT_RETURN_IF_ERROR(this->device_copy_func_(target, source, nullptr, DeviceCopyDirection::deviceToDevice));
  }

  return status;
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
