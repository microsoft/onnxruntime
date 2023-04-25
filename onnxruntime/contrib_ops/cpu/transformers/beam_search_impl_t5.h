// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/span_utils.h"
#include "contrib_ops/cpu/transformers/generation_shared.h"  // for DEBUG_GENERATION
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
               Stream* ort_stream,
               IConsoleDumper* cuda_dumper,
               BeamSearchParameters& params,
               const GenerationDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
               const GenerationDeviceHelper::ReorderPastStateFunc& reorder_past_state_func,
               const GenerationDeviceHelper::InitCacheIndirFunc& init_cache_indir_func,
               const GenerationDeviceHelper::TopkFunc& topk_func,
               const GenerationDeviceHelper::ProcessLogitsFunc<T>& process_logits_func,
               const GenerationDeviceHelper::InitBeamStateFunc<T>& init_beam_state_func,
               const GenerationDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
               const GenerationDeviceHelper::DeviceCopyFunc<int32_t>& device_copy_int32_func,
               const GenerationDeviceHelper::CreateEncoderInputsFunc& create_encoder_inputs_func,
               const GenerationDeviceHelper::UpdateDecoderFeedsFunc<T>& update_decoder_feeds_func,
               const GenerationDeviceHelper::ExpandBufferFunc<int32_t>& expand_buffer_int32_func,
               const GenerationDeviceHelper::ExpandBufferFunc<float>& expand_buffer_float_func,
               const GenerationDeviceHelper::ExpandBufferFunc<MLFloat16>& expand_buffer_float16_func,
               const void* cuda_device_prop,
               int cuda_device_arch)
      : BeamSearchBase<T>(context, decoder_session_state, thread_pool,
                          ort_stream, cuda_dumper, params,
                          topk_func, process_logits_func, device_copy_func, device_copy_int32_func),
        encoder_session_state_(encoder_session_state),
        encoder_subgraph_(encoder_subgraph),
        decoder_subgraph_(decoder_subgraph),
        add_to_feeds_func_(add_to_feeds_func),
        init_beam_state_func_(init_beam_state_func),
        reorder_past_state_func_(reorder_past_state_func),
        init_cache_indir_func_(init_cache_indir_func),
        create_encoder_inputs_func_(create_encoder_inputs_func),
        update_decoder_feeds_func_(update_decoder_feeds_func),
        expand_buffer_int32_func_(expand_buffer_int32_func),
        expand_buffer_float_func_(expand_buffer_float_func),
        expand_buffer_float16_func_(expand_buffer_float16_func),
        cuda_device_prop_(cuda_device_prop),
        cuda_device_arch_(cuda_device_arch) {
    if (decoder_subgraph_.has_decoder_masked_attention_) {
      ORT_ENFORCE(cuda_device_arch_ >= 530,
                  "Decoder masked multihead attention can only be used on "
                  "GPU cards of compute capability 5.3 or higher. "
                  "This card has compute capability ",
                  cuda_device_arch_);
    }
  }

  // Execute beam search in iterations util stopping criteria is reached.
  Status Execute(const FeedsFetchesManager& encoder_feeds_fetches_manager,
                 const FeedsFetchesManager& decoder_feeds_fetches_manager);

 private:
  const SessionState& encoder_session_state_;

  T5EncoderSubgraph& encoder_subgraph_;
  T5DecoderSubgraph& decoder_subgraph_;

  // Device specific functions
  GenerationDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  GenerationDeviceHelper::InitBeamStateFunc<T> init_beam_state_func_;
  GenerationDeviceHelper::ReorderPastStateFunc reorder_past_state_func_;
  GenerationDeviceHelper::InitCacheIndirFunc init_cache_indir_func_;
  GenerationDeviceHelper::CreateEncoderInputsFunc create_encoder_inputs_func_;
  GenerationDeviceHelper::UpdateDecoderFeedsFunc<T> update_decoder_feeds_func_;
  GenerationDeviceHelper::ExpandBufferFunc<int32_t> expand_buffer_int32_func_;
  GenerationDeviceHelper::ExpandBufferFunc<float> expand_buffer_float_func_;
  GenerationDeviceHelper::ExpandBufferFunc<MLFloat16> expand_buffer_float16_func_;

  const void* cuda_device_prop_ = nullptr;
  int cuda_device_arch_ = 0;
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

  const OrtValue* encoder_attn_mask_value = this->context_.GetInputOrtValue(9);

  BeamSearchCpuState cpu_state;
  cpu_state.Init(this->cpu_allocator_,
                 static_cast<size_t>(parameters->BatchBeamSize()),
                 parameters->max_length,
                 parameters->sequence_length,
                 this->IsCuda());

  IAllocatorUniquePtr<char> buffer;
  OrtValue decoder_input_ids;  // Tensor in CPU, and it will be used to initialize sequence in cpu_state
  ORT_RETURN_IF_ERROR(this->encoder_subgraph_.CreateInitialFeeds(
      encoder_input_ids,
      encoder_attn_mask_value,
      this->implicit_inputs_,
      parameters->pad_token_id,
      parameters->decoder_start_token_id,
      encoder_feeds,
      this->create_encoder_inputs_func_,
      this->add_to_feeds_func_,
      buffer,
      decoder_input_ids,
      this->ort_stream_));

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
  const_cast<SessionState&>(this->encoder_session_state_).IncrementGraphExecutionCounter();
#endif
  ORT_RETURN_IF_ERROR(utils::ExecuteSubgraph(this->encoder_session_state_,
                                             encoder_feeds_fetches_manager,
                                             encoder_feeds,
                                             encoder_fetches,
                                             {},
                                             ExecutionMode::ORT_SEQUENTIAL,
                                             this->context_.GetTerminateFlag(),
                                             this->context_.Logger(),
                                             this->ort_stream_));

#ifdef DEBUG_GENERATION
  const IConsoleDumper* dumper = this->GetConsoleDumper();
  for (int i = 0; i < this->encoder_subgraph_.num_subgraph_inputs; i++) {
    dumper->Print("encoder_feeds", static_cast<int>(i), true);
    dumper->Print("", encoder_feeds[i]);
  }

  for (int i = 0; i <= encoder_subgraph_.GetFirstPresentOutputIndex(); i++) {
    dumper->Print("encoder_fetches", i, true);
    dumper->Print("", encoder_fetches[i]);
  }
#endif

  // ------------------------------------
  // Initialize resources
  // ------------------------------------

  // Copy decoder_input_ids (in CPU) to sequence. It contains decoder_start_token_id for each beam.
  cpu_state.SetSequence(decoder_input_ids.Get<Tensor>().DataAsSpan<int32_t>(),
                        static_cast<size_t>(parameters->BatchBeamSize()),
                        parameters->num_beams,
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
                  parameters->num_heads,
                  parameters->head_size,
                  decoder_subgraph_.has_decoder_masked_attention_,
                  parameters->output_scores,
                  use_position);

  init_beam_state_func_(&beam_state,
                        cpu_state.sequence_lengths,
                        parameters->batch_size,
                        parameters->num_beams,
                        this->ort_stream_);

  // ------------------------------------------------------------------------------
  // Generate next token from logits output from encoder, and initialize decoder inputs.
  // ------------------------------------------------------------------------------
  gsl::span<int32_t> beam_next_tokens;
  gsl::span<int32_t> beam_indices;

  int iteration_counter = 0;
  std::vector<OrtValue> decoder_feeds;
  int current_length = parameters->sequence_length;

  std::vector<OrtValue> decoder_fetches;

  if (current_length + 1 < parameters->max_length) {
    ++iteration_counter;
    ORT_RETURN_IF_ERROR(this->GenerateNextToken(encoder_fetches[0],
                                                beam_next_tokens,
                                                beam_indices,
                                                beam_state,
                                                cpu_state,
                                                iteration_counter));
    ++current_length;  // Increase sequence length after a new token is generated.

    ORT_RETURN_IF_ERROR(decoder_subgraph_.CreateInitialFeeds(this->cpu_allocator_,
                                                             ReinterpretAsSpan<const int32_t>(beam_next_tokens),
                                                             this->implicit_inputs_,
                                                             encoder_feeds,
                                                             encoder_fetches,
                                                             decoder_feeds,
                                                             this->device_copy_int32_func_,
                                                             this->expand_buffer_int32_func_,
                                                             this->expand_buffer_float_func_,
                                                             this->expand_buffer_float16_func_,
                                                             parameters->num_beams,
                                                             this->ort_stream_,
                                                             decoder_subgraph_.UseSequenceAsInputIds(),
                                                             current_length,
                                                             cpu_state.sequences,
                                                             parameters->max_length,
                                                             decoder_subgraph_.has_decoder_masked_attention_));

    if (decoder_subgraph_.past_present_share_buffer_) {
      decoder_fetches.reserve(static_cast<int64_t>(decoder_subgraph_.GetFirstPresentOutputIndex()) + 2 * static_cast<int64_t>(decoder_subgraph_.num_layers));
      decoder_fetches.resize(decoder_subgraph_.GetFirstPresentOutputIndex(), OrtValue());
      for (int layer = 0; layer < 2 * decoder_subgraph_.num_layers; layer++) {
        int feed_idx = decoder_subgraph_.GetFirstPastInputIndex() + layer;
        OrtValue& past_tensor_value = decoder_feeds[feed_idx];
        Tensor* past_tensor = past_tensor_value.GetMutable<Tensor>();
        OrtValue present_tensor_value;
        Tensor::InitOrtValue(past_tensor->DataType(), past_tensor->Shape(), past_tensor->MutableData<T>(),
                             past_tensor->Location(), present_tensor_value);
        decoder_fetches.push_back(present_tensor_value);
      }
    }

    if (decoder_subgraph_.has_decoder_masked_attention_) {
      size_t offset = static_cast<size_t>(decoder_subgraph_.GetFirstPastInputIndex());
      // Here we only need to reorder the past key for self-attention and cross-attention.
      for (size_t i = 0; i < 2 * static_cast<size_t>(decoder_subgraph_.num_layers); ++i) {
        ORT_RETURN_IF_ERROR(reorder_past_state_func_(cuda_device_prop_,
                                                     *decoder_feeds[offset + 2 * i].GetMutable<Tensor>(),
                                                     beam_state.staging_for_past_state_reorder,
                                                     this->ort_stream_));
      }
      size_t cache_indir_input_offset = static_cast<size_t>(decoder_subgraph_.GetFirstPastInputIndex()) + 4 * static_cast<size_t>(decoder_subgraph_.num_layers) + 2;
      ORT_RETURN_IF_ERROR(init_cache_indir_func_(*decoder_feeds[cache_indir_input_offset].GetMutable<Tensor>(), this->ort_stream_));
    }
  }

  while (current_length < parameters->max_length) {
    iteration_counter++;
#ifdef DEBUG_GENERATION
    auto cur_len = std::to_string(current_length);
    dumper->Print("***CurrentLength", cur_len, true);

    for (int i = 0; i <= decoder_subgraph_.GetFirstPastInputIndex(); i++) {
      dumper->Print("decoder_feeds", i, true);
      dumper->Print("", decoder_feeds[i]);
    }
    auto offset = decoder_subgraph_.GetFirstPastInputIndex() + 4 * decoder_subgraph_.num_layers;
    dumper->Print("past_sequence_length", offset, true);
    dumper->Print("", decoder_feeds[offset]);
    dumper->Print("beam_width", offset + 1, true);
    dumper->Print("", decoder_feeds[offset + 1]);
    dumper->Print("cache_redir", offset + 2, true);
    dumper->Print("", decoder_feeds[offset + 2]);
#endif

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
    const_cast<SessionState&>(this->decoder_session_state_).IncrementGraphExecutionCounter();
#endif
    status = utils::ExecuteSubgraph(this->decoder_session_state_,
                                    decoder_feeds_fetches_manager,
                                    decoder_feeds,
                                    decoder_fetches,
                                    {},
                                    ExecutionMode::ORT_SEQUENTIAL,
                                    this->context_.GetTerminateFlag(),
                                    this->context_.Logger(),
                                    this->ort_stream_);

    ORT_RETURN_IF_ERROR(status);

#ifdef DEBUG_GENERATION
    for (int i = 0; i <= decoder_subgraph_.GetFirstPresentOutputIndex(); i++) {
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
      gsl::span<const int32_t> place_holder;
      const int num_present_outputs = 2 * parameters->num_layers;  // number of outputs with name like present_*
      ORT_RETURN_IF_ERROR(this->update_decoder_feeds_func_(
          this->temp_space_allocator_,
          this->ort_stream_,
          decoder_fetches,
          decoder_feeds,
          num_present_outputs,
          ReinterpretAsSpan<const int32_t>(beam_next_tokens),
          ReinterpretAsSpan<const int32_t>(beam_indices),
          decoder_subgraph_.has_decoder_masked_attention_
              ? ReinterpretAsSpan<const int32_t>(beam_state.chosen_indices)
              : place_holder,
          parameters->num_beams,
          decoder_subgraph_.GetFirstPastInputIndex(),
          decoder_subgraph_.GetFirstPresentOutputIndex(),
          decoder_subgraph_.UseSequenceAsInputIds(),
          current_length,
          parameters->sequence_length,
          decoder_subgraph_.past_present_share_buffer_,
          decoder_subgraph_.has_decoder_masked_attention_,
          cpu_state.sequences,
          this->GetConsoleDumper()));
    }

    if (decoder_subgraph_.past_present_share_buffer_) {
      // clear fetched values before presents[]
      for (int idx = 0; idx < decoder_subgraph_.GetFirstPresentOutputIndex(); idx++) {
        decoder_fetches[idx] = OrtValue();
      }
    } else {
      decoder_fetches.clear();
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
