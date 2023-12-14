// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/span_utils.h"
#include "contrib_ops/cpu/transformers/generation_shared.h"  // for DEBUG_GENERATION
#include "contrib_ops/cpu/transformers/beam_search_impl_base.h"
#include "contrib_ops/cpu/transformers/subgraph_whisper_encoder.h"
#include "contrib_ops/cpu/transformers/subgraph_whisper_decoder.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

// Beam search implementation for Whisper model.
template <typename T>
class BeamSearchWhisper : public BeamSearchBase<T> {
 public:
  BeamSearchWhisper(OpKernelContextInternal& context,
                    const SessionState& encoder_session_state,
                    const SessionState& decoder_session_state,
                    WhisperEncoderSubgraph& encoder_subgraph,
                    WhisperDecoderSubgraph& decoder_subgraph,
                    concurrency::ThreadPool* thread_pool,
                    Stream* ort_stream,
                    IConsoleDumper* cuda_dumper,
                    BeamSearchParameters& params,
                    const GenerationDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
                    const GenerationDeviceHelper::TopkFunc& topk_func,
                    const GenerationDeviceHelper::ProcessLogitsFunc<T>& process_logits_func,
                    const GenerationDeviceHelper::InitBeamStateFunc<T>& init_beam_state_func,
                    const GenerationDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
                    const GenerationDeviceHelper::DeviceCopyFunc<int32_t>& device_copy_int32_func,
                    const GenerationDeviceHelper::CreateWhisperEncoderInputsFunc& create_encoder_inputs_func,
                    const GenerationDeviceHelper::UpdateDecoderFeedsFunc<T>& update_decoder_feeds_func,
                    const GenerationDeviceHelper::ExpandBufferFunc<float>& expand_buffer_float_func,
                    const GenerationDeviceHelper::ExpandBufferFunc<MLFloat16>& expand_buffer_float16_func,
                    const GenerationDeviceHelper::CreateBeamScorer& create_beam_scorer_func,
                    const GenerationDeviceHelper::UpdateDecoderCrossQKFunc& update_decoder_cross_qk_func,
                    const GenerationDeviceHelper::FinalizeDecoderCrossQKFunc& finalize_decoder_cross_qk_func)
      : BeamSearchBase<T>(context, decoder_session_state, thread_pool,
                          ort_stream, cuda_dumper, params,
                          topk_func, process_logits_func, device_copy_func, device_copy_int32_func),
        encoder_session_state_(encoder_session_state),
        encoder_subgraph_(encoder_subgraph),
        decoder_subgraph_(decoder_subgraph),
        add_to_feeds_func_(add_to_feeds_func),
        init_beam_state_func_(init_beam_state_func),
        create_encoder_inputs_func_(create_encoder_inputs_func),
        update_decoder_feeds_func_(update_decoder_feeds_func),
        expand_buffer_float_func_(expand_buffer_float_func),
        expand_buffer_float16_func_(expand_buffer_float16_func),
        create_beam_scorer_func_(create_beam_scorer_func),
        update_decoder_cross_qk_func_(update_decoder_cross_qk_func),
        finalize_decoder_cross_qk_func_(finalize_decoder_cross_qk_func),
        cuda_device_prop_(nullptr),
        cuda_device_arch_(0) {}

#ifdef USE_CUDA
  Status InitializeCuda(
      const GenerationDeviceHelper::ReorderPastStateFunc& reorder_past_state_func,
      const GenerationDeviceHelper::InitCacheIndirFunc& init_cache_indir_func,
      const void* cuda_device_prop,
      int cuda_device_arch) {
    reorder_past_state_func_ = reorder_past_state_func;
    init_cache_indir_func_ = init_cache_indir_func;
    cuda_device_prop_ = cuda_device_prop;
    cuda_device_arch_ = cuda_device_arch;
    if (decoder_subgraph_.has_decoder_masked_attention_) {
      ORT_RETURN_IF_NOT(cuda_device_arch_ >= 530,
                        "Decoder masked multihead attention can only be used on "
                        "GPU cards of compute capability 5.3 or higher. "
                        "This card has compute capability ",
                        cuda_device_arch_);
    }
    return Status::OK();
  }
#endif

  // Execute beam search in iterations util stopping criteria is reached.
  Status Execute(const FeedsFetchesManager& encoder_feeds_fetches_manager,
                 const FeedsFetchesManager& decoder_feeds_fetches_manager);

 private:
  const SessionState& encoder_session_state_;

  WhisperEncoderSubgraph& encoder_subgraph_;
  WhisperDecoderSubgraph& decoder_subgraph_;

  // Device specific functions
  GenerationDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  GenerationDeviceHelper::InitBeamStateFunc<T> init_beam_state_func_;
#ifdef USE_CUDA
  GenerationDeviceHelper::ReorderPastStateFunc reorder_past_state_func_;
  GenerationDeviceHelper::InitCacheIndirFunc init_cache_indir_func_;
#endif
  GenerationDeviceHelper::CreateWhisperEncoderInputsFunc create_encoder_inputs_func_;
  GenerationDeviceHelper::UpdateDecoderFeedsFunc<T> update_decoder_feeds_func_;
  GenerationDeviceHelper::ExpandBufferFunc<float> expand_buffer_float_func_;
  GenerationDeviceHelper::ExpandBufferFunc<MLFloat16> expand_buffer_float16_func_;
  GenerationDeviceHelper::CreateBeamScorer create_beam_scorer_func_;

  const GenerationDeviceHelper::UpdateDecoderCrossQKFunc update_decoder_cross_qk_func_;
  const GenerationDeviceHelper::FinalizeDecoderCrossQKFunc finalize_decoder_cross_qk_func_;
  const void* cuda_device_prop_ = nullptr;
  int cuda_device_arch_ = 0;
};

template <typename T>
Status BeamSearchWhisper<T>::Execute(const FeedsFetchesManager& encoder_feeds_fetches_manager,
                                     const FeedsFetchesManager& decoder_feeds_fetches_manager) {
  auto status = Status::OK();

  const BeamSearchParameters* parameters = this->parameters_;

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

  if (parameters->no_speech_probs_output_id > 0) {
    TensorShape no_speech_probs_shape{parameters->batch_size};
    Tensor* no_speech_probs = this->context_.Output(parameters->no_speech_probs_output_id, no_speech_probs_shape);
    if (no_speech_probs && no_speech_probs->MutableData<T>()) {
      ORT_ENFORCE(parameters->no_speech_token >= 0 && parameters->no_speech_token < parameters->vocab_size,
                  "no_speech_token id out of range, it is ", parameters->no_speech_token,
                  ", vocab_size is ", parameters->vocab_size);
      this->parameters_->no_speech_probs = (void*)no_speech_probs->MutableData<T>();
    }
  }

  // Update the flag to indicate whether scores exists in output
  this->parameters_->output_scores = (output_scores != nullptr);

  // ------------------------------------
  // Call encoder subgraph.
  // ------------------------------------
  std::vector<OrtValue> encoder_feeds;
  std::vector<OrtValue> encoder_fetches;

  const OrtValue* encoder_input_ids_value = this->context_.GetInputOrtValue(0);
  const Tensor& encoder_input_ids = encoder_input_ids_value->Get<Tensor>();

  BeamSearchCpuState cpu_state{*parameters,
                               this->cpu_allocator_,
                               this->IsCuda(),
                               this->ort_stream_};

  IAllocatorUniquePtr<char> buffer;

  const OrtValue* initial_decoder_input_ids_value = this->context_.GetInputOrtValue(10);

  OrtValue decoder_input_ids;  // Tensor in CPU, and it will be used to initialize sequence in cpu_state
  ORT_RETURN_IF_ERROR(this->encoder_subgraph_.CreateInitialFeeds(
      encoder_input_ids,
      initial_decoder_input_ids_value,
      parameters->decoder_start_token_id,
      this->implicit_inputs_,
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

  BeamSearchState<T> beam_state{*parameters,
                                this->temp_space_allocator_,
                                decoder_subgraph_.has_decoder_masked_attention_,
                                false /* use_position */,
                                this->ort_stream_};

  init_beam_state_func_(&beam_state,
                        cpu_state.sequence_lengths,
                        parameters->batch_size,
                        parameters->num_beams,
                        this->ort_stream_);

  // Copy decoder_input_ids (in CPU) to sequence. It contains the initial decoder token ids for each beam.
  cpu_state.SetUnexpandedSequence(decoder_input_ids.Get<Tensor>().DataAsSpan<int32_t>());

  // beam_state.sequences_device is the GPU version of cpu_state.sequences_space,
  // this copies it over to the GPU after setting it up on the CPU
  if (this->IsCuda()) {
    cpu_state.sequences.InitDevice(beam_state.sequences_device);
    ORT_RETURN_IF_ERROR(this->device_copy_int32_func_(beam_state.sequences_device.subspan(0, beam_state.sequences_device.size() / 2),
                                                      cpu_state.sequences_space.subspan(0, cpu_state.sequences_space.size() / 2),
                                                      nullptr,
                                                      DeviceCopyDirection::hostToDevice));
  }

  this->beam_scorer_ = create_beam_scorer_func_
                           ? create_beam_scorer_func_(*parameters, this->temp_space_allocator_, this->cpu_allocator_, this->ort_stream_)
                           : std::make_unique<BeamSearchScorer>(*parameters, this->cpu_allocator_);

  // ------------------------------------------------------------------------------
  // Generate next token from logits output from encoder, and initialize decoder inputs.
  // ------------------------------------------------------------------------------
  gsl::span<int32_t> beam_next_tokens;

  int iteration_counter = 0;
  std::vector<OrtValue> decoder_feeds;
  int current_length = parameters->sequence_length;

  // for decoder subgraph output cross qk
  int64_t frames_of_k = 0LL;
  Tensor* cross_qk_output = nullptr;  // output tensor
  int64_t cross_qk_layer_head_pair_count = 0LL;
  OrtValue cross_qk_buffer_value;
  float* cross_qk_buffer_data = nullptr;
  std::vector<int32_t> cross_qk_all_layer_heads;
  const int32_t* cross_qk_layer_head_pairs = nullptr;
  IAllocatorUniquePtr<float*> qk_layer_pointers;  // if needed, device array hold the cross qk data pointers, shape of [num_layers]

  std::vector<OrtValue> decoder_fetches;

  if (current_length + 1 < parameters->max_length) {
    ++iteration_counter;
    ORT_RETURN_IF_ERROR(this->GenerateNextToken(encoder_fetches[0],
                                                beam_next_tokens,
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
      decoder_fetches.reserve(static_cast<size_t>(decoder_subgraph_.GetFirstPresentOutputIndex()) +
                              2 * static_cast<size_t>(decoder_subgraph_.num_layers));
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

    if (decoder_subgraph_.output_cross_qk_) {
      ORT_ENFORCE(decoder_subgraph_.has_decoder_masked_attention_, "decoder subgraph: output_cross_qk could only work with has_decoder_masked_attention");
      ORT_ENFORCE(decoder_subgraph_.past_present_share_buffer_, "decoder subgraph: output_cross_qk could only work with past_present_share_buffer");

      cross_qk_layer_head_pair_count = parameters->num_layers * parameters->num_heads;
      const auto* input_tensor_cross_qk_layer_head = this->context_.template Input<Tensor>(parameters->cross_qk_layer_head_input_id);
      ORT_ENFORCE(input_tensor_cross_qk_layer_head != nullptr, "Must specify input cross_qk_layer_head");
      cross_qk_layer_head_pair_count = input_tensor_cross_qk_layer_head->Shape()[0];
      cross_qk_layer_head_pairs = input_tensor_cross_qk_layer_head->template Data<int32_t>();  // it is on GPU

      size_t decoder_input_first_cross_key = static_cast<size_t>(decoder_subgraph_.GetFirstPastInputIndex()) + (2 * decoder_subgraph_.num_layers);
      auto first_cross_attention_key = decoder_feeds[decoder_input_first_cross_key].GetMutable<Tensor>();
      frames_of_k = first_cross_attention_key->Shape()[2];

      TensorShape layer_cross_qk_shape{
          static_cast<int64_t>(parameters->BatchBeamSize()),
          static_cast<int64_t>(parameters->num_heads),
          1LL,
          static_cast<int64_t>(frames_of_k)};
      for (int layer = 0; layer < decoder_subgraph_.num_layers; layer++) {
        OrtValue cross_qk_value;
        Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), layer_cross_qk_shape, this->temp_space_allocator_, cross_qk_value);
        decoder_fetches.emplace_back(cross_qk_value);
      }

      TensorShape cross_qk_shape{
          static_cast<int64_t>(parameters->batch_size),
          static_cast<int64_t>(parameters->num_beams),
          cross_qk_layer_head_pair_count,
          static_cast<int64_t>(parameters->max_length),
          frames_of_k};
      Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), cross_qk_shape, this->temp_space_allocator_, cross_qk_buffer_value);
      cross_qk_buffer_data = cross_qk_buffer_value.GetMutable<Tensor>()->MutableData<float>();
    }

    if (decoder_subgraph_.has_decoder_masked_attention_) {
      size_t offset = static_cast<size_t>(decoder_subgraph_.GetFirstPastInputIndex());
      // Need to check cross attention's past key tensor size, suppose all layers cross attention key size are same
      auto first_cross_attention_key = decoder_feeds[offset + 2 * static_cast<size_t>(decoder_subgraph_.num_layers)].GetMutable<Tensor>();
      auto cross_attention_past_key_sz = first_cross_attention_key->Shape().Size();
      beam_state.EnsurePastStateReorderStagingBuffer(this->temp_space_allocator_, cross_attention_past_key_sz);

#ifdef USE_CUDA
      // Here we only need to reorder the past key for self-attention and cross-attention.
      for (size_t i = 0; i < 2 * static_cast<size_t>(decoder_subgraph_.num_layers); ++i) {
        ORT_RETURN_IF_ERROR(reorder_past_state_func_(cuda_device_prop_,
                                                     *decoder_feeds[offset + 2 * i].GetMutable<Tensor>(),
                                                     beam_state.staging_for_past_state_reorder,
                                                     this->ort_stream_));
      }
      size_t cache_indir_input_offset = static_cast<size_t>(decoder_subgraph_.GetFirstPastInputIndex()) + 4 * static_cast<size_t>(decoder_subgraph_.num_layers) + 2;
      ORT_RETURN_IF_ERROR(init_cache_indir_func_(*decoder_feeds[cache_indir_input_offset].GetMutable<Tensor>(), this->ort_stream_));
#endif
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

    if (decoder_subgraph_.output_cross_qk_) {
      int decoder_output_first_cross_qk = decoder_subgraph_.GetFirstPresentOutputIndex() + (2 * decoder_subgraph_.num_layers);
      ORT_RETURN_IF_ERROR(this->update_decoder_cross_qk_func_(
          iteration_counter,
          this->ort_stream_,
          &decoder_fetches[decoder_output_first_cross_qk],
          qk_layer_pointers,
          parameters->num_layers,
          static_cast<int>(cross_qk_layer_head_pair_count),
          cross_qk_layer_head_pairs,
          cross_qk_buffer_data,
          parameters->max_length,
          this->temp_space_allocator_));
    }

#ifdef DEBUG_GENERATION
    for (int i = 0; i <= decoder_subgraph_.GetFirstPresentOutputIndex(); i++) {
      dumper->Print("decoder_fetches", i, true);
      dumper->Print("", decoder_fetches[i]);
    }
#endif

    const OrtValue& logits = decoder_fetches[0];
    ORT_RETURN_IF_ERROR(this->GenerateNextToken(logits,
                                                beam_next_tokens,
                                                beam_state,
                                                cpu_state,
                                                iteration_counter));

    // When all batches are finished, stop earlier to avoid wasting computation.
    if (this->beam_scorer_->IsDone()) {
      break;
    }

    // TODO: If this is safe to do after update_decoder_feeds_func, move it later so that we can speculatively run the next steps while we wait
    // for the done result to transfer to the CPU
    if (this->beam_scorer_->IsDoneLater()) {
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
          decoder_subgraph_.has_decoder_masked_attention_
              ? place_holder
              : ReinterpretAsSpan<const int32_t>(this->beam_scorer_->GetNextIndicesCPU()),
          decoder_subgraph_.has_decoder_masked_attention_
              ? ReinterpretAsSpan<const int32_t>(this->beam_scorer_->GetNextIndicesGPU())
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

  if (decoder_subgraph_.output_cross_qk_) {
    TensorShape cross_qk_shape{
        static_cast<int64_t>(parameters->batch_size),
        static_cast<int64_t>(parameters->num_return_sequences),
        cross_qk_layer_head_pair_count,
        static_cast<int64_t>(iteration_counter - 1),
        frames_of_k};
    cross_qk_output = this->context_.Output(parameters->cross_qk_output_id, cross_qk_shape);

    size_t cache_indir_input_offset = static_cast<size_t>(decoder_subgraph_.GetFirstPastInputIndex()) + 4 * static_cast<size_t>(decoder_subgraph_.num_layers) + 2;
    const int* cache_indir_data = decoder_feeds[cache_indir_input_offset].GetMutable<Tensor>()->Data<int32_t>();
    auto beam_indices = this->beam_scorer_->GetNextIndicesGPU();  // currently only support on GPU
    ORT_RETURN_IF_ERROR(this->finalize_decoder_cross_qk_func_(
        this->ort_stream_,
        iteration_counter,
        parameters->sequence_length,
        parameters->batch_size,
        parameters->num_beams,
        parameters->max_length,
        static_cast<int>(cross_qk_layer_head_pair_count),
        cross_qk_layer_head_pairs,
        static_cast<int>(frames_of_k),
        cross_qk_buffer_data,
        cross_qk_output->MutableData<float>(),
        parameters->num_return_sequences,
        cache_indir_data,
        beam_indices));
  }

  gsl::span<const float> final_beam_scores = beam_state.beam_scores;
  this->beam_scorer_->Finalize(cpu_state.sequences,
                               final_beam_scores,
                               output_sequences,
                               output_sequences_scores);

  // Output per token scores
  gsl::span<const float> per_token_scores = beam_state.scores;
  this->beam_scorer_->OutputScores(per_token_scores, output_scores);

  return status;
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
