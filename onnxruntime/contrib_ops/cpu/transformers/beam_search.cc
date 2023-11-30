// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include <memory>
#include <assert.h>
#include <functional>
#include "core/common/safeint.h"
#include "core/providers/cpu/math/top_k.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/framework/allocator.h"
#include "core/framework/framework_common.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/framework/session_options.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/allocator.h"
#include "core/framework/ort_value.h"
#include "core/common/gsl.h"
#include "contrib_ops/cpu/transformers/beam_search.h"
#include "contrib_ops/cpu/transformers/logits_processor.h"
#include "contrib_ops/cpu/transformers/sequences.h"
#include "contrib_ops/cpu/transformers/dump_tensor.h"
#include "contrib_ops/cpu/transformers/beam_search_scorer.h"
#include "contrib_ops/cpu/transformers/beam_search_impl_gpt.h"
#include "contrib_ops/cpu/transformers/beam_search_impl_t5.h"
#include "contrib_ops/cpu/transformers/beam_search_impl_whisper.h"
#include "contrib_ops/cpu/transformers/greedy_search_impl_gpt.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      BeamSearch,                                                 \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      transformers::BeamSearch);                                  \
                                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      WhisperBeamSearch,                                          \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      transformers::WhisperBeamSearch);

REGISTER_KERNEL_TYPED(float)

namespace transformers {

void BeamSearch::Init(const OpKernelInfo& info) {
  parameters_->ParseFromAttributes(info);

  // Model_type could be either 0 (GPT-2), 1 (encoder-decoder like T5), or 2 (Whisper)
  ORT_ENFORCE(parameters_->model_type == IGenerationParameters::kModelTypeGpt ||
              parameters_->model_type == IGenerationParameters::kModelTypeT5 ||
              parameters_->model_type == IGenerationParameters::kModelTypeWhisper);

  ONNX_NAMESPACE::GraphProto proto;

  if (parameters_->model_type != IGenerationParameters::kModelTypeGpt) {
    // Make sure the encoder sub-graph attribute is present for the T5 and Whisper models.
    ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("encoder", &proto).IsOK());
  }

  if (parameters_->model_type == IGenerationParameters::kModelTypeGpt) {
    // Check if the init_decoder sub-graph attribute is present for the GPT2 model.
    if (info.GetAttr<ONNX_NAMESPACE::GraphProto>("init_decoder", &proto).IsOK()) {
      has_init_decoder_ = true;
    }
  }

  // Make sure the decoder sub-graph attribute is present for all model types.
  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("decoder", &proto).IsOK());

  ORT_IGNORE_RETURN_VALUE(proto);
}

Status BeamSearch::SetupSubgraphExecutionInfo(const SessionState& session_state,
                                              const std::string& attribute_name,
                                              const SessionState& subgraph_session_state) {
  const auto& node = Node();
  if (parameters_->model_type == IGenerationParameters::kModelTypeGpt) {
    if (attribute_name == "decoder") {
      ORT_ENFORCE(gpt_subgraph_ == nullptr, "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
      auto res = gpt_details::CreateGptSubgraphAndUpdateParameters(node, session_state, attribute_name,
                                                                   subgraph_session_state, *parameters_);

      auto status = res.first;
      if (!status.IsOK()) {
        return status;
      }

      gpt_subgraph_ = std::move(res.second);
      decoder_feeds_fetches_manager_ = gpt_subgraph_->GetFeedsFetchesManager();
    } else if (attribute_name == "init_decoder") {
      ORT_ENFORCE(init_run_gpt_subgraph_ == nullptr, "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
      // TODO (hasesh): If 'init_decoder' is present, then we update 'parameters_' again based on its subgraph (it would have been
      // updated once for the 'decoder' attribute). In future, find a way to update 'parameters' only once based on only one subgraph
      // attribute.
      auto res = gpt_details::CreateGptSubgraphAndUpdateParameters(node, session_state, attribute_name,
                                                                   subgraph_session_state, *parameters_);

      auto status = res.first;
      if (!status.IsOK()) {
        return status;
      }

      init_run_gpt_subgraph_ = std::move(res.second);
      init_run_decoder_feeds_fetches_manager_ = init_run_gpt_subgraph_->GetFeedsFetchesManager();
    }
  } else if (parameters_->model_type == IGenerationParameters::kModelTypeT5) {
    if (attribute_name == "encoder") {
      ORT_ENFORCE(t5_encoder_subgraph_ == nullptr,
                  "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
      t5_encoder_subgraph_ = std::make_unique<T5EncoderSubgraph>(node,
                                                                 attribute_name,
                                                                 subgraph_session_state.GetGraphViewer());
      ORT_RETURN_IF_ERROR(t5_encoder_subgraph_->Setup(session_state, subgraph_session_state));
      encoder_feeds_fetches_manager_ = t5_encoder_subgraph_->GetFeedsFetchesManager();

      if (parameters_->decoder_start_token_id < 0) {
        ORT_RETURN_IF(t5_encoder_subgraph_->num_subgraph_inputs != 2,
                      "Encoder subgraph shall have 2 inputs when decoder_start_token_id attribute is empty");
      } else {
        ORT_RETURN_IF(t5_encoder_subgraph_->num_subgraph_inputs != 3 && t5_encoder_subgraph_->num_subgraph_inputs != 4,
                      "Encoder subgraph shall have 3 or 4 inputs when decoder_start_token_id attribute is available");
      }
    } else if (attribute_name == "decoder") {
      ORT_ENFORCE(t5_decoder_subgraph_ == nullptr,
                  "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
      t5_decoder_subgraph_ = std::make_unique<T5DecoderSubgraph>(node,
                                                                 attribute_name,
                                                                 subgraph_session_state.GetGraphViewer());
      ORT_RETURN_IF_ERROR(t5_decoder_subgraph_->Setup(session_state, subgraph_session_state));
      decoder_feeds_fetches_manager_ = t5_decoder_subgraph_->GetFeedsFetchesManager();
      parameters_->SetSubgraphParameters(t5_decoder_subgraph_->vocab_size,
                                         t5_decoder_subgraph_->num_heads,
                                         t5_decoder_subgraph_->head_size,
                                         t5_decoder_subgraph_->num_layers);
    }
  } else if (parameters_->model_type == IGenerationParameters::kModelTypeWhisper) {
    if (attribute_name == "encoder") {
      ORT_ENFORCE(whisper_encoder_subgraph_ == nullptr,
                  "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
      whisper_encoder_subgraph_ = std::make_unique<WhisperEncoderSubgraph>(node,
                                                                           attribute_name,
                                                                           subgraph_session_state.GetGraphViewer());
      ORT_RETURN_IF_ERROR(whisper_encoder_subgraph_->Setup(session_state, subgraph_session_state));
      encoder_feeds_fetches_manager_ = whisper_encoder_subgraph_->GetFeedsFetchesManager();

      ORT_RETURN_IF(whisper_encoder_subgraph_->num_subgraph_inputs != 2,
                    "Encoder subgraph shall have 2 inputs (encoder_input_ids, decoder_input_ids)");
    } else if (attribute_name == "decoder") {
      ORT_ENFORCE(whisper_decoder_subgraph_ == nullptr,
                  "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
      whisper_decoder_subgraph_ = std::make_unique<WhisperDecoderSubgraph>(node,
                                                                           attribute_name,
                                                                           subgraph_session_state.GetGraphViewer());
      ORT_RETURN_IF_ERROR(whisper_decoder_subgraph_->Setup(session_state, subgraph_session_state));
      decoder_feeds_fetches_manager_ = whisper_decoder_subgraph_->GetFeedsFetchesManager();
      parameters_->SetSubgraphParameters(whisper_decoder_subgraph_->vocab_size,
                                         whisper_decoder_subgraph_->num_heads,
                                         whisper_decoder_subgraph_->head_size,
                                         whisper_decoder_subgraph_->num_layers);
    }
  }

  return Status::OK();
}

Status BeamSearch::Compute(OpKernelContext* ctx) const {
  auto* ctx_internal = static_cast<OpKernelContextInternal*>(ctx);

  auto* decoder_session_state = ctx_internal->SubgraphSessionState("decoder");
  ORT_ENFORCE(decoder_session_state, "Subgraph SessionState was not found for 'decoder' attribute.");
  ORT_ENFORCE(decoder_feeds_fetches_manager_, "CreateFeedsFetchesManager must be called prior to execution of graph.");

  auto* init_run_decoder_session_state = ctx_internal->SubgraphSessionState("init_decoder");
  if (has_init_decoder_) {
    ORT_ENFORCE(init_run_decoder_session_state, "Subgraph SessionState was not found for 'decoder' attribute.");
    ORT_ENFORCE(init_run_decoder_feeds_fetches_manager_, "CreateFeedsFetchesManager must be called prior to execution of graph.");
    ORT_ENFORCE(init_run_gpt_subgraph_ && gpt_subgraph_ && init_run_gpt_subgraph_->past_present_share_buffer_ == gpt_subgraph_->past_present_share_buffer_,
                "past_present_share_buffer mode must be same for init decoder and decoder subgraphes");
  }

  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  // Make a copy of parameters since we will update it based on inputs later
  BeamSearchParameters parameters = *parameters_;

  if (parameters.model_type == IGenerationParameters::kModelTypeGpt) {
    if (!gpt_subgraph_->IsOutputFloat16()) {  // Output float32
      BeamSearchGpt<float> impl{
          *ctx_internal,
          has_init_decoder_ ? init_run_decoder_session_state : nullptr,
          has_init_decoder_ ? init_run_gpt_subgraph_.get() : nullptr,
          *decoder_session_state,
          *gpt_subgraph_,
          thread_pool, ctx->GetComputeStream(), dumper_, parameters,
          GenerationCpuDeviceHelper::CreateGptInputs,
          add_to_feeds_func_ ? add_to_feeds_func_ : GenerationCpuDeviceHelper::AddToFeeds,
          topk_func_ ? topk_func_ : GenerationCpuDeviceHelper::TopK,
          process_logits_func_ ? process_logits_func_ : GenerationCpuDeviceHelper::ProcessLogits<float>,
          init_beam_state_func_ ? init_beam_state_func_ : GenerationCpuDeviceHelper::InitBeamState<float>,
          device_copy_func_ ? device_copy_func_ : GenerationCpuDeviceHelper::DeviceCopy<float>,
          device_copy_int32_func_ ? device_copy_int32_func_ : GenerationCpuDeviceHelper::DeviceCopy<int32_t>,
          update_gpt_feeds_func_ ? update_gpt_feeds_func_ : GenerationCpuDeviceHelper::UpdateGptFeeds<float>,
          create_beam_scorer_func_};
#ifdef USE_CUDA
      ORT_RETURN_IF_ERROR(impl.InitializeCuda(reorder_past_state_func_, cuda_device_prop_, cuda_device_arch_));
#endif
      ORT_RETURN_IF_ERROR(impl.Initialize());

      return impl.Execute(init_run_decoder_feeds_fetches_manager_, *decoder_feeds_fetches_manager_);
    } else {  // Output float16
      BeamSearchGpt<MLFloat16> impl{
          *ctx_internal,
          has_init_decoder_ ? init_run_decoder_session_state : nullptr,
          has_init_decoder_ ? init_run_gpt_subgraph_.get() : nullptr,
          *decoder_session_state,
          *gpt_subgraph_,
          thread_pool, ctx->GetComputeStream(), dumper_, parameters,
          GenerationCpuDeviceHelper::CreateGptInputs,
          add_to_feeds_func_ ? add_to_feeds_func_ : GenerationCpuDeviceHelper::AddToFeeds,
          topk_func_ ? topk_func_ : GenerationCpuDeviceHelper::TopK,
          process_logits_fp16_func_,
          init_beam_state_fp16_func_,
          device_copy_func_,
          device_copy_int32_func_,
          update_gpt_feeds_fp16_func_,
          create_beam_scorer_func_};
#ifdef USE_CUDA
      ORT_RETURN_IF_ERROR(impl.InitializeCuda(reorder_past_state_func_, cuda_device_prop_, cuda_device_arch_));
#endif
      ORT_RETURN_IF_ERROR(impl.Initialize());

      return impl.Execute(init_run_decoder_feeds_fetches_manager_, *decoder_feeds_fetches_manager_);
    }
  }

  auto* encoder_session_state = ctx_internal->SubgraphSessionState("encoder");
  ORT_ENFORCE(encoder_session_state, "Subgraph SessionState was not found for 'encoder' attribute.");
  ORT_ENFORCE(encoder_feeds_fetches_manager_, "CreateFeedsFetchesManager must be called prior to execution of graph.");

  if (parameters.model_type == IGenerationParameters::kModelTypeT5) {
    // Subgraph has constraint that the output is either float or float16
    if (!t5_decoder_subgraph_->IsOutputFloat16()) {
      BeamSearchT5<float> impl{
          *ctx_internal, *encoder_session_state, *decoder_session_state, *t5_encoder_subgraph_,
          *t5_decoder_subgraph_, thread_pool, ctx->GetComputeStream(), dumper_, parameters,
          add_to_feeds_func_ ? add_to_feeds_func_ : GenerationCpuDeviceHelper::AddToFeeds,
          topk_func_ ? topk_func_ : GenerationCpuDeviceHelper::TopK,
          process_logits_func_ ? process_logits_func_ : GenerationCpuDeviceHelper::ProcessLogits<float>,
          init_beam_state_func_ ? init_beam_state_func_ : GenerationCpuDeviceHelper::InitBeamState<float>,
          device_copy_func_ ? device_copy_func_ : GenerationCpuDeviceHelper::DeviceCopy<float>,
          device_copy_int32_func_ ? device_copy_int32_func_ : GenerationCpuDeviceHelper::DeviceCopy<int32_t>,
          create_encoder_inputs_func_ ? create_encoder_inputs_func_ : GenerationCpuDeviceHelper::CreateEncoderInputs<float>,
          update_decoder_feeds_func_ ? update_decoder_feeds_func_ : GenerationCpuDeviceHelper::UpdateDecoderFeeds<float>,
          expand_buffer_int32_func_ ? expand_buffer_int32_func_ : GenerationCpuDeviceHelper::ExpandBuffer<int32_t>,
          expand_buffer_float_func_ ? expand_buffer_float_func_ : GenerationCpuDeviceHelper::ExpandBuffer<float>,
          expand_buffer_float16_func_ ? expand_buffer_float16_func_ : GenerationCpuDeviceHelper::ExpandBuffer<MLFloat16>,
          create_beam_scorer_func_};
#ifdef USE_CUDA
      ORT_RETURN_IF_ERROR(impl.InitializeCuda(reorder_past_state_func_, init_cache_indir_func_, cuda_device_prop_, cuda_device_arch_));
#endif
      ORT_RETURN_IF_ERROR(impl.Initialize());

      return impl.Execute(*encoder_feeds_fetches_manager_, *decoder_feeds_fetches_manager_);
    } else {
      BeamSearchT5<MLFloat16> impl{
          *ctx_internal, *encoder_session_state, *decoder_session_state, *t5_encoder_subgraph_,
          *t5_decoder_subgraph_, thread_pool, ctx->GetComputeStream(), dumper_, parameters,
          add_to_feeds_func_ ? add_to_feeds_func_ : GenerationCpuDeviceHelper::AddToFeeds,
          topk_func_ ? topk_func_ : GenerationCpuDeviceHelper::TopK,
          process_logits_fp16_func_,
          init_beam_state_fp16_func_,
          device_copy_func_,
          device_copy_int32_func_,
          create_encoder_inputs_func_ ? create_encoder_inputs_func_ : GenerationCpuDeviceHelper::CreateEncoderInputs<MLFloat16>,
          update_decoder_feeds_fp16_func_,
          expand_buffer_int32_func_,
          expand_buffer_float_func_,
          expand_buffer_float16_func_,
          create_beam_scorer_func_};
#ifdef USE_CUDA
      ORT_RETURN_IF_ERROR(impl.InitializeCuda(reorder_past_state_func_, init_cache_indir_func_, cuda_device_prop_, cuda_device_arch_));
#endif
      ORT_RETURN_IF_ERROR(impl.Initialize());

      return impl.Execute(*encoder_feeds_fetches_manager_, *decoder_feeds_fetches_manager_);
    }
  }

  // Change the CreateEncoderInputs function for Whisper shapes
  if (parameters.model_type == IGenerationParameters::kModelTypeWhisper) {
    // Subgraph has constraint that the output is either float or float16
    if (!whisper_decoder_subgraph_->IsOutputFloat16()) {
      BeamSearchWhisper<float> impl{
          *ctx_internal, *encoder_session_state, *decoder_session_state, *whisper_encoder_subgraph_,
          *whisper_decoder_subgraph_, thread_pool, ctx->GetComputeStream(), dumper_, parameters,
          add_to_feeds_func_ ? add_to_feeds_func_ : GenerationCpuDeviceHelper::AddToFeeds,
          topk_func_ ? topk_func_ : GenerationCpuDeviceHelper::TopK,
          process_logits_func_ ? process_logits_func_ : GenerationCpuDeviceHelper::ProcessLogits<float>,
          init_beam_state_func_ ? init_beam_state_func_ : GenerationCpuDeviceHelper::InitBeamState<float>,
          device_copy_func_ ? device_copy_func_ : GenerationCpuDeviceHelper::DeviceCopy<float>,
          device_copy_int32_func_ ? device_copy_int32_func_ : GenerationCpuDeviceHelper::DeviceCopy<int32_t>,
          create_whisper_encoder_inputs_func_ ? create_whisper_encoder_inputs_func_ : GenerationCpuDeviceHelper::CreateWhisperEncoderInputs<float>,
          update_decoder_feeds_func_ ? update_decoder_feeds_func_ : GenerationCpuDeviceHelper::UpdateDecoderFeeds<float>,
          expand_buffer_float_func_ ? expand_buffer_float_func_ : GenerationCpuDeviceHelper::ExpandBuffer<float>,
          expand_buffer_float16_func_ ? expand_buffer_float16_func_ : GenerationCpuDeviceHelper::ExpandBuffer<MLFloat16>,
          create_beam_scorer_func_,
          update_decoder_cross_qk_func_ ? update_decoder_cross_qk_func_ : GenerationCpuDeviceHelper::UpdateDecoderCrossQK,
          finalize_decoder_cross_qk_func_ ? finalize_decoder_cross_qk_func_ : GenerationCpuDeviceHelper::FinalizeDecoderCrossQK};

#ifdef USE_CUDA
      ORT_RETURN_IF_ERROR(impl.InitializeCuda(reorder_past_state_func_, init_cache_indir_func_, cuda_device_prop_, cuda_device_arch_));
#endif
      ORT_RETURN_IF_ERROR(impl.Initialize());

      return impl.Execute(*encoder_feeds_fetches_manager_, *decoder_feeds_fetches_manager_);
    } else {
      BeamSearchWhisper<MLFloat16> impl{
          *ctx_internal, *encoder_session_state, *decoder_session_state, *whisper_encoder_subgraph_,
          *whisper_decoder_subgraph_, thread_pool, ctx->GetComputeStream(), dumper_, parameters,
          add_to_feeds_func_ ? add_to_feeds_func_ : GenerationCpuDeviceHelper::AddToFeeds,
          topk_func_ ? topk_func_ : GenerationCpuDeviceHelper::TopK,
          process_logits_fp16_func_,
          init_beam_state_fp16_func_,
          device_copy_func_,
          device_copy_int32_func_,
          create_whisper_encoder_inputs_func_ ? create_whisper_encoder_inputs_func_ : GenerationCpuDeviceHelper::CreateWhisperEncoderInputs<MLFloat16>,
          update_decoder_feeds_fp16_func_ ? update_decoder_feeds_fp16_func_ : GenerationCpuDeviceHelper::UpdateDecoderFeeds<MLFloat16>,
          expand_buffer_float_func_,
          expand_buffer_float16_func_,
          create_beam_scorer_func_,
          update_decoder_cross_qk_func_ ? update_decoder_cross_qk_func_ : GenerationCpuDeviceHelper::UpdateDecoderCrossQK,
          finalize_decoder_cross_qk_func_ ? finalize_decoder_cross_qk_func_ : GenerationCpuDeviceHelper::FinalizeDecoderCrossQK};

#ifdef USE_CUDA
      ORT_RETURN_IF_ERROR(impl.InitializeCuda(reorder_past_state_func_, init_cache_indir_func_, cuda_device_prop_, cuda_device_arch_));
#endif
      ORT_RETURN_IF_ERROR(impl.Initialize());

      return impl.Execute(*encoder_feeds_fetches_manager_, *decoder_feeds_fetches_manager_);
    }
  }

  // Model type not supported in IGenerationParameters
  ORT_THROW("Model type is not supported.");
}

Status WhisperBeamSearch::Compute(OpKernelContext* ctx) const {
  return BeamSearch::Compute(ctx);
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
