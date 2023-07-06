// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include <assert.h>
#include <functional>
#include <string>
#include <utility>
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
#include "core/framework/ort_value.h"
#include "core/common/gsl.h"
#include "contrib_ops/cpu/transformers/greedy_search.h"
#include "contrib_ops/cpu/transformers/logits_processor.h"
#include "contrib_ops/cpu/transformers/sequences.h"
#include "contrib_ops/cpu/transformers/dump_tensor.h"
#include "contrib_ops/cpu/transformers/greedy_search_impl_gpt.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GreedySearch,                                               \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      transformers::GreedySearch);

REGISTER_KERNEL_TYPED(float)

namespace transformers {

namespace gpt_details {
std::pair<Status, std::unique_ptr<GptSubgraph>> CreateGptSubgraphAndUpdateParameters(
    const Node& node,
    const SessionState& session_state,
    const std::string& attribute_name,
    const SessionState& subgraph_session_state,
    /*out*/ BeamSearchParameters& parameters) {
  auto gpt_subgraph = std::make_unique<GptSubgraph>(node, attribute_name, subgraph_session_state.GetGraphViewer());
  auto status = gpt_subgraph->Setup(session_state, subgraph_session_state);
  if (!status.IsOK()) {
    return std::make_pair(status, std::move(gpt_subgraph));
  }

  parameters.SetSubgraphParameters(gpt_subgraph->vocab_size,
                                   gpt_subgraph->num_heads,
                                   gpt_subgraph->head_size,
                                   gpt_subgraph->num_layers);

  return std::make_pair(status, std::move(gpt_subgraph));
}
}  // namespace gpt_details

void GreedySearch::Init(const OpKernelInfo& info) {
  parameters_.ParseFromAttributes(info);
  parameters_.vocab_size = (parameters_.vocab_size == 0 ? -1 : parameters_.vocab_size);

  // Model_type could be either 0 (GPT-2) or 1 (encoder-decoder like T5)
  ORT_ENFORCE(parameters_.model_type == IGenerationParameters::kModelTypeGpt);

  ONNX_NAMESPACE::GraphProto proto;
  if (parameters_.model_type != IGenerationParameters::kModelTypeGpt) {
    // Make sure the encoder sub-graph attribute is present for the T5 model.
    ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("encoder", &proto).IsOK());
  }

  if (parameters_.model_type == IGenerationParameters::kModelTypeGpt) {
    // Check if the init_decoder sub-graph attribute is present for the GPT2 model.
    if (info.GetAttr<ONNX_NAMESPACE::GraphProto>("init_decoder", &proto).IsOK()) {
      has_init_decoder_ = true;
    }
  }

  // Make sure the decoder sub-graph attribute is present for all model types.
  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("decoder", &proto).IsOK());
}

Status GreedySearch::SetupSubgraphExecutionInfo(const SessionState& session_state,
                                                const std::string& attribute_name,
                                                const SessionState& subgraph_session_state) {
  const auto& node = Node();
  if (parameters_.model_type == IGenerationParameters::kModelTypeGpt) {  // GPT-2
    if (attribute_name == "decoder") {
      ORT_ENFORCE(gpt_subgraph_ == nullptr, "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
      auto res = gpt_details::CreateGptSubgraphAndUpdateParameters(node, session_state, attribute_name,
                                                                   subgraph_session_state, parameters_);

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
                                                                   subgraph_session_state, parameters_);

      auto status = res.first;
      if (!status.IsOK()) {
        return status;
      }

      init_run_gpt_subgraph_ = std::move(res.second);
      init_run_decoder_feeds_fetches_manager_ = init_run_gpt_subgraph_->GetFeedsFetchesManager();
    }
  } else if (parameters_.model_type == IGenerationParameters::kModelTypeT5) {  // encoder-decoder like T5
    ORT_THROW("Not Implemented");
    // if (attribute_name == "encoder") {
    //   ORT_ENFORCE(t5_encoder_subgraph_ == nullptr,
    //               "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
    //   t5_encoder_subgraph_ = std::make_unique<T5EncoderSubgraph>(
    //                            node,
    //                            attribute_name,
    //                            subgraph_session_state.GetGraphViewer());
    //   ORT_RETURN_IF_ERROR(t5_encoder_subgraph_->Setup(session_state, subgraph_session_state));
    //   encoder_feeds_fetches_manager_ = t5_encoder_subgraph_->GetFeedsFetchesManager();
    // } else if (attribute_name == "decoder") {
    //   ORT_ENFORCE(t5_decoder_subgraph_ == nullptr,
    //               "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
    //   t5_decoder_subgraph_ = std::make_unique<T5DecoderSubgraph>(
    //                            node,
    //                            attribute_name,
    //                            subgraph_session_state.GetGraphViewer());
    //   ORT_RETURN_IF_ERROR(t5_decoder_subgraph_->Setup(session_state, subgraph_session_state));
    //   decoder_feeds_fetches_manager_ = t5_decoder_subgraph_->GetFeedsFetchesManager();
    //   parameters_.SetSubgraphParameters(t5_decoder_subgraph_->vocab_size,
    //                                     t5_decoder_subgraph_->num_heads,
    //                                     t5_decoder_subgraph_->head_size,
    //                                     t5_decoder_subgraph_->num_layers);
    // }
  }

  return Status::OK();
}

Status GreedySearch::Compute(OpKernelContext* ctx) const {
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

  // make a copy since we will update the parameters based on inputs later
  GreedySearchParameters parameters = parameters_;

  if (parameters_.model_type == 0) {  // GPT-2
    // Subgraph has constraint that the output is either float or float16
    if (!gpt_subgraph_->IsOutputFloat16()) {
      GreedySearchGpt<float, GreedySearchParameters> impl{
          *ctx_internal,
          has_init_decoder_ ? init_run_decoder_session_state : nullptr,
          has_init_decoder_ ? init_run_gpt_subgraph_.get() : nullptr,
          *decoder_session_state,
          *gpt_subgraph_,
          thread_pool,
          ctx->GetComputeStream(),
          dumper_,
          parameters,
          GenerationCpuDeviceHelper::CreateGptInputs,
          add_to_feeds_func_ ? add_to_feeds_func_ : GenerationCpuDeviceHelper::AddToFeeds,
          topk_func_ ? topk_func_ : GenerationCpuDeviceHelper::TopK,
          process_logits_func_ ? process_logits_func_ : GenerationCpuDeviceHelper::GreedySearchProcessLogits<float>,
          init_greedy_state_func_ ? init_greedy_state_func_ : GenerationCpuDeviceHelper::InitGreedyState<float>,
          device_copy_func_ ? device_copy_func_ : GenerationCpuDeviceHelper::DeviceCopy<float>,
          update_gpt_feeds_func_ ? update_gpt_feeds_func_ : GenerationCpuDeviceHelper::UpdateGptFeeds<float>};
#ifdef USE_CUDA
      ORT_RETURN_IF_ERROR(impl.InitializeCuda(reorder_past_state_func_, cuda_device_prop_, cuda_device_arch_));
#endif
      ORT_RETURN_IF_ERROR(impl.Initialize());

      return impl.Execute(init_run_decoder_feeds_fetches_manager_, *decoder_feeds_fetches_manager_);
    } else {
      GreedySearchGpt<MLFloat16, GreedySearchParameters> impl{
          *ctx_internal,
          has_init_decoder_ ? init_run_decoder_session_state : nullptr,
          has_init_decoder_ ? init_run_gpt_subgraph_.get() : nullptr,
          *decoder_session_state,
          *gpt_subgraph_,
          thread_pool,
          ctx->GetComputeStream(),
          dumper_,
          parameters,
          GenerationCpuDeviceHelper::CreateGptInputs,
          add_to_feeds_func_ ? add_to_feeds_func_ : GenerationCpuDeviceHelper::AddToFeeds,
          topk_func_ ? topk_func_ : GenerationCpuDeviceHelper::TopK,
          process_logits_fp16_func_,
          init_greedy_state_fp16_func_,
          device_copy_func_,
          update_gpt_feeds_fp16_func_};
#ifdef USE_CUDA
      ORT_RETURN_IF_ERROR(impl.InitializeCuda(reorder_past_state_func_, cuda_device_prop_, cuda_device_arch_));
#endif
      ORT_RETURN_IF_ERROR(impl.Initialize());

      return impl.Execute(init_run_decoder_feeds_fetches_manager_, *decoder_feeds_fetches_manager_);
    }
  }

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
