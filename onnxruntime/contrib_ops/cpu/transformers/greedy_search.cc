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

void GreedySearch::Init(const OpKernelInfo& info) {
  parameters_.ParseFromAttributes(info);

  // Check model_type 0 (GPT-2) and 1 (encoder-decoder like T5)
  ORT_ENFORCE(parameters_.model_type == 0 || parameters_.model_type == 1);

  // Make sure the decoder attribute was present even though we don't need it here.
  ONNX_NAMESPACE::GraphProto proto;
  if (parameters_.model_type != 0) {
    ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("encoder", &proto).IsOK());
  }

  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("decoder", &proto).IsOK());
  ORT_IGNORE_RETURN_VALUE(proto);
}

Status GreedySearch::SetupSubgraphExecutionInfo(const SessionState& session_state,
                                                const std::string& attribute_name,
                                                const SessionState& subgraph_session_state) {
  const auto& node = Node();
  if (parameters_.model_type == IBeamSearchParameters::kModelTypeGpt) {  // GPT-2
    if (attribute_name == "decoder") {
      ORT_ENFORCE(gpt_subgraph_ == nullptr,
                  "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
      gpt_subgraph_ = std::make_unique<GptSubgraph>(node, attribute_name, subgraph_session_state.GetGraphViewer());
      ORT_RETURN_IF_ERROR(gpt_subgraph_->Setup(session_state, subgraph_session_state));
      decoder_feeds_fetches_manager_ = gpt_subgraph_->GetFeedsFetchesManager();
      parameters_.SetSubgraphParameters(gpt_subgraph_->vocab_size,
                                        gpt_subgraph_->num_heads,
                                        gpt_subgraph_->head_size,
                                        gpt_subgraph_->num_layers);
    }
  } else if (parameters_.model_type == IBeamSearchParameters::kModelTypeT5) {  // encoder-decoder like T5
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

  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  // make a copy since we will update the parameters based on inputs later
  GreedySearchParameters parameters = parameters_;

  if (parameters_.model_type == 0) {  // GPT-2
    // Subgraph has constraint that the output is either float or float16
    if (!gpt_subgraph_->IsOutputFloat16()) {
      GreedySearchGpt<float> impl{
          *ctx_internal,
          *decoder_session_state,
          *gpt_subgraph_,
          thread_pool,
          cuda_stream_,
          dumper_,
          parameters,
          GenerationCpuDeviceHelper::CreateGptInputs,
          add_to_feeds_func_ ? add_to_feeds_func_ : GenerationCpuDeviceHelper::AddToFeeds,
          topk_func_ ? topk_func_ : GenerationCpuDeviceHelper::TopK,
          process_logits_func_ ? process_logits_func_ : GenerationCpuDeviceHelper::GreedySearchProcessLogits<float>,
          init_greedy_state_func_ ? init_greedy_state_func_ : GenerationCpuDeviceHelper::InitGreedyState<float>,
          device_copy_func_ ? device_copy_func_ : GenerationCpuDeviceHelper::DeviceCopy<float>,
          update_gpt_feeds_func_ ? update_gpt_feeds_func_ : GenerationCpuDeviceHelper::UpdateGptFeeds<float>};
      ORT_RETURN_IF_ERROR(impl.Initialize());

      return impl.Execute(*decoder_feeds_fetches_manager_);
    } else {
      GreedySearchGpt<MLFloat16> impl{
          *ctx_internal,
          *decoder_session_state,
          *gpt_subgraph_,
          thread_pool,
          cuda_stream_,
          dumper_,
          parameters,
          GenerationCpuDeviceHelper::CreateGptInputs,
          add_to_feeds_func_ ? add_to_feeds_func_ : GenerationCpuDeviceHelper::AddToFeeds,
          topk_func_ ? topk_func_ : GenerationCpuDeviceHelper::TopK,
          process_logits_fp16_func_,
          init_greedy_state_fp16_func_,
          device_copy_func_,
          update_gpt_feeds_fp16_func_};
      ORT_RETURN_IF_ERROR(impl.Initialize());

      return impl.Execute(*decoder_feeds_fetches_manager_);
    }
  }

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
