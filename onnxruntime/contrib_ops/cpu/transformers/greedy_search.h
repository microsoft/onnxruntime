// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include <string>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "contrib_ops/cpu/transformers/greedy_search_parameters.h"
#include "contrib_ops/cpu/transformers/subgraph_gpt.h"
#include "contrib_ops/cpu/transformers/subgraph_t5_encoder.h"
#include "contrib_ops/cpu/transformers/subgraph_t5_decoder.h"
#include "contrib_ops/cpu/transformers/generation_device_helper.h"

namespace onnxruntime {
class FeedsFetchesManager;

namespace contrib {
namespace transformers {

using namespace onnxruntime::controlflow;  // namespace of IControlFlowKernel

// bugbug: refactor
class GreedySearch : public IControlFlowKernel {
 public:
  explicit GreedySearch(const OpKernelInfo& info)
      : IControlFlowKernel(info),
        // encoder_feeds_fetches_manager_(nullptr),
        decoder_feeds_fetches_manager_(nullptr),
        cuda_stream_(nullptr),
        dumper_(nullptr) {
    Init(info);
  }

  void Init(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

  Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                    const std::string& attribute_name,
                                    const SessionState& subgraph_session_state) override;

 protected:
  void SetComputeStream(void* stream) { cuda_stream_ = stream; }
  void SetConsoleDumper(IConsoleDumper* dumper) { dumper_ = dumper; }

  // device helpers that is same for both GPT and encoder-decoder models.
  void SetDeviceHelpers(
      const GenerationDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
      const GenerationDeviceHelper::TopkFunc& topk_func,
      const GenerationDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
      const GenerationDeviceHelper::GreedySearchProcessLogitsFunc<float>& process_logits_func,
      const GenerationDeviceHelper::GreedySearchProcessLogitsFunc<MLFloat16>& process_logits_fp16_func,
      const GenerationDeviceHelper::InitGreedyStateFunc<float>& init_greedy_state_func,
      const GenerationDeviceHelper::InitGreedyStateFunc<MLFloat16>& init_greedy_state_fp16_func) {
    add_to_feeds_func_ = add_to_feeds_func;
    topk_func_ = topk_func;
    device_copy_func_ = device_copy_func;
    process_logits_func_ = process_logits_func;
    process_logits_fp16_func_ = process_logits_fp16_func;
    init_greedy_state_func_ = init_greedy_state_func;
    init_greedy_state_fp16_func_ = init_greedy_state_fp16_func;
  }

  void SetDeviceHelpers_Gpt(
      const GenerationDeviceHelper::UpdateGptFeedsFunc<float>& update_gpt_feeds_func,
      const GenerationDeviceHelper::UpdateGptFeedsFunc<MLFloat16>& update_gpt_feeds_fp16_func) {
    update_gpt_feeds_func_ = update_gpt_feeds_func;
    update_gpt_feeds_fp16_func_ = update_gpt_feeds_fp16_func;
  }

 private:
  // Device specific functions
  GenerationDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  GenerationDeviceHelper::TopkFunc topk_func_;
  GenerationDeviceHelper::DeviceCopyFunc<float> device_copy_func_;

  GenerationDeviceHelper::GreedySearchProcessLogitsFunc<float> process_logits_func_;
  GenerationDeviceHelper::GreedySearchProcessLogitsFunc<MLFloat16> process_logits_fp16_func_;

  GenerationDeviceHelper::InitGreedyStateFunc<float> init_greedy_state_func_;
  GenerationDeviceHelper::InitGreedyStateFunc<MLFloat16> init_greedy_state_fp16_func_;

  //------------------------------------------------------------
  // Device specific functions for GPT
  //------------------------------------------------------------
  GenerationDeviceHelper::UpdateGptFeedsFunc<float> update_gpt_feeds_func_;
  GenerationDeviceHelper::UpdateGptFeedsFunc<MLFloat16> update_gpt_feeds_fp16_func_;

  //------------------------------------------------------------
  // Subgraph and FeedsFetchesManager re-used for each subgraph execution.
  //------------------------------------------------------------
  std::unique_ptr<GptSubgraph> gpt_subgraph_;
  // std::unique_ptr<T5EncoderSubgraph> t5_encoder_subgraph_;
  // std::unique_ptr<T5DecoderSubgraph> t5_decoder_subgraph_;
  // FeedsFetchesManager* encoder_feeds_fetches_manager_;
  FeedsFetchesManager* decoder_feeds_fetches_manager_;

  void* cuda_stream_;

  IConsoleDumper* dumper_;

  GreedySearchParameters parameters_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
