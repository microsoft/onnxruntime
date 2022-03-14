// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "beam_search_parameters.h"
#include "gpt_subgraph.h"
#include "beam_search_device_helper.h"

namespace onnxruntime {
class FeedsFetchesManager;

namespace contrib {
namespace transformers {

using namespace onnxruntime::controlflow;  // namespace of IControlFlowKernel

class BeamSearch : public IControlFlowKernel {
 public:
  BeamSearch(const OpKernelInfo& info)
      : IControlFlowKernel(info), feeds_fetches_manager_(nullptr), cuda_stream_(nullptr), dumper_(nullptr) {
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

  void SetDeviceHelpers(
      // const BeamSearchDeviceHelper::CreateInputsFunc& create_inputs_func,
      const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
      const BeamSearchDeviceHelper::TopkFunc& topk_func) {
    // create_inputs_func_ = create_inputs_func;
    add_to_feeds_func_ = add_to_feeds_func;
    topk_func_ = topk_func;
  }

  // Type dependent helpers: float
  void SetDeviceHelpers(
      const BeamSearchDeviceHelper::ProcessLogitsFunc<float>& process_logits_func,
      const BeamSearchDeviceHelper::InitBeamStateFunc<float>& init_beam_state_func,
      const BeamSearchDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
      const BeamSearchDeviceHelper::UpdateFeedsFunc<float>& update_feeds_func) {
    process_logits_func_ = process_logits_func;
    init_beam_state_func_ = init_beam_state_func;
    device_copy_func_ = device_copy_func;
    update_feeds_func_ = update_feeds_func;
  }

  // Type dependent helpers: MLFloat16
  void SetDeviceHelpers(
      const BeamSearchDeviceHelper::ProcessLogitsFunc<MLFloat16>& process_logits_func,
      const BeamSearchDeviceHelper::InitBeamStateFunc<MLFloat16>& init_beam_state_func,
      const BeamSearchDeviceHelper::UpdateFeedsFunc<MLFloat16>& update_feeds_func) {
    process_logits_fp16_func_ = process_logits_func;
    init_beam_state_fp16_func_ = init_beam_state_func;
    update_feeds_fp16_func_ = update_feeds_func;
  }

 private:
  // Device specific functions
  BeamSearchDeviceHelper::CreateInputsFunc create_inputs_func_;
  BeamSearchDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  BeamSearchDeviceHelper::TopkFunc topk_func_;
  BeamSearchDeviceHelper::ProcessLogitsFunc<float> process_logits_func_;
  BeamSearchDeviceHelper::InitBeamStateFunc<float> init_beam_state_func_;
  BeamSearchDeviceHelper::DeviceCopyFunc<float> device_copy_func_;
  BeamSearchDeviceHelper::UpdateFeedsFunc<float> update_feeds_func_;

  BeamSearchDeviceHelper::ProcessLogitsFunc<MLFloat16> process_logits_fp16_func_;
  BeamSearchDeviceHelper::InitBeamStateFunc<MLFloat16> init_beam_state_fp16_func_;
  BeamSearchDeviceHelper::UpdateFeedsFunc<MLFloat16> update_feeds_fp16_func_;

  // Subgraph and FeedsFetchesManager re-used for each subgraph execution.
  std::unique_ptr<GptSubgraph> gpt_subgraph_;
  FeedsFetchesManager* feeds_fetches_manager_;

  void* cuda_stream_;

  IConsoleDumper* dumper_;

  BeamSearchParameters parameters_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
