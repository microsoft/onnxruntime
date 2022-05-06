// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "beam_search_parameters.h"
#include "subgraph_gpt.h"
#include "subgraph_t5_encoder.h"
#include "subgraph_t5_decoder.h"
#include "beam_search_device_helper.h"

namespace onnxruntime {
class FeedsFetchesManager;

namespace contrib {
namespace transformers {

using namespace onnxruntime::controlflow;  // namespace of IControlFlowKernel

class BeamSearch : public IControlFlowKernel {
 public:
  BeamSearch(const OpKernelInfo& info)
      : IControlFlowKernel(info),
        encoder_feeds_fetches_manager_(nullptr),
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
      const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
      const BeamSearchDeviceHelper::TopkFunc& topk_func,
      const BeamSearchDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
      const BeamSearchDeviceHelper::ProcessLogitsFunc<float>& process_logits_func,
      const BeamSearchDeviceHelper::ProcessLogitsFunc<MLFloat16>& process_logits_fp16_func,
      const BeamSearchDeviceHelper::InitBeamStateFunc<float>& init_beam_state_func,
      const BeamSearchDeviceHelper::InitBeamStateFunc<MLFloat16>& init_beam_state_fp16_func) {
    add_to_feeds_func_ = add_to_feeds_func;
    topk_func_ = topk_func;
    device_copy_func_ = device_copy_func;
    process_logits_func_ = process_logits_func;
    process_logits_fp16_func_ = process_logits_fp16_func;
    init_beam_state_func_ = init_beam_state_func;
    init_beam_state_fp16_func_ = init_beam_state_fp16_func;
  }

  // device helpers for GPT model
  // void SetDeviceHelpers_Gpt(const BeamSearchDeviceHelper::CreateGptInputsFunc& create_gpt_inputs_func) {
  //   create_gpt_inputs_func_ = create_gpt_inputs_func;
  // }

  void SetDeviceHelpers_Gpt(
      const BeamSearchDeviceHelper::UpdateGptFeedsFunc<float>& update_gpt_feeds_func,
      const BeamSearchDeviceHelper::UpdateGptFeedsFunc<MLFloat16>& update_gpt_feeds_fp16_func) {
    update_gpt_feeds_func_ = update_gpt_feeds_func;
    update_gpt_feeds_fp16_func_ = update_gpt_feeds_fp16_func;
  }

  // device helpers for encoder-decoder model like T5
  void SetDeviceHelpers_EncoderDecoder(
      const BeamSearchDeviceHelper::CreateEncoderInputsFunc& create_encoder_inputs_func,
      const BeamSearchDeviceHelper::InitDecoderFeedsFunc<float>& init_decoder_feeds_func,
      const BeamSearchDeviceHelper::UpdateDecoderFeedsFunc<float>& update_decoder_feeds_func,
      const BeamSearchDeviceHelper::InitDecoderFeedsFunc<MLFloat16>& init_decoder_feeds_fp16_func,
      const BeamSearchDeviceHelper::UpdateDecoderFeedsFunc<MLFloat16>& update_decoder_feeds_fp16_func) {
    create_encoder_inputs_func_ = create_encoder_inputs_func;
    init_decoder_feeds_func_ = init_decoder_feeds_func;
    update_decoder_feeds_func_ = update_decoder_feeds_func;
    init_decoder_feeds_fp16_func_ = init_decoder_feeds_fp16_func;
    update_decoder_feeds_fp16_func_ = update_decoder_feeds_fp16_func;
  }

 private:
  // Device specific functions
  BeamSearchDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  BeamSearchDeviceHelper::TopkFunc topk_func_;
  BeamSearchDeviceHelper::DeviceCopyFunc<float> device_copy_func_;

  BeamSearchDeviceHelper::ProcessLogitsFunc<float> process_logits_func_;
  BeamSearchDeviceHelper::ProcessLogitsFunc<MLFloat16> process_logits_fp16_func_;

  BeamSearchDeviceHelper::InitBeamStateFunc<float> init_beam_state_func_;
  BeamSearchDeviceHelper::InitBeamStateFunc<MLFloat16> init_beam_state_fp16_func_;

  //------------------------------------------------------------
  // Device specific functions for GPT
  //------------------------------------------------------------
  BeamSearchDeviceHelper::CreateGptInputsFunc create_gpt_inputs_func_;
  BeamSearchDeviceHelper::UpdateGptFeedsFunc<float> update_gpt_feeds_func_;
  BeamSearchDeviceHelper::UpdateGptFeedsFunc<MLFloat16> update_gpt_feeds_fp16_func_;

  //------------------------------------------------------------
  // Device specific functions for encoder-decoder model like T5
  //------------------------------------------------------------
  BeamSearchDeviceHelper::CreateEncoderInputsFunc create_encoder_inputs_func_;

  BeamSearchDeviceHelper::InitDecoderFeedsFunc<float> init_decoder_feeds_func_;
  BeamSearchDeviceHelper::InitDecoderFeedsFunc<MLFloat16> init_decoder_feeds_fp16_func_;

  BeamSearchDeviceHelper::UpdateDecoderFeedsFunc<float> update_decoder_feeds_func_;
  BeamSearchDeviceHelper::UpdateDecoderFeedsFunc<MLFloat16> update_decoder_feeds_fp16_func_;

  //------------------------------------------------------------
  // Subgraph and FeedsFetchesManager re-used for each subgraph execution.
  //------------------------------------------------------------
  std::unique_ptr<GptSubgraph> gpt_subgraph_;
  std::unique_ptr<T5EncoderSubgraph> t5_encoder_subgraph_;
  std::unique_ptr<T5DecoderSubgraph> t5_decoder_subgraph_;
  FeedsFetchesManager* encoder_feeds_fetches_manager_;
  FeedsFetchesManager* decoder_feeds_fetches_manager_;

  void* cuda_stream_;

  IConsoleDumper* dumper_;

  BeamSearchParameters parameters_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
