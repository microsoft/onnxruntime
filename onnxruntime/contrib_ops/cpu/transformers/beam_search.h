// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "contrib_ops/cpu/transformers/beam_search_parameters.h"
#include "contrib_ops/cpu/transformers/subgraph_gpt.h"
#include "contrib_ops/cpu/transformers/subgraph_t5_encoder.h"
#include "contrib_ops/cpu/transformers/subgraph_t5_decoder.h"
#include "contrib_ops/cpu/transformers/subgraph_whisper_encoder.h"
#include "contrib_ops/cpu/transformers/subgraph_whisper_decoder.h"
#include "contrib_ops/cpu/transformers/generation_device_helper.h"

namespace onnxruntime {
class FeedsFetchesManager;

namespace contrib {
namespace transformers {

using namespace onnxruntime::controlflow;  // namespace of IControlFlowKernel

class BeamSearch : public IControlFlowKernel {
 public:
  BeamSearch(const OpKernelInfo& info, std::unique_ptr<BeamSearchParameters> param = std::make_unique<BeamSearchParameters>())
      : IControlFlowKernel(info),
        encoder_feeds_fetches_manager_(nullptr),
        decoder_feeds_fetches_manager_(nullptr),
        dumper_(nullptr) {
    parameters_.swap(param);
    Init(info);
  }

  void Init(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

  Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                    const std::string& attribute_name,
                                    const SessionState& subgraph_session_state) override;

 protected:
  void SetConsoleDumper(IConsoleDumper* dumper) { dumper_ = dumper; }

  // device helpers that is same for both GPT and encoder-decoder models.
  void SetDeviceHelpers(
      const GenerationDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
      const GenerationDeviceHelper::TopkFunc& topk_func,
      const GenerationDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
      const GenerationDeviceHelper::DeviceCopyFunc<int32_t>& device_copy_int32_func,
      const GenerationDeviceHelper::ProcessLogitsFunc<float>& process_logits_func,
      const GenerationDeviceHelper::ProcessLogitsFunc<MLFloat16>& process_logits_fp16_func,
      const GenerationDeviceHelper::InitBeamStateFunc<float>& init_beam_state_func,
      const GenerationDeviceHelper::InitBeamStateFunc<MLFloat16>& init_beam_state_fp16_func,
      const GenerationDeviceHelper::CreateBeamScorer& create_beam_scorer_func) {
    add_to_feeds_func_ = add_to_feeds_func;
    topk_func_ = topk_func;
    device_copy_func_ = device_copy_func;
    device_copy_int32_func_ = device_copy_int32_func;
    process_logits_func_ = process_logits_func;
    process_logits_fp16_func_ = process_logits_fp16_func;
    init_beam_state_func_ = init_beam_state_func;
    init_beam_state_fp16_func_ = init_beam_state_fp16_func;
    create_beam_scorer_func_ = create_beam_scorer_func;
  }

#ifdef USE_CUDA
  void SetDeviceHelpers_Cuda(
      const GenerationDeviceHelper::ReorderPastStateFunc& reorder_past_state_func,
      const GenerationDeviceHelper::InitCacheIndirFunc& init_cache_indir_func) {
    reorder_past_state_func_ = reorder_past_state_func;
    init_cache_indir_func_ = init_cache_indir_func;
  }
#endif

  void SetDeviceHelpers_Gpt(
      const GenerationDeviceHelper::UpdateGptFeedsFunc<float>& update_gpt_feeds_func,
      const GenerationDeviceHelper::UpdateGptFeedsFunc<MLFloat16>& update_gpt_feeds_fp16_func) {
    update_gpt_feeds_func_ = update_gpt_feeds_func;
    update_gpt_feeds_fp16_func_ = update_gpt_feeds_fp16_func;
  }

  // device helpers for encoder-decoder model like T5
  void SetDeviceHelpers_EncoderDecoder(
      const GenerationDeviceHelper::UpdateDecoderFeedsFunc<float>& update_decoder_feeds_func,
      const GenerationDeviceHelper::UpdateDecoderFeedsFunc<MLFloat16>& update_decoder_feeds_fp16_func,
      const GenerationDeviceHelper::ExpandBufferFunc<int32_t>& expand_buffer_int32_func,
      const GenerationDeviceHelper::ExpandBufferFunc<float>& expand_buffer_float_func,
      const GenerationDeviceHelper::ExpandBufferFunc<MLFloat16>& expand_buffer_float16_func,
      const GenerationDeviceHelper::UpdateDecoderCrossQKFunc& update_decoder_cross_qk_func,
      const GenerationDeviceHelper::FinalizeDecoderCrossQKFunc& finalize_decoder_cross_qk_func) {
    update_decoder_feeds_func_ = update_decoder_feeds_func;
    update_decoder_feeds_fp16_func_ = update_decoder_feeds_fp16_func;
    expand_buffer_int32_func_ = expand_buffer_int32_func;
    expand_buffer_float_func_ = expand_buffer_float_func;
    expand_buffer_float16_func_ = expand_buffer_float16_func;
    update_decoder_cross_qk_func_ = update_decoder_cross_qk_func;
    finalize_decoder_cross_qk_func_ = finalize_decoder_cross_qk_func;
  }

#ifdef USE_CUDA
  const void* cuda_device_prop_ = nullptr;
  int cuda_device_arch_ = 0;
#endif

 protected:
  // Device specific functions
  GenerationDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  GenerationDeviceHelper::TopkFunc topk_func_;
  GenerationDeviceHelper::DeviceCopyFunc<float> device_copy_func_;
  GenerationDeviceHelper::DeviceCopyFunc<int32_t> device_copy_int32_func_;

  GenerationDeviceHelper::ProcessLogitsFunc<float> process_logits_func_;
  GenerationDeviceHelper::ProcessLogitsFunc<MLFloat16> process_logits_fp16_func_;

  GenerationDeviceHelper::InitBeamStateFunc<float> init_beam_state_func_;
  GenerationDeviceHelper::InitBeamStateFunc<MLFloat16> init_beam_state_fp16_func_;
  GenerationDeviceHelper::CreateBeamScorer create_beam_scorer_func_;

#ifdef USE_CUDA
  GenerationDeviceHelper::ReorderPastStateFunc reorder_past_state_func_;
  GenerationDeviceHelper::InitCacheIndirFunc init_cache_indir_func_;
#endif

  //------------------------------------------------------------
  // Device specific functions for GPT
  //------------------------------------------------------------
  GenerationDeviceHelper::UpdateGptFeedsFunc<float> update_gpt_feeds_func_;
  GenerationDeviceHelper::UpdateGptFeedsFunc<MLFloat16> update_gpt_feeds_fp16_func_;

  //------------------------------------------------------------
  // Device specific functions for encoder-decoder model like T5
  //------------------------------------------------------------
  GenerationDeviceHelper::CreateEncoderInputsFunc create_encoder_inputs_func_;
  GenerationDeviceHelper::UpdateDecoderFeedsFunc<float> update_decoder_feeds_func_;
  GenerationDeviceHelper::UpdateDecoderFeedsFunc<MLFloat16> update_decoder_feeds_fp16_func_;

  //------------------------------------------------------------
  // Device specific functions for Whisper
  //------------------------------------------------------------
  GenerationDeviceHelper::CreateWhisperEncoderInputsFunc create_whisper_encoder_inputs_func_;

  GenerationDeviceHelper::ExpandBufferFunc<int32_t> expand_buffer_int32_func_;
  GenerationDeviceHelper::ExpandBufferFunc<float> expand_buffer_float_func_;
  GenerationDeviceHelper::ExpandBufferFunc<MLFloat16> expand_buffer_float16_func_;

  //------------------------------------------------------------
  // Subgraph and FeedsFetchesManager re-used for each subgraph execution.
  //------------------------------------------------------------

  // Relevant only for GPT2
  // The init_run_gpt_subgraph_ (if the `init_decoder` attribute is present) will be
  // used for the first decoding run and the gpt_subgraph_ will be used
  // for subsequent runs.
  // If the `init_decoder` attribute is missing, the `gpt_subgraph_` will be
  // used for all decoding runs.
  std::unique_ptr<GptSubgraph> init_run_gpt_subgraph_;
  std::unique_ptr<GptSubgraph> gpt_subgraph_;

  // Relevant only for T5
  // Same concept as above.
  // The encoder will be used for the first run and the decoder will
  // be used for subsequent runs.
  std::unique_ptr<T5EncoderSubgraph> t5_encoder_subgraph_;
  std::unique_ptr<T5DecoderSubgraph> t5_decoder_subgraph_;

  // Relevant only for Whisper
  std::unique_ptr<WhisperEncoderSubgraph> whisper_encoder_subgraph_;
  std::unique_ptr<WhisperDecoderSubgraph> whisper_decoder_subgraph_;

  FeedsFetchesManager* encoder_feeds_fetches_manager_;
  FeedsFetchesManager* decoder_feeds_fetches_manager_;
  FeedsFetchesManager* init_run_decoder_feeds_fetches_manager_;

  IConsoleDumper* dumper_;

  std::unique_ptr<BeamSearchParameters> parameters_;

  bool has_init_decoder_ = false;

  GenerationDeviceHelper::UpdateDecoderCrossQKFunc update_decoder_cross_qk_func_;

  GenerationDeviceHelper::FinalizeDecoderCrossQKFunc finalize_decoder_cross_qk_func_;
};

class WhisperBeamSearch : public BeamSearch {
 public:
  WhisperBeamSearch(const OpKernelInfo& info)
      : BeamSearch(info, std::unique_ptr<BeamSearchParameters>(new WhisperBeamSearchParameters())) {}

  Status Compute(OpKernelContext* ctx) const override;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
