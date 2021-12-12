// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "gsl/gsl"
#include "core/common/common.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "beam_search_parameters.h"
#include "beam_search_scorer.h"
#include "gpt_subgraph.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

template <typename T>
class BeamSearch : public controlflow::IControlFlowKernel {
 public:
  BeamSearch(const OpKernelInfo& info) : IControlFlowKernel(info) { Init(info); }
  void Init(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

  Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                    const std::string& attribute_name,
                                    const SessionState& subgraph_session_state) override;

  static std::unique_ptr<OpKernel> Create(const OpKernelInfo& info, void* stream);

 protected:
  void SetComputeStream(void* stream) { stream_ = stream; }

 private:
  // Subgraph and FeedsFetchesManager re-used for each subgraph execution.
  std::unique_ptr<GptSubgraph> gpt_subgraph_;
  FeedsFetchesManager* feeds_fetches_manager_;

  void* stream_;

  BeamSearchParameters parameters_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
