// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "contrib_ops/cpu/transformers/greedy_search.h"
#include "contrib_ops/cpu/transformers/sampling_parameters.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

class Sampling : public GreedySearch {
 public:
  explicit Sampling(const OpKernelInfo& info) : GreedySearch(info) {}

  void Init(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

  Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                    const std::string& attribute_name,
                                    const SessionState& subgraph_session_state) override;

 private:
  SamplingParameters parameters_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
