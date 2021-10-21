// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "gsl/gsl"

#include "core/providers/cpu/controlflow/utils.h"

namespace onnxruntime {
class SessionState;

class If : public controlflow::IControlFlowKernel {
 public:
  If(const OpKernelInfo& info) : IControlFlowKernel(info) { Init(info); }
  void Init(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

  Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                    const std::string& attribute_name,
                                    const SessionState& subgraph_session_state) override;

  struct Info {
    Info(const onnxruntime::Node& node, const GraphViewer& subgraph_in);
    const GraphViewer& subgraph;

    std::vector<bool> used_implicit_inputs;
    int num_implicit_inputs;
    int num_outputs;

    std::vector<std::string> subgraph_output_names;
  };

 private:
  // Info and FeedsFetchesManager re-used for each subgraph execution.
  std::unique_ptr<Info> then_info_;
  std::unique_ptr<Info> else_info_;
  std::unique_ptr<FeedsFetchesManager> then_feeds_fetches_manager_;
  std::unique_ptr<FeedsFetchesManager> else_feeds_fetches_manager_;
};
}  // namespace onnxruntime
