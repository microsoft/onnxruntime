// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "gsl/gsl"

#include "core/common/common.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/controlflow/utils.h"

namespace onnxruntime {

class Loop final : public OpKernel, public controlflow::IControlFlowKernel {
 public:
  Loop(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

  common::Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                            const std::string& attribute_name,
                                            const SessionState& subgraph_session_state) override;

  // hide internal implementation details via forward declaration.
  struct Info;
  ~Loop();

 private:
  // Info and FeedsFetchesManager re-used for each subgraph execution.
  std::unique_ptr<Info> info_;
  std::unique_ptr<FeedsFetchesManager> feeds_fetches_manager_;
};
}  // namespace onnxruntime
