// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "gsl/gsl"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/providers/cpu/controlflow/utils.h"

namespace onnxruntime {
template <int OpSet>
class Scan final : public OpKernel, public controlflow::IControlFlowKernel {
 public:
  Scan(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

  common::Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                            const std::string& attribute_name,
                                            const SessionState& subgraph_session_state) override;

  // hide internal implementation details via forward declaration.
  struct Info;
  ~Scan();

 private:
  int64_t num_scan_inputs_;
  std::vector<int64_t> input_directions_;
  std::vector<int64_t> output_directions_;
  std::vector<int64_t> input_axes_;
  std::vector<int64_t> output_axes_;

  // Info and FeedsFetchesManager re-used for each subgraph execution.
  std::unique_ptr<Info> info_;
  std::unique_ptr<FeedsFetchesManager> feeds_fetches_manager_;
};
}  // namespace onnxruntime
