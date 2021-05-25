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

class Loop : public controlflow::IControlFlowKernel {
 public:
  Loop(const OpKernelInfo& info) : IControlFlowKernel(info) { Init(info); }
  void Init(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

  Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                    const std::string& attribute_name,
                                    const SessionState& subgraph_session_state) override;

  struct Info {
    Info(const onnxruntime::Node& node, const GraphViewer& subgraph_in);

    const GraphViewer& subgraph;

    int num_loop_carried_vars;
    int num_implicit_inputs;
    int num_outputs;

    int num_subgraph_inputs;
    int num_subgraph_outputs;

    std::vector<std::string> subgraph_input_names;
    std::vector<std::string> subgraph_output_names;
  };

  // function to concatenate the OrtValue instances from each Loop iteration into a single output buffer.
  // @param per_iteration_output OrtValue instances from each iteration. Never empty. All should have the same shape.
  // @param output Pre-allocated output buffer. On device specific to the ExecutionProvider running the Loop node.
  using ConcatOutput = std::function<Status(void* stream, std::vector<OrtValue>& per_iteration_output,
                                            void* output, size_t output_size_in_bytes)>;

  static std::unique_ptr<OpKernel> Create(const OpKernelInfo& info, const ConcatOutput& concat_output_func, void* stream);

 protected:
  // derived class can provide implementation for handling concatenation of Loop output on a different device
  void SetConcatOutputFunc(const ConcatOutput& concat_output_func) { concat_output_func_ = concat_output_func; }
  void SetComputeStream(void* stream) { stream_ = stream; }

 private:
  // Info and FeedsFetchesManager re-used for each subgraph execution.
  std::unique_ptr<Info> info_;
  std::unique_ptr<FeedsFetchesManager> feeds_fetches_manager_;
  ConcatOutput concat_output_func_;
  void* stream_;
};
}  // namespace onnxruntime
