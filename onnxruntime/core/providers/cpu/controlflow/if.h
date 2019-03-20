// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "gsl/gsl_util"

#include "core/common/common.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
class SessionState;

class If final : public OpKernel {
 public:
  If(const OpKernelInfo& info) : OpKernel(info) {
    // make sure the required attributes are present even though we don't need it here.
    // The GraphProto attributes are loaded as a Graph instance by main Graph::Resolve,
    // and a SessionState instance for executing the subgraph is created by InferenceSession.
    // This is available via Info().GetSubgraphSessionState("attribute_name") when Compute is called.
    ONNX_NAMESPACE::GraphProto then_proto;
    ONNX_NAMESPACE::GraphProto else_proto;
    ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("then_branch", &then_proto).IsOK());
    ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("else_branch", &else_proto).IsOK());

    CheckForPassthroughOptimization(info, then_proto, else_proto);
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  // check if either subgraph is simply passing through an input using a single Identity node.
  // if so we can skip subgraph execution
  void CheckForPassthroughOptimization(const OpKernelInfo& info,
                                       const ONNX_NAMESPACE::GraphProto& then_proto,
                                       const ONNX_NAMESPACE::GraphProto& else_proto);

  std::string then_passthrough_input_name_ = {};
  std::string else_passthrough_input_name_ = {};

  mutable std::unique_ptr<FeedsFetchesManager> cached_then_feeds_fetches_manager_;
  mutable std::unique_ptr<FeedsFetchesManager> cached_else_feeds_fetches_manager_;
};
}  // namespace onnxruntime
