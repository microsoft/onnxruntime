// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "gsl/gsl_util"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
class Loop final : public OpKernel {
 public:
  Loop(const OpKernelInfo& info) : OpKernel(info) {
    // make sure the attribute was present even though we don't need it here.
    // The GraphProto is loaded as a Graph instance by main Graph::Resolve,
    // and a SessionState instance for executing the subgraph is created by InferenceSession.
    // This is available via Info().GetSubgraphSessionState("attribute_name") when Compute is called.
    ONNX_NAMESPACE::GraphProto proto;
    ONNXRUNTIME_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("body", &proto).IsOK());
    ONNXRUNTIME_IGNORE_RETURN_VALUE(proto);
  }

  Status Compute(OpKernelContext* ctx) const override;
};
}  // namespace onnxruntime
