// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "gsl/gsl_util"

#include "core/common/common.h"
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
    ONNX_NAMESPACE::GraphProto proto;
    ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("then_branch", &proto).IsOK());
    ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("else_branch", &proto).IsOK());
    ORT_IGNORE_RETURN_VALUE(proto);
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
};
}  // namespace onnxruntime
