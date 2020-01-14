// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {
// helper class for unary_ops with alpha
class FuncWithAlpha {
 public:
  FuncWithAlpha(const Node& node) {
    ProtoHelperNodeContext ctx(node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);
    ORT_ENFORCE(attrs.GetAttr<float>("alpha", &alpha_).IsOK());
  }

 protected:
  float alpha_;
};

// helper class for unary_ops with alpha and beta
class FuncWithAlphaBeta {
 public:
  FuncWithAlphaBeta(const Node& node) {
    ProtoHelperNodeContext ctx(node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);
    ORT_ENFORCE(attrs.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(attrs.GetAttr<float>("beta", &beta_).IsOK());
  }

 protected:
  float alpha_;
  float beta_;
};

// helper class for unary_ops with alpha and gamma
class FuncWithAlphaGamma {
 public:
  FuncWithAlphaGamma(const Node& node) {
    ProtoHelperNodeContext ctx(node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);
    ORT_ENFORCE(attrs.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(attrs.GetAttr<float>("gamma", &gamma_).IsOK());
  }

 protected:
  float alpha_;
  float gamma_;
};
}  // namespace tvm_codegen
}  // namespace onnxruntime
