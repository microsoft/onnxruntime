// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/target/generic/op_ir_creator/all_ops.h"

#include "core/codegen/common/op_macro.h"
#include "core/codegen/mti/math/unary_ops.h"
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

// helper macro declares unary_ops helper class without attribute
#define FuncClass(name)                                  \
  class Func##name {                                     \
   public:                                               \
    Func##name(const Node&) {}                           \
    tvm::Tensor operator()(const tvm::Tensor& X) const { \
      return name(X);                                    \
    }                                                    \
  }

// helper macro declares unary_ops helper class with alpha
#define FuncClassAlpha(name)                              \
  class Func##name : public FuncWithAlpha {               \
   public:                                                \
    Func##name(const Node& node) : FuncWithAlpha(node) {} \
    tvm::Tensor operator()(const tvm::Tensor& X) const {  \
      return name(X, alpha_);                             \
    }                                                     \
  }

// helper macro declares unary_ops helper class with alpha and beta
#define FuncClassAlphaBeta(name)                              \
  class Func##name : public FuncWithAlphaBeta {               \
   public:                                                    \
    Func##name(const Node& node) : FuncWithAlphaBeta(node) {} \
    tvm::Tensor operator()(const tvm::Tensor& X) const {      \
      return name(X, alpha_, beta_);                          \
    }                                                         \
  }

// helper macro declares unary_ops helper class with alpha and gamma
#define FuncClassAlphaGamma(name)                              \
  class Func##name : public FuncWithAlphaGamma {               \
   public:                                                     \
    Func##name(const Node& node) : FuncWithAlphaGamma(node) {} \
    tvm::Tensor operator()(const tvm::Tensor& X) const {       \
      return name(X, alpha_, gamma_);                          \
    }                                                          \
  }

FuncClass(Abs);
FuncClassAlphaBeta(Affine);
FuncClass(Ceil);
FuncClassAlpha(Elu);
FuncClass(Exp);
FuncClass(Floor);
FuncClassAlphaBeta(HardSigmoid);
FuncClassAlpha(LeakyRelu);
FuncClass(Log);
FuncClass(Neg);
FuncClassAlphaBeta(ParametricSoftplus);
FuncClass(Reciprocal);
FuncClass(Relu);
FuncClassAlphaBeta(ScaledTanh);
FuncClassAlphaGamma(Selu);
FuncClass(Sigmoid);
FuncClass(Softplus);
FuncClass(Softsign);
FuncClass(Sqrt);
FuncClass(Tanh);
FuncClassAlpha(ThresholdedRelu);

// helper macro defines Evaluate of UNARY_OP OpIRCreators
#define UNARY_OP(name)                                \
  Status GENERIC_OP_IR_CREATOR_CLASS(name)::Evaluate( \
      const tvm::Array<tvm::Tensor>& inputs,          \
      const Node& node,                               \
      CodeGenContext&,                                \
      tvm::Array<tvm::Tensor>& outputs) {             \
    tvm::Tensor Y = Func##name(node)(inputs[0]);      \
    outputs.push_back(Y);                             \
    return Status::OK();                              \
  }

// helper local macros to replace some calls in LIST_UNARY_OPS
LIST_UNARY_OPS()

#undef UNARY_OP

}  // namespace tvm_codegen
}  // namespace onnxruntime
