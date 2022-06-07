// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

#include "core/codegen/passes/op_ir_creator/math/unary_funcs.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/nuphar/mti_x86/math/unary_ops.h"

namespace onnxruntime {
namespace nuphar {

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
  class Func##name : public tvm_codegen::FuncWithAlpha {  \
   public:                                                \
    Func##name(const Node& node) : FuncWithAlpha(node) {} \
    tvm::Tensor operator()(const tvm::Tensor& X) const {  \
      return name(X, alpha_);                             \
    }                                                     \
  }

// helper macro declares unary_ops helper class with alpha and beta
#define FuncClassAlphaBeta(name)                              \
  class Func##name : public tvm_codegen::FuncWithAlphaBeta {  \
   public:                                                    \
    Func##name(const Node& node) : FuncWithAlphaBeta(node) {} \
    tvm::Tensor operator()(const tvm::Tensor& X) const {      \
      return name(X, alpha_, beta_);                          \
    }                                                         \
  }

// helper macro declares unary_ops helper class with alpha and gamma
#define FuncClassAlphaGamma(name)                              \
  class Func##name : public tvm_codegen::FuncWithAlphaGamma {  \
   public:                                                     \
    Func##name(const Node& node) : FuncWithAlphaGamma(node) {} \
    tvm::Tensor operator()(const tvm::Tensor& X) const {       \
      return name(X, alpha_, gamma_);                          \
    }                                                          \
  }

FuncClass(Erf);
FuncClass(Exp);
FuncClass(Log);
FuncClassAlphaBeta(ParametricSoftplus);
FuncClassAlphaBeta(ScaledTanh);
FuncClassAlphaGamma(Selu);
FuncClass(Sigmoid);
FuncClass(Softplus);
FuncClass(Tanh);

// helper macro defines Evaluate of UNARY_OP OpIRCreators
#define UNARY_OP(name)                                       \
  Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(name)::Evaluate( \
      const tvm::Array<tvm::Tensor>& inputs,                 \
      const Node& node,                                      \
      tvm_codegen::CodeGenContext&,                          \
      tvm::Array<tvm::Tensor>& outputs) {                    \
    tvm::Tensor Y = Func##name(node)(inputs[0]);             \
    outputs.push_back(Y);                                    \
    return Status::OK();                                     \
  }

// helper local macros to replace some calls in LIST_UNARY_OPS
LIST_X86_UNARY_OPS()

#undef UNARY_OP

}  // namespace nuphar
}  // namespace onnxruntime
