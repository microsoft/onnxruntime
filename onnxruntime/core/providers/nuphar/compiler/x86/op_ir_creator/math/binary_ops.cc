// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

#include "core/codegen/common/op_macro.h"
#include "core/codegen/mti/math/binary_ops.h"
#include "core/codegen/mti/tensor/cast_ops.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/common.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"
#include "core/providers/nuphar/mti_x86/math/pow.h"

namespace onnxruntime {
using namespace tvm_codegen;

namespace nuphar {

bool HandleConstantScalar(tvm::Expr& scalar, size_t i, const Node& node, CodeGenContext& ctx_codegen) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);
  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);

  ORT_ENFORCE(i < node.InputDefs().size());
  const auto* tensor = ctx_nuphar->GetOrtInitializerTensor(node.InputDefs()[i]->Name());

  if (!tensor || tensor->Shape().Size() > 1)
    return false;  // return if not constant or not scalar

#define ASSIGN_TVM_SCALAR(tvm_type, tensor_type)                      \
  if (utils::IsPrimitiveDataType<tensor_type>(tensor->DataType())) {  \
    scalar = tvm::make_const(tvm_type, *tensor->Data<tensor_type>()); \
  }

#define ASSIGN_TVM_SCALAR_ELSE(tvm_type, tensor_type) \
  else ASSIGN_TVM_SCALAR(tvm_type, tensor_type)

  ASSIGN_TVM_SCALAR(HalideIR::Float(32), float)
  ASSIGN_TVM_SCALAR_ELSE(HalideIR::Int(64), int64_t)
  ASSIGN_TVM_SCALAR_ELSE(HalideIR::Int(32), int32_t)
  ASSIGN_TVM_SCALAR_ELSE(HalideIR::UInt(64), uint64_t)
  ASSIGN_TVM_SCALAR_ELSE(HalideIR::UInt(32), uint32_t)
  ASSIGN_TVM_SCALAR_ELSE(HalideIR::Float(64), double)
  else {
    return false;
  }

#undef ASSIGN_TVM_SCALAR

  return true;
}

// helper local macro defines Evaluate of BINARY_OP OpIRCreators
#define BINARY_OP(name)                                                     \
  Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(name)::Evaluate(                \
      const tvm::Array<tvm::Tensor>& inputs,                                \
      const Node& node,                                                     \
      CodeGenContext& ctx_codegen,                                          \
      tvm::Array<tvm::Tensor>& outputs) {                                   \
    tvm::Expr scalar0, scalar1;                                             \
    bool use_scalar0 = HandleConstantScalar(scalar0, 0, node, ctx_codegen); \
    bool use_scalar1 = HandleConstantScalar(scalar1, 1, node, ctx_codegen); \
    tvm::Tensor Y;                                                          \
    if (use_scalar0)                                                        \
      Y = name(scalar0, inputs[1], node.Name());                            \
    else if (use_scalar1)                                                   \
      Y = name(inputs[0], scalar1, node.Name());                            \
    else                                                                    \
      Y = name(inputs[0], inputs[1], node.Name());                          \
    outputs.push_back(Y);                                                   \
    return Status::OK();                                                    \
  }

LIST_X86_BINARY_OPS()

#undef BINARY_OP

// helper local macro defines Evaluate of BINARY_CMP_OP OpIRCreators
#define BINARY_CMP_OP(name)                                                 \
  Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(name)::Evaluate(                \
      const tvm::Array<tvm::Tensor>& inputs,                                \
      const Node& node,                                                     \
      CodeGenContext& ctx_codegen,                                          \
      tvm::Array<tvm::Tensor>& outputs) {                                   \
    tvm::Expr scalar0, scalar1;                                             \
    bool use_scalar0 = HandleConstantScalar(scalar0, 0, node, ctx_codegen); \
    bool use_scalar1 = HandleConstantScalar(scalar1, 1, node, ctx_codegen); \
    tvm::Tensor Y;                                                          \
    if (use_scalar0)                                                        \
      Y = name(scalar0, inputs[1], node.Name());                            \
    else if (use_scalar1)                                                   \
      Y = name(inputs[0], scalar1, node.Name());                            \
    else                                                                    \
      Y = name(inputs[0], inputs[1], node.Name());                          \
    Y = Cast(Y, HalideIR::UInt(8), "cast_bool_" #name);                     \
    outputs.push_back(Y);                                                   \
    return Status::OK();                                                    \
  }

LIST_X86_BINARY_CMP_OPS()

#undef BINARY_CMP_OP

}  // namespace nuphar
}  // namespace onnxruntime
