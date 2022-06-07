// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/passes/utils/codegen_context.h"
#include "core/codegen/passes/op_ir_creator/tvm_op_creator.h"

namespace onnxruntime {
namespace nuphar {

// Declare a TVM IR builder based on the ORT OP type
// with postfix NupharTVMX86
#define DECLARE_NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(OP) \
  DECLARE_OP_IR_CREATOR_CLASS_EX(OP, NupharTVM, X86)

// Return a TVM IR builder class name such as OP type
// with postfix NupharTVMX86
#define NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(OP) \
  OP_IR_CREATOR_CLASS_EX(OP, NupharTVM, X86)

#define NUPHAR_TVM_X86_OP_IR_CREATOR_STRING(OP) \
  STRINGIZE(NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(OP))

#define LIST_X86_POOL_OPS() \
  POOL_OP(MaxPool)          \
  POOL_OP(AveragePool)      \
  POOL_OP(GlobalMaxPool)    \
  POOL_OP(GlobalAveragePool)

#define LIST_X86_UNARY_OPS()   \
  UNARY_OP(Erf)                \
  UNARY_OP(Exp)                \
  UNARY_OP(Log)                \
  UNARY_OP(ParametricSoftplus) \
  UNARY_OP(ScaledTanh)         \
  UNARY_OP(Selu)               \
  UNARY_OP(Sigmoid)            \
  UNARY_OP(Softplus)           \
  UNARY_OP(Tanh)

#define LIST_REDUCE_V_OPS() \
  REDUCE_V_OP(ReduceMax)    \
  REDUCE_V_OP(ReduceMin)    \
  REDUCE_V_OP(ReduceSum)    \
  REDUCE_V_OP(ReduceMean)

#define LIST_ALL_X86_OPS()     \
  LIST_REDUCE_V_OPS()          \
  LIST_X86_POOL_OPS()          \
  LIST_X86_UNARY_OPS()         \
  ADD_OP_ITEM(Dropout)         \
  ADD_OP_ITEM(Gemm)            \
  ADD_OP_ITEM(LogSoftmax)      \
  ADD_OP_ITEM(MatMul)          \
  ADD_OP_ITEM(MatMulInteger)   \
  ADD_OP_ITEM(MatMulInteger16) \
  ADD_OP_ITEM(Pow)             \
  ADD_OP_ITEM(Scatter)         \
  ADD_OP_ITEM(ScatterElements) \
  ADD_OP_ITEM(Slice)           \
  ADD_OP_ITEM(Softmax)         \
  ADD_OP_ITEM(Tile)

// Define all OPs for NupharTVMX86
#define ADD_OP_ITEM(OP) DECLARE_NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(OP)
#define POOL_OP(OP) ADD_OP_ITEM(OP)
#define REDUCE_V_OP(OP) ADD_OP_ITEM(OP)
#define UNARY_OP(OP) ADD_OP_ITEM(OP)

LIST_ALL_X86_OPS()

#undef ADD_OP_ITEM
#undef REDUCE_V_OP
#undef POOL_OP
#undef UNARY_OP

}  // namespace nuphar
}  // namespace onnxruntime
