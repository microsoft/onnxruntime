// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/target/tvm_op_creator.h"
#include "core/codegen/target/tvm_context.h"

namespace onnxruntime {
namespace tvm_codegen {

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

#define LIST_REDUCE_V_OPS() \
  REDUCE_V_OP(ReduceMax)    \
  REDUCE_V_OP(ReduceMin)    \
  REDUCE_V_OP(ReduceSum)

#define LIST_ALL_X86_OPS()   \
  LIST_REDUCE_V_OPS()        \
  ADD_OP_ITEM(LogSoftmax)    \
  ADD_OP_ITEM(MatMul)        \
  ADD_OP_ITEM(MatMulInteger) \
  ADD_OP_ITEM(Softmax)

// Define all OPs for NupharTVMX86
#define ADD_OP_ITEM(OP) DECLARE_NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(OP)
#define REDUCE_V_OP(OP) ADD_OP_ITEM(OP)

LIST_ALL_X86_OPS()

#undef ADD_OP_ITEM
#undef REDUCE_V_OP

}  // namespace tvm_codegen
}  // namespace onnxruntime
