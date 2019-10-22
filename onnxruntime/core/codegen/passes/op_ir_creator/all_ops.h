// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/passes/utils/codegen_context.h"
#include "core/codegen/common/op_macro.h"
#include "core/codegen/passes/op_ir_creator/tvm_op_creator.h"

namespace onnxruntime {
namespace tvm_codegen {

// This macro declares a TVM IR builder
// based on ORT OP type with postfix DefaultTVM
#define DECLARE_GENERIC_OP_IR_CREATOR_CLASS(OP) \
  DECLARE_OP_IR_CREATOR_CLASS(OP, DefaultTVM)

// This macro returns a TVM IR builder class name
// based ORT OP type with postfix DefaultTVM
#define GENERIC_OP_IR_CREATOR_CLASS(OP) \
  CREATOR_CLASS(OP, DefaultTVM##IRCreator)

#define GENERIC_OP_IR_CREATOR_STRING(OP) \
  STRINGIZE(GENERIC_OP_IR_CREATOR_CLASS(OP))

// define all ops for DefaultTVM
#define ADD_OP_ITEM(OP) DECLARE_GENERIC_OP_IR_CREATOR_CLASS(OP)
#define BINARY_OP(OP) ADD_OP_ITEM(OP)
#define BINARY_CMP_OP(OP) ADD_OP_ITEM(OP)
#define POOL_OP(OP) ADD_OP_ITEM(OP)
#define UNARY_OP(OP) ADD_OP_ITEM(OP)
#define VARIADIC_OP(OP) ADD_OP_ITEM(OP)
#define REDUCE_INDEXED_OP(OP) ADD_OP_ITEM(OP)
#define REDUCE_OP(OP) ADD_OP_ITEM(OP)

LIST_ALL_GENERIC_OPS()

#undef ADD_OP_ITEM
#undef BINARY_OP
#undef BINARY_CMP_OP
#undef POOL_OP
#undef REDUCE_OP
#undef REDUCE_INDEXED_OP
#undef UNARY_OP
#undef VARIADIC_OP

}  // namespace tvm_codegen
}  // namespace onnxruntime
