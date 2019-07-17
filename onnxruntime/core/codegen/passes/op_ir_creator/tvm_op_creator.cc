// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/tvm_op_creator.h"

#include "core/codegen/common/common.h"
#include "core/codegen/common/dispatcher.h"
#include "core/codegen/passes/utils/codegen_context.h"

namespace onnxruntime {
namespace codegen {
// Explicit instantiation for OpIRCreator
template class CreatorBase<const tvm::Array<tvm::Tensor>&,
                           const Node&,
                           tvm_codegen::CodeGenContext&,
                           tvm::Array<tvm::Tensor>&,
                           Status>;

// Explicit instantiation for OpIRCreators' dispatcher
template class DispatcherBase<tvm_codegen::OpIRCreator*>;

}  // namespace codegen

namespace tvm_codegen {

// One dispatcher is based on ORT OpType
OpIRCreator* OP_IR_DISPATCHER_CLASS(OpType)::Find(const Node& node) {
  return DispatcherBase::Get(node.OpType());
}

// Another dispatcher is based ORT NodeArg name (GetKey)
OpIRCreator* OP_IR_DISPATCHER_CLASS(NodeName)::Find(const Node& node) {
  return DispatcherBase::Get(GetKey(&node));
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
