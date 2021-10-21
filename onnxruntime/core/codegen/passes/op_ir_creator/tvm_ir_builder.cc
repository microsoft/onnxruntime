// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/tvm_ir_builder.h"

#include "core/codegen/common/op_macro.h"
#include "core/codegen/passes/op_ir_creator/all_ops.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace tvm_codegen {

TVMIRBuilder::TVMIRBuilder(const std::string& name)
    : name_(name) {}

const std::string& TVMIRBuilder::Name() const {
  return name_;
}

void TVMIRBuilder::InsertDispatcher(std::unique_ptr<OpIRDispatcher>&& ptr) {
  dispatchers_.push_back(std::move(ptr));
}

void TVMIRBuilder::ClearAllDispatchers() {
  dispatchers_.clear();
}

void TVMIRBuilder::DumpAllOpIRCreators() const {
  int count = 0;
  for (auto& d : dispatchers_) {
    std::cout << "************ TVM OpIRDispatcher "
              << count << " : "
              << d->Name()
              << " ************" << std::endl;

    d->ForEach([](const std::string& key, OpIRCreator* builder) {
      std::cout << "Key " << key
                << ", Creator " << builder->Name() << std::endl;
    });

    ++count;
  }
}

// Evaluate finds ONE proper OpIRCreator and build the corresponding OpIR
// If a TVMIRBuilder has more than one OpIRCreator for an ORT Op,
// the first one will be used.
// Please adjust registration order and dispatcher in TVMIRBuilder
// to make sure the proper OpIRCreator is called.
Status TVMIRBuilder::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  OpIRCreator* candidate = nullptr;
  for (auto& d : dispatchers_) {
    candidate = d->Find(node);
    if (nullptr != candidate)
      break;
  }

  if (nullptr == candidate) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Not implemented: ", node.OpType());
  }

  ORT_RETURN_IF_ERROR(candidate->Evaluate(inputs, node, ctx_codegen, outputs));

  return Status::OK();
}

// BEGIN: Generic IR creator classes
#define ADD_OP_ITEM(name) \
  op_ir_registry->Register(std::make_unique<GENERIC_OP_IR_CREATOR_CLASS(name)>());

#define BINARY_OP(name) ADD_OP_ITEM(name)
#define BINARY_CMP_OP(name) ADD_OP_ITEM(name)
#define POOL_OP(name) ADD_OP_ITEM(name)
#define REDUCE_OP(name) ADD_OP_ITEM(name)
#define REDUCE_INDEXED_OP(name) ADD_OP_ITEM(name)
#define UNARY_OP(name) ADD_OP_ITEM(name)
#define VARIADIC_OP(name) ADD_OP_ITEM(name)

void RegisterAllGenericOpIRCreators(OpIRRegistry* op_ir_registry) {
  LIST_ALL_GENERIC_OPS();
}

#undef ADD_OP_ITEM
#undef BINARY_OP
#undef BINARY_CMP_OP
#undef POOL_OP
#undef REDUCE_OP
#undef REDUCE_INDEXED_OP
#undef UNARY_OP
#undef VARIADIC_OP

// BEGIN: Plugin Generic IR creator classes
#define ADD_OP_ITEM(name) \
  dispatcher->Register(#name, registry->Get(GENERIC_OP_IR_CREATOR_STRING(name)));

#define BINARY_OP(name) ADD_OP_ITEM(name)
#define BINARY_CMP_OP(name) ADD_OP_ITEM(name)
#define POOL_OP(name) ADD_OP_ITEM(name)
#define REDUCE_OP(name) ADD_OP_ITEM(name)
#define REDUCE_INDEXED_OP(name) ADD_OP_ITEM(name)
#define UNARY_OP(name) ADD_OP_ITEM(name)
#define VARIADIC_OP(name) ADD_OP_ITEM(name)

void RegisterGenericOrtOpTypeDispatcher(const std::shared_ptr<TVMIRBuilder>& builder,
                                        const OpIRRegistry* registry) {
  auto dispatcher = std::make_unique<OP_IR_DISPATCHER_CLASS(OpType)>("GenericOrtOpTypeOpIRCreators");
  LIST_ALL_GENERIC_OPS()
  builder->InsertDispatcher(std::move(dispatcher));
}

#undef ADD_OP_ITEM
#undef BINARY_OP
#undef BINARY_CMP_OP
#undef POOL_OP
#undef REDUCE_OP
#undef REDUCE_INDEXED_OP
#undef UNARY_OP
// END: Generic IR creators classes

}  // namespace tvm_codegen
}  // namespace onnxruntime
