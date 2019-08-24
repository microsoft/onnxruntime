// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/passes/utils/codegen_context.h"
#include "core/codegen/passes/op_ir_creator/tvm_op_creator.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace tvm_codegen {

// TVMIRBuilder contains all applicable TVM OpIRCreators
// OpIRCreators are stored in multiple dispatchers
// that check different conditions of an ORT Node.

// If an ORT Node satisfies more than one OpIRCreators,
// the first dispatched pass will be applied.

class TVMIRBuilder {
 public:
  TVMIRBuilder(const std::string& name);
  ~TVMIRBuilder() = default;

  // A debug dumps all existing in this TVMIRBuilders
  void DumpAllOpIRCreators() const;

  // Evaluates an OpIRCreator that first satisfies condtions of all dispatchers
  Status Evaluate(
      const tvm::Array<tvm::Tensor>& inputs,
      const Node& node,
      CodeGenContext& ctx,
      tvm::Array<tvm::Tensor>& outputs);

  // Inserts a dispatcher and move its ownership to this TVMIRBuilder
  void InsertDispatcher(std::unique_ptr<OpIRDispatcher>&& ptr);

  // Clears all dispatchers in this TVMIRBuilder
  void ClearAllDispatchers();

  // Dumps the name of this TVMIRBuilder
  const std::string& Name() const;

 private:
  std::vector<std::unique_ptr<OpIRDispatcher>> dispatchers_;
  std::string name_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TVMIRBuilder);
};

// Utility function to register all builtin generic OpIRCreators into an OpIRRegistry.
// It creates instances of all generic OpIRCreators
// and registers them to op_ir_registry
void RegisterAllGenericOpIRCreators(OpIRRegistry* op_ir_registry);

// Utility function to bind all builtin generic OpIRCreators to a TVMIRBuilder.
// It creates an instance of a Dispatcher that contains all generic OpIRCreators created above
// and uses OrtOpType to dispatch OpIRCreators.
// Then, it registers the created Dispatcher to a TVMIRBuilder, builder.
void RegisterGenericOrtOpTypeDispatcher(const std::shared_ptr<TVMIRBuilder>& builder,
                                        const OpIRRegistry* registry);

}  // namespace tvm_codegen
}  // namespace onnxruntime
