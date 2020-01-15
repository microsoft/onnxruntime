// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/expand.h"
#include "core/codegen/passes/utils/ort_tvm_utils.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of Expand OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Expand)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  tvm::Tensor Y = Expand(inputs[0], ShapeToTvmArray(node.OutputDefs()[0], ctx_codegen), node.Name() + "_Expand");
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
