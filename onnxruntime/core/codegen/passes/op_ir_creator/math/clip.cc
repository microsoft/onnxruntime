// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/math/unary_ops.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of Clip OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Clip)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);

  int version = ctx_codegen.GetCodeGenHandle()->domain_version_lookup_func(node.Domain());
  tvm::Expr min_value, max_value;
  if (version < 11) {
    float max_v, min_v;
    info.GetAttrOrDefault("min", &min_v, std::numeric_limits<float>::lowest());
    info.GetAttrOrDefault("max", &max_v, std::numeric_limits<float>::max());
    min_value = tvm::make_const(tvm::Float(32), min_v);
    max_value = tvm::make_const(tvm::Float(32), max_v);
  } else {
    // for op_version >= 11, max and min are optional inputs
    min_value = tvm::make_const(tvm::Float(32), std::numeric_limits<float>::lowest());
    max_value = tvm::make_const(tvm::Float(32), std::numeric_limits<float>::max());
    auto num_inputs = inputs.size();
    if (num_inputs >= 2 && inputs[1].defined()) {
      min_value = inputs[1]();
    }
    if (num_inputs == 3 && inputs[2].defined()) {
      max_value = inputs[2]();
    }
  }

  tvm::Tensor Y = Clip(inputs[0], min_value, max_value, node.Name() + "_Clip");
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
