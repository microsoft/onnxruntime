// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/reshape_ops.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/common.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"

namespace onnxruntime {
namespace nuphar {

// Evaluate of Dropout OpIRCreator
Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(Dropout)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  tvm::Tensor Y = tvm_codegen::Identity(inputs[0]);
  outputs.push_back(Y);

  // optional mask
  // Support skipped trailing outputs
  if (node.OutputDefs().size() > 1 && node.OutputDefs()[1]->Exists()) {
    NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);
    int version = ctx_nuphar->GetCodeGenHandle()->domain_version_lookup_func(node.Domain());
    if (version >= 12) {
      // A fake mask with all ones for opset 12+
      auto l = [&](const tvm::Array<tvm::Var>& /*indices*/) {
        return tvm::make_const(tvm::UInt(8), 1);
      };
      tvm::Tensor mask = tvm::compute(inputs[0]->shape, l, "mask");
      outputs.push_back(mask);
    } else {
      // for opset < 12, masks are all zero
      tvm::Tensor mask = tvm_codegen::MakeZeroTensor(inputs[0]->shape, inputs[0]->dtype, "mask");
      outputs.push_back(mask);
    }
  }

  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
