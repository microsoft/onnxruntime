// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

#include "core/codegen/mti/tensor/tile.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/common.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"

namespace onnxruntime {
namespace nuphar {

// Evaluate of Tile OpIRCreator
Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(Tile)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);
  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);
  const auto* repeats = ctx_nuphar->GetOrtInitializerTensor(node.InputDefs()[1]->Name());
  ORT_RETURN_IF_NOT(repeats != nullptr, "repeats == nullptr");
  ORT_RETURN_IF_NOT(repeats->Shape().Size() == gsl::narrow<int64_t>(inputs[0]->shape.size()),
                    "repeats->Shape().Size() != inputs[0]->shape.size()");
  const int64_t* repeats_data = repeats->Data<int64_t>();
  const auto repeats_vector = std::vector<int64_t>(repeats_data, repeats_data + inputs[0]->shape.size());
  tvm::Tensor Y = tvm_codegen::Tile(inputs[0], repeats_vector, node.Name() + "_Tile");
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
