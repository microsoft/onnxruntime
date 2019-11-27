// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

#include "core/codegen/mti/tensor/tile.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/common.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"

namespace onnxruntime {
namespace tvm_codegen {

// Forwarding
Status SliceCommon(const tvm::Array<tvm::Tensor>& inputs,
                   const Node& node,
                   tvm::Array<tvm::Tensor>& outputs,
                   const std::vector<int64_t>& starts,
                   const std::vector<int64_t>& ends,
                   const std::vector<int64_t>& axes,
                   const std::vector<int64_t>& steps);

}  // namespace tvm_codegen

namespace nuphar {

// Evaluate of Slice OpIRCreator
Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(Slice)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);
  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);

  std::vector<std::vector<int64_t>> slice_params;
  int version = ctx_codegen.GetCodeGenHandle()->domain_version_lookup_func(node.Domain());
  if (version <= 9) {
    std::vector<int64_t> starts, ends, axes, steps;
    ORT_RETURN_IF_ERROR(info.GetAttrs<int64_t>("starts", starts));
    ORT_RETURN_IF_ERROR(info.GetAttrs<int64_t>("ends", ends));
    ORT_RETURN_IF_NOT(starts.size() == ends.size());
    axes = info.GetAttrsOrDefault<int64_t>("axes");
    slice_params.push_back(starts);
    slice_params.push_back(ends);
    slice_params.push_back(axes);
    slice_params.push_back(steps);
  } else {
    // for opset 10 Slice, input 1/2/3/4 are starts/ends/axes/steps
    // while axes and steps are optional
    for (size_t i = 1; i < 5; ++i) {
      if (i < node.InputDefs().size()) {
        const auto* tensor = ctx_nuphar->GetOrtInitializerTensor(node.InputDefs()[i]->Name());
        if (tensor) {
          if (utils::IsPrimitiveDataType<int64_t>(tensor->DataType())) {
            const int64_t* data = tensor->Data<int64_t>();
            slice_params.push_back(std::vector<int64_t>(data, data + tensor->Shape().Size()));
          } else {
            const int32_t* data = tensor->Data<int32_t>();
            slice_params.push_back(std::vector<int64_t>(data, data + tensor->Shape().Size()));
          }
          continue;
        }
      }
      slice_params.push_back(std::vector<int64_t>());
    }
  }
  return tvm_codegen::SliceCommon(inputs, node, outputs, slice_params[0], slice_params[1], slice_params[2], slice_params[3]);
}

}  // namespace nuphar
}  // namespace onnxruntime
