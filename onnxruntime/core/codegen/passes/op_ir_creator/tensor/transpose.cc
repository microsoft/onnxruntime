// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/transpose.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of Transpose OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Transpose)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext&,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

  size_t input_0_shape_rank = inputs[0]->shape.size();
  std::vector<int64_t> permute;
  bool is_ok = attrs.GetAttrs<int64_t>("perm", permute).IsOK();
  if (permute.size() != 0 && permute.size() != input_0_shape_rank)
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Transpose: Incorrect permute size");

  std::vector<int64_t> default_permute;
  const std::vector<int64_t>* perm;
  // either we don't have perm attribute or the perm attribute is empty
  bool use_default_perm = !is_ok || permute.size() == 0;
  if (use_default_perm) {
    default_permute.resize(input_0_shape_rank);
    for (size_t i = 0; i < input_0_shape_rank; ++i) {
      default_permute[i] = gsl::narrow<int64_t>(input_0_shape_rank - 1 - i);
    }
    perm = &default_permute;
  } else {
    perm = &permute;
  }

  tvm::Tensor Y = Transpose(inputs[0], ToTvmArrayInt(*perm), node.Name() + "_Transpose");
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
