// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/tensor/cast_ops.h"
#include "core/codegen/passes/utils/ort_tvm_utils.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of Cast OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Cast)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext&,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

  int64_t to;
  ORT_RETURN_IF_ERROR(attrs.GetAttr<int64_t>("to", &to));
  auto to_type_proto = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);

  tvm::Tensor X = inputs[0];
  tvm::Tensor Y;
  if (to_type_proto == ONNX_NAMESPACE::TensorProto_DataType_BOOL) {
    // special case for bool as ONNX bool is uint8, while in tvm it's uint1
    Y = CastToUInt8Bool(X, node.Name() + "_Cast");
  } else {
    Y = Cast(X, ToTvmType(to_type_proto), node.Name() + "_Cast");
  }

  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
