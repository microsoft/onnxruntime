// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/nn/pool_ops.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/cpu/nn/pool_attributes.h"

namespace onnxruntime {
namespace tvm_codegen {

// A local macro to create Pool Ops

// helper macro defines Evaluate of of POOL_OP OpIRCreators
#define POOL_OP(name)                                                                                                         \
  Status GENERIC_OP_IR_CREATOR_CLASS(name)::Evaluate(                                                                         \
      const tvm::Array<tvm::Tensor>& inputs,                                                                                  \
      const Node& node,                                                                                                       \
      CodeGenContext& ctx_codegen,                                                                                            \
      tvm::Array<tvm::Tensor>& outputs) {                                                                                     \
    ORT_RETURN_IF_NOT(outputs.size() == 1, "multiple outputs are not supported yet!");                                        \
    ProtoHelperNodeContext ctx(node);                                                                                         \
    OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);                                                                     \
    int version = ctx_codegen.GetCodeGenHandle()->domain_version_lookup_func(node.Domain());                                  \
    PoolAttributes pool_attrs(info, #name, version);                                                                          \
    for (auto n : pool_attrs.dilations) {                                                                                     \
      ORT_RETURN_IF_NOT(n <= 1, "dilations are not supported yet!");                                                          \
    }                                                                                                                         \
    if (pool_attrs.global_pooling) {                                                                                          \
      if (inputs[0]->shape.size() != 4) {                                                                                     \
        ORT_NOT_IMPLEMENTED(gsl::narrow_cast<int64_t>(inputs[0]->shape.size()) - 2, "d global pooling is not implementated"); \
      }                                                                                                                       \
    } else {                                                                                                                  \
      if (pool_attrs.kernel_shape.size() != 2) {                                                                              \
        ORT_NOT_IMPLEMENTED(pool_attrs.kernel_shape.size(), "d pooling is not implementated");                                \
      }                                                                                                                       \
    }                                                                                                                         \
    tvm::Array<tvm::Expr> dummy_output_shape;                                                                                 \
    tvm::Tensor Y = name(inputs[0], pool_attrs, dummy_output_shape);                                                          \
    outputs.push_back(Y);                                                                                                     \
    return Status::OK();                                                                                                      \
  }

LIST_POOL_OPS()

#undef POOL_OP

}  // namespace tvm_codegen
}  // namespace onnxruntime
