// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

#include "core/codegen/passes/utils/ort_tvm_utils.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/providers/nuphar/mti_x86/nn/pool_ops.h"

namespace onnxruntime {
namespace nuphar {

static tvm::Array<tvm::Expr> GetOutputShapeAndPads(const Node& node,
                                                   PoolAttributes& pool_attrs,
                                                   tvm_codegen::CodeGenContext& ctx_codegen) {
  const NodeArg* input = node.InputDefs()[0];
  ORT_ENFORCE(input);
  const ONNX_NAMESPACE::TensorShapeProto* shape_proto = input->Shape();
  size_t num_input_dims = shape_proto->dim_size();
  ORT_ENFORCE(num_input_dims >= 2);

  tvm::Array<tvm::Expr> output_shape;
  // batch dimenion
  output_shape.push_back(ShapeDimToTvmDim(shape_proto->dim(0), ctx_codegen));
  // output channel
  output_shape.push_back(ShapeDimToTvmDim(shape_proto->dim(1), ctx_codegen));

  size_t kernel_sz = pool_attrs.kernel_shape.size();
  if (pool_attrs.global_pooling) {
    pool_attrs.pads.assign(kernel_sz, 0);
    // skip batch and channel dimensions, so dim starts from 2
    for (size_t dim = 2; dim < num_input_dims; dim++) {
      output_shape.push_back(tvm::make_const(tvm::Int(32), 1));
    }
  } else {
    ORT_ENFORCE(num_input_dims > kernel_sz);
    size_t kernel_idx_offset = num_input_dims - kernel_sz;
    for (size_t dim = 0; dim < kernel_sz; dim++) {
      // TODO: handle symbolic dimensions
      ORT_ENFORCE(ShapeHasValue(input, dim + kernel_idx_offset));
      int64_t dim_val = ShapeValue(input, dim + kernel_idx_offset);
      int64_t dim_size = 0;
      pool_attrs.ComputeSizePadDilations(static_cast<int>(dim_val),
                                         pool_attrs.strides[dim],
                                         pool_attrs.kernel_shape[dim],
                                         &(pool_attrs.pads[dim]),
                                         &(pool_attrs.pads[kernel_sz + dim]),
                                         pool_attrs.dilations[dim],
                                         &dim_size);
      output_shape.push_back(tvm::make_const(tvm::Int(32), dim_size));
    }
  }
  return output_shape;
}

#define POOL_OP(name)                                                                             \
  Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(name)::Evaluate(                                      \
      const tvm::Array<tvm::Tensor>& inputs,                                                      \
      const Node& node,                                                                           \
      tvm_codegen::CodeGenContext& ctx_codegen,                                                   \
      tvm::Array<tvm::Tensor>& outputs) {                                                         \
    ORT_RETURN_IF_NOT(node.OutputDefs().size() == 1, " multiple outputs are not supported yet!"); \
    ORT_RETURN_IF_NOT(inputs[0]->dtype == HalideIR::Float(32), " non-float32 not supported yet"); \
    ProtoHelperNodeContext ctx(node);                                                             \
    OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);                                         \
    int version = ctx_codegen.GetCodeGenHandle()->domain_version_lookup_func(node.Domain());      \
    PoolAttributes pool_attrs(info, #name, version);                                              \
    for (auto n : pool_attrs.dilations) {                                                         \
      ORT_RETURN_IF_NOT(n <= 1, "dilations are not supported yet!");                              \
    }                                                                                             \
    tvm::Array<tvm::Expr> output_shape = GetOutputShapeAndPads(node, pool_attrs, ctx_codegen);    \
    tvm::Tensor Y = name(inputs[0], pool_attrs, output_shape);                                    \
    outputs.push_back(Y);                                                                         \
    return Status::OK();                                                                          \
  }                                                                                               \

LIST_X86_POOL_OPS()

#undef POOL_OP

}  // namespace nuphar
}  // namespace onnxruntime
