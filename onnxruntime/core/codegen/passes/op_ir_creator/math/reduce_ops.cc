// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/common/op_macro.h"
#include "core/codegen/mti/math/reduce_ops.h"
#include "core/codegen/mti/tensor/cast_ops.h"
#include "core/codegen/mti/tensor/reshape_ops.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace tvm_codegen {

using ReduceIndexedFunc = tvm::Tensor (*)(const tvm::Tensor& X, int64_t axis, bool keep_dims, const std::string& name);
using ReduceFunc = tvm::Tensor (*)(const tvm::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name);

// helper class for for REDUCE_INDEXED_OP
class FuncReduceIndexed {
 public:
  FuncReduceIndexed(const Node& node, ReduceIndexedFunc func, const std::string& name) {
    ProtoHelperNodeContext ctx(node);
    OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 0);
    int64_t keepdims_i = 1;
    ORT_ENFORCE(info.GetAttr("keepdims", &keepdims_i).IsOK());
    keep_dims_ = (keepdims_i == 1);
    func_ = func;
    name_ = name;
  }

  tvm::Tensor operator()(const tvm::Tensor& X) const {
    auto axis = HandleNegativeAxis(axis_, gsl::narrow_cast<int64_t>(X->shape.size()));
    tvm::Tensor index32 = func_(X, axis, keep_dims_, name_);
    return Cast(index32, tvm::Int(64));
  }

 private:
  int64_t axis_;
  bool keep_dims_;
  ReduceIndexedFunc func_;
  std::string name_;
};

// helper class for REDUCE_OP
class FuncReduce {
 public:
  FuncReduce(const Node& node, ReduceFunc func, const std::string& name) {
    ProtoHelperNodeContext ctx(node);
    OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);
    axes_ = info.GetAttrsOrDefault<int64_t>("axes");
    int64_t keepdims_i = 1;
    ORT_ENFORCE(info.GetAttr("keepdims", &keepdims_i).IsOK());
    keep_dims_ = (keepdims_i == 1);
    func_ = func;
    name_ = name;
  }

  tvm::Tensor operator()(const tvm::Tensor& X) const {
    std::vector<int64_t> axes;
    for (auto i : axes_)
      axes.push_back(HandleNegativeAxis(i, gsl::narrow_cast<int64_t>(X->shape.size())));

    return func_(X, axes, keep_dims_, name_);
  }

 private:
  std::vector<int64_t> axes_;
  bool keep_dims_;
  ReduceFunc func_;
  std::string name_;
};

// helper macro defines Evaluate of REDUCE_OP OpIRCreators
#define REDUCE_OP(name)                                             \
  Status GENERIC_OP_IR_CREATOR_CLASS(name)::Evaluate(               \
      const tvm::Array<tvm::Tensor>& inputs,                        \
      const Node& node,                                             \
      CodeGenContext&,                                              \
      tvm::Array<tvm::Tensor>& outputs) {                           \
    tvm::Tensor Y;                                                  \
    if (ShapeRank(node.OutputDefs()[0]) == 0) {                     \
      tvm::Tensor temp = FuncReduce(node, &name, #name)(inputs[0]); \
      Y = Reshape(temp, {});                                        \
    } else {                                                        \
      Y = FuncReduce(node, &name, #name)(inputs[0]);                \
    }                                                               \
    outputs.push_back(Y);                                           \
    return Status::OK();                                            \
  }

// helper macro defines Evaluate of REDUCE_INDEXED_OP OpIRCreators
#define REDUCE_INDEXED_OP(name)                                       \
  Status GENERIC_OP_IR_CREATOR_CLASS(name)::Evaluate(                 \
      const tvm::Array<tvm::Tensor>& inputs,                          \
      const Node& node,                                               \
      CodeGenContext&,                                                \
      tvm::Array<tvm::Tensor>& outputs) {                             \
    tvm::Tensor Y = FuncReduceIndexed(node, &name, #name)(inputs[0]); \
    outputs.push_back(Y);                                             \
    return Status::OK();                                              \
  }

LIST_REDUCE_OPS()

#undef REDUCE_OP
#undef REDUCE_INDEXED_OP

}  // namespace tvm_codegen
}  // namespace onnxruntime
