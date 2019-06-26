// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/target/generic/op_ir_creator/all_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/nn/pool_ops.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

// helper class for pool_ops with arguments
class FuncWithPoolingArgument {
 public:
  FuncWithPoolingArgument(const Node& node, const std::string& op_name) {
    ProtoHelperNodeContext ctx(node);
    OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);
    int64_t storage_order{0};  // MaxPool_8 only. 0 is row major, and 1 is column major. Default is 0.

    ORT_ENFORCE(info.GetAttrs<int64_t>("kernel_shape", kernel_shape_).IsOK(), "No kernel shape is set.");
    if (kernel_shape_.size() != 2)
      ORT_NOT_IMPLEMENTED(kernel_shape_.size(), "d pooling is not implementated");
    if (!info.GetAttrs<int64_t>("pads", pads_).IsOK() || pads_.empty()) {
      pads_.resize(kernel_shape_.size() * 2, 0);
    }
    if (!info.GetAttrs<int64_t>("strides", strides_).IsOK() || strides_.empty()) {
      strides_.resize(kernel_shape_.size(), 1);
    }
    if (op_name == "AveragePool") {
      int64_t temp;
      ORT_ENFORCE(info.GetAttr<int64_t>("count_include_pad", &temp).IsOK());
      count_include_pad_ = (temp != 0);
    }

    if (op_name == "MaxPool") {
      // TODO: add version check or not? remove version check since only after version 8 would have storage_order, otherwise, it would be zero
      storage_order = info.GetAttrOrDefault<int64_t>("storage_order", 0 /*default_value*/);
      if (storage_order != 1) {
        layout_ = "NCWH";
      }
    }
  }

  std::vector<int64_t> kernel_shape_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  std::string layout_ = "NCHW";
  bool count_include_pad_ = false;
};

// A local macro to create Pool Ops

// helper macro defines Evaluate of of POOL_OP OpIRCreators
#define POOL_OP(name)                                                                                                                                                         \
  Status GENERIC_OP_IR_CREATOR_CLASS(name)::Evaluate(                                                                                                                         \
      const tvm::Array<tvm::Tensor>& inputs,                                                                                                                                  \
      const Node& node,                                                                                                                                                       \
      CodeGenContext&,                                                                                                                                                        \
      tvm::Array<tvm::Tensor>& outputs) {                                                                                                                                     \
    if (outputs.size() > 1) ORT_NOT_IMPLEMENTED("output size = 2 is not implementated");                                                                                      \
    FuncWithPoolingArgument argment(node, #name);                                                                                                                             \
    tvm::Tensor Y = name(inputs[0], ToTvmArray(argment.kernel_shape_), ToTvmArray(argment.strides_), ToTvmArray(argment.pads_), argment.layout_, argment.count_include_pad_); \
    outputs.push_back(Y);                                                                                                                                                     \
    return Status::OK();                                                                                                                                                      \
  }  // namespace tvm_codegen

POOL_OP(MaxPool)
POOL_OP(AveragePool)

#undef POOL_OP

// helper macro defines Evaluate of of GlobalPOOL_OP OpIRCreators
#define POOL_OP(name)                                                                                                       \
  Status GENERIC_OP_IR_CREATOR_CLASS(name)::Evaluate(                                                                       \
      const tvm::Array<tvm::Tensor>& inputs,                                                                                \
      const Node& node,                                                                                                     \
      CodeGenContext&,                                                                                                      \
      tvm::Array<tvm::Tensor>& outputs) {                                                                                   \
    if (inputs[0]->shape.size() != 4)                                                                                       \
      ORT_NOT_IMPLEMENTED(gsl::narrow_cast<int64_t>(inputs[0]->shape.size()) - 2, "d global pooling is not implementated"); \
    tvm::Tensor Y = name(inputs[0], "NCHW");                                                                                \
    outputs.push_back(Y);                                                                                                   \
    return Status::OK();                                                                                                    \
  }

POOL_OP(GlobalMaxPool)
POOL_OP(GlobalAveragePool)

#undef POOL_OP

}  // namespace tvm_codegen
}  // namespace onnxruntime
