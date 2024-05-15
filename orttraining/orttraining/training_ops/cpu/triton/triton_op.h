// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRITON

#pragma once

#include "core/common/inlined_containers.h"

#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {
namespace contrib {

class TritonOp final : public OpKernel {
 public:
  TritonOp(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(info.GetAttr("func_name", &func_name_));
    ORT_THROW_IF_ERROR(info.GetAttr("onnx_key", &onnx_key_));
    ORT_THROW_IF_ERROR(info.GetAttr("onnx_string", &onnx_string_));
    for (const auto& attr : info.node().GetAttributes()) {
      if (attr.first.rfind("_", 0) == 0 || attr.first == "func_name" || attr.first == "onnx_key" ||
          attr.first == "onnx_string") {
        continue;
      }
      // Support int64, float and string only for now, skip other types.
      if (attr.second.type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_INT) {
        kwargs_.insert({attr.first, {std::to_string(attr.second.i()), ONNX_NAMESPACE::TensorProto_DataType_INT64}});
      } else if (attr.second.type() ==
                 ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOAT) {
        kwargs_.insert({attr.first, {std::to_string(attr.second.f()), ONNX_NAMESPACE::TensorProto_DataType_FLOAT}});
      } else if (attr.second.type() ==
                 ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_STRING) {
        kwargs_.insert({attr.first, {attr.second.s(), ONNX_NAMESPACE::TensorProto_DataType_STRING}});
      }
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  InlinedHashSet<size_t> GetBoolOutputs(size_t output_size) const;

  std::string func_name_;
  int64_t onnx_key_;
  std::string onnx_string_;
  InlinedHashMap<std::string, std::pair<std::string, int>> kwargs_;
};

bool IsTritonOpExecutorInitialized();
Status ExecuteTritonOpByFuncName(OpKernelContext* p_ctx, const std::string& func_name, size_t input_count,
                                 size_t output_count,
                                 const InlinedHashMap<std::string, std::pair<std::string, int>>& kwargs);

}  // namespace contrib
}  // namespace onnxruntime

#endif
