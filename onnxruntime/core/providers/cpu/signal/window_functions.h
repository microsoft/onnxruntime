// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class VariableOutputDataTypeBase : public OpKernel {
 protected:
  onnx::TensorProto_DataType data_type_;

 public:
  VariableOutputDataTypeBase(const OpKernelInfo& info) : OpKernel(info) {
    data_type_ = static_cast<onnx::TensorProto_DataType>(info.GetAttrOrDefault<int64_t>("output_datatype", onnx::TensorProto_DataType::TensorProto_DataType_FLOAT));
  }
};

class MelWeightMatrix final : public VariableOutputDataTypeBase {
 public:
  explicit MelWeightMatrix(const OpKernelInfo& info) : VariableOutputDataTypeBase(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

}  // namespace onnxruntime
