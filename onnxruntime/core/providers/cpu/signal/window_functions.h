// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class VariableOutputDataTypeBase : public OpKernel {
 protected:
  onnx::TensorProto_DataType data_type_;

 public:
  explicit VariableOutputDataTypeBase(const OpKernelInfo& info) : OpKernel(info) {
    data_type_ = static_cast<onnx::TensorProto_DataType>(  //
        info.GetAttrOrDefault<int64_t>("output_datatype", onnx::TensorProto_DataType::TensorProto_DataType_FLOAT));
  }
};

class HannWindow final : public VariableOutputDataTypeBase {
 public:
  explicit HannWindow(const OpKernelInfo& info) : VariableOutputDataTypeBase(info) {
    is_periodic_ = static_cast<bool>(info.GetAttrOrDefault<int64_t>("periodic", 1));
  }
  Status Compute(OpKernelContext* ctx) const override;

 private:
  bool is_periodic_ = true;
};

class HammingWindow final : public VariableOutputDataTypeBase {
 public:
  explicit HammingWindow(const OpKernelInfo& info) : VariableOutputDataTypeBase(info) {
    is_periodic_ = static_cast<bool>(info.GetAttrOrDefault<int64_t>("periodic", 1));
  }
  Status Compute(OpKernelContext* ctx) const override;

 private:
  bool is_periodic_ = true;
};

class BlackmanWindow final : public VariableOutputDataTypeBase {
 public:
  explicit BlackmanWindow(const OpKernelInfo& info) : VariableOutputDataTypeBase(info) {
    is_periodic_ = static_cast<bool>(info.GetAttrOrDefault<int64_t>("periodic", 1));
  }
  Status Compute(OpKernelContext* ctx) const override;

 private:
  bool is_periodic_ = true;
};

class MelWeightMatrix final : public VariableOutputDataTypeBase {
 public:
  explicit MelWeightMatrix(const OpKernelInfo& info) : VariableOutputDataTypeBase(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

}  // namespace onnxruntime
