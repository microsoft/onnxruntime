// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {
namespace contrib {

class WindowFunctionBase : public OpKernel {
 protected:
  onnx::TensorProto_DataType data_type_;

 public:
  WindowFunctionBase(const OpKernelInfo& info) : OpKernel(info) {
    data_type_ = static_cast<onnx::TensorProto_DataType>(info.GetAttrOrDefault<int64_t>("border", onnx::TensorProto_DataType::TensorProto_DataType_FLOAT));
  }
};

class HannWindow final : public WindowFunctionBase {
 public:
  explicit HannWindow(const OpKernelInfo& info) : WindowFunctionBase(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

class HammingWindow final : public WindowFunctionBase {
 public:
  explicit HammingWindow(const OpKernelInfo& info) : WindowFunctionBase(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

class BlackmanWindow final : public WindowFunctionBase {
 public:
  explicit BlackmanWindow(const OpKernelInfo& info) : WindowFunctionBase(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
