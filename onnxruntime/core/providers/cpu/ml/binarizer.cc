// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/binarizer.h"
#include <cmath>
/**
https://github.com/onnx/onnx/blob/main/onnx/defs/traditionalml/defs.cc

ONNX_OPERATOR_SCHEMA(Binarizer)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
Makes values 1 or 0 based on a single threshold.
)DOC")
.Input(0, "X", "Data to be binarized", "T")
.Output(0, "Y", "Binarized output data", "T")
.TypeConstraint(
"T",
{ "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
" allowed types.")
.Attr(
"threshold",
"Values greater than this are set to 1, else set to 0",
AttributeProto::FLOAT,
OPTIONAL);
*/

namespace onnxruntime {
namespace ml {
ONNX_CPU_OPERATOR_ML_KERNEL(
    Binarizer,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    BinarizerOp<float>);

template <typename T>
BinarizerOp<T>::BinarizerOp(const OpKernelInfo& info)
    : OpKernel(info), threshold_(info.GetAttrOrDefault<float>("threshold", 1.0f)) {}

template <typename T>
common::Status BinarizerOp<T>::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const TensorShape& x_shape = X.Shape();
  Tensor* Y = context->Output(0, x_shape);
  const T* x_data = X.Data<T>();
  T* y_data = Y->MutableData<T>();
  size_t x_size = onnxruntime::narrow<size_t>(x_shape.Size());

  common::Status status = common::Status::OK();
  for (size_t i = 0; i < x_size; ++i) {
    T x_val = x_data[i];
    T& y_val = y_data[i];

    auto tmp = static_cast<float>(x_val);  // this cast is necessary because isnan doesn't work otherwise.
    if (std::isnan(tmp)) {
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Input data with index: " + std::to_string(i) + " is NaN");
    }
    y_val = x_val > threshold_ ? static_cast<T>(1) : static_cast<T>(0);
  }
  return status;
}
}  // namespace ml
}  // namespace onnxruntime
