// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/scaler.h"

/**
https://github.com/onnx/onnx/blob/master/onnx/defs/traditionalml/defs.cc
ONNX_OPERATOR_SCHEMA(Scaler)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
  Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
  )DOC")
.Input(0, "X", "Data to be scaled", "T")
.Output(0, "Y", "Scaled output data", "tensor(float)")
.TypeConstraint(
  "T",
  { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
  " allowed types.")
.Attr(
  "scale",
  "second, multiply by this, can be length of features or length 1",
  AttributeProto::FLOATS,
  OPTIONAL)
.Attr(
  "offset",
  "first, offset by this, must be same length as scale",
  AttributeProto::FLOATS,
  OPTIONAL);
*/
using namespace ::onnxruntime::common;
using namespace std;
namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    Scaler,
    1,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).MayInplace(0, 0),
    ScalerOp<float>);

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    Scaler,
    1,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()).MayInplace(0, 0),
    ScalerOp<double>);

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    Scaler,
    1,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()).MayInplace(0, 0),
    ScalerOp<int64_t>);

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    Scaler,
    1,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()).MayInplace(0, 0),
    ScalerOp<int32_t>);

static constexpr int kParallelizationThreshold = 10 * 1000;

template <typename T>
ScalerOp<T>::ScalerOp(const OpKernelInfo& info) : OpKernel(info),
                                                  scale_(info.GetAttrsOrDefault<float>("scale")),
                                                  offset_(info.GetAttrsOrDefault<float>("offset")) {
  ORT_ENFORCE(!scale_.empty(), "Empty scale in attributes");
  ORT_ENFORCE(scale_.size() == offset_.size(),
              "Scale size: (" + std::to_string(scale_.size()) + ") != (" + std::to_string(offset_.size()) + ")");
}

template <typename T>
common::Status ScalerOp<T>::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const TensorShape& x_shape = X.Shape();
  Tensor* Y = context->Output(0, x_shape);
  const T* x_data = X.template Data<T>();
  auto* y_data = Y->template MutableData<float>();
  auto x_dims = x_shape.GetDims();
  if (x_dims.empty()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid argument: input has empty dimensions.");
  }

  size_t x_size = x_shape.Size();
  int64_t stride = x_dims.size() == 1 ? x_dims[0] : x_dims[1];
  auto* ttp = context->GetOperatorThreadPool();
  auto conditional_batch_call = [ttp, x_size](std::function<void(ptrdiff_t)> f) {
    if (x_size < kParallelizationThreshold) {  // TODO: tune this, arbitrary threshold
      for (size_t i = 0; i < x_size; ++i) {
        f(i);
      }
    } else {
      concurrency::ThreadPool::TryBatchParallelFor(ttp, x_size, f, 0);
    }
  };

  if (static_cast<int64_t>(offset_.size()) == stride &&
      static_cast<int64_t>(scale_.size()) == stride) {
    auto fn = [this, y_data, x_data, stride](ptrdiff_t i) {
      y_data[i] = static_cast<float>((x_data[i] - offset_[i % stride]) * scale_[i % stride]);
    };
    conditional_batch_call(fn);
  } else if (offset_.size() == 1 && scale_.size() == 1) {
    auto fn = [this, y_data, x_data](ptrdiff_t i) {
      y_data[i] = static_cast<float>((x_data[i] - offset_[0]) * scale_[0]);
    };
    conditional_batch_call(fn);
  } else {
    std::ostringstream err_msg;
    err_msg << "Either both scale and offset can be of feature size (" << stride << ") or 1";
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, err_msg.str());
  }
  return Status::OK();
}
}  // namespace ml
}  // namespace onnxruntime
