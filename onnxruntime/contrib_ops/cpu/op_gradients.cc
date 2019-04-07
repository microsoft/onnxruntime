// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_gradients.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/util/math.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "gsl/gsl_algorithm"
#include "gsl/gsl_util"

namespace onnxruntime {
namespace contrib {

std::vector<VectorInt64> InferOutputShapes(OpKernelInfo info) {
  std::vector<VectorInt64> output_tensor_shapes = {};

  auto& node = info.node();
  auto output_defs = node.OutputDefs();
  auto outputCount = output_defs.size();

  for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
    output_tensor_shapes.push_back({});
    if (!output_defs[outputIndex]->Exists())
      continue;

    auto shape = output_defs[outputIndex]->Shape();
    for (auto dim : shape->dim()) {
      output_tensor_shapes[outputIndex].push_back(dim.dim_value());
    }
  }
  return output_tensor_shapes;
}

template <typename T>
auto MakeEigenArrayMap(Tensor& t) { return EigenVectorArrayMap<T>(t.template MutableData<T>(), t.Shape().Size()); }
template <typename T>
auto MakeEigenArrayMap(const Tensor& t) { return ConstEigenVectorArrayMap<T>(t.template Data<T>(), t.Shape().Size()); }

template <typename T>
auto MakeEigenArrayMap(Tensor* t) { return EigenVectorArrayMap<T>(t->template MutableData<T>(), t->Shape().Size()); }
template <typename T>
auto MakeEigenArrayMap(const Tensor* t) { return ConstEigenVectorArrayMap<T>(t->template Data<T>(), t->Shape().Size()); }

ONNX_CPU_OPERATOR_KERNEL(
    SinGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SinGrad<float>);

template <typename T>
Status SinGrad<T>::Compute(OpKernelContext* context) const {
  auto& dY = *context->Input<Tensor>(0);
  auto& X = *context->Input<Tensor>(1);
  auto& dX = *context->Output(0, X.Shape());
  MakeEigenArrayMap<float>(dX) = MakeEigenArrayMap<float>(dY) * MakeEigenArrayMap<float>(X).cos();
  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    MulGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MulGrad<float>);

template <typename T>
Status MulGrad<T>::Compute(OpKernelContext* context) const {
  auto& dZ = *context->Input<Tensor>(0);
  auto& X = *context->Input<Tensor>(1);
  auto& Y = *context->Input<Tensor>(2);

  auto& dX = *context->Output(0, X.Shape());
  auto& dY = *context->Output(1, Y.Shape());

  MakeEigenArrayMap<float>(dX) = MakeEigenArrayMap<float>(dZ) * MakeEigenArrayMap<float>(Y);
  MakeEigenArrayMap<float>(dY) = MakeEigenArrayMap<float>(dZ) * MakeEigenArrayMap<float>(X);

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    FlattenGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FlattenGrad<float>);

template <typename T>
Status FlattenGrad<T>::Compute(OpKernelContext* context) const {
  auto& dZ = *context->Input<Tensor>(0);
  auto& dA = *context->Output(0, dZ.Shape());

  // unimplemented
  MakeEigenArrayMap<float>(dA) = MakeEigenArrayMap<float>(dZ);
  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    UnsqueezeGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    UnsqueezeGrad<float>);

template <typename T>
Status UnsqueezeGrad<T>::Compute(OpKernelContext* context) const {
  auto& dZ = *context->Input<Tensor>(0);
  auto& dA = *context->Output(0, dZ.Shape());

  // unimplemented
  MakeEigenArrayMap<float>(dA) = MakeEigenArrayMap<float>(dZ);
  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    ReluGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReluGrad<float>);

template <typename T>
Status ReluGrad<T>::Compute(OpKernelContext* context) const {
  auto& dY = *context->Input<Tensor>(0);
  auto& Y = *context->Input<Tensor>(1);
  auto& dX = *context->Output(0, dY.Shape());

  EigenVectorArrayMap<float>(dX.template MutableData<T>(), dX.Shape().Size()) =
      (ConstEigenVectorArrayMap<float>(Y.template Data<T>(), Y.Shape().Size()) > T(0))
          .select(ConstEigenVectorArrayMap<float>(dY.template Data<T>(), dY.Shape().Size()), T(0));

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    AddGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    AddGrad<float>);

// The Current implementation assumes X1.Shape() == X2.Shape()
// TODO: Implement Grad for BroadCastAdd
template <typename T>
Status AddGrad<T>::Compute(OpKernelContext* context) const {
  auto dY = context->Input<Tensor>(0);

  if (!output_tensor_shapes_[0].empty()) {
    auto dX1 = context->Output(0, TensorShape::ReinterpretBaseType(output_tensor_shapes_[0]));

    auto out = gsl::make_span(dX1->template MutableData<float>(), dX1->Shape().Size());
    auto in = gsl::make_span(dY->Data<float>(), dY->Shape().Size());

    auto iter = out.begin();
    auto iter2 = in.begin();
    for (; iter != out.end() && iter2 != in.end(); iter++, iter2++) {
      *iter = static_cast<float>(*iter2);
    }
  }

  if (!output_tensor_shapes_[1].empty()) {
    auto dX2 = context->Output(1, TensorShape::ReinterpretBaseType(output_tensor_shapes_[1]));
    MakeEigenArrayMap<float>(dX2) = MakeEigenArrayMap<float>(dY);
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    MatMulGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMulGrad<float>);

// impl supports 2D matrix only
template <typename T>
Status MatMulGrad<T>::Compute(OpKernelContext* context) const {
  auto& dz = *context->Input<Tensor>(0);
  auto& x = *context->Input<Tensor>(1);
  auto& y = *context->Input<Tensor>(2);

  // dx = dz * transpose(y)
  auto rows = dz.Shape().GetDims()[0];
  auto cols = y.Shape().GetDims()[0];

  std::vector<int64_t> dxDims{rows, cols};
  Tensor* dx = context->Output(0, dxDims);

  if (dx) {
    math::Gemm<float, CPUMathUtil>(
        CblasNoTrans,
        CblasTrans,
        rows,
        cols,
        dz.Shape().GetDims()[1],
        /* alpha */ 1.0f,
        dz.template Data<float>(),
        y.template Data<float>(),
        /* beta */ 0.0f,
        dx->template MutableData<float>(),
        &CPUMathUtil::Instance());
  }
  // dy = transpose(x) * y
  rows = x.Shape().GetDims()[1];
  cols = dz.Shape().GetDims()[1];

  std::vector<int64_t> dyDims{rows, cols};
  Tensor* dy = context->Output(1, TensorShape(dyDims));

  if (dy) {
    math::Gemm<float, CPUMathUtil>(
        CblasTrans,
        CblasNoTrans,
        rows,
        cols,
        dz.Shape().GetDims()[0],
        /* alpha */ 1.0f,
        x.template Data<float>(),
        dz.template Data<float>(),
        /* beta */ 0.0f,
        dy->template MutableData<float>(),
        &CPUMathUtil::Instance());
  }
  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    SubGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SubGrad<float>);

template <typename T>
Status SubGrad<T>::Compute(OpKernelContext* context) const {
  auto& dZ = *context->Input<Tensor>(0);

  if (!output_tensor_shapes_[0].empty()) {
    auto dX1 = context->Output(0, TensorShape::ReinterpretBaseType(output_tensor_shapes_[0]));
    MakeEigenArrayMap<float>(dX1) = MakeEigenArrayMap<float>(dZ);
  }

  if (!output_tensor_shapes_[1].empty()) {
    auto dX2 = context->Output(1, TensorShape::ReinterpretBaseType(output_tensor_shapes_[1]));
    MakeEigenArrayMap<float>(dX2) = -1 * MakeEigenArrayMap<float>(dZ);
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    PowGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    PowGrad<float>);

// This is currently implemented for when a is a single element.
template <typename T>
Status PowGrad<T>::Compute(OpKernelContext* context) const {
  auto& dz = *context->Input<Tensor>(0);
  auto& w = *context->Input<Tensor>(1);
  auto& a = *context->Input<Tensor>(2);

  auto& dw = *context->Output(0, w.Shape());

  // df/dw = a * w^(a-1) - all operations are element wise
  float scalarA = a.Data<float>()[0];
  MakeEigenArrayMap<float>(dw) = scalarA * MakeEigenArrayMap<float>(w).pow(scalarA - 1) * MakeEigenArrayMap<float>(dz);

  // df/da =  w^a * ln w
  // this is not implemented yet . needs ln

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    ReduceMeanGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReduceMeanGrad<float>);

// This impl will only work for 1-D tensor input (x), with Do not keepDims option
template <typename T>
Status ReduceMeanGrad<T>::Compute(OpKernelContext* context) const {
  auto& dy = *context->Input<Tensor>(0);

  if (!output_tensor_shapes_[0].empty()) {
    auto dx_shape = TensorShape::ReinterpretBaseType(output_tensor_shapes_[0]);
    auto dx = context->Output(0, dx_shape);

    float value = dy.Data<float>()[0] / dx_shape.Size();  //only one value expected for this case since we support 1-D input only

    auto out = gsl::make_span(dx->template MutableData<T>(), dx->Shape().Size());
    std::for_each(out.begin(), out.end(), [&value](T& v) { v = static_cast<T>(value); });
  }
  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    SigmoidGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SigmoidGrad<float>);

template <typename T>
Status SigmoidGrad<T>::Compute(OpKernelContext* context) const {
  auto& dY = *context->Input<Tensor>(0);
  auto& Y = *context->Input<Tensor>(1);
  auto& dX = *context->Output(0, Y.Shape());

  // dx = dy * y * (1 - y)
  // TODO: Would this be preferable as dx = dy * sigmoid(x) * (1 - sigmoid(x)) ???
  MakeEigenArrayMap<float>(dX) = MakeEigenArrayMap<float>(dY) * MakeEigenArrayMap<float>(Y) * (T(1) - MakeEigenArrayMap<float>(Y));
  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    SoftmaxGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SoftmaxGrad<float>);

template <typename T>
Status SoftmaxGrad<T>::Compute(OpKernelContext* context) const {
  auto& dY = *context->Input<Tensor>(0);
  auto& Y = *context->Input<Tensor>(1);
  const TensorShape input_shape{Y.Shape()};
  auto& dX = *context->Output(0, Y.Shape());

  auto axis = HandleNegativeAxis(axis_, Y.Shape().NumDimensions());

  size_t N = input_shape.SizeToDimension(axis);
  size_t D = input_shape.SizeFromDimension(axis);

  if (N == 0) {
    return Status::OK();
  }

  std::vector<float> scale_(N);
  std::vector<float> sum_multiplier_(D, 1.f);  // initialize all multiplier values to 1.0
  const int n = gsl::narrow_cast<int>(N);
  const int d = gsl::narrow_cast<int>(D);
  const int nd = gsl::narrow_cast<int>(N * D);

  float* scaledata = scale_.data();
  const float* Ydata = Y.template Data<float>();
  const float* dYdata = dY.template Data<float>();
  float* dXdata = dX.template MutableData<float>();

  gsl::copy(gsl::make_span(dYdata, nd), gsl::make_span(dXdata, nd));

  for (int i = 0; i < N; ++i) {
    math::Dot<float, CPUMathUtil>(d, Ydata + i * d, dYdata + i * d,
                                  scaledata + i, nullptr);
  }

  math::Gemm<float, CPUMathUtil>(CblasNoTrans, CblasNoTrans, n, d, 1, -1,
                                 scaledata, sum_multiplier_.data(), 1,
                                 dXdata, nullptr);

  math::Mul<float, CPUMathUtil>(gsl::narrow_cast<int>(Y.Shape().Size()), dXdata, Ydata, dXdata, nullptr);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
