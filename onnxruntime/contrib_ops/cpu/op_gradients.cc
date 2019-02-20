/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#include "op_gradients.h"
#include "core/util/math_cpuonly.h"
#include "core/common/common.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/util/math.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace contrib {

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
    8,
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
    8,
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
    ConvGrad,
    8,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ConvGrad<float>);

template <typename T>
Status ConvGrad<T>::Compute(OpKernelContext* context) const {
  auto& dZ = *context->Input<Tensor>(0);
  //auto& X = *context->Input<Tensor>(1);
  //auto& Y = *context->Input<Tensor>(2);

  auto& dA = *context->Output(0, dZ.Shape());
  auto& dB = *context->Output(1, dZ.Shape());
  auto& dC = *context->Output(2, dZ.Shape());

  // unimplemented

  MakeEigenArrayMap<float>(dA) = MakeEigenArrayMap<float>(dZ);
  MakeEigenArrayMap<float>(dB) = MakeEigenArrayMap<float>(dZ);
  MakeEigenArrayMap<float>(dC) = MakeEigenArrayMap<float>(dZ);

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    FlattenGrad,
    8,
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
    8,
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
    8,
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
    8,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    AddGrad<float>);

// The Current implementation assumes X1.Shape() == X2.Shape()
// TODO: Implement Grad for BroadCastAdd
template <typename T>
Status AddGrad<T>::Compute(OpKernelContext* context) const {
  auto dY = context->Input<Tensor>(0);

  if (!output_tensor_shapes_[0].empty()) {
    auto dX1 = context->Output(0, TensorShape::ReinterpretBaseType(output_tensor_shapes_[0]));

    auto out = gsl::make_span(dX1->MutableData<float>(), dX1->Shape().Size());
    auto in = gsl::make_span(dY->Data<float>(), dY->Shape().Size());

    auto iter = out.begin();
    auto iter2 = in.begin();
    for (; iter != out.end(), iter2 != in.end(); iter++, iter2++) {
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
    8,
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
    8,
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
    8,
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
    8,
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

    auto out = gsl::make_span(dx->MutableData<T>(), dx->Shape().Size());
    std::for_each(out.begin(), out.end(), [&value](T& v) { v = static_cast<T>(value); });
  }
  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
