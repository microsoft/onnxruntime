// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/lp_norm.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"

namespace onnxruntime {
#define REGISTER_LPNORMALISATION_KERNEL(type, sinceVersion)                        \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                  \
      LpNormalization, sinceVersion, type,                                         \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), \
      LpNorm<type>);

REGISTER_LPNORMALISATION_KERNEL(float, 1)
REGISTER_LPNORMALISATION_KERNEL(double, 1)

using InnerStride = Eigen::InnerStride<Eigen::Dynamic>;

template <typename T>
using StridedVec = Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>, 0, InnerStride>;

template <typename T>
using ConstStridedVec = Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>, 0, InnerStride>;

template <typename T>
void DoNormalizeP2(
    const T* xData,
    T* yData,
    const int64_t m,
    const int64_t n,
    const int64_t sf) {
  for (int i = 0; i < n; ++i) {
    auto base = (i / sf) * sf * m + (i % sf);
    ConstStridedVec<T> xVec(xData + base, 1, m, InnerStride(sf));
    StridedVec<T> yVec(yData + base, 1, m, InnerStride(sf));

    auto norm = xVec.template lpNorm<2>();
    if (norm != 0) {
      yVec = xVec / norm;
    } else {
      // norm is zero, so set the result to zero
      yVec.setZero();
    }
  }
};

template <typename T>
void DoNormalizeP1(
    const T* xData,
    T* yData,
    const int64_t m,
    const int64_t n,
    const int64_t sf) {
  for (int i = 0; i < n; ++i) {
    auto base = (i / sf) * sf * m + (i % sf);
    ConstStridedVec<T> xVec(xData + base, 1, m, InnerStride(sf));
    StridedVec<T> yVec(yData + base, 1, m, InnerStride(sf));

    auto norm = xVec.template lpNorm<1>();
    if (norm != 0) {
      yVec = xVec / norm;
    } else {
      // norm is zero - set the result to zero
      yVec.setZero();
    }
  }
};

template <typename T>
Status LpNorm<T>::Compute(OpKernelContext* p_op_kernel_context) const {
  const auto* input = p_op_kernel_context->Input<Tensor>(0);
  const TensorShape& input_shape = input->Shape();
  Tensor* output = p_op_kernel_context->Output(0, input_shape);

  const auto canonical_axis = HandleNegativeAxis(axis_, static_cast<int64_t>(input_shape.NumDimensions()));
  const int64_t m = input_shape.GetDims()[canonical_axis];
  const int64_t n = input_shape.Size() / m;
  const int64_t sf = input_shape.SizeFromDimension(canonical_axis + 1);

  if (p_ == 1) {
    DoNormalizeP1(input->template Data<T>(), output->template MutableData<T>(), m, n, sf);
  } else if (p_ == 2) {
    DoNormalizeP2(input->template Data<T>(), output->template MutableData<T>(), m, n, sf);
  }

  return Status::OK();
}
}  // namespace onnxruntime
