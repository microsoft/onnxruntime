// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/hardmax.h"
#include "core/util/math_cpuonly.h"
#include "core/util/math.h"
#include "core/providers/common.h"

namespace onnxruntime {

template <>
Status Hardmax<float>::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  const TensorShape& input_shape = X->Shape();
  const auto* Xdata = X->template Data<float>();

  const auto axis_corrected = HandleNegativeAxis(axis_, static_cast<int64_t>(input_shape.NumDimensions()));
  size_t tmpN = input_shape.SizeToDimension(axis_corrected);
  size_t tmpD = input_shape.SizeFromDimension(axis_corrected);

  // Math::RowwiseMax expects int N and D.
  if (tmpN * tmpD > INT32_MAX || tmpN > INT32_MAX || tmpD > INT32_MAX) {
    std::ostringstream ss;
    ss << "Hardmax inputs N, D and N * D must be < " << INT32_MAX << ". N=" << tmpN << ", D=" << tmpD;
    std::string msg = ss.str();

    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, msg);
  }

  const int N = gsl::narrow_cast<int>(tmpN);
  const int D = gsl::narrow_cast<int>(tmpD);

  std::vector<float> rowmax_(N);
  float* rowmax_data = rowmax_.data();
  math::RowwiseMax<float, CPUMathUtil>(N, D, Xdata, rowmax_data, nullptr);

  Tensor* Y = ctx->Output(0, input_shape);
  auto* Ydata = Y->template MutableData<float>();
  math::Set<float, CPUMathUtil>(input_shape.Size(), 0.f, Ydata, &CPUMathUtil::Instance());

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      if (Xdata[i * D + j] == rowmax_data[i]) {
        Ydata[i * D + j] = 1;
        break;
      }
    }
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    Hardmax,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Hardmax<float>);

}  // namespace onnxruntime
