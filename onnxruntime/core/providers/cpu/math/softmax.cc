// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/softmax.h"

#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/softmax_shared.h"
#include "core/util/math.h"

namespace onnxruntime {

template <>
Status Softmax<float>::Compute(OpKernelContext* ctx) const {
  auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();

  const auto* tensor_pointer = ctx->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *tensor_pointer;
  const TensorShape& input_shape{X.Shape()};

  VLOGS(ctx->Logger(), 2) << "Input tensor shape: " << input_shape;

  Tensor* Y = ctx->Output(0, input_shape);

  const int64_t axis = HandleNegativeAxis(axis_, input_shape.NumDimensions());

  size_t N = input_shape.SizeToDimension(axis);
  size_t D = input_shape.SizeFromDimension(axis);

  auto* Ydata = Y->template MutableData<float>();

  std::vector<float> scale_(N);
  std::vector<float> rowmax_(N);
  std::vector<float> sum_multiplier_(D, 1.f);  // initialize all multiplier values to 1.0

  const bool logarithmic = false;
  auto status = SoftmaxCPU(N, D, X.template Data<float>(), Ydata,
                           scale_.data(), sum_multiplier_.data(), logarithmic, rowmax_.data(), tp);

  return status;
}

ONNX_CPU_OPERATOR_KERNEL(
    Softmax,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

}  // namespace onnxruntime
