// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinear_global_average_pool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

Status QLinearGlobalAveragePool::Compute(OpKernelContext* context) const {
  const auto tensor_x_scale = context->Input<Tensor>(1);
  const auto tensor_x_zero_point = context->Input<Tensor>(2);
  const auto tensor_y_scale = context->Input<Tensor>(3);
  const auto tensor_y_zero_point = context->Input<Tensor>(4);

  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_x_scale),
              "Input x_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_x_zero_point),
              "input x_zero_point must be a scalar or 1D tensor of size 1 if given");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_y_scale),
              "input y_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_y_zero_point),
              "input y_zero_point must be a scalar or 1D tensor of size 1 if given");

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  const auto& X = *context->Input<Tensor>(0);
  const auto& x_shape = X.Shape().GetDims();

  ORT_RETURN_IF_NOT(x_shape.size() >= 3, "Input dimension cannot be less than 3.");
  const size_t spatial_dim_start = channels_last_ ? 1 : 2;
  const size_t spatial_dim_end = spatial_dim_start + (x_shape.size() - 2);

  int64_t N = x_shape[0];
  int64_t C = (channels_last_ ? x_shape.back() : x_shape[1]);
  int64_t image_size = std::accumulate(x_shape.cbegin() + spatial_dim_start, x_shape.cbegin() + spatial_dim_end,
                                       1LL, std::multiplies<int64_t>());

  std::vector<int64_t> output_dims(x_shape.begin(), x_shape.end());
  std::transform(x_shape.cbegin() + spatial_dim_start, x_shape.cbegin() + spatial_dim_end,
                 output_dims.begin() + spatial_dim_start, [](const int64_t&) { return int64_t{1}; });
  Tensor& Y = *context->Output(0, output_dims);

  const float x_scale = *(tensor_x_scale->Data<float>());
  const float y_scale = *(tensor_y_scale->Data<float>());

  auto dtype = X.GetElementType();
  if (dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    return ComputeQLinearGlobalAvgPool(X.Data<uint8_t>(), x_scale, *(tensor_x_zero_point->Data<uint8_t>()),
                                       Y.MutableData<uint8_t>(), y_scale, *(tensor_y_zero_point->Data<uint8_t>()),
                                       N, C, image_size, channels_last_, tp);
  } else {
    return ComputeQLinearGlobalAvgPool(X.Data<int8_t>(), x_scale, *(tensor_x_zero_point->Data<int8_t>()),
                                       Y.MutableData<int8_t>(), y_scale, *(tensor_y_zero_point->Data<int8_t>()),
                                       N, C, image_size, channels_last_, tp);
  }
}

ONNX_OPERATOR_KERNEL_EX(QLinearGlobalAveragePool,
                        kMSDomain,
                        1,
                        kCpuExecutionProvider,
                        KernelDefBuilder(),
                        QLinearGlobalAveragePool);

}  // namespace contrib
}  // namespace onnxruntime
