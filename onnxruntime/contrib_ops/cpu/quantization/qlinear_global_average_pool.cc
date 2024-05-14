// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinear_global_average_pool.h"
#include "core/common/narrow.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/platform/threadpool.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
#include <functional>

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

template <typename T8Bits>
Status ComputeQLinearGlobalAvgPool(
    const T8Bits* x,
    float x_scale,
    T8Bits x_zero_point,
    T8Bits* y,
    float y_scale,
    T8Bits y_zero_point,
    int64_t N,
    int64_t C,
    int64_t image_size,
    bool channels_last,
    concurrency::ThreadPool* tp) {
  if (!channels_last || C == 1) {
    auto worker = [=](std::ptrdiff_t first, std::ptrdiff_t last) {
      const T8Bits* input = (const T8Bits*)(x + (first * image_size));
      T8Bits* output = (T8Bits*)(y + first);
      std::vector<int32_t> acc_buffer(MlasQLinearSafePaddingElementCount(sizeof(int32_t), last - first));
      MlasQLinearGlobalAveragePoolNchw(input, x_scale, x_zero_point, output, y_scale, y_zero_point, last - first, narrow<size_t>(image_size), acc_buffer.data());
    };
    concurrency::ThreadPool::TryParallelFor(
        tp, static_cast<std::ptrdiff_t>(N * C), {1.0 * image_size, 1.0, 8.0 * image_size}, worker);
  } else {
    auto worker = [=](std::ptrdiff_t first, std::ptrdiff_t last) {
      const T8Bits* input = x + first * C * image_size;
      T8Bits* output = y + first * C;
      std::vector<int32_t> acc_buffer(MlasQLinearSafePaddingElementCount(sizeof(int32_t), narrow<size_t>(C)));
      std::vector<T8Bits> zero_buffer(MlasQLinearSafePaddingElementCount(sizeof(T8Bits), narrow<size_t>(C)), 0);
      MlasQLinearGlobalAveragePoolNhwc(
          input, x_scale, x_zero_point, output, y_scale, y_zero_point,
          last - first, narrow<size_t>(image_size), narrow<size_t>(C), narrow<size_t>(C), acc_buffer.data(), zero_buffer.data());
    };
    concurrency::ThreadPool::TryParallelFor(
        tp, static_cast<std::ptrdiff_t>(N),
        {1.0 * image_size * C, 1.0 * C, 8.0 * image_size * C},
        worker);
  }
  return Status::OK();
}

// GCC's unexplained behavior:
// GCC wouldn't generate corresponding symbols versus function instances below when "--disable-exceptions"
// and "--minimal-build" are combined on linux build.
// But this two symbols are required by qlinear_pool.cc.
// The other compilers wouldn't hit it and works fine, and we also didn't see it in the other platforms, such as Android.
// So we are doing explicit instantiation here for every compilers/platforms happy.
template Status ComputeQLinearGlobalAvgPool<int8_t>(
    const int8_t* x,
    float x_scale,
    int8_t x_zero_point,
    int8_t* y,
    float y_scale,
    int8_t y_zero_point,
    int64_t N,
    int64_t C,
    int64_t image_size,
    bool channels_last,
    concurrency::ThreadPool* tp);

template Status ComputeQLinearGlobalAvgPool<uint8_t>(
    const uint8_t* x,
    float x_scale,
    uint8_t x_zero_point,
    uint8_t* y,
    float y_scale,
    uint8_t y_zero_point,
    int64_t N,
    int64_t C,
    int64_t image_size,
    bool channels_last,
    concurrency::ThreadPool* tp);

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
  int64_t image_size = std::accumulate(x_shape.begin() + spatial_dim_start, x_shape.begin() + spatial_dim_end,
                                       1LL, std::multiplies<int64_t>());

  std::vector<int64_t> output_dims(x_shape.begin(), x_shape.end());
  std::transform(x_shape.begin() + spatial_dim_start, x_shape.begin() + spatial_dim_end,
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
