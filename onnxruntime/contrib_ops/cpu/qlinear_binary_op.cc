// Copyright (c Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinear_binary_op.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

template <typename T, typename Input0Scalar, typename Input1Scalar, typename General>
void QLinearBroadcastLoop(TBroadcaster<T, T>& bc, TBroadcastOutput<T>& output, Input0Scalar input0scalar, Input1Scalar input1scalar, General general,
                          float A_scale, float B_scale, float C_scale, T A_zero_point, T B_zero_point, T C_zero_point) {
  if (bc.IsInput0Scalar()) {
    while (output)
      input0scalar(output.NextSpanOutput(), bc.NextScalar0(), bc.NextSpan1(), A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
  } else if (bc.IsInput1Scalar()) {
    while (output)
      input1scalar(output.NextSpanOutput(), bc.NextSpan0(), bc.NextScalar1(), A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
  } else {
    while (output)
      general(output.NextSpanOutput(), bc.NextSpan0(), bc.NextSpan1(), A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
  }
}

template <typename T, typename Input0Scalar, typename Input1Scalar, typename General>
void QLinearBroadcastOneSpan(ThreadPool* tp, double unit_cost,
                             gsl::span<T> output_span, gsl::span<const T> input0_span, gsl::span<const T> input1_span,
                             Input0Scalar input0scalar, Input1Scalar input1scalar, General general,
                             float A_scale, float B_scale, float C_scale, T A_zero_point, T B_zero_point, T C_zero_point) {
  if (input0_span.size() == 1) {
    ThreadPool::TryParallelFor(tp, output_span.size(), unit_cost,
                               [=](std::ptrdiff_t first, std::ptrdiff_t last) {
                                 size_t count = static_cast<size_t>(last - first);
                                 input0scalar(output_span.subspan(first, count), *input0_span.data(), input1_span.subspan(first, count),
                                              A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
                               });
  } else if (input1_span.size() == 1) {
    ThreadPool::TryParallelFor(tp, output_span.size(), unit_cost,
                               [=](std::ptrdiff_t first, std::ptrdiff_t last) {
                                 size_t count = static_cast<size_t>(last - first);
                                 input1scalar(output_span.subspan(first, count), input0_span.subspan(first, count), *input1_span.data(),
                                              A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
                               });
  } else {
    ThreadPool::TryParallelFor(tp, output_span.size(), unit_cost,
                               [=](std::ptrdiff_t first, std::ptrdiff_t last) {
                                 size_t count = static_cast<size_t>(last - first);
                                 general(output_span.subspan(first, count), input0_span.subspan(first, count), input1_span.subspan(first, count),
                                         A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
                               });
  }
}

template <typename T, typename Input0Scalar, typename Input1Scalar, typename General>
Status QLinearBroadcastTwo(OpKernelContext& context, Input0Scalar input0scalar, Input1Scalar input1scalar, General general, double unit_cost) {
  auto tensor_a_scale = context.Input<Tensor>(1);
  auto tensor_a_zero_point = context.Input<Tensor>(2);
  auto tensor_b_scale = context.Input<Tensor>(4);
  auto tensor_b_zero_point = context.Input<Tensor>(5);
  auto tensor_c_scale = context.Input<Tensor>(6);
  auto tensor_c_zero_point = context.Input<Tensor>(7);

  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_a_scale),
              "MatmulInteger : input1 A_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_a_zero_point == nullptr || IsScalarOr1ElementVector(tensor_a_zero_point),
              "MatmulInteger : input1 A_zero_point must be a scalar or 1D tensor of size 1 if given");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_b_scale),
              "MatmulInteger : input1 B_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_b_zero_point == nullptr || IsScalarOr1ElementVector(tensor_b_zero_point),
              "MatmulInteger : input1 B_zero_point must be a scalar or 1D tensor of size 1 if given");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_c_scale),
              "MatmulInteger : input1 C_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_c_zero_point == nullptr || IsScalarOr1ElementVector(tensor_c_zero_point),
              "MatmulInteger : input1 C_zero_point must be a scalar or 1D tensor of size 1 if given");

  const float A_scale = *(tensor_a_scale->Data<float>());
  const T A_zero_point = (nullptr == tensor_a_zero_point) ? static_cast<T>(0) : *(tensor_a_zero_point->template Data<T>());
  const float B_scale = *(tensor_b_scale->Data<float>());
  const T B_zero_point = (nullptr == tensor_b_zero_point) ? static_cast<T>(0) : *(tensor_b_zero_point->template Data<T>());
  const float C_scale = *(tensor_c_scale->Data<float>());
  const T C_zero_point = (nullptr == tensor_c_zero_point) ? static_cast<T>(0) : *(tensor_c_zero_point->template Data<T>());

  TBroadcaster<T, T> bc(*context.Input<Tensor>(0), *context.Input<Tensor>(3));
  Tensor& output_tensor = *context.Output(0, bc.GetOutputShape());
  auto span_size = bc.GetSpanSize();
  TBroadcastOutput<T> output(span_size, output_tensor);
  int64_t output_len = output_tensor.Shape().Size();

  ThreadPool* tp = context.GetOperatorThreadPool();
  if (output_len == static_cast<int64_t>(span_size)) {  // Only one big span for all data, parallel inside it
    auto span0 = bc.IsInput0Scalar() ? gsl::span<const T>(&bc.NextScalar0(), 1) : bc.NextSpan0();
    auto span1 = bc.IsInput1Scalar() ? gsl::span<const T>(&bc.NextScalar1(), 1) : bc.NextSpan1();
    QLinearBroadcastOneSpan(tp, unit_cost, output.NextSpanOutput(), span0, span1,
                            input0scalar, input1scalar, general,
                            A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
  } else {
    ThreadPool::TryParallelFor(
        tp, output_len / span_size, unit_cost * span_size,
        [=, &bc, &output_tensor](std::ptrdiff_t first_span, std::ptrdiff_t last_span) {
          TBroadcaster<T, T> span_bc(bc);
          TBroadcastOutput<T> span_output(span_size, output_tensor, first_span * span_size, last_span * span_size);
          span_bc.AdvanceBy(first_span * span_size);
          QLinearBroadcastLoop(span_bc, span_output, input0scalar, input1scalar, general,
                               A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
        });
  }
  return Status::OK();
}

template <typename T>
Status QLinearAdd<T>::Compute(OpKernelContext* context) const {
  return QLinearBroadcastTwo<T>(
      *context,
      [](gsl::span<T> output, const T& input0, gsl::span<const T> input1,
         float A_scale, float B_scale, float C_scale, T A_zero_point, T B_zero_point, T C_zero_point) {
        MlasQLinearAdd(input1.data(), B_scale, B_zero_point,
                       &input0, A_scale, A_zero_point,
                       C_scale, C_zero_point, output.data(), output.size(), true);
      },
      [](gsl::span<T> output, gsl::span<const T> input0, const T& input1,
         float A_scale, float B_scale, float C_scale, T A_zero_point, T B_zero_point, T C_zero_point) {
        MlasQLinearAdd(input0.data(), A_scale, A_zero_point,
                       &input1, B_scale, B_zero_point,
                       C_scale, C_zero_point, output.data(), output.size(), true);
      },
      [](gsl::span<T> output, gsl::span<const T> input0, gsl::span<const T> input1,
         float A_scale, float B_scale, float C_scale, T A_zero_point, T B_zero_point, T C_zero_point) {
        MlasQLinearAdd(input0.data(), A_scale, A_zero_point,
                       input1.data(), B_scale, B_zero_point,
                       C_scale, C_zero_point, output.data(), output.size(), false);
      },
      1.0);
}

#define REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(op_name, version, data_type, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(                                                    \
      op_name, version, data_type,                                                      \
      KernelDefBuilder()                                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),               \
      KERNEL_CLASS<data_type>);

REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearAdd, 1, int8_t, QLinearAdd);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearAdd, 1, uint8_t, QLinearAdd);

}  // namespace contrib
}  // namespace onnxruntime
