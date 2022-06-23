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

namespace {
struct QLinearBroadcastHelper : public BroadcastHelper {
  QLinearBroadcastHelper(InputBroadcaster& input_broadcaster,
                         OutputBroadcaster& output_broadcaster,
                         ThreadPool* threadpool,
                         double unit_cost,
                         float A_scale_in, float B_scale_in, float C_scale_in,
                         uint8_t A_zero_point_in, uint8_t B_zero_point_in, uint8_t C_zero_point_in)
      : BroadcastHelper{input_broadcaster, output_broadcaster, nullptr, threadpool, unit_cost},
        A_scale{A_scale_in},
        B_scale{B_scale_in},
        C_scale{C_scale_in},
        A_zero_point{A_zero_point_in},
        B_zero_point{B_zero_point_in},
        C_zero_point{C_zero_point_in} {
  }

  QLinearBroadcastHelper(const QLinearBroadcastHelper& rhs, size_t offset, size_t num_elements)
      : BroadcastHelper(rhs, offset, num_elements),
        A_scale{rhs.A_scale},
        B_scale{rhs.B_scale},
        C_scale{rhs.C_scale},
        A_zero_point{rhs.A_zero_point},
        B_zero_point{rhs.B_zero_point},
        C_zero_point{rhs.C_zero_point} {
  }

  float A_scale;
  float B_scale;
  float C_scale;
  // storage for these is uint8_t but original value may be uint8_t or int8_t.
  // typed code that uses values needs to cast to correct representation
  uint8_t A_zero_point;
  uint8_t B_zero_point;
  uint8_t C_zero_point;
};

template <typename T>
void QLinearImpl(OpKernelContext& context, double unit_cost, const ProcessBroadcastSpanFuncs& functors) {
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
  const T A_zero_point = (nullptr == tensor_a_zero_point) ? T{} : *(tensor_a_zero_point->template Data<T>());
  const float B_scale = *(tensor_b_scale->Data<float>());
  const T B_zero_point = (nullptr == tensor_b_zero_point) ? T{} : *(tensor_b_zero_point->template Data<T>());
  const float C_scale = *(tensor_c_scale->Data<float>());
  const T C_zero_point = (nullptr == tensor_c_zero_point) ? T{} : *(tensor_c_zero_point->template Data<T>());

  InputBroadcaster input_broadcaster{*context.Input<Tensor>(0), *context.Input<Tensor>(3)};
  OutputBroadcaster output_broadcaster{input_broadcaster.GetSpanSize(),
                                       *context.Output(0, input_broadcaster.GetOutputShape())};

  QLinearBroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster,
                                          context.GetOperatorThreadPool(), unit_cost,
                                          A_scale, B_scale, C_scale,
                                          static_cast<uint8_t>(A_zero_point),
                                          static_cast<uint8_t>(B_zero_point),
                                          static_cast<uint8_t>(C_zero_point));

  BroadcastLooper(broadcast_helper, functors);
}
}  // namespace

template <typename T>
Status QLinearAdd<T>::Compute(OpKernelContext* context) const {
  const ProcessBroadcastSpanFuncs functors = {
      [](BroadcastHelper& per_iter_bh) {
        QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        const T input0 = per_iter_bh.ScalarInput0<T>();
        auto input1 = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        MlasQLinearAdd(input1.data(),
                       qlbh.B_scale, static_cast<T>(qlbh.B_zero_point),
                       &input0,
                       qlbh.A_scale, static_cast<T>(qlbh.A_zero_point),
                       qlbh.C_scale, static_cast<T>(qlbh.C_zero_point),
                       output.data(), output.size(), true);
      },
      [](BroadcastHelper& per_iter_bh) {
        QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        auto input0 = per_iter_bh.SpanInput0<T>();
        const T input1 = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();
        MlasQLinearAdd(input0.data(),
                       qlbh.A_scale, static_cast<T>(qlbh.A_zero_point),
                       &input1,
                       qlbh.B_scale, static_cast<T>(qlbh.B_zero_point),
                       qlbh.C_scale, static_cast<T>(qlbh.C_zero_point),
                       output.data(), output.size(), true);
      },
      [](BroadcastHelper& per_iter_bh) {
        QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        auto input0 = per_iter_bh.SpanInput0<T>();
        auto input1 = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        MlasQLinearAdd(input0.data(),
                       qlbh.A_scale, static_cast<T>(qlbh.A_zero_point),
                       input1.data(),
                       qlbh.B_scale, static_cast<T>(qlbh.B_zero_point),
                       qlbh.C_scale, static_cast<T>(qlbh.C_zero_point),
                       output.data(), output.size(), false);
      }};

  QLinearImpl<T>(*context, 1.0, functors);

  return Status::OK();
}

template <typename T>
Status QLinearMul<T>::Compute(OpKernelContext* context) const {
  const ProcessBroadcastSpanFuncs functors = {
      [](BroadcastHelper& per_iter_bh) {
        QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        const T input0 = per_iter_bh.ScalarInput0<T>();
        auto input1 = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        MlasQLinearMul(input1.data(),
                       qlbh.B_scale, static_cast<T>(qlbh.B_zero_point),
                       &input0,
                       qlbh.A_scale, static_cast<T>(qlbh.A_zero_point),
                       qlbh.C_scale, static_cast<T>(qlbh.C_zero_point),
                       output.data(), output.size(), true);
      },
      [](BroadcastHelper& per_iter_bh) {
        QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        auto input0 = per_iter_bh.SpanInput0<T>();
        const T input1 = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();
        MlasQLinearMul(input0.data(),
                       qlbh.A_scale, static_cast<T>(qlbh.A_zero_point),
                       &input1,
                       qlbh.B_scale, static_cast<T>(qlbh.B_zero_point),
                       qlbh.C_scale, static_cast<T>(qlbh.C_zero_point),
                       output.data(), output.size(), true);
      },
      [](BroadcastHelper& per_iter_bh) {
        QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        auto input0 = per_iter_bh.SpanInput0<T>();
        auto input1 = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        MlasQLinearMul(input0.data(),
                       qlbh.A_scale, static_cast<T>(qlbh.A_zero_point),
                       input1.data(),
                       qlbh.B_scale, static_cast<T>(qlbh.B_zero_point),
                       qlbh.C_scale, static_cast<T>(qlbh.C_zero_point),
                       output.data(), output.size(), false);
      }};

  QLinearImpl<T>(*context, 1.0, functors);

  return Status::OK();
}

#define REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(op_name, version, data_type, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(                                                    \
      op_name, version, data_type,                                                      \
      KernelDefBuilder()                                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),               \
      KERNEL_CLASS<data_type>);

REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearAdd, 1, int8_t, QLinearAdd);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearAdd, 1, uint8_t, QLinearAdd);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearMul, 1, int8_t, QLinearMul);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearMul, 1, uint8_t, QLinearMul);

}  // namespace contrib
}  // namespace onnxruntime
