// Copyright (c Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear_element_wise.h"
#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {
namespace contrib {

template <typename TBroadcaster, typename Output, typename Input0Scalar, typename Input1Scalar, typename General>
void QLinearBroadcastLoop(TBroadcaster& bc, Output& output, Input0Scalar input0scalar, Input1Scalar input1scalar,
                          General general, float A_scale, float B_scale, float C_scale, int A_zero_point, int B_zero_point, int C_zero_point) {
  if (bc.IsInput0Scalar()) {
    while (output)
      input0scalar(output.NextEigenOutput(), bc.NextScalar0(), bc.NextEigen1(), A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
  } else if (bc.IsInput1Scalar()) {
    while (output)
      input1scalar(output.NextEigenOutput(), bc.NextEigen0(), bc.NextScalar1(), A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
  } else {
    while (output)
      general(output.NextEigenOutput(), bc.NextEigen0(), bc.NextEigen1(), A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
  }
}

template <typename T, typename Input0Scalar, typename Input1Scalar, typename General>
Status QLinearBroadcastTwo(OpKernelContext& context, Input0Scalar input0scalar, Input1Scalar input1scalar, General general) {
  const float A_scale = *(context.Input<Tensor>(1)->Data<float>());
  const T A_zero_point = (nullptr == context.Input<Tensor>(2)) ? static_cast<T>(0) : *(context.Input<Tensor>(2)->template Data<T>());
  const float B_scale = *(context.Input<Tensor>(4)->Data<float>());
  const T B_zero_point = (nullptr == context.Input<Tensor>(5)) ? static_cast<T>(0) : *(context.Input<Tensor>(5)->template Data<T>());
  const float C_scale = *(context.Input<Tensor>(6)->Data<float>());
  const T C_zero_point = (nullptr == context.Input<Tensor>(7)) ? static_cast<T>(0) : *(context.Input<Tensor>(7)->template Data<T>());

  TBroadcaster<T, T> bc(*context.Input<Tensor>(0), *context.Input<Tensor>(3));
  TBroadcastOutput<T> output(bc.GetSpanSize(), *context.Output(0, bc.GetOutputShape()));
  QLinearBroadcastLoop(bc, output, input0scalar, input1scalar, general, A_scale, B_scale, C_scale, static_cast<int>(A_zero_point), static_cast<int>(B_zero_point), static_cast<int>(C_zero_point));
  return Status::OK();
}

template <typename T>
Status QLinearAdd<T>::Compute(OpKernelContext* context) const {
  return QLinearBroadcastTwo<T>(
      *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1, float A_scale, float B_scale, float C_scale, int A_zero_point, int B_zero_point, int C_zero_point) {
        float a_value = A_scale * (static_cast<int>(input0) - A_zero_point);
        output = (((((input1.array().template cast<int>() - B_zero_point).template cast<float>() * B_scale) + a_value) / C_scale).round() - static_cast<float>(C_zero_point)).template cast<T>();
      },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1, float A_scale, float B_scale, float C_scale, int A_zero_point, int B_zero_point, int C_zero_point) {
        float b_value = B_scale * (static_cast<int>(input1) - B_zero_point);
        output = (((((input0.array().template cast<int>() - A_zero_point).template cast<float>() * A_scale) + b_value) / C_scale).round() - static_cast<float>(C_zero_point)).template cast<T>();
      },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1, float A_scale, float B_scale, float C_scale, int A_zero_point, int B_zero_point, int C_zero_point) {
        output = (((((input0.array().template cast<int>() - A_zero_point).template cast<float>() * A_scale) +
                    ((input1.array().template cast<int>() - B_zero_point).template cast<float>() * B_scale)) / C_scale).round() - static_cast<float>(C_zero_point)).template cast<T>();
      });
}

template <typename T>
Status QLinearMul<T>::Compute(OpKernelContext* context) const {
  return QLinearBroadcastTwo<T>(
      *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1, float A_scale, float B_scale, float C_scale, int A_zero_point, int B_zero_point, int C_zero_point) {
        float a_value_scaled_b_c = A_scale * (static_cast<int>(input0) - A_zero_point) * B_scale / C_scale;
        output = (((input1.array().template cast<int>() - B_zero_point).template cast<float>() * a_value_scaled_b_c).round() - static_cast<float>(C_zero_point)).template cast<T>();
      },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1, float A_scale, float B_scale, float C_scale, int A_zero_point, int B_zero_point, int C_zero_point) {
        float b_value_scaled_a_c = B_scale * (static_cast<int>(input1) - B_zero_point) * A_scale / C_scale;
        output = (((input0.array().template cast<int>() - A_zero_point).template cast<float>() * b_value_scaled_a_c).round() - static_cast<float>(C_zero_point)).template cast<T>();
      },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1, float A_scale, float B_scale, float C_scale, int A_zero_point, int B_zero_point, int C_zero_point) {
        output = (((((input0.array().template cast<int>() - A_zero_point).template cast<float>() * A_scale) *
                    ((input1.array().template cast<int>() - B_zero_point).template cast<float>() * B_scale)) / C_scale).round() - static_cast<float>(C_zero_point)).template cast<T>();
      });
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
