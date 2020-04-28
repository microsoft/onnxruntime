// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinear_lookup_table.h"
#include <limits>
#include <cmath>

namespace onnxruntime {
namespace contrib {

static void QLinearLookupTableTransform(const uint8_t* x, const uint8_t table[256], uint8_t* y, size_t n) {
  for (; n >= 4; n -= 4) {
    const size_t x_value0 = x[0];
    const size_t x_value1 = x[1];
    const size_t x_value2 = x[2];
    const size_t x_value3 = x[3];
    x += 4;
    const uint8_t table_value0 = table[x_value0];
    const uint8_t table_value1 = table[x_value1];
    const uint8_t table_value2 = table[x_value2];
    const uint8_t table_value3 = table[x_value3];

    y[0] = table_value0;
    y[1] = table_value1;
    y[2] = table_value2;
    y[3] = table_value3;
    y += 4;
  }
  for (; n != 0; --n) {
    const size_t x_value0 = *x++;
    const uint8_t table_value0 = table[x_value0];
    *y++ = table_value0;
  }
}

template <typename T>
static void BuildQLinearLookupTable(
    uint8_t table[256], const float X_scale, const T X_zero_point, const float Y_scale, const T Y_zero_point, float alpha) {
  constexpr int qmin = std::numeric_limits<T>::min();
  constexpr int qmax = std::numeric_limits<T>::max();
  for (int i = 0; i < 256; ++i) {
    T x = static_cast<T>(i);
    float x_dequantized = X_scale * (static_cast<int>(x) - static_cast<int>(X_zero_point));
    float y = x_dequantized >= 0.0f ? x_dequantized : alpha * x_dequantized;
    int y_quantized = static_cast<int>(std::round(y / Y_scale)) + static_cast<int>(Y_zero_point);
    table[i] = static_cast<uint8_t>(static_cast<T>(std::min(qmax, std::max(qmin, y_quantized))));
  }
}

template <typename T>
QLinearLeakyRelu<T>::QLinearLeakyRelu(const OpKernelInfo& info)
    : OpKernel(info), alpha_(info.GetAttrOrDefault("alpha", 0.01f)) {
  const Tensor* X_scale_tensor = nullptr;
  const Tensor* X_zero_point_tensor = nullptr;
  const Tensor* Y_scale_tensor = nullptr;
  const Tensor* Y_zero_point_tensor = nullptr;

  bool X_scale_get = info.TryGetConstantInput(1, &X_scale_tensor);
  bool X_zero_point_get = info.TryGetConstantInput(2, &X_zero_point_tensor);
  bool Y_scale_get = info.TryGetConstantInput(3, &Y_scale_tensor);
  bool Y_zero_point_get = info.TryGetConstantInput(4, &Y_zero_point_tensor);
  is_fixed_parameters_ = X_scale_get && X_zero_point_get && Y_scale_get && Y_zero_point_get;

  if (is_fixed_parameters_) {
    const float X_scale = *(X_scale_tensor->Data<float>());
    const T X_zero_point = (X_zero_point_tensor == nullptr) ? static_cast<T>(0) : *(X_zero_point_tensor->template Data<T>());
    const float Y_scale = *(Y_scale_tensor->Data<float>());
    const T Y_zero_point = (Y_zero_point_tensor == nullptr) ? static_cast<T>(0) : *(Y_zero_point_tensor->template Data<T>());
    BuildQLinearLookupTable(fixed_lookup_table_, X_scale, X_zero_point, Y_scale, Y_zero_point, alpha_);
  }
}

template <typename T>
Status QLinearLeakyRelu<T>::Compute(OpKernelContext* context) const {
  const auto& X = *context->Input<Tensor>(0);
  const auto& input_shape = X.Shape();
  const auto N = input_shape.Size();
  auto& Y = *context->Output(0, input_shape);

  uint8_t table[256];
  if (!is_fixed_parameters_) {
    const float X_scale = *(context->Input<Tensor>(1)->Data<float>());
    const T X_zero_point = (nullptr == context->Input<Tensor>(2)) ? static_cast<T>(0) : *(context->Input<Tensor>(2)->template Data<T>());
    const float Y_scale = *(context->Input<Tensor>(3)->Data<float>());
    const T Y_zero_point = (nullptr == context->Input<Tensor>(4)) ? static_cast<T>(0) : *(context->Input<Tensor>(4)->template Data<T>());
    BuildQLinearLookupTable(table, X_scale, X_zero_point, Y_scale, Y_zero_point, alpha_);
  }

  QLinearLookupTableTransform(
      reinterpret_cast<const uint8_t*>(X.template Data<T>()),
      is_fixed_parameters_ ? fixed_lookup_table_ : table,
      reinterpret_cast<uint8_t*>(Y.template MutableData<T>()),
      static_cast<size_t>(N));

  return Status::OK();
}

#define REGISTER_QLINEAR_LOOKUPTABLE_TYPED_KERNEL(op_name, version, data_type, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(                                                         \
      op_name, version, data_type,                                                           \
      KernelDefBuilder()                                                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),                    \
      KERNEL_CLASS<data_type>);

REGISTER_QLINEAR_LOOKUPTABLE_TYPED_KERNEL(QLinearLeakyRelu, 1, int8_t, QLinearLeakyRelu);
REGISTER_QLINEAR_LOOKUPTABLE_TYPED_KERNEL(QLinearLeakyRelu, 1, uint8_t, QLinearLeakyRelu);

}  // namespace contrib
}  // namespace onnxruntime
