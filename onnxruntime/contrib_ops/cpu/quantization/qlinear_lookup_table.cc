// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinear_lookup_table.h"

#include "core/mlas/inc/mlas.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

template <typename TOutput>
void QLinearLookupTableTransform(const uint8_t* x, const TOutput* table, TOutput* y, size_t n) {
  for (; n >= 4; n -= 4) {
    const size_t x_value0 = x[0];
    const size_t x_value1 = x[1];
    const size_t x_value2 = x[2];
    const size_t x_value3 = x[3];
    x += 4;
    const TOutput table_value0 = table[x_value0];
    const TOutput table_value1 = table[x_value1];
    const TOutput table_value2 = table[x_value2];
    const TOutput table_value3 = table[x_value3];

    y[0] = table_value0;
    y[1] = table_value1;
    y[2] = table_value2;
    y[3] = table_value3;
    y += 4;
  }
  for (; n != 0; --n) {
    const size_t x_value0 = *x++;
    const TOutput table_value0 = table[x_value0];
    *y++ = table_value0;
  }
}

template void QLinearLookupTableTransform(const uint8_t* x, const uint8_t* table, uint8_t* y, size_t n);
template void QLinearLookupTableTransform(const uint8_t* x, const float* table, float* y, size_t n);

template <typename T>
void QlinearBuildLookupTable(uint8_t* table,
                             const Tensor* tensor_x_scale,
                             const Tensor* tensor_x_zero_point,
                             const Tensor* tensor_y_scale,
                             const Tensor* tensor_y_zero_point,
                             const LookupTableArrayTransformer& array_values_transformer) {
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_x_scale),
              "QlinearBuildLookupTable : input X_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_x_zero_point == nullptr || IsScalarOr1ElementVector(tensor_x_zero_point),
              "QlinearBuildLookupTable : input X_zero_point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_y_scale),
              "QlinearBuildLookupTable : input Y_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_y_zero_point == nullptr || IsScalarOr1ElementVector(tensor_y_zero_point),
              "QlinearBuildLookupTable : input Y_zero_point must be a scalar or 1D tensor of size 1");

  const float X_scale = *(tensor_x_scale->Data<float>());
  const T X_zero_point =
    (tensor_x_zero_point == nullptr) ? static_cast<T>(0) : *(tensor_x_zero_point->Data<T>());
  const float Y_scale = *(tensor_y_scale->Data<float>());
  const T Y_zero_point =
    (tensor_y_zero_point == nullptr) ? static_cast<T>(0) : *(tensor_y_zero_point->Data<T>());

  float dequantized_input[256];
  float dequantized_output[256];
  for (int i = 0; i < 256; ++i) {
    T x = static_cast<T>(i);
    dequantized_input[i] = X_scale * (static_cast<int>(x) - static_cast<int>(X_zero_point));
  }
  array_values_transformer(dequantized_input, dequantized_output, 256);
  MlasQuantizeLinear(dequantized_output, (T*)table, 256, Y_scale, Y_zero_point);
}

template <typename T>
void QlinearBuildLookupTable(uint8_t* table,
                             const Tensor* tensor_x_scale,
                             const Tensor* tensor_x_zero_point,
                             const Tensor* tensor_y_scale,
                             const Tensor* tensor_y_zero_point,
                             const LookupTableScalarTransformer& value_transformer) {
  LookupTableArrayTransformer array_values_transformer =
      [&value_transformer](const float* input, float* output, size_t length) {
        for (size_t i = 0; i < length; ++i) {
          *output++ = value_transformer(*input++);
        }
      };
  return QlinearBuildLookupTable<T>(table, tensor_x_scale, tensor_x_zero_point,
                                    tensor_y_scale, tensor_y_zero_point, array_values_transformer);
}

template void QlinearBuildLookupTable<uint8_t>(uint8_t* table,
                                               const Tensor* tensor_x_scale,
                                               const Tensor* tensor_x_zero_point,
                                               const Tensor* tensor_y_scale,
                                               const Tensor* tensor_y_zero_point,
                                               const LookupTableScalarTransformer& value_transformer);

template void QlinearBuildLookupTable<int8_t>(uint8_t* table,
                                              const Tensor* tensor_x_scale,
                                              const Tensor* tensor_x_zero_point,
                                              const Tensor* tensor_y_scale,
                                              const Tensor* tensor_y_zero_point,
                                              const LookupTableScalarTransformer& value_transformer);

}  // namespace contrib
}  // namespace onnxruntime
