// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cmath>
#include "core/framework/print_tensor_utils.h"

namespace onnxruntime {
namespace utils {

template <typename T>
int my_fpclassify(const T& val) {
  return std::fpclassify(val);
}

template <>
int my_fpclassify(const MLFloat16& val) {
  return std::fpclassify(val.ToFloat());
}

template <>
int my_fpclassify(const BFloat16& val) {
  return std::fpclassify(val.ToFloat());
}

template <typename T>
void PrintFloatStats(const T* data, size_t count) {
  size_t inf = 0;
  size_t nan = 0;
  size_t zero = 0;
  size_t subnormal = 0;
  for (size_t i = 0; i < count; i++) {
    switch (my_fpclassify(*data)) {
      case FP_INFINITE:
        inf++;
        break;
      case FP_NAN:
        nan++;
        break;
      case FP_SUBNORMAL:
        subnormal++;
        break;
      case FP_ZERO:
        zero++;
        break;
      default:
        break;
    }
  }

  if (inf)
    std::cout << ",Inf=" << inf;
  if (nan)
    std::cout << ",NaN=" << nan;
  if (zero)
    std::cout << ",Zero=" << zero;
  if (subnormal)
    std::cout << ",SubNormal=" << subnormal;
}

template <typename T>
void PrintCommonStats(const T* data, size_t count) {
  T min = data[0];
  T max = min;
  for (size_t i = 1; i < count; i++) {
    auto value = data[i];
    if (value > max) {
      max = value;
    }
    if (value < min) {
      min = value;
    }
  }

  std::cout << "Min=";
  PrintValue(min);

  std::cout << ",Max=";
  PrintValue(max);
}

template <typename T>
void PrintHalfStats(const T* data, size_t count) {
  float min = data[0].ToFloat();
  float max = min;
  for (size_t i = 1; i < count; i++) {
    float value = data[i].ToFloat();
    if (value > max) {
      max = value;
    }

    if (value < min) {
      min = value;
    }
  }

  std::cout << "Min=";
  PrintValue(min);

  std::cout << ",Max=";
  PrintValue(max);
}

template <typename T>
void PrintTensorStats(const T* tensor, size_t count) {
  PrintCommonStats<T>(tensor, count);
}

template <>
void PrintTensorStats<float>(const float* tensor, size_t count) {
  PrintCommonStats<float>(tensor, count);
  PrintFloatStats<float>(tensor, count);
}

template <>
void PrintTensorStats<double>(const double* tensor, size_t count) {
  PrintCommonStats<double>(tensor, count);
  PrintFloatStats<double>(tensor, count);
}

template <>
void PrintTensorStats<MLFloat16>(const MLFloat16* tensor, size_t count) {
  PrintHalfStats<MLFloat16>(tensor, count);
  PrintFloatStats<MLFloat16>(tensor, count);
}

template <>
void PrintTensorStats<BFloat16>(const BFloat16* tensor, size_t count) {
  PrintHalfStats<BFloat16>(tensor, count);
  PrintFloatStats<BFloat16>(tensor, count);
}

template <typename T>
void PrintCpuTensorStats(const Tensor& tensor) {
  const auto& shape = tensor.Shape();
  auto num_items = shape.Size();
  if (num_items == 0) {
    return;
  }

  const T* data = tensor.Data<T>();
  PrintTensorStats<T>(data, num_items);
  std::cout << std::endl;
}

template <>
void PrintCpuTensorStats<std::string>(const Tensor&) {
}

}  // namespace utils
}  // namespace onnxruntime
