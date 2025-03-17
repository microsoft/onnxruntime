// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cmath>
#include "core/framework/print_tensor_utils.h"

namespace onnxruntime {
namespace utils {

// Currently we only store statistics data for float tensors that printed to stdout.
// It can be extended to other types if needed.
struct TensorStatisticsData {
  bool is_float = false;
  float float_min;
  float float_max;
};

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
    switch (my_fpclassify(data[i])) {
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
void PrintCommonStats(const T* data, size_t count, TensorStatisticsData& tensor_statistics) {
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

  // Statistics for float and double only for now.
  if constexpr (std::is_same<T, float>::value) {
    tensor_statistics.is_float = true;
    tensor_statistics.float_min = static_cast<double>(min);
    tensor_statistics.float_max = static_cast<double>(max);
  }
}

#define DEF_PRINT_COMMON_STATS_INT4(INT4_TYPE)                      \
  template <>                                                       \
  inline void PrintCommonStats<INT4_TYPE>(                          \
      const INT4_TYPE* data, size_t count, TensorStatisticsData&) { \
    using UnpackedType = typename INT4_TYPE::UnpackedType;          \
    UnpackedType min = data[0].GetElem(0);                          \
    UnpackedType max = min;                                         \
    for (size_t i = 1; i < count; i++) {                            \
      auto indices = INT4_TYPE::GetTensorElemIndices(i);            \
      auto value = data[indices.first].GetElem(indices.second);     \
      if (value > max) {                                            \
        max = value;                                                \
      }                                                             \
      if (value < min) {                                            \
        min = value;                                                \
      }                                                             \
    }                                                               \
                                                                    \
    std::cout << "Min=";                                            \
    PrintValue(min);                                                \
                                                                    \
    std::cout << ",Max=";                                           \
    PrintValue(max);                                                \
  }

DEF_PRINT_COMMON_STATS_INT4(Int4x2)
DEF_PRINT_COMMON_STATS_INT4(UInt4x2)

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
void PrintTensorStats(const T* tensor, size_t count, TensorStatisticsData& tensor_statistics) {
  PrintCommonStats<T>(tensor, count, tensor_statistics);
}

template <>
void PrintTensorStats<float>(const float* tensor, size_t count, TensorStatisticsData& tensor_statistics) {
  PrintCommonStats<float>(tensor, count, tensor_statistics);
  PrintFloatStats<float>(tensor, count);
}

template <>
void PrintTensorStats<double>(const double* tensor, size_t count, TensorStatisticsData& tensor_statistics) {
  PrintCommonStats<double>(tensor, count, tensor_statistics);
  PrintFloatStats<double>(tensor, count);
}

template <>
void PrintTensorStats<MLFloat16>(const MLFloat16* tensor, size_t count, TensorStatisticsData&) {
  PrintHalfStats<MLFloat16>(tensor, count);
  PrintFloatStats<MLFloat16>(tensor, count);
}

template <>
void PrintTensorStats<BFloat16>(const BFloat16* tensor, size_t count, TensorStatisticsData&) {
  PrintHalfStats<BFloat16>(tensor, count);
  PrintFloatStats<BFloat16>(tensor, count);
}

template <typename T>
void PrintCpuTensorStats(const Tensor& tensor, TensorStatisticsData& tensor_statistics) {
  const auto& shape = tensor.Shape();
  auto num_items = shape.Size();
  if (num_items == 0) {
    return;
  }

  const T* data = tensor.Data<T>();
  PrintTensorStats<T>(data, num_items, tensor_statistics);
  std::cout << std::endl;
}

template <>
void PrintCpuTensorStats<std::string>(const Tensor&, TensorStatisticsData&) {
}

}  // namespace utils
}  // namespace onnxruntime
