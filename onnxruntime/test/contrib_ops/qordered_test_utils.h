// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#include <vector>
#include <functional>

namespace onnxruntime {
namespace test {

enum OrderCublasLt {
  ORDER_COL = 0,
  ORDER_ROW = 1,
};

class OrderedIndex {
  OrderCublasLt order_;
  int64_t rows_;
  int64_t cols_;

 public:
  OrderedIndex(OrderCublasLt order, int64_t rows, int64_t cols)
      : order_(order), rows_(rows), cols_(cols) {}

  int64_t operator()(int64_t r, int64_t c) {
    switch (order_) {
      case ORDER_ROW:
        return r * cols_ + c;
      case ORDER_COL:
        return c * rows_ + r;
      default:
        return 0;
    }
  }
};

template <typename T>
inline std::vector<T> GenData(std::vector<int64_t> const& shape, float scale) {
  int64_t n = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

  scale = std::is_same<T, int8_t>::value ? 1.0f : scale;  // using scale = 1.0f to generate int8_t data,
  std::vector<T> r(n);
  RandomValueGenerator random{};
  std::vector<int> tmp = random.Uniform<int32_t>(shape, -128, 127);
  for (int64_t i = 0; i < n; i++) {
    r[i] = static_cast<T>(tmp[i] * scale);
  }
  return r;
}

inline void BatchRowColFromShape(std::vector<int64_t> const& shape, int64_t& batch, int64_t& rows, int64_t& cols) {
  cols = shape.back();
  rows = (shape.size() > 1 ? shape[shape.size() - 2] : 1LL);
  batch = (shape.size() <= 2)
              ? 1LL
              : std::accumulate(shape.data(), shape.data() + (shape.size() - 2), 1LL, std::multiplies<int64_t>());
}

}  // namespace test
}  // namespace onnxruntime
