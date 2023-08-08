// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <functional>
#include <iomanip>
#include <iostream>

namespace onnxruntime {
namespace utils {

constexpr int64_t kDefaultSnippetEdgeItems = 3;
constexpr int64_t kDefaultSnippetThreshold = 200;

// Skip non edge items in last dimension
#define SKIP_NON_EDGE_ITEMS_LAST_DIM(dim_size, index, edge_items)                          \
  if (dim_size > 2 * edge_items && index >= edge_items && index + edge_items < dim_size) { \
    if (index == edge_items) {                                                             \
      std::cout << ", ... ";                                                               \
    }                                                                                      \
    continue;                                                                              \
  }

// Skip non edge items in other dimensions except the last dimension
#define SKIP_NON_EDGE_ITEMS(dim_size, index, edge_items)                                   \
  if (dim_size > 2 * edge_items && index >= edge_items && index + edge_items < dim_size) { \
    if (index == edge_items) {                                                             \
      std::cout << "..." << std::endl;                                                     \
    }                                                                                      \
    continue;                                                                              \
  }

template <typename T>
inline void PrintValue(const T& value) {
  if (std::is_floating_point<T>::value)
    std::cout << std::setprecision(8) << value;
  else
    std::cout << value;
}

// Explicit specialization
template <>
inline void PrintValue(const MLFloat16& value) {
  std::cout << std::setprecision(8) << value.ToFloat();
}

template <>
inline void PrintValue(const BFloat16& value) {
  std::cout << std::setprecision(8) << value.ToFloat();
}

template <>
inline void PrintValue(const uint8_t& value) {
  std::cout << static_cast<uint32_t>(value);
}

template <>
inline void PrintValue(const int8_t& value) {
  std::cout << static_cast<int32_t>(value);
}

// Print snippet of 2D tensor with shape (dim0, dim1)
template <typename T>
void PrintCpuTensorSnippet(const T* tensor, int64_t dim0, int64_t dim1, int64_t edge_items) {
  for (int64_t i = 0; i < dim0; i++) {
    SKIP_NON_EDGE_ITEMS(dim0, i, edge_items);
    PrintValue(tensor[i * dim1]);
    for (int64_t j = 1; j < dim1; j++) {
      SKIP_NON_EDGE_ITEMS_LAST_DIM(dim1, j, edge_items);
      std::cout << ", ";
      PrintValue(tensor[i * dim1 + j]);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Print snippet of 3D tensor with shape (dim0, dim1, dim2)
template <typename T>
void PrintCpuTensorSnippet(const T* tensor, int64_t dim0, int64_t dim1, int64_t dim2, int64_t edge_items) {
  for (int64_t i = 0; i < dim0; i++) {
    SKIP_NON_EDGE_ITEMS(dim0, i, edge_items);
    for (int64_t j = 0; j < dim1; j++) {
      SKIP_NON_EDGE_ITEMS(dim1, j, edge_items);
      PrintValue(tensor[i * dim1 * dim2 + j * dim2]);
      for (int64_t k = 1; k < dim2; k++) {
        SKIP_NON_EDGE_ITEMS_LAST_DIM(dim2, k, edge_items);
        std::cout << ", ";
        PrintValue(tensor[i * dim1 * dim2 + j * dim2 + k]);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Print 2D tensor
template <typename T>
void PrintCpuTensorFull(const T* tensor, int64_t dim0, int64_t dim1) {
  for (int64_t i = 0; i < dim0; i++) {
    PrintValue(tensor[i * dim1]);
    for (int64_t j = 1; j < dim1; j++) {
      std::cout << ", ";
      PrintValue(tensor[i * dim1 + j]);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Print 3D tensor
template <typename T>
void PrintCpuTensorFull(const T* tensor, int64_t dim0, int64_t dim1, int64_t dim2) {
  for (int64_t i = 0; i < dim0; i++) {
    for (int64_t j = 0; j < dim1; j++) {
      PrintValue(tensor[i * dim1 * dim2 + j * dim2]);
      for (int64_t k = 1; k < dim2; k++) {
        std::cout << ", ";
        PrintValue(tensor[i * dim1 * dim2 + j * dim2 + k]);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T>
void PrintCpuTensor(const Tensor& tensor, int threshold = kDefaultSnippetThreshold, int edge_items = kDefaultSnippetEdgeItems) {
  const auto& shape = tensor.Shape();
  auto num_items = shape.Size();
  if (num_items == 0) {
    std::cout << "no data";
    return;
  }

  auto data = tensor.Data<T>();
  bool is_snippet = (threshold > 0 && static_cast<int64_t>(threshold) < num_items);
  size_t num_dims = shape.NumDimensions();
  if (num_dims >= 3) {
    int64_t dim0 = shape.SizeToDimension(num_dims - 2);
    int64_t dim1 = shape[num_dims - 2];
    int64_t dim2 = shape[num_dims - 1];
    if (is_snippet) {
      PrintCpuTensorSnippet<T>(data, dim0, dim1, dim2, edge_items);
    } else {
      PrintCpuTensorFull<T>(data, dim0, dim1, dim2);
    }
    return;
  }

  int64_t num_rows = 1;
  if (num_dims > 1) {
    num_rows = shape[0];
  }
  int64_t row_size = num_items / num_rows;

  if (is_snippet) {
    PrintCpuTensorSnippet<T>(data, num_rows, row_size, edge_items);
  } else {
    PrintCpuTensorFull<T>(data, num_rows, row_size);
  }
}

}  // namespace utils
}  // namespace onnxruntime
