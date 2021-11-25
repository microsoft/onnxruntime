// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <iomanip>
#include <string>
#include "core/framework/tensorprotoutils.h"

#ifndef NDEBUG
//#define DEBUG_BEAM_SEARCH 1  // uncomment it for debugging beam search
#endif

namespace onnxruntime {
namespace contrib {
namespace transformers {

#define MAX_ROW_OR_COLUMN 8

#define SKIP_IF_MORE_THAN(row_or_column_size, i, max_n, new_line)                           \
  if (row_or_column_size > max_n && i >= max_n / 2 && i + max_n / 2 < row_or_column_size) { \
    if (i == max_n / 2) {                                                                   \
      std::cout << ", ...";                                                                 \
      if (new_line)                                                                         \
        std::cout << std::endl;                                                             \
    }                                                                                       \
    continue;                                                                               \
  }

#define SKIP_IF_TOO_MANY(row_or_column_size, i, new_line) SKIP_IF_MORE_THAN(row_or_column_size, i, MAX_ROW_OR_COLUMN, new_line)

extern bool g_enable_tensor_dump;  // global variance to turn on/off dump

template <typename T>
void PrintValue(const T& value) {
  if (std::is_floating_point<T>::value)
    std::cout << std::setprecision(8) << value;
  else
    std::cout << value;
}

template <typename T>
void DumpTensor(const char* name, const Tensor& tensor) {
  if (!g_enable_tensor_dump)
    return;

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  const auto& shape = tensor.Shape();
  auto num_items = shape.Size();

  if (num_items == 0) {
    std::cout << "no data";
    return;
  }

  size_t num_dims = shape.NumDimensions();
  size_t num_rows = 1;
  if (num_dims > 1) {
    num_rows = static_cast<size_t>(shape[0]);
  }

  size_t row_size = num_items / num_rows;

  auto data = tensor.DataAsSpan<T>();

  for (size_t row = 0; row < num_rows; ++row) {
    SKIP_IF_TOO_MANY(num_rows, row, true);
    std::cout << "[" << row << "]:";
    for (size_t i = 0; i < row_size; ++i) {
      SKIP_IF_TOO_MANY(row_size, i, false);

      if (i > 0)
        std::cout << ", ";

      PrintValue(data[row * row_size + i]);
    }
    std::cout << "\n";
  }

  std::cout << std::endl;
}

void DumpOrtValue(const char* name, const OrtValue& value);

template <typename T>
void DumpTensor(const char* name, const T* tensor, int dim0, int dim1) {
  if (!g_enable_tensor_dump)
    return;

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  for (int i = 0; i < dim0; i++) {
    SKIP_IF_TOO_MANY(dim0, i, true);
    std::cout << "[" << i << "]:";
    for (int j = 0; j < dim1; j++) {
      SKIP_IF_TOO_MANY(dim1, j, false);
      if (j > 0)
        std::cout << ", ";
      T value = tensor[i * dim1 + j];
      PrintValue<T>(value);
    }
    std::cout << std::endl;
  }
}

void DumpString(const char* name, int index, bool end_line);

void DumpString(const char* name, std::string value, bool end_line);

template <typename T>
void DumpTensor(const char* name, const T* tensor, int dim0, int dim1, int dim2) {
  if (!g_enable_tensor_dump)
    return;

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  for (int i = 0; i < dim0; i++) {
    SKIP_IF_TOO_MANY(dim0, i, true);
    for (int j = 0; j < dim1; j++) {
      SKIP_IF_TOO_MANY(dim1, j, true);
      std::cout << "[" << i << "][" << j << "]:";
      for (int k = 0; k < dim2; k++) {
        SKIP_IF_TOO_MANY(dim2, k, false);
        if (k > 0)
          std::cout << ", ";
        T value = tensor[i * dim1 * dim2 + j * dim2 + k];
        PrintValue<T>(value);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void ConfigureTensorDump();

void DisableTensorDump();

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime