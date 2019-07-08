// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cassert>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace onnxruntime {

template <typename T1, typename T2>
void DumpArrayRecursive(const T1* data, int64_t& data_offset, const std::vector<T2>& shape, int idx) {
  int dim = static_cast<int>(shape.size());
  if (dim == 0) {
    std::cout << "[]\n";
    return;
  }

  assert(idx < dim);
  int sz = shape[idx];

  std::cout << "[";
  if (idx < dim - 1) {
    for (auto i = 0; i < sz; ++i) {
      DumpArrayRecursive(data, data_offset, shape, idx + 1);
      if (i < sz - 1) {
        std::cout << ",";
        // print multiple newlines after ',' when necessary
        for (int j = idx + 1; j < dim; j++)
          std::cout << "\n";
        // print leading spaces before "[" when necessary
        for (int j = 0; j < idx + 1; ++j)
          std::cout << " ";
      }
    }
  } else {
    for (auto i = 0; i < sz; ++i) {
      if (std::is_same<T1, int8_t>::value || std::is_same<T1, uint8_t>::value)
        std::cout << std::setw(3) << static_cast<int>(*(data + data_offset));
      else
        std::cout << std::setw(12) << std::setprecision(8) << *(data + data_offset);
      data_offset++;
      if (i < sz - 1)
        std::cout << ",";
    }
  }
  std::cout << "]";
}

// A helper function to dump multidimensional arrays in a way similar to numpy
template <typename T1, typename T2>
void DumpArray(const std::string& tag, const T1* data, const std::vector<T2>& shape) {
  std::cout << tag << "\n";
  int64_t data_offset = 0;
  DumpArrayRecursive(data, data_offset, shape, 0);
  assert(data_offset == TotalSize(shape));
  std::cout << std::endl;
}

}  // namespace onnxruntime
