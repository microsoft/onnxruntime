#pragma once

#include <cstdint>
#if __cpluscplus >= 202002L
#include <span>
#else
#include "common/simple_span.h"
#endif
#include <vector>

namespace onnx_c_ops {

#define py_array_t py::array_t
#define py_array_style py::array::c_style | py::array::forcecast

#define array2vector(vec, arr, dtype)                                                          \
  {                                                                                            \
    if (arr.size() > 0) {                                                                      \
      auto n = arr.size();                                                                     \
      auto p = (dtype *)arr.data(0);                                                           \
      vec = std::vector<dtype>(p, p + n);                                                      \
    }                                                                                          \
  }

#define arrayshape2vector(vec, arr)                                                            \
  {                                                                                            \
    if (arr.size() > 0) {                                                                      \
      vec.resize(arr.ndim());                                                                  \
      for (std::size_t i = 0; i < vec.size(); ++i)                                             \
        vec[i] = (int64_t)arr.shape(i);                                                        \
    }                                                                                          \
  }

template <class NTYPE> NTYPE flattened_dimension(const std::vector<NTYPE> &values) {
  NTYPE r = 1;
  for (auto it = values.begin(); it != values.end(); ++it)
    r *= *it;
  return r;
}

#if __cpluscplus >= 202002L
template <class NTYPE> NTYPE flattened_dimension(const std::span<NTYPE> &values) {
#else
template <class NTYPE> NTYPE flattened_dimension(const std_::span<NTYPE> &values) {
#endif
  NTYPE r = 1;
  for (auto it = values.begin(); it != values.end(); ++it)
    r *= *it;
  return r;
}

template <class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE> &values, int64_t first) {
  NTYPE r = 1;
  auto end = values.begin() + first;
  for (auto it = values.begin(); it != end; ++it)
    r *= *it;
  return r;
}

template <class DIMTYPE, class NTYPE>
void shape2strides(const std::vector<DIMTYPE> &shape, std::vector<DIMTYPE> &strides, NTYPE) {
  strides.resize(shape.size());
  strides[strides.size() - 1] = sizeof(NTYPE);
  for (int64_t i = (int64_t)strides.size() - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
}

template <class DIMTYPE>
DIMTYPE SizeFromDimension(const std::vector<DIMTYPE> &shape, std::size_t start,
                          std::size_t end) {
  DIMTYPE size = 1;
  for (std::size_t i = start; i < end; i++) {
    if (shape[i] < 0)
      return -1;
    size *= shape[i];
  }
  return size;
}

inline int64_t HandleNegativeAxis(int64_t axis, int64_t tensor_rank) {
  return axis < 0 ? axis + tensor_rank : axis;
}

} // namespace onnx_c_ops
