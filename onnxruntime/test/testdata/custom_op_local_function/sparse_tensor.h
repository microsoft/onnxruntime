#pragma once

#include "onnx_extended_helpers.h"
#include <cstring>
#include <map>
#include <vector>

namespace onnx_sparse {

template <typename T> struct CTypeToElementType {
  uint32_t onnx_type() const;
};
template <> struct CTypeToElementType<float> {
  inline uint32_t onnx_type() const { return 1; }
};
template <> struct CTypeToElementType<double> {
  inline uint32_t onnx_type() const { return 11; }
};

/**
 * This structure defines a 1D to 5D sparse tensor.
 * It assumes the sparse tensor has less than 4Gb non null elements.
 *
 */
struct sparse_struct {
  uint32_t fix_value;
  uint32_t n_dims;
  int64_t shape[4];
  uint32_t n_elements;
  uint32_t onnx_type;
  uint32_t begin;

  inline uint32_t *indices() const { return (uint32_t *)&begin; }
  inline float *values() const { return (float *)(indices() + n_elements); }
  static std::size_t element_size(uint32_t onnx_type) {
    switch (onnx_type) {
    case 1:
      return sizeof(float);
    case 11:
      return sizeof(double);
    default:
      EXT_THROW("Unsupported sparse element type.");
    }
  }
  inline std::size_t element_size() const { return element_size(onnx_type); }
  static inline std::size_t size_float(uint32_t n_elements, uint32_t onnx_type) {
    std::size_t el_size = element_size(onnx_type);
    return sizeof(sparse_struct) + n_elements + n_elements * el_size / 4 +
           (el_size % 4 ? 1 : 0);
  }
  inline std::size_t size_float() const { return size_float(n_elements, onnx_type); }

  void set(const std::vector<int64_t> &sh, uint32_t n, uint32_t dtype) {
    EXT_ENFORCE(sh.size() <= 5, "Sparse tensor must be 5D or less.");
    fix_value = 0b10101010101010101010101010101010;
    n_dims = static_cast<uint32_t>(sh.size());
    for (std::size_t i = 0; i < sh.size(); ++i)
      shape[i] = sh[i];
    onnx_type = dtype;
    n_elements = n;
  }

  static void copy(const std::vector<int64_t> &shape, const std::vector<uint32_t> &indices,
                   const std::vector<float> &values, std::vector<float> &result) {
    EXT_ENFORCE(shape.size() <= 5, "Sparse tensor must be 5D or less.");
    EXT_ENFORCE(indices.size() == values.size(), "indices and values must have the same size.");
    sparse_struct sp;
    sp.set(shape, static_cast<uint32_t>(indices.size()), 1);
    result.resize(sp.size_float());
    std::memcpy(static_cast<void *>(result.data()), static_cast<void *>(&sp), sizeof(sp) - 4);
    if (!indices.empty()) {
      sparse_struct &sps = *(sparse_struct *)result.data();
      std::memcpy(sps.indices(), indices.data(), indices.size() * sizeof(uint32_t));
      std::memcpy(sps.values(), values.data(), values.size() * sizeof(float));
    }
  }

  void unmake(uint32_t &dims, uint32_t &n, const int64_t *&out_shape, uint32_t *&out_indices,
              float *&out_values) const {
    EXT_ENFORCE(fix_value, 0b10101010101010101010101010101010,
                "The structure is not a sparse tensor.");
    EXT_ENFORCE(onnx_type == 1,
                "The structure does not contain float values, onnx_type=", onnx_type, ".");
    dims = n_dims;
    out_shape = shape;
    out_indices = indices();
    out_values = values();
    n = n_elements;
  }

  template <typename T> // std::vector<std::unordered_map<uint32_t, float>>
  void to_maps(T &maps) const {
    EXT_ENFORCE(n_dims == 2, "to_maps only works with 2D matrices.");
    maps.resize(shape[0]);
    for (auto it : maps)
      it.clear();
    // The implementation could be parallelized.
    uint32_t row, col, pos;
    const uint32_t *ind = indices();
    const float *val = values();
    for (uint32_t i = 0; i < n_elements; ++i) {
      pos = ind[i];
      row = pos / shape[1];
      col = pos % shape[1];
      // EXT_ENFORCE(row < shape[0]);
      maps[row][col] = val[i];
    }
  }

  inline int64_t flatten_shape(int last = -1) const {
    int64_t res = 1;
    std::size_t end = last == -1 ? n_dims : static_cast<std::size_t>(last);
    for (std::size_t i = 0; i < end; ++i)
      res *= shape[i];
    return res;
  }

  void csr(std::vector<uint32_t> &rows_index) const {
    if (n_elements == 0) {
      rows_index.reserve(1);
      rows_index.push_back(static_cast<uint32_t>(0));
      return;
    }
    uint32_t *ind = indices();
    uint32_t last_dim = static_cast<uint32_t>(shape[n_dims - 1]);
    std::size_t expected =
        static_cast<std::size_t>(n_dims == 2 ? shape[0] : flatten_shape(n_dims - 1));
    rows_index.reserve(expected + 1);
    uint32_t row, new_row;
    // The implementation could be parallelized assuming rows have
    // approxatively the same amount of values.
    new_row = ind[0] / last_dim;
    for (row = 0; row <= new_row; ++row) {
      rows_index.push_back(static_cast<uint32_t>(0));
    }
    for (uint32_t i = 1; i < n_elements; ++i) {
      new_row = ind[i] / last_dim;
      EXT_ENFORCE(ind[i] < ind[i + 1], "indices are not sorted,", ind[i], ">=", ind[i + 1],
                  ".");
      for (; row != new_row; ++row) {
        rows_index.push_back(static_cast<uint32_t>(i));
      }
    }
    while (rows_index.size() <= expected)
      rows_index.push_back(static_cast<uint32_t>(n_elements));
  }

  void csr(std::vector<uint32_t> &rows_index, std::vector<uint32_t> &element_indices) const {
    if (n_elements == 0) {
      rows_index.reserve(1);
      rows_index.push_back(static_cast<uint32_t>(0));
      return;
    }
    element_indices.resize(n_elements);
    uint32_t *ind = indices();
    uint32_t last_dim = static_cast<uint32_t>(shape[n_dims - 1]);
    std::size_t expected =
        static_cast<std::size_t>(n_dims == 2 ? shape[0] : flatten_shape(n_dims - 1));
    rows_index.reserve(expected + 1);
    uint32_t row, new_row;
    // The implementation could be parallelized assuming rows have
    // approxatively the same amount of values.
    new_row = ind[0] / last_dim;
    for (row = 0; row <= new_row; ++row) {
      rows_index.push_back(static_cast<uint32_t>(0));
    }
    element_indices[0] = ind[0] - row * last_dim;
    for (uint32_t i = 1; i < n_elements; ++i) {
      new_row = ind[i] / last_dim;
      EXT_ENFORCE(ind[i] < ind[i + 1], "indices are not sorted.");
      for (; row != new_row; ++row) {
        rows_index.push_back(static_cast<uint32_t>(i));
      }
      element_indices[i] = ind[i] - row * last_dim;
    }
    while (rows_index.size() <= expected)
      rows_index.push_back(static_cast<uint32_t>(n_elements));
  }

}; // struct sparse_struct

} // namespace onnx_sparse
