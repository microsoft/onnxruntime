// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(DISABLE_SPARSE_TENSORS)

#include "core/framework/allocator.h"
#include "gsl/gsl"
#include <functional>

#ifndef SHARED_PROVIDER
#include "core/framework/sparse_tensor.h"
#endif

namespace onnxruntime {
#ifndef SHARED_PROVIDER
class Tensor;
class DataTransferManager;

namespace common {
class Status;
}
#endif

namespace sparse_utils {
#if !defined(ORT_MINIMAL_BUILD)
/// <summary>
/// This function converts dense tensor into Csr format.
/// Conversion takes place on CPU. Thus if the source
/// is not on CPU, the source is copied to CPU. Likewise, if the destination
/// is not on CPU, the function would perform a copy.
/// std::string src and destination are assumed to be both on CPU.
/// </summary>
/// <param name="data_manager">used for X-device copy</param>
/// <param name="src">source dense tensor</param>
/// <param name="cpu_allocator">CPU based allocator</param>
/// <param name="dst_allocator">destination device allocator</param>
/// <param name="dst">output tensor</param>
///   <example>
///   <code>
///   Tensor src; // Tensor to convert
///   SparseTensor dst;
///   ORT_RETURN_IF_ERROR(DenseTensorToSparse(data_manager, src, cpu_allocator, dst_allocator, dst));
///   </code>
///   </example>
/// <returns>Status instance</returns>
Status DenseTensorToSparseCsr(const DataTransferManager& data_manager, const Tensor& src, const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, SparseTensor& dst);

/// <summary>
/// Converts Csr format to Dense matrix.
/// Conversion takes place on CPU. Thus if the source
/// is not on CPU, the source is copied to CPU. Likewise, if the destination
/// is not on CPU, the function would perform a copy.
/// std::string src and destination are assumed to be both on CPU.
/// </summary>
/// <param name="data_manager">used for X-device copy</param>
/// <param name="src">sparse tensor to convert to dense</param>
/// <param name="cpu_allocator">CPU based allocator</param>
/// <param name="dst_allocator">destination device allocator</param>
/// <param name="dst">output tensor</param>
///   <example>
///   <code>
///   SparseTensor src; // Tensor to make dense
///   Tensor dst;
///   ORT_RETURN_IF_ERROR(SparseCsrToDenseTensor(data_manager, src, cpu_allocator, dst_allocator, dst));
///   </code>
/// <returns>Status instance</returns>
Status SparseCsrToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src, const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, Tensor& dst);

/// <summary>
/// Convert COO format to dense matrix.
/// Conversion takes place on CPU. Thus if the source
/// is not on CPU, the source is copied to CPU. Likewise, if the destination
/// is not on CPU, the function would perform a copy.
/// std::string src and destination are assumed to be both on CPU.
/// </summary>
/// <param name="data_manager">used for X-device copy</param>
/// <param name="src">sparse tensor to convert to dense</param>
/// <param name="cpu_allocator">CPU based allocator</param>
/// <param name="dst_allocator">destination device allocator</param>
/// <param name="dst">output sparse tensor</param>
/// <returns>Status instance</returns>
Status SparseCooToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src, const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, Tensor& dst);
#endif  //!ORT_MINIMAL_BUILD

/// <summary>
/// Convert Dense Tensor to COO format.
/// Conversion takes place on CPU. Thus if the source
/// is not on CPU, the source is copied to CPU. Likewise, if the destination
/// is not on CPU, the function would perform a copy.
/// std::string src and destination are assumed to be both on CPU.
/// </summary>
/// <param name="data_manager">used for X-device copy</param>
/// <param name="src">dense tensor</param>
/// <param name="cpu_allocator">CPU based allocator</param>
/// <param name="dst_allocator">destination device allocator</param>
/// <param name="dst">output sparse tensor</param>
/// <returns>Status instance</returns>
Status DenseTensorToSparseCoo(const DataTransferManager& data_manager, const Tensor& src, const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, bool linear_index, SparseTensor& dst);

// Determines if this is a type specific zero
using IsZeroFunc = bool (*)(const void*);
// Copy element
using CopyElementFunc = void (*)(void* dest, const void* src, int64_t dest_index, int64_t src_index);

// Here we are not using tolerance for FP types since these dense tensors were
// created from sparse initializers where zeros were absolute
template <typename T>
inline bool IsZero(const void* p) {
  return (static_cast<T>(0) == *reinterpret_cast<const T*>(p));
}

template <typename T>
struct NotZero {
  bool operator()(T v) const {
    return v != T{0};
  }
};

template <>
struct NotZero<std::string> {
  bool operator()(const std::string& s) const {
    return !s.empty();
  }
};

/// <summary>
/// Useful when dealing with non-aligned buffers so we copy data byte by byte
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="dst"></param>
/// <param name="src"></param>
/// <param name="dst_index"></param>
/// <param name="src_index"></param>
template <typename T>
inline void CopyElement(void* dst, const void* src, int64_t dst_index, int64_t src_index) {
  const auto* src_p = reinterpret_cast<const T*>(src) + src_index;
  auto* dst_p = reinterpret_cast<T*>(dst) + dst_index;
  memcpy(dst_p, src_p, sizeof(T));
}

template <>
inline void CopyElement<uint8_t>(void* dst, const void* src, int64_t dst_index, int64_t src_index) {
  reinterpret_cast<uint8_t*>(dst)[dst_index] = reinterpret_cast<const uint8_t*>(src)[src_index];
}

template <>
inline void CopyElement<std::string>(void* dst, const void* src, int64_t dst_index, int64_t src_index) {
  reinterpret_cast<std::string*>(dst)[dst_index] = reinterpret_cast<const std::string*>(src)[src_index];
}

template <typename T>
inline void CopyElementAligned(void* dst, const void* src, int64_t dst_index, int64_t src_index) {
  reinterpret_cast<T*>(dst)[dst_index] = reinterpret_cast<const T*>(src)[src_index];
}

/// <summary>
/// Advance through binary data in element_size increments.
/// </summary>
struct Advance {
  size_t element_size_;
  explicit Advance(size_t element_size) : element_size_(element_size) {}
  const void* operator()(const void* start, size_t elements) const {
    return (reinterpret_cast<const uint8_t*>(start) + elements * element_size_);
  }
};

/// <summary>
/// Converts 2-D COO indices into 1-D flat indices.
/// The data is assumed to be on CPU.
/// </summary>
/// <param name="cols">cols for the 2-D dense shape</param>
/// <param name="input_span">original @-d indices</param>
/// <param name="output_span"></param>
/// <returns>Status</returns>
Status Convert2DCooIndicesTo1D(int64_t cols, gsl::span<const int64_t> input_span, gsl::span<int64_t> output_span);

/// <summary>
/// Calls Convert2DCooIndicesTo1D() and copies into the coo_mutator
/// The data is assumed to be on CPU.
/// </summary>
/// <param name="input_sparse"></param>
/// <param name="coo_mutator"></param>
/// <returns></returns>
Status ConvertIndicesTo1DAndCopy(const SparseTensor& input_sparse, SparseTensor::CooMutator& coo_mutator);

/// <summary>
/// The function performs conversion of input COO indices into
/// CSR indices and places results into inner_indices and outer_indices vectors.
/// In case we have a 1-D dense shape (a vector) then the inner indices is a just a copy
/// of input_indices and we have two entries in the outer_indices. Thus, the CSR produced
/// as if the input was a row vector. Make sure to flip the transpose flag and swap the dims
/// if the vector is really a column vector.
/// </summary>
/// <param name="computed_dims">shape that was adjusted for 2-D</param>
/// <param name="input_indices_ndims">indices either 1-D or 2-D indices</param>
/// <param name="input_indices"></param>
/// <param name="inner_indices"></param>
/// <param name="outer_indices"></param>
/// <returns></returns>
Status ConvertCooIndicesToCsrIndices(const std::vector<int64_t>& computed_dims, size_t input_indices_ndims,
                                     const gsl::span<const int64_t>& input_indices,
                                     std::vector<int64_t>& inner_indices,
                                     std::vector<int64_t>& outer_indices);

/// <summary>
/// Converts Csr indices into 1-D COO indices
/// </summary>
/// <param name="cols"></param>
/// <param name="input_inner"></param>
/// <param name="input_outer"></param>
/// <param name="output_indices"></param>
/// <returns></returns>
Status ConvertCsrIndicesToCooIndices(int64_t cols, gsl::span<const int64_t> input_inner,
                                     gsl::span<const int64_t> input_outer,
                                     gsl::span<int64_t> output_indices);

/// <summary>
/// Copies one tensor to another on CPU
/// </summary>
/// <param name="src"></param>
/// <param name="dst"></param>
void CopyCpuTensor(const Tensor& src, Tensor& dst);

/// <summary>
/// Copies values from input to destination values tensor
/// </summary>
/// <param name="input_sparse"></param>
/// <param name="output_values"></param>
inline void CopySparseCpuValues(const SparseTensor& src, Tensor& output_values) {
  CopyCpuTensor(src.Values(), output_values);
}


/// <summary>
/// Utility to copy data from one sparse tensor to another.
/// SparseTensor::Copy requires tensors to have same shapes, otherwise it
/// is preferable.
/// This assumes the destination has a compatible shapes. I.e.
/// all source indices remain valid.
/// </summary>
/// <param name="src"></param>
/// <param name="tgt"></param>
void CopyCpuSparseCooTensor(const SparseTensor& src, SparseTensor& tgt);

/// <summary>
/// Scans input indices for matching indices and fires callback with its
/// relative offsets so one can retrieve values and indices. Used for products
/// of sparse matrices
/// </summary>
/// <param name="a_indices">1d coo indices</param>
/// <param name="b_indices">1d coo indices</param>
/// <param name="match_cb">callback for match of indices</param>
void ScanForSparseMatches(const gsl::span<const int64_t>& a_indices,
                          const gsl::span<const int64_t>& b_indices,
                          std::function<void(size_t, size_t)> match_cb);

}  // namespace sparse_utils
}  // namespace onnxruntime

#endif  //!defined(DISABLE_SPARSE_TENSORS)