// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#pragma once

#include "core/framework/allocator.h"
#include "gsl/gsl"

namespace onnxruntime {
#ifndef SHARED_PROVIDER
class Tensor;
class SparseTensor;
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

template<>
inline void CopyElement<std::string>(void* dst, const void* src, int64_t dst_index, int64_t src_index) {
  reinterpret_cast<std::string*>(dst)[dst_index] = reinterpret_cast<const std::string*>(src)[src_index];
}

Status Convert2DCooIndicesTo1D(int64_t cols, gsl::span<const int64_t> input, std::vector<int64_t>& output);

}  // namespace sparse_utils
}  // namespace onnxruntime

#endif  //!defined(DISABLE_SPARSE_TENSORS)