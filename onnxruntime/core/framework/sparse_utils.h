// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#pragma once

#include "core/framework/allocator.h"

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

}  // namespace sparse_utils
}  // namespace onnxruntime

#endif  //!defined(DISABLE_SPARSE_TENSORS)