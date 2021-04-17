// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
/// <summary>
/// This function converts dense tensor into Csr format.
/// Conversion takes place on CPU. Thus if the source
/// is not on CPU, a copy if made first. Likewise, if the destination
/// is not on CPU, the function would perform a copy.
/// </summary>
/// <param name="data_manager"></param>
/// <param name="src">dense tensor</param>
/// <param name="cpu_allocator"></param>
/// <param name="dst_allocator">destination allocator</param>
/// <param name="dst">output</param>
/// <returns>Status</returns>
common::Status DenseTensorToSparseCsr(const DataTransferManager& data_manager, const Tensor& src, const AllocatorPtr& cpu_allocator,
                                      const AllocatorPtr& dst_allocator, SparseTensor& dst);

/// <summary>
/// Converts Csr format to Dense matrix
/// Conversion takes place on CPU. Thus if the source
/// is not on CPU, a copy if made first. Likewise, if the destination
/// is not on CPU, the function would perform a copy.
/// </summary>
/// <param name="data_manager"></param>
/// <param name="src">SparseTensor</param>
/// <param name="cpu_allocator"></param>
/// <param name="dst_allocator">destination allocator</param>
/// <param name="dst">out parameter</param>
/// <returns>Status</returns>
common::Status SparseCsrToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src, const AllocatorPtr& cpu_allocator,
                                      const AllocatorPtr& dst_allocator, Tensor& dst);

/// <summary>
/// Convert Dense Tensor to COO format
/// Conversion takes place on CPU. Thus if the source
/// is not on CPU, a copy if made first. Likewise, if the destination
/// is not on CPU, the function would perform a copy.
/// </summary>
/// <param name="data_manager">for X-dev copy</param>
/// <param name="src">dense tensor</param>
/// <param name="cpu_allocator">cpu_based allocator</param>
/// <param name="dst_allocator">destination_allocator</param>
/// <param name="liner_index">true if we want 1-D index and 2-D otehrwise</param>
/// <param name="dst">output parameter</param>
/// <returns>Status instance</returns>
common::Status DenseTensorToSparseCoo(const DataTransferManager& data_manager, const Tensor& src, const AllocatorPtr& cpu_allocator,
                                      const AllocatorPtr& dst_allocator, bool linear_indexs, SparseTensor& dst);

common::Status SparseCooToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src, const AllocatorPtr& cpu_allocator,
                                      const AllocatorPtr& dst_allocator, Tensor& dst);
}  // namespace sparse_utils
}  // namespace onnxruntime
