// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/framework/data_types.h"
#include "core/framework/sparse_tensor.h"

using namespace onnxruntime::common;

namespace onnxruntime {

SparseTensor::SparseTensor(MLDataType elt_type,
                           const TensorShape& shape,
                           size_t nnz,
                           void* values_data,
                           void* indices_data,
                           const OrtMemoryInfo& memory_info)
    : values_(elt_type, TensorShape({static_cast<int64_t>(nnz)}), values_data, memory_info),
      indices_(DataTypeImpl::GetType<int64_t>(),
               TensorShape({static_cast<int64_t>(nnz), static_cast<int64_t>(shape.NumDimensions())}),
               indices_data, memory_info, 0),
      shape_(shape) {}

SparseTensor::SparseTensor(MLDataType elt_type,
                           const TensorShape& shape,
                           size_t nnz,
                           std::shared_ptr<IAllocator> allocator)
    : values_(elt_type, TensorShape({static_cast<int64_t>(nnz)}), allocator),
      indices_(DataTypeImpl::GetType<int64_t>(),
               TensorShape({static_cast<int64_t>(nnz), static_cast<int64_t>(shape.NumDimensions())}),
               allocator),
      shape_(shape) {}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
