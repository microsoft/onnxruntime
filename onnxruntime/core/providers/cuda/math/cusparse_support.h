// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime {

class OpKernelContext;
namespace cuda {

class CudaKernel;

namespace cusparse_helper {
Status ConvertToBlockedEll(const CudaKernel* kernel,
                           int64_t ell_block_size, int64_t K, int64_t N, bool transpose, int32_t element_type, size_t element_size,
                           const void* input_data_initialier, IAllocatorUniquePtr<uint8_t>& ell_indicies_buffer, IAllocatorUniquePtr<uint8_t>& ell_values_buffer,
                           int64_t& ell_cols);

Status PrePack(const CudaKernel* kernel, const Tensor& tensor, const OpKernel::PrepackParam& prepack_param,
               bool transb, int32_t expected_kernel_type, cudaDataType cuda_type, OpKernel::PrepackParam& final_param,
               std::vector<IAllocatorUniquePtr<uint8_t>>& sparse_buffers,
               cusparseSpMatDescr_t& sparse_desc, bool& is_packed);

Status Compute(const CudaKernel* kernel, OpKernelContext* ctx, const OpKernel::PrepackParam& param,
               const TensorShape& right_shape, cusparseSpMatDescr_t sparse_desc,
               float alpha, bool transa, bool transb, cudaDataType cuda_type);

}  // namespace cusparse_helper

}  // namespace cuda
}  // namespace onnxruntime
