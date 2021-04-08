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
/// <summary>
/// Converts an input CUDA dense buffer into a Blocked Ell format
/// This is currently a utility function for Prepack testing.
/// The function serves MatMul purposes and as cuSparse requires a sparse
/// argument to be the first arg, we need to transpose by default and not transpose
/// if transpose is false.
/// </summary>
/// <param name="kernel">Prepacking kernel</param>
/// <param name="ell_block_size">ell block size</param>
/// <param name="K">rows in row major</param>
/// <param name="N">col</param>
/// <param name="transpose">transpose flag. The meaning is reverse</param>
/// <param name="element_type"></param>
/// <param name="element_size"></param>
/// <param name="input_data_initialier">device buffer ptr</param>
/// <param name="ell_indicies_buffer">out ell indicies</param>
/// <param name="ell_values_buffer">out ell values</param>
/// <param name="ell_cols">number of ell columns in the returned index</param>
/// <returns></returns>
Status ConvertToBlockedEll(const CudaKernel* kernel,
                           int64_t ell_block_size, int64_t K, int64_t N, bool transpose, int32_t element_type, size_t element_size,
                           const void* input_data_initialier, IAllocatorUniquePtr<uint8_t>& ell_indicies_buffer, IAllocatorUniquePtr<uint8_t>& ell_values_buffer,
                           int64_t& ell_cols);

/// <summary>
/// Executes prepack by converting a dense initializer buffer into NVIDIA Blocked Ell format.
/// If hardware A 100 or V 100 is not available, we default to CSR format and reflect the fact
/// in the final_param
/// </summary>
/// <param name="kernel"></param>
/// <param name="tensor">initializer tensor</param>
/// <param name="prepack_param"></param>
/// <param name="transb">transpose flag for B in MatMal. Reverse meaning.</param>
/// <param name="expected_kernel_type"></param>
/// <param name="cuda_type"></param>
/// <param name="final_param">a potentially modified copy of prepack params.</param>
/// <param name="sparse_buffers">output sparse buffers. The number and meaning depend on the format.</param>
/// <param name="sparse_desc">cusparse sparse matrix descriptor</param>
/// <param name="is_packed">true if packing succeeded</param>
/// <returns>OK if it is OK to continue to load the model</returns>
Status PrePack(const CudaKernel* kernel, const Tensor& tensor, const OpKernel::PrepackParam& prepack_param,
               bool transb, int32_t expected_kernel_type, cudaDataType cuda_type, OpKernel::PrepackParam& final_param,
               std::vector<IAllocatorUniquePtr<uint8_t>>& sparse_buffers,
               cusparseSpMatDescr_t& sparse_desc, bool& is_packed);

/// <summary>
/// Executes cuSparse MatMul
/// </summary>
/// <param name="kernel"></param>
/// <param name="ctx"></param>
/// <param name="param">Stored prepack param</param>
/// <param name="right_shape">original initialier shape</param>
/// <param name="sparse_desc">sparse initializer matrix descriptor</param>
/// <param name="alpha"></param>
/// <param name="transa"></param>
/// <param name="transb"></param>
/// <param name="cuda_type"></param>
/// <returns>status</returns>
Status Compute(const CudaKernel* kernel, OpKernelContext* ctx, const OpKernel::PrepackParam& param,
               const TensorShape& right_shape, cusparseSpMatDescr_t sparse_desc,
               float alpha, bool transa, bool transb, cudaDataType cuda_type);

}  // namespace cusparse_helper

}  // namespace cuda
}  // namespace onnxruntime
