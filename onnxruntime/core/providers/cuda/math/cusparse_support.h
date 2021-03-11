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
/// Captures Prepack() information along the data and its shape
/// </summary>
struct SparseInfo {
  OpKernel::PrepackParam param_;
  TensorShape shape_;
  ptrdiff_t K_, N_;                                            // computed in PrePack() and here for verification
  std::vector<IAllocatorUniquePtr<uint8_t>> prepack_buffers_;  // Typed buffer
#ifdef USE_CUSPARSE
  onnxruntime::optional<cusparseLtHandle_t> handle_lt_;
#endif
  onnxruntime::optional<cusparseSpMatDescr_t> sparse_desc_;

  SparseInfo(const OpKernel::PrepackParam& p, const TensorShape& shape)
      : param_(p), shape_(shape), prepack_buffers_() {}

  SparseInfo(const SparseInfo&) = delete;
  SparseInfo& operator=(const SparseInfo&) = delete;

  ~SparseInfo() {
    if (sparse_desc_.has_value()) {
      cusparseDestroySpMat(*sparse_desc_);
    }
#ifdef USE_CUSPARSE
    if (handle_lt_.has_value()) {
      cusparseLtDestroy(&*handle_lt_);
    }
#endif
  }
};

template <typename T>
struct IsNotZero {
  bool operator()(T v) const noexcept {
    return v != static_cast<T>(0);
  }
};

static const MLFloat16 zero_ml16(0.f);

template <>
struct IsNotZero<MLFloat16> {
  bool operator()(MLFloat16 v) const noexcept {
    return zero_ml16.val != v.val;
  }
};

static const BFloat16 zero_b16(0.f);

template <>
struct IsNotZero<BFloat16> {
  bool operator()(BFloat16 v) const noexcept {
    return zero_b16.val != v.val;
  }
};

/// <summary>
/// Finds the first non-zero entry and computes its col index.
/// Advances restart past the found entry. restart is nullptr if
/// reached the end of the block row.
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
struct FindNotZero {
  // returns nullptr if not found
  void operator()(int64_t N, int64_t ell_block_size,
                  const uint8_t* block_row_begin,
                  const uint8_t*& restart, const uint8_t* block_row_end,
                  int64_t& block_col) const {
    const T* block_row_begin_T = reinterpret_cast<const T*>(block_row_begin);
    const T* start = reinterpret_cast<const T*>(restart);
    const T* block_row_end_T = reinterpret_cast<const T*>(block_row_end);
    assert(start <= block_row_end_T);
    auto hit = std::find_if(start, block_row_end_T, IsNotZero<T>());
    if (hit != block_row_end_T) {
      block_col = ((hit - block_row_begin_T) % N) / ell_block_size;
      restart = reinterpret_cast<const uint8_t*>(hit + 1);
    } else {
      restart = nullptr;
    }
  }
};

Status ConvertToBlockedEll(const CudaKernel* kernel,
                           int64_t ell_block_size, int64_t K, int64_t N, bool transpose, int32_t element_type, size_t element_size,
                           const void* input_data_initialier, IAllocatorUniquePtr<uint8_t>& ell_indicies_buffer, IAllocatorUniquePtr<uint8_t>& ell_values_buffer,
                           int64_t& ell_cols);

Status PrePack(const CudaKernel* kernel, const Tensor& tensor, const OpKernel::PrepackParam& prepack_param,
               bool transb, int32_t expected_kernel_type, cudaDataType cuda_type,
               std::unique_ptr<SparseInfo>& sparse_info, bool& is_packed);

Status Compute(const CudaKernel* kernel, OpKernelContext* ctx, const SparseInfo& sparse_info,
               float alpha, bool transa, bool transb, cudaDataType cuda_type);

}  // namespace cusparse_helper

}  // namespace cuda
}  // namespace onnxruntime
