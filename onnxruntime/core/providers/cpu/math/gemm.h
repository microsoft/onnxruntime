// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gemm_base.h"

#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include "core/util/math.h"
#include "core/providers/cpu/activation/activations.h"
#include "core/providers/cpu/mlas_backend_kernel_selector_config_utils.h"

namespace onnxruntime {

void Gemm_MLFloat16(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,                                 // 0, 1
                    ptrdiff_t M, ptrdiff_t N, ptrdiff_t K,                                            // 2, 3, 4
                    MLFloat16 alpha,                                                                  // 5
                    const MLFloat16* a_data, const MLFloat16* b_data,                                 // 6, 7
                    MLFloat16 beta,                                                                   // 8
                    const MLFloat16* c_data, const TensorShape* c_shape,                              // 9, 10
                    MLFloat16* y_data,                                                                // 11
                    concurrency::ThreadPool* thread_pool,                                             // 12
                    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* mlas_backend_kernel_selector_config);  // 13

template <typename T>
class Gemm : protected GemmBase, public OpKernel {
 public:
  Gemm(const OpKernelInfo& info) : GemmBase(info), OpKernel(info) {
    SetupMlasBackendKernelSelectorFromConfigOptions(mlas_backend_kernel_selector_config_, info.GetConfigOptions());
  }

  Status Compute(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

  static void ComputeGemm(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
                          ptrdiff_t M, ptrdiff_t N, ptrdiff_t K,
                          T alpha,
                          const T* a_data, const T* b_data,
                          T beta,
                          const T* c_data, const TensorShape* c_shape,
                          T* y_data,
                          concurrency::ThreadPool* thread_pool,
                          const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* mlas_backend_kernel_selector_config);

 protected:
  TensorShape b_shape_;
  IAllocatorUniquePtr<void> packed_b_;

  // For fused gemm + activation
  std::unique_ptr<functors::ElementWiseRangedTransform<T>> activation_;

  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config_;

  void ComputeActivation(_Inout_updates_(y_size) T* y_data, ptrdiff_t y_size, _Inout_opt_ concurrency::ThreadPool* thread_pool) const;
};

}  // namespace onnxruntime
