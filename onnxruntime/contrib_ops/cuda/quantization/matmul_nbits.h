// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define MatMulNBits operator, it is basically
// matmul float with right hand side being a 2-D matrix
// pre-packed and block-compacted into int4
//
#pragma once
#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/llm/gemmProfiler.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_profiler.h"
#include "contrib_ops/cuda/quantization/fpA_intB_gemm.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;
using ort_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner;
using ort_llm::kernels::weight_only::GemmPluginProfilerManager;
using ort_llm::kernels::weight_only::WeightOnlyGroupwiseQuantGemmPluginProfiler;
using ort_llm::kernels::weight_only::GemmIdCore;
using ort_llm::kernels::weight_only::GemmDims;
using GemmProfilerPtr = std::shared_ptr<WeightOnlyGroupwiseQuantGemmPluginProfiler>;
using WeightOnlyGemmRunnerPtr=std::shared_ptr<ort_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface>;

template <typename T>
class MatMulNBits final : public CudaKernel {
 public:
  MatMulNBits(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("bits", &nbits_));

    constexpr size_t kInputIndexScale = 2;
    constexpr size_t kInputIndexZeroPoints = 3;
    constexpr size_t kInputIndexGroupIndex = 4;
    constexpr size_t kInputIndexBias = 5;

    has_zero_points_ = info.GetInputCount() > kInputIndexZeroPoints && info.node().InputDefs()[kInputIndexZeroPoints]->Exists();
    has_g_idx_ = info.GetInputCount() > kInputIndexGroupIndex && info.node().InputDefs()[kInputIndexGroupIndex]->Exists();
    has_bias_ = info.GetInputCount() > kInputIndexBias && info.node().InputDefs()[kInputIndexBias]->Exists();

    if (has_zero_points_) {
      int32_t zero_point_type = info.node().InputDefs()[kInputIndexZeroPoints]->TypeAsProto()->tensor_type().elem_type();
      int32_t scale_type = info.node().InputDefs()[kInputIndexScale]->TypeAsProto()->tensor_type().elem_type();
      is_zero_points_scale_same_type_ = (zero_point_type == scale_type);
    }

    if constexpr (std::is_same<T, MLFloat16>::value) {
      if ((block_size_ == 64 || (nbits_ == 4 && block_size_ == 128)) && (nbits_ == 4 || nbits_ == 8) && !has_g_idx_ && has_zero_points_ && !has_bias_ && N_ % (nbits_ == 8 ? 32 : 64) == 0 && K_ % block_size_ == 0) {
        fpA_intB_gemm::KernelType cuda_kernel_type = (nbits_ == 8)
                                                         ? fpA_intB_gemm::KernelType::FP16Int8Groupwise
                                                         : fpA_intB_gemm::KernelType::FP16Int4Groupwise;
        int sm = this->GetDeviceProp().major * 10 + this->GetDeviceProp().minor;
        if (fpA_intB_gemm::is_supported(sm, cuda_kernel_type)) {
          has_fpA_intB_gemv_ = true;
        }

        if (sm >= 75) {
          RunGemmProfile(has_fpA_intB_gemv_, sm, 2048);
          has_fpA_intB_gemm_ = true;
        }
      }
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  void RunGemmProfile(bool hasCudaKernel, int sm, int max_m);

  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t nbits_;
  bool column_wise_quant_blk_{true};

  bool has_g_idx_{false};
  bool has_bias_{false};
  bool has_zero_points_{false};
  bool is_zero_points_scale_same_type_{false};
  bool has_fpA_intB_gemv_{false};
  bool has_fpA_intB_gemm_{false};

  WeightOnlyGemmRunnerPtr weightOnlyGemmRunner_{nullptr};
  mutable GemmProfilerPtr gemmProfiler_{nullptr};
  GemmIdCore gemmId_{};


  mutable std::once_flag fpA_intB_init_once_flag_;
  mutable IAllocatorUniquePtr<void> fpA_intB_weight_buffer_;
  mutable IAllocatorUniquePtr<void> fpA_intB_scale_buffer_;
  mutable IAllocatorUniquePtr<void> fpA_intB_zero_buffer_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
