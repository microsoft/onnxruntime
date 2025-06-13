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
#include "contrib_ops/cuda/llm/fpA_intB_gemm_profiler.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cpu/utils/measure_latency.h"

#include <iostream>

#define FPA_INTB_GEMM_LATENCY 1

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;
using onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner;
using onnxruntime::llm::kernels::weight_only::GemmDims;
using onnxruntime::llm::kernels::weight_only::GemmIdCore;
using onnxruntime::llm::kernels::weight_only::GemmPluginProfilerManager;
using onnxruntime::llm::kernels::weight_only::WeightOnlyGroupwiseQuantGemmPluginProfiler;
using GemmProfilerPtr = std::shared_ptr<WeightOnlyGroupwiseQuantGemmPluginProfiler>;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface>;

// Environment variable to configure fpA_intB_gemm for experiments. Set it to 0 to disable, 1 to eanble all.
constexpr const char* kFpAIntBGemmOption = "ORT_FPA_INTB_GEMM";
constexpr int kFpAIntBGemmOption_All = 0x01;
constexpr int kFpAIntBGemmOption_Gemv = 0x02;
constexpr int kFpAIntBGemmOption_Int4 = 0x04;
constexpr int kFpAIntBGemmOption_Int8 = 0x08;

constexpr const char* kFpAIntBGemmInitM = "ORT_FPA_INTB_GEMM_INIT_M";

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
    sm_ = this->GetDeviceProp().major * 10 + this->GetDeviceProp().minor;

    if (has_zero_points_) {
      int32_t zero_point_type = info.node().InputDefs()[kInputIndexZeroPoints]->TypeAsProto()->tensor_type().elem_type();
      int32_t scale_type = info.node().InputDefs()[kInputIndexScale]->TypeAsProto()->tensor_type().elem_type();
      is_zero_points_scale_same_type_ = (zero_point_type == scale_type);
    }

    if constexpr (std::is_same<T, MLFloat16>::value) {
      int option = ParseEnvironmentVariableWithDefault<int>(kFpAIntBGemmOption, 0);
      if ((option & (static_cast<int>(nbits_) | kFpAIntBGemmOption_All)) != 0 &&
          (block_size_ == 64 || block_size_ == 128) &&
          (nbits_ == 4 || nbits_ == 8) &&
          !has_g_idx_ && !has_bias_ &&
          N_ % (nbits_ == 8 ? 32 : 64) == 0 &&
          K_ % block_size_ == 0 &&
          sm_ >= 75) {
        if ((option & (kFpAIntBGemmOption_Gemv | kFpAIntBGemmOption_All)) != 0) {
          using onnxruntime::llm::kernels::fpA_intB_gemv::KernelType;
          KernelType cuda_kernel_type = (nbits_ == 8) ? KernelType::FP16Int8Groupwise : KernelType::FP16Int4Groupwise;
          if (onnxruntime::llm::kernels::fpA_intB_gemv::is_supported(sm_, cuda_kernel_type)) {
            has_fpA_intB_gemv_ = true;
          }
        }

        InitGemmProfiler(sm_);

        int max_m = ParseEnvironmentVariableWithDefault<int>(kFpAIntBGemmInitM, 16);

#ifdef FPA_INTB_GEMM_LATENCY
        std::cout << "Gemm Profile for N=" << N_ << ", K=" << K_ << ", M=1~" << max_m << std::endl;
        auto latency_us = measure_latency([&]() {
#endif
          RunGemmProfile(has_fpA_intB_gemv_, 1, max_m);

#ifdef FPA_INTB_GEMM_LATENCY
        });
        std::cout << "Latency: " << latency_us << " microseconds" << std::endl;;
#endif

        has_fpA_intB_gemm_ = true;
        max_m_ = max_m;
      }
    }

#ifndef NDEBUG
    printf("n=%d, k=%d, block_size=%d, bits=%d, zp_bits=%d, g_idx=%d, bias=%d, gemv=%d, gemm=%d\n",
           int(N_), int(K_), int(block_size_), int(nbits_),
           has_zero_points_ ? (is_zero_points_scale_same_type_ ? int(sizeof(T)) * 8 : int(nbits_)) : int(0),
           int(has_g_idx_ ? 1 : 0), int(has_bias_ ? 1 : 0),
           int(has_fpA_intB_gemv_), int(has_fpA_intB_gemm_));
#endif
  }

  Status ComputeInternal(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;

 private:
  void InitGemmProfiler(int sm);
  void RunGemmProfile(bool hasWeightOnlyCudaKernel, int min_m, int max_m);

  Status PrePack_B(const Tensor& tensor, AllocatorPtr alloc, cudaStream_t stream);
  Status PrePack_Scale(const Tensor& tensor, AllocatorPtr alloc, cudaStream_t stream);
  Status PrePack_ZeroPoint(const Tensor& tensor, AllocatorPtr alloc, cudaStream_t stream);

  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t nbits_;
  int sm_{0};
  bool column_wise_quant_blk_{true};

  bool has_g_idx_{false};
  bool has_bias_{false};
  bool has_zero_points_{false};
  bool is_zero_points_scale_same_type_{false};
  bool has_fpA_intB_gemv_{false};
  bool has_fpA_intB_gemm_{false};

  mutable int max_m_{0};

  WeightOnlyGemmRunnerPtr weightOnlyGemmRunner_{nullptr};
  mutable GemmProfilerPtr gemmProfiler_{nullptr};
  GemmIdCore gemmId_{};

  IAllocatorUniquePtr<void> fpA_intB_weight_buffer_;
  IAllocatorUniquePtr<void> fpA_intB_scale_buffer_;
  IAllocatorUniquePtr<void> fpA_intB_zero_buffer_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
