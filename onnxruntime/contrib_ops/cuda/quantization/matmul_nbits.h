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
#include "contrib_ops/cuda/llm/weightOnlyGemmProfiler.h"
#include "contrib_ops/cuda/quantization/fpA_intB_gemm.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;
// using ort_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner;
// using ort_llm::kernels::weight_only::WeightOnlyQuantGemmPluginProfiler;
// using ort_llm::kernels::weight_only::GemmIdCore;
// using ort_llm::kernels::weight_only::GemmDims;
// using GemmProfilerPtr = std::shared_ptr<WeightOnlyQuantGemmPluginProfiler>;
// using WeightOnlyGemmRunnerPtr=std::shared_ptr<CutlassFpAIntBGemmRunnerInterface>;

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
    // constexpr size_t kInputIndexBias = 5;

    has_zero_points_ = info.GetInputCount() > kInputIndexZeroPoints && info.node().InputDefs()[kInputIndexZeroPoints]->Exists();
    has_g_idx_ = info.GetInputCount() > kInputIndexGroupIndex && info.node().InputDefs()[kInputIndexGroupIndex]->Exists();
    // has_bias_ = info.GetInputCount() > kInputIndexBias && info.node().InputDefs()[kInputIndexBias]->Exists();

    if (has_zero_points_) {
      int32_t zero_point_type = info.node().InputDefs()[kInputIndexZeroPoints]->TypeAsProto()->tensor_type().elem_type();
      int32_t scale_type = info.node().InputDefs()[kInputIndexScale]->TypeAsProto()->tensor_type().elem_type();
      is_zero_points_scale_same_type_ = (zero_point_type == scale_type);
    }

    if constexpr (std::is_same<T, MLFloat16>::value) {
      if ((block_size_ == 64 || (nbits_ == 4 && block_size_ == 128)) && (nbits_ == 4 || nbits_ == 8) && !has_g_idx_ && has_zero_points_ && N_ % (nbits_ == 8 ? 32 : 64) == 0 && K_ % 64 == 0) {
        fpA_intB_gemm::KernelType cuda_kernel_type = (nbits_ == 8)
                                                         ? fpA_intB_gemm::KernelType::FP16Int8Groupwise
                                                         : fpA_intB_gemm::KernelType::FP16Int4Groupwise;
        int sm = this->GetDeviceProp().major * 10 + this->GetDeviceProp().minor;
        if (fpA_intB_gemm::is_supported(sm, cuda_kernel_type)) {
          use_fpA_intB_gemm_ = true;
        }
      }
    }

    /*
    gemmProfiler_ = std::make_shared<WeightOnlyQuantGemmPluginProfiler>(); // TODO: use factory to create singleton instance.

    if constexpr (std::is_same<T, MLFloat16>::value) {
      if ((block_size_ == 64 || block_size_ == 128) && (nbits_ == 4 || nbits_ == 8) && has_zero_points_ && !has_g_idx_ && is_zero_points_scale_same_type_) {
        use_fpA_intB_gemm_ = true;
        gemmId_ = GemmIdCore(N_, K_, ort_llm::nvinfer1::DataType::kHALF);

        if (nbits_ == 8) {
          weightOnlyGemmRunner_ = std::make_shared<CutlassFpAIntBGemmRunner<half, uint8_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
          // gemmProfiler_->setWeightTypeId(WeightTypeId::INT8);
        }
        else if (nbits_ == 4) {
          weightOnlyGemmRunner_ = std::make_shared<CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
          //gemmProfiler_->setWeightTypeId(WeightTypeId::INT4);
          //gemmProfiler_->setCudaKernelType(ort_llm::kernels::weight_only::KernelType::FP16Int4Groupwise, 80);
        }

        int minM = 1;
        int maxM = 8192;
        GemmDims dims = {minM, maxM, N_, K_};

        // size_t smoothedActSize = static_cast<size_t>(maxM) * static_cast<size_t>(maxK)
        //     * (in[0].desc.type == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(half));
        // m_workspaceMaxSize = smoothedActSize + weightOnlyGemmRunner_->getWorkspaceSize(maxM, N_, K_);


        bool hasWeightOnlyCudaKernel = false;
        gemmProfiler_->profileTactics(weightOnlyGemmRunner_, ort_llm::nvinfer1::DataType::kHALF, dims, gemmId_, hasWeightOnlyCudaKernel);
      }
    }*/
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t nbits_;
  bool column_wise_quant_blk_{true};

  bool has_g_idx_{false};
  // bool has_bias_{false};
  bool has_zero_points_{false};
  bool is_zero_points_scale_same_type_{false};
  bool use_fpA_intB_gemm_{false};

  // WeightOnlyGemmRunnerPtr weightOnlyGemmRunner_{nullptr};
  // mutable GemmProfilerPtr gemmProfiler_{nullptr};
  // GemmIdCore gemmId_{};
  mutable std::once_flag fpA_intB_init_once_flag_;
  mutable IAllocatorUniquePtr<void> fpA_intB_weight_buffer_;
  mutable IAllocatorUniquePtr<void> fpA_intB_scale_buffer_;
  mutable IAllocatorUniquePtr<void> fpA_intB_zero_buffer_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
