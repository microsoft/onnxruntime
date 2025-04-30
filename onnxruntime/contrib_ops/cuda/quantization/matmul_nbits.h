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
namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;
using ort_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner;
using ort_llm::kernels::weight_only::WeightOnlyQuantGemmPluginProfiler;
using ort_llm::kernels::weight_only::GemmIdCore;
using GemmProfilerPtr = std::shared_ptr<WeightOnlyQuantGemmPluginProfiler>;
//using WeightOnlyGemmRunnerPtr=std::shared_ptr<CutlassFpAIntBGemmRunnerInterface>;

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
    //constexpr size_t kInputIndexBias = 5;


    has_zero_points_ = info.GetInputCount() > kInputIndexZeroPoints && info.node().InputDefs()[kInputIndexZeroPoints]->Exists();
    has_g_idx_ = info.GetInputCount() > kInputIndexGroupIndex && info.node().InputDefs()[kInputIndexGroupIndex]->Exists();
    //has_bias_ = info.GetInputCount() > kInputIndexBias && info.node().InputDefs()[kInputIndexBias]->Exists();

    if (has_zero_points_){
      int32_t zero_point_type = info.node().InputDefs()[kInputIndexZeroPoints]->TypeAsProto()->tensor_type().elem_type();
      int32_t scale_type = info.node().InputDefs()[kInputIndexScale]->TypeAsProto()->tensor_type().elem_type();
      is_zero_points_scale_same_type_ = (zero_point_type == scale_type);
    }

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
      }
    }    
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t nbits_;
  bool column_wise_quant_blk_{true};

  bool has_g_idx_{false};
  //bool has_bias_{false};
  bool has_zero_points_{false};
  bool is_zero_points_scale_same_type_{false};
  bool use_fpA_intB_gemm_{false};

  WeightOnlyGemmRunnerPtr weightOnlyGemmRunner_{nullptr};
  mutable GemmProfilerPtr gemmProfiler_{nullptr};
  GemmIdCore gemmId_{};
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
