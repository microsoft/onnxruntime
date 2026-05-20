// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/moe/moe_base.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.h"

#include <mutex>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class QMoE final : public CudaKernel, public MoEBase {
 public:
  explicit QMoE(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;

 private:
  // PrePack helpers - each handles one category of input tensor.
  void PrePackTransposeAndPack(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                               IAllocatorUniquePtr<void>& packed_buf, bool& is_packed);
  void PrePackComputeBias(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                          const IAllocatorUniquePtr<void>& packed_scale,
                          IAllocatorUniquePtr<void>& packed_bias, bool& is_packed);
  void PrePackCopyToGpu(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                        IAllocatorUniquePtr<void>& packed_buf, bool& is_packed);
  void PrePackSwizzleBlockScales(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                 IAllocatorUniquePtr<void>& packed_buf, bool& is_packed);
  void PrePackRepackFP4Weights(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                               IAllocatorUniquePtr<void>& packed_buf, bool& is_packed);
  int64_t expert_weight_bits_;
  bool is_fp16_;
  bool use_fp4_dequant_fallback_ = false;
  // Dequantizes FP8 weights to FP16/BF16 scratch buffers before invoking the A16 MoE runner.
  bool use_fp8_dequant_fallback_ = false;
  // WFP4AFP8 (W4A8) requires SM100+ (Blackwell) block-scaled tensor ops. On older GPUs we
  // dequantize MXFP4 weights to FP16/BF16 and run the dense A16 MoE runner.
  bool use_wfp4afp8_dequant_fallback_ = false;
  std::string quant_type_;  // "int", "fp4", "fp8", or "wfp4afp8"

  std::unique_ptr<onnxruntime::llm::kernels::cutlass_kernels::CutlassMoeFCRunnerInterface> m_moe_runner;

  // Pre-packed buffers
  // Note: For QMoE, we need both Scales (for dequant) and Bias (derived from ZP/Scale) during inference.
  // PrePack logic:
  // - Copies scales to GPU buffer (if in CPU) or just keeps them. For simplicity, we allocate and copy.
  // - Computes Bias from ZP and Scale using PrePack kernel.
  IAllocatorUniquePtr<void> packed_fc1_scales_;
  IAllocatorUniquePtr<void> packed_fc1_bias_;
  IAllocatorUniquePtr<void> packed_fc2_scales_;
  IAllocatorUniquePtr<void> packed_fc2_bias_;

  // FP4 pre-packed buffers
  IAllocatorUniquePtr<void> packed_fp4_fc1_weights_;
  IAllocatorUniquePtr<void> packed_fp4_fc2_weights_;
  IAllocatorUniquePtr<void> packed_fp4_fc1_block_scales_;
  IAllocatorUniquePtr<void> packed_fp4_fc2_block_scales_;

  // Per-expert global weight scales used by FP4 and FP8 modes.
  IAllocatorUniquePtr<void> packed_fc1_global_scale_;
  IAllocatorUniquePtr<void> packed_fc2_global_scale_;

  // Per-tensor or per-expert FP8 activation global scales used by W4A8 (WFP4AFP8) Variant A.
  // Inputs 17/18 in the QMoE schema. Optional; absent for the MXFP8 block-scaled variant.
  IAllocatorUniquePtr<void> packed_fc1_act_scale_;
  IAllocatorUniquePtr<void> packed_fc2_act_scale_;

  mutable onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler mGemmProfiler;
  mutable onnxruntime::llm::kernels::cutlass_kernels::MoeGemmId mGemmId1;
  mutable onnxruntime::llm::kernels::cutlass_kernels::MoeGemmId mGemmId2;
  mutable std::mutex mGemmProfilerMutex;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
