// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define MatMulNBits operator, it is basically
// matmul float with right hand side being a 2-D matrix
// pre-packed and block-compacted into int4
//
#pragma once
#include <string>
#include <vector>
#include "core/common/safeint.h"
#include "core/common/string_utils.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_profiler.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;

// Environment variables to force the chunked dequant+GEMM path for testing.
// ORT_MATMULNBITS_FORCE_CHUNKED=1 bypasses the scratch-size and min-N guards.
// ORT_MATMULNBITS_CHUNK_SIZE overrides the default chunk size (32768).
constexpr const char* kForceChunkedEnvVar = "ORT_MATMULNBITS_FORCE_CHUNKED";
constexpr const char* kChunkSizeEnvVar = "ORT_MATMULNBITS_CHUNK_SIZE";
constexpr int64_t kDefaultChunkTargetRows = 32768;

#if USE_FPA_INTB_GEMM
using onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner;
using onnxruntime::llm::kernels::weight_only::GemmDims;
using onnxruntime::llm::kernels::weight_only::GemmIdCore;
using onnxruntime::llm::kernels::weight_only::GemmPluginProfilerManager;
using onnxruntime::llm::kernels::weight_only::WeightOnlyGroupwiseQuantGemmPluginProfiler;
using GemmProfilerPtr = std::shared_ptr<WeightOnlyGroupwiseQuantGemmPluginProfiler>;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface>;

// Environment variable to enable/disable the fpA_intB path: unset/0/off to disable, other value to enable.
// This only affects nodes whose weights are NOT prepacked (see the constructor).
constexpr const char* kFpAIntBGemmOption = "ORT_FPA_INTB_GEMM";

constexpr int64_t kMatMulNBitsWeightNotPrepacked = 0;
constexpr int64_t kMatMulNBitsWeightPrepackedSm80 = 1;
constexpr int64_t kMatMulNBitsWeightPrepackedSm90 = 2;

// Session-option config keys. These are readable by BOTH the built-in CUDA EP and the CUDA plugin
// EP: every kernel is created via KernelRegistryManager::CreateKernel, which injects the
// session-level ConfigOptions, and the plugin CUDA EP wraps a CUDAExecutionProvider that reuses this
// same kernel. Each key overrides its ORT_* environment-variable equivalent (config wins).
//   ep.cuda.fpa_intb_gemm       <-> ORT_FPA_INTB_GEMM       (0/off, 1/on)
//   ep.cuda.fpa_intb_profile_m  <-> ORT_FPA_INTB_PROFILE_M  (initial profile M buckets)
constexpr const char* kConfigFpAIntBGemm = "ep.cuda.fpa_intb_gemm";
constexpr const char* kConfigFpAIntBProfileM = "ep.cuda.fpa_intb_profile_m";

// Resolves a setting from the session config first (per-session, EP-agnostic), then the environment
// variable, else empty. Session config wins so a model/session can override a process-wide env var.
inline std::string ResolveFpAIntBConfigOrEnv(const OpKernelInfo& info, const char* config_key,
                                             const char* env_key) {
  const auto from_config = info.GetConfigOptions().GetConfigEntry(config_key);
  if (from_config.has_value()) {
    return *from_config;
  }
  return ParseEnvironmentVariableWithDefault<std::string>(env_key, "");
}

// Parses the fpA_intB enable flag. "on" enables the full fpA_intB path (the CUTLASS GEMM and, where
// supported, the GEMV decode kernel; they share one weight layout and cannot be split). Accepts
// (case-insensitive): "" / "0" / "off" -> disabled; otherwise, enabled (a non-zero numeric value
// still enables, for backward compatibility).
inline bool ParseFpAIntBEnabled(const std::string& value) {
  const std::string lowered = onnxruntime::utils::GetLowercaseString(onnxruntime::utils::TrimString(value));
  if (lowered.empty() || lowered == "0" || lowered == "off") {
    return false;
  }
  return true;
}
#endif

template <typename T>
class MatMulNBits final : public CudaKernel {
 public:
  MatMulNBits(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("bits", &nbits_));

    constexpr int kInputIndexScale = 2;
    constexpr int kInputIndexZeroPoints = 3;
    constexpr int kInputIndexGroupIndex = 4;
    constexpr int kInputIndexBias = 5;

#ifdef BUILD_CUDA_EP_AS_PLUGIN
    // PLUGIN BUILD ADAPTATION: The adapter Node does not expose InputDefs(),
    // so we cannot check whether optional inputs (zero_points, g_idx, bias)
    // truly exist at construction time. Instead, we check input count here
    // and verify actual tensor presence in ComputeInternal.
    ORT_UNUSED_PARAMETER(kInputIndexScale);  // only used in non-plugin path for type checking
    has_zero_points_ = info.GetInputCount() > kInputIndexZeroPoints;
    has_g_idx_ = info.GetInputCount() > kInputIndexGroupIndex;
    has_bias_ = info.GetInputCount() > kInputIndexBias;
    // is_zero_points_scale_same_type_ defaults to false; checked at runtime in plugin path.
#else
    has_zero_points_ = info.GetInputCount() > kInputIndexZeroPoints && info.node().InputDefs()[kInputIndexZeroPoints]->Exists();
    has_g_idx_ = info.GetInputCount() > kInputIndexGroupIndex && info.node().InputDefs()[kInputIndexGroupIndex]->Exists();
    has_bias_ = info.GetInputCount() > kInputIndexBias && info.node().InputDefs()[kInputIndexBias]->Exists();

    if (has_zero_points_) {
      int32_t zero_point_type = info.node().InputDefs()[kInputIndexZeroPoints]->TypeAsProto()->tensor_type().elem_type();
      int32_t scale_type = info.node().InputDefs()[kInputIndexScale]->TypeAsProto()->tensor_type().elem_type();
      is_zero_points_scale_same_type_ = (zero_point_type == scale_type);
    }
#endif
    sm_ = this->GetDeviceProp().major * 10 + this->GetDeviceProp().minor;

    force_chunked_ = ParseEnvironmentVariableWithDefault<int>(kForceChunkedEnvVar, 0) != 0;
    chunk_target_rows_ = ParseEnvironmentVariableWithDefault<int64_t>(kChunkSizeEnvVar, kDefaultChunkTargetRows);
    if (chunk_target_rows_ < 1) {
      chunk_target_rows_ = kDefaultChunkTargetRows;
    }

#if USE_FPA_INTB_GEMM
    weight_prepacked_ = info.GetAttrOrDefault<int64_t>("weight_prepacked", kMatMulNBitsWeightNotPrepacked);
    ORT_ENFORCE(weight_prepacked_ == kMatMulNBitsWeightNotPrepacked ||
                    weight_prepacked_ == kMatMulNBitsWeightPrepackedSm80 ||
                    weight_prepacked_ == kMatMulNBitsWeightPrepackedSm90,
                "weight_prepacked must be 0 (not prepacked), 1 (SM80 layout), or 2 (SM90 layout), but got ",
                weight_prepacked_);
    if (weight_prepacked_ == kMatMulNBitsWeightPrepackedSm90) {
      // The native SM90 (Hopper TMA/WGMMA) mixed-GEMM kernel requires a compute-capability 9.0
      // device and a block_size that is a multiple of the Hopper K tile (128 / sizeof(half) = 64).
      // block_size=32 is only supported by the SM80/Ampere-class kernel + GEMV path.
      ORT_ENFORCE(sm_ == 90,
                  "weight_prepacked=2 (SM90 layout) requires a compute capability 9.0 (Hopper) device, but got sm ", sm_);
      ORT_ENFORCE(block_size_ == 64 || block_size_ == 128,
                  "weight_prepacked=2 (SM90 layout) supports block_size 64 or 128 only, but got ", block_size_);
    }

    if constexpr (std::is_same<T, MLFloat16>::value || std::is_same<T, BFloat16>::value) {
      const bool prepacked = weight_prepacked_ != kMatMulNBitsWeightNotPrepacked;
      // The enable flag (session config ep.cuda.fpa_intb_gemm, else ORT_FPA_INTB_GEMM env) only
      // chooses the path for weights that are NOT prepacked. A prepacked weight is already stored in
      // the fpA_intB layout, so the choice was made at export time and cannot be turned off here.
      const bool enable_fpa_intb =
          prepacked ||
          ParseFpAIntBEnabled(ResolveFpAIntBConfigOrEnv(info, kConfigFpAIntBGemm, kFpAIntBGemmOption));
      // Note: a fused bias (input[5]) is fully supported by the fpA_intB GEMV, CUTLASS SM80/SM90
      // GEMM (EpilogueOpBias), and the tactic profiler, so bias-bearing nodes (e.g. gpt-oss
      // qkv_proj/o_proj) are eligible. Only g_idx/reorder remains unsupported by this path.
      if (enable_fpa_intb &&
          (block_size_ == 32 || block_size_ == 64 || block_size_ == 128) &&
          (nbits_ == 4 || nbits_ == 8) &&
          !has_g_idx_ &&
          N_ % (nbits_ == 8 ? 32 : 64) == 0 &&
          K_ % block_size_ == 0 &&
          sm_ >= 75) {
        // The CUTLASS GEMM and the GEMV decode kernel consume the same fpA_intB weight layout, so
        // enable GEMV whenever it is supported; a node cannot mix fpA_intB and legacy layouts.
        using onnxruntime::llm::kernels::fpA_intB_gemv::KernelType;
        KernelType cuda_kernel_type;
        if constexpr (std::is_same<T, MLFloat16>::value) {
          cuda_kernel_type = (nbits_ == 8) ? KernelType::FP16Int8Groupwise : KernelType::FP16Int4Groupwise;
        } else if constexpr (std::is_same<T, BFloat16>::value) {
          cuda_kernel_type = (nbits_ == 8) ? KernelType::BF16Int8Groupwise : KernelType::BF16Int4Groupwise;
        }
        if (onnxruntime::llm::kernels::fpA_intB_gemv::is_supported(sm_, cuda_kernel_type)) {
          has_fpA_intB_gemv_ = true;
        }

        InitGemmProfiler(FpAIntBPackingSmForKernel());

        // Initial profile M buckets from session config (ep.cuda.fpa_intb_profile_m) with
        // ORT_FPA_INTB_PROFILE_M env fallback; empty -> profiler uses its default bucket set.
        std::vector<int> profile_m = WeightOnlyGroupwiseQuantGemmPluginProfiler::ParseProfileMList(
            ResolveFpAIntBConfigOrEnv(info, kConfigFpAIntBProfileM,
                                      onnxruntime::llm::kernels::weight_only::kEnvProfileM));
        gemmProfiler_->setProfileMOverride(profile_m);

        int max_m = profile_m.empty() ? onnxruntime::llm::kernels::weight_only::kDefaultProfileMaxM
                                      : profile_m.back();
        RunGemmProfile(has_fpA_intB_gemv_, 1, max_m);
        has_fpA_intB_gemm_ = true;
      }

      if (prepacked) {
        ORT_ENFORCE(has_fpA_intB_gemm_,
                    "weight_prepacked requires the fpA_intB path, but it is unsupported for this node "
                    "(check bits, block_size, N/K alignment, g_idx, and compute capability >= 7.5)");
        ORT_ENFORCE(weight_prepacked_ == RequiredWeightPrepackedFormat(),
                    "weight_prepacked=", weight_prepacked_, " does not match the format required by the selected fpA_intB kernel: ",
                    RequiredWeightPrepackedFormat());
      }
    } else {
      ORT_ENFORCE(weight_prepacked_ == kMatMulNBitsWeightNotPrepacked,
                  "weight_prepacked requires fp16 or bf16 input A so the CUDA fpA_intB path can consume input B");
    }

#ifndef NDEBUG
    printf("n=%d, k=%d, block_size=%d, bits=%d, zp_bits=%d, g_idx=%d, bias=%d, gemv=%d, gemm=%d\n",
           int(N_), int(K_), int(block_size_), int(nbits_),
           has_zero_points_ ? (is_zero_points_scale_same_type_ ? int(sizeof(T)) * 8 : int(nbits_)) : int(0),
           int(has_g_idx_ ? 1 : 0), int(has_bias_ ? 1 : 0),
           int(has_fpA_intB_gemv_), int(has_fpA_intB_gemm_));
#endif
#else
    const int64_t weight_prepacked = info.GetAttrOrDefault<int64_t>("weight_prepacked", static_cast<int64_t>(0));
    ORT_ENFORCE(weight_prepacked == 0,
                "weight_prepacked requires an ONNX Runtime build with onnxruntime_USE_FPA_INTB_GEMM=ON");
#endif
  }

  Status ComputeInternal(OpKernelContext* context) const override;
#if USE_FPA_INTB_GEMM
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;
#endif

 private:
#if USE_FPA_INTB_GEMM
  int FpAIntBPackingSmForKernel() const;
  int64_t RequiredWeightPrepackedFormat() const;

  void InitGemmProfiler(int sm);
  void RunGemmProfile(bool hasWeightOnlyCudaKernel, int min_m, int max_m);

  Status PrePack_B(const Tensor& tensor, AllocatorPtr alloc, cudaStream_t stream, bool& is_packed);
  Status PrePack_Scale(const Tensor& tensor, AllocatorPtr alloc, cudaStream_t stream);
  Status PrePack_ZeroPoint(const Tensor& tensor, AllocatorPtr alloc, cudaStream_t stream);
#endif

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
  bool force_chunked_{false};
  int64_t chunk_target_rows_{kDefaultChunkTargetRows};

#if USE_FPA_INTB_GEMM
  bool has_fpA_intB_gemv_{false};
  bool has_fpA_intB_gemm_{false};
  int64_t weight_prepacked_{kMatMulNBitsWeightNotPrepacked};

  bool is_prepacked_weight_{false};
  bool is_prepacked_scale_{false};
  bool is_prepacked_zero_point_{false};

  WeightOnlyGemmRunnerPtr weightOnlyGemmRunner_{nullptr};
  mutable GemmProfilerPtr gemmProfiler_{nullptr};
  GemmIdCore gemmId_{};

  IAllocatorUniquePtr<void> fpA_intB_weight_buffer_;
  IAllocatorUniquePtr<void> fpA_intB_scale_buffer_;
  IAllocatorUniquePtr<void> fpA_intB_zero_buffer_;
#endif
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
