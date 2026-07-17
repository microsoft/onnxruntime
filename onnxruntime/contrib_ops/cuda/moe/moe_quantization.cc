// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#include "contrib_ops/cuda/moe/moe_quantization.h"
#include <type_traits>
#include "core/common/float8.h"
#include "cutlass/numeric_types.h"
#include "core/common/safeint.h"
#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "contrib_ops/cuda/llm/common/env_utils.h"
#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_adaptor.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_util_kernels.h"

#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cpu/utils/debug_macros.h"

#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace {
void LogQMoESwigluFusionRemapOnce() {
  static std::once_flag log_warning;
  std::call_once(log_warning, []() {
    LOGS_DEFAULT(WARNING) << "QMoE swiglu_fusion is 0; assuming interleaved SwiGLU layout "
                             "for backward compatibility.";
  });
}

// Per-shape autotune of the fused FP4 GEMV CtaN/Threads tiling. It is opt-in because profiling
// synchronizes the inference stream; set ORT_FP4_GEMV_AUTOTUNE=1 to enable it.
// ORT_FP4_GEMV_AUTOTUNE_LOG=1 logs the chosen configs per shape.
bool Fp4GemvAutotuneEnabled() {
  return onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_AUTOTUNE", 0) == 1;
}

bool Fp4GemvAutotuneLogEnabled() {
  return onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_AUTOTUNE_LOG", 0) == 1;
}

const char* QMoEGemvConfigName(onnxruntime::llm::kernels::moe_gemv::MoeGemvConfig config) {
  using onnxruntime::llm::kernels::moe_gemv::MoeGemvConfig;
  if (config == MoeGemvConfig::kCtaN16) {
    return "ctan16";
  }
  if (config == MoeGemvConfig::kThreads64) {
    return "threads64";
  }
  return "default";
}

bool TryGetStaticDim(const ONNX_NAMESPACE::TensorShapeProto* shape, int dim_index, int64_t& value) {
  if (shape == nullptr || dim_index >= shape->dim_size() || !shape->dim(dim_index).has_dim_value()) {
    return false;
  }
  value = shape->dim(dim_index).dim_value();
  return value > 0;
}

bool StaticFp4CutlassShapeSupported(const OpKernelInfo& op_kernel_info, bool is_swiglu) {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  ORT_UNUSED_PARAMETER(op_kernel_info);
  ORT_UNUSED_PARAMETER(is_swiglu);
  return false;
#else
  const auto& input_defs = op_kernel_info.node().InputDefs();
  if (input_defs.size() <= 5 || input_defs[2] == nullptr || input_defs[5] == nullptr) {
    return false;
  }

  const auto* fc1_shape = input_defs[2]->Shape();
  const auto* fc2_shape = input_defs[5]->Shape();
  if (fc1_shape == nullptr || fc2_shape == nullptr || fc1_shape->dim_size() != 3 || fc2_shape->dim_size() != 3) {
    return false;
  }

  int64_t fc1_k = 0;
  int64_t fc1_packed_n = 0;
  int64_t fc2_k = 0;
  int64_t fc2_packed_n = 0;
  if (!TryGetStaticDim(fc1_shape, 1, fc1_k) || !TryGetStaticDim(fc1_shape, 2, fc1_packed_n) ||
      !TryGetStaticDim(fc2_shape, 1, fc2_k) || !TryGetStaticDim(fc2_shape, 2, fc2_packed_n)) {
    return false;
  }

  const int64_t hidden_size = fc1_k;
  const int64_t hidden_size_from_fc2 = fc2_packed_n * 2;
  const int64_t fc1_n = fc1_packed_n * 2;
  if (hidden_size != hidden_size_from_fc2 || (is_swiglu && fc1_n % 2 != 0)) {
    return false;
  }

  const int64_t inter_size = fc2_k;
  const int64_t inter_size_from_fc1 = is_swiglu ? fc1_n / 2 : fc1_n;
  if (inter_size != inter_size_from_fc1) {
    return false;
  }

  // The SM90 WFP4A16 block-scaled grouped GEMM packs block scales in groups of 8 k-blocks
  // (PackedScalesNum = CTA_K(256) / group_size(32)); the GEMM reads ceil(K/256) of those
  // packed groups. PrePackFp4ScalesForTmaWs zero-pads each expert's k-block count up to a
  // multiple of 8 (and the runner strides experts by the padded count), so K no longer has
  // to be 256-aligned: the partial last group's tail blocks are zero and are only read by
  // the GEMM's last CTA-K-tile, where the matching A/B K elements are TMA-zeroed. The
  // binding requirements are now the MXFP4 block-scale vector size (k % 32 == 0 for an
  // integral k-block count) and the weight/activation TMA alignment. Gate on 64 as a
  // conservative margin above 32; this admits gpt-oss-20b (hidden=inter=2880) while leaving
  // headroom for any untested N-dim/CTA tiling alignment, and can be lowered to 32 once
  // broader shapes are validated.
  constexpr int64_t kMxfp4CutlassAlignment = 64;
  return hidden_size % kMxfp4CutlassAlignment == 0 && inter_size % kMxfp4CutlassAlignment == 0;
#endif
}
}  // namespace

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      QMoE,                                                                    \
      kMSDomain,                                                               \
      1,                                                                       \
      T,                                                                       \
      kCudaExecutionProvider,                                                  \
      (*KernelDefBuilder::Create())                                            \
          .MayInplace(0, 0)                                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .TypeConstraint("T1", {DataTypeImpl::GetTensorType<uint8_t>(),       \
                                 DataTypeImpl::GetTensorType<Float8E4M3FN>()}) \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<T>(),             \
                                 DataTypeImpl::GetTensorType<Float8E8M0>(),    \
                                 DataTypeImpl::GetTensorType<Float8E4M3FN>()}) \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<float>()),         \
      QMoE);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

QMoE::QMoE(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info), MoEBase(op_kernel_info, GetDeviceProp()) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4,
              "expert_weight_bits must be 4 or 8, but got ", expert_weight_bits_);

  block_size_ = op_kernel_info.GetAttrOrDefault<int64_t>("block_size", -1);
  this->quant_type_ = op_kernel_info.GetAttrOrDefault<std::string>("quant_type", "int");
  ORT_ENFORCE(quant_type_ == "int" || quant_type_ == "fp4" || quant_type_ == "nvfp4" ||
                  quant_type_ == "fp8" || quant_type_ == "wfp4afp8",
              "quant_type must be 'int', 'fp4', 'nvfp4', 'fp8', or 'wfp4afp8', but got '", quant_type_, "'");
  if (quant_type_ == "nvfp4") {
    constexpr int64_t kNvfp4BlockSize = 16;
    ORT_ENFORCE(block_size_ == -1 || block_size_ == kNvfp4BlockSize,
                "QMoE quant_type='nvfp4' requires block_size=", kNvfp4BlockSize,
                ", but got ", block_size_);
    block_size_ = kNvfp4BlockSize;
  } else if (quant_type_ == "fp4" || quant_type_ == "wfp4afp8") {
    constexpr int64_t kMxfp4BlockSize = 32;
    ORT_ENFORCE(block_size_ == -1 || block_size_ == kMxfp4BlockSize,
                "QMoE quant_type='", quant_type_, "' requires block_size=", kMxfp4BlockSize,
                ", but got ", block_size_);
    block_size_ = kMxfp4BlockSize;
  }
  // ``weights_prepacked`` is an optional tri-state attribute (default -1) that
  // declares the layout of the int4/int8 fc1/fc2 weight initializers. The
  // concrete prepacked layouts selected by -1 and 1 are determined by the
  // execution provider. The CUDA EP maps the tri-state as:
  //   -1 (default): already prepacked in the EP's default int weight layout.
  //    1: already prepacked in an alternate EP-selected int weight layout.
  //    0: raw [E, N, K/pack] initializers; the PrePack hook lays them out.
  //
  // Important: the CUDA QMoE int4/int8 MoE GEMM always dispatches to the
  // Ampere (SM80) grouped-GEMM kernel -- even on SM90 -- because mixed
  // int-weight + fp16/bf16 activation is not a valid Hopper TMA warp-specialized
  // specialisation (see isValidHopperMOESpecialisation). The kernel therefore
  // consumes the SM80/Ampere CUTLASS fpA_intB layout on every GPU. As a result
  // the EP default (-1) is the SM80 layout regardless of the runtime device SM,
  // and SM80-format weights are valid on SM90 (they run via the SM80 kernel).
  // For CUDA today, -1 and 1 are equivalent (both SM80 layout), and 1 is
  // reserved for a possible future Hopper-specific layout.
  // PrePack (weights_prepacked=0) packs for the SM80 layout accordingly.
  const int64_t weights_prepacked_mode =
      op_kernel_info.GetAttrOrDefault<int64_t>("weights_prepacked", static_cast<int64_t>(-1));
  ORT_ENFORCE(weights_prepacked_mode == -1 || weights_prepacked_mode == 0 || weights_prepacked_mode == 1,
              "weights_prepacked must be -1 (default), 0, or 1, but got ", weights_prepacked_mode);
  weights_prepacked_ = (weights_prepacked_mode != 0);
#if !defined(ENABLE_FP4) || !defined(USE_FP4_QMOE)
  ORT_ENFORCE(quant_type_ != "fp4", "QMoE quant_type='fp4' requires USE_FP4_QMOE with CUDA 12.8 or newer.");
  ORT_ENFORCE(quant_type_ != "nvfp4", "QMoE quant_type='nvfp4' requires USE_FP4_QMOE with CUDA 12.8 or newer.");
  ORT_ENFORCE(quant_type_ != "wfp4afp8",
              "QMoE quant_type='wfp4afp8' requires USE_FP4_QMOE with CUDA 12.8 or newer.");
#endif
#if !defined(ENABLE_FP8) || !defined(USE_FP8_QMOE)
  ORT_ENFORCE(quant_type_ != "fp8", "QMoE quant_type='fp8' requires USE_FP8_QMOE with CUDA 11.8 or newer.");
  ORT_ENFORCE(quant_type_ != "wfp4afp8", "QMoE quant_type='wfp4afp8' requires USE_FP8_QMOE with CUDA 11.8 or newer.");
#endif

  using namespace onnxruntime::llm::kernels::cutlass_kernels;

#ifdef BUILD_CUDA_EP_AS_PLUGIN
  auto input_type = op_kernel_info.GetKernelInfo().GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  bool is_fp16 = input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
#else
  int32_t input_type = op_kernel_info.node().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  bool is_fp16 = input_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16;
#endif
  is_fp16_ = is_fp16;

  if (quant_type_ == "fp4" || quant_type_ == "nvfp4" || quant_type_ == "fp8" || quant_type_ == "wfp4afp8") {
    if (quant_type_ == "fp4") {
      ORT_ENFORCE(expert_weight_bits_ == 4, "FP4 quantization requires expert_weight_bits=4");
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
      use_fp4_dequant_fallback_ = sm_ < 120;
      const bool requested_fp4_cutlass_gemm =
          onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_ENABLE_FP4_CUTLASS_GEMM", 0) == 1;
      const bool allow_unsafe_fp4_cutlass_gemm =
          onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_ENABLE_FP4_CUTLASS_UNSAFE", 0) == 1;
      const bool fp4_cutlass_shape_supported = StaticFp4CutlassShapeSupported(
          op_kernel_info, activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu);
      enable_fp4_cutlass_gemm_ = is_fp16_ && sm_ == 90 && requested_fp4_cutlass_gemm &&
                                 allow_unsafe_fp4_cutlass_gemm && fp4_cutlass_shape_supported;
      if (requested_fp4_cutlass_gemm && is_fp16_ && sm_ == 90 && !allow_unsafe_fp4_cutlass_gemm) {
        LOGS_DEFAULT(WARNING) << "QMoE native FP4 CUTLASS GEMM was requested, but it is disabled by default "
                                 "while its SM90 MXFP4 conversion/layout path is still being validated; "
                                 "falling back to the dequant/GEMV path. Set ORT_ENABLE_FP4_CUTLASS_UNSAFE=1 "
                                 "only for debugging the experimental native path.";
      } else if (requested_fp4_cutlass_gemm && is_fp16_ && sm_ == 90 && !fp4_cutlass_shape_supported) {
        LOGS_DEFAULT(WARNING) << "QMoE native FP4 CUTLASS GEMM was requested, but the static FP4 weight "
                                 "shape does not meet the current WFP4A16 alignment requirements; falling "
                                 "back to the dequant/GEMV path.";
      }
      if (enable_fp4_cutlass_gemm_) {
        use_fp4_dequant_fallback_ = false;
      }
      // Fused MXFP4 GEMV (W4A16) decode path for the SM<120 fallback regime. This is the
      // default: on real decode shapes it is ~18x faster than re-dequantizing all experts to
      // dense BF16/FP16 every token, and it is validated bit-exact against the fallback. Set
      // ORT_ENABLE_FP4_GEMV=0 to force the dequant fallback (e.g. for debugging). Prefill and
      // any unsupported shape still fall through to the dequant path at dispatch time.
      if (use_fp4_dequant_fallback_) {
        const char* v = std::getenv("ORT_ENABLE_FP4_GEMV");
        enable_fp4_gemv_ = (v == nullptr || v[0] == '\0' || v[0] != '0');
      } else if (enable_fp4_cutlass_gemm_) {
        // Native CUTLASS WFP4A16 scales well with M but is underfilled for decode (small M):
        // route prefill (M >= ORT_FP4_PREFILL_MIN_TOKENS) through native, and small-decode
        // shapes through the fused GEMV. Both weight/scale layouts are pre-packed alongside
        // each other so a single QMoE node can serve both regimes.
        enable_fp4_gemv_ = true;
        fp4_prefill_min_tokens_ = onnxruntime::ParseEnvironmentVariableWithDefault<int64_t>(
            "ORT_FP4_PREFILL_MIN_TOKENS", 64);
        // Above this average per-expert token count, prefill routes to the dense A16 fallback
        // runner instead of the native FP4 grouped GEMM (measured crossover ~128 tokens/expert
        // on H200). 0 disables the upper bound. See ComputeInternal dispatch.
        fp4_native_max_tokens_per_expert_ = onnxruntime::ParseEnvironmentVariableWithDefault<int64_t>(
            "ORT_FP4_NATIVE_MAX_TOKENS_PER_EXPERT", 128);
      }
      // SM80 FP4 grouped GEMM (port of the INT4 fused-dequant Ampere path to MXFP4).
      // Only meaningful on Ampere through pre-Blackwell in the dequant-fallback regime
      // (80 <= sm_ < 120, e.g. H200), where the native SM90 TMA FP4 path is the slow prefill path.
      // This SM80 grouped GEMM is several times faster at the gpt-oss-20b prefill regime, so it is enabled by DEFAULT for FP16/BF16;
      // set ORT_FP4_SM80_GEMM=0 to fall back to the dequant path.
      // If the user EXPLICITLY
      // requested the native CUTLASS GEMM (ORT_ENABLE_FP4_CUTLASS_GEMM=1) we honor that intent
      // and do not take the SM80 path — this keeps the kernel-side moeUseSm80Fp4() (which reads
      // the same two env vars) in lock-step with this decision in every regime, including the
      // native-requested-but-shape-unsupported fallback (which then uses the dequant path).
      // When enabled we force the GEMV prepack (which also produces the SM80 CUTLASS-interleaved
      // e2m1 weights + activation-dtype group scales) and later override the runner to the FP4 runner so
      // prefill can dispatch to the SM80 DqMma grouped GEMM (see moeUseSm80Fp4 in the kernels).
      enable_fp4_sm80_gemm_ =
          use_fp4_dequant_fallback_ && sm_ >= 80 && sm_ < 120 && !requested_fp4_cutlass_gemm &&
          onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_SM80_GEMM", 1) == 1;
      if (enable_fp4_sm80_gemm_) {
        enable_fp4_gemv_ = true;
      }
      // Capture the fused MXFP4 GEMV autotune knobs once here (at op-construction time) instead of
      // re-reading the environment on every inference call, so a session's autotune behavior cannot
      // change underneath it if the process mutates the environment after construction (mirrors the
      // ORT_FP4_SM80_GEMM constructor-plumbed decision below).
      enable_fp4_gemv_autotune_ = Fp4GemvAutotuneEnabled();
      enable_fp4_gemv_autotune_log_ = Fp4GemvAutotuneLogEnabled();
#else
      use_fp4_dequant_fallback_ = true;
#endif
    } else if (quant_type_ == "nvfp4") {
      ORT_ENFORCE(expert_weight_bits_ == 4, "NVFP4 quantization requires expert_weight_bits=4");
      // Native block-scaled CUTLASS GEMM for NVFP4 is Blackwell-only. On all currently
      // supported GPUs (including SM90/H200) NVFP4 always uses the dequant-to-A16 fallback:
      // dequantize E2M1 weights (E4M3 block scales, block size 16, per-expert global scale)
      // to FP16/BF16 and run the dense A16 MoE runner.
      use_fp4_dequant_fallback_ = true;
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
      // Fused FP4 GEMV (W4A16) decode fast path, shared with MXFP4 but with block size 16 and
      // Float8E4M3FN block scales. Default-on (opt-out via ORT_ENABLE_FP4_GEMV=0): small-decode
      // shapes route through the fused GEMV instead of re-dequantizing every expert to dense
      // BF16/FP16 each token, and any unsupported shape (prefill / large batch) falls through to
      // the dequant fallback. NVFP4 has no native CUTLASS/SM80 path, so those stay disabled.
      {
        const char* v = std::getenv("ORT_ENABLE_FP4_GEMV");
        enable_fp4_gemv_ = (v == nullptr || v[0] == '\0' || v[0] != '0');
      }
      enable_fp4_gemv_autotune_ = Fp4GemvAutotuneEnabled();
      enable_fp4_gemv_autotune_log_ = Fp4GemvAutotuneLogEnabled();
#endif
    } else if (quant_type_ == "wfp4afp8") {
      ORT_ENFORCE(expert_weight_bits_ == 4, "WFP4AFP8 (W4A8) quantization requires expert_weight_bits=4");
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE) && defined(ENABLE_FP8)
      // The native FP8 x MXFP4 path uses CUTLASS block-scaled tensor ops which require SM100+ (Blackwell).
      // The activation BF16/FP16 -> FP8 quantization is performed inside the runner's
      // expandInputRowsKernel using the MXFP8 branch: the runner is constructed with T=__nv_fp8_e4m3,
      // InputType=half/bf16, and the QuantParams sets mxfp8_mxfp4.fc{1,2}.weight_block_scale to the MXFP4
      // weight block scales. Activation block scales are written to fc1_fp4_act_scale_ at runtime.
      // On older GPUs we fall back to dequantizing MXFP4 weights to BF16/FP16 and using the A16 runner.
      use_wfp4afp8_dequant_fallback_ = sm_ < 100;
#else
      use_wfp4afp8_dequant_fallback_ = true;
#endif
    } else {
      ORT_ENFORCE(expert_weight_bits_ == 8, "FP8 quantization requires expert_weight_bits=8");
      // Use native W8A16-FP8 on SM90+ (Hopper/H200), fallback to dequant on older GPUs
      if (sm_ >= 90) {
        use_fp8_dequant_fallback_ = false;
      } else {
        use_fp8_dequant_fallback_ = true;
      }
    }
    if (quant_type_ == "fp4" && (!use_fp4_dequant_fallback_ || enable_fp4_sm80_gemm_)) {
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
      if (is_fp16) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, __nv_fp4_e2m1, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_fp4_e2m1, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
      // Dense A16 fallback runner for the large-per-expert-M prefill regime, where dequantizing
      // the MXFP4 weights to FP16/BF16 and running the dense grouped GEMM beats the native FP4
      // grouped GEMM. Constructed alongside the FP4 runner so a single QMoE node can route per
      // call (see fp4_native_max_tokens_per_expert_). Not used by the SM80 grouped-GEMM path
      // (enable_fp4_sm80_gemm_), which routes prefill through the FP4 runner directly.
      if (is_fp16) {
        m_fp4_dense_fallback_runner_ = std::make_unique<CutlassMoeFCRunner<half, half, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {
        m_fp4_dense_fallback_runner_ = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
      // Capture the SM80-FP4 routing decision (made above from the environment at op-construction
      // time) into the runner, so inference-time config/tactic selection does not re-read the
      // environment (which may have changed since the session was created, e.g. in unit tests).
      m_moe_runner->setUseSm80Fp4(enable_fp4_sm80_gemm_);
#endif
    } else if (quant_type_ == "wfp4afp8" && !use_wfp4afp8_dequant_fallback_) {
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE) && defined(ENABLE_FP8) && defined(USE_FP8_QMOE)
      // Native W4A8: FP8 e4m3 activations + MXFP4 weights, BF16/FP16 input/output.
      // Template parameters: <T=fp8, WeightType=fp4, OutputType=BF16/FP16, InputType=BF16/FP16>.
      // CUTLASS routes this through the SM100+ block-scaled tensor op path. The runner accepts
      // BF16/FP16 input from the caller and quantizes it to FP8 inside expandInputRowsKernel
      // (MXFP8 branch, triggered by mxfp8_mxfp4.fc{1,2}.weight_block_scale being non-null).
      if (is_fp16) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
#endif
    } else if (quant_type_ == "fp8" && !use_fp8_dequant_fallback_) {
#if defined(ENABLE_FP8) && defined(USE_FP8_QMOE)
      // Native W8A16-FP8: activations are half/bf16, weights are __nv_fp8_e4m3
      if (is_fp16) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, __nv_fp8_e4m3, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
#endif
    } else {
      // FP4/NVFP4/WFP4AFP8 dequant fallback or FP8 dequant fallback: use A16 runner
      if (is_fp16) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, half, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {  // BFloat16
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
    }
  } else {
    // Integer quantization (INT4/INT8)
    if (is_fp16) {
      if (expert_weight_bits_ == 4) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, cutlass::uint4b_t, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {  // expert_weight_bits_ == 8
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, uint8_t, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
    }
#if !defined(ORT_QUICK_BUILD) && defined(ENABLE_BF16)
    else {  // BFloat16
      if (expert_weight_bits_ == 4) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {  // expert_weight_bits_ == 8
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, uint8_t, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
    }
#endif
  }  // end integer quantization

  ORT_ENFORCE(m_moe_runner != nullptr,
              "QMoE: failed to construct MoE runner for quant_type='", quant_type_,
              "', expert_weight_bits=", expert_weight_bits_,
              ", input_type=", (is_fp16 ? "float16" : "bfloat16"),
              ". Build configuration may be missing the corresponding kernel.");
}

Status QMoE::ComputeInternal(OpKernelContext* context) const {
  const bool is_fp4 = (quant_type_ == "fp4");
  const bool is_nvfp4 = (quant_type_ == "nvfp4");
  // NVFP4 shares the E2M1 4-bit weight format, [E,K,N/2] weight layout, and the per-expert
  // global-scale (15/16) + block-scale (3/6) inputs with MXFP4 "fp4"; it differs only in the
  // block-scale decode (Float8E4M3FN vs Float8E8M0) and block size (16 vs 32). Treat both as an
  // fp4-family type wherever weight-scale handling is shared.
  const bool is_fp4_family = is_fp4 || is_nvfp4;
  const bool is_fp8 = (quant_type_ == "fp8");
  const bool is_wfp4afp8 = (quant_type_ == "wfp4afp8");
  const bool is_int = (quant_type_ == "int");
  // Modes that consume FP4 weight block scales (inputs 3/6) and per-expert global weight scales.
  const bool uses_fp4_weight_scales = is_fp4_family || is_wfp4afp8;
  // Modes that consume per-expert FP-format global weight scales (inputs 15/16).
  const bool uses_global_weight_scales = is_fp4_family || is_fp8 || is_wfp4afp8;
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  // When PrePack consumed the int4/int8 expert-weight initializers
  // (``weights_prepacked == false`` opt-in path), the original tensors
  // were freed; ``context->Input<Tensor>(2)/(5)`` would return nothing.
  // Mirror how ``MatMulNBits`` reads its prepacked B input.
  // Gate on *both* prepacked buffers being present. If only fc1 were prepacked
  // (e.g. a partial prepack from an earlier failure or a future refactor), this
  // path must not null out fc2_experts_weights and feed a null fc2 weight/shape
  // to the runner.
  const bool int_weights_consumed_by_prepack =
      is_int && !weights_prepacked_ && packed_fc1_weights_ != nullptr && packed_fc2_weights_ != nullptr;
  // When ``weights_prepacked == 0`` the raw ``[E, N, K/pack]`` int weights must be
  // converted to the CUTLASS fpA_intB layout by PrePack before the runner can consume
  // them. If PrePack never ran (e.g. ``session.disable_prepacking`` is set), the prepack
  // buffers stay null and falling through to the raw initializer pointers would feed
  // non-CUTLASS bytes to the runner, producing silently wrong output. Fail loudly instead.
  if (is_int && !weights_prepacked_ &&
      (packed_fc1_weights_ == nullptr || packed_fc2_weights_ == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "QMoE weights_prepacked=0 requires PrePack to run, but the int weight "
                           "buffers were not produced (is session.disable_prepacking set?). Provide "
                           "CUTLASS-prepacked weights with weights_prepacked=1, or enable prepacking.");
  }
  const Tensor* fc1_experts_weights = int_weights_consumed_by_prepack ? nullptr : context->Input<Tensor>(2);
  const Tensor* fc1_scales = (is_int && !packed_fc1_scales_) ? context->Input<Tensor>(3) : nullptr;
  const Tensor* fc1_experts_bias_optional = context->Input<Tensor>(4);
  const Tensor* fc2_experts_weights = int_weights_consumed_by_prepack ? nullptr : context->Input<Tensor>(5);
  const Tensor* fc2_scales = (is_int && !packed_fc2_scales_) ? context->Input<Tensor>(6) : nullptr;
  const Tensor* fc2_experts_bias_optional = context->Input<Tensor>(7);
  // The CUTLASS MoE runner has no separate FC3 GEMM — gate and up projection weights must be
  // pre-concatenated into fc1 with doubled output dimension.
  ORT_ENFORCE(context->Input<Tensor>(8) == nullptr,
              "QMoE in CUDA execution provider does not support separate fc3_experts_weights. "
              "Gate and up projection weights must be pre-concatenated into fc1.");

  // Backward compatibility: the published gpt-oss-20b model (and any model exported by ORT < 1.27)
  // hard-coded the interleaved SwiGLU fusion layout and did not emit a swiglu_fusion attribute, so it
  // falls back to the default of 0 ("not fused"). QMoE never has a separate FC3 (enforced above), so a
  // SwiGLU activation with swiglu_fusion == 0 means the gate and value projections are actually pre-fused
  // into FC1 (interleaved layout). Treat this as swiglu_fusion == 1 so those legacy models keep working.
  int swiglu_fusion = swiglu_fusion_;
  if (activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu &&
      swiglu_fusion == 0) {
    swiglu_fusion = 1;
    LogQMoESwigluFusionRemapOnce();
  }

  const Tensor* fc1_zeros = packed_fc1_bias_ ? nullptr : context->Input<Tensor>(11);
  const Tensor* fc2_zeros = packed_fc2_bias_ ? nullptr : context->Input<Tensor>(12);

  auto check_weight_type = [](const Tensor* tensor, const char* name, bool expect_fp8) -> Status {
    ORT_RETURN_IF_NOT(tensor != nullptr, "Input '", name, "' is required.");
    if (expect_fp8) {
      ORT_RETURN_IF_NOT(tensor->IsDataType<Float8E4M3FN>(), name, " must be a float8e4m3fn tensor when quant_type='fp8'.");
    } else {
      ORT_RETURN_IF_NOT(tensor->IsDataType<uint8_t>(), name, " must be a uint8 tensor when quant_type is 'int' or 'fp4'.");
    }
    return Status::OK();
  };

  // When PrePack consumed the int weight initializers, the dtype check
  // is no longer applicable (we know they were uint8 — that's what
  // PrePackIntExpertWeights validated and consumed).
  if (!int_weights_consumed_by_prepack) {
    ORT_RETURN_IF_ERROR(check_weight_type(fc1_experts_weights, "fc1_experts_weights", is_fp8));
    ORT_RETURN_IF_ERROR(check_weight_type(fc2_experts_weights, "fc2_experts_weights", is_fp8));
  }

  // Unified FP4 inputs: block scales in fc*_scales (3/6), global scales in 15/16.
  // FP4 scales are copied to GPU during PrePack but intentionally remain live inputs. The raw
  // tensors provide the dtype and shape metadata required to validate packed buffers before any
  // dequant or GEMV kernel indexes them.
  const Tensor* fp4_fc1_block_scales = uses_fp4_weight_scales ? context->Input<Tensor>(3) : nullptr;
  const Tensor* fp4_fc2_block_scales = uses_fp4_weight_scales ? context->Input<Tensor>(6) : nullptr;
  const Tensor* fc1_global_scale = uses_global_weight_scales ? context->Input<Tensor>(15) : nullptr;
  const Tensor* fc2_global_scale = uses_global_weight_scales ? context->Input<Tensor>(16) : nullptr;

  // W4A8 (WFP4AFP8) optional Variant A activation scales (per-tensor or per-expert FP8 global act scale).
  const Tensor* fc1_act_scale = (is_wfp4afp8 && !packed_fc1_act_scale_) ? context->Input<Tensor>(17) : nullptr;
  const Tensor* fc2_act_scale = (is_wfp4afp8 && !packed_fc2_act_scale_) ? context->Input<Tensor>(18) : nullptr;

  const bool has_any_zero_point = (fc1_zeros != nullptr || fc2_zeros != nullptr ||
                                   packed_fc1_bias_ != nullptr || packed_fc2_bias_ != nullptr);

  // Row-wise quantization path does not support asymmetric zero-points in QMoE.
  // QuantParams::Int only carries scales (no zero/bias tensor).
  if (block_size_ <= 0 && has_any_zero_point) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "QMoE row-wise quantization (block_size <= 0) does not support zero_points. "
                           "Remove fc*_zero_points or use block-wise quantization.");
  }
  if (block_size_ > 0 && block_size_ < 32 && has_any_zero_point) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "QMoE asymmetric zero_points are currently supported only when block_size >= 32. "
                           "Use block_size >= 32 or remove fc*_zero_points.");
  }

  int64_t pack_size = expert_weight_bits_ == 4 ? 2 : 1;
  bool is_fused_swiglu = activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu;
  MoEParameters moe_params;
  // Prefer the cached shapes when PrePack consumed the source initializer.
  const TensorShape& fc1_shape = int_weights_consumed_by_prepack ? fc1_weights_shape_ : fc1_experts_weights->Shape();
  const TensorShape& fc2_shape = int_weights_consumed_by_prepack ? fc2_weights_shape_ : fc2_experts_weights->Shape();
  ORT_RETURN_IF_ERROR(onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs, &fc1_shape,
      fc1_experts_bias_optional, fc1_scales, fc1_zeros,
      &fc2_shape, fc2_experts_bias_optional, fc2_scales, fc2_zeros,
      nullptr, nullptr, nullptr, nullptr,
      pack_size, is_fused_swiglu, block_size_));
  ORT_RETURN_IF_NOT(k_ > 0 && k_ <= moe_params.num_experts,
                    "QMoE requires 0 < k <= num_experts, got k=", k_,
                    " and num_experts=", moe_params.num_experts);

  // The INT4/INT8 weight-only path stores B in the column-interleaved layout (ColumnMajorTileInterleave),
  // whose CUTLASS pitchlinear iterators require the GEMM reduction dim K to be a whole multiple of the
  // interleave tile (kInterleaveKTile == 64 for fp16/bf16 activations). For the two MoE GEMMs the
  // reduction dims are fc1.K == hidden_size and fc2.K == inter_size. A partial final K tile is read past
  // the valid range and silently yields garbage/NaN (the single-matrix fpA_intB GEMM throws on this; the
  // grouped MoE GEMM and the decode GEMV have no such guard), so reject it up front with a clear error.
  if (quant_type_ == "int") {
    constexpr int64_t kInterleaveKTile = 64;
    ORT_RETURN_IF_NOT(moe_params.hidden_size % kInterleaveKTile == 0,
                      "QMoE int weight-only quantization requires hidden_size to be a multiple of ",
                      kInterleaveKTile, " (the interleaved-weight K tile), got hidden_size=", moe_params.hidden_size, ".");
    ORT_RETURN_IF_NOT(moe_params.inter_size % kInterleaveKTile == 0,
                      "QMoE int weight-only quantization requires inter_size to be a multiple of ",
                      kInterleaveKTile, " (the interleaved-weight K tile), got inter_size=", moe_params.inter_size, ".");
  }

  if (uses_fp4_weight_scales) {
    // MXFP4 ("fp4"/"wfp4afp8") uses block size 32 with Float8E8M0 block scales; NVFP4 ("nvfp4")
    // uses block size 16 with Float8E4M3FN block scales. Both are consumed as raw uint8 bytes.
    const int64_t fp4_block_size = is_nvfp4 ? 16 : 32;
    const char* fp4_mode_label = is_nvfp4 ? "quant_type='nvfp4'" : "quant_type='fp4'/'wfp4afp8'";
    const char* fp4_scale_label = is_nvfp4 ? "NVFP4 block scales" : "MXFP4 block scales";
    ORT_RETURN_IF_NOT(moe_params.hidden_size % fp4_block_size == 0,
                      "QMoE ", fp4_mode_label, " requires hidden_size to be a multiple of ",
                      fp4_block_size, " for ", fp4_scale_label, ", got hidden_size=", moe_params.hidden_size, ".");
    ORT_RETURN_IF_NOT(moe_params.inter_size % fp4_block_size == 0,
                      "QMoE ", fp4_mode_label, " requires inter_size to be a multiple of ",
                      fp4_block_size, " for ", fp4_scale_label, ", got inter_size=", moe_params.inter_size, ".");
    const int64_t fc1_out_size = is_fused_swiglu ? moe_params.inter_size * 2 : moe_params.inter_size;
    auto check_fp4_block_scale = [is_nvfp4](const Tensor* tensor, const char* name, int64_t num_experts,
                                            int64_t n, int64_t k) -> Status {
      ORT_RETURN_IF_NOT(tensor != nullptr, "QMoE quant_type='fp4'/'nvfp4'/'wfp4afp8' requires ", name, ".");
      if (is_nvfp4) {
        ORT_RETURN_IF_NOT(tensor->IsDataType<Float8E4M3FN>(), name, " must be a float8e4m3fn NVFP4 block-scale tensor.");
      } else {
        ORT_RETURN_IF_NOT(tensor->IsDataType<Float8E8M0>(), name, " must be a float8e8m0 MXFP block-scale tensor.");
      }
      const auto& dims = tensor->Shape().GetDims();
      ORT_RETURN_IF_NOT(dims.size() == 3 && dims[0] == num_experts && dims[1] == n && dims[2] == k,
                        name, " must have shape (", num_experts, ", ", n, ", ", k, "), got ", tensor->Shape().ToString(), ".");
      return Status::OK();
    };
    auto check_global_scale = [](const Tensor* tensor, const char* name, int64_t num_experts, const char* quant_type) -> Status {
      ORT_RETURN_IF_NOT(tensor != nullptr, "QMoE quant_type='", quant_type, "' requires ", name, ".");
      ORT_RETURN_IF_NOT(tensor->IsDataType<float>(), name, " must be a float tensor.");
      const auto& dims = tensor->Shape().GetDims();
      ORT_RETURN_IF_NOT(dims.size() == 1 && dims[0] == num_experts,
                        name, " must have shape (", num_experts, "), got ", tensor->Shape().ToString(), ".");
      return Status::OK();
    };

    if (fp4_fc1_block_scales) {
      ORT_RETURN_IF_ERROR(check_fp4_block_scale(fp4_fc1_block_scales, "fc1_scales", moe_params.num_experts,
                                                fc1_out_size, moe_params.hidden_size / fp4_block_size));
    }
    if (fp4_fc2_block_scales) {
      ORT_RETURN_IF_ERROR(check_fp4_block_scale(fp4_fc2_block_scales, "fc2_scales", moe_params.num_experts,
                                                moe_params.hidden_size, moe_params.inter_size / fp4_block_size));
    }
    if (fc1_global_scale) {
      ORT_RETURN_IF_ERROR(check_global_scale(fc1_global_scale, "fc1_global_scale", moe_params.num_experts, quant_type_.c_str()));
    }
    if (fc2_global_scale) {
      ORT_RETURN_IF_ERROR(check_global_scale(fc2_global_scale, "fc2_global_scale", moe_params.num_experts, quant_type_.c_str()));
    }
  }

  if (is_wfp4afp8) {
    auto check_act_scale = [](const Tensor* tensor, const char* name, int64_t num_experts) -> Status {
      ORT_RETURN_IF_NOT(tensor != nullptr, "QMoE quant_type='wfp4afp8' Variant A requires ", name, ".");
      ORT_RETURN_IF_NOT(tensor->IsDataType<float>(), name, " must be a float tensor.");
      const auto& dims = tensor->Shape().GetDims();
      ORT_RETURN_IF_NOT(dims.size() == 1 && (dims[0] == 1 || dims[0] == num_experts),
                        name, " must have shape (1,) or (", num_experts, "), got ", tensor->Shape().ToString(), ".");
      return Status::OK();
    };
    // fc*_act_scale are optional; when absent the runner uses the MXFP8 block-scaled Variant B.
    if (fc1_act_scale) {
      ORT_RETURN_IF_ERROR(check_act_scale(fc1_act_scale, "fc1_act_scale", moe_params.num_experts));
    }
    if (fc2_act_scale) {
      ORT_RETURN_IF_ERROR(check_act_scale(fc2_act_scale, "fc2_act_scale", moe_params.num_experts));
    }
  }

  if (is_fp8) {
    auto check_global_scale = [](const Tensor* tensor, const char* name, int64_t num_experts) -> Status {
      ORT_RETURN_IF_NOT(tensor != nullptr, "QMoE quant_type='fp8' requires ", name, ".");
      ORT_RETURN_IF_NOT(tensor->IsDataType<float>(), name, " must be a float tensor.");
      const auto& dims = tensor->Shape().GetDims();
      ORT_RETURN_IF_NOT(dims.size() == 1 && dims[0] == num_experts,
                        name, " must have shape (", num_experts, "), got ", tensor->Shape().ToString(), ".");
      return Status::OK();
    };
    if (fc1_global_scale) {
      ORT_RETURN_IF_ERROR(check_global_scale(fc1_global_scale, "fc1_global_scale", moe_params.num_experts));
    }
    if (fc2_global_scale) {
      ORT_RETURN_IF_ERROR(check_global_scale(fc2_global_scale, "fc2_global_scale", moe_params.num_experts));
    }
  }

  // Validate minimum dimensions for CUTLASS kernels.
  // SM >= 90 TMA WarpSpecialized: smallest tile is 128x16x128B (N=16 for FP16). K < tile_K handled by TMA.
  // SM < 90 Ampere GemmGrouped: smallest instantiated tile N=128, but CUTLASS predicates N < tile_N.
  // On SM90 with mixed-type (INT4/INT8), the Ampere fallback is used — same predication applies.
  // Alignment of dimensions to 128 bits is enforced separately in moe_kernels.cu.
  {
    constexpr int min_dim = 16;
    ORT_RETURN_IF(moe_params.hidden_size < min_dim,
                  "QMoE CUDA kernel requires hidden_size >= ", min_dim,
                  " for SM", sm_, ", got ", moe_params.hidden_size);
    ORT_RETURN_IF(moe_params.inter_size < min_dim,
                  "QMoE CUDA kernel requires inter_size >= ", min_dim,
                  " for SM", sm_, ", got ", moe_params.inter_size);
  }

  bool use_awq = (fc1_zeros != nullptr) || (packed_fc1_bias_ != nullptr);
  onnxruntime::llm::kernels::cutlass_kernels::MOEParallelismConfig parallelism_config{};

  // Per-call native-vs-dense routing for the FP4 prefill regime. The native FP4 grouped GEMM wins
  // when each expert processes few tokens, but the dense A16 grouped GEMM (MXFP4 weights
  // dequantized to FP16/BF16) is faster once the average per-expert token count is large
  // (compute-bound; measured crossover ~128 tokens/expert on H200). When native FP4 is enabled and
  // the average exceeds fp4_native_max_tokens_per_expert_, route this call through the dense
  // fallback runner instead. avg_tokens_per_expert = num_rows * top_k / num_experts.
  const bool fp4_native_available = is_fp4 && !use_fp4_dequant_fallback_ && m_fp4_dense_fallback_runner_ != nullptr;
  const int64_t avg_tokens_per_expert =
      moe_params.num_experts > 0
          ? (static_cast<int64_t>(moe_params.num_rows) * static_cast<int64_t>(k_)) /
                static_cast<int64_t>(moe_params.num_experts)
          : static_cast<int64_t>(moe_params.num_rows) * static_cast<int64_t>(k_);
  const bool route_native_fp4 =
      fp4_native_available &&
      (fp4_native_max_tokens_per_expert_ <= 0 || avg_tokens_per_expert <= fp4_native_max_tokens_per_expert_);
  // SM80 FP4 grouped-GEMM prefill path. Active only when ORT_FP4_SM80_GEMM=1 and the
  // GEMV prepack produced the SM80 CUTLASS-interleaved e2m1 weights + activation-dtype group scales. Decode
  // shapes are still served by the fused GEMV (which returns early below); everything that
  // falls through to the runner here (prefill / GEMV-unsupported shapes) runs on the FP4
  // runner via the Ampere/SM80 DqMma grouped GEMM with QuantParams::GroupWise(32, ...).
  const bool fp4_sm80_prefill =
      is_fp4 && enable_fp4_sm80_gemm_ &&
      gemv_fp4_fc1_weights_ != nullptr && gemv_fp4_fc2_weights_ != nullptr &&
      gemv_fp4_fc1_scales_ != nullptr && gemv_fp4_fc2_scales_ != nullptr;
  // The dense fallback runner is selected only when native FP4 is available but this call exceeds
  // the per-expert threshold. In every other configuration (native chosen, or native unavailable so
  // m_moe_runner is itself the dense A16 runner) we use m_moe_runner.
  onnxruntime::llm::kernels::cutlass_kernels::CutlassMoeFCRunnerInterface* active_runner =
      (fp4_native_available && !route_native_fp4) ? m_fp4_dense_fallback_runner_.get() : m_moe_runner.get();

  // Profile and capture the best tactics under the profiler mutex, then release the mutex so
  // that scratch allocation, weight dequantization, scale prepping, softmax, and other
  // CPU-bound work can proceed concurrently across QMoE inferences. The mutex is reacquired
  // around setTactic + runMoe because they mutate shared `m_moe_runner` state.
  std::optional<onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler::Config> config1;
  std::optional<onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler::Config> config2;
  size_t workspace_size = 0;
  {
    std::lock_guard<std::mutex> profiler_lock(mGemmProfilerMutex);

    // Profiling launches grouped-GEMM kernels, records/synchronizes CUDA events, and
    // allocates/frees scratch from the temp allocator on the compute stream. All of these are
    // illegal while that stream is being captured into a CUDA graph; performing them corrupts the
    // capture and later surfaces as an illegal memory access (CUDA 700) reported at a downstream
    // MoE kernel launch (e.g. moe_kernels.cu cudaFuncSetAttribute). During capture we therefore
    // skip profiling and reuse a config cached from an earlier non-capturing run, falling back to
    // the default tactic when nothing is cached.
    cudaStream_t compute_stream = Stream(context);
    // compute_stream == nullptr is the legacy default stream (stream 0), which can itself be
    // under CUDA-graph capture; cudaStreamIsCapturing handles it, so query it directly rather
    // than assuming a null stream is never capturing.
    const bool stream_is_capturing = onnxruntime::llm::common::isCapturing(compute_stream);

    // Use profiler with proper weight type for quantized weights
    if (onnxruntime::llm::common::getEnvForceDeterministicMOE()) {
      auto tactics = active_runner->getTactics();
      if (!tactics.empty()) {
        config1 = tactics[0];
        config2 = tactics[0];
        active_runner->setTactic(config1, config2);
      }
    } else {
      AllocatorPtr allocator;
      ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
      mGemmProfiler.setAllocator(std::move(allocator));
      mGemmProfiler.setProfilerParams(static_cast<int>(moe_params.num_experts), static_cast<int>(k_),
                                      static_cast<int64_t>(moe_params.hidden_size), static_cast<int64_t>(moe_params.inter_size),
                                      fp4_sm80_prefill ? int64_t{32} : static_cast<int64_t>(block_size_), activation_type_,
                                      false, true, parallelism_config, sm_);

      onnxruntime::llm::nvinfer::DataType dtype = is_fp16_ ? onnxruntime::llm::nvinfer::DataType::kHALF : onnxruntime::llm::nvinfer::DataType::kBF16;
      if (is_wfp4afp8 && !use_wfp4afp8_dequant_fallback_) {
        dtype = onnxruntime::llm::nvinfer::DataType::kFP8;
      }
      // Weight type: FP4 for MXFP4, INT4 for 4-bit integer, INT8 for 8-bit integer. When the
      // native FP4 path routes to the dense fallback for this call, profile the dense (A16) tactic.
      onnxruntime::llm::nvinfer::DataType wtype;
      if (is_nvfp4) {
        // NVFP4 always uses the dequant fallback, so profile against the dense (A16) tactic.
        wtype = dtype;
      } else if (is_fp4) {
        // fp4_sm80_prefill runs the e2m1 weights through the SM80 fused-dequant grouped GEMM,
        // whose scratch + groupwise scale layout match INT4-groupwise (4-bit weight + activation-dtype
        // scales). Profile against kINT4 so the workspace is sized for the groupwise path
        // rather than the FP4 TMA block-scale layout; the launched kernel is still the e2m1
        // kernel (templated on the runner's WeightType), and the profiler's per-tactic
        // try/catch discards tile shapes that fail can_implement.
        wtype = fp4_sm80_prefill ? onnxruntime::llm::nvinfer::DataType::kINT4
                                 : (route_native_fp4 ? onnxruntime::llm::nvinfer::DataType::kFP4 : dtype);
      } else if (is_wfp4afp8) {
        // Native W4A8 path uses FP8 activation + FP4 weights through the block-scaled dispatch.
        // Profile against the FP4 weight tactic; fall back to dense dtype when the dequant path is selected.
        wtype = use_wfp4afp8_dequant_fallback_ ? dtype : onnxruntime::llm::nvinfer::DataType::kFP4;
      } else if (is_fp8) {
        wtype = use_fp8_dequant_fallback_ ? dtype : onnxruntime::llm::nvinfer::DataType::kFP8;
      } else {
        wtype = (expert_weight_bits_ == 4) ? onnxruntime::llm::nvinfer::DataType::kINT4
                                           : onnxruntime::llm::nvinfer::DataType::kINT8;
      }

      using onnxruntime::llm::kernels::cutlass_kernels::MoeGemmId;
      using onnxruntime::llm::kernels::weight_only::GemmDims;

      // For gated activations (SwiGLU), fc1_out_size is doubled
      int64_t fc1_out_size = static_cast<int64_t>(moe_params.inter_size);
      if (is_fused_swiglu) {
        fc1_out_size = static_cast<int64_t>(moe_params.inter_size) * 2;
      }

      // GEMM 1: N=fc1_out_size (doubled for gated), K=hidden_size
      MoeGemmId id1(static_cast<int>(fc1_out_size), static_cast<int>(moe_params.hidden_size), dtype, wtype, MoeGemmId::GemmType::Gemm1);
      if (!stream_is_capturing) {
        // profileTactics caches per (GemmId, M bucket); calling it every forward lets decode
        // (small M) and prefill (large M) each profile and select their own best tile shape.
        GemmDims dims(static_cast<int64_t>(moe_params.num_rows), static_cast<int64_t>(moe_params.num_rows),
                      fc1_out_size, static_cast<int64_t>(moe_params.hidden_size));
        mGemmProfiler.profileTactics(active_runner, dims, id1, compute_stream);
      }
      config1 = mGemmProfiler.getBestConfig(static_cast<int>(moe_params.num_rows), id1);

      // GEMM 2
      MoeGemmId id2(static_cast<int>(moe_params.hidden_size), static_cast<int>(moe_params.inter_size), dtype, wtype, MoeGemmId::GemmType::Gemm2);
      if (!stream_is_capturing) {
        GemmDims dims(static_cast<int64_t>(moe_params.num_rows), static_cast<int64_t>(moe_params.num_rows),
                      static_cast<int64_t>(moe_params.hidden_size), static_cast<int64_t>(moe_params.inter_size));
        mGemmProfiler.profileTactics(active_runner, dims, id2, compute_stream);
      }
      config2 = mGemmProfiler.getBestConfig(static_cast<int>(moe_params.num_rows), id2);

      // Capture-safe fallback: if profiling was skipped (graph capture) and no tuned config was
      // cached from a prior non-capturing run, use the runner's default tactic instead of leaving
      // the config unset.
      if (!config1 || !config2) {
        auto tactics = active_runner->getTactics();
        if (!tactics.empty()) {
          if (!config1) {
            config1 = tactics[0];
          }
          if (!config2) {
            config2 = tactics[0];
          }
        }
      }

      active_runner->setTactic(config1, config2);
    }

    workspace_size = active_runner->getWorkspaceSize(
        moe_params.num_rows, moe_params.hidden_size, moe_params.inter_size, moe_params.num_experts, k_,
        activation_type_, parallelism_config, use_awq);
  }
  // Lock released — concurrent QMoE inferences can now run prep work in parallel.

  // Scratch buffer for workspace + expert_scales + expert_indices + permutation_map.
  // Use checked arithmetic: these byte counts derive adjacent pointer offsets inside one allocation.
  // expert_scales: num_rows * k * sizeof(float)
  // expert_indices: num_rows * k * sizeof(int)
  size_t expanded_rows = SafeInt<size_t>(moe_params.num_rows) * SafeInt<size_t>(k_);
  size_t scales_bytes = expanded_rows * sizeof(float);
  size_t indices_bytes = expanded_rows * sizeof(int);
  size_t permutation_bytes = expanded_rows * sizeof(int);
  size_t total_scratch_bytes = SafeInt<size_t>(workspace_size) + scales_bytes + indices_bytes + permutation_bytes;

  auto work_space = GetScratchBuffer<void>(total_scratch_bytes, GetComputeStream(context));
  char* workspace_ptr = reinterpret_cast<char*>(work_space.get());
  float* expert_scales = reinterpret_cast<float*>(workspace_ptr + workspace_size);
  int* expert_indices = reinterpret_cast<int*>(workspace_ptr + workspace_size + scales_bytes);
  int* unpermuted_row_to_permuted_row = reinterpret_cast<int*>(workspace_ptr + workspace_size + scales_bytes + indices_bytes);

  cudaStream_t stream = Stream(context);

  // Perform Softmax + TopK
  // Input router_probs is (num_rows, num_experts)
  bool is_fp16 = input->IsDataType<MLFloat16>();
  bool is_bf16 = input->IsDataType<BFloat16>();
  if (is_fp16) {
    LaunchSoftmaxTopK(
        reinterpret_cast<const half*>(router_probs->DataRaw()),
        expert_scales,
        expert_indices,
        static_cast<int>(moe_params.num_rows),
        static_cast<int>(moe_params.num_experts),
        static_cast<int>(k_),
        normalize_routing_weights_,
        stream);
  } else if (is_bf16) {
    LaunchSoftmaxTopK(
        reinterpret_cast<const __nv_bfloat16*>(router_probs->DataRaw()),
        expert_scales,
        expert_indices,
        static_cast<int>(moe_params.num_rows),
        static_cast<int>(moe_params.num_experts),
        static_cast<int>(k_),
        normalize_routing_weights_,
        stream);
  } else {
    // Fallback for float
    LaunchSoftmaxTopK(
        reinterpret_cast<const float*>(router_probs->DataRaw()),
        expert_scales,
        expert_indices,
        static_cast<int>(moe_params.num_rows),
        static_cast<int>(moe_params.num_experts),
        static_cast<int>(k_),
        normalize_routing_weights_,
        stream);
  }

  // Holders for packed tensors (if packing is needed for SwiGLU)
  IAllocatorUniquePtr<void> packed_fc1_scales_holder;
  IAllocatorUniquePtr<void> packed_fc1_zp_holder;
  IAllocatorUniquePtr<void> transposed_fc1_scales_holder;
  IAllocatorUniquePtr<void> transposed_fc2_scales_holder;
  IAllocatorUniquePtr<void> transposed_fc1_zp_holder;
  IAllocatorUniquePtr<void> transposed_fc2_zp_holder;

  // Determine effective pointers for scales and zero points
  const void* p_fc1_scales = nullptr;
  const void* p_fc1_zp = nullptr;
  const void* p_fc2_scales = nullptr;
  const void* p_fc2_zp = nullptr;

  // Use pre-packed buffers if available, otherwise use input tensors (and potentially compute bias on the fly)
  IAllocatorUniquePtr<void> transient_fc1_bias;
  IAllocatorUniquePtr<void> transient_fc2_bias;

  auto prepare_scale_zp = [&](const Tensor* scales, const Tensor* zeros,
                              const IAllocatorUniquePtr<void>& packed_scale, const IAllocatorUniquePtr<void>& packed_bias,
                              IAllocatorUniquePtr<void>& transposed_scale_holder,
                              IAllocatorUniquePtr<void>& transposed_zp_holder,
                              IAllocatorUniquePtr<void>& transient_bias,
                              const void*& eff_scale, const void*& eff_zp) {
    if (packed_scale) {
      eff_scale = packed_scale.get();
    } else if (scales) {
      eff_scale = scales->DataRaw();

      // For block-wise quantization, Cutlass expects scales laid out as [Experts, Blocks, N].
      // Input tensors are provided as [Experts, N, Blocks], so transpose when PrePack is not used.
      auto scale_shape = scales->Shape();
      if (block_size_ > 0 && scale_shape.NumDimensions() == 3 && scale_shape[2] > 1) {
        size_t rows = scale_shape[1];   // N
        size_t cols = scale_shape[2];   // Blocks
        size_t batch = scale_shape[0];  // Experts
        size_t bytes = scales->SizeInBytes();

        transposed_scale_holder = GetScratchBuffer<void>(bytes, GetComputeStream(context));
        eff_scale = transposed_scale_holder.get();

        if (scales->IsDataType<MLFloat16>()) {
          LaunchQMoETranspose2D(static_cast<const half*>(scales->DataRaw()), static_cast<half*>(transposed_scale_holder.get()), batch, rows, cols, stream);
        } else if (scales->IsDataType<BFloat16>()) {
          LaunchQMoETranspose2D(static_cast<const __nv_bfloat16*>(scales->DataRaw()), static_cast<__nv_bfloat16*>(transposed_scale_holder.get()), batch, rows, cols, stream);
        } else {
          LaunchQMoETranspose2D(static_cast<const float*>(scales->DataRaw()), static_cast<float*>(transposed_scale_holder.get()), batch, rows, cols, stream);
        }
      }
    }

    if (packed_bias) {
      eff_zp = packed_bias.get();
    } else if (zeros) {
      if (expert_weight_bits_ == 4 || (expert_weight_bits_ == 8 && block_size_ > 0)) {
        // Compute bias on the fly: bias = -zp * scale
        // We need 'eff_scale' to be available.
        if (eff_scale && block_size_ > 0) {
          size_t num_elements = zeros->Shape().Size();
          // Determine type size based on scale type
          bool is_fp16 = scales->IsDataType<MLFloat16>();
          bool is_bf16 = scales->IsDataType<BFloat16>();
          size_t bytes = num_elements * (is_fp16 ? 2 : 4);

          transient_bias = GetScratchBuffer<void>(bytes, GetComputeStream(context));
          eff_zp = transient_bias.get();

          const uint8_t* p_zp = static_cast<const uint8_t*>(zeros->DataRaw());

          // Determine whether zeros are stored packed (two uint4 ZP per byte) or unpacked.
          // For block-wise 4-bit quantization, scales have shape [E, N, K_blocks] and zeros
          // have shape either [E, N, K_blocks] (unpacked) or [E, N, ceil(K_blocks/2)] (packed).
          // Compare the last dim of zeros vs scales explicitly instead of relying on a fragile
          // numeric heuristic on Shape().Size() ratios, which can mis-classify pathological
          // shapes (e.g., K_blocks=1 where ceil(1/2)=1 makes packed indistinguishable from
          // unpacked by element count alone).
          bool zp_is_packed_4bit = false;
          if (expert_weight_bits_ == 4) {
            const auto& zeros_shape = zeros->Shape();
            const auto& scales_shape = scales->Shape();
            ORT_ENFORCE(zeros_shape.NumDimensions() == 3 && scales_shape.NumDimensions() == 3,
                        "Block-wise 4-bit zeros and scales must be 3D, got zeros=",
                        zeros_shape.ToString(), ", scales=", scales_shape.ToString());
            ORT_ENFORCE(zeros_shape[0] == scales_shape[0] && zeros_shape[1] == scales_shape[1],
                        "Block-wise 4-bit zeros and scales must agree on the first two dims, got zeros=",
                        zeros_shape.ToString(), ", scales=", scales_shape.ToString());
            const int64_t scales_k = scales_shape[2];
            const int64_t zeros_k = zeros_shape[2];
            const int64_t expected_packed_k = (scales_k + 1) / 2;
            if (zeros_k == scales_k) {
              zp_is_packed_4bit = false;
            } else if (zeros_k == expected_packed_k) {
              zp_is_packed_4bit = true;
            } else {
              ORT_THROW("Block-wise 4-bit zeros last dim must be ", scales_k,
                        " (unpacked) or ", expected_packed_k, " (packed). Got zeros=",
                        zeros_shape.ToString(), ", scales=", scales_shape.ToString());
            }
          }

          // Transpose ZP if needed (for 3D ZP)
          auto shape = zeros->Shape();
          IAllocatorUniquePtr<void> temp_zp_transposed;
          if (shape.NumDimensions() == 3 && shape[2] > 1) {
            size_t rows = shape[1];   // N
            size_t cols = shape[2];   // Blocks
            size_t batch = shape[0];  // Experts
            size_t zp_bytes = zeros->SizeInBytes();
            temp_zp_transposed = GetScratchBuffer<void>(zp_bytes, GetComputeStream(context));
            LaunchQMoETranspose2D(p_zp, static_cast<uint8_t*>(temp_zp_transposed.get()), batch, rows, cols, stream);
            p_zp = static_cast<const uint8_t*>(temp_zp_transposed.get());
          }

          if (is_fp16) {
            if (expert_weight_bits_ == 8) {
              LaunchQMoEPrePackOffsetBias(
                  p_zp,
                  static_cast<const half*>(eff_scale),
                  static_cast<half*>(transient_bias.get()),
                  static_cast<int>(num_elements),
                  128.0f,
                  stream);
            } else if (zp_is_packed_4bit) {
              size_t scale_el = scales->Shape().Size();
              int N_stride = static_cast<int>(zeros->Shape()[1]);
              LaunchQMoEPrePackPacked4BitZPKernel(
                  p_zp,
                  static_cast<const half*>(eff_scale),
                  static_cast<half*>(transient_bias.get()),
                  static_cast<int>(scale_el),
                  N_stride,
                  stream);
            } else {
              LaunchQMoEPrePackZP(
                  p_zp,
                  static_cast<const half*>(eff_scale),
                  static_cast<half*>(transient_bias.get()),
                  static_cast<int>(num_elements),
                  stream);
            }
          } else if (is_bf16) {
            if (expert_weight_bits_ == 8) {
              LaunchQMoEPrePackOffsetBias(
                  p_zp,
                  static_cast<const __nv_bfloat16*>(eff_scale),
                  static_cast<__nv_bfloat16*>(transient_bias.get()),
                  static_cast<int>(num_elements),
                  128.0f,
                  stream);
            } else if (zp_is_packed_4bit) {
              size_t scale_el = scales->Shape().Size();
              int N_stride = static_cast<int>(zeros->Shape()[1]);
              LaunchQMoEPrePackPacked4BitZPKernel(
                  p_zp,
                  static_cast<const __nv_bfloat16*>(eff_scale),
                  static_cast<__nv_bfloat16*>(transient_bias.get()),
                  static_cast<int>(scale_el),
                  N_stride,
                  stream);
            } else {
              LaunchQMoEPrePackZP(
                  p_zp,
                  static_cast<const __nv_bfloat16*>(eff_scale),
                  static_cast<__nv_bfloat16*>(transient_bias.get()),
                  static_cast<int>(num_elements),
                  stream);
            }
          } else {
            if (expert_weight_bits_ == 8) {
              LaunchQMoEPrePackOffsetBias(
                  p_zp,
                  static_cast<const float*>(eff_scale),
                  static_cast<float*>(transient_bias.get()),
                  static_cast<int>(num_elements),
                  128.0f,
                  stream);
            } else if (zp_is_packed_4bit) {
              size_t scale_el = scales->Shape().Size();
              int N_stride = static_cast<int>(zeros->Shape()[1]);
              LaunchQMoEPrePackPacked4BitZPKernel(
                  p_zp,
                  static_cast<const float*>(eff_scale),
                  static_cast<float*>(transient_bias.get()),
                  static_cast<int>(scale_el),
                  N_stride,
                  stream);
            } else {
              LaunchQMoEPrePackZP(
                  p_zp,
                  static_cast<const float*>(eff_scale),
                  static_cast<float*>(transient_bias.get()),
                  static_cast<int>(num_elements),
                  stream);
            }
          }
        }
      } else {
        // For 8-bit, ZP is used as is (or transposed).
        // Since we are not packing, we use the raw pointer unless transpose is needed.
        // Transpose on the fly is tricky without allocation. BUT, ComputeInternal is usually called
        // with pre-packed weights/scales if coming from unit tests or offline tools.
        // If not pre-packed (e.g. dynamic graph), we might need to transpose if 3D.
        // For now, assuming standard path or 1D ZP for 2D weights.
        // If 3D, we must transpose.
        auto shape = zeros->Shape();
        if (shape.NumDimensions() == 3 && shape[2] > 1) {
          // Need temporary buffer for transpose
          size_t bytes = zeros->SizeInBytes();
          transposed_zp_holder = GetScratchBuffer<void>(bytes, GetComputeStream(context));
          eff_zp = transposed_zp_holder.get();

          size_t rows = shape[1];   // N
          size_t cols = shape[2];   // Blocks
          size_t batch = shape[0];  // Experts
          LaunchQMoETranspose2D(static_cast<const uint8_t*>(zeros->DataRaw()), static_cast<uint8_t*>(transposed_zp_holder.get()), batch, rows, cols, stream);
        } else {
          eff_zp = zeros->DataRaw();
        }
      }
    }
  };

  prepare_scale_zp(fc1_scales, fc1_zeros, packed_fc1_scales_, packed_fc1_bias_,
                   transposed_fc1_scales_holder, transposed_fc1_zp_holder, transient_fc1_bias, p_fc1_scales, p_fc1_zp);
  prepare_scale_zp(fc2_scales, fc2_zeros, packed_fc2_scales_, packed_fc2_bias_,
                   transposed_fc2_scales_holder, transposed_fc2_zp_holder, transient_fc2_bias, p_fc2_scales, p_fc2_zp);

  onnxruntime::llm::kernels::cutlass_kernels::QuantParams quant_params;
  if (is_fp4 && fp4_sm80_prefill) {
    // SM80 FP4 grouped-GEMM prefill: the e2m1 weights are in the SM80 CUTLASS interleaved
    // layout and the scales are the activation-dtype group scales (group=32, [E, K/32, N]) the GEMV path
    // already built (gemv_fp4_fc*_scales_ = e8m0_block_scale * global_scale). Feed them through
    // the generic fine-grained groupwise path — identical plumbing to INT4 block-wise QMoE.
    quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::GroupWise(
        /*group_size=*/32,
        gemv_fp4_fc1_scales_.get(),
        gemv_fp4_fc2_scales_.get(),
        nullptr,
        nullptr,
        nullptr,
        nullptr);
  } else if (is_fp4) {
    // FP4 quantization: use QuantParams::FP4 with block scales and global scales
    const void* p_fc1_block_scales = packed_fp4_fc1_block_scales_ ? packed_fp4_fc1_block_scales_.get()
                                                                  : (fp4_fc1_block_scales ? fp4_fc1_block_scales->DataRaw() : nullptr);
    const void* p_fc1_global_scale = packed_fc1_global_scale_ ? packed_fc1_global_scale_.get()
                                                              : (fc1_global_scale ? fc1_global_scale->DataRaw() : nullptr);
    const void* p_fc2_block_scales = packed_fp4_fc2_block_scales_ ? packed_fp4_fc2_block_scales_.get()
                                                                  : (fp4_fc2_block_scales ? fp4_fc2_block_scales->DataRaw() : nullptr);
    const void* p_fc2_global_scale = packed_fc2_global_scale_ ? packed_fc2_global_scale_.get()
                                                              : (fc2_global_scale ? fc2_global_scale->DataRaw() : nullptr);
    ORT_RETURN_IF_NOT(p_fc1_block_scales && p_fc1_global_scale && p_fc2_block_scales && p_fc2_global_scale,
                      "QMoE quant_type='fp4' requires fc1_scales, fc2_scales, fc1_global_scale, and fc2_global_scale.");
    // Only the native FP4 runner consumes block scales via quant_params; the dense fallback runner
    // receives weights already dequantized to FP16/BF16 and leaves quant_params empty.
    if (route_native_fp4) {
      using MXFPXElementSF = onnxruntime::llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF;
      quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::MXFP8MXFP4(
          static_cast<const MXFPXElementSF*>(p_fc1_block_scales),
          static_cast<const float*>(p_fc1_global_scale),
          static_cast<const MXFPXElementSF*>(p_fc2_block_scales),
          static_cast<const float*>(p_fc2_global_scale));
    }
  } else if (is_wfp4afp8) {
    // W4A8 (WFP4AFP8): MXFP4 weights + FP8 e4m3 activations.
    //   - Weight block scales (uint8 MXFPX) are read from fc*_scales (inputs 3/6)
    //   - Per-expert weight global scales come from inputs 15/16
    //   - Optional per-expert/per-tensor FP8 activation global scales come from inputs 18/19
    const void* p_fc1_block_scales = packed_fp4_fc1_block_scales_ ? packed_fp4_fc1_block_scales_.get()
                                                                  : (fp4_fc1_block_scales ? fp4_fc1_block_scales->DataRaw() : nullptr);
    const void* p_fc1_global_scale = packed_fc1_global_scale_ ? packed_fc1_global_scale_.get()
                                                              : (fc1_global_scale ? fc1_global_scale->DataRaw() : nullptr);
    const void* p_fc2_block_scales = packed_fp4_fc2_block_scales_ ? packed_fp4_fc2_block_scales_.get()
                                                                  : (fp4_fc2_block_scales ? fp4_fc2_block_scales->DataRaw() : nullptr);
    const void* p_fc2_global_scale = packed_fc2_global_scale_ ? packed_fc2_global_scale_.get()
                                                              : (fc2_global_scale ? fc2_global_scale->DataRaw() : nullptr);
    ORT_RETURN_IF_NOT(p_fc1_block_scales && p_fc1_global_scale && p_fc2_block_scales && p_fc2_global_scale,
                      "QMoE quant_type='wfp4afp8' requires fc1_scales, fc2_scales, fc1_global_scale, and fc2_global_scale.");
    if (!use_wfp4afp8_dequant_fallback_) {
      // Native W4A8 path (SM100+): use QuantParams::MXFP8MXFP4 (Variant B). The activation
      // is quantized BF16/FP16 -> MXFP8 (FP8 + per-block ue8m0 scales) inside the runner's
      // expandInputRowsKernel; the activation block scales are written to fc1_fp4_act_scale_
      // at runtime. The mxfp8_mxfp4 weight_block_scale field holds the MXFP4 weight block
      // scales (same uint8 ue8m0 element type as MXFP8 activation block scales) and is
      // checked by the expansion kernel as a marker to take the MXFP8 quantization path.
      //
      // Variant A (global-scaled FP8 activation) would consume the per-expert/per-tensor
      // scale from inputs 18/19 via QuantParams::FP8MXFP4. That path requires the user to
      // feed FP8 input directly, which the QMoE op does not support (its input is BF16/FP16),
      // so we use Variant B instead. The act_scale inputs are still validated and pre-packed
      // for forward compatibility.
      using MXFPXElementSF = onnxruntime::llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF;
      quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::MXFP8MXFP4(
          static_cast<const MXFPXElementSF*>(p_fc1_block_scales),
          static_cast<const float*>(p_fc1_global_scale),
          static_cast<const MXFPXElementSF*>(p_fc2_block_scales),
          static_cast<const float*>(p_fc2_global_scale));
    }
  } else if (is_fp8 && !use_fp8_dequant_fallback_) {
    // Native W8A16-FP8: per-expert global scale applied via alpha_scale_ptr_array in the epilogue.
    const void* p_fc1_global_scale = packed_fc1_global_scale_ ? packed_fc1_global_scale_.get()
                                                              : (fc1_global_scale ? fc1_global_scale->DataRaw() : nullptr);
    const void* p_fc2_global_scale = packed_fc2_global_scale_ ? packed_fc2_global_scale_.get()
                                                              : (fc2_global_scale ? fc2_global_scale->DataRaw() : nullptr);
    ORT_RETURN_IF_NOT(p_fc1_global_scale && p_fc2_global_scale,
                      "QMoE native W8A16-FP8 requires fc1_global_scale and fc2_global_scale.");
    quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::FP8(
        static_cast<const float*>(p_fc1_global_scale),  // dequant_fc1 = per-expert weight global scale
        nullptr,                                        // quant_fc2 (not used for W8A16)
        static_cast<const float*>(p_fc2_global_scale),  // dequant_fc2 = per-expert weight global scale
        nullptr,                                        // quant_final
        nullptr,                                        // dequant_input
        false);                                         // fc2_use_per_expert_act_scale
  } else if (block_size_ > 0) {
    quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::GroupWise(
        block_size_,
        p_fc1_scales,
        p_fc2_scales,
        nullptr,
        nullptr,
        p_fc1_zp,
        p_fc2_zp);
  } else {
    // Per-column quantization
    quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::Int(
        p_fc1_scales,
        p_fc2_scales);
  }

  Tensor* output = context->Output(0, input->Shape());

  // ---------------------------------------------------------------------------
  // Fused MXFP4 GEMV (W4A16) decode fast path. Default-on (opt-out via ORT_ENABLE_FP4_GEMV=0)
  // on the SM<120 fallback regime. Instead of dequantizing every active expert's MXFP4 weights
  // to dense BF16/FP16 HBM, route small-decode shapes through a standalone fused pipeline:
  //   build expert maps -> expand permuted activations -> fc1 SwiGLU GEMV -> fc2 GEMV ->
  //   finalize routing. The pre-packed [E,n,k/2] weights and [E,k/32,n] scales are produced
  //   by PrePack/TryBuildGemvFp4Scales. Unsupported shapes (prefill / large batch / missing
  //   pre-pack buffers) fall through to the dequant fallback below.
  // MXFP4 uses a fixed group size of 32 (intrinsic to the e2m1 + e8m0 block format); the
  // block_size_ attribute is unset (-1) for fp4, so it is not part of the gate.
  // When native CUTLASS WFP4A16 is enabled, GEMV serves only the decode regime
  // (num_rows < fp4_prefill_min_tokens_); prefill (M >= threshold) falls through to native.
  const bool fp4_decode_regime =
      use_fp4_dequant_fallback_ ||
      (enable_fp4_cutlass_gemm_ && fp4_prefill_min_tokens_ > 0 &&
       moe_params.num_rows < fp4_prefill_min_tokens_);
  const bool fp4_gemv_buffers_ready =
      (enable_fp4_sm80_gemm_
           ? (gemv_fp4_fc1_weights_decode_ != nullptr && gemv_fp4_fc2_weights_decode_ != nullptr)
           : (gemv_fp4_fc1_weights_ != nullptr && gemv_fp4_fc2_weights_ != nullptr)) &&
      gemv_fp4_fc1_scales_ != nullptr && gemv_fp4_fc2_scales_ != nullptr;
  if (is_fp4_family && fp4_decode_regime && enable_fp4_gemv_ && is_fused_swiglu &&
      fp4_gemv_buffers_ready) {
    namespace gemv = onnxruntime::llm::kernels::moe_gemv;
    namespace ck = onnxruntime::llm::kernels::cutlass_kernels;
    const int num_experts = static_cast<int>(moe_params.num_experts);
    const int64_t num_rows = moe_params.num_rows;
    const int64_t expanded = num_rows * static_cast<int64_t>(k_);
    const int64_t hidden = moe_params.hidden_size;
    const int64_t inter = moe_params.inter_size;
    const int64_t fc1_n = inter * 2;
    // MXFP4 uses block size 32 (e2m1 + e8m0); NVFP4 uses block size 16 (e2m1 + e4m3).
    const int gemv_group_size = is_nvfp4 ? 16 : 32;
    const bool fc1_gemv_supported = gemv::is_moe_gemv_fp4_supported(sm_, expanded, fc1_n, hidden, gemv_group_size);
    const bool fc2_gemv_supported = gemv::is_moe_gemv_fp4_supported(sm_, expanded, hidden, inter, gemv_group_size);
    if (num_rows > 0 && num_rows <= 256 && expanded > 0 && fc1_gemv_supported && fc2_gemv_supported) {
      // When the SM80 grouped-GEMM prefill repurposes gemv_fp4_fc*_weights_ for its pair-
      // interleaved layout, the decode GEMV reads its own ColToRow copy in
      // gemv_fp4_fc*_weights_decode_ instead (the SM80 layout is not GEMV-decodable).
      const uint8_t* gemv_fc1_weight = static_cast<const uint8_t*>(
          (enable_fp4_sm80_gemm_ ? gemv_fp4_fc1_weights_decode_ : gemv_fp4_fc1_weights_).get());
      const uint8_t* gemv_fc2_weight = static_cast<const uint8_t*>(
          (enable_fp4_sm80_gemm_ ? gemv_fp4_fc2_weights_decode_ : gemv_fp4_fc2_weights_).get());
      auto p_r2u_buf = GetScratchBuffer<int>(expanded, GetComputeStream(context));
      auto p_exp_buf = GetScratchBuffer<int>(expanded, GetComputeStream(context));
      auto p_efto_buf = GetScratchBuffer<int64_t>(num_experts + 1, GetComputeStream(context));
      const size_t elt = is_fp16_ ? sizeof(half) : sizeof(__nv_bfloat16);
      auto p_act_buf = GetScratchBuffer<void>(SafeInt<size_t>(expanded) * hidden * elt, GetComputeStream(context));
      auto p_fc1_buf = GetScratchBuffer<void>(SafeInt<size_t>(expanded) * inter * elt, GetComputeStream(context));
      auto p_fc2_buf = GetScratchBuffer<void>(SafeInt<size_t>(expanded) * hidden * elt, GetComputeStream(context));
      int* p_r2u = p_r2u_buf.get();
      int* p_exp = p_exp_buf.get();
      int64_t* p_efto = p_efto_buf.get();

      ck::ActivationParams act_params(activation_type_);
      act_params.alpha = activation_alpha_;
      act_params.beta = activation_beta_;
      act_params.swiglu_fusion = swiglu_fusion;
      act_params.limit = swiglu_limit_;

      ck::fusedBuildExpertMapsSortFirstToken(
          expert_indices, p_r2u, unpermuted_row_to_permuted_row, p_exp, p_efto,
          num_rows, num_experts, static_cast<int>(k_), 0, num_experts, stream);

      const void* fc1_bias = fc1_experts_bias_optional ? fc1_experts_bias_optional->DataRaw() : nullptr;
      const void* fc2_bias = fc2_experts_bias_optional ? fc2_experts_bias_optional->DataRaw() : nullptr;

      using MoeGemvConfig = gemv::MoeGemvConfig;

      // Choose the fc1 (SwiGLU) and fc2 GEMV tiling configs. CtaN/Threads are pure tiling
      // knobs (numerically bit-exact), so the only goal is picking the fastest. Reuse a
      // cached per-shape result when available; otherwise profile on a non-captured (warmup)
      // call and freeze the choice for CUDA-graph replay. During capture (or when autotune is
      // off) fall back to the default tiling.
      MoeGemvConfig fc1_config = MoeGemvConfig::kDefault;
      MoeGemvConfig fc2_config = MoeGemvConfig::kDefault;

      const int64_t row_bucket =
          onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler::bucketM(expanded);
      const Fp4GemvTuneKey tune_key{is_fp16_, row_bucket, hidden, inter, sm_};
      bool have_tune = false;
      {
        std::lock_guard<std::mutex> lock(fp4_gemv_tune_cache_mutex_);
        const auto cached_tune = fp4_gemv_tune_cache_.find(tune_key);
        have_tune = cached_tune != fp4_gemv_tune_cache_.end();
        if (have_tune) {
          fc1_config = cached_tune->second.fc1_config;
          fc2_config = cached_tune->second.fc2_config;
        }
      }

      cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
      CUDA_CALL_THROW(cudaStreamIsCapturing(stream, &capture_status));
      const bool is_capturing = capture_status != cudaStreamCaptureStatusNone;
      const bool do_tune = enable_fp4_gemv_autotune_ && !have_tune && !is_capturing;

      auto run_fused = [&](auto* t_ptr) {
        using T = std::remove_pointer_t<decltype(t_ptr)>;
        ck::expandInputRowsKernelLauncher<T, T>(
            static_cast<const T*>(input->DataRaw()), static_cast<T*>(p_act_buf.get()),
            nullptr, nullptr, p_r2u, num_rows, hidden, static_cast<int>(k_), num_experts,
            quant_params, false, p_efto, nullptr, nullptr, nullptr, stream);

        auto launch_fc1 = [&](MoeGemvConfig cfg) {
          gemv::launch_moe_gemv_fp4_symmetric_interleaved_swiglu<T>(
              static_cast<const T*>(p_act_buf.get()),
              gemv_fc1_weight,
              static_cast<const T*>(gemv_fp4_fc1_scales_.get()),
              static_cast<const T*>(fc1_bias), static_cast<T*>(p_fc1_buf.get()),
              p_efto, p_exp, num_experts, expanded, inter, hidden, gemv_group_size, sm_, act_params, cfg, stream);
        };
        auto launch_fc2 = [&](MoeGemvConfig cfg) {
          gemv::launch_moe_gemv_fp4_symmetric<T>(
              static_cast<const T*>(p_fc1_buf.get()),
              gemv_fc2_weight,
              static_cast<const T*>(gemv_fp4_fc2_scales_.get()),
              static_cast<const T*>(fc2_bias), static_cast<T*>(p_fc2_buf.get()),
              p_efto, p_exp, num_experts, expanded, hidden, inter, gemv_group_size, sm_, cfg, stream);
        };

        if (do_tune) {
          constexpr MoeGemvConfig kCandidates[] = {
              MoeGemvConfig::kDefault, MoeGemvConfig::kCtaN16, MoeGemvConfig::kThreads64};
          constexpr int kWarmup = 3;
          constexpr int kIters = 20;

          cudaEvent_t start_event = nullptr;
          cudaEvent_t stop_event = nullptr;
          CUDA_CALL_THROW(cudaEventCreate(&start_event));
          std::unique_ptr<CUevent_st, decltype(&cudaEventDestroy)> start_event_guard(start_event, cudaEventDestroy);
          CUDA_CALL_THROW(cudaEventCreate(&stop_event));
          std::unique_ptr<CUevent_st, decltype(&cudaEventDestroy)> stop_event_guard(stop_event, cudaEventDestroy);

          auto time_launch = [&](auto&& launch_fn) -> float {
            for (int i = 0; i < kWarmup; ++i) {
              launch_fn();
            }
            CUDA_CALL_THROW(cudaStreamSynchronize(stream));
            CUDA_CALL_THROW(cudaEventRecord(start_event, stream));
            for (int i = 0; i < kIters; ++i) {
              launch_fn();
            }
            CUDA_CALL_THROW(cudaEventRecord(stop_event, stream));
            CUDA_CALL_THROW(cudaEventSynchronize(stop_event));
            float elapsed_ms = 0.0f;
            CUDA_CALL_THROW(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
            return elapsed_ms;
          };

          // fc1 reads p_act_buf (populated by the expand above).
          const bool log_tune = enable_fp4_gemv_autotune_log_;
          float best_fc1 = std::numeric_limits<float>::max();
          for (MoeGemvConfig cfg : kCandidates) {
            if (!gemv::is_moe_gemv_fp4_supported(sm_, expanded, fc1_n, hidden, gemv_group_size, cfg)) {
              continue;
            }
            const float ms = time_launch([&] { launch_fc1(cfg); });
            if (log_tune) {
              LOGS_DEFAULT(WARNING) << "FP4 GEMV autotune candidate: fc1 expanded=" << expanded
                                    << " n=" << fc1_n << " k=" << hidden << " cfg=" << QMoEGemvConfigName(cfg)
                                    << " " << ms << "ms/" << kIters << "it";
            }
            if (ms < best_fc1) {
              best_fc1 = ms;
              fc1_config = cfg;
            }
          }
          // fc2 reads p_fc1_buf; populate it once with the chosen fc1 config before timing fc2.
          launch_fc1(fc1_config);
          float best_fc2 = std::numeric_limits<float>::max();
          for (MoeGemvConfig cfg : kCandidates) {
            if (!gemv::is_moe_gemv_fp4_supported(sm_, expanded, hidden, inter, gemv_group_size, cfg)) {
              continue;
            }
            const float ms = time_launch([&] { launch_fc2(cfg); });
            if (log_tune) {
              LOGS_DEFAULT(WARNING) << "FP4 GEMV autotune candidate: fc2 expanded=" << expanded
                                    << " n=" << hidden << " k=" << inter << " cfg=" << QMoEGemvConfigName(cfg)
                                    << " " << ms << "ms/" << kIters << "it";
            }
            if (ms < best_fc2) {
              best_fc2 = ms;
              fc2_config = cfg;
            }
          }

          {
            std::lock_guard<std::mutex> lock(fp4_gemv_tune_cache_mutex_);
            fp4_gemv_tune_cache_.try_emplace(tune_key, Fp4GemvTuneResult{fc1_config, fc2_config});
          }
          if (log_tune) {
            LOGS_DEFAULT(WARNING) << "FP4 GEMV autotune: is_fp16=" << is_fp16_ << " expanded=" << expanded
                                  << " hidden=" << hidden << " inter=" << inter << " fc1="
                                  << QMoEGemvConfigName(fc1_config) << " (" << best_fc1 << "ms/" << kIters
                                  << "it) fc2=" << QMoEGemvConfigName(fc2_config) << " (" << best_fc2 << "ms/"
                                  << kIters << "it)";
          }
        }

        launch_fc1(fc1_config);
        launch_fc2(fc2_config);
        ck::finalizeMoeRoutingKernelLauncher<T, T, T>(
            static_cast<const T*>(p_fc2_buf.get()), static_cast<T*>(output->MutableDataRaw()),
            nullptr, expert_scales, unpermuted_row_to_permuted_row, p_r2u, expert_indices,
            p_efto, num_rows, hidden, static_cast<int64_t>(k_), num_experts,
            parallelism_config, false, stream);
      };

      if (is_fp16_) {
        run_fused(static_cast<half*>(nullptr));
      } else {
        run_fused(static_cast<__nv_bfloat16*>(nullptr));
      }
      return Status::OK();
    }
  }

  const void* fc1_weight_data = fc1_experts_weights ? fc1_experts_weights->DataRaw() : nullptr;
  const void* fc2_weight_data = fc2_experts_weights ? fc2_experts_weights->DataRaw() : nullptr;
  if (fp4_sm80_prefill) {
    // SM80 FP4 grouped GEMM: consume the e2m1 weights in the SM80 CUTLASS interleaved layout
    // that PrePack produced into the GEMV interleaved-layout buffers (same layout the INT4 SM80
    // grouped GEMM uses). The activation-dtype group scales are wired via quant_params above.
    fc1_weight_data = gemv_fp4_fc1_weights_.get();
    fc2_weight_data = gemv_fp4_fc2_weights_.get();
  } else if ((is_fp4 && route_native_fp4) || (is_wfp4afp8 && !use_wfp4afp8_dequant_fallback_)) {
    // The native CUTLASS FP4 paths consume weights in the repacked FP4
    // layout produced by PrePack. If PrePack never ran (e.g.
    // ``session.disable_prepacking`` is set) the repacked buffers stay null and
    // falling through to the raw initializer bytes would feed a non-CUTLASS
    // layout to the runner, producing silently wrong output. Fail loudly.
    if (packed_fp4_fc1_weights_ == nullptr || packed_fp4_fc2_weights_ == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "QMoE native FP4 requires PrePack to run, but the repacked FP4 weight "
                             "buffers were not produced (is session.disable_prepacking set?). "
                             "Enable prepacking to use the native FP4 path.");
    }
    fc1_weight_data = packed_fp4_fc1_weights_.get();
    fc2_weight_data = packed_fp4_fc2_weights_.get();
  } else if (int_weights_consumed_by_prepack) {
    // PrePack converted the raw int4/int8 weights to the CUTLASS fpA_intB
    // layout that the runner consumes and freed the source initializer
    // (``is_packed = true``). Gate on ``int_weights_consumed_by_prepack``
    // (which already requires both packed weight buffers) rather than
    // just ``is_int && !weights_prepacked_``: when prepacking is disabled at
    // the session level (``session.disable_prepacking``) PrePack never runs,
    // the prepack buffers stay null, and the raw initializer pointers read
    // above must be kept so the runner is not handed null weight pointers.
    fc1_weight_data = packed_fc1_weights_.get();
    fc2_weight_data = packed_fc2_weights_.get();
  }
  IAllocatorUniquePtr<void> dequant_fc1_weights;
  IAllocatorUniquePtr<void> dequant_fc2_weights;
  // FP4 (W4A16) and WFP4AFP8 (W4A8) share the MXFP4 weight format (Float8E8M0 block scales, block 32).
  // NVFP4 (W4A16) uses Float8E4M3FN block scales with block 16 and always runs the dequant fallback.
  // When the native CUTLASS path is unavailable on the current SM (always for NVFP4), or when native
  // FP4 routes this call to the dense fallback for the large per-expert-M regime, dequantize the E2M1
  // weights to FP16/BF16 and run the dense A16 runner.
  if (is_nvfp4 ||
      (((is_fp4 && !route_native_fp4) || (is_wfp4afp8 && use_wfp4afp8_dequant_fallback_)) && !fp4_sm80_prefill)) {
    // The dequant kernel expects raw [E, n, k_blocks] block scales. When native FP4 is enabled
    // (this is the large per-expert-M fallback), packed_fp4_*_block_scales_ holds the TMA-swizzled
    // layout, so use the raw copy kept in gemv_fp4_*_block_raw_ instead. NVFP4 and the SM<90
    // dequant-only build have no raw copy, so packed_fp4_*_block_scales_ already holds the raw scales.
    const void* p_fc1_block_scales = gemv_fp4_fc1_block_raw_ ? gemv_fp4_fc1_block_raw_.get()
                                                             : (packed_fp4_fc1_block_scales_ ? packed_fp4_fc1_block_scales_.get()
                                                                                             : (fp4_fc1_block_scales ? fp4_fc1_block_scales->DataRaw() : nullptr));
    const void* p_fc1_global_scale = packed_fc1_global_scale_ ? packed_fc1_global_scale_.get()
                                                              : (fc1_global_scale ? fc1_global_scale->DataRaw() : nullptr);
    const void* p_fc2_block_scales = gemv_fp4_fc2_block_raw_ ? gemv_fp4_fc2_block_raw_.get()
                                                             : (packed_fp4_fc2_block_scales_ ? packed_fp4_fc2_block_scales_.get()
                                                                                             : (fp4_fc2_block_scales ? fp4_fc2_block_scales->DataRaw() : nullptr));
    const void* p_fc2_global_scale = packed_fc2_global_scale_ ? packed_fc2_global_scale_.get()
                                                              : (fc2_global_scale ? fc2_global_scale->DataRaw() : nullptr);
    ORT_RETURN_IF_NOT(p_fc1_block_scales && p_fc1_global_scale && p_fc2_block_scales && p_fc2_global_scale,
                      "QMoE FP4 dequant fallback requires block and global scales for fc1 and fc2.");

    int fc1_n = static_cast<int>(is_fused_swiglu ? moe_params.inter_size * 2 : moe_params.inter_size);
    int fc1_k = static_cast<int>(moe_params.hidden_size);
    int fc2_n = static_cast<int>(moe_params.hidden_size);
    int fc2_k = static_cast<int>(moe_params.inter_size);
    int num_experts = static_cast<int>(moe_params.num_experts);
    size_t element_size = is_fp16_ ? sizeof(half) : sizeof(__nv_bfloat16);
    size_t fc1_bytes = SafeInt<size_t>(num_experts) * fc1_n * fc1_k * element_size;
    size_t fc2_bytes = SafeInt<size_t>(num_experts) * fc2_n * fc2_k * element_size;
    dequant_fc1_weights = GetScratchBuffer<void>(fc1_bytes, GetComputeStream(context));
    dequant_fc2_weights = GetScratchBuffer<void>(fc2_bytes, GetComputeStream(context));

    // Choose the FP4 (MXFP4 / E8M0, block 32) or NVFP4 (E4M3, block 16) dequant launcher.
    auto dequant = [&](const uint8_t* weights, const uint8_t* block_scales, const float* global_scale,
                       void* out, int n, int k) {
      if (is_fp16_) {
        half* out_h = static_cast<half*>(out);
        if (is_nvfp4) {
          LaunchQMoEDequantizeNvfp4Weights(weights, block_scales, global_scale, out_h, num_experts, n, k, stream);
        } else {
          LaunchQMoEDequantizeFp4Weights(weights, block_scales, global_scale, out_h, num_experts, n, k, stream);
        }
      } else {
        __nv_bfloat16* out_b = static_cast<__nv_bfloat16*>(out);
        if (is_nvfp4) {
          LaunchQMoEDequantizeNvfp4Weights(weights, block_scales, global_scale, out_b, num_experts, n, k, stream);
        } else {
          LaunchQMoEDequantizeFp4Weights(weights, block_scales, global_scale, out_b, num_experts, n, k, stream);
        }
      }
    };
    // This dequant fallback is reachable only for the FP4 family (fp4/nvfp4) and the
    // WFP4AFP8 dequant path -- never for quant_type='int'. Only the int path nulls out
    // fc*_experts_weights (int_weights_consumed_by_prepack), so the raw weight pointers are
    // guaranteed live here. Enforce that invariant explicitly so a future mode-guard change
    // that lets int fall through cannot silently dereference a null weight tensor.
    ORT_ENFORCE(fc1_experts_weights != nullptr && fc2_experts_weights != nullptr,
                "QMoE FP4/NVFP4 dequant fallback requires the raw expert-weight tensors; got null "
                "(this path must not be reached in int-weight prepack mode).");
    dequant(static_cast<const uint8_t*>(fc1_experts_weights->DataRaw()),
            static_cast<const uint8_t*>(p_fc1_block_scales),
            static_cast<const float*>(p_fc1_global_scale),
            dequant_fc1_weights.get(), fc1_n, fc1_k);
    dequant(static_cast<const uint8_t*>(fc2_experts_weights->DataRaw()),
            static_cast<const uint8_t*>(p_fc2_block_scales),
            static_cast<const float*>(p_fc2_global_scale),
            dequant_fc2_weights.get(), fc2_n, fc2_k);
    fc1_weight_data = dequant_fc1_weights.get();
    fc2_weight_data = dequant_fc2_weights.get();
  } else if (is_fp8 && use_fp8_dequant_fallback_) {
    const void* p_fc1_global_scale = packed_fc1_global_scale_ ? packed_fc1_global_scale_.get()
                                                              : (fc1_global_scale ? fc1_global_scale->DataRaw() : nullptr);
    const void* p_fc2_global_scale = packed_fc2_global_scale_ ? packed_fc2_global_scale_.get()
                                                              : (fc2_global_scale ? fc2_global_scale->DataRaw() : nullptr);
    ORT_RETURN_IF_NOT(p_fc1_global_scale && p_fc2_global_scale,
                      "QMoE FP8 dequant fallback requires fc1_global_scale and fc2_global_scale.");

    int fc1_n = static_cast<int>(is_fused_swiglu ? moe_params.inter_size * 2 : moe_params.inter_size);
    int fc1_k = static_cast<int>(moe_params.hidden_size);
    int fc2_n = static_cast<int>(moe_params.hidden_size);
    int fc2_k = static_cast<int>(moe_params.inter_size);
    int num_experts = static_cast<int>(moe_params.num_experts);
    size_t element_size = is_fp16_ ? sizeof(half) : sizeof(__nv_bfloat16);
    size_t fc1_bytes = SafeInt<size_t>(num_experts) * fc1_n * fc1_k * element_size;
    size_t fc2_bytes = SafeInt<size_t>(num_experts) * fc2_n * fc2_k * element_size;
    dequant_fc1_weights = GetScratchBuffer<void>(fc1_bytes, GetComputeStream(context));
    dequant_fc2_weights = GetScratchBuffer<void>(fc2_bytes, GetComputeStream(context));

    if (is_fp16_) {
      LaunchQMoEDequantizeFp8Weights(static_cast<const uint8_t*>(fc1_experts_weights->DataRaw()),
                                     static_cast<const float*>(p_fc1_global_scale),
                                     static_cast<half*>(dequant_fc1_weights.get()), num_experts, fc1_n, fc1_k, stream);
      LaunchQMoEDequantizeFp8Weights(static_cast<const uint8_t*>(fc2_experts_weights->DataRaw()),
                                     static_cast<const float*>(p_fc2_global_scale),
                                     static_cast<half*>(dequant_fc2_weights.get()), num_experts, fc2_n, fc2_k, stream);
    } else {
      LaunchQMoEDequantizeFp8Weights(static_cast<const uint8_t*>(fc1_experts_weights->DataRaw()),
                                     static_cast<const float*>(p_fc1_global_scale),
                                     static_cast<__nv_bfloat16*>(dequant_fc1_weights.get()), num_experts, fc1_n, fc1_k, stream);
      LaunchQMoEDequantizeFp8Weights(static_cast<const uint8_t*>(fc2_experts_weights->DataRaw()),
                                     static_cast<const float*>(p_fc2_global_scale),
                                     static_cast<__nv_bfloat16*>(dequant_fc2_weights.get()), num_experts, fc2_n, fc2_k, stream);
    }
    fc1_weight_data = dequant_fc1_weights.get();
    fc2_weight_data = dequant_fc2_weights.get();
  }

  // Set tactic and run MoE. Must hold the mutex since setTactic mutates runner state.
  {
    std::lock_guard<std::mutex> profiler_lock(mGemmProfilerMutex);
    active_runner->setTactic(config1, config2);
    active_runner->runMoe(
        input->DataRaw(),
        nullptr,
        expert_indices,
        expert_scales,
        fc1_weight_data,
        fc1_experts_bias_optional ? fc1_experts_bias_optional->DataRaw() : nullptr,
        activation_type_,
        fc2_weight_data,
        fc2_experts_bias_optional ? fc2_experts_bias_optional->DataRaw() : nullptr,
        quant_params,
        moe_params.num_rows,
        moe_params.hidden_size,
        moe_params.inter_size,
        moe_params.num_experts,
        k_,
        workspace_ptr,
        output->MutableDataRaw(),
        unpermuted_row_to_permuted_row,
        parallelism_config,
        [&]() {
          onnxruntime::llm::kernels::cutlass_kernels::ActivationParams params(activation_type_);
          params.alpha = activation_alpha_;
          params.beta = activation_beta_;
          params.swiglu_fusion = swiglu_fusion;
          params.limit = swiglu_limit_;
          return params;
        }(),
        stream);
  }

  return Status::OK();
}

Status QMoE::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                     bool& is_packed, PrePackedWeights* prepacked_weights) {
  ORT_UNUSED_PARAMETER(prepacked_weights);
  is_packed = false;

  cudaStream_t stream = 0;  // Use default stream for PrePack operations

  DUMP_TENSOR_INIT();

#if DUMP_TENSOR_LEVEL >= 1
  auto dump_tensor = [&](const char* name, const IAllocatorUniquePtr<void>& packed_scales, const Tensor& scales) {
    auto shape = scales.Shape();
    if (shape.NumDimensions() == 3 && is_fp16_) {
      size_t rows = shape[1];
      size_t cols = shape[2];
      size_t batch = shape[0];
      if (expert_weight_bits_ == 8 && block_size_ <= 0 && strstr(name, "bias") != nullptr) {
        DUMP_TENSOR(name, static_cast<const uint8_t*>(packed_scales.get()), int(batch), int(cols), int(rows));
      } else {
        DUMP_TENSOR(name, static_cast<const half*>(packed_scales.get()), int(batch), int(cols), int(rows));
      }
    }
  };
#define DUMP_PACK_TENSOR(name, packed_scales, scales) dump_tensor(name, packed_scales, scales)
#else
#define DUMP_PACK_TENSOR(name, packed_scales, scales)
#endif

  if (input_idx == 2 && ((quant_type_ == "fp4" && !use_fp4_dequant_fallback_) ||
                         (quant_type_ == "wfp4afp8" && !use_wfp4afp8_dequant_fallback_))) {
    PrePackRepackFP4Weights(tensor, stream, alloc, packed_fp4_fc1_weights_, is_packed);
    // Native CUTLASS + GEMV coexist: also pre-pack the GEMV layout for decode (interleaved
    // layout when ORT_FP4_GEMV_INTERLEAVED=1, else the [E,n,k/2] row-major layout).
    if (quant_type_ == "fp4" && enable_fp4_gemv_) {
      bool local_packed = false;
      PrePackRepackFP4Weights(tensor, stream, alloc, gemv_fp4_fc1_weights_, local_packed,
                              onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvUseInterleaved());
    }
    is_packed = false;
  } else if (input_idx == 5 && ((quant_type_ == "fp4" && !use_fp4_dequant_fallback_) ||
                                (quant_type_ == "wfp4afp8" && !use_wfp4afp8_dequant_fallback_))) {
    PrePackRepackFP4Weights(tensor, stream, alloc, packed_fp4_fc2_weights_, is_packed);
    if (quant_type_ == "fp4" && enable_fp4_gemv_) {
      bool local_packed = false;
      PrePackRepackFP4Weights(tensor, stream, alloc, gemv_fp4_fc2_weights_, local_packed,
                              onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvUseInterleaved());
    }
    is_packed = false;
  } else if (input_idx == 2 && (quant_type_ == "fp4" || quant_type_ == "nvfp4") && enable_fp4_gemv_) {
    // Fused FP4 GEMV: lay out fc1 weights as [E, 2*inter, hidden/2] row-major. Keep
    // is_packed = false so the raw [E, hidden, n/2] initializer remains available for the
    // dequant fallback used by shapes the GEMV does not support. When the SM80 grouped-GEMM
    // port is enabled (MXFP4 only), force the SM80 CUTLASS ColumnMajorTileInterleave layout
    // so this buffer feeds the prefill grouped GEMM; the decode GEMV then reads its
    // own ColToRow copy in gemv_fp4_fc1_weights_decode_ (packed below) since it cannot consume
    // that layout. NVFP4 (block 16) has no native/SM80 path and always uses the plain ColToRow
    // layout the non-interleaved GEMV consumes.
    const bool nvfp4 = (quant_type_ == "nvfp4");
    const bool use_interleave =
        !nvfp4 && (onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvUseInterleaved() || enable_fp4_sm80_gemm_);
    const bool sm80_pair = !nvfp4 && enable_fp4_sm80_gemm_;
    bool local_packed = false;
    PrePackRepackFP4Weights(tensor, stream, alloc, gemv_fp4_fc1_weights_, local_packed,
                            use_interleave, /*sm80_pair_interleave=*/sm80_pair);
    if (enable_fp4_sm80_gemm_) {
      bool decode_packed = false;
      PrePackRepackFP4Weights(tensor, stream, alloc, gemv_fp4_fc1_weights_decode_, decode_packed,
                              onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvUseInterleaved(),
                              /*sm80_pair_interleave=*/false);
    }
  } else if (input_idx == 5 && (quant_type_ == "fp4" || quant_type_ == "nvfp4") && enable_fp4_gemv_) {
    const bool nvfp4 = (quant_type_ == "nvfp4");
    const bool use_interleave =
        !nvfp4 && (onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvUseInterleaved() || enable_fp4_sm80_gemm_);
    const bool sm80_pair = !nvfp4 && enable_fp4_sm80_gemm_;
    bool local_packed = false;
    PrePackRepackFP4Weights(tensor, stream, alloc, gemv_fp4_fc2_weights_, local_packed,
                            use_interleave, /*sm80_pair_interleave=*/sm80_pair);
    if (enable_fp4_sm80_gemm_) {
      bool decode_packed = false;
      PrePackRepackFP4Weights(tensor, stream, alloc, gemv_fp4_fc2_weights_decode_, decode_packed,
                              onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvUseInterleaved(),
                              /*sm80_pair_interleave=*/false);
    }
  } else if (input_idx == 2 && quant_type_ == "int" && !weights_prepacked_) {
    // Caller opted in (``weights_prepacked=0`` attribute) to having ORT
    // do the CUTLASS fpA_intB layout transform internally, instead of
    // shipping pre-prepacked bytes. Mirrors ``MatMulNBits::PrePack_B``
    // looped over the E experts of ``[E, N, K/pack]``. We cache the
    // source shape in ``fc1_weights_shape_`` so ``CheckInputs`` can be
    // satisfied without holding the original initializer alive, then
    // set ``is_packed = true`` to let ORT free it.
    fc1_weights_shape_ = tensor.Shape();
    PrePackIntExpertWeights(tensor, stream, alloc, packed_fc1_weights_, is_packed);
  } else if (input_idx == 5 && quant_type_ == "int" && !weights_prepacked_) {
    fc2_weights_shape_ = tensor.Shape();
    PrePackIntExpertWeights(tensor, stream, alloc, packed_fc2_weights_, is_packed);
  } else if (input_idx == 3) {  // fc1_scales
    DUMP_TENSOR("fc1_scales", tensor);
    if (quant_type_ == "nvfp4") {
      ORT_ENFORCE(tensor.IsDataType<Float8E4M3FN>() && tensor.Shape().NumDimensions() == 3,
                  "QMoE NVFP4 fc1_scales must be a 3-D float8e4m3fn tensor.");
    }
    if (quant_type_ == "fp4" && !use_fp4_dequant_fallback_) {
      PrePackFp4ScalesForTmaWs(tensor, stream, alloc, packed_fp4_fc1_block_scales_, is_packed);
      // Native CUTLASS swizzles the block scales; keep a raw copy + dims so GEMV decode can
      // build its combined scale layout.
      if (enable_fp4_gemv_ && tensor.Shape().NumDimensions() == 3) {
        bool raw_packed = false;
        PrePackCopyToGpu(tensor, stream, alloc, gemv_fp4_fc1_block_raw_, raw_packed);
        gemv_fp4_fc1_scale_e_ = tensor.Shape()[0];
        gemv_fp4_fc1_scale_n_ = tensor.Shape()[1];
        gemv_fp4_fc1_scale_kb_ = tensor.Shape()[2];
        TryBuildGemvFp4Scales(1, stream, alloc);
      }
    } else if (quant_type_ == "wfp4afp8" && !use_wfp4afp8_dequant_fallback_) {
      PrePackSwizzleBlockScales(tensor, stream, alloc, packed_fp4_fc1_block_scales_, is_packed);
    } else if (quant_type_ == "fp4" || quant_type_ == "nvfp4" || quant_type_ == "wfp4afp8") {
      PrePackCopyToGpu(tensor, stream, alloc, packed_fp4_fc1_block_scales_, is_packed);
      if ((quant_type_ == "fp4" || quant_type_ == "nvfp4") && enable_fp4_gemv_ && tensor.Shape().NumDimensions() == 3) {
        gemv_fp4_fc1_scale_e_ = tensor.Shape()[0];
        gemv_fp4_fc1_scale_n_ = tensor.Shape()[1];
        gemv_fp4_fc1_scale_kb_ = tensor.Shape()[2];
        TryBuildGemvFp4Scales(1, stream, alloc);
      }
    } else if (quant_type_ == "int") {
      PrePackTransposeAndPack(tensor, stream, alloc, packed_fc1_scales_, is_packed);
      DUMP_PACK_TENSOR("packed_fc1_scales", packed_fc1_scales_, tensor);
    }
    if (quant_type_ == "fp4" || quant_type_ == "nvfp4" || quant_type_ == "wfp4afp8") {
      is_packed = false;
    }
  } else if (input_idx == 6) {  // fc2_scales
    DUMP_TENSOR("fc2_scales", tensor);
    if (quant_type_ == "nvfp4") {
      ORT_ENFORCE(tensor.IsDataType<Float8E4M3FN>() && tensor.Shape().NumDimensions() == 3,
                  "QMoE NVFP4 fc2_scales must be a 3-D float8e4m3fn tensor.");
    }
    if (quant_type_ == "fp4" && !use_fp4_dequant_fallback_) {
      PrePackFp4ScalesForTmaWs(tensor, stream, alloc, packed_fp4_fc2_block_scales_, is_packed);
      if (enable_fp4_gemv_ && tensor.Shape().NumDimensions() == 3) {
        bool raw_packed = false;
        PrePackCopyToGpu(tensor, stream, alloc, gemv_fp4_fc2_block_raw_, raw_packed);
        gemv_fp4_fc2_scale_e_ = tensor.Shape()[0];
        gemv_fp4_fc2_scale_n_ = tensor.Shape()[1];
        gemv_fp4_fc2_scale_kb_ = tensor.Shape()[2];
        TryBuildGemvFp4Scales(2, stream, alloc);
      }
    } else if (quant_type_ == "wfp4afp8" && !use_wfp4afp8_dequant_fallback_) {
      PrePackSwizzleBlockScales(tensor, stream, alloc, packed_fp4_fc2_block_scales_, is_packed);
    } else if (quant_type_ == "fp4" || quant_type_ == "nvfp4" || quant_type_ == "wfp4afp8") {
      PrePackCopyToGpu(tensor, stream, alloc, packed_fp4_fc2_block_scales_, is_packed);
      if ((quant_type_ == "fp4" || quant_type_ == "nvfp4") && enable_fp4_gemv_ && tensor.Shape().NumDimensions() == 3) {
        gemv_fp4_fc2_scale_e_ = tensor.Shape()[0];
        gemv_fp4_fc2_scale_n_ = tensor.Shape()[1];
        gemv_fp4_fc2_scale_kb_ = tensor.Shape()[2];
        TryBuildGemvFp4Scales(2, stream, alloc);
      }
    } else if (quant_type_ == "int") {
      PrePackTransposeAndPack(tensor, stream, alloc, packed_fc2_scales_, is_packed);
      DUMP_PACK_TENSOR("packed_fc2_scales", packed_fc2_scales_, tensor);
    }
    if (quant_type_ == "fp4" || quant_type_ == "nvfp4" || quant_type_ == "wfp4afp8") {
      is_packed = false;
    }
  } else if (input_idx == 11) {  // fc1_zeros
    DUMP_TENSOR("fc1_zeros", tensor);
    PrePackComputeBias(tensor, stream, alloc, packed_fc1_scales_, packed_fc1_bias_, is_packed);
    DUMP_PACK_TENSOR("packed_fc1_bias", packed_fc1_bias_, tensor);
  } else if (input_idx == 12) {  // fc2_zeros
    DUMP_TENSOR("fc2_zeros", tensor);
    PrePackComputeBias(tensor, stream, alloc, packed_fc2_scales_, packed_fc2_bias_, is_packed);
    DUMP_PACK_TENSOR("packed_fc2_bias", packed_fc2_bias_, tensor);
  } else if ((input_idx == 15 || input_idx == 16) &&
             (quant_type_ == "fp4" || quant_type_ == "nvfp4" || quant_type_ == "fp8" || quant_type_ == "wfp4afp8")) {
    ORT_ENFORCE(tensor.IsDataType<float>() && tensor.Shape().NumDimensions() == 1,
                "QMoE ", quant_type_, " global scales must be 1-D float tensors.");
    if (input_idx == 15) {
      PrePackCopyToGpu(tensor, stream, alloc, packed_fc1_global_scale_, is_packed);
      fp4_fc1_global_scale_e_ = tensor.Shape()[0];
      TryBuildGemvFp4Scales(1, stream, alloc);
    } else {
      PrePackCopyToGpu(tensor, stream, alloc, packed_fc2_global_scale_, is_packed);
      fp4_fc2_global_scale_e_ = tensor.Shape()[0];
      TryBuildGemvFp4Scales(2, stream, alloc);
    }
    is_packed = false;
  } else if ((input_idx == 17 || input_idx == 18) && quant_type_ == "wfp4afp8") {
    if (input_idx == 17) {
      PrePackCopyToGpu(tensor, stream, alloc, packed_fc1_act_scale_, is_packed);
    } else {
      PrePackCopyToGpu(tensor, stream, alloc, packed_fc2_act_scale_, is_packed);
    }
  }

  return Status::OK();
}

// ---------------------------------------------------------------------------
// PrePack helper: Transpose [E, N, Blocks] -> [E, Blocks, N] and copy to GPU.
// ---------------------------------------------------------------------------
void QMoE::PrePackTransposeAndPack(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                   IAllocatorUniquePtr<void>& packed_buf, bool& is_packed) {
  auto shape = tensor.Shape();
  size_t bytes = tensor.SizeInBytes();
  packed_buf = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

  const void* p_src = tensor.DataRaw();
  IAllocatorUniquePtr<void> temp_src_gpu;
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    temp_src_gpu = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);
    CUDA_CALL_THROW(cudaMemcpyAsync(temp_src_gpu.get(), p_src, bytes, cudaMemcpyDefault, stream));
    p_src = temp_src_gpu.get();
  }

  if (shape.NumDimensions() == 3 && shape[2] > 1) {
    size_t rows = shape[1];   // N
    size_t cols = shape[2];   // Blocks
    size_t batch = shape[0];  // Experts
    auto type = tensor.DataType();
    if (type == DataTypeImpl::GetType<MLFloat16>()) {
      LaunchQMoETranspose2D(static_cast<const half*>(p_src), static_cast<half*>(packed_buf.get()), batch, rows, cols, stream);
    } else if (type == DataTypeImpl::GetType<BFloat16>()) {
      LaunchQMoETranspose2D(static_cast<const __nv_bfloat16*>(p_src), static_cast<__nv_bfloat16*>(packed_buf.get()), batch, rows, cols, stream);
    } else if (type == DataTypeImpl::GetType<float>()) {
      LaunchQMoETranspose2D(static_cast<const float*>(p_src), static_cast<float*>(packed_buf.get()), batch, rows, cols, stream);
    } else if (type == DataTypeImpl::GetType<uint8_t>()) {
      LaunchQMoETranspose2D(static_cast<const uint8_t*>(p_src), static_cast<uint8_t*>(packed_buf.get()), batch, rows, cols, stream);
    } else if (type == DataTypeImpl::GetType<Float8E8M0>()) {
      LaunchQMoETranspose2D(static_cast<const uint8_t*>(p_src), static_cast<uint8_t*>(packed_buf.get()), batch, rows, cols, stream);
    } else {
      ORT_THROW("Unsupported data type for scale transposition");
    }
  } else {
    CUDA_CALL_THROW(cudaMemcpyAsync(packed_buf.get(), p_src, bytes, cudaMemcpyDefault, stream));
  }

  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
}

// ---------------------------------------------------------------------------
// PrePack helper: Copy tensor to GPU without transformation.
// ---------------------------------------------------------------------------
void QMoE::PrePackCopyToGpu(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                            IAllocatorUniquePtr<void>& packed_buf, bool& is_packed) {
  size_t bytes = tensor.SizeInBytes();
  packed_buf = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);
  const void* p_src = tensor.DataRaw();
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    CUDA_CALL_THROW(cudaMemcpyAsync(packed_buf.get(), p_src, bytes, cudaMemcpyHostToDevice, stream));
  } else {
    CUDA_CALL_THROW(cudaMemcpyAsync(packed_buf.get(), p_src, bytes, cudaMemcpyDeviceToDevice, stream));
  }
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
}

// ---------------------------------------------------------------------------
// PrePack helper: int4/int8 per-expert weights → CUTLASS fpA_intB layout.
// ---------------------------------------------------------------------------
// Mirrors ``MatMulNBits::PrePack_B`` but loops over the leading E (experts)
// dimension. Input ``tensor`` is the row-major 3-D ``[E, N, K/(8/bits)]``
// quantized weight initializer; output is a GPU buffer in the
// kernel-expected ``[E, K, N/(8/bits)]`` layout.
void QMoE::PrePackIntExpertWeights(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                   IAllocatorUniquePtr<void>& packed_buf, bool& is_packed) {
  ORT_ENFORCE(expert_weight_bits_ == 4 || expert_weight_bits_ == 8,
              "PrePackIntExpertWeights: only 4 and 8 bits are supported, got ", expert_weight_bits_);
  ORT_ENFORCE(sm_ >= 75,
              "PrePackIntExpertWeights: quant_type='int' with weights_prepacked=0 requires SM75+ CUDA hardware, got SM",
              sm_);
  const auto& shape = tensor.Shape();
  ORT_ENFORCE(shape.NumDimensions() == 3,
              "PrePackIntExpertWeights: expected 3-D weight tensor [E, N, K/pack], got ndim=",
              shape.NumDimensions());

  const int bits = static_cast<int>(expert_weight_bits_);
  const int pack_factor = 8 / bits;
  const int64_t num_experts = shape[0];
  const int64_t n = shape[1];
  const int64_t k_packed = shape[2];
  const int64_t k = k_packed * pack_factor;

  // The CUDA QMoE int4/int8 MoE GEMM always dispatches to the Ampere (SM80)
  // grouped-GEMM kernel -- even on SM90 -- because mixed int-weight + fp16/bf16
  // is not a valid Hopper TMA warp-specialized specialisation. The kernel thus
  // consumes the SM80 CUTLASS fpA_intB layout on every GPU, so the weights must
  // always be preprocessed for SM80 regardless of the runtime device SM.
  // (Using get_arch_for_mixed_gemm_weight_preprocess(sm_) here would emit the
  // SM90 layout on Hopper, which the SM80 kernel cannot consume -> wrong output.)
  const int packing_sm =
      onnxruntime::llm::kernels::weight_only::get_arch_for_mixed_gemm_weight_preprocess(80);

  // Per-expert sizes.
  const size_t per_expert_bytes = static_cast<size_t>(n) * static_cast<size_t>(k) / pack_factor;
  const size_t total_bytes = per_expert_bytes * static_cast<size_t>(num_experts);

  // Output buffer holds all E prepacked experts back-to-back in
  // [E, K, N/pack_factor] layout.
  packed_buf = IAllocator::MakeUniquePtr<void>(alloc, total_bytes, /*use_reserve=*/true);
  int8_t* dst_all = reinterpret_cast<int8_t*>(packed_buf.get());

  // Two transient per-expert scratch buffers reused across experts.
  IAllocatorUniquePtr<void> transposed_scratch =
      this->GetTransientScratchBuffer<void>(per_expert_bytes);
  int8_t* transposed_scratch_ptr = reinterpret_cast<int8_t*>(transposed_scratch.get());

  IAllocatorUniquePtr<void> src_gpu_scratch;
  const uint8_t* src_base_gpu = nullptr;
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    src_gpu_scratch = this->GetTransientScratchBuffer<void>(total_bytes);
    CUDA_CALL_THROW(cudaMemcpyAsync(src_gpu_scratch.get(), tensor.DataRaw(), total_bytes,
                                    cudaMemcpyHostToDevice, stream));
    src_base_gpu = reinterpret_cast<const uint8_t*>(src_gpu_scratch.get());
  } else {
    src_base_gpu = reinterpret_cast<const uint8_t*>(tensor.DataRaw());
  }

  IAllocatorUniquePtr<int32_t> permutation_map = this->GetTransientScratchBuffer<int32_t>(32);

  using onnxruntime::llm::kernels::weight_only::QuantType;
  const QuantType quant_type = (bits == 4) ? QuantType::W4_A16 : QuantType::W8_A16;

  for (int64_t e = 0; e < num_experts; ++e) {
    const uint8_t* src_e = src_base_gpu + static_cast<size_t>(e) * per_expert_bytes;
    int8_t* dst_e = dst_all + static_cast<size_t>(e) * per_expert_bytes;

    // Step 1: transpose + (for int4) unpack/zero-point bias into the
    // transposed-int8 scratch buffer. Mirrors MatMulNBits's PrePack_B.
    if (bits == 4) {
      onnxruntime::llm::kernels::fpA_intB_gemv::unpack_uint4_transposed_to_int8_direct_cuda(
          stream, transposed_scratch_ptr, src_e, static_cast<int>(n), static_cast<int>(k));
    } else {
      onnxruntime::llm::kernels::fpA_intB_gemv::transpose_uint8_matrix_and_convert_to_int8(
          stream, transposed_scratch_ptr, src_e, static_cast<int>(n), static_cast<int>(k));
    }

    // Step 2: apply the CUTLASS fpA_intB row-permutation / column-interleave /
    // bias / pair-interleave transform into the per-expert output slot.
    // ``synchronize=false``: avoid one host-blocking ``cudaStreamSynchronize``
    // per expert (which would scale model-load time with ``num_experts``).
    // Stream ordering guarantees expert e's transform finishes before expert
    // e+1 reuses the shared transpose scratch, and a single sync after the loop
    // makes the whole batch complete before the scratch buffers are freed.
    onnxruntime::llm::kernels::weight_only::preprocess_weights_for_mixed_gemm_cuda(
        stream,
        packing_sm,
        dst_e,
        transposed_scratch_ptr,
        permutation_map.get(),
        {static_cast<size_t>(k), static_cast<size_t>(n)},
        quant_type,
        /*synchronize=*/false);
  }

  // Single host-blocking sync after all experts: this guarantees every
  // per-expert transform (and the CPU->GPU staging copy above) is complete, so
  // the transient scratch buffers are safe to free on return.
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
}

// ---------------------------------------------------------------------------
// PrePack helper: Swizzle MXFP block scales for SM120 TMA layout using GPU kernel.
// ---------------------------------------------------------------------------
void QMoE::PrePackSwizzleBlockScales(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                     IAllocatorUniquePtr<void>& packed_buf, bool& is_packed) {
  auto shape = tensor.Shape();
  ORT_ENFORCE(shape.NumDimensions() == 3, "Expected 3D FP4 block scales for WFP4AFP8 native prepack");
  ORT_ENFORCE(tensor.IsDataType<Float8E8M0>(), "Expected Float8E8M0 FP4 block scales for WFP4AFP8 native prepack");

  const int64_t experts = shape[0];
  const int64_t rows = shape[1];
  const int64_t scale_cols = shape[2];
  ORT_ENFORCE(experts > 0 && rows > 0 && scale_cols > 0,
              "FP4 block scales must have positive dimensions, got ", shape.ToString());
  const int64_t rows_padded_i64 = ((rows + 127) / 128) * 128;
  const int64_t cols_padded_i64 = ((scale_cols + 3) / 4) * 4;
  ORT_ENFORCE(experts <= std::numeric_limits<int>::max() && rows <= std::numeric_limits<int>::max() &&
                  scale_cols <= std::numeric_limits<int>::max() &&
                  rows_padded_i64 <= std::numeric_limits<int>::max() &&
                  cols_padded_i64 <= std::numeric_limits<int>::max(),
              "FP4 block-scale dimensions exceed CUDA launch int range, got ", shape.ToString());
  const int rows_padded = static_cast<int>(rows_padded_i64);
  const int cols_padded = static_cast<int>(cols_padded_i64);
  const size_t dst_bytes = SafeInt<size_t>(experts) * SafeInt<size_t>(rows_padded) *
                           SafeInt<size_t>(cols_padded) * sizeof(uint8_t);

  // Ensure input is on GPU
  const void* p_src = tensor.DataRaw();
  IAllocatorUniquePtr<void> temp_src_gpu;
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    temp_src_gpu = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
    CUDA_CALL_THROW(cudaMemcpyAsync(temp_src_gpu.get(), p_src, tensor.SizeInBytes(), cudaMemcpyHostToDevice, stream));
    p_src = temp_src_gpu.get();
  }

  // QMoEBlockScaleInterleaveKernel writes every byte of the output buffer
  // (the (batch, row, col) -> offset map is a bijection over
  // [0, batch_size) x [0, rows_padded) x [0, cols_padded), and padded
  // source positions are written as 0), so no explicit memset is required.
  packed_buf = IAllocator::MakeUniquePtr<void>(alloc, dst_bytes, true);

  int multi_processor_count = 0;
  int device_id = 0;
  CUDA_CALL_THROW(cudaGetDevice(&device_id));
  CUDA_CALL_THROW(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, device_id));

  LaunchQMoEBlockScaleInterleave(
      static_cast<const uint8_t*>(p_src),
      static_cast<uint8_t*>(packed_buf.get()),
      static_cast<int>(experts),
      static_cast<int>(rows),
      static_cast<int>(scale_cols),
      rows_padded,
      cols_padded,
      multi_processor_count,
      stream);

  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
}

void QMoE::PrePackFp4ScalesForTmaWs(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                    IAllocatorUniquePtr<void>& packed_buf, bool& is_packed) {
  auto shape = tensor.Shape();
  ORT_ENFORCE(shape.NumDimensions() == 3, "Expected 3D FP4 block scales for WFP4A16 native prepack");
  ORT_ENFORCE(tensor.IsDataType<Float8E8M0>(), "Expected Float8E8M0 FP4 block scales for WFP4A16 native prepack");

  const int64_t experts = shape[0];
  const int64_t rows = shape[1];
  const int64_t k_blocks = shape[2];
  ORT_ENFORCE(experts > 0 && rows > 0 && k_blocks > 0,
              "FP4 block scales must have positive dimensions, got ", shape.ToString());
  ORT_ENFORCE(experts <= std::numeric_limits<int>::max() && rows <= std::numeric_limits<int>::max() &&
                  k_blocks <= std::numeric_limits<int>::max(),
              "FP4 block-scale dimensions exceed CUDA launch int range, got ", shape.ToString());

  const size_t bytes = tensor.SizeInBytes();
  const void* p_src = tensor.DataRaw();
  IAllocatorUniquePtr<void> temp_src_gpu;
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    temp_src_gpu = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);
    CUDA_CALL_THROW(cudaMemcpyAsync(temp_src_gpu.get(), p_src, bytes, cudaMemcpyHostToDevice, stream));
    p_src = temp_src_gpu.get();
  }

  // LaunchQMoEPackFp4ScalesForTmaWs pads the k-block count up to a multiple of the packed
  // K-tile (8). Allocate the output buffer for the padded count so the GEMM's last partial
  // CTA-K-tile can read a full packed scale group. The kernel writes every padded element
  // (real blocks copied, tail blocks zeroed), so the buffer is fully initialized.
  constexpr int64_t kPackedScalesPerKTile = 8;
  const int64_t k_blocks_padded =
      ((k_blocks + kPackedScalesPerKTile - 1) / kPackedScalesPerKTile) * kPackedScalesPerKTile;
  const size_t out_bytes =
      SafeInt<size_t>(experts) * SafeInt<size_t>(rows) * SafeInt<size_t>(k_blocks_padded);
  packed_buf = IAllocator::MakeUniquePtr<void>(alloc, out_bytes, true);
  LaunchQMoEPackFp4ScalesForTmaWs(
      static_cast<const uint8_t*>(p_src),
      static_cast<uint8_t*>(packed_buf.get()),
      static_cast<int>(experts),
      static_cast<int>(rows),
      static_cast<int>(k_blocks),
      stream);

  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
}

// ---------------------------------------------------------------------------
// PrePack helper: Repack column-major FP4 weights to row-major using GPU kernel.
// ---------------------------------------------------------------------------
void QMoE::PrePackRepackFP4Weights(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                   IAllocatorUniquePtr<void>& packed_buf, bool& is_packed,
                                   bool gemv_interleaved, bool sm80_pair_interleave) {
  auto shape = tensor.Shape();
  ORT_ENFORCE(shape.NumDimensions() == 3, "Expected 3D FP4 weights for WFP4AFP8 native prepack");
  ORT_ENFORCE(tensor.IsDataType<uint8_t>(), "Expected uint8 FP4 weights for WFP4AFP8 native prepack");

  const int64_t experts = shape[0];
  const int64_t k = shape[1];
  const int64_t n = shape[2] * 2;  // Packed: n/2 bytes per row in source
  ORT_ENFORCE(experts > 0 && k > 0 && n > 0, "FP4 weights must have positive dimensions, got ", shape.ToString());
  ORT_ENFORCE(k % 2 == 0 && n % 2 == 0,
              "FP4 weight repack requires even k and n dimensions, got k=", k, ", n=", n);
  ORT_ENFORCE(experts <= std::numeric_limits<int>::max(),
              "FP4 weight expert count exceeds CUDA launch int range, got ", experts);
  const size_t bytes = tensor.SizeInBytes();

  // Ensure input is on GPU
  const void* p_src = tensor.DataRaw();
  IAllocatorUniquePtr<void> temp_src_gpu;
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    temp_src_gpu = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);
    CUDA_CALL_THROW(cudaMemcpyAsync(temp_src_gpu.get(), p_src, bytes, cudaMemcpyHostToDevice, stream));
    p_src = temp_src_gpu.get();
  }

  packed_buf = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

  if (gemv_interleaved) {
    // Interleaved layout: produce the INT4-style ColumnMajorInterleaved FP4 weight layout instead of the
    // [E, n, k/2] row-major ColToRow layout. The source per expert is [k, n/2] bytes == a
    // [K, N] row-major W4 (e2m1) tensor, which is exactly what the CUTLASS fpA_intB SM80 W4_A16
    // preprocessor consumes (shape {k, n}). The preprocessor's layout-only steps 1-3 (row-
    // permute + subbyte-transpose + column-interleave) apply to e2m1 unchanged; step 4 (the
    // integer +bias + pair-interleave) MUST be skipped (apply_bias_interleave=false) because it
    // would corrupt the floating-point nibbles. Block scales are unchanged (kStepK=32 == MXFP4
    // block size), so TryBuildGemvFp4Scales's [E,k/32,n] layout already maps one scale per
    // column per K-block.
    using onnxruntime::llm::kernels::weight_only::QuantType;
    // The fused FP4 GEMV always runs on the SM80 fpA_intB layout (mixed W4 + fp16/bf16 act is
    // not a Hopper TMA specialisation), so preprocess for SM80 regardless of the runtime device.
    const int packing_sm =
        onnxruntime::llm::kernels::weight_only::get_arch_for_mixed_gemm_weight_preprocess(80);
    const size_t per_expert_bytes = static_cast<size_t>(k) * static_cast<size_t>(n) / 2;
    int8_t* dst_all = reinterpret_cast<int8_t*>(packed_buf.get());
    const int8_t* src_all = reinterpret_cast<const int8_t*>(p_src);

    // The preprocessor mutates its input buffers in place (it ping-pongs between the dst and a
    // writable src across the layout steps), so feed it a writable per-expert scratch copy of
    // the (const) source rather than the source initializer itself. Reused across experts.
    IAllocatorUniquePtr<void> src_scratch = this->GetTransientScratchBuffer<void>(per_expert_bytes);
    int8_t* src_scratch_ptr = reinterpret_cast<int8_t*>(src_scratch.get());
    IAllocatorUniquePtr<int32_t> permutation_map = this->GetTransientScratchBuffer<int32_t>(32);

    for (int64_t e = 0; e < experts; ++e) {
      CUDA_CALL_THROW(cudaMemcpyAsync(src_scratch_ptr, src_all + static_cast<size_t>(e) * per_expert_bytes,
                                      per_expert_bytes, cudaMemcpyDeviceToDevice, stream));
      // synchronize=false: one host-blocking sync after the loop (stream ordering guarantees
      // expert e finishes before e+1 reuses src_scratch). apply_bias_interleave=false for e2m1.
      // When this buffer feeds the SM80 MoE grouped GEMM (sm80_pair_interleave), additionally
      // apply step 4's nibble pair-interleave WITHOUT the +8 bias, which is the layout that
      // GEMM's e2m1 dequant converter inverts. The fused GEMV decode kernel consumes the plain
      // steps-1-3 layout, so it packs its own copy with sm80_pair_interleave=false.
      onnxruntime::llm::kernels::weight_only::preprocess_weights_for_mixed_gemm_cuda(
          stream,
          packing_sm,
          dst_all + static_cast<size_t>(e) * per_expert_bytes,
          src_scratch_ptr,
          permutation_map.get(),
          {static_cast<size_t>(k), static_cast<size_t>(n)},
          QuantType::W4_A16,
          /*synchronize=*/false,
          /*apply_bias_interleave=*/false,
          /*interleave_without_bias=*/sm80_pair_interleave);
    }
    CUDA_CALL_THROW(cudaStreamSynchronize(stream));
    is_packed = true;
    return;
  }

  LaunchQMoERepackFP4ColToRow(
      static_cast<const uint8_t*>(p_src),
      static_cast<uint8_t*>(packed_buf.get()),
      static_cast<int>(experts),
      k, n, stream);

  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
}

void QMoE::TryBuildGemvFp4Scales(int fc, cudaStream_t stream, AllocatorPtr alloc) {
  const bool is_nvfp4 = (quant_type_ == "nvfp4");
  if (!enable_fp4_gemv_ || (quant_type_ != "fp4" && !is_nvfp4)) {
    return;
  }
  IAllocatorUniquePtr<void>& block = (fc == 1) ? packed_fp4_fc1_block_scales_ : packed_fp4_fc2_block_scales_;
  IAllocatorUniquePtr<void>& raw_block = (fc == 1) ? gemv_fp4_fc1_block_raw_ : gemv_fp4_fc2_block_raw_;
  // Native CUTLASS swizzles packed_fp4_*_block_scales_; in that regime read the raw e8m0 copy.
  IAllocatorUniquePtr<void>& src_block = (raw_block != nullptr) ? raw_block : block;
  IAllocatorUniquePtr<void>& global = (fc == 1) ? packed_fc1_global_scale_ : packed_fc2_global_scale_;
  IAllocatorUniquePtr<void>& out = (fc == 1) ? gemv_fp4_fc1_scales_ : gemv_fp4_fc2_scales_;
  const int64_t experts = (fc == 1) ? gemv_fp4_fc1_scale_e_ : gemv_fp4_fc2_scale_e_;
  const int64_t n = (fc == 1) ? gemv_fp4_fc1_scale_n_ : gemv_fp4_fc2_scale_n_;
  const int64_t k_blocks = (fc == 1) ? gemv_fp4_fc1_scale_kb_ : gemv_fp4_fc2_scale_kb_;
  const int64_t global_scale_e = (fc == 1) ? fp4_fc1_global_scale_e_ : fp4_fc2_global_scale_e_;

  // Build exactly once, and only when both inputs are present and dimensions are known.
  if (out != nullptr || src_block == nullptr || global == nullptr || experts <= 0 || n <= 0 || k_blocks <= 0 ||
      global_scale_e != experts) {
    return;
  }

  const size_t element_size = is_fp16_ ? sizeof(half) : sizeof(__nv_bfloat16);
  const size_t bytes = SafeInt<size_t>(experts) * SafeInt<size_t>(n) * SafeInt<size_t>(k_blocks) * element_size;
  out = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

  if (is_fp16_) {
    if (is_nvfp4) {
      LaunchQMoECombineNvfp4ScalesForGemv(
          static_cast<const uint8_t*>(src_block.get()),
          static_cast<const float*>(global.get()),
          static_cast<half*>(out.get()),
          static_cast<int>(experts), static_cast<int>(n), static_cast<int>(k_blocks), stream);
    } else {
      LaunchQMoECombineFp4ScalesForGemv(
          static_cast<const uint8_t*>(src_block.get()),
          static_cast<const float*>(global.get()),
          static_cast<half*>(out.get()),
          static_cast<int>(experts), static_cast<int>(n), static_cast<int>(k_blocks), stream);
    }
  } else {
    if (is_nvfp4) {
      LaunchQMoECombineNvfp4ScalesForGemv(
          static_cast<const uint8_t*>(src_block.get()),
          static_cast<const float*>(global.get()),
          static_cast<__nv_bfloat16*>(out.get()),
          static_cast<int>(experts), static_cast<int>(n), static_cast<int>(k_blocks), stream);
    } else {
      LaunchQMoECombineFp4ScalesForGemv(
          static_cast<const uint8_t*>(src_block.get()),
          static_cast<const float*>(global.get()),
          static_cast<__nv_bfloat16*>(out.get()),
          static_cast<int>(experts), static_cast<int>(n), static_cast<int>(k_blocks), stream);
    }
  }
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
}

// ---------------------------------------------------------------------------
// PrePack helper: Compute bias from zero-points and scales.
// ---------------------------------------------------------------------------
void QMoE::PrePackComputeBias(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                              const IAllocatorUniquePtr<void>& packed_scale,
                              IAllocatorUniquePtr<void>& packed_bias, bool& is_packed) {
  if ((expert_weight_bits_ == 4) && !packed_scale) {
    return;
  }

  size_t num_elements = tensor.Shape().Size();
  auto shape = tensor.Shape();

  if (expert_weight_bits_ == 8) {
    if (block_size_ > 0) {
      bool is_fp16 = is_fp16_;
      bool is_bf16 = !is_fp16_;
      size_t bytes = num_elements * (is_fp16 || is_bf16 ? 2 : 4);
      packed_bias = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

      const void* p_src_zp = tensor.DataRaw();
      IAllocatorUniquePtr<void> temp_zp_gpu;
      if (tensor.Location().device.Type() == OrtDevice::CPU) {
        temp_zp_gpu = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
        CUDA_CALL_THROW(cudaMemcpyAsync(temp_zp_gpu.get(), p_src_zp, tensor.SizeInBytes(), cudaMemcpyDefault, stream));
        p_src_zp = temp_zp_gpu.get();
      }

      const void* p_zp_for_calc = p_src_zp;
      IAllocatorUniquePtr<void> temp_zp_transposed;

      if (shape.NumDimensions() == 3 && shape[2] > 1) {
        size_t rows = shape[1];
        size_t cols = shape[2];
        size_t batch = shape[0];
        temp_zp_transposed = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
        LaunchQMoETranspose2D(static_cast<const uint8_t*>(p_src_zp), static_cast<uint8_t*>(temp_zp_transposed.get()), batch, rows, cols, stream);
        p_zp_for_calc = temp_zp_transposed.get();
      }

      if (is_fp16) {
        LaunchQMoEPrePackOffsetBias(static_cast<const uint8_t*>(p_zp_for_calc), static_cast<const half*>(packed_scale.get()), static_cast<half*>(packed_bias.get()), num_elements, 128.0f, stream);
      } else if (is_bf16) {
        LaunchQMoEPrePackOffsetBias(static_cast<const uint8_t*>(p_zp_for_calc), static_cast<const __nv_bfloat16*>(packed_scale.get()), static_cast<__nv_bfloat16*>(packed_bias.get()), num_elements, 128.0f, stream);
      } else {
        LaunchQMoEPrePackOffsetBias(static_cast<const uint8_t*>(p_zp_for_calc), static_cast<const float*>(packed_scale.get()), static_cast<float*>(packed_bias.get()), num_elements, 128.0f, stream);
      }
    } else {
      size_t bytes = num_elements * sizeof(uint8_t);
      packed_bias = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

      const void* p_src_zp = tensor.DataRaw();
      IAllocatorUniquePtr<void> temp_zp_gpu;
      if (tensor.Location().device.Type() == OrtDevice::CPU) {
        temp_zp_gpu = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
        CUDA_CALL_THROW(cudaMemcpyAsync(temp_zp_gpu.get(), p_src_zp, tensor.SizeInBytes(), cudaMemcpyDefault, stream));
        p_src_zp = temp_zp_gpu.get();
      }

      if (shape.NumDimensions() == 3 && shape[2] > 1) {
        size_t rows = shape[1];
        size_t cols = shape[2];
        size_t batch = shape[0];
        LaunchQMoETranspose2D(static_cast<const uint8_t*>(p_src_zp), static_cast<uint8_t*>(packed_bias.get()), batch, rows, cols, stream);
      } else {
        CUDA_CALL_THROW(cudaMemcpyAsync(packed_bias.get(), p_src_zp, bytes, cudaMemcpyDefault, stream));
      }
    }
  } else {
    if (block_size_ <= 0) {
      return;
    }

    ORT_ENFORCE(shape.NumDimensions() == 3, "Expected 3D zeros for block-wise 4-bit");
    ORT_ENFORCE(shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
                "4-bit block-wise zeros must have positive dimensions, got ", shape.ToString());
    // packed_k_blocks is doubled to k_blocks below; constrain it to half of INT_MAX to keep the
    // doubled value (and the int dims passed into LaunchQMoEScaledZP4BitBatched) within int range.
    constexpr int64_t kMaxPackedKBlocks = std::numeric_limits<int>::max() / 2;
    ORT_ENFORCE(shape[0] <= std::numeric_limits<int>::max() &&
                    shape[1] <= std::numeric_limits<int>::max() &&
                    shape[2] <= kMaxPackedKBlocks,
                "4-bit block-wise zeros dimensions exceed CUDA launch int range, got ", shape.ToString());
    const int experts = static_cast<int>(shape[0]);
    const int n = static_cast<int>(shape[1]);
    const int packed_k_blocks = static_cast<int>(shape[2]);
    const int k_blocks = packed_k_blocks * 2;
    // QMoE only supports FP16/BF16 inputs (is_fp16_ is set in the ctor), both of which are 2 bytes.
    size_t output_count = static_cast<size_t>(experts) * static_cast<size_t>(k_blocks) * static_cast<size_t>(n);
    size_t bytes = output_count * sizeof(uint16_t);
    packed_bias = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

    const void* p_src_zp = tensor.DataRaw();
    IAllocatorUniquePtr<void> temp_zp_gpu;
    if (tensor.Location().device.Type() == OrtDevice::CPU) {
      temp_zp_gpu = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
      CUDA_CALL_THROW(cudaMemcpyAsync(temp_zp_gpu.get(), p_src_zp, tensor.SizeInBytes(), cudaMemcpyDefault, stream));
      p_src_zp = temp_zp_gpu.get();
    }

    const uint8_t* zp_ptr = static_cast<const uint8_t*>(p_src_zp);
    constexpr float kDefaultZeroPoint4Bit = 8.0f;
    if (is_fp16_) {
      LaunchQMoEScaledZP4BitBatched(
          zp_ptr,
          static_cast<const half*>(packed_scale.get()),
          static_cast<half*>(packed_bias.get()),
          experts, n, k_blocks, kDefaultZeroPoint4Bit, stream);
    } else {
      LaunchQMoEScaledZP4BitBatched(
          zp_ptr,
          static_cast<const __nv_bfloat16*>(packed_scale.get()),
          static_cast<__nv_bfloat16*>(packed_bias.get()),
          experts, n, k_blocks, kDefaultZeroPoint4Bit, stream);
    }
  }
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
