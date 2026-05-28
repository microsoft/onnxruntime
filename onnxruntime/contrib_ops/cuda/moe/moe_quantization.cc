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

#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cpu/utils/debug_macros.h"

#include <cstring>
#include <limits>
#include <vector>

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

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
                                 DataTypeImpl::GetTensorType<Float8E8M0>()})   \
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
  ORT_ENFORCE(quant_type_ == "int" || quant_type_ == "fp4" || quant_type_ == "fp8" || quant_type_ == "wfp4afp8",
              "quant_type must be 'int', 'fp4', 'fp8', or 'wfp4afp8', but got '", quant_type_, "'");
#if !defined(ENABLE_FP4) || !defined(USE_FP4_QMOE)
  ORT_ENFORCE(quant_type_ != "fp4", "QMoE quant_type='fp4' requires USE_FP4_QMOE with CUDA 12.8 or newer.");
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

  if (quant_type_ == "fp4" || quant_type_ == "fp8" || quant_type_ == "wfp4afp8") {
    if (quant_type_ == "fp4") {
      ORT_ENFORCE(expert_weight_bits_ == 4, "FP4 quantization requires expert_weight_bits=4");
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
      use_fp4_dequant_fallback_ = sm_ < 120;
#else
      use_fp4_dequant_fallback_ = true;
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
    if (quant_type_ == "fp4" && !use_fp4_dequant_fallback_) {
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
      if (is_fp16) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, __nv_fp4_e2m1, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_fp4_e2m1, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
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
      // FP4/WFP4AFP8 dequant fallback or FP8 dequant fallback: use A16 runner
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
  const bool is_fp8 = (quant_type_ == "fp8");
  const bool is_wfp4afp8 = (quant_type_ == "wfp4afp8");
  const bool is_int = (quant_type_ == "int");
  // Modes that consume MXFP4 weight block scales (inputs 3/6) and per-expert global weight scales.
  const bool uses_fp4_weight_scales = is_fp4 || is_wfp4afp8;
  // Modes that consume per-expert FP-format global weight scales (inputs 15/16).
  const bool uses_global_weight_scales = is_fp4 || is_fp8 || is_wfp4afp8;
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc1_scales = (is_int && !packed_fc1_scales_) ? context->Input<Tensor>(3) : nullptr;
  const Tensor* fc1_experts_bias_optional = context->Input<Tensor>(4);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(5);
  const Tensor* fc2_scales = (is_int && !packed_fc2_scales_) ? context->Input<Tensor>(6) : nullptr;
  const Tensor* fc2_experts_bias_optional = context->Input<Tensor>(7);
  // The CUTLASS MoE runner has no separate FC3 GEMM — gate and up projection weights must be
  // pre-concatenated into fc1 with doubled output dimension.
  ORT_ENFORCE(context->Input<Tensor>(8) == nullptr,
              "QMoE in CUDA execution provider does not support separate fc3_experts_weights. "
              "Gate and up projection weights must be pre-concatenated into fc1.");

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

  ORT_RETURN_IF_ERROR(check_weight_type(fc1_experts_weights, "fc1_experts_weights", is_fp8));
  ORT_RETURN_IF_ERROR(check_weight_type(fc2_experts_weights, "fc2_experts_weights", is_fp8));

  // Unified FP4 inputs: block scales in fc*_scales (3/6), global scales in 15/16.
  const Tensor* fp4_fc1_block_scales = (uses_fp4_weight_scales && !packed_fp4_fc1_block_scales_) ? context->Input<Tensor>(3) : nullptr;
  const Tensor* fp4_fc2_block_scales = (uses_fp4_weight_scales && !packed_fp4_fc2_block_scales_) ? context->Input<Tensor>(6) : nullptr;
  const Tensor* fc1_global_scale = (uses_global_weight_scales && !packed_fc1_global_scale_) ? context->Input<Tensor>(15) : nullptr;
  const Tensor* fc2_global_scale = (uses_global_weight_scales && !packed_fc2_global_scale_) ? context->Input<Tensor>(16) : nullptr;

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
  if (block_size_ > 0 && block_size_ < 64 && has_any_zero_point) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "QMoE asymmetric zero_points are currently supported only when block_size >= 64. "
                           "Use block_size >= 64 or remove fc*_zero_points.");
  }

  int64_t pack_size = expert_weight_bits_ == 4 ? 2 : 1;
  bool is_fused_swiglu = activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu;
  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs, fc1_experts_weights,
      fc1_experts_bias_optional, fc1_scales, fc1_zeros,
      fc2_experts_weights, fc2_experts_bias_optional, fc2_scales, fc2_zeros,
      nullptr, nullptr, nullptr, nullptr,
      pack_size, is_fused_swiglu, block_size_));

  if (uses_fp4_weight_scales) {
    constexpr int64_t fp4_block_size = 32;
    const int64_t fc1_out_size = is_fused_swiglu ? moe_params.inter_size * 2 : moe_params.inter_size;
    auto check_fp4_block_scale = [](const Tensor* tensor, const char* name, int64_t num_experts,
                                    int64_t n, int64_t k) -> Status {
      ORT_RETURN_IF_NOT(tensor != nullptr, "QMoE quant_type='fp4'/'wfp4afp8' requires ", name, ".");
      ORT_RETURN_IF_NOT(tensor->IsDataType<Float8E8M0>(), name, " must be a float8e8m0 MXFP block-scale tensor.");
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

  // Profile and capture the best tactics under the profiler mutex, then release the mutex so
  // that scratch allocation, weight dequantization, scale prepping, softmax, and other
  // CPU-bound work can proceed concurrently across QMoE inferences. The mutex is reacquired
  // around setTactic + runMoe because they mutate shared `m_moe_runner` state.
  std::optional<onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler::Config> config1;
  std::optional<onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler::Config> config2;
  size_t workspace_size = 0;
  {
    std::lock_guard<std::mutex> profiler_lock(mGemmProfilerMutex);

    // Use profiler with proper weight type for quantized weights
    if (onnxruntime::llm::common::getEnvForceDeterministicMOE()) {
      auto tactics = m_moe_runner->getTactics();
      if (!tactics.empty()) {
        config1 = tactics[0];
        config2 = tactics[0];
        m_moe_runner->setTactic(config1, config2);
      }
    } else {
      AllocatorPtr allocator;
      ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
      mGemmProfiler.setAllocator(std::move(allocator));
      mGemmProfiler.setProfilerParams(static_cast<int>(moe_params.num_experts), static_cast<int>(k_),
                                      static_cast<int64_t>(moe_params.hidden_size), static_cast<int64_t>(moe_params.inter_size),
                                      static_cast<int64_t>(block_size_), activation_type_,
                                      false, true, parallelism_config, sm_);

      onnxruntime::llm::nvinfer::DataType dtype = is_fp16_ ? onnxruntime::llm::nvinfer::DataType::kHALF : onnxruntime::llm::nvinfer::DataType::kBF16;
      if (is_wfp4afp8 && !use_wfp4afp8_dequant_fallback_) {
        dtype = onnxruntime::llm::nvinfer::DataType::kFP8;
      }
      // Weight type: FP4 for MXFP4, INT4 for 4-bit integer, INT8 for 8-bit integer
      onnxruntime::llm::nvinfer::DataType wtype;
      if (is_fp4) {
        wtype = use_fp4_dequant_fallback_ ? dtype : onnxruntime::llm::nvinfer::DataType::kFP4;
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
      if (mGemmId1 != id1) {
        mGemmId1 = id1;
        GemmDims dims(static_cast<int64_t>(moe_params.num_rows), static_cast<int64_t>(moe_params.num_rows),
                      fc1_out_size, static_cast<int64_t>(moe_params.hidden_size));
        mGemmProfiler.profileTactics(m_moe_runner.get(), dtype, dims, id1);
      }
      config1 = mGemmProfiler.getBestConfig(static_cast<int>(moe_params.num_rows), mGemmId1);

      // GEMM 2
      MoeGemmId id2(static_cast<int>(moe_params.hidden_size), static_cast<int>(moe_params.inter_size), dtype, wtype, MoeGemmId::GemmType::Gemm2);
      if (mGemmId2 != id2) {
        mGemmId2 = id2;
        GemmDims dims(static_cast<int64_t>(moe_params.num_rows), static_cast<int64_t>(moe_params.num_rows),
                      static_cast<int64_t>(moe_params.hidden_size), static_cast<int64_t>(moe_params.inter_size));
        mGemmProfiler.profileTactics(m_moe_runner.get(), dtype, dims, id2);
      }
      config2 = mGemmProfiler.getBestConfig(static_cast<int>(moe_params.num_rows), mGemmId2);

      m_moe_runner->setTactic(config1, config2);
    }

    workspace_size = m_moe_runner->getWorkspaceSize(
        moe_params.num_rows, moe_params.hidden_size, moe_params.inter_size, moe_params.num_experts, k_,
        activation_type_, parallelism_config, use_awq);
  }
  // Lock released — concurrent QMoE inferences can now run prep work in parallel.

  // Scratch buffer for workspace + expert_scales + expert_indices
  // expert_scales: num_rows * k * sizeof(float)
  // expert_indices: num_rows * k * sizeof(int)
  size_t scales_bytes = moe_params.num_rows * k_ * sizeof(float);
  size_t indices_bytes = moe_params.num_rows * k_ * sizeof(int);
  size_t permutation_bytes = moe_params.num_rows * k_ * sizeof(int);
  size_t total_scratch_bytes = workspace_size + scales_bytes + indices_bytes + permutation_bytes;

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
  if (is_fp4) {
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
    if (!use_fp4_dequant_fallback_) {
      using NVFP4ElementSF = onnxruntime::llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF;
      quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::FP4(
          nullptr,  // fc1_act_global_scale (no activation quantization for W4A16)
          static_cast<const NVFP4ElementSF*>(p_fc1_block_scales),
          static_cast<const float*>(p_fc1_global_scale),
          nullptr,  // fc2_act_global_scale
          static_cast<const NVFP4ElementSF*>(p_fc2_block_scales),
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

  const void* fc1_weight_data = fc1_experts_weights->DataRaw();
  const void* fc2_weight_data = fc2_experts_weights->DataRaw();
  if (is_wfp4afp8 && !use_wfp4afp8_dequant_fallback_) {
    fc1_weight_data = packed_fp4_fc1_weights_ ? packed_fp4_fc1_weights_.get() : fc1_weight_data;
    fc2_weight_data = packed_fp4_fc2_weights_ ? packed_fp4_fc2_weights_.get() : fc2_weight_data;
  }
  IAllocatorUniquePtr<void> dequant_fc1_weights;
  IAllocatorUniquePtr<void> dequant_fc2_weights;
  // FP4 (W4A16) and WFP4AFP8 (W4A8) share the MXFP4 weight format. When the native CUTLASS path
  // is unavailable on the current SM, dequantize MXFP4 weights to FP16/BF16 and run the dense A16 runner.
  if ((is_fp4 && use_fp4_dequant_fallback_) || (is_wfp4afp8 && use_wfp4afp8_dequant_fallback_)) {
    const void* p_fc1_block_scales = packed_fp4_fc1_block_scales_ ? packed_fp4_fc1_block_scales_.get()
                                                                  : (fp4_fc1_block_scales ? fp4_fc1_block_scales->DataRaw() : nullptr);
    const void* p_fc1_global_scale = packed_fc1_global_scale_ ? packed_fc1_global_scale_.get()
                                                              : (fc1_global_scale ? fc1_global_scale->DataRaw() : nullptr);
    const void* p_fc2_block_scales = packed_fp4_fc2_block_scales_ ? packed_fp4_fc2_block_scales_.get()
                                                                  : (fp4_fc2_block_scales ? fp4_fc2_block_scales->DataRaw() : nullptr);
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

    if (is_fp16_) {
      LaunchQMoEDequantizeFp4Weights(static_cast<const uint8_t*>(fc1_experts_weights->DataRaw()),
                                     static_cast<const uint8_t*>(p_fc1_block_scales),
                                     static_cast<const float*>(p_fc1_global_scale),
                                     static_cast<half*>(dequant_fc1_weights.get()), num_experts, fc1_n, fc1_k, stream);
      LaunchQMoEDequantizeFp4Weights(static_cast<const uint8_t*>(fc2_experts_weights->DataRaw()),
                                     static_cast<const uint8_t*>(p_fc2_block_scales),
                                     static_cast<const float*>(p_fc2_global_scale),
                                     static_cast<half*>(dequant_fc2_weights.get()), num_experts, fc2_n, fc2_k, stream);
    } else {
      LaunchQMoEDequantizeFp4Weights(static_cast<const uint8_t*>(fc1_experts_weights->DataRaw()),
                                     static_cast<const uint8_t*>(p_fc1_block_scales),
                                     static_cast<const float*>(p_fc1_global_scale),
                                     static_cast<__nv_bfloat16*>(dequant_fc1_weights.get()), num_experts, fc1_n, fc1_k, stream);
      LaunchQMoEDequantizeFp4Weights(static_cast<const uint8_t*>(fc2_experts_weights->DataRaw()),
                                     static_cast<const uint8_t*>(p_fc2_block_scales),
                                     static_cast<const float*>(p_fc2_global_scale),
                                     static_cast<__nv_bfloat16*>(dequant_fc2_weights.get()), num_experts, fc2_n, fc2_k, stream);
    }
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
    m_moe_runner->setTactic(config1, config2);
    m_moe_runner->runMoe(
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
          params.swiglu_fusion = swiglu_fusion_;
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

  if (input_idx == 2 && quant_type_ == "wfp4afp8" && !use_wfp4afp8_dequant_fallback_) {
    PrePackRepackFP4Weights(tensor, stream, alloc, packed_fp4_fc1_weights_, is_packed);
    is_packed = false;
  } else if (input_idx == 5 && quant_type_ == "wfp4afp8" && !use_wfp4afp8_dequant_fallback_) {
    PrePackRepackFP4Weights(tensor, stream, alloc, packed_fp4_fc2_weights_, is_packed);
    is_packed = false;
  } else if (input_idx == 3) {  // fc1_scales
    DUMP_TENSOR("fc1_scales", tensor);
    if (quant_type_ == "wfp4afp8" && !use_wfp4afp8_dequant_fallback_) {
      PrePackSwizzleBlockScales(tensor, stream, alloc, packed_fp4_fc1_block_scales_, is_packed);
    } else if (quant_type_ == "fp4" || quant_type_ == "wfp4afp8") {
      PrePackCopyToGpu(tensor, stream, alloc, packed_fp4_fc1_block_scales_, is_packed);
    } else if (quant_type_ == "int") {
      PrePackTransposeAndPack(tensor, stream, alloc, packed_fc1_scales_, is_packed);
      DUMP_PACK_TENSOR("packed_fc1_scales", packed_fc1_scales_, tensor);
    }
  } else if (input_idx == 6) {  // fc2_scales
    DUMP_TENSOR("fc2_scales", tensor);
    if (quant_type_ == "wfp4afp8" && !use_wfp4afp8_dequant_fallback_) {
      PrePackSwizzleBlockScales(tensor, stream, alloc, packed_fp4_fc2_block_scales_, is_packed);
    } else if (quant_type_ == "fp4" || quant_type_ == "wfp4afp8") {
      PrePackCopyToGpu(tensor, stream, alloc, packed_fp4_fc2_block_scales_, is_packed);
    } else if (quant_type_ == "int") {
      PrePackTransposeAndPack(tensor, stream, alloc, packed_fc2_scales_, is_packed);
      DUMP_PACK_TENSOR("packed_fc2_scales", packed_fc2_scales_, tensor);
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
             (quant_type_ == "fp4" || quant_type_ == "fp8" || quant_type_ == "wfp4afp8")) {
    if (input_idx == 15) {
      PrePackCopyToGpu(tensor, stream, alloc, packed_fc1_global_scale_, is_packed);
    } else {
      PrePackCopyToGpu(tensor, stream, alloc, packed_fc2_global_scale_, is_packed);
    }
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

// ---------------------------------------------------------------------------
// PrePack helper: Repack column-major FP4 weights to row-major using GPU kernel.
// ---------------------------------------------------------------------------
void QMoE::PrePackRepackFP4Weights(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                   IAllocatorUniquePtr<void>& packed_buf, bool& is_packed) {
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

  LaunchQMoERepackFP4ColToRow(
      static_cast<const uint8_t*>(p_src),
      static_cast<uint8_t*>(packed_buf.get()),
      static_cast<int>(experts),
      k, n, stream);

  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
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
