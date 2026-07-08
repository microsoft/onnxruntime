// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "contrib_ops/cuda/moe/moe.h"
#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "contrib_ops/cuda/llm/common/env_utils.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"

#include <mutex>

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
void LogSwigluFusionRemapOnce() {
  static std::once_flag log_warning;
  std::call_once(log_warning, []() {
    LOGS_DEFAULT(WARNING) << "MoE swiglu_fusion is 0 with no fc3_experts_weights; assuming interleaved "
                             "SwiGLU layout for backward compatibility.";
  });
}
}  // namespace

#define REGISTER_KERNEL_TYPED(T)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                    \
      MoE, kMSDomain, 1, T, kCudaExecutionProvider, \
      (*KernelDefBuilder::Create()).MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), MoE<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
MoE<T>::MoE(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info), MoEBase(op_kernel_info, GetDeviceProp()) {
}

template <typename T>
Status MoE<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc1_experts_bias_optional = context->Input<Tensor>(3);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(4);
  const Tensor* fc2_experts_bias_optional = context->Input<Tensor>(5);
  const Tensor* fc3_experts_weights_optional = context->Input<Tensor>(6);
  const Tensor* fc3_experts_bias_optional = context->Input<Tensor>(7);

  using onnxruntime::llm::kernels::cutlass_kernels::ActivationType;

  // Backward compatibility: the published gpt-oss-20b model (and any model exported by ORT < 1.27)
  // hard-coded the interleaved SwiGLU fusion layout and did not emit a swiglu_fusion attribute, so it
  // falls back to the default of 0 ("not fused"). When the activation is SwiGLU, swiglu_fusion is 0,
  // and there is no separate FC3 weight, the gate and value projections are actually pre-fused into FC1
  // (interleaved layout). Treat this as swiglu_fusion == 1 so those legacy models keep working.
  int swiglu_fusion = swiglu_fusion_;
  if (activation_type_ == ActivationType::Swiglu && swiglu_fusion == 0 &&
      fc3_experts_weights_optional == nullptr) {
    swiglu_fusion = 1;
    LogSwigluFusionRemapOnce();
  }

  bool is_fused_swiglu = (activation_type_ == ActivationType::Swiglu) &&
                         (swiglu_fusion != 0) &&
                         (fc3_experts_weights_optional == nullptr);

  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights, fc1_experts_bias_optional, nullptr, nullptr,
      fc2_experts_weights, fc2_experts_bias_optional, nullptr, nullptr,
      fc3_experts_weights_optional, fc3_experts_bias_optional, nullptr, nullptr,
      1,  //  no quantization so pack size is 1
      is_fused_swiglu,
      0));  // no block-wise quantization for regular MoE
  ORT_RETURN_IF_NOT(k_ > 0 && k_ <= moe_params.num_experts,
                    "MoE requires 0 < k <= num_experts, got k=", k_,
                    " and num_experts=", moe_params.num_experts);

  using CudaT = typename OrtToCudaType<T>::type;

  void* stream_obj = GetComputeStream(context);
  cudaStream_t stream = Stream(context);

  auto& device_prop = GetDeviceProp();
  int sm = device_prop.major * 10 + device_prop.minor;

  // SM90 TMA WS kernels only support f16/bf16, not float32.
  // Force SM80 path for float32 to use legacy kernels.
  if constexpr (std::is_same_v<T, float>) {
    if (sm >= 90) {
      sm = 80;
    }
  }

  // Validate minimum dimensions for CUTLASS kernels.
  // SM >= 90 TMA WarpSpecialized: smallest tile is 128x16x128B (N=16 for FP16). K < tile_K handled by TMA.
  // SM < 90 Ampere GemmGrouped: smallest instantiated tile N=128, but CUTLASS predicates N < tile_N.
  // Alignment of dimensions to 128 bits is enforced separately in moe_kernels.cu.
  {
    constexpr int min_dim = 16;
    ORT_RETURN_IF(moe_params.hidden_size < min_dim,
                  "MoE CUDA kernel requires hidden_size >= ", min_dim,
                  " for SM", sm, ", got ", moe_params.hidden_size);
    ORT_RETURN_IF(moe_params.inter_size < min_dim,
                  "MoE CUDA kernel requires inter_size >= ", min_dim,
                  " for SM", sm, ", got ", moe_params.inter_size);
  }

  using onnxruntime::llm::kernels::cutlass_kernels::ActivationType;
  ActivationType kernel_activation_type = activation_type_;
  if (activation_type_ == ActivationType::Silu && fc3_experts_weights_optional != nullptr) {
    // Mixtral case: SiLU activation with separate FC3.
    // Kernel supports SwiGLU which is Linear * SiLU(Gate).
    // We map Mixtral to SwiGLU by packing weights as [FC3, FC1] (Linear, Gate).
    kernel_activation_type = ActivationType::Swiglu;
  }

  onnxruntime::llm::kernels::cutlass_kernels::CutlassMoeFCRunner<CudaT, CudaT> moe_runner(sm,
                                                                                          kernel_activation_type,
                                                                                          normalize_routing_weights_,
                                                                                          use_sparse_mixer_);

  constexpr bool use_awq = false;
  onnxruntime::llm::kernels::cutlass_kernels::MOEParallelismConfig parallelism_config{};

  if (onnxruntime::llm::common::getEnvForceDeterministicMOE()) {
    auto tactics = moe_runner.getTactics();
    if (!tactics.empty()) {
      moe_runner.setTactic(tactics[0], tactics[0]);
    }
  } else {
    std::lock_guard<std::mutex> profiler_lock(mGemmProfilerMutex);
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    mGemmProfiler.setAllocator(std::move(allocator));
    mGemmProfiler.setProfilerParams(static_cast<int>(moe_params.num_experts), static_cast<int>(this->k_),
                                    static_cast<int64_t>(moe_params.hidden_size), static_cast<int64_t>(moe_params.inter_size),
                                    static_cast<int64_t>(this->block_size_), kernel_activation_type,
                                    false, true, parallelism_config, sm);

    // Profiling launches grouped-GEMM kernels, records/synchronizes CUDA events, and
    // allocates/frees scratch from the temp allocator on the compute stream. All of these are
    // illegal while that stream is being captured into a CUDA graph; performing them corrupts the
    // capture. During capture we therefore skip profiling and reuse a config cached from an earlier
    // non-capturing run, falling back to the default tactic when nothing is cached.
    const bool stream_is_capturing = onnxruntime::llm::common::isCapturing(stream);

    onnxruntime::llm::nvinfer::DataType dtype = onnxruntime::llm::nvinfer::DataType::kFLOAT;
    if constexpr (std::is_same_v<CudaT, half>) {
      dtype = onnxruntime::llm::nvinfer::DataType::kHALF;
    } else if constexpr (std::is_same_v<CudaT, __nv_bfloat16>) {
      dtype = onnxruntime::llm::nvinfer::DataType::kBF16;
    }

    using onnxruntime::llm::kernels::cutlass_kernels::MoeGemmId;
    using onnxruntime::llm::kernels::weight_only::GemmDims;

    // GEMM 1
    MoeGemmId id1(static_cast<int>(moe_params.inter_size), static_cast<int>(moe_params.hidden_size), dtype, MoeGemmId::GemmType::Gemm1);
    if (!stream_is_capturing) {
      // profileTactics caches per (GemmId, M bucket); calling it every forward lets decode
      // (small M) and prefill (large M) each profile and select their own best tile shape.
      GemmDims dims(static_cast<int64_t>(moe_params.num_rows), static_cast<int64_t>(moe_params.num_rows),
                    static_cast<int64_t>(moe_params.inter_size), static_cast<int64_t>(moe_params.hidden_size));
      mGemmProfiler.profileTactics(&moe_runner, dims, id1, stream);
    }
    auto config1 = mGemmProfiler.getBestConfig(static_cast<int>(moe_params.num_rows), id1);

    // GEMM 2
    MoeGemmId id2(static_cast<int>(moe_params.hidden_size), static_cast<int>(moe_params.inter_size), dtype, MoeGemmId::GemmType::Gemm2);
    if (!stream_is_capturing) {
      GemmDims dims(static_cast<int64_t>(moe_params.num_rows), static_cast<int64_t>(moe_params.num_rows),
                    static_cast<int64_t>(moe_params.hidden_size), static_cast<int64_t>(moe_params.inter_size));
      mGemmProfiler.profileTactics(&moe_runner, dims, id2, stream);
    }
    auto config2 = mGemmProfiler.getBestConfig(static_cast<int>(moe_params.num_rows), id2);

    // Capture-safe fallback: if profiling was skipped (graph capture) and no tuned config was
    // cached from a prior non-capturing run, use the runner's default tactic instead of leaving
    // the config unset.
    if (!config1 || !config2) {
      auto tactics = moe_runner.getTactics();
      if (!tactics.empty()) {
        if (!config1) {
          config1 = tactics[0];
        }
        if (!config2) {
          config2 = tactics[0];
        }
      }
    }

    moe_runner.setTactic(config1, config2);
  }

  size_t ws_size = moe_runner.getWorkspaceSize(
      static_cast<size_t>(moe_params.num_rows), static_cast<size_t>(moe_params.hidden_size),
      static_cast<size_t>(moe_params.inter_size), static_cast<size_t>(moe_params.num_experts), static_cast<size_t>(k_),
      kernel_activation_type, parallelism_config, use_awq);

  // Scratch buffer for workspace + expert_scales + expert_indices + permutation_map.
  // Use checked arithmetic: these byte counts derive adjacent pointer offsets inside one allocation.
  size_t expanded_rows = SafeInt<size_t>(moe_params.num_rows) * SafeInt<size_t>(k_);
  size_t scales_bytes = expanded_rows * sizeof(float);
  size_t indices_bytes = expanded_rows * sizeof(int);
  size_t permutation_bytes = expanded_rows * sizeof(int);
  size_t total_scratch_bytes = SafeInt<size_t>(ws_size) + scales_bytes + indices_bytes + permutation_bytes;

  auto work_space = GetScratchBuffer<void>(total_scratch_bytes, stream_obj);
  char* workspace_ptr = reinterpret_cast<char*>(work_space.get());
  float* expert_scales = reinterpret_cast<float*>(workspace_ptr + ws_size);
  int* expert_indices = reinterpret_cast<int*>(workspace_ptr + ws_size + scales_bytes);
  int* unpermuted_row_to_permuted_row = reinterpret_cast<int*>(workspace_ptr + ws_size + scales_bytes + indices_bytes);

  // Perform Softmax + TopK
  bool is_fp16 = input->IsDataType<MLFloat16>();

  if (use_sparse_mixer_) {
    ORT_ENFORCE(k_ == 2, "Sparse mixer only supports k=2");
    ORT_ENFORCE(moe_params.num_experts == 8 || moe_params.num_experts == 16,
                "Sparse mixer only supports 8 or 16 experts, got ", moe_params.num_experts);

    if (is_fp16) {
      LaunchSparseMixerTop2(
          reinterpret_cast<const half*>(router_probs->DataRaw()),
          expert_scales,
          expert_indices,
          unpermuted_row_to_permuted_row,  // source_rows
          static_cast<int>(moe_params.num_rows),
          static_cast<int>(moe_params.num_experts),
          stream);
    } else {
      LaunchSparseMixerTop2(
          reinterpret_cast<const float*>(router_probs->DataRaw()),
          expert_scales,
          expert_indices,
          unpermuted_row_to_permuted_row,
          static_cast<int>(moe_params.num_rows),
          static_cast<int>(moe_params.num_experts),
          stream);
    }
  } else {
    // Standard Softmax + TopK
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
    } else {
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
  }

  Tensor* output = context->Output(0, input->Shape());

  onnxruntime::llm::kernels::cutlass_kernels::QuantParams quant_params{};

  // =============================================================================
  // WEIGHT PACKING
  // =============================================================================
  // Prepare buffers for CutlassMoeFCRunner.
  // For standard MoE, we copy weights directly.
  // For SwiGLU with separate gates (e.g. Mixtral), we interleave FC1 and FC3 weights.
  // =============================================================================

  // Calculate buffer sizes
  size_t fc1_block_size = static_cast<size_t>(moe_params.inter_size) * static_cast<size_t>(moe_params.hidden_size);
  int E = static_cast<int>(moe_params.num_experts);

  // FC1 Handling
  const CudaT* fc1_input_ptr = reinterpret_cast<const CudaT*>(fc1_experts_weights->DataRaw());
  const CudaT* fc1_processed_ptr = fc1_input_ptr;
  IAllocatorUniquePtr<void> fc1_processed_buffer;

  // Detect fused SwiGLU weights: swiglu_fusion_ != 0 indicates FC1 contains pre-fused gate+value weights
  // When fused, FC1 has shape [E, 2*I, H] instead of [E, I, H] and FC3 is not provided
  // Must also check activation_type is Swiglu to avoid false positives for other activations

  if (fc3_experts_weights_optional != nullptr) {
    // Gated activation with separate FC1 and FC3 weights (e.g., Mixtral's silu + FC3)
    // Kernel expects weights in shape [E, 2*I, H] for gated activation GEMM.
    // Each expert should have FC1_weights and FC3_weights horizontally stacked:
    //   Buffer layout: [Expert0: FC1|FC3][Expert1: FC1|FC3]...
    //   Each expert has 2*I*H elements = 2 * fc1_block_size
    const CudaT* fc3_input_ptr = reinterpret_cast<const CudaT*>(fc3_experts_weights_optional->DataRaw());
    size_t fc1_total_size = E * 2 * fc1_block_size * sizeof(CudaT);
    fc1_processed_buffer = GetScratchBuffer<void>(fc1_total_size, stream_obj);
    CudaT* fc1_fc3_processed_ptr = reinterpret_cast<CudaT*>(fc1_processed_buffer.get());
    fc1_processed_ptr = fc1_fc3_processed_ptr;

    for (int e = 0; e < E; ++e) {
      // Horizontally stack [FC3 | FC1] within each expert's block to match SwiGLU convention
      // Kernel computes: Linear(1st half) * SiLU(Gate(2nd half))
      // Mixtral wants: FC3 * SiLU(FC1)
      // So: 1st half = FC3 (Linear), 2nd half = FC1 (Gate)
      CudaT* dest_fc1 = fc1_fc3_processed_ptr + e * 2 * fc1_block_size;                   // First half of expert e (Gate/FC1)
      CudaT* dest_fc3 = fc1_fc3_processed_ptr + e * 2 * fc1_block_size + fc1_block_size;  // Second half of expert e (Linear/FC3)

      // Copy [I, H] directly
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dest_fc1, fc1_input_ptr + e * fc1_block_size, fc1_block_size * sizeof(CudaT), cudaMemcpyDeviceToDevice, stream));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dest_fc3, fc3_input_ptr + e * fc1_block_size, fc1_block_size * sizeof(CudaT), cudaMemcpyDeviceToDevice, stream));
    }
  }

  // FC2 Handling
  const CudaT* fc2_input_ptr = reinterpret_cast<const CudaT*>(fc2_experts_weights->DataRaw());
  // Layout matches kernel expectation [H, I]. Use directly.
  const CudaT* fc2_processed_ptr = fc2_input_ptr;

  moe_runner.runMoe(
      reinterpret_cast<const CudaT*>(input->template Data<T>()),
      nullptr,         // input_sf
      expert_indices,  // token_selected_experts
      expert_scales,   // token_final_scales
      fc1_processed_ptr,
      fc1_experts_bias_optional == nullptr ? nullptr : reinterpret_cast<const CudaT*>(fc1_experts_bias_optional->template Data<T>()),
      kernel_activation_type,
      fc2_processed_ptr,
      fc2_experts_bias_optional == nullptr ? nullptr : reinterpret_cast<const CudaT*>(fc2_experts_bias_optional->template Data<T>()),
      quant_params,
      static_cast<int>(moe_params.num_rows), static_cast<int>(moe_params.hidden_size),
      static_cast<int>(moe_params.inter_size), static_cast<int>(moe_params.num_experts),
      static_cast<int>(k_),
      workspace_ptr,
      reinterpret_cast<void*>(output->template MutableData<T>()),
      unpermuted_row_to_permuted_row,
      parallelism_config,
      [&]() {
        onnxruntime::llm::kernels::cutlass_kernels::ActivationParams params(kernel_activation_type);
        params.alpha = activation_alpha_;
        params.beta = activation_beta_;
        params.swiglu_fusion = swiglu_fusion;
        params.limit = swiglu_limit_;
        return params;
      }(),
      stream);

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
