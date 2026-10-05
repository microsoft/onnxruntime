// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/common/string_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "contrib_ops/cpu/moe/moe_cpu_offload.h"
#include "contrib_ops/cuda/moe/moe.h"
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
#include "contrib_ops/cuda/moe/kernel_pilot_moe_expert_selection_cuda.h"
#include "core/framework/kernel_pilot.h"
#endif
#include "contrib_ops/cuda/moe/moe_kernels.h"
#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "contrib_ops/cuda/llm/common/env_utils.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"

#include <cstring>
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
  if constexpr (std::is_same_v<T, MLFloat16>) {
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
    const auto cpu_offload_experts = op_kernel_info.GetConfigOptions().GetConfigOrDefault(
        kOrtSessionOptionsConfigMoeCpuOffloadExperts, "0");
    int64_t cpu_offload_expert_count = -1;
    ORT_ENFORCE(TryParseStringWithClassicLocale(cpu_offload_experts, cpu_offload_expert_count) &&
                    cpu_offload_expert_count >= 0,
                kOrtSessionOptionsConfigMoeCpuOffloadExperts,
                " must be a non-negative integer. Received: \"", cpu_offload_experts, "\".");
    cpu_offload_enabled_ = cpu_offload_expert_count > 0;
    if (cpu_offload_enabled_) {
      CUDA_CALL_THROW(cudaStreamCreateWithFlags(&input_copy_stream_, cudaStreamNonBlocking));
    }
#endif
    cuda_allocator_ = op_kernel_info.GetAllocator(OrtMemTypeDefault);
  }
}

template <typename T>
MoE<T>::~MoE() {
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  if (input_copy_stream_ != nullptr) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamDestroy(input_copy_stream_)));
  }
#endif
}

template <typename T>
Status MoE<T>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr,
                       bool& is_packed, PrePackedWeights* prepacked_weights) {
  is_packed = false;
  ORT_UNUSED_PARAMETER(prepacked_weights);
  if constexpr (!std::is_same_v<T, MLFloat16>) {
    return Status::OK();
  }

  if (!cpu_offload_enabled_ || input_idx < 2 || input_idx > 7) {
    return Status::OK();
  }

  auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
  ORT_RETURN_IF(packed.present, "MoE input ", input_idx, " was prepacked more than once.");
  packed.shape = tensor.Shape();
  packed.bytes = tensor.SizeInBytes();
  packed.present = true;

  ORT_RETURN_IF_NOT(packed.bytes % sizeof(MLFloat16) == 0,
                    "FP16 MoE input ", input_idx, " has an invalid byte size.");
  packed.cpu_data.resize(packed.bytes / sizeof(MLFloat16));
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    std::memcpy(packed.cpu_data.data(), tensor.DataRaw(), packed.bytes);
  } else {
    CUDA_RETURN_IF_ERROR(cudaMemcpy(packed.cpu_data.data(), tensor.DataRaw(), packed.bytes, cudaMemcpyDeviceToHost));
  }

  is_packed = true;
  return Status::OK();
}

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
template <typename T>
Status MoE<T>::InitializeKernelPilot(KernelPilot* pilot) {
  if constexpr (!std::is_same_v<T, MLFloat16>) {
    return Status::OK();
  }
  if (!cpu_offload_enabled_) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(pilot, "FP16 MoE CPU offload requires a KernelPilot.");
  gsl::span<const int> cuda_experts;
  ORT_RETURN_IF_ERROR(pilot->GetMoeCudaExperts(cuda_experts));
  return InitializeCudaExpertWeights(cuda_experts);
}

template <typename T>
Status MoE<T>::InitializeCudaExpertWeights(gsl::span<const int> cuda_experts) {
  if constexpr (!std::is_same_v<T, MLFloat16>) {
    ORT_UNUSED_PARAMETER(cuda_experts);
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(packed_inputs_[2].present && packed_inputs_[4].present,
                    "FP16 MoE CPU offload requires constant FC1 and FC2 weights.");
  for (int input_idx : {3, 5, 6, 7}) {
    const auto& input_defs = Info().node().InputDefs();
    const bool connected = static_cast<size_t>(input_idx) < input_defs.size() &&
                           input_defs[static_cast<size_t>(input_idx)]->Exists();
    ORT_RETURN_IF(connected && !packed_inputs_[static_cast<size_t>(input_idx)].present,
                  "FP16 MoE CPU offload requires optional expert input ", input_idx,
                  " to be constant or absent.");
  }
  ORT_RETURN_IF(packed_inputs_[6].present || packed_inputs_[7].present,
                "FP16 MoE CPU offload does not yet support separate FC3 weights or bias.");
  ORT_RETURN_IF(activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu &&
                    swiglu_fusion_ == 2,
                "FP16 MoE CPU offload does not support chunked SwiGLU.");

  const auto& fc1_shape = packed_inputs_[2].shape;
  const auto& fc2_shape = packed_inputs_[4].shape;
  ORT_RETURN_IF_NOT(fc1_shape.NumDimensions() == 3 && fc2_shape.NumDimensions() == 3 &&
                        fc1_shape[0] > 0 && fc2_shape[0] == fc1_shape[0],
                    "FP16 MoE FC1 and FC2 weights must be rank 3 with matching positive expert dimensions.");
  const size_t num_experts = static_cast<size_t>(fc1_shape[0]);
  const auto& input_defs = Info().node().InputDefs();
  const auto* input_shape = input_defs[0]->Shape();
  ORT_RETURN_IF_NOT(input_shape != nullptr && input_shape->dim_size() > 0 &&
                        input_shape->dim(input_shape->dim_size() - 1).has_dim_value(),
                    "FP16 MoE CPU offload requires a static input hidden dimension.");
  const int64_t hidden_size = input_shape->dim(input_shape->dim_size() - 1).dim_value();
  const int64_t fc2_elements_per_expert = fc2_shape.SizeFromDimension(1);
  ORT_RETURN_IF_NOT(hidden_size > 0 && fc2_elements_per_expert % hidden_size == 0,
                    "FP16 MoE FC2 weights have an invalid shape for hidden size ", hidden_size, ".");
  const int64_t inter_size = fc2_elements_per_expert / hidden_size;
  const bool is_fused_swiglu =
      activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu &&
      swiglu_fusion_ != 2 && !packed_inputs_[6].present;
  const int64_t fc1_output_size = is_fused_swiglu ? 2 * inter_size : inter_size;
  const bool legacy_shape =
      (hidden_size != inter_size && fc2_shape[1] == inter_size) ||
      (hidden_size == inter_size && is_fused_swiglu && fc1_shape[1] == hidden_size);
  const std::array<int64_t, 8> logical_output_sizes{
      0, 0, fc1_output_size, fc1_output_size, hidden_size, hidden_size, 0, 0};
  for (const auto& [bias_idx, weight_idx] : {std::pair{3, 2}, std::pair{5, 4}}) {
    const auto& bias = packed_inputs_[static_cast<size_t>(bias_idx)];
    const auto& weight = packed_inputs_[static_cast<size_t>(weight_idx)];
    ORT_RETURN_IF(bias.present &&
                      (weight.shape.NumDimensions() != 3 ||
                       bias.shape.NumDimensions() != 2 ||
                       bias.shape[0] != static_cast<int64_t>(num_experts) ||
                       bias.shape[1] != logical_output_sizes[static_cast<size_t>(bias_idx)]),
                  "FP16 MoE input ", bias_idx, " has an invalid bias shape.");
  }
  cuda_experts_.assign(cuda_experts.begin(), cuda_experts.end());
  expert_map_.assign(num_experts, -1);
  for (size_t index = 0; index < cuda_experts_.size(); ++index) {
    const int expert = cuda_experts_[index];
    ORT_RETURN_IF(expert < 0 || static_cast<size_t>(expert) >= num_experts,
                  "KernelPilot selected an invalid CUDA MoE expert: ", expert);
    expert_map_[static_cast<size_t>(expert)] = static_cast<int>(index);
  }

  for (int input_idx : {2, 3, 4, 5}) {
    auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
    if (!packed.present) {
      continue;
    }
    ORT_RETURN_IF_NOT(packed.shape.NumDimensions() >= 1 &&
                          packed.shape[0] == static_cast<int64_t>(num_experts) &&
                          packed.bytes % num_experts == 0,
                      "FP16 MoE input ", input_idx, " has an invalid expert-major layout.");
    const size_t expert_bytes = packed.bytes / num_experts;
    const size_t cuda_bytes = SafeInt<size_t>(cuda_experts_.size()) * expert_bytes;
    if (cuda_bytes == 0) {
      continue;
    }
    packed.cuda_data = IAllocator::MakeUniquePtr<void>(cuda_allocator_, cuda_bytes, true);
    ORT_RETURN_IF_NOT(packed.cuda_data, "Failed to allocate CUDA storage for MoE input ", input_idx, ".");
    for (size_t index = 0; index < cuda_experts_.size(); ++index) {
      CUDA_RETURN_IF_ERROR(cudaMemcpy(
          static_cast<char*>(packed.cuda_data.get()) + index * expert_bytes,
          reinterpret_cast<const char*>(packed.cpu_data.data()) +
              static_cast<size_t>(cuda_experts_[index]) * expert_bytes,
          expert_bytes, cudaMemcpyHostToDevice));
    }
  }

  for (int input_idx : {2, 4}) {
    auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
    if (cuda_experts_.size() == num_experts) {
      continue;
    }
    const size_t output_size = static_cast<size_t>(
        logical_output_sizes[static_cast<size_t>(input_idx)]);
    const size_t input_size = static_cast<size_t>(input_idx == 2 ? hidden_size : inter_size);
    const size_t expert_element_count = output_size * input_size;
    ORT_RETURN_IF_NOT(
        packed.shape.SizeFromDimension(1) == static_cast<int64_t>(expert_element_count) &&
            (legacy_shape ? packed.shape[1] == static_cast<int64_t>(input_size)
                          : packed.shape[1] == static_cast<int64_t>(output_size)),
        "FP16 MoE input ", input_idx, " has an invalid ",
        legacy_shape ? "legacy" : "standard", " expert weight shape.");
    packed.cpu_gemm_data.resize(packed.cpu_data.size());
    for (size_t expert = 0; expert < num_experts; ++expert) {
      const MLFloat16* source = packed.cpu_data.data() + expert * expert_element_count;
      MLFloat16* destination = packed.cpu_gemm_data.data() + expert * expert_element_count;
      for (size_t output = 0; output < output_size; ++output) {
        for (size_t input = 0; input < input_size; ++input) {
          destination[input * output_size + output] = source[output * input_size + input];
        }
      }
    }
    std::vector<MLFloat16>{}.swap(packed.cpu_data);
  }

  if (!cuda_experts_.empty()) {
    device_expert_map_ =
        IAllocator::MakeUniquePtr<void>(cuda_allocator_, expert_map_.size() * sizeof(int), true);
    ORT_RETURN_IF_NOT(device_expert_map_, "Failed to allocate the CUDA MoE expert map.");
    CUDA_RETURN_IF_ERROR(cudaMemcpy(device_expert_map_.get(), expert_map_.data(),
                                    expert_map_.size() * sizeof(int), cudaMemcpyHostToDevice));
  }
  return Status::OK();
}
#endif

template <typename T>
Status MoE<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const auto input_if_not_packed = [context, this](int index) {
    return packed_inputs_[index].present ? nullptr : context->Input<Tensor>(index);
  };
  const Tensor* fc1_experts_weights = input_if_not_packed(2);
  const Tensor* fc1_experts_bias_optional = input_if_not_packed(3);
  const Tensor* fc2_experts_weights = input_if_not_packed(4);
  const Tensor* fc2_experts_bias_optional = input_if_not_packed(5);
  const Tensor* fc3_experts_weights_optional = input_if_not_packed(6);
  const Tensor* fc3_experts_bias_optional = input_if_not_packed(7);
  const bool use_packed_fp16_weights =
      std::is_same_v<T, MLFloat16> && packed_inputs_[2].present && packed_inputs_[4].present;

  const TensorShape* fc1_experts_weights_shape =
      use_packed_fp16_weights ? &packed_inputs_[2].shape : &fc1_experts_weights->Shape();
  const TensorShape* fc2_experts_weights_shape =
      use_packed_fp16_weights ? &packed_inputs_[4].shape : &fc2_experts_weights->Shape();
  const TensorShape* fc3_experts_weights_shape =
      use_packed_fp16_weights
          ? (packed_inputs_[6].present ? &packed_inputs_[6].shape : nullptr)
          : (fc3_experts_weights_optional != nullptr ? &fc3_experts_weights_optional->Shape() : nullptr);

  using onnxruntime::llm::kernels::cutlass_kernels::ActivationType;

  // Backward compatibility: the published gpt-oss-20b model (and any model exported by ORT < 1.27)
  // hard-coded the interleaved SwiGLU fusion layout and did not emit a swiglu_fusion attribute, so it
  // falls back to the default of 0 ("not fused"). When the activation is SwiGLU, swiglu_fusion is 0,
  // and there is no separate FC3 weight, the gate and value projections are actually pre-fused into FC1
  // (interleaved layout). Treat this as swiglu_fusion == 1 so those legacy models keep working.
  int swiglu_fusion = swiglu_fusion_;
  if (activation_type_ == ActivationType::Swiglu && swiglu_fusion == 0 &&
      fc3_experts_weights_shape == nullptr) {
    swiglu_fusion = 1;
    LogSwigluFusionRemapOnce();
  }

  bool is_fused_swiglu = (activation_type_ == ActivationType::Swiglu) &&
                         (swiglu_fusion != 0) &&
                         (fc3_experts_weights_shape == nullptr);

  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights_shape, use_packed_fp16_weights ? nullptr : fc1_experts_bias_optional, nullptr, nullptr,
      fc2_experts_weights_shape, use_packed_fp16_weights ? nullptr : fc2_experts_bias_optional, nullptr, nullptr,
      fc3_experts_weights_shape, use_packed_fp16_weights ? nullptr : fc3_experts_bias_optional, nullptr, nullptr,
      1, is_fused_swiglu, 0));
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
  const int cuda_runner_num_experts = static_cast<int>(cuda_experts_.size());
  const bool run_cuda_experts = !cpu_offload_enabled_ || cuda_runner_num_experts > 0;
  const bool run_cpu_experts =
      cpu_offload_enabled_ && cuda_runner_num_experts < static_cast<int>(moe_params.num_experts);
  const int workspace_num_experts =
      cpu_offload_enabled_
          ? std::max(static_cast<int>(k_), cuda_runner_num_experts)
          : static_cast<int>(moe_params.num_experts);

  if (run_cuda_experts) {
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
      mGemmProfiler.setProfilerParams(workspace_num_experts, static_cast<int>(this->k_),
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
  }

  size_t ws_size = run_cuda_experts
                       ? moe_runner.getWorkspaceSize(
                             static_cast<size_t>(moe_params.num_rows), static_cast<size_t>(moe_params.hidden_size),
                             static_cast<size_t>(moe_params.inter_size), static_cast<size_t>(workspace_num_experts),
                             static_cast<size_t>(k_), kernel_activation_type, parallelism_config, use_awq)
                       : 0;

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

  ORT_RETURN_IF(cpu_offload_enabled_ && use_sparse_mixer_,
                "FP16 MoE CPU offload does not support sparse_mixer.");

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  IAllocatorUniquePtr<MLFloat16> host_input;
  IAllocatorUniquePtr<int> host_expert_indices;
  IAllocatorUniquePtr<float> host_expert_scales;
  KernelPilot* moe_pilot = nullptr;
  cudaEvent_t input_ready = nullptr;
  cudaEvent_t input_copy_ready = nullptr;
  cudaEvent_t routing_copy_ready = nullptr;
  bool input_copy_complete = false;
  bool routing_copy_pending = false;
  bool routing_copy_recorded = false;
  bool routing_copy_complete = false;
  auto release_input_copy_events = gsl::finally([&]() {
    if (routing_copy_pending && !routing_copy_complete) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(routing_copy_recorded
                                            ? cudaEventSynchronize(routing_copy_ready)
                                            : cudaStreamSynchronize(stream)));
    }
    if (run_cpu_experts && !input_copy_complete && input_copy_stream_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamSynchronize(input_copy_stream_)));
    }
    if (routing_copy_ready != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaEventDestroy(routing_copy_ready)));
    }
    if (input_copy_ready != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaEventDestroy(input_copy_ready)));
    }
    if (input_ready != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaEventDestroy(input_ready)));
    }
  });
  if (cpu_offload_enabled_) {
    host_expert_indices = AllocateBufferOnCPUPinned<int>(expanded_rows);
    ORT_RETURN_IF_NOT(host_expert_indices,
                      "Failed to allocate a pinned routing buffer for FP16 MoE CPU offload.");
    CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&routing_copy_ready, cudaEventDisableTiming));
    if (run_cpu_experts) {
      const size_t host_element_count =
          static_cast<size_t>(SafeInt<int64_t>(moe_params.num_rows) * moe_params.hidden_size);
      host_input = AllocateBufferOnCPUPinned<MLFloat16>(host_element_count);
      host_expert_scales = AllocateBufferOnCPUPinned<float>(expanded_rows);
      ORT_RETURN_IF_NOT(host_input && host_expert_scales,
                        "Failed to allocate pinned host buffers for FP16 MoE CPU execution.");

      CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming));
      CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&input_copy_ready, cudaEventDisableTiming));
      {
        std::lock_guard<std::mutex> input_copy_lock(input_copy_mutex_);
        CUDA_RETURN_IF_ERROR(cudaEventRecord(input_ready, stream));
        CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(input_copy_stream_, input_ready, 0));
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_input.get(), input->DataRaw(),
                                             input->SizeInBytes(), cudaMemcpyDeviceToHost,
                                             input_copy_stream_));
        CUDA_RETURN_IF_ERROR(cudaEventRecord(input_copy_ready, input_copy_stream_));
      }
    }
  }
#endif

  // Perform Softmax + TopK
  bool is_fp16 = input->IsDataType<MLFloat16>();
  bool is_bf16 = input->IsDataType<BFloat16>();

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
    } else if (is_bf16) {
      LaunchSparseMixerTop2(
          reinterpret_cast<const __nv_bfloat16*>(router_probs->DataRaw()),
          expert_scales,
          expert_indices,
          unpermuted_row_to_permuted_row,
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
  const int* runner_expert_indices = expert_indices;
  IAllocatorUniquePtr<void> remapped_expert_indices_buffer;
  IAllocatorUniquePtr<void> cpu_output_device_buffer;

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  if (cpu_offload_enabled_) {
    ORT_RETURN_IF_NOT(use_packed_fp16_weights && expert_map_.size() == static_cast<size_t>(moe_params.num_experts),
                      "FP16 MoE CPU offload weights were not initialized.");
    moe_pilot = context->GetKernelPilot();
    ORT_RETURN_IF_NOT(moe_pilot, "FP16 MoE CPU offload requires a KernelPilot.");
    ORT_RETURN_IF_ERROR(moe_pilot->Moe().BeginInvocation(static_cast<size_t>(moe_params.num_experts)));

    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_expert_indices.get(), expert_indices,
                                         indices_bytes, cudaMemcpyDeviceToHost, stream));
    routing_copy_pending = true;
    if (run_cpu_experts) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_expert_scales.get(), expert_scales,
                                           scales_bytes, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_RETURN_IF_ERROR(cudaEventRecord(routing_copy_ready, stream));
    routing_copy_recorded = true;

    if (!cuda_experts_.empty()) {
      remapped_expert_indices_buffer = GetScratchBuffer<void>(indices_bytes, stream_obj);
      auto* remapped_expert_indices =
          static_cast<int*>(remapped_expert_indices_buffer.get());
      LaunchRemapMoeExpertIndices(expert_indices, remapped_expert_indices,
                                  static_cast<const int*>(device_expert_map_.get()),
                                  expanded_rows, stream);
      runner_expert_indices = remapped_expert_indices;
    }
  }
#endif

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  if (routing_snapshot_ && !cpu_offload_enabled_) {
    auto* pilot = context->GetKernelPilot();
    ORT_RETURN_IF_NOT(pilot, "MoE expert tracking is enabled but its collector is unavailable.");
    ORT_RETURN_IF_ERROR(routing_snapshot_->BeginInvocation(pilot->Moe(), static_cast<size_t>(moe_params.num_experts)));
    ORT_RETURN_IF_ERROR(routing_snapshot_->Capture(expert_indices, expanded_rows, stream));
  }
#endif

  onnxruntime::llm::kernels::cutlass_kernels::QuantParams quant_params{};

  // =============================================================================
  // WEIGHT PACKING
  // =============================================================================
  // Prepare buffers for CutlassMoeFCRunner.
  // For standard MoE, we copy weights directly.
  // For SwiGLU with separate gates (e.g. Mixtral), we interleave FC1 and FC3 weights.
  // =============================================================================

  if (run_cuda_experts) {
    // Calculate buffer sizes
    size_t fc1_block_size = static_cast<size_t>(moe_params.inter_size) * static_cast<size_t>(moe_params.hidden_size);
    int E = cpu_offload_enabled_ ? cuda_runner_num_experts : static_cast<int>(moe_params.num_experts);

    std::array<IAllocatorUniquePtr<void>, 8> runtime_cuda_inputs;
    const auto packed_cuda_data = [&](int input_idx, const Tensor* tensor) -> const CudaT* {
      if constexpr (std::is_same_v<T, MLFloat16>) {
        const auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
        if (packed.present) {
          return static_cast<const CudaT*>(packed.cuda_data.get());
        }
        if (tensor->Location().device.Type() == OrtDevice::CPU) {
          auto& staged = runtime_cuda_inputs[static_cast<size_t>(input_idx)];
          staged = GetScratchBuffer<void>(tensor->SizeInBytes(), stream_obj);
          CUDA_CALL_THROW(cudaMemcpyAsync(staged.get(), tensor->DataRaw(), tensor->SizeInBytes(),
                                          cudaMemcpyHostToDevice, stream));
          return static_cast<const CudaT*>(staged.get());
        }
      }
      return reinterpret_cast<const CudaT*>(tensor->DataRaw());
    };

    // FC1 Handling
    const CudaT* fc1_input_ptr = packed_cuda_data(2, fc1_experts_weights);
    const CudaT* fc1_processed_ptr = fc1_input_ptr;
    IAllocatorUniquePtr<void> fc1_processed_buffer;

    if (fc3_experts_weights_shape != nullptr) {
      const CudaT* fc3_input_ptr = packed_cuda_data(6, fc3_experts_weights_optional);
      size_t fc1_total_size = E * 2 * fc1_block_size * sizeof(CudaT);
      fc1_processed_buffer = GetScratchBuffer<void>(fc1_total_size, stream_obj);
      CudaT* fc1_fc3_processed_ptr = reinterpret_cast<CudaT*>(fc1_processed_buffer.get());
      fc1_processed_ptr = fc1_fc3_processed_ptr;

      for (int e = 0; e < E; ++e) {
        CudaT* dest_fc1 = fc1_fc3_processed_ptr + e * 2 * fc1_block_size;
        CudaT* dest_fc3 = fc1_fc3_processed_ptr + e * 2 * fc1_block_size + fc1_block_size;
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dest_fc1, fc1_input_ptr + e * fc1_block_size,
                                             fc1_block_size * sizeof(CudaT), cudaMemcpyDeviceToDevice, stream));
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dest_fc3, fc3_input_ptr + e * fc1_block_size,
                                             fc1_block_size * sizeof(CudaT), cudaMemcpyDeviceToDevice, stream));
      }
    }

    const CudaT* fc2_processed_ptr = packed_cuda_data(4, fc2_experts_weights);
    const CudaT* fc1_bias_ptr =
        use_packed_fp16_weights
            ? (packed_inputs_[3].present ? static_cast<const CudaT*>(packed_inputs_[3].cuda_data.get()) : nullptr)
            : (fc1_experts_bias_optional == nullptr ? nullptr : packed_cuda_data(3, fc1_experts_bias_optional));
    const CudaT* fc2_bias_ptr =
        use_packed_fp16_weights
            ? (packed_inputs_[5].present ? static_cast<const CudaT*>(packed_inputs_[5].cuda_data.get()) : nullptr)
            : (fc2_experts_bias_optional == nullptr ? nullptr : packed_cuda_data(5, fc2_experts_bias_optional));

    moe_runner.runMoe(
        reinterpret_cast<const CudaT*>(input->template Data<T>()),
        nullptr,
        runner_expert_indices,
        expert_scales,
        fc1_processed_ptr,
        fc1_bias_ptr,
        kernel_activation_type,
        fc2_processed_ptr,
        fc2_bias_ptr,
        quant_params,
        static_cast<int>(moe_params.num_rows), static_cast<int>(moe_params.hidden_size),
        static_cast<int>(moe_params.inter_size), E,
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
        onnxruntime::llm::kernels::cutlass_kernels::FusedRoutingParams{},
        stream);
  }

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  if (cpu_offload_enabled_) {
    // The CUDA path is already enqueued above, so mixed execution overlaps its kernels with the host GEMMs.
    ORT_RETURN_IF_NOT(moe_pilot, "FP16 MoE CPU offload requires a KernelPilot.");
    CUDA_RETURN_IF_ERROR(cudaEventSynchronize(routing_copy_ready));
    routing_copy_complete = true;
    ORT_RETURN_IF_ERROR(
        moe_pilot->Moe().Collect(gsl::make_span(host_expert_indices.get(), expanded_rows)));

    if (run_cpu_experts) {
      CUDA_RETURN_IF_ERROR(cudaEventSynchronize(input_copy_ready));
      input_copy_complete = true;

      auto cpu_activation_type = ::onnxruntime::contrib::ActivationType::Identity;
      switch (activation_type_) {
        case ActivationType::Relu:
          cpu_activation_type = ::onnxruntime::contrib::ActivationType::Relu;
          break;
        case ActivationType::Gelu:
          cpu_activation_type = ::onnxruntime::contrib::ActivationType::Gelu;
          break;
        case ActivationType::Silu:
          cpu_activation_type = ::onnxruntime::contrib::ActivationType::Silu;
          break;
        case ActivationType::Identity:
          cpu_activation_type = ::onnxruntime::contrib::ActivationType::Identity;
          break;
        case ActivationType::Swiglu:
          cpu_activation_type = ::onnxruntime::contrib::ActivationType::SwiGLU;
          break;
        default:
          return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                                 "Unsupported FP16 MoE CPU-offload activation.");
      }

      auto host_data = [&](int input_idx) {
        return gsl::make_span(packed_inputs_[static_cast<size_t>(input_idx)].cpu_gemm_data);
      };
      auto optional_host_data = [&](int input_idx) -> gsl::span<const MLFloat16> {
        const auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
        return packed.present ? gsl::make_span(packed.cpu_data) : gsl::span<const MLFloat16>{};
      };

      const size_t host_element_count =
          static_cast<size_t>(SafeInt<int64_t>(moe_params.num_rows) * moe_params.hidden_size);
      auto host_cpu_output = AllocateBufferOnCPUPinned<MLFloat16>(host_element_count);
      ORT_RETURN_IF_NOT(host_cpu_output, "Failed to allocate the pinned FP16 MoE CPU output buffer.");
      const ::onnxruntime::contrib::MoeCpuOffloadParameters cpu_parameters{
          cpu_activation_type, activation_alpha_, activation_beta_, swiglu_limit_, is_fused_swiglu};
      ORT_RETURN_IF_ERROR(::onnxruntime::contrib::ComputeMoeCpuOffloadedExpertsFp16(
          gsl::make_span(host_input.get(), host_element_count),
          gsl::make_span(host_expert_indices.get(), expanded_rows),
          gsl::make_span(host_expert_scales.get(), expanded_rows), expert_map_,
          host_data(2), optional_host_data(3), host_data(4), optional_host_data(5),
          moe_params.num_rows, moe_params.hidden_size, moe_params.inter_size, moe_params.num_experts, k_,
          cpu_parameters, gsl::make_span(host_cpu_output.get(), host_element_count),
          context->GetOperatorThreadPool()));

      void* cpu_output_destination = output->MutableDataRaw();
      if (!cuda_experts_.empty()) {
        cpu_output_device_buffer = GetScratchBuffer<void>(input->SizeInBytes(), stream_obj);
        cpu_output_destination = cpu_output_device_buffer.get();
      }
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cpu_output_destination, host_cpu_output.get(),
                                           input->SizeInBytes(), cudaMemcpyHostToDevice, stream));
      AddDeferredReleaseCPUPtr(host_cpu_output.release(), stream_obj);
    }
  }
#endif

  if (run_cpu_experts && !cuda_experts_.empty()) {
    LaunchAddMoeFp16Output(
        reinterpret_cast<half*>(output->MutableDataRaw()),
        static_cast<const half*>(cpu_output_device_buffer.get()),
        static_cast<size_t>(output->Shape().Size()), stream);
  }

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  if (routing_snapshot_ && !cpu_offload_enabled_) {
    ORT_RETURN_IF_ERROR(routing_snapshot_->Consume());
  }
#endif

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
