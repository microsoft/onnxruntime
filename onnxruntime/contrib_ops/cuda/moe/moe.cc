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
                "Invalid ", kOrtSessionOptionsConfigMoeCpuOffloadExperts,
                " value: ", cpu_offload_experts);
    cpu_offload_enabled_ = cpu_offload_expert_count > 0;
#endif
    cpu_allocator_ = op_kernel_info.GetAllocator(OrtMemTypeCPU);
    cuda_allocator_ = op_kernel_info.GetAllocator(OrtMemTypeDefault);
  }
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

  if (cpu_offload_enabled_) {
    packed.cpu_data = IAllocator::MakeUniquePtr<void>(cpu_allocator_, packed.bytes, true);
    ORT_RETURN_IF_NOT(packed.cpu_data, "Failed to allocate CPU storage for MoE input ", input_idx, ".");
    if (tensor.Location().device.Type() == OrtDevice::CPU) {
      std::memcpy(packed.cpu_data.get(), tensor.DataRaw(), packed.bytes);
    } else {
      CUDA_RETURN_IF_ERROR(cudaMemcpy(packed.cpu_data.get(), tensor.DataRaw(), packed.bytes, cudaMemcpyDeviceToHost));
    }
  } else {
    packed.cuda_data = IAllocator::MakeUniquePtr<void>(cuda_allocator_, packed.bytes, true);
    ORT_RETURN_IF_NOT(packed.cuda_data, "Failed to allocate CUDA storage for MoE input ", input_idx, ".");
    CUDA_RETURN_IF_ERROR(cudaMemcpy(packed.cuda_data.get(), tensor.DataRaw(), packed.bytes, cudaMemcpyDefault));
  }

  is_packed = true;
  return Status::OK();
}

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
  ORT_RETURN_IF(packed_inputs_[6].present,
                "FP16 MoE CPU offload does not yet support separate FC3 weights.");
  ORT_RETURN_IF(activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu &&
                    swiglu_fusion_ == 2,
                "FP16 MoE CPU offload does not support chunked SwiGLU.");

  const auto& fc1_shape = packed_inputs_[2].shape;
  ORT_RETURN_IF_NOT(fc1_shape.NumDimensions() == 3 && fc1_shape[0] > 0,
                    "FP16 MoE FC1 weights must have a positive expert dimension.");
  const size_t num_experts = static_cast<size_t>(fc1_shape[0]);
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
          static_cast<const char*>(packed.cpu_data.get()) +
              static_cast<size_t>(cuda_experts_[index]) * expert_bytes,
          expert_bytes, cudaMemcpyHostToDevice));
    }
  }

  if (!expert_map_.empty()) {
    device_expert_map_ =
        IAllocator::MakeUniquePtr<void>(cuda_allocator_, expert_map_.size() * sizeof(int), true);
    ORT_RETURN_IF_NOT(device_expert_map_, "Failed to allocate the CUDA MoE expert map.");
    CUDA_RETURN_IF_ERROR(cudaMemcpy(device_expert_map_.get(), expert_map_.data(),
                                    expert_map_.size() * sizeof(int), cudaMemcpyHostToDevice));
  }
  return Status::OK();
}

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
  const int workspace_num_experts =
      cpu_offload_enabled_
          ? std::max(static_cast<int>(k_), std::max(1, cuda_runner_num_experts))
          : static_cast<int>(moe_params.num_experts);

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

  size_t ws_size = moe_runner.getWorkspaceSize(
      static_cast<size_t>(moe_params.num_rows), static_cast<size_t>(moe_params.hidden_size),
      static_cast<size_t>(moe_params.inter_size), static_cast<size_t>(workspace_num_experts), static_cast<size_t>(k_),
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
  bool is_bf16 = input->IsDataType<BFloat16>();

  if (use_sparse_mixer_) {
    ORT_RETURN_IF(cpu_offload_enabled_, "FP16 MoE CPU offload does not support sparse_mixer.");
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
    auto* pilot = context->GetKernelPilot();
    ORT_RETURN_IF_NOT(pilot, "FP16 MoE CPU offload requires a KernelPilot.");
    auto& usage = pilot->Moe();
    ORT_RETURN_IF_ERROR(usage.BeginInvocation(static_cast<size_t>(moe_params.num_experts)));

    std::vector<MLFloat16> host_input(
        static_cast<size_t>(SafeInt<int64_t>(moe_params.num_rows) * moe_params.hidden_size));
    std::vector<int> host_expert_indices(expanded_rows);
    std::vector<float> host_expert_scales(expanded_rows);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_input.data(), input->DataRaw(),
                                         host_input.size() * sizeof(MLFloat16),
                                         cudaMemcpyDeviceToHost, stream));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_expert_indices.data(), expert_indices,
                                         indices_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_expert_scales.data(), expert_scales,
                                         scales_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    ORT_RETURN_IF_ERROR(usage.Collect(host_expert_indices));

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

    auto host_weights = [&](int input_idx) {
      const auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
      return gsl::make_span(static_cast<const MLFloat16*>(packed.cpu_data.get()),
                            packed.bytes / sizeof(MLFloat16));
    };
    auto optional_host_weights = [&](int input_idx) -> gsl::span<const MLFloat16> {
      const auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
      return packed.present ? host_weights(input_idx) : gsl::span<const MLFloat16>{};
    };

    std::vector<MLFloat16> host_cpu_output(host_input.size());
    const ::onnxruntime::contrib::MoeCpuOffloadParameters cpu_parameters{
        cpu_activation_type, activation_alpha_, activation_beta_, swiglu_limit_, is_fused_swiglu};
    ORT_RETURN_IF_ERROR(::onnxruntime::contrib::ComputeMoeCpuOffloadedExpertsFp16(
        host_input, host_expert_indices, host_expert_scales, expert_map_,
        host_weights(2), optional_host_weights(3), host_weights(4), optional_host_weights(5),
        moe_params.num_rows, moe_params.hidden_size, moe_params.inter_size, moe_params.num_experts, k_,
        cpu_parameters, host_cpu_output));

    cpu_output_device_buffer =
        GetScratchBuffer<void>(host_cpu_output.size() * sizeof(MLFloat16), stream_obj);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cpu_output_device_buffer.get(), host_cpu_output.data(),
                                         host_cpu_output.size() * sizeof(MLFloat16),
                                         cudaMemcpyHostToDevice, stream));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

    if (!cuda_experts_.empty()) {
      remapped_expert_indices_buffer = GetScratchBuffer<void>(indices_bytes, stream_obj);
      auto* remapped_expert_indices =
          static_cast<int*>(remapped_expert_indices_buffer.get());
      LaunchRemapMoeExpertIndices(expert_indices, remapped_expert_indices,
                                  static_cast<const int*>(device_expert_map_.get()),
                                  expanded_rows, stream);
      runner_expert_indices = remapped_expert_indices;
    } else {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output->MutableDataRaw(), cpu_output_device_buffer.get(),
                                           host_cpu_output.size() * sizeof(MLFloat16),
                                           cudaMemcpyDeviceToDevice, stream));
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

  if (!cpu_offload_enabled_ || !cuda_experts_.empty()) {
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

  if (cpu_offload_enabled_ && !cuda_experts_.empty()) {
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
