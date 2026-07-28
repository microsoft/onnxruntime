// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "sharded_moe.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

#define REGISTER_KERNEL_TYPED(T)                                                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                            \
      ShardedMoE, kMSDomain, 1, T, kCudaExecutionProvider,                                                  \
      (*KernelDefBuilder::Create()).MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ShardedMoE<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
ShardedMoE<T>::ShardedMoE(const OpKernelInfo& op_kernel_info)
    : NcclKernel(op_kernel_info), MoEBase(op_kernel_info, GetDeviceProp()) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("tensor_shards", &tensor_shards_).IsOK());
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("local_experts_start_index", &local_experts_start_index_).IsOK());
}

template <typename T>
Status ShardedMoE<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;
  using onnxruntime::llm::kernels::cutlass_kernels::ActivationType;
  using onnxruntime::llm::kernels::cutlass_kernels::MOEParallelismConfig;

  cudaStream_t stream = Stream(context);
  void* stream_obj = GetComputeStream(context);

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc1_experts_bias_optional = context->Input<Tensor>(3);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(4);
  const Tensor* fc2_experts_bias_optional = context->Input<Tensor>(5);
  const Tensor* fc3_experts_weights_optional = context->Input<Tensor>(6);
  const Tensor* fc3_experts_bias_optional = context->Input<Tensor>(7);

  // Backward compatibility: see MoE::ComputeInternal (moe.cc) for the full rationale. Legacy
  // exports leave swiglu_fusion at its default of 0 with FC1/FC3 pre-fused (interleaved).
  int swiglu_fusion = swiglu_fusion_;
  if (activation_type_ == ActivationType::Swiglu && swiglu_fusion == 0 &&
      fc3_experts_weights_optional == nullptr) {
    swiglu_fusion = 1;
  }

  bool is_fused_swiglu = (activation_type_ == ActivationType::Swiglu) &&
                         (swiglu_fusion != 0) &&
                         (fc3_experts_weights_optional == nullptr);

  MoEParameters moe_params(tensor_shards_);
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights, fc1_experts_bias_optional, nullptr, nullptr,
      fc2_experts_weights, fc2_experts_bias_optional, nullptr, nullptr,
      fc3_experts_weights_optional, fc3_experts_bias_optional, nullptr, nullptr,
      1,  // no quantization so pack size is 1
      is_fused_swiglu,
      0));  // no block-wise quantization for sharded MoE

  MOEParallelismConfig parallelism_config;
  if (moe_params.parallel_type == MoEParallelType::TP) {
    ORT_ENFORCE(moe_params.tensor_shards == nccl_->Size());
    parallelism_config = MOEParallelismConfig(nccl_->Size(), nccl_->Rank(), /*ep_size*/ 1, /*ep_rank*/ 0);
  } else if (moe_params.parallel_type == MoEParallelType::EP) {
    ORT_RETURN_IF_NOT(moe_params.num_experts % nccl_->Size() == 0, "num_experts should be divisible by world_size");
    int64_t const ep_rank = local_experts_start_index_ / moe_params.local_num_experts;
    ORT_ENFORCE(ep_rank * moe_params.local_num_experts == local_experts_start_index_,
                "ShardedMoE requires experts to be split contiguously and evenly across ranks");
    parallelism_config = MOEParallelismConfig(/*tp_size*/ 1, /*tp_rank*/ 0,
                                              static_cast<int>(nccl_->Size()), static_cast<int>(ep_rank));
  } else if (moe_params.parallel_type == MoEParallelType::EPAndTP) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Expert and Tensor Parallelism is not supported yet");
  }

  auto& device_prop = GetDeviceProp();
  int sm = device_prop.major * 10 + device_prop.minor;
  // SM90 TMA WS kernels only support f16/bf16, not float32. Force SM80 path for float32.
  if constexpr (std::is_same_v<T, float>) {
    if (sm >= 90) {
      sm = 80;
    }
  }

  {
    constexpr int min_dim = 16;
    ORT_RETURN_IF(moe_params.hidden_size < min_dim,
                  "MoE CUDA kernel requires hidden_size >= ", min_dim, " for SM", sm,
                  ", got ", moe_params.hidden_size);
    ORT_RETURN_IF(moe_params.inter_size < min_dim,
                  "MoE CUDA kernel requires inter_size >= ", min_dim, " for SM", sm,
                  ", got ", moe_params.inter_size);
  }

  onnxruntime::llm::kernels::cutlass_kernels::ActivationType kernel_activation_type = activation_type_;
  if (activation_type_ == ActivationType::Silu && fc3_experts_weights_optional != nullptr) {
    // Mixtral case: SiLU activation with separate FC3 mapped onto SwiGLU (Linear * SiLU(Gate)).
    kernel_activation_type = ActivationType::Swiglu;
  }

  onnxruntime::llm::kernels::cutlass_kernels::CutlassMoeFCRunner<CudaT, CudaT> moe_runner(
      sm, kernel_activation_type, normalize_routing_weights_, use_sparse_mixer_);

  size_t ws_size = moe_runner.getWorkspaceSize(
      static_cast<size_t>(moe_params.num_rows), static_cast<size_t>(moe_params.hidden_size),
      static_cast<size_t>(moe_params.inter_size), static_cast<size_t>(moe_params.num_experts),
      static_cast<size_t>(k_), kernel_activation_type, parallelism_config, /*use_awq*/ false);

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

  // Perform Softmax + TopK. router_probs shares type T with input, so BFloat16 needs its own
  // branch: reinterpreting it as float32 (as a naive is_fp16-only check would) reads garbage.
  bool is_fp16 = input->IsDataType<MLFloat16>();
  bool is_bf16 = input->IsDataType<BFloat16>();

  if (use_sparse_mixer_) {
    ORT_ENFORCE(k_ == 2, "Sparse mixer only supports k=2");
    ORT_ENFORCE(moe_params.num_experts == 8 || moe_params.num_experts == 16,
                "Sparse mixer only supports 8 or 16 experts, got ", moe_params.num_experts);

    if (is_fp16) {
      LaunchSparseMixerTop2(
          reinterpret_cast<const half*>(router_probs->DataRaw()), expert_scales, expert_indices,
          unpermuted_row_to_permuted_row, static_cast<int>(moe_params.num_rows),
          static_cast<int>(moe_params.num_experts), stream);
    } else if (is_bf16) {
      LaunchSparseMixerTop2(
          reinterpret_cast<const __nv_bfloat16*>(router_probs->DataRaw()), expert_scales, expert_indices,
          unpermuted_row_to_permuted_row, static_cast<int>(moe_params.num_rows),
          static_cast<int>(moe_params.num_experts), stream);
    } else {
      LaunchSparseMixerTop2(
          reinterpret_cast<const float*>(router_probs->DataRaw()), expert_scales, expert_indices,
          unpermuted_row_to_permuted_row, static_cast<int>(moe_params.num_rows),
          static_cast<int>(moe_params.num_experts), stream);
    }
  } else {
    if (is_fp16) {
      LaunchSoftmaxTopK(
          reinterpret_cast<const half*>(router_probs->DataRaw()), expert_scales, expert_indices,
          static_cast<int>(moe_params.num_rows), static_cast<int>(moe_params.num_experts),
          static_cast<int>(k_), normalize_routing_weights_, stream);
    } else if (is_bf16) {
      LaunchSoftmaxTopK(
          reinterpret_cast<const __nv_bfloat16*>(router_probs->DataRaw()), expert_scales, expert_indices,
          static_cast<int>(moe_params.num_rows), static_cast<int>(moe_params.num_experts),
          static_cast<int>(k_), normalize_routing_weights_, stream);
    } else {
      LaunchSoftmaxTopK(
          reinterpret_cast<const float*>(router_probs->DataRaw()), expert_scales, expert_indices,
          static_cast<int>(moe_params.num_rows), static_cast<int>(moe_params.num_experts),
          static_cast<int>(k_), normalize_routing_weights_, stream);
    }
  }

  Tensor* output = context->Output(0, input->Shape());

  // FC1/FC2 weight packing: identical to MoE::ComputeInternal (moe.cc) — the runner expects FC1
  // fused as [E, 2*I, H] when a gated activation ships gate/value as separate FC1/FC3 weights.
  size_t fc1_block_size = static_cast<size_t>(moe_params.inter_size) * static_cast<size_t>(moe_params.hidden_size);
  int E = static_cast<int>(moe_params.num_experts);

  const CudaT* fc1_input_ptr = reinterpret_cast<const CudaT*>(fc1_experts_weights->DataRaw());
  const CudaT* fc1_processed_ptr = fc1_input_ptr;
  IAllocatorUniquePtr<void> fc1_processed_buffer;

  if (fc3_experts_weights_optional != nullptr) {
    const CudaT* fc3_input_ptr = reinterpret_cast<const CudaT*>(fc3_experts_weights_optional->DataRaw());
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

  const CudaT* fc2_processed_ptr = reinterpret_cast<const CudaT*>(fc2_experts_weights->DataRaw());

  // finalizeMoeRoutingKernelLauncher (invoked inside runMoe) already zero-fills tokens that don't
  // route to a rank-local expert (EP) and only applies fc2 bias on tp_rank == 0 (TP), so summing
  // the per-rank finalized output across ranks is equivalent to the pre-finalize reduce+finalize
  // ordering the old ft_moe runner used, without needing a separate collective buffer per rank.
  bool needs_all_reduce = moe_params.parallel_type == MoEParallelType::TP ||
                          moe_params.parallel_type == MoEParallelType::EP;

  IAllocatorUniquePtr<void> local_output_buffer;
  void* final_output_ptr;
  if (needs_all_reduce) {
    size_t output_bytes = static_cast<size_t>(moe_params.num_rows) * static_cast<size_t>(moe_params.hidden_size) * sizeof(CudaT);
    local_output_buffer = GetScratchBuffer<void>(output_bytes, stream_obj);
    final_output_ptr = local_output_buffer.get();
  } else {
    final_output_ptr = output->MutableDataRaw();
  }

  moe_runner.runMoe(
      reinterpret_cast<const CudaT*>(input->template Data<T>()),
      nullptr,  // input_sf
      expert_indices,
      expert_scales,
      fc1_processed_ptr,
      fc1_experts_bias_optional == nullptr
          ? nullptr
          : reinterpret_cast<const CudaT*>(fc1_experts_bias_optional->template Data<T>()),
      kernel_activation_type,
      fc2_processed_ptr,
      fc2_experts_bias_optional == nullptr
          ? nullptr
          : reinterpret_cast<const CudaT*>(fc2_experts_bias_optional->template Data<T>()),
      onnxruntime::llm::kernels::cutlass_kernels::QuantParams{},
      static_cast<int64_t>(moe_params.num_rows), static_cast<int64_t>(moe_params.hidden_size),
      static_cast<int64_t>(moe_params.inter_size), static_cast<int>(moe_params.num_experts),
      static_cast<int>(k_), workspace_ptr, final_output_ptr,
      unpermuted_row_to_permuted_row, parallelism_config,
      [&]() {
        onnxruntime::llm::kernels::cutlass_kernels::ActivationParams params(kernel_activation_type);
        params.alpha = activation_alpha_;
        params.beta = activation_beta_;
        params.swiglu_fusion = swiglu_fusion;
        params.limit = swiglu_limit_;
        return params;
      }(),
      stream);

  if (needs_all_reduce) {
    ORT_RETURN_IF_ERROR(FuncCustomAllReduce(
        nccl_, stream, final_output_ptr, output->MutableDataRaw(),
        static_cast<int64_t>(moe_params.num_rows) * static_cast<int64_t>(moe_params.hidden_size),
        input->DataType(),
        collective::IPCMemoryResourcePack::GetGlobalInstance()));
  }

  return Status::OK();
}

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
