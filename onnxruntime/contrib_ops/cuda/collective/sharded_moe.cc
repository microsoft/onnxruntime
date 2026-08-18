// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <utility>

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/transformer_cuda_common.h"
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

template <typename T>
ShardedMoE<T>::ShardedMoE(const OpKernelInfo& op_kernel_info)
    : NcclKernel(op_kernel_info), MoEBase(op_kernel_info, GetDeviceProp()) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("tensor_shards", &tensor_shards_).IsOK());
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("local_experts_start_index", &local_experts_start_index_).IsOK());
  ORT_ENFORCE(tensor_shards_ >= 1, "tensor_shards must be positive, got ", tensor_shards_);
  ORT_ENFORCE(local_experts_start_index_ >= 0,
              "local_experts_start_index must not be negative, got ", local_experts_start_index_);
}

template <typename T>
Status ShardedMoE<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc1_experts_bias_optional = context->Input<Tensor>(3);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(4);
  const Tensor* fc2_experts_bias_optional = context->Input<Tensor>(5);
  const Tensor* fc3_experts_weights_optional = context->Input<Tensor>(6);
  const Tensor* fc3_experts_bias_optional = context->Input<Tensor>(7);

  MoEParameters moe_params(tensor_shards_);
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights, fc1_experts_bias_optional, nullptr, nullptr,
      fc2_experts_weights, fc2_experts_bias_optional, nullptr, nullptr,
      fc3_experts_weights_optional, fc3_experts_bias_optional, nullptr, nullptr,
      1,  // no quantization so pack size is 1
      IsFusedSwiglu(fc3_experts_weights_optional != nullptr),
      0));  // no block-wise quantization for sharded MoE
  ORT_RETURN_IF_NOT(k_ > 0 && k_ <= moe_params.num_experts,
                    "ShardedMoE requires 0 < k <= num_experts, got k=", k_,
                    " and num_experts=", moe_params.num_experts);

  const int world_size = nccl_->Size();

  // The kernel computes the experts (expert parallelism) or the slice of the intermediate dimension
  // (tensor parallelism) that this rank owns, so every rank only produces a partial result that has
  // to be combined with an all-reduce afterwards.
  onnxruntime::llm::kernels::cutlass_kernels::MOEParallelismConfig parallelism_config{};
  switch (moe_params.parallel_type) {
    case MoEParallelType::None:
      break;
    case MoEParallelType::EP: {
      ORT_RETURN_IF_NOT(moe_params.local_num_experts * world_size == moe_params.num_experts,
                        "num_experts (", moe_params.num_experts, ") should be local_num_experts (",
                        moe_params.local_num_experts, ") times world_size (", world_size, ")");
      ORT_RETURN_IF_NOT(local_experts_start_index_ % moe_params.local_num_experts == 0,
                        "local_experts_start_index (", local_experts_start_index_,
                        ") should be a multiple of local_num_experts (", moe_params.local_num_experts, ")");
      const int ep_rank = static_cast<int>(local_experts_start_index_ / moe_params.local_num_experts);
      ORT_RETURN_IF_NOT(ep_rank < world_size,
                        "local_experts_start_index (", local_experts_start_index_, ") is out of range");
      parallelism_config = onnxruntime::llm::kernels::cutlass_kernels::MOEParallelismConfig(
          /*tp_size*/ 1, /*tp_rank*/ 0, /*ep_size*/ world_size, ep_rank);
      break;
    }
    case MoEParallelType::TP: {
      ORT_RETURN_IF_NOT(moe_params.tensor_shards == world_size,
                        "tensor_shards (", moe_params.tensor_shards, ") should be equal to world_size (",
                        world_size, ")");
      parallelism_config = onnxruntime::llm::kernels::cutlass_kernels::MOEParallelismConfig(
          /*tp_size*/ world_size, /*tp_rank*/ nccl_->Rank(), /*ep_size*/ 1, /*ep_rank*/ 0);
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Expert and Tensor Parallelism is not supported yet");
  }

  Tensor* output = context->Output(0, input->Shape());
  auto ort_stream = GetOrtStream(context);

  if (moe_params.parallel_type == MoEParallelType::None) {
    return RunMoe<T>(context, ort_stream.get(), moe_params, parallelism_config, mGemmProfiler, mGemmProfilerMutex,
                     output->MutableDataRaw());
  }

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  auto partial_output = Tensor::Create(input->DataType(), input->Shape(), allocator);
  ORT_RETURN_IF_ERROR(RunMoe<T>(context, ort_stream.get(), moe_params, parallelism_config, mGemmProfiler, mGemmProfilerMutex,
                                partial_output->MutableDataRaw()));

  // Tokens that are not routed to a local expert contribute zero, and the bias of the second GEMM is
  // only applied by rank 0, so summing the partial results of all ranks gives the final output.
  return FuncAllReduce(nccl_->Comm(), Stream(context), partial_output.get(), output);
}

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
