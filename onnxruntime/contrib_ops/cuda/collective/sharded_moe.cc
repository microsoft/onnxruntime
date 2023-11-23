// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "sharded_moe.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ShardedMoE,                                                 \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .MayInplace(0, 0)                                       \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ShardedMoE<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
ShardedMoE<T>::ShardedMoE(const OpKernelInfo& op_kernel_info) : NcclKernel(op_kernel_info), MoEBase(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("local_experts_start_index", &local_experts_start_index_).IsOK());
}

template <typename T>
Status ShardedMoE<T>::ComputeInternal(OpKernelContext* context) const {
  const ncclComm_t comm = nccl_->Comm();

  typedef typename ToCudaType<T>::MappedType CudaT;
  auto stream = context->GetComputeStream();

  auto& device_prop = GetDeviceProp();
  const int sm = device_prop.major * 10 + device_prop.minor;

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  // Creating a {Rank, ExpertsStartIndex} map on Host.
  std::vector<int64_t> rank_to_experts_start_index(nccl_->Size());
  IAllocatorUniquePtr<int64_t> experts_start_index_d =
        IAllocator::MakeUniquePtr<int64_t>(allocator, 1, false, stream);
  IAllocatorUniquePtr<int64_t> rank_to_experts_start_index_d =
        IAllocator::MakeUniquePtr<int64_t>(allocator, nccl_->Size(), false, stream);

  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(experts_start_index_d.get(),
                                       &local_experts_start_index_,
                                       sizeof(int64_t),
                                       cudaMemcpyHostToDevice,
                                       Stream(context)));
  NCCL_RETURN_IF_ERROR(ncclAllGather(reinterpret_cast<const char*>(experts_start_index_d.get()),
                                     reinterpret_cast<char*>(rank_to_experts_start_index_d.get()),
                                     1,
                                     ncclInt64,
                                     comm,
                                     Stream(context)));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(rank_to_experts_start_index.data(),
                                       rank_to_experts_start_index_d.get(),
                                       nccl_->Size() * sizeof(int64_t),
                                       cudaMemcpyDeviceToHost,
                                       Stream(context)));

  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(Stream(context)));

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(3);
  const Tensor* fc1_experts_bias_optional = context->Input<Tensor>(4);
  const Tensor* fc2_experts_bias_optional = context->Input<Tensor>(5);

  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(CheckInputs(moe_params, input, router_probs, fc1_experts_weights, fc2_experts_weights,
                                  fc1_experts_bias_optional, fc2_experts_bias_optional));
  ORT_RETURN_IF_NOT(moe_params.num_experts % nccl_->Size() == 0,
                    "num_experts should be divisible by world_size");

  ort_fastertransformer::CutlassMoeFCRunner<CudaT, CudaT> moe_runner(sm);

  size_t ws_size =
      moe_runner.getWorkspaceSize(static_cast<int>(moe_params.num_rows), static_cast<int>(moe_params.hidden_size),
                                  static_cast<int>(moe_params.inter_size), static_cast<int>(moe_params.num_experts),
                                  static_cast<int>(k_));

  size_t fc2_output_size = k_ * moe_params.num_rows * moe_params.hidden_size * sizeof(CudaT);
  size_t expert_scales_size = k_ * moe_params.num_rows * sizeof(CudaT);
  size_t expanded_source_row_to_expanded_dest_row_size = k_ * moe_params.num_rows * sizeof(int);
  size_t expert_for_source_row_size = k_ * moe_params.num_rows * sizeof(int);

  // TODO: allocate one buffer and reuse it.
  IAllocatorUniquePtr<void> work_space = IAllocator::MakeUniquePtr<void>(allocator, ws_size, false, stream);
  IAllocatorUniquePtr<void> fc2_output = IAllocator::MakeUniquePtr<void>(allocator, fc2_output_size, false, stream);
  IAllocatorUniquePtr<void> fc2_output_bc = IAllocator::MakeUniquePtr<void>(allocator, fc2_output_size, false, stream);
  IAllocatorUniquePtr<void> expert_scales =
      IAllocator::MakeUniquePtr<void>(allocator, expert_scales_size, false, stream);
  IAllocatorUniquePtr<void> expanded_source_row_to_expanded_dest_row =
      IAllocator::MakeUniquePtr<void>(allocator, expanded_source_row_to_expanded_dest_row_size, false, stream);
  IAllocatorUniquePtr<void> expert_for_source_row =
      IAllocator::MakeUniquePtr<void>(allocator, expert_for_source_row_size, false, stream);

  // fc1_scales and fc2_scales are used in quantized MoE
  const CudaT* fc1_scales_ptr = nullptr;
  const CudaT* fc2_scales_ptr = nullptr;

  moe_runner.run_moe_fc(reinterpret_cast<const CudaT*>(input->template Data<T>()),
                        reinterpret_cast<const CudaT*>(router_probs->template Data<T>()),
                        reinterpret_cast<const CudaT*>(fc1_experts_weights->template Data<T>()),
                        std::move(fc1_scales_ptr),
                        fc1_experts_bias_optional == nullptr
                            ? nullptr
                            : reinterpret_cast<const CudaT*>(fc1_experts_bias_optional->template Data<T>()),
                        activation_type_, reinterpret_cast<const CudaT*>(fc2_experts_weights->template Data<T>()),
                        std::move(fc2_scales_ptr), static_cast<int>(moe_params.num_rows),
                        static_cast<int>(moe_params.hidden_size),
                        static_cast<int>(moe_params.inter_size), static_cast<int>(moe_params.num_experts),
                        static_cast<int>(moe_params.local_num_experts), static_cast<int>(local_experts_start_index_),
                        static_cast<int>(k_), reinterpret_cast<char*>(work_space.get()),
                        reinterpret_cast<CudaT*>(fc2_output.get()), reinterpret_cast<CudaT*>(expert_scales.get()),
                        reinterpret_cast<int*>(expanded_source_row_to_expanded_dest_row.get()),
                        reinterpret_cast<int*>(expert_for_source_row.get()), Stream(context));

  Tensor* output = context->Output(0, input->Shape());

  size_t base_offset = moe_params.hidden_size * sizeof(CudaT);
  ncclDataType_t dtype = GetNcclDataType(input->DataType());
  // NCCL_RETURN_IF_ERROR(ncclGroupStart()); flacky results with ncclBroadcast + ncclGroupStart/End
  for (int r = 0; r < nccl_->Size(); ++r) {
    int64_t experts_start_index = rank_to_experts_start_index[r];
    int64_t total_past_rows = 0;
    int64_t total_covered_rows = 0;
    moe_runner.get_total_rows_info(experts_start_index, moe_params.local_num_experts, total_past_rows, total_covered_rows);
    std::cout << "rank: " << r << ", experts_start_index: " << experts_start_index << ", total_past_rows: " << total_past_rows << ", total_covered_rows: " << total_covered_rows << std::endl;
    NCCL_RETURN_IF_ERROR(ncclBroadcast(reinterpret_cast<const char*>(fc2_output.get()) + total_past_rows * base_offset,
                                       reinterpret_cast<char*>(fc2_output_bc.get()) + total_past_rows * base_offset,
                                       total_covered_rows * base_offset,
                                       dtype,
                                       r,
                                       comm,
                                       Stream(context)));
  }
  //NCCL_RETURN_IF_ERROR(ncclGroupEnd());

  ort_fastertransformer::finalize_moe_routing_kernelLauncher(
      reinterpret_cast<CudaT*>(fc2_output_bc.get()), reinterpret_cast<CudaT*>(output->template MutableData<T>()),
      fc2_experts_bias_optional == nullptr
          ? nullptr
          : reinterpret_cast<const CudaT*>(fc2_experts_bias_optional->template Data<T>()),
      reinterpret_cast<CudaT*>(expert_scales.get()),
      reinterpret_cast<int*>(expanded_source_row_to_expanded_dest_row.get()),
      reinterpret_cast<int*>(expert_for_source_row.get()), static_cast<int>(moe_params.num_rows),
      static_cast<int>(moe_params.hidden_size), static_cast<int>(k_), Stream(context));

  return Status::OK();
}

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
