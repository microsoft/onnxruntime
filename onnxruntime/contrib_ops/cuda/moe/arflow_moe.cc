// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "arflow_moe.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                    \
      ArflowMoE, kMSDomain, 1, T, kCudaExecutionProvider, \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), ArflowMoE<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

struct ArflowMoEParameters {
  ArflowMoEParameters() {}
  int64_t num_rows;
  int64_t num_experts;
  int64_t in_features;
  int64_t interm_features;
  int64_t out_features;
};

template <typename T>
ArflowMoE<T>::ArflowMoE(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info), MoEBase(op_kernel_info) {
}

template <typename T>
Status ArflowMoE<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0); //(num_rows, in_features)
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2); //(num_experts, in_features, interm_features)
  const Tensor* fc1_experts_bias = context->Input<Tensor>(3); //(num_experts, interm_features)
  const Tensor* fc2_experts_weights = context->Input<Tensor>(4); //(num_experts, interm_features, interm_features)
  const Tensor* fc2_experts_bias = context->Input<Tensor>(5); //(num_experts, interm_features)
  const Tensor* fc3_experts_weights = context->Input<Tensor>(6); //(num_experts, interm_features, interm_features)
  const Tensor* fc3_experts_bias = context->Input<Tensor>(7); //(num_experts, interm_features)
  const Tensor* fc4_experts_weights = context->Input<Tensor>(8); //(num_experts, interm_features, out_features)
  const Tensor* fc4_experts_bias = context->Input<Tensor>(9); //(num_experts, out_features)

  ArflowMoEParameters moe_params;
  moe_params.num_rows = input->Shape()[0];
  moe_params.num_experts = fc1_experts_weights->Shape()[0];
  moe_params.in_features = fc1_experts_weights->Shape()[1];
  moe_params.interm_features = fc1_experts_weights->Shape()[2];
  moe_params.out_features = fc4_experts_weights->Shape()[2];

  typedef typename ToCudaType<T>::MappedType CudaT;
  auto stream = context->GetComputeStream();

  auto& device_prop = GetDeviceProp();
  const int sm = device_prop.major * 10 + device_prop.minor;

  ort_fastertransformer::CutlassMoeFCRunner<CudaT, CudaT> moe_runner(sm, true, // use fc3_bufffer as pingpong buffer
                                                                     normalize_routing_weights_);

  size_t ws_size = moe_runner.getWorkspaceSize(
      static_cast<size_t>(moe_params.num_rows), static_cast<size_t>(moe_params.in_features),
      static_cast<size_t>(moe_params.interm_features), static_cast<size_t>(moe_params.num_experts), static_cast<size_t>(k_));
  size_t fc2_output_size = k_ * moe_params.num_rows * moe_params.out_features * sizeof(CudaT);
  size_t expert_scales_size = k_ * moe_params.num_rows * sizeof(CudaT);
  size_t expanded_source_row_to_expanded_dest_row_size = k_ * moe_params.num_rows * sizeof(int);
  size_t expert_for_source_row_size = k_ * moe_params.num_rows * sizeof(int);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  // TODO: allocate one buffer and reuse it.
  IAllocatorUniquePtr<void> work_space = IAllocator::MakeUniquePtr<void>(allocator, ws_size, false, stream);
  IAllocatorUniquePtr<void> fc2_output = IAllocator::MakeUniquePtr<void>(allocator, fc2_output_size, false, stream);
  IAllocatorUniquePtr<void> expert_scales =
      IAllocator::MakeUniquePtr<void>(allocator, expert_scales_size, false, stream);
  IAllocatorUniquePtr<void> expanded_source_row_to_expanded_dest_row =
      IAllocator::MakeUniquePtr<void>(allocator, expanded_source_row_to_expanded_dest_row_size, false, stream);
  IAllocatorUniquePtr<void> expert_for_source_row =
      IAllocator::MakeUniquePtr<void>(allocator, expert_for_source_row_size, false, stream);

  moe_runner.run_moe_fc_arflow(
      reinterpret_cast<const CudaT*>(input->template Data<T>()),
      reinterpret_cast<const CudaT*>(router_probs->template Data<T>()),
      reinterpret_cast<const CudaT*>(fc1_experts_weights->DataRaw()),
      reinterpret_cast<const CudaT*>(fc1_experts_bias->template Data<T>()),
      reinterpret_cast<const CudaT*>(fc2_experts_weights->DataRaw()),
      reinterpret_cast<const CudaT*>(fc2_experts_bias->template Data<T>()),
      reinterpret_cast<const CudaT*>(fc3_experts_weights->DataRaw()),
      reinterpret_cast<const CudaT*>(fc3_experts_bias->template Data<T>()),
      reinterpret_cast<const CudaT*>(fc4_experts_weights->DataRaw()),
      static_cast<int>(moe_params.num_rows), static_cast<int>(moe_params.in_features),
      static_cast<int>(moe_params.interm_features), static_cast<int>(moe_params.out_features),
      static_cast<int>(moe_params.num_experts),
      static_cast<int>(moe_params.num_experts), 0 /*local_experts_start_index_ used in sharded MoE*/,
      static_cast<int>(k_), reinterpret_cast<char*>(work_space.get()), reinterpret_cast<CudaT*>(fc2_output.get()),
      static_cast<int>(moe_params.num_rows), // active rows
      reinterpret_cast<CudaT*>(expert_scales.get()),
      reinterpret_cast<int*>(expanded_source_row_to_expanded_dest_row.get()),
      reinterpret_cast<int*>(expert_for_source_row.get()), Stream(context));

  TensorShape output_shape({moe_params.num_rows, moe_params.out_features});
  Tensor* output = context->Output(0, output_shape);

  ort_fastertransformer::finalize_moe_routing_kernelLauncher(
      reinterpret_cast<CudaT*>(fc2_output.get()), reinterpret_cast<CudaT*>(output->template MutableData<T>()),
      reinterpret_cast<const CudaT*>(fc4_experts_bias->template Data<T>()),
      reinterpret_cast<CudaT*>(expert_scales.get()),
      reinterpret_cast<int*>(expanded_source_row_to_expanded_dest_row.get()),
      reinterpret_cast<int*>(expert_for_source_row.get()), static_cast<int>(moe_params.num_rows),
      static_cast<int>(moe_params.out_features), static_cast<int>(k_), Stream(context));

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
