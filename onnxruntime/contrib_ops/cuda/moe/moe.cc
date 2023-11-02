// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "moe.h"
#include "moe_kernel.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MoEBlock,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MoEBlock<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
MoEBlock<T>::MoEBlock(const OpKernelInfo& info) : CudaKernel(info) {
}

template <typename T>
Status MoEBlock<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* gated_output = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(3);
  const Tensor* fc1_experts_bias = context->Input<Tensor>(4);
  const Tensor* fc2_experts_bias = context->Input<Tensor>(5);

  // Shape
  const auto& input_dims = input->Shape().GetDims();
  const auto& fc1_experts_weights_dims = fc1_experts_weights->Shape().GetDims();

  const int64_t num_rows = input_dims[0];
  const int64_t hidden_size = input_dims[1];
  const int64_t num_experts = fc1_experts_weights_dims[0];
  const int64_t inter_size = fc1_experts_weights_dims[2];
  const int64_t k = 1;

  typedef typename ToCudaType<T>::MappedType CudaT;
  auto stream = context->GetComputeStream();

  fastertransformer::CutlassMoeFCRunner<CudaT, CudaT> moe_runner;

  size_t ws_size = moe_runner.getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, k);
  size_t fc2_output_size = k * num_rows * hidden_size * sizeof(CudaT);
  size_t expert_scales_size = k * num_rows * sizeof(CudaT);
  size_t expanded_source_row_to_expanded_dest_row_size = k * num_rows * sizeof(int);
  size_t expert_for_source_row_size = k * num_rows * sizeof(int);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  IAllocatorUniquePtr<void> work_space = IAllocator::MakeUniquePtr<void>(allocator, ws_size, false, stream);
  IAllocatorUniquePtr<void> fc2_output = IAllocator::MakeUniquePtr<void>(allocator, fc2_output_size, false, stream);
  IAllocatorUniquePtr<void> expert_scales = IAllocator::MakeUniquePtr<void>(allocator, expert_scales_size, false, stream);
  IAllocatorUniquePtr<void> expanded_source_row_to_expanded_dest_row = IAllocator::MakeUniquePtr<void>(allocator, expanded_source_row_to_expanded_dest_row_size, false, stream);
  IAllocatorUniquePtr<void> expert_for_source_row = IAllocator::MakeUniquePtr<void>(allocator, expert_for_source_row_size, false, stream);

  const CudaT* fc1_scales_ptr = nullptr;
  const CudaT* fc2_scales_ptr = nullptr;

  // bugbug: use a string to select from different activationType
  moe_runner.run_moe_fc(reinterpret_cast<const CudaT*>(input->template Data<T>()),
                        reinterpret_cast<const CudaT*>(gated_output->template Data<T>()),
                        reinterpret_cast<const CudaT*>(fc1_experts_weights->template Data<T>()),
                        std::move(fc1_scales_ptr),
                        reinterpret_cast<const CudaT*>(fc1_experts_bias->template Data<T>()),
                        fastertransformer::ActivationType::Gelu,
                        reinterpret_cast<const CudaT*>(fc2_experts_weights->template Data<T>()),
                        std::move(fc2_scales_ptr),
                        static_cast<int>(num_rows),
                        static_cast<int>(hidden_size),
                        static_cast<int>(inter_size),
                        static_cast<int>(num_experts),
                        static_cast<int>(k),
                        reinterpret_cast<char*>(work_space.get()),
                        reinterpret_cast<CudaT*>(fc2_output.get()),
                        reinterpret_cast<CudaT*>(expert_scales.get()),
                        reinterpret_cast<int*>(expanded_source_row_to_expanded_dest_row.get()),
                        reinterpret_cast<int*>(expert_for_source_row.get()),
                        Stream(context));

  Tensor* output = context->Output(0, input->Shape());

  // bugbug: support no skip in moe_kernel
  IAllocatorUniquePtr<void> skip_layer = IAllocator::MakeUniquePtr<void>(allocator, num_rows * hidden_size * sizeof(T), false, stream);
  fastertransformer::finalize_moe_routing_kernelLauncher(reinterpret_cast<CudaT*>(fc2_output.get()),
                                                         reinterpret_cast<CudaT*>(output->template MutableData<T>()),
                                                         reinterpret_cast<CudaT*>(skip_layer.get()),
                                                         reinterpret_cast<const CudaT*>(fc2_experts_bias->template Data<T>()),
                                                         reinterpret_cast<CudaT*>(expert_scales.get()),
                                                         reinterpret_cast<int*>(expanded_source_row_to_expanded_dest_row.get()),
                                                         reinterpret_cast<int*>(expert_for_source_row.get()),
                                                         static_cast<int>(num_rows),
                                                         static_cast<int>(hidden_size),
                                                         static_cast<int>(k),
                                                         Stream(context));

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
