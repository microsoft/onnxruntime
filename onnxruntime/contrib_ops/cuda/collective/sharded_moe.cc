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
Status ShardedMoE<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(3);
  const Tensor* fc1_experts_bias_optional = context->Input<Tensor>(4);
  const Tensor* fc2_experts_bias_optional = context->Input<Tensor>(5);

  const auto& input_dims = input->Shape().GetDims();
  const auto& router_probs_dims = router_probs->Shape().GetDims();
  const auto& fc1_experts_weights_dims = fc1_experts_weights->Shape().GetDims();
  const auto& fc2_experts_weights_dims = fc2_experts_weights->Shape().GetDims();

  const int64_t num_rows = input_dims.size() == 2 ? input_dims[0] : input_dims[0] * input_dims[1];
  const int64_t hidden_size = input_dims[input_dims.size() - 1];
  const int64_t num_experts = fc1_experts_weights_dims[0];
  const int64_t inter_size = fc1_experts_weights_dims[2];

  // TODO: refactor to helper function.
  if (fc1_experts_weights_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_weights_dims must be 3D, got ",
                           fc1_experts_weights_dims.size());
  }
  if (fc2_experts_weights_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc2_experts_weights_dims must be 3D, got ",
                           fc2_experts_weights_dims.size());
  }
  if (fc1_experts_weights_dims[1] != hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "fc1_experts_weights_dims[1] must be equal to hidden_size, got ",
                           fc1_experts_weights_dims[1], " and ", hidden_size);
  }
  if (fc2_experts_weights_dims[1] != inter_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "fc2_experts_weights_dims[1] must be equal to inter_size, got ", fc2_experts_weights_dims[1],
                           " and ", inter_size);
  }
  if (fc1_experts_weights_dims[2] != inter_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "fc1_experts_weights_dims[2] must be equal to inter_size, got ", fc1_experts_weights_dims[2],
                           " and ", inter_size);
  }
  if (fc2_experts_weights_dims[2] != hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "fc2_experts_weights_dims[2] must be equal to hidden_size, got ",
                           fc2_experts_weights_dims[2], " and ", hidden_size);
  }
  if (router_probs_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "router_probs_dims must be 2D, got ",
                           router_probs_dims.size());
  }
  if (router_probs_dims[0] != num_rows) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "router_probs_dims[0] must be equal to num_rows, got ",
                           router_probs_dims[0], " and ", num_rows);
  }
  if (router_probs_dims[1] != num_experts) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "router_probs_dims[1] must be equal to num_experts, got ",
                           router_probs_dims[1], " and ", num_experts);
  }
  if (fc1_experts_bias_optional != nullptr && fc2_experts_bias_optional == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_bias is set but fc2_experts_bias is not set");
  }
  if (fc1_experts_bias_optional == nullptr && fc2_experts_bias_optional != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_bias is not set but fc2_experts_bias is set");
  }
  if (fc1_experts_bias_optional != nullptr && fc2_experts_bias_optional != nullptr) {
    const auto& fc1_experts_bias_dims = fc1_experts_bias_optional->Shape().GetDims();
    const auto& fc2_experts_bias_dims = fc2_experts_bias_optional->Shape().GetDims();
    if (fc1_experts_bias_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_bias_dims must be 2D, got ",
                             fc1_experts_bias_dims.size());
    }
    if (fc2_experts_bias_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc2_experts_bias_dims must be 2D, got ",
                             fc2_experts_bias_dims.size());
    }
    if (fc1_experts_bias_dims[0] != num_experts) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "fc1_experts_bias_dims[0] must be equal to num_experts, got ", fc1_experts_bias_dims[0],
                             " and ", num_experts);
    }
    if (fc2_experts_bias_dims[0] != num_experts) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "fc2_experts_bias_dims[0] must be equal to num_experts, got ", fc2_experts_bias_dims[0],
                             " and ", num_experts);
    }
    if (fc1_experts_bias_dims[1] != inter_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "fc1_experts_bias_dims[1] must be equal to inter_size, got ", fc1_experts_bias_dims[1],
                             " and ", inter_size);
    }
    if (fc2_experts_bias_dims[1] != hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "fc2_experts_bias_dims[1] must be equal to hidden_size, got ", fc2_experts_bias_dims[1],
                             " and ", hidden_size);
    }
  }

  typedef typename ToCudaType<T>::MappedType CudaT;
  auto stream = context->GetComputeStream();

  auto& device_prop = GetDeviceProp();
  const int sm = device_prop.major * 10 + device_prop.minor;

  ort_fastertransformer::CutlassMoeFCRunner<CudaT, CudaT> moe_runner(sm);

  size_t ws_size =
      moe_runner.getWorkspaceSize(static_cast<int>(num_rows), static_cast<int>(hidden_size),
                                  static_cast<int>(inter_size), static_cast<int>(num_experts), static_cast<int>(k_));
  size_t fc2_output_size = k_ * num_rows * hidden_size * sizeof(CudaT);
  size_t expert_scales_size = k_ * num_rows * sizeof(CudaT);
  size_t expanded_source_row_to_expanded_dest_row_size = k_ * num_rows * sizeof(int);
  size_t expert_for_source_row_size = k_ * num_rows * sizeof(int);

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
                        std::move(fc2_scales_ptr), static_cast<int>(num_rows), static_cast<int>(hidden_size),
                        static_cast<int>(inter_size), static_cast<int>(num_experts), static_cast<int>(k_),
                        reinterpret_cast<char*>(work_space.get()), reinterpret_cast<CudaT*>(fc2_output.get()),
                        reinterpret_cast<CudaT*>(expert_scales.get()),
                        reinterpret_cast<int*>(expanded_source_row_to_expanded_dest_row.get()),
                        reinterpret_cast<int*>(expert_for_source_row.get()), Stream(context));

  Tensor* output = context->Output(0, input->Shape());

  ort_fastertransformer::finalize_moe_routing_kernelLauncher(
      reinterpret_cast<CudaT*>(fc2_output.get()), reinterpret_cast<CudaT*>(output->template MutableData<T>()),
      fc2_experts_bias_optional == nullptr
          ? nullptr
          : reinterpret_cast<const CudaT*>(fc2_experts_bias_optional->template Data<T>()),
      reinterpret_cast<CudaT*>(expert_scales.get()),
      reinterpret_cast<int*>(expanded_source_row_to_expanded_dest_row.get()),
      reinterpret_cast<int*>(expert_for_source_row.get()), static_cast<int>(num_rows), static_cast<int>(hidden_size),
      static_cast<int>(k_), Stream(context));

  return Status::OK();
}

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
