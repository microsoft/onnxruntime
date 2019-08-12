// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "optimizers.h"
#include "core/providers/cuda/reduction/reduction_ops.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    SGDOptimizer,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder()
      .Alias(1, 0) // Update weights in-place
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SGDOptimizer);

Status SGDOptimizer::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& W = *ctx->MutableInput<Tensor>(1);
  const Tensor& G = *ctx->Input<Tensor>(2);
  Tensor& NW = *ctx->Output(0, W.Shape());

  ORT_ENFORCE(W.Shape() == G.Shape());

  SGDOptimizerImpl(
      ETA.template Data<float>(),
      W.template Data<float>(),
      G.template Data<float>(),
      NW.template MutableData<float>(),
      W.Shape().Size());

  return Status::OK();
}

  // TODO: Once Schema is checked in to onnx lets fix this to match that
#define REGISTER_ADAM_KERNEL_TYPED(T1, T2, T3, T4)                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                    \
      AdamOptimizer,                                                \
      kOnnxDomain,                                                  \
      9,                                                            \
      T1##_##T2##_##T3##_##T4,                                      \
      kCudaExecutionProvider,                                       \
      KernelDefBuilder()                                            \
          .Alias(1, 3) /* Update step count in-place */             \
          .Alias(2, 0) /* Update weights in-place */                \
          .Alias(4, 1) /* Update moment-1 in-place */               \
          .Alias(5, 2) /* Update moment-2 in-place */               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())  \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T3>())  \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<T4>()), \
      AdamOptimizer<T1, T2, T3, T4>);

REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float)
REGISTER_ADAM_KERNEL_TYPED(MLFloat16, int64_t, float, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, MLFloat16)

template <typename T1, typename T2, typename T3, typename T4>
Status AdamOptimizer<T1, T2, T3, T4>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T1>::MappedType CudaT1;
  typedef typename ToCudaType<T2>::MappedType CudaT2;
  typedef typename ToCudaType<T3>::MappedType CudaT3;
  typedef typename ToCudaType<T4>::MappedType CudaT4;

  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& S = *ctx->Input<Tensor>(1);
  const Tensor& W = *ctx->Input<Tensor>(2);
  const Tensor& G = *ctx->Input<Tensor>(3);
  const Tensor& M1 = *ctx->Input<Tensor>(4);
  const Tensor& M2 = *ctx->Input<Tensor>(5);

  Tensor& NW = *ctx->Output(0, W.Shape());
  Tensor& NM1 = *ctx->Output(1, M1.Shape());
  Tensor& NM2 = *ctx->Output(2, M2.Shape());
  Tensor& NS = *ctx->Output(3, S.Shape());

  AdamOptimizerImpl(
      reinterpret_cast<const CudaT1*>(ETA.template Data<T1>()),
      reinterpret_cast<const CudaT2*>(S.template Data<T2>()),
      reinterpret_cast<const CudaT3*>(W.template Data<T3>()),
      reinterpret_cast<const CudaT4*>(G.template Data<T4>()),
      reinterpret_cast<const CudaT4*>(M1.template Data<T4>()),
      reinterpret_cast<const CudaT4*>(M2.template Data<T4>()),
      ToCudaType<T4>::FromFloat(alpha_),
      ToCudaType<T4>::FromFloat(beta_),
      ToCudaType<T4>::FromFloat(lambda_),
      ToCudaType<T4>::FromFloat(epsilon_),
      reinterpret_cast<CudaT3*>(NW.template MutableData<T3>()),
      reinterpret_cast<CudaT4*>(NM1.template MutableData<T4>()),
      reinterpret_cast<CudaT4*>(NM2.template MutableData<T4>()),
      reinterpret_cast<CudaT2*>(NS.template MutableData<T2>()),
      W.Shape().Size());

  return Status::OK();
}

// TODO: Once Schema is checked in to onnx lets fix this to match that
ONNX_OPERATOR_KERNEL_EX(
    LambOptimizer,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder()
      .Alias(1, 0) // Allow in-place update to weight.
      .Alias(2, 1) // Allow in-place update to 1st-order momentum.
      .Alias(3, 2) // Allow in-place update to 2nd-order momentum.
      .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),
    LambOptimizer);

Status LambOptimizer::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor &eta_tensor = *ctx->Input<Tensor>(0);
  const Tensor &weights_tensor = *ctx->Input<Tensor>(1);
  const Tensor &gradients_tensor = *ctx->Input<Tensor>(2);
  const Tensor &moment_1_tensor = *ctx->Input<Tensor>(3);
  const Tensor &moment_2_tensor = *ctx->Input<Tensor>(4);

  const TensorShape &weight_tensor_shape = weights_tensor.Shape();
  const size_t weight_tensor_rank =  weight_tensor_shape.NumDimensions();
  const auto weight_tensor_size = weights_tensor.Shape().Size();

  ORT_ENFORCE(weight_tensor_shape == gradients_tensor.Shape());
  ORT_ENFORCE(weight_tensor_shape == moment_1_tensor.Shape());
  ORT_ENFORCE(weight_tensor_shape == moment_2_tensor.Shape());

  // It's an alias of weights_tensor, which leads to an in-place update.
  Tensor &weights_tensor_updated = *ctx->Output(0, weight_tensor_shape);
  Tensor &moment_1_tensor_updated = *ctx->Output(1, weight_tensor_shape);
  Tensor &moment_2_tensor_updated = *ctx->Output(2, weight_tensor_shape);

  // Compute update direction and the 1st and the 2nd momentums.
  IAllocatorUniquePtr<float> update_direction_buffer = GetScratchBuffer<float>(weight_tensor_size);
  LambComputeDirectionImpl(
      weights_tensor.template Data<float>(),
      gradients_tensor.template Data<float>(),
      moment_1_tensor.template Data<float>(),
      moment_2_tensor.template Data<float>(),
      alpha_,
      beta_,
      lambda_,
      epsilon_,
      update_direction_buffer.get(),
      moment_1_tensor_updated.template MutableData<float>(),
      moment_2_tensor_updated.template MutableData<float>(),
      weight_tensor_size);

  // Allocate buffer for reduction computation of update direction.
  IAllocatorUniquePtr<float> direction_norm_buffer = GetScratchBuffer<float>(1);
  // Allocate buffer for reduction computation of weight tensor.
  IAllocatorUniquePtr<float> weights_norm_buffer = GetScratchBuffer<float>(1);

  // Do reduction to compute the 2-norm of the weight tensor and
  // the 2-norm of the update direction. They will be used to
  // adjust the learning rate.
  if (weight_tensor_size > 1)
  {
    const cudnnDataType_t cudnn_reduction_type = CudnnTensor::GetDataType<float>();
    std::vector<int64_t> cudnn_reduction_input_dims = weight_tensor_shape.GetDims();
    std::vector<int64_t> cudnn_reduction_output_dims(weight_tensor_rank, 1);

    if (weight_tensor_rank > 8)
      return ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL, "CUDNN only supports up to 8-D tensors in reduction, \
        so the current LAMB optimizer cannot update tensors with more than 8 axes.");

    // CUDNN's reduction doesn't work with 1-D and 2-D tensors. 
    if (weight_tensor_rank < 3) {
      std::vector<int64_t> pads(3 - weight_tensor_rank, 1);
      cudnn_reduction_input_dims.insert(cudnn_reduction_input_dims.end(), pads.begin(), pads.end());
      cudnn_reduction_output_dims.insert(cudnn_reduction_output_dims.end(), pads.begin(), pads.end());
    }

    // Create shape and type information for input tensor and output tensor
    // for CUDNN reduction computation. It's useful when allocating memory. 
    CudnnTensor cudnn_input_tensor_desc;
    CudnnTensor cudnn_output_tensor_desc;
    cudnn_input_tensor_desc.Set(cudnn_reduction_input_dims, cudnn_reduction_type);
    cudnn_output_tensor_desc.Set(cudnn_reduction_output_dims, cudnn_reduction_type);

    // Create a wrapper of cudnnReduceTensorDescriptor_t, which controls
    // the configuration of CUDA reduction.
    CudnnReduceDescriptor cudnn_reduction_desc;
    cudnn_reduction_desc.Set(
      CUDNN_REDUCE_TENSOR_NORM2,
      cudnn_reduction_type,
      CUDNN_REDUCE_TENSOR_NO_INDICES);

    // Pre-allocate memory needed in CUDNN reduction.
    size_t cudnn_workspace_bytes = 0;
    CUDNN_RETURN_IF_ERROR(
      cudnnGetReductionWorkspaceSize(
        CudnnHandle(),
        cudnn_reduction_desc,
        cudnn_input_tensor_desc,
        cudnn_output_tensor_desc, &cudnn_workspace_bytes));

    auto cudnn_workspace = GetScratchBuffer<ToCudaType<float>::MappedType>(cudnn_workspace_bytes);

    // Compute pre-allocated memory amount needed in CUDNN reduction.
    size_t cudnn_indices_bytes = 0;
    CUDNN_RETURN_IF_ERROR(
      cudnnGetReductionIndicesSize(
        CudnnHandle(),
        cudnn_reduction_desc,
        cudnn_input_tensor_desc,
        cudnn_output_tensor_desc,
        &cudnn_indices_bytes));

    auto cudnn_indices_workspace = GetScratchBuffer<uint32_t>(cudnn_indices_bytes);

    // Allocate constants needed in the CUDNN reduction.
    const auto one = Consts<ToCudaType<float>::MappedType>::One;
    const auto zero = Consts<ToCudaType<float>::MappedType>::Zero;

    // Compute reductions of the update direction and the weight tensor. Note that
    // we reuse cudnn_reduction_desc because the two reductions have the same shapes.
    CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
        CudnnHandle(),
        cudnn_reduction_desc,
        cudnn_indices_workspace.get(),
        cudnn_indices_bytes,
        cudnn_workspace.get(),
        cudnn_workspace_bytes,
        &one,
        cudnn_input_tensor_desc,
        update_direction_buffer.get(),
        &zero,
        cudnn_output_tensor_desc,
        direction_norm_buffer.get()));

    CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
        CudnnHandle(),
        cudnn_reduction_desc,
        cudnn_indices_workspace.get(),
        cudnn_indices_bytes,
        cudnn_workspace.get(),
        cudnn_workspace_bytes,
        &one,
        cudnn_input_tensor_desc,
        weights_tensor.template Data<float>(),
        &zero,
        cudnn_output_tensor_desc,
        weights_norm_buffer.get()));
  }
  else
  {
    // CUDA reduction doesn't support one-element case, so we do it by our own CUDA kernel.
    LambScalarL2NormReductionImpl(update_direction_buffer.get(), direction_norm_buffer.get());
    LambScalarL2NormReductionImpl(weights_tensor.template Data<float>(), weights_norm_buffer.get());
  }

  // Use the update direction and the computed norms to compute
  // the new weights.
  LambUpdateImpl(
    eta_tensor.template Data<float>(),
    direction_norm_buffer.get(),
    weights_norm_buffer.get(),
    weights_tensor.template Data<float>(),
    update_direction_buffer.get(),
    weights_tensor_updated.template MutableData<float>(),
    weight_tensor_size);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
