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
  const Tensor& W = *ctx->Input<Tensor>(1);
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
#define REGISTER_ADAM_KERNEL_TYPED(T1, T2, T3, T4, T_GRAD)                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      AdamOptimizer,                                                        \
      kOnnxDomain,                                                          \
      9,                                                                    \
      T1##_##T2##_##T3##_##T4##_##T_GRAD,                                   \
      kCudaExecutionProvider,                                               \
      KernelDefBuilder()                                                    \
          .Alias(1, 3) /* Update step count in-place */                     \
          .Alias(2, 0) /* Update weights in-place */                        \
          .Alias(4, 1) /* Update moment-1 in-place */                       \
          .Alias(5, 2) /* Update moment-2 in-place */                       \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())          \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())          \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T3>())          \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<T4>())          \
          .TypeConstraint("T_GRAD", DataTypeImpl::GetTensorType<T_GRAD>()), \
      AdamOptimizer<T1, T2, T3, T4, T_GRAD>);

REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float, float)
REGISTER_ADAM_KERNEL_TYPED(MLFloat16, int64_t, float, MLFloat16, float)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, MLFloat16, float)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(MLFloat16, int64_t, float, MLFloat16, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, MLFloat16, MLFloat16)

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD>
Status AdamOptimizer<T1, T2, T3, T4, T_GRAD>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T1>::MappedType CudaT1;
  typedef typename ToCudaType<T2>::MappedType CudaT2;
  typedef typename ToCudaType<T3>::MappedType CudaT3;
  typedef typename ToCudaType<T4>::MappedType CudaT4;
  typedef typename ToCudaType<T_GRAD>::MappedType CudaT_GRAD;

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
      reinterpret_cast<const CudaT_GRAD*>(G.template Data<T_GRAD>()),
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
#define REGISTER_LAMB_KERNEL_TYPED(T1,T2,T3,T4)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                       \
      LambOptimizer,                                                   \
      kOnnxDomain,                                                     \
      9,                                                               \
      T1##_##T2##_##T3##_##T4,                                         \
      kCudaExecutionProvider,                                          \
      KernelDefBuilder()                                               \
          .Alias(1, 0)                                                 \
          .Alias(2, 1)                                                 \
          .Alias(3, 2)                                                 \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())     \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T3>())     \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<T4>()),    \
      LambOptimizer<T1, T2, T3, T4>);

REGISTER_LAMB_KERNEL_TYPED(float, float, float, float)
REGISTER_LAMB_KERNEL_TYPED(double, double, double, double)
REGISTER_LAMB_KERNEL_TYPED(MLFloat16, float, MLFloat16, MLFloat16)
REGISTER_LAMB_KERNEL_TYPED(MLFloat16, float, MLFloat16, float)

template<typename T1, typename T2, typename T3, typename T4>
Status LambOptimizer<T1, T2, T3, T4>::ComputeInternal(OpKernelContext* ctx) const {
  // CudaT* are types used to invoke CUDA-based functions. It, for example, maps
  // MLFloat16 in ONNXRuntime to half in CUDA.
  typedef typename ToCudaType<T1>::MappedType CudaT1;
  typedef typename ToCudaType<T2>::MappedType CudaT2;
  typedef typename ToCudaType<T3>::MappedType CudaT3;
  typedef typename ToCudaType<T4>::MappedType CudaT4;

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
  // Gradient type controls the direction's type.
  IAllocatorUniquePtr<T3> update_direction_buffer = GetScratchBuffer<T3>(weight_tensor_size);
  LambComputeDirectionImpl(
      reinterpret_cast<const CudaT2*>(weights_tensor.template Data<T2>()),
      reinterpret_cast<const CudaT3*>(gradients_tensor.template Data<T3>()),
      reinterpret_cast<const CudaT4*>(moment_1_tensor.template Data<T4>()),
      reinterpret_cast<const CudaT4*>(moment_2_tensor.template Data<T4>()),
      ToCudaType<T4>::FromFloat(alpha_),
      ToCudaType<T4>::FromFloat(beta_),
      ToCudaType<T2>::FromFloat(lambda_),
      ToCudaType<T4>::FromFloat(epsilon_),
      reinterpret_cast<CudaT3*>(update_direction_buffer.get()),
      reinterpret_cast<CudaT4*>(moment_1_tensor_updated.template MutableData<T4>()),
      reinterpret_cast<CudaT4*>(moment_2_tensor_updated.template MutableData<T4>()),
      weight_tensor_size);

  // Allocate buffer for reduction computation of update direction.
  // We reduce type T3 tensor to type T2 scalar. An example is that T3=float16
  // and T2=float.
  IAllocatorUniquePtr<T2> direction_norm_buffer = GetScratchBuffer<T2>(1);
  // Allocate buffer for reduction computation of weight tensor.
  // We reduce type T3 tensor to type T2 scalar. An example is that T3=float16
  // and T2=float.
  IAllocatorUniquePtr<T2> weights_norm_buffer = GetScratchBuffer<T2>(1);

  // Do reduction to compute the 2-norm of the weight tensor and
  // the 2-norm of the update direction. They will be used to
  // adjust the learning rate.
  if (weight_tensor_size > 1)
  {
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

    // The reduction should be as precise as the weight, so T2 is the type
    // that subsequent tensors reduced to.
    const cudnnDataType_t cudnn_reduction_type_T2 = CudnnTensor::GetDataType<CudaT2>();
    const cudnnDataType_t cudnn_reduction_type_T3 = CudnnTensor::GetDataType<CudaT3>();

    // Create shape and type information for input tensor and output tensor
    // for CUDNN reduction computation. It's useful when allocating memory. 
    CudnnTensor cudnn_input_tensor_desc_T2;
    CudnnTensor cudnn_input_tensor_desc_T3;
    CudnnTensor cudnn_output_tensor_desc;
    cudnn_input_tensor_desc_T2.Set(cudnn_reduction_input_dims, cudnn_reduction_type_T2);
    cudnn_input_tensor_desc_T3.Set(cudnn_reduction_input_dims, cudnn_reduction_type_T3);
    cudnn_output_tensor_desc.Set(cudnn_reduction_output_dims, cudnn_reduction_type_T2);

    // Create a wrapper of cudnnReduceTensorDescriptor_t, which controls
    // the configuration of CUDA reduction.
    // Subsequently, there will be two reductions. One reduces T2 tensor to T2 scalar
    // and the other one is T3-to-T2 reduction. Because in general, T2's precision
    // is higher than T3, T2 is the type that subsequent tensors reduced to.
    CudnnReduceDescriptor cudnn_reduction_desc;
    cudnn_reduction_desc.Set(
      CUDNN_REDUCE_TENSOR_NORM2,
      cudnn_reduction_type_T2,
      CUDNN_REDUCE_TENSOR_NO_INDICES);

    // Pre-allocate memory needed in CUDNN reduction.
    size_t cudnn_workspace_bytes_T2 = 0;
    CUDNN_RETURN_IF_ERROR(
      cudnnGetReductionWorkspaceSize(
        CudnnHandle(),
        cudnn_reduction_desc,
        cudnn_input_tensor_desc_T2,
        cudnn_output_tensor_desc,
        &cudnn_workspace_bytes_T2));

    auto cudnn_workspace_T2 = GetScratchBuffer<T2>(cudnn_workspace_bytes_T2);

    // Compute pre-allocated memory amount needed in CUDNN reduction.
    size_t cudnn_indices_bytes_T2 = 0;
    CUDNN_RETURN_IF_ERROR(
      cudnnGetReductionIndicesSize(
        CudnnHandle(),
        cudnn_reduction_desc,
        cudnn_input_tensor_desc_T2,
        cudnn_output_tensor_desc,
        &cudnn_indices_bytes_T2));

    auto cudnn_indices_workspace_T2 = GetScratchBuffer<uint32_t>(cudnn_indices_bytes_T2);

    // Pre-allocate memory needed in CUDNN reduction.
    size_t cudnn_workspace_bytes_T3 = 0;
    CUDNN_RETURN_IF_ERROR(
      cudnnGetReductionWorkspaceSize(
        CudnnHandle(),
        cudnn_reduction_desc,
        cudnn_input_tensor_desc_T3,
        cudnn_output_tensor_desc,
        &cudnn_workspace_bytes_T3));

    auto cudnn_workspace_T3 = GetScratchBuffer<T3>(cudnn_workspace_bytes_T3);

    // Compute pre-allocated memory amount needed in CUDNN reduction.
    size_t cudnn_indices_bytes_T3 = 0;
    CUDNN_RETURN_IF_ERROR(
      cudnnGetReductionIndicesSize(
        CudnnHandle(),
        cudnn_reduction_desc,
        cudnn_input_tensor_desc_T3,
        cudnn_output_tensor_desc,
        &cudnn_indices_bytes_T3));

    auto cudnn_indices_workspace_T3 = GetScratchBuffer<uint32_t>(cudnn_indices_bytes_T3);

    // Allocate constants needed in the CUDNN reduction.
    const auto one = Consts<T2>::One;
    const auto zero = Consts<T2>::Zero;

    // Compute reductions of the update direction and the weight tensor. Note that
    // we reuse cudnn_reduction_desc because the two reductions have the same shapes.
    CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
        CudnnHandle(),
        cudnn_reduction_desc,
        cudnn_indices_workspace_T3.get(),
        cudnn_indices_bytes_T3,
        cudnn_workspace_T3.get(),
        cudnn_workspace_bytes_T3,
        &one,
        cudnn_input_tensor_desc_T3,
        update_direction_buffer.get(),
        &zero,
        cudnn_output_tensor_desc,
        direction_norm_buffer.get()));

    CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
        CudnnHandle(),
        cudnn_reduction_desc,
        cudnn_indices_workspace_T2.get(),
        cudnn_indices_bytes_T2,
        cudnn_workspace_T2.get(),
        cudnn_workspace_bytes_T2,
        &one,
        cudnn_input_tensor_desc_T2,
        weights_tensor.template Data<T2>(),
        &zero,
        cudnn_output_tensor_desc,
        weights_norm_buffer.get()));
  }
  else
  {
    // CUDA reduction doesn't support one-element case, so we do it
    // by our own CUDA kernel.
    LambScalarL2NormReductionImpl(
      reinterpret_cast<const CudaT3*>(update_direction_buffer.get()),
      direction_norm_buffer.get());
    LambScalarL2NormReductionImpl(
      weights_tensor.template Data<T2>(),
      weights_norm_buffer.get());
  }

  // Use the update direction and the computed norms to compute
  // the new weights.
  LambUpdateImpl(
    reinterpret_cast<const CudaT1*>(eta_tensor.template Data<T1>()),
    reinterpret_cast<const CudaT2*>(direction_norm_buffer.get()),
    reinterpret_cast<const CudaT2*>(weights_norm_buffer.get()),
    reinterpret_cast<const CudaT2*>(weights_tensor.template Data<T2>()),
    reinterpret_cast<CudaT3*>(update_direction_buffer.get()),
    reinterpret_cast<CudaT2*>(weights_tensor_updated.template MutableData<T2>()),
    weight_tensor_size);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
