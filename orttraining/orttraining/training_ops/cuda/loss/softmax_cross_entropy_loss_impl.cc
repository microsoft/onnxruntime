// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/softmax.h"
#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/framework/ort_value.h"
#include "orttraining/training_ops/cpu/loss/softmax_cross_entropy_loss.h"
#include "orttraining/training_ops/cuda/loss/softmax_cross_entropy_loss_impl.h"

namespace onnxruntime {
namespace cuda {

OrtValue AllocateTensorInMLValue(const MLDataType data_type, const TensorShape& shape, AllocatorPtr& allocator) {
  auto new_tensor = Tensor::Create(data_type, shape, allocator);

  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  return OrtValue{new_tensor.release(), ml_tensor,
                  ml_tensor->GetDeleteFunc()};
};

#define REGISTER_KERNEL_VERSIONED_TYPED_TWO_TYPES(Class, T, Tin, domain, startver, endver) \
  ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX(                                             \
      Class,                                                                               \
      domain,                                                                              \
      startver, endver,                                                                    \
      T, Tin,                                                                              \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create())                                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                           \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<Tin>()),                     \
      Class<T, Tin>);

#define REGISTER_KERNEL_TYPED_TWO_TYPES(Class, T, Tin, domain, version) \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                    \
      Class,                                                            \
      domain,                                                           \
      version,                                                          \
      T, Tin,                                                           \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<Tin>()),  \
      Class<T, Tin>);

template <typename T, typename Tin>
Status SoftmaxCrossEntropyLoss<T, Tin>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const Tensor& logit = *ctx->Input<Tensor>(0);
  const Tensor& label = *ctx->Input<Tensor>(1);
  const Tensor* p_weight = ctx->Input<Tensor>(2);
  const Tensor* p_ignore_index = ctx->Input<Tensor>(3);
  int64_t ignore_index = ignore_index_;
  if (p_ignore_index) {
    ORT_ENFORCE(p_ignore_index->Shape().IsScalar(), "ignore_index should be a scalar.");
    ignore_index = *(p_ignore_index->template Data<int64_t>());
  }

  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};
  onnxruntime::contrib::VerifyLogitWeightAndLabelShape(logit_shape, label_shape,
                                                       p_weight ? &p_weight->Shape() : nullptr);

  // N_D = N * D1 * D2...D*K
  int64_t N_D;
  int64_t C;
  onnxruntime::contrib::GetNDCFromLogitAndLabelShape(logit_shape, label_shape, N_D, C);
  const TensorShape logit_reshape({N_D, C});
  Tensor* total_loss = ctx->Output(0, reduction_ == ReductionType::NONE ? TensorShape(label.Shape()) : TensorShape({}));
  T* total_loss_data = total_loss->template MutableData<T>();
  T* tmp_loss_sample_buffer = nullptr;
  IAllocatorUniquePtr<T> tmp_loss_sample;
  if (reduction_ == ReductionType::NONE) {
    tmp_loss_sample_buffer = total_loss_data;
  } else {
    tmp_loss_sample = GetScratchBuffer<T>(N_D, ctx->GetComputeStream());
    tmp_loss_sample_buffer = tmp_loss_sample.get();
  }

  const T* logit_data = logit.template Data<T>();
  const Tin* label_data = label.template Data<Tin>();

  T* log_prob_data = nullptr;
  Tensor* log_prob = nullptr;
  IAllocatorUniquePtr<T> log_prob_scratch_buffer;
  if (ctx->OutputCount() > 1) {
    log_prob = ctx->Output(1, logit_shape);
    log_prob_data = log_prob->template MutableData<T>();
  } else {
    log_prob_scratch_buffer = GetScratchBuffer<T>(logit_shape.Size(), ctx->GetComputeStream());
    log_prob_data = log_prob_scratch_buffer.get();
  }

  OrtValue transpose_output;
  TensorShapeVector new_shape;
  std::vector<size_t> permutations;
  AllocatorPtr alloc;
  const OpKernelInfo& info = OpKernel::Info();

  // Transpose logit from [N, C, D1, D2 .. Dk] to [N, D1, D2...Dk, C]
  if (logit_shape.NumDimensions() > 2) {
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
    onnxruntime::contrib::GetPermutationAndShape(true, logit_shape, new_shape, permutations);
    transpose_output = AllocateTensorInMLValue(logit.DataType(), new_shape, alloc);
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(cuda::Transpose(info), ctx->GetComputeStream(), permutations, logit, *transpose_output.GetMutable<Tensor>()));
    logit_data = (*transpose_output.GetMutable<Tensor>()).template Data<T>();
  }

  // calculate logsoftmax
  auto status = SoftMaxComputeHelper<T, true>(Stream(ctx),
                                              logit_data,
                                              logit_reshape,
                                              log_prob_data,
                                              1);
  ORT_RETURN_IF_ERROR(status);

  const T* weight_data = nullptr;
  if (p_weight) {
    const Tensor& weight = *p_weight;
    weight_data = weight.template Data<T>();
  }

  IAllocatorUniquePtr<T> weight_data_nd = GetScratchBuffer<T>(N_D, ctx->GetComputeStream());
  T* weight_data_nd_data = weight_data_nd.get();
  ComputeSoftmaxCrossEntropyWeightsImpl(Stream(ctx),
                                        label_data,
                                        reinterpret_cast<const CudaT*>(weight_data),
                                        N_D, C,
                                        ignore_index,
                                        reinterpret_cast<CudaT*>(weight_data_nd_data));

  // Compute buffer size in byte for reduction APIs.
  const auto buffer_size =
      compute_reduction_buffer_size<CudaT>(static_cast<int>(N_D));
  // Allocate reduction buffer whose size is buffer_size bytes, or nullptr if no reduction.
  IAllocatorUniquePtr<void> reduction_buffer = GetScratchBuffer<void>(
      reduction_ != ReductionType::NONE ? buffer_size : 0, ctx->GetComputeStream());

  typedef AccumulationType_t<CudaT> TBuf;
  auto normalize_factor_data = GetScratchBuffer<TBuf>(1, ctx->GetComputeStream());
  if (reduction_ == ReductionType::MEAN) {
    ORT_RETURN_IF_ERROR(reduce_sum(
        Stream(ctx),
        reinterpret_cast<CudaT*>(weight_data_nd_data),
        normalize_factor_data.get(),
        static_cast<int>(N_D),
        reduction_buffer.get(),
        buffer_size));
  } else {
    constexpr TBuf normalize_factor = static_cast<TBuf>(1.0f);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(TBuf), cudaMemcpyHostToDevice, Stream(ctx)));
  }

  SoftmaxCrossEntropyLossImpl(Stream(ctx),
                              reinterpret_cast<CudaT*>(log_prob_data),
                              label_data,
                              reinterpret_cast<CudaT*>(weight_data_nd_data),
                              normalize_factor_data.get(),
                              N_D,
                              C,
                              ignore_index,
                              reinterpret_cast<CudaT*>(tmp_loss_sample_buffer));

  // Transpose log probability from [N, D1, D2...Dk, C] to [N, C, D1, D2 .. Dk].
  if (logit_shape.NumDimensions() > 2 && log_prob != nullptr) {
    TensorShape log_prob_shape = new_shape;
    new_shape.clear();
    permutations.clear();
    onnxruntime::contrib::GetPermutationAndShape(false, log_prob_shape, new_shape, permutations);
    auto* transposed_data = (*transpose_output.GetMutable<Tensor>()).template MutableData<T>();
    transpose_output.GetMutable<Tensor>()->Reshape(log_prob->Shape());
    log_prob->Reshape(log_prob_shape);
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(cuda::Transpose(info), ctx->GetComputeStream(), permutations, *log_prob, *transpose_output.GetMutable<Tensor>()));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(log_prob_data, transposed_data, sizeof(T) * logit_shape.Size(), cudaMemcpyDeviceToDevice, Stream(ctx)));
    log_prob->Reshape(new_shape);
  }

  if (reduction_ != ReductionType::NONE) {
    // ReduceSum on loss_per_sample
    ORT_RETURN_IF_ERROR(reduce_sum(
        Stream(ctx),
        reinterpret_cast<CudaT*>(tmp_loss_sample_buffer),
        reinterpret_cast<CudaT*>(total_loss_data),
        static_cast<int>(N_D),
        reduction_buffer.get(),
        buffer_size));
  }

  return Status::OK();
}

template <typename T, typename Tin>
Status SoftmaxCrossEntropyLossGrad<T, Tin>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const Tensor& dY = *ctx->Input<Tensor>(0);
  const Tensor& log_prob = *ctx->Input<Tensor>(1);
  const Tensor& label = *ctx->Input<Tensor>(2);
  const Tensor* p_weight = ctx->Input<Tensor>(3);
  const Tensor* p_ignore_index = ctx->Input<Tensor>(4);
  const Tensor* p_bias = ctx->Input<Tensor>(5);
  int64_t ignore_index = ignore_index_;
  if (p_ignore_index) {
    ORT_ENFORCE(p_ignore_index->Shape().IsScalar(), "ignore_index should be a scalar.");
    ignore_index = *(p_ignore_index->Data<int64_t>());
  }

  const TensorShape probability_shape{log_prob.Shape()};
  const TensorShape label_shape{label.Shape()};
  onnxruntime::contrib::VerifyLogitWeightAndLabelShape(probability_shape, label_shape,
                                                       p_weight ? &p_weight->Shape() : nullptr);

  // N_D = N * D1 * D2...D*K
  int64_t N_D;
  int64_t C;
  onnxruntime::contrib::GetNDCFromLogitAndLabelShape(probability_shape, label_shape, N_D, C);
  Tensor* d_logit = ctx->Output(0, probability_shape);
  const T* dY_data = dY.Data<T>();
  const T* log_prob_data = log_prob.Data<T>();
  const Tin* label_data = label.Data<Tin>();
  T* d_logit_data = d_logit->MutableData<T>();
  const T* weight_data = nullptr;
  OrtValue transpose_output;
  TensorShapeVector new_shape;
  std::vector<size_t> permutations;
  AllocatorPtr alloc;
  const OpKernelInfo& info = OpKernel::Info();

  // Transpose logit from [N, C, D1, D2 .. Dk] to [N, D1, D2...Dk, C]
  if (probability_shape.NumDimensions() > 2) {
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
    onnxruntime::contrib::GetPermutationAndShape(true, probability_shape, new_shape, permutations);
    transpose_output = AllocateTensorInMLValue(log_prob.DataType(), new_shape, alloc);
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(cuda::Transpose(info), ctx->GetComputeStream(), permutations, log_prob, *transpose_output.GetMutable<Tensor>()));
    log_prob_data = (*transpose_output.GetMutable<Tensor>()).Data<T>();
  }

  if (p_weight) {
    const Tensor& weight = *p_weight;
    weight_data = weight.Data<T>();
  }

  IAllocatorUniquePtr<T> weight_data_nd = GetScratchBuffer<T>(N_D, ctx->GetComputeStream());
  T* weight_data_nd_data = weight_data_nd.get();
  ComputeSoftmaxCrossEntropyWeightsImpl(Stream(ctx),
                                        label_data,
                                        reinterpret_cast<const CudaT*>(weight_data),
                                        N_D, C,
                                        ignore_index,
                                        reinterpret_cast<CudaT*>(weight_data_nd_data));
  typedef AccumulationType_t<CudaT> TBuf;
  auto normalize_factor_data = GetScratchBuffer<TBuf>(1, ctx->GetComputeStream());
  if (reduction_ == ReductionType::MEAN) {
    // Compute buffer size in byte for reduction APIs.
    const auto buffer_size =
        compute_reduction_buffer_size<CudaT>(static_cast<int>(N_D));
    // Allocate reduction buffer whose size is buffer_size bytes.
    IAllocatorUniquePtr<void> reduction_buffer = GetScratchBuffer<void>(
        buffer_size, ctx->GetComputeStream());
    ORT_RETURN_IF_ERROR(reduce_sum(
        Stream(ctx),
        reinterpret_cast<const CudaT*>(weight_data_nd_data),
        normalize_factor_data.get(),
        static_cast<int>(N_D),
        reduction_buffer.get(),
        buffer_size));
  } else {
    constexpr TBuf normalize_factor = static_cast<TBuf>(1.0f);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(TBuf), cudaMemcpyHostToDevice, Stream(ctx)));
  }

  const T* bias_data = p_bias ? p_bias->Data<T>() : nullptr;

  SoftmaxCrossEntropyLossGradImpl(Stream(ctx),
                                  reinterpret_cast<const CudaT*>(dY_data),
                                  reinterpret_cast<const CudaT*>(log_prob_data),
                                  label_data,
                                  reinterpret_cast<const CudaT*>(weight_data_nd_data),
                                  normalize_factor_data.get(),
                                  reinterpret_cast<const CudaT*>(bias_data),
                                  N_D,
                                  C,
                                  ReductionType::NONE == reduction_,
                                  reinterpret_cast<CudaT*>(d_logit_data));

  // Transpose logit from [N, D1, D2...Dk, C] to [N, C, D1, D2 .. Dk]
  if (probability_shape.NumDimensions() > 2) {
    TensorShape logit_shape = new_shape;
    new_shape.clear();
    permutations.clear();
    onnxruntime::contrib::GetPermutationAndShape(false, logit_shape, new_shape, permutations);
    transpose_output.GetMutable<Tensor>()->Reshape(d_logit->Shape());
    d_logit->Reshape(logit_shape);
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(cuda::Transpose(info), ctx->GetComputeStream(), permutations, *d_logit, *transpose_output.GetMutable<Tensor>()));
    auto* transposed_data = (*transpose_output.GetMutable<Tensor>()).template Data<T>();
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(d_logit_data, transposed_data, sizeof(T) * probability_shape.Size(), cudaMemcpyDeviceToDevice, Stream(ctx)));
    d_logit->Reshape(new_shape);
  }

  return Status::OK();
}

#define INSTANTIATE_VERSIONED_COMPUTE_SPARSE(Class, T, Tin, domain, startver, endvar) \
  REGISTER_KERNEL_VERSIONED_TYPED_TWO_TYPES(Class, T, Tin, domain, startver, endvar)

#define INSTANTIATE_COMPUTE_SPARSE(Class, T, Tin, domain, version) \
  REGISTER_KERNEL_TYPED_TWO_TYPES(Class, T, Tin, domain, version)  \
  template Status Class<T, Tin>::ComputeInternal(OpKernelContext* ctx) const;

INSTANTIATE_VERSIONED_COMPUTE_SPARSE(SoftmaxCrossEntropyLoss, float, int64_t, kOnnxDomain, 12, 12)
INSTANTIATE_VERSIONED_COMPUTE_SPARSE(SoftmaxCrossEntropyLoss, MLFloat16, int64_t, kOnnxDomain, 12, 12)
INSTANTIATE_COMPUTE_SPARSE(SoftmaxCrossEntropyLoss, float, int64_t, kOnnxDomain, 13)
INSTANTIATE_COMPUTE_SPARSE(SoftmaxCrossEntropyLoss, MLFloat16, int64_t, kOnnxDomain, 13)
INSTANTIATE_COMPUTE_SPARSE(SoftmaxCrossEntropyLoss, BFloat16, int64_t, kOnnxDomain, 13)
INSTANTIATE_COMPUTE_SPARSE(SoftmaxCrossEntropyLossGrad, float, int64_t, kMSDomain, 1)
INSTANTIATE_COMPUTE_SPARSE(SoftmaxCrossEntropyLossGrad, MLFloat16, int64_t, kMSDomain, 1)
INSTANTIATE_COMPUTE_SPARSE(SoftmaxCrossEntropyLossGrad, BFloat16, int64_t, kMSDomain, 1)

#define REGISTER_KERNEL_INTERNAL_TYPED(OpName, ClassName, T, Tin, CpuInputIndex)                      \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(OpName, kMSDomain, 1, T, Tin, kCudaExecutionProvider,             \
                                    (*KernelDefBuilder::Create())                                     \
                                        .InputMemoryType(OrtMemTypeCPUInput, CpuInputIndex)           \
                                        .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
                                        .TypeConstraint("Tind", DataTypeImpl::GetTensorType<Tin>())   \
                                        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
                                    ClassName<T, Tin>);

REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternal, SoftmaxCrossEntropyLoss, float, int64_t, 3)
REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternal, SoftmaxCrossEntropyLoss, MLFloat16, int64_t, 3)
REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternal, SoftmaxCrossEntropyLoss, BFloat16, int64_t, 3)
REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternalGrad, SoftmaxCrossEntropyLossGrad, float, int64_t, 4)
REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternalGrad, SoftmaxCrossEntropyLossGrad, MLFloat16, int64_t, 4)
REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternalGrad, SoftmaxCrossEntropyLossGrad, BFloat16, int64_t, 4)

}  // namespace cuda
}  // namespace onnxruntime
