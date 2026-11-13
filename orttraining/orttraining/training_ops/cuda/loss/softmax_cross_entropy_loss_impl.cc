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

#define REGISTER_KERNEL_VERSIONED_TYPED_TWO_TYPES(Class, T, TLabel, domain, startver, endver) \
  ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX(                                                \
      Class,                                                                                  \
      domain,                                                                                 \
      startver, endver,                                                                       \
      T, TLabel,                                                                              \
      kCudaExecutionProvider,                                                                 \
      (*KernelDefBuilder::Create())                                                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                              \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TLabel>()),                     \
      Class<T, TLabel, T>);

#define REGISTER_KERNEL_TYPED_TWO_TYPES(Class, T, TLabel, domain, version) \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                       \
      Class,                                                               \
      domain,                                                              \
      version,                                                             \
      T, TLabel,                                                           \
      kCudaExecutionProvider,                                              \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())           \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TLabel>()),  \
      Class<T, TLabel, T>);

template <typename T, typename TLabel, typename TOut>
Status SoftmaxCrossEntropyLoss<T, TLabel, TOut>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT_IN;
  typedef typename ToCudaType<TOut>::MappedType CudaT_OUT;

  const Tensor& logit = *ctx->Input<Tensor>(0);
  const Tensor& label = *ctx->Input<Tensor>(1);
  const Tensor* p_weight = ctx->Input<Tensor>(2);        // optional input
  const Tensor* p_ignore_index = ctx->Input<Tensor>(3);  // optional input

  // Prefer ignore index from input over attributes.
  int64_t ignore_index = ignore_index_;
  if (p_ignore_index) {
    ORT_ENFORCE(p_ignore_index->Shape().IsScalar(), "ignore_index should be a scalar.");
    ignore_index = *(p_ignore_index->template Data<int64_t>());
  }

  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};
  onnxruntime::contrib::VerifyLogitWeightAndLabelShape(logit_shape, label_shape,
                                                       p_weight ? &p_weight->Shape() : nullptr);

  // N_D = N * D1 * D2...Dk
  int64_t N_D, C;
  onnxruntime::contrib::GetNDCFromLogitAndLabelShape(logit_shape, label_shape, N_D, C);
  const TensorShape logit_reshape({N_D, C});
  Tensor* total_loss = ctx->Output(0, reduction_ == ReductionType::NONE ? TensorShape(label.Shape()) : TensorShape({}));
  TOut* total_loss_data = total_loss->template MutableData<TOut>();
  TOut* tmp_loss_sample_buffer = nullptr;
  IAllocatorUniquePtr<TOut> tmp_loss_sample;
  if (reduction_ == ReductionType::NONE) {
    tmp_loss_sample_buffer = total_loss_data;
  } else {
    tmp_loss_sample = GetScratchBuffer<TOut>(N_D, ctx->GetComputeStream());
    tmp_loss_sample_buffer = tmp_loss_sample.get();
  }

  const T* logit_data = logit.template Data<T>();
  const TLabel* label_data = label.template Data<TLabel>();

  TOut* log_prob_data = nullptr;
  Tensor* log_prob = nullptr;
  IAllocatorUniquePtr<TOut> log_prob_scratch_buffer;
  if (ctx->OutputCount() > 1) {
    log_prob = ctx->Output(1, logit_shape);
    log_prob_data = log_prob->template MutableData<TOut>();
  } else {
    log_prob_scratch_buffer = GetScratchBuffer<TOut>(logit_shape.Size(), ctx->GetComputeStream());
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
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(cuda::Transpose(info), ctx->GetComputeStream(), permutations, logit,
                                                     *transpose_output.GetMutable<Tensor>()));
    logit_data = (*transpose_output.GetMutable<Tensor>()).template Data<T>();
  }

  // Calculate logsoftmax
  auto status = SoftMaxComputeHelper<T, TOut, true>(Stream(ctx), logit_data, logit_reshape, log_prob_data, 1);
  ORT_RETURN_IF_ERROR(status);

  const T* weight_data = nullptr;
  if (p_weight) {
    const Tensor& weight = *p_weight;
    weight_data = weight.template Data<T>();
  }

  IAllocatorUniquePtr<CudaT_OUT> weight_data_nd = GetScratchBuffer<CudaT_OUT>(N_D, ctx->GetComputeStream());
  CudaT_OUT* weight_data_nd_data = weight_data_nd.get();
  ComputeSoftmaxCrossEntropyWeightsImpl(Stream(ctx),
                                        label_data,
                                        reinterpret_cast<const CudaT_IN*>(weight_data),
                                        N_D, C,
                                        ignore_index,
                                        reinterpret_cast<CudaT_OUT*>(weight_data_nd_data));

  // Compute buffer size in byte for reduction APIs.
  const auto buffer_size = compute_reduction_buffer_size<CudaT_OUT>(static_cast<int>(N_D));
  // Allocate reduction buffer whose size is buffer_size bytes, or nullptr if no reduction.
  IAllocatorUniquePtr<void> reduction_buffer = GetScratchBuffer<void>(
      reduction_ != ReductionType::NONE ? buffer_size : 0, ctx->GetComputeStream());

  typedef AccumulationType_t<CudaT_OUT> TBuf;
  auto normalize_factor_data = GetScratchBuffer<TBuf>(1, ctx->GetComputeStream());
  if (reduction_ == ReductionType::MEAN) {
    ORT_RETURN_IF_ERROR(reduce_sum(
        Stream(ctx),
        reinterpret_cast<CudaT_OUT*>(weight_data_nd_data),
        normalize_factor_data.get(),
        static_cast<int>(N_D),
        reduction_buffer.get(),
        buffer_size));
  } else {
    constexpr TBuf normalize_factor = static_cast<TBuf>(1.0f);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(TBuf),
                                         cudaMemcpyHostToDevice, Stream(ctx)));
  }

  SoftmaxCrossEntropyLossImpl(Stream(ctx),
                              reinterpret_cast<CudaT_OUT*>(log_prob_data),
                              label_data,
                              reinterpret_cast<CudaT_OUT*>(weight_data_nd_data),
                              normalize_factor_data.get(),
                              N_D,
                              C,
                              ignore_index,
                              reinterpret_cast<CudaT_OUT*>(tmp_loss_sample_buffer));

  // Transpose log probability from [N, D1, D2...Dk, C] to [N, C, D1, D2 .. Dk].
  if (logit_shape.NumDimensions() > 2 && log_prob != nullptr) {
    TensorShape log_prob_shape = new_shape;
    new_shape.clear();
    permutations.clear();
    onnxruntime::contrib::GetPermutationAndShape(false, log_prob_shape, new_shape, permutations);
    auto* transposed_data = (*transpose_output.GetMutable<Tensor>()).template MutableData<TOut>();
    transpose_output.GetMutable<Tensor>()->Reshape(log_prob->Shape());
    log_prob->Reshape(log_prob_shape);
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(cuda::Transpose(info), ctx->GetComputeStream(), permutations, *log_prob,
                                                     *transpose_output.GetMutable<Tensor>()));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(log_prob_data, transposed_data, sizeof(TOut) * logit_shape.Size(),
                                         cudaMemcpyDeviceToDevice, Stream(ctx)));
    log_prob->Reshape(new_shape);
  }

  if (reduction_ != ReductionType::NONE) {
    // ReduceSum on loss_per_sample
    ORT_RETURN_IF_ERROR(reduce_sum(
        Stream(ctx),
        reinterpret_cast<CudaT_OUT*>(tmp_loss_sample_buffer),
        reinterpret_cast<CudaT_OUT*>(total_loss_data),
        static_cast<int>(N_D),
        reduction_buffer.get(),
        buffer_size));
  }

  return Status::OK();
}

template <typename T, typename TLabel, typename TOut>
Status SoftmaxCrossEntropyLossGrad<T, TLabel, TOut>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT_IN;
  typedef typename ToCudaType<TOut>::MappedType CudaT_OUT;

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

  // N_D = N * D1 * D2...Dk
  int64_t N_D = 0, C = 0;
  onnxruntime::contrib::GetNDCFromLogitAndLabelShape(probability_shape, label_shape, N_D, C);
  Tensor* d_logit = ctx->Output(0, probability_shape);
  const T* dY_data = dY.Data<T>();
  const T* log_prob_data = log_prob.Data<T>();
  const TLabel* label_data = label.Data<TLabel>();
  TOut* d_logit_data = d_logit->MutableData<TOut>();
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
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(cuda::Transpose(info), ctx->GetComputeStream(), permutations,
                                                     log_prob, *transpose_output.GetMutable<Tensor>()));
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
                                        reinterpret_cast<const CudaT_IN*>(weight_data),
                                        N_D, C,
                                        ignore_index,
                                        reinterpret_cast<CudaT_IN*>(weight_data_nd_data));
  typedef AccumulationType_t<CudaT_IN> TBuf;
  auto normalize_factor_data = GetScratchBuffer<TBuf>(1, ctx->GetComputeStream());
  if (reduction_ == ReductionType::MEAN) {
    // Compute buffer size in byte for reduction APIs.
    const auto buffer_size =
        compute_reduction_buffer_size<CudaT_IN>(static_cast<int>(N_D));
    // Allocate reduction buffer whose size is buffer_size bytes.
    IAllocatorUniquePtr<void> reduction_buffer = GetScratchBuffer<void>(
        buffer_size, ctx->GetComputeStream());
    ORT_RETURN_IF_ERROR(reduce_sum(
        Stream(ctx),
        reinterpret_cast<const CudaT_IN*>(weight_data_nd_data),
        normalize_factor_data.get(),
        static_cast<int>(N_D),
        reduction_buffer.get(),
        buffer_size));
  } else {
    constexpr TBuf normalize_factor = static_cast<TBuf>(1.0f);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(TBuf),
                                         cudaMemcpyHostToDevice, Stream(ctx)));
  }

  const TOut* bias_data = p_bias ? p_bias->Data<TOut>() : nullptr;

  SoftmaxCrossEntropyLossGradImpl(Stream(ctx),
                                  reinterpret_cast<const CudaT_IN*>(dY_data),
                                  reinterpret_cast<const CudaT_IN*>(log_prob_data),
                                  label_data,
                                  reinterpret_cast<const CudaT_IN*>(weight_data_nd_data),
                                  normalize_factor_data.get(),
                                  reinterpret_cast<const CudaT_OUT*>(bias_data),
                                  N_D,
                                  C,
                                  ReductionType::NONE == reduction_,
                                  reinterpret_cast<CudaT_OUT*>(d_logit_data));

  // Transpose logit from [N, D1, D2...Dk, C] to [N, C, D1, D2 .. Dk]
  if (probability_shape.NumDimensions() > 2) {
    TensorShape logit_shape = new_shape;
    new_shape.clear();
    permutations.clear();
    onnxruntime::contrib::GetPermutationAndShape(false, logit_shape, new_shape, permutations);
    transpose_output.GetMutable<Tensor>()->Reshape(d_logit->Shape());
    d_logit->Reshape(logit_shape);
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(cuda::Transpose(info), ctx->GetComputeStream(), permutations,
                                                     *d_logit, *transpose_output.GetMutable<Tensor>()));
    auto* transposed_data = (*transpose_output.GetMutable<Tensor>()).template Data<TOut>();
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(d_logit_data, transposed_data, sizeof(TOut) * probability_shape.Size(),
                                         cudaMemcpyDeviceToDevice, Stream(ctx)));
    d_logit->Reshape(new_shape);
  }

  return Status::OK();
}

#define INSTANTIATE_VERSIONED_COMPUTE_SPARSE(Class, T, TLabel, domain, startver, endvar) \
  REGISTER_KERNEL_VERSIONED_TYPED_TWO_TYPES(Class, T, TLabel, domain, startver, endvar)

#define INSTANTIATE_COMPUTE_SPARSE(Class, T, TLabel, domain, version) \
  REGISTER_KERNEL_TYPED_TWO_TYPES(Class, T, TLabel, domain, version)  \
  template Status Class<T, TLabel, T>::ComputeInternal(OpKernelContext* ctx) const;

INSTANTIATE_VERSIONED_COMPUTE_SPARSE(SoftmaxCrossEntropyLoss, float, int64_t, kOnnxDomain, 12, 12)
INSTANTIATE_VERSIONED_COMPUTE_SPARSE(SoftmaxCrossEntropyLoss, MLFloat16, int64_t, kOnnxDomain, 12, 12)
INSTANTIATE_COMPUTE_SPARSE(SoftmaxCrossEntropyLoss, float, int64_t, kOnnxDomain, 13)
INSTANTIATE_COMPUTE_SPARSE(SoftmaxCrossEntropyLoss, MLFloat16, int64_t, kOnnxDomain, 13)
INSTANTIATE_COMPUTE_SPARSE(SoftmaxCrossEntropyLoss, BFloat16, int64_t, kOnnxDomain, 13)
INSTANTIATE_COMPUTE_SPARSE(SoftmaxCrossEntropyLossGrad, float, int64_t, kMSDomain, 1)
INSTANTIATE_COMPUTE_SPARSE(SoftmaxCrossEntropyLossGrad, MLFloat16, int64_t, kMSDomain, 1)
INSTANTIATE_COMPUTE_SPARSE(SoftmaxCrossEntropyLossGrad, BFloat16, int64_t, kMSDomain, 1)

#define REGISTER_KERNEL_INTERNAL_TYPED(OpName, ClassName, T, TLabel, TOut, CpuInputIndex)            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(OpName, kMSDomain, 1, T##_##TLabel##_##TOut, kCudaExecutionProvider, \
                                (*KernelDefBuilder::Create())                                        \
                                    .InputMemoryType(OrtMemTypeCPUInput, CpuInputIndex)              \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())           \
                                    .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TLabel>())   \
                                    .TypeConstraint("TOut", DataTypeImpl::GetTensorType<TOut>())     \
                                    .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),    \
                                ClassName<T, TLabel, TOut>);

REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternal, SoftmaxCrossEntropyLoss, MLFloat16, int64_t, float, 3)
REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternal, SoftmaxCrossEntropyLoss, float, int64_t, float, 3)
REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternal, SoftmaxCrossEntropyLoss, MLFloat16, int64_t, MLFloat16, 3)
REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternal, SoftmaxCrossEntropyLoss, BFloat16, int64_t, BFloat16, 3)

REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternalGrad, SoftmaxCrossEntropyLossGrad, float, int64_t, float, 4)
REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternalGrad, SoftmaxCrossEntropyLossGrad, float, int64_t, MLFloat16, 4)
REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternalGrad, SoftmaxCrossEntropyLossGrad, MLFloat16, int64_t, MLFloat16, 4)
REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternalGrad, SoftmaxCrossEntropyLossGrad, BFloat16, int64_t, BFloat16, 4)

}  // namespace cuda
}  // namespace onnxruntime
