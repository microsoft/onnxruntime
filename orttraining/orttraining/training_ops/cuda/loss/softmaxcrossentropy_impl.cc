// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/providers/cuda/math/softmax.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cpu/controlflow/scan_utils.h"
#include "orttraining/training_ops/cpu/loss/softmax_cross_entropy_loss.h"
#include "orttraining/training_ops/cuda/loss/softmaxcrossentropy_impl.h"

namespace onnxruntime {
namespace cuda {
#define REGISTER_KERNEL_TYPED(Class, T, domain, version)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      Class,                                                                    \
      domain,                                                                   \
      version,                                                                  \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Class<T>);

#define REGISTER_KERNEL_TYPED_TWO_TYPES(Class, T, Tin, domain, version) \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                    \
      Class,                                                            \
      domain,                                                           \
      version,                                                          \
      T, Tin,                                                           \
      kCudaExecutionProvider,                                           \
      KernelDefBuilder()                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("Tin", DataTypeImpl::GetTensorType<Tin>()),   \
      Class<T, Tin>);

template <typename T>
Status SoftmaxCrossEntropy<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& logit = *ctx->Input<Tensor>(0);
  const Tensor& label = *ctx->Input<Tensor>(1);

  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};
  ORT_ENFORCE(label_shape == logit_shape, "The shape in logits and labels is not identical");

  int64_t N = logit_shape.SizeToDimension(logit_shape.NumDimensions() - 1);
  int64_t D = logit_shape[logit_shape.NumDimensions() - 1];
  const TensorShape logit_reshape({N, D});

  Tensor* log_prob = ctx->Output(1, logit_shape);

  const T* logit_data = logit.template Data<T>();
  const T* label_data = label.template Data<T>();
  T* log_prob_data = log_prob->template MutableData<T>();

  // calculate logsoftmax
  auto status = SoftMaxComputeHelper<T, true>(logit_data,
                                              logit_reshape,
                                              log_prob_data,
                                              CudnnHandle(),
                                              1 /*axis default*/);
  ORT_RETURN_IF_ERROR(status);

  size_t normalize_factor = N;
  if (reduction_ == ReductionType::SUM) {
    normalize_factor = static_cast<size_t>(1);
  }

  // calculate (label * log(softmax)) for each element
  IAllocatorUniquePtr<T> temp_X = GetScratchBuffer<T>(N * D);
  SoftMaxCrossEntropyImpl(
      log_prob_data,     // logsoftmax result
      label_data,        // label
      normalize_factor,  // normalize_factor
      temp_X.get(),      // -(label * log(softmax))
      N * D);

  std::vector<int64_t> output_dims(2, 1);
  Tensor* Y = ctx->Output(0, TensorShape({}));
  // Sum((label * log(softmax)) using Reduction
  ReduceKernelShared<T, T, CUDNN_REDUCE_TENSOR_NO_INDICES>(
      temp_X.get(),
      logit_reshape,
      Y->template MutableData<T>(),
      TensorShape({}),
      CUDNN_REDUCE_TENSOR_ADD,
      output_dims);

  return Status::OK();
}

template <typename T>
Status SoftmaxCrossEntropyGrad<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& dY = *ctx->Input<Tensor>(0);
  const Tensor& log_prob = *ctx->Input<Tensor>(1);
  const Tensor& label = *ctx->Input<Tensor>(2);

  const TensorShape probability_shape{log_prob.Shape()};
  const TensorShape label_shape{label.Shape()};
  ORT_ENFORCE(label_shape == probability_shape, "The shape in probability and label is not identical");

  int64_t N = probability_shape.SizeToDimension(probability_shape.NumDimensions() - 1);
  int64_t ND = probability_shape.Size();

  Tensor* d_logits = ctx->Output(0, probability_shape);

  const T* dY_data = dY.template Data<T>();
  const T* log_prob_data = log_prob.template Data<T>();
  const T* label_data = label.template Data<T>();

  size_t normalize_factor = N;
  if (reduction_ == ReductionType::SUM) {
    normalize_factor = static_cast<size_t>(1);
  }

  T* d_logits_data = d_logits->template MutableData<T>();

  SoftMaxCrossEntropyGradImpl(
      dY_data,           // Dy
      log_prob_data,     // log(pi)
      label_data,        // Label
      normalize_factor,  // normalize_factor
      d_logits_data,     // gradient
      ND);

  return Status::OK();
}

template <typename T, typename Tin>
Status SparseSoftmaxCrossEntropy<T, Tin>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& logit = *ctx->Input<Tensor>(0);
  const Tensor& label = *ctx->Input<Tensor>(1);

  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};
  ORT_ENFORCE(logit_shape.NumDimensions() == label_shape.NumDimensions() + 1,
              "logits_shape must be (1 + label_shape)");
  for (size_t i = 0; i < label_shape.NumDimensions(); i++) {
    ORT_ENFORCE(label_shape[i] == logit_shape[i], "The shape in logits and labels does not match");
  }

  int64_t N = label_shape.Size();
  int64_t D = logit_shape[logit_shape.NumDimensions() - 1];
  const TensorShape logit_reshape({N, D});
  const TensorShape label_reshape({N});

  IAllocatorUniquePtr<T> tmp_loss_sample = GetScratchBuffer<T>(N);
  Tensor* total_loss = ctx->Output(0, reduction_ == ReductionType::NONE ? TensorShape({label_shape[0]}) : TensorShape({}));
  Tensor* log_prob = ctx->Output(1, logit_shape);

  const T* logit_data = logit.template Data<T>();
  const Tin* label_data = label.template Data<Tin>();
  T* total_loss_data = total_loss->template MutableData<T>();
  T* log_prob_data = log_prob->template MutableData<T>();

  // calculate logsoftmax
  auto status = SoftMaxComputeHelper<T, true>(logit_data,
                                              logit_reshape,
                                              log_prob_data,
                                              CudnnHandle(),
                                              1 /*axis default*/);
  ORT_RETURN_IF_ERROR(status);

  // calculate  (label * log(softmax)) for each sample
  const T* weight_data = nullptr;
  if (OpKernel::Node().InputDefs().size() == 3) {
    const Tensor& weight = *ctx->Input<Tensor>(2);
    const TensorShape weight_shape{weight.Shape()};
    ORT_ENFORCE(weight_shape == label_shape, "The shape in weights and labels is different");
    weight_data = weight.template Data<T>();
  }

  auto normalize_factor_data = GetScratchBuffer<T>(1);
  if (reduction_ == ReductionType::SUM) {
    const T normalize_factor = static_cast<T>(1);
    cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(T), cudaMemcpyHostToDevice);
  } else if (reduction_ == ReductionType::MEAN) {
    if (weight_data == nullptr) {
      const T normalize_factor = static_cast<T>(N);
      cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(T), cudaMemcpyHostToDevice);
    } else {
      // Compute buffer size in byte for reduction APIs.
      const auto buffer_size = static_cast<size_t>(
          compute_reduction_buffer_size(
              static_cast<int>(sizeof(T)), static_cast<int>(N)));
      // Allocate reduction buffer whose size is buffer_size bytes.
      IAllocatorUniquePtr<void> reduction_buffer = GetScratchBuffer<void>(
          buffer_size);
      reduce_sum(weight_data,
                 normalize_factor_data.get(),
                 static_cast<int>(N),
                 reinterpret_cast<T*>(reduction_buffer.get()));
    }
  }

  SparseSoftmaxCrossEntropyImpl(log_prob_data,
                                label_data,
                                weight_data,
                                normalize_factor_data.get(),
                                tmp_loss_sample.get(),
                                N,
                                D);

  // ReduceSum on loss_per_sample
  std::vector<int64_t> output_dims(1, 1);
  ReduceKernelShared<T, T, CUDNN_REDUCE_TENSOR_NO_INDICES>(
      tmp_loss_sample.get(),
      label_reshape,
      total_loss_data,
      TensorShape({}),
      CUDNN_REDUCE_TENSOR_ADD,
      output_dims);

  return Status::OK();
}

template <typename T, typename Tin>
Status SparseSoftmaxCrossEntropyGrad<T, Tin>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& dY = *ctx->Input<Tensor>(0);
  const Tensor& log_prob = *ctx->Input<Tensor>(1);
  const Tensor& label = *ctx->Input<Tensor>(2);

  const TensorShape probability_shape{log_prob.Shape()};
  const TensorShape label_shape{label.Shape()};
  ORT_ENFORCE(probability_shape.NumDimensions() == label_shape.NumDimensions() + 1,
              "probability_shape must be (1 + label_shape)");
  for (size_t i = 0; i < label_shape.NumDimensions(); i++) {
    ORT_ENFORCE(label_shape[i] == probability_shape[i], "The shape in probability and labels does not match");
  }

  int64_t N = label_shape.Size();
  int64_t D = probability_shape[probability_shape.NumDimensions() - 1];

  Tensor* d_logit = ctx->Output(0, probability_shape);

  const T* dY_data = dY.template Data<T>();
  const T* log_prob_data = log_prob.template Data<T>();
  const Tin* label_data = label.template Data<Tin>();
  T* d_logit_data = d_logit->template MutableData<T>();

  const T* weight_data = nullptr;
  if (OpKernel::Node().InputDefs().size() == 4) {
    const Tensor& weight = *ctx->Input<Tensor>(3);
    const TensorShape weight_shape{weight.Shape()};
    ORT_ENFORCE(weight_shape == label_shape, "The shape in weights and labels is different");
    weight_data = weight.template Data<T>();
  }

  auto normalize_factor_data = GetScratchBuffer<T>(1);
  if (reduction_ == ReductionType::SUM) {
    const T normalize_factor = static_cast<T>(1);
    cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(T), cudaMemcpyHostToDevice);
  } else if (reduction_ == ReductionType::MEAN) {
    if (weight_data == nullptr) {
      const T normalize_factor = static_cast<T>(N);
      cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(T), cudaMemcpyHostToDevice);
    } else {
      // Compute buffer size in byte for reduction APIs.
      const auto buffer_size = static_cast<size_t>(
          compute_reduction_buffer_size(
              static_cast<int>(sizeof(T)), static_cast<int>(N)));
      // Allocate reduction buffer whose size is buffer_size bytes.
      IAllocatorUniquePtr<void> reduction_buffer = GetScratchBuffer<void>(
          buffer_size);
      reduce_sum(weight_data,
                 normalize_factor_data.get(),
                 static_cast<int>(N),
                 reinterpret_cast<T*>(reduction_buffer.get()));
    }
  }

  SparseSoftmaxCrossEntropyGradImpl(dY_data,
                                    log_prob_data,
                                    label_data,
                                    weight_data,
                                    normalize_factor_data.get(),
                                    d_logit_data,
                                    N,
                                    D);

  return Status::OK();
}

template <typename T, typename Tin>
Status SoftmaxCrossEntropyLoss<T, Tin>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& logit = *ctx->Input<Tensor>(0);
  const Tensor& label = *ctx->Input<Tensor>(1);
  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};
  onnxruntime::contrib::VerifyLogitAndLabelShape(logit_shape, label_shape);

  int64_t N;
  int64_t D;
  int64_t C;
  onnxruntime::contrib::GetNDCFromLogitAndLabelShape(logit_shape, label_shape, N, D, C);
  int64_t N_D = N * D;
  const TensorShape logit_reshape({N_D, C});
  const TensorShape label_reshape({N_D});
  IAllocatorUniquePtr<T> tmp_loss_sample = GetScratchBuffer<T>(N_D);
  const T* logit_data = logit.template Data<T>();
  const Tin* label_data = label.template Data<Tin>();

  // Output 0: Loss data.
  Tensor* total_loss = ctx->Output(0, reduction_ == ReductionType::NONE ? TensorShape(label.Shape()) : TensorShape({}));
  T* total_loss_data = total_loss->template MutableData<T>();

  // Output 1: Log probability data.
  T* log_prob_data = nullptr;
  Tensor* log_prob = nullptr;
  IAllocatorUniquePtr<T> log_prob_scratch_buffer;
  if (ctx->OutputCount() > 1) {
    log_prob = ctx->Output(1, logit_shape);
    log_prob_data = log_prob->template MutableData<T>();
  } else {
    log_prob_scratch_buffer = GetScratchBuffer<T>(logit_shape.Size());
    log_prob_data = log_prob_scratch_buffer.get();
  }

  ORT_ENFORCE(cudaMemset(log_prob_data, 0, N_D * C * sizeof(T)) == cudaSuccess);
  ORT_ENFORCE(cudaMemset(total_loss_data, 0, sizeof(T)) == cudaSuccess);

  OrtValue transpose_output;
  Tensor transpose_tensor;
  std::vector<int64_t> new_shape;
  std::vector<size_t> permutations;
  AllocatorPtr alloc;
  const OpKernelInfo& info = OpKernel::Info();

  // Transpose logit from [N, C, D1, D2 .. Dk] to [N, D1, D2...Dk, C]
  // REVIEW(codemzs): In PyTorch they seem to only handle 3-D and 4-D case and are able to avoid these transposes for
  // performance. However in Tensorflow they always transpose and handle N-D case.
  if (logit_shape.NumDimensions() > 2) {
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
    onnxruntime::contrib::GetPermutationAndShape(true, logit_shape, new_shape, permutations);
    transpose_output = scan::detail::AllocateTensorInMLValue(logit.DataType(), new_shape, alloc);
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(cuda::Transpose(info), permutations, logit, *transpose_output.GetMutable<Tensor>()));
    logit_data = (*transpose_output.GetMutable<Tensor>()).template Data<T>();
  }

  // calculate logsoftmax
  auto status = SoftMaxComputeHelper<T, true>(logit_data,
                                              logit_reshape,
                                              log_prob_data,
                                              CudnnHandle(),
                                              1);
  ORT_RETURN_IF_ERROR(status);

  // calculate  (label * log(softmax)) for each sample
  const T* weight_data = nullptr;
  if (OpKernel::Node().InputDefs().size() == 3) {
    const Tensor& weight = *ctx->Input<Tensor>(2);
    const TensorShape weight_shape{weight.Shape()};
    ORT_ENFORCE(1 == weight_shape.NumDimensions(), "Weights tensor is not 1-D.");
    weight_data = weight.template Data<T>();
  }

  //REVIEW(codemzs): Inefficient, just there for correctness, will get rid of this and come up with parallel
  //implementation and also avoid allocation weight buffer.
  IAllocatorUniquePtr<T> weight_data_nd = GetScratchBuffer<T>(N_D);
  T* weight_data_nd_data = weight_data_nd.get();
  ORT_ENFORCE(cudaMemset(weight_data_nd_data, 0, N_D * sizeof(T)) == cudaSuccess);
  ComputeWeightsSoftmaxCrossEntropyImpl(weight_data_nd_data, label_data, weight_data, N_D, C, ignore_index_);

  auto normalize_factor_data = GetScratchBuffer<T>(1);
  if (reduction_ == ReductionType::MEAN) {
    // Compute buffer size in byte for reduction APIs.
    const auto buffer_size = static_cast<size_t>(
        compute_reduction_buffer_size(
            static_cast<int>(sizeof(T)), static_cast<int>(N_D)));
    // Allocate reduction buffer whose size is buffer_size bytes.
    IAllocatorUniquePtr<void> reduction_buffer = GetScratchBuffer<void>(
        buffer_size);
    reduce_sum(weight_data_nd_data,
               normalize_factor_data.get(),
               static_cast<int>(N_D),
               reinterpret_cast<T*>(reduction_buffer.get()));
  } else {
    const T normalize_factor = static_cast<T>(1);
    cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(T), cudaMemcpyHostToDevice);
  }

  SoftmaxCrossEntropyLossImpl(log_prob_data,
                              label_data,
                              weight_data_nd_data,
                              normalize_factor_data.get(),
                              tmp_loss_sample.get(),
                              N_D,
                              C,
                              ignore_index_);

  // Transpose log probability from [N, D1, D2...Dk, C] to [N, C, D1, D2 .. Dk].
  if (logit_shape.NumDimensions() > 2 && log_prob != nullptr) {
    TensorShape log_prob_shape = new_shape;
    new_shape.clear();
    permutations.clear();
    onnxruntime::contrib::GetPermutationAndShape(false, log_prob_shape, new_shape, permutations);
    auto* transposed_data = (*transpose_output.GetMutable<Tensor>()).template MutableData<T>();
    transpose_output.GetMutable<Tensor>()->Reshape(log_prob->Shape());
    log_prob->Reshape(log_prob_shape);
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(cuda::Transpose(info), permutations, *log_prob, *transpose_output.GetMutable<Tensor>()));
    cudaMemcpyAsync(log_prob_data, transposed_data, sizeof(T) * logit_shape.Size(), cudaMemcpyDeviceToDevice);
    log_prob->Reshape(new_shape);
  }

  if (reduction_ == ReductionType::NONE) {
    cudaMemcpyAsync(total_loss_data, tmp_loss_sample.get(), sizeof(T) * N_D, cudaMemcpyDeviceToDevice);
    return Status::OK();
  }

  // ReduceSum on loss_per_sample
  std::vector<int64_t> output_dims(1, 1);
  ReduceKernelShared<T, T, CUDNN_REDUCE_TENSOR_NO_INDICES>(
      tmp_loss_sample.get(),
      label_reshape,
      total_loss_data,
      TensorShape({}),
      CUDNN_REDUCE_TENSOR_ADD,
      output_dims);

  return Status::OK();
}

template <typename T, typename Tin>
Status SoftmaxCrossEntropyLossGrad<T, Tin>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& dY = *ctx->Input<Tensor>(0);
  const Tensor& log_prob = *ctx->Input<Tensor>(1);
  const Tensor& label = *ctx->Input<Tensor>(2);
  const TensorShape probability_shape{log_prob.Shape()};
  const TensorShape label_shape{label.Shape()};
  onnxruntime::contrib::VerifyLogitAndLabelShape(probability_shape, label_shape);

  int64_t N;
  int64_t D;
  int64_t C;
  onnxruntime::contrib::GetNDCFromLogitAndLabelShape(probability_shape, label_shape, N, D, C);
  int64_t N_D = N * D;
  Tensor* d_logit = ctx->Output(0, probability_shape);
  const T* dY_data = dY.template Data<T>();
  const T* log_prob_data = log_prob.template Data<T>();
  const Tin* label_data = label.template Data<Tin>();
  T* d_logit_data = d_logit->template MutableData<T>();
  const T* weight_data = nullptr;
  OrtValue transpose_output;
  std::vector<int64_t> new_shape;
  std::vector<size_t> permutations;
  AllocatorPtr alloc;
  const OpKernelInfo& info = OpKernel::Info();

  // Transpose logit from [N, C, D1, D2 .. Dk] to [N, D1, D2...Dk, C]
  // REVIEW(codemzs): In PyTorch they seem to only handle 3-D and 4-D case and are able to avoid these transposes for
  // performance. However in Tensorflow they always transpose and handle N-D case.
  if (probability_shape.NumDimensions() > 2) {
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
    onnxruntime::contrib::GetPermutationAndShape(true, probability_shape, new_shape, permutations);
    transpose_output = scan::detail::AllocateTensorInMLValue(log_prob.DataType(), new_shape, alloc);
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(cuda::Transpose(info), permutations, log_prob, *transpose_output.GetMutable<Tensor>()));
    log_prob_data = (*transpose_output.GetMutable<Tensor>()).template Data<T>();
  }

  if (OpKernel::Node().InputDefs().size() == 4) {
    const Tensor& weight = *ctx->Input<Tensor>(3);
    const TensorShape weight_shape{weight.Shape()};
    ORT_ENFORCE(1 == weight_shape.NumDimensions(), "Weights tensor is not 1-D.");
    weight_data = weight.template Data<T>();
  }

  //REVIEW(codemzs): Inefficient, just there for correctness, will get rid of this and come up with parallel
  //implementation and also avoid allocation weight buffer.
  IAllocatorUniquePtr<T> weight_data_nd = GetScratchBuffer<T>(N_D);
  T* weight_data_nd_data = weight_data_nd.get();
  ORT_ENFORCE(cudaMemset(weight_data_nd_data, 0, N_D * sizeof(T)) == cudaSuccess);
  ComputeWeightsSoftmaxCrossEntropyImpl(weight_data_nd_data, label_data, weight_data, N_D, C, ignore_index_);

  auto normalize_factor_data = GetScratchBuffer<T>(1);
  if (reduction_ == ReductionType::NONE) {
    const T normalize_factor = static_cast<T>(1);
    cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(T), cudaMemcpyHostToDevice);
  } else {
    if (reduction_ == ReductionType::MEAN) {
      // Compute buffer size in byte for reduction APIs.
      const auto buffer_size = static_cast<size_t>(
          compute_reduction_buffer_size(
              static_cast<int>(sizeof(T)), static_cast<int>(N_D)));
      // Allocate reduction buffer whose size is buffer_size bytes.
      IAllocatorUniquePtr<void> reduction_buffer = GetScratchBuffer<void>(
          buffer_size);
      reduce_sum(weight_data_nd_data,
                 normalize_factor_data.get(),
                 static_cast<int>(N_D),
                 reinterpret_cast<T*>(reduction_buffer.get()));
    } else {
      const T normalize_factor = static_cast<T>(1);
      cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(T), cudaMemcpyHostToDevice);
    }
  }

  SoftmaxCrossEntropyLossGradImpl(dY_data,
                                  log_prob_data,
                                  label_data,
                                  weight_data_nd_data,
                                  normalize_factor_data.get(),
                                  d_logit_data,
                                  N_D,
                                  C,
                                  ignore_index_,
                                  ReductionType::NONE == reduction_);

  // Transpose logit from [N, D1, D2...Dk, C] to [N, C, D1, D2 .. Dk]
  if (probability_shape.NumDimensions() > 2) {
    TensorShape logit_shape = new_shape;
    new_shape.clear();
    permutations.clear();
    onnxruntime::contrib::GetPermutationAndShape(false, logit_shape, new_shape, permutations);
    transpose_output.GetMutable<Tensor>()->Reshape(d_logit->Shape());
    d_logit->Reshape(logit_shape);
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(cuda::Transpose(info), permutations, *d_logit, *transpose_output.GetMutable<Tensor>()));
    auto* transposed_data = (*transpose_output.GetMutable<Tensor>()).template Data<T>();
    cudaMemcpyAsync(d_logit_data, transposed_data, sizeof(T) * probability_shape.Size(), cudaMemcpyDeviceToDevice);
    d_logit->Reshape(new_shape);
  }

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(Class, T, domain, version) \
  REGISTER_KERNEL_TYPED(Class, T, domain, version)     \
  template Status Class<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(SoftmaxCrossEntropy, float, kMSDomain, 1)
SPECIALIZED_COMPUTE(SoftmaxCrossEntropyGrad, float, kMSDomain, 1)

#define SPECIALIZED_COMPUTE_SPARSE(Class, T, Tin, domain, version) \
  REGISTER_KERNEL_TYPED_TWO_TYPES(Class, T, Tin, domain, version)  \
  template Status Class<T, Tin>::ComputeInternal(OpKernelContext* ctx) const;

// SPECIALIZED_COMPUTE_SPARSE(SparseSoftmaxCrossEntropy, float, int32_t, kOnnxDomain, 9)
SPECIALIZED_COMPUTE_SPARSE(SparseSoftmaxCrossEntropy, float, int64_t, kOnnxDomain, 9)
// SPECIALIZED_COMPUTE_SPARSE(SparseSoftmaxCrossEntropyGrad, float, int32_t, kOnnxDomain, 9)
SPECIALIZED_COMPUTE_SPARSE(SparseSoftmaxCrossEntropyGrad, float, int64_t, kOnnxDomain, 9)
// SPECIALIZED_COMPUTE_SPARSE(SparseSoftmaxCrossEntropy, float, int32_t, kOnnxDomain, 9)
SPECIALIZED_COMPUTE_SPARSE(SoftmaxCrossEntropyLoss, float, int64_t, kOnnxDomain, 12)
// SPECIALIZED_COMPUTE_SPARSE(SparseSoftmaxCrossEntropyGrad, float, int32_t, kOnnxDomain, 9)
SPECIALIZED_COMPUTE_SPARSE(SoftmaxCrossEntropyLossGrad, float, int64_t, kMSDomain, 1)

}  // namespace cuda
}  // namespace onnxruntime
