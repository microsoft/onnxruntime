// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/providers/cuda/math/softmax.h"
#include "softmaxcrossentropy_impl.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cpu/controlflow/scan_utils.h"

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
  Tensor* total_loss = ctx->Output(0, TensorShape({}));
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
                                D,
                                (Tin)-100);

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
                                    D,
                                    (Tin)-100);

  return Status::OK();
}

template <typename T, typename Tin>
Status SoftmaxCrossEntropyLoss<T, Tin>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& logit = *ctx->Input<Tensor>(0);
  const Tensor& label = *ctx->Input<Tensor>(1);

  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};
  const size_t label_dims = label_shape.NumDimensions();
  ORT_ENFORCE(logit_shape.NumDimensions() == label_dims + 1,
              "logit_shape must be (1 + label_shape)");

  ORT_ENFORCE(label_shape[0] == logit_shape[0], "The shape of logit and label does not match");

  if (label_dims >= 2) {
    for (size_t i = 0; i < label_shape.NumDimensions() - 1; i++) {
      ORT_ENFORCE(label_shape[i + 1] == logit_shape[i + 2], "The shape of logit and label does not match");
    }
  }

  int64_t N = logit_shape[0];
  int64_t D = logit_shape.NumDimensions() > 2 ? label_shape.Size() / N : 1;
  int64_t N_D = N * D;
  int64_t C = logit_shape.Size() / N_D;

  const TensorShape logit_reshape({N_D, C});
  const TensorShape label_reshape({N_D});

  IAllocatorUniquePtr<T> tmp_loss_sample = GetScratchBuffer<T>(N_D);
  const T* logit_data = logit.template Data<T>();
  const Tin* label_data = label.template Data<Tin>();

  // Output 0: Loss data.
  Tensor* total_loss = ctx->Output(0, reduction_ == ReductionType::NONE ? TensorShape({label_shape[0]}) : TensorShape({}));
  T* total_loss_data = total_loss->template MutableData<T>();

  // Output 1: Log probability data.
  T* log_prob_data;
  if (ctx->OutputCount() > 1) {
    log_prob_data = ctx->Output(1, logit_shape)->template MutableData<T>();
  } else {
    log_prob_data = GetScratchBuffer<T>(logit_shape.Size()).get();
  }

  OrtValue transpose_output;
  Tensor transpose_tensor;
  std::vector<int64_t> new_shape;
  std::vector<size_t> permutations;
  AllocatorPtr alloc;
  const OpKernelInfo& info = OpKernel::Info();
  // Transpose logit from [N, C, D1, D2 .. Dk] to [N, D1, D2...Dk, C]
  if (logit_shape.NumDimensions() > 2) {
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

    new_shape.emplace_back(logit_shape[0]);
    permutations.emplace_back(0);
    for (int index = 2; index < logit_shape.NumDimensions(); index += 1) {
      new_shape.emplace_back(logit_shape[index]);
      permutations.emplace_back(index);
    }

    new_shape.emplace_back(logit_shape[1]);
    permutations.emplace_back(1);

    transpose_output = scan::detail::AllocateTensorInMLValue(logit.DataType(), new_shape, alloc);
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(cuda::Transpose(info), permutations, logit, *transpose_output.GetMutable<Tensor>()));
    logit_data = (*transpose_output.GetMutable<Tensor>()).template Data<T>();
  }

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
    ORT_ENFORCE(1 == weight_shape.NumDimensions(), "Weights tensor is not 1-D.");
    weight_data = weight.template Data<T>();
  }

  auto normalize_factor_data = GetScratchBuffer<T>(1);
  if (reduction_ == ReductionType::SUM) {
    const T normalize_factor = static_cast<T>(1);
    cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(T), cudaMemcpyHostToDevice);
  } else if (reduction_ == ReductionType::MEAN) {
    if (weight_data == nullptr) {
      const T normalize_factor = static_cast<T>(N_D);
      cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(T), cudaMemcpyHostToDevice);
    } else {
      // Compute buffer size in byte for reduction APIs.
      const auto buffer_size = static_cast<size_t>(
          compute_reduction_buffer_size(
              static_cast<int>(sizeof(T)), static_cast<int>(N_D)));
      // Allocate reduction buffer whose size is buffer_size bytes.
      IAllocatorUniquePtr<void> reduction_buffer = GetScratchBuffer<void>(
          buffer_size);
      reduce_sum(weight_data,
                 normalize_factor_data.get(),
                 static_cast<int>(N_D),
                 reinterpret_cast<T*>(reduction_buffer.get()));
    }
  }

  SparseSoftmaxCrossEntropyImpl(log_prob_data,
                                label_data,
                                weight_data,
                                normalize_factor_data.get(),
                                tmp_loss_sample.get(),
                                N_D,
                                C,
                                ignore_index_);

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
  ORT_ENFORCE(probability_shape.NumDimensions() == label_shape.NumDimensions() + 1,
              "logit_shape must be (1 + label_shape)");

  ORT_ENFORCE(label_shape[0] == probability_shape[0], "The shape of logit and label does not match");

  if (label_shape.NumDimensions() >= 2) {
    for (size_t i = 0; i < label_shape.NumDimensions() - 1; i++) {
      ORT_ENFORCE(label_shape[i + 1] == probability_shape[i + 2], "The shape of logit and label does not match");
    }
  }

  int64_t N = probability_shape[0];
  int64_t D = probability_shape.NumDimensions() > 2 ? label_shape.Size() / N : 1;
  int64_t N_D = N * D;
  int64_t C = probability_shape.Size() / N_D;

  Tensor* d_logit = ctx->Output(0, probability_shape);

  const T* dY_data = dY.template Data<T>();
  const T* log_prob_data = log_prob.template Data<T>();
  const Tin* label_data = label.template Data<Tin>();
  T* d_logit_data = d_logit->template MutableData<T>();

  const T* weight_data = nullptr;
  if (OpKernel::Node().InputDefs().size() == 4) {
    const Tensor& weight = *ctx->Input<Tensor>(3);
    const TensorShape weight_shape{weight.Shape()};
    ORT_ENFORCE(1 == weight_shape.NumDimensions(), "Weights tensor is not 1-D.");
    weight_data = weight.template Data<T>();
  }

  auto normalize_factor_data = GetScratchBuffer<T>(1);
  if (reduction_ == ReductionType::SUM) {
    const T normalize_factor = static_cast<T>(1);
    cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(T), cudaMemcpyHostToDevice);
  } else if (reduction_ == ReductionType::MEAN) {
    if (weight_data == nullptr) {
      const T normalize_factor = static_cast<T>(N_D);
      cudaMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(T), cudaMemcpyHostToDevice);
    } else {
      // Compute buffer size in byte for reduction APIs.
      const auto buffer_size = static_cast<size_t>(
          compute_reduction_buffer_size(
              static_cast<int>(sizeof(T)), static_cast<int>(N_D)));
      // Allocate reduction buffer whose size is buffer_size bytes.
      IAllocatorUniquePtr<void> reduction_buffer = GetScratchBuffer<void>(
          buffer_size);
      reduce_sum(weight_data,
                 normalize_factor_data.get(),
                 static_cast<int>(N_D),
                 reinterpret_cast<T*>(reduction_buffer.get()));
    }
  }

  SparseSoftmaxCrossEntropyGradImpl(dY_data,
                                    log_prob_data,
                                    label_data,
                                    weight_data,
                                    normalize_factor_data.get(),
                                    d_logit_data,
                                    N_D,
                                    C,
                                    ignore_index_);

  // Transpose logit from [N, D1, D2...Dk, C] to [N, C, D1, D2 .. Dk]
  OrtValue transpose_output;
  Tensor transpose_tensor;
  std::vector<size_t> permutations;
  AllocatorPtr alloc;
  const OpKernelInfo& info = OpKernel::Info();

  if (probability_shape.NumDimensions() > 2) {
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

    permutations.emplace_back(0);
    permutations.emplace_back(probability_shape.NumDimensions() - 1);
    for (int index = 1; index < probability_shape.NumDimensions() - 1; index += 1) {
      permutations.emplace_back(index);
    }

    transpose_output = scan::detail::AllocateTensorInMLValue(log_prob.DataType(), probability_shape, alloc);
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(cuda::Transpose(info), permutations, *d_logit, *transpose_output.GetMutable<Tensor>()));
    auto transposed_data = (*transpose_output.GetMutable<Tensor>()).template Data<T>();
    cudaMemcpyAsync(d_logit_data, transposed_data, sizeof(T), cudaMemcpyDeviceToDevice);
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
