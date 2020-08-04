// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/providers/cuda/math/softmax.h"
#include "softmaxcrossentropy_impl.h"
#include "torch/torch.h"

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

// Type mapping for MLFloat16 to half
template <typename T>
at::ScalarType get_torch_type() {
  return at::kFloat;
}

template <>
at::ScalarType get_torch_type<MLFloat16>() {
  return at::kHalf;
}

template <>
at::ScalarType get_torch_type<float>() {
  return at::kFloat;
}

template <>
at::ScalarType get_torch_type<double>() {
  return at::kDouble;
}

template <>
at::ScalarType get_torch_type<int32_t>() {
  return at::kInt;
}

template <>
at::ScalarType get_torch_type<int64_t>() {
  return at::kLong;
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

  if (true) {
    // calculate  (label * log(softmax)) for each sample
    const T* weight_data = nullptr;
    if (OpKernel::Node().InputDefs().size() == 3) {
      const Tensor& weight = *ctx->Input<Tensor>(2);
      const TensorShape weight_shape{weight.Shape()};
      ORT_ENFORCE(weight_shape == label_shape, "The shape in weights and labels is different");
      weight_data = weight.template Data<T>();
    }

    int torch_device = 0;
    cudaGetDevice(&torch_device);

    auto torch_tensor_options = torch::TensorOptions().dtype(get_torch_type<T>()).device(torch::kCUDA, torch_device);
    torch::Tensor torch_logit = torch::zeros(c10::IntArrayRef{N, D}, torch_tensor_options);
    cudaMemcpy(torch_logit.data_ptr(), logit_data, N * D * sizeof(T), cudaMemcpyDeviceToDevice);
    ORT_ENFORCE(cudaGetLastError() == cudaSuccess);

    auto torch_label_tensor_options = torch::TensorOptions().dtype(get_torch_type<Tin>()).device(torch::kCUDA, torch_device);
    torch::Tensor torch_label = torch::zeros(c10::IntArrayRef{N}, torch_label_tensor_options);
    cudaMemcpy(torch_label.data_ptr(), label_data, N * sizeof(Tin), cudaMemcpyDeviceToDevice);
    ORT_ENFORCE(cudaGetLastError() == cudaSuccess);

    auto torch_weight_tensor_options = torch::TensorOptions().dtype(get_torch_type<T>()).device(torch::kCUDA, torch_device);
    torch::Tensor torch_weight = torch::ones(c10::IntArrayRef{N}, torch_weight_tensor_options);
    if (weight_data) {
      cudaMemcpy(torch_weight.data_ptr(), weight_data, N * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    ORT_ENFORCE(cudaGetLastError() == cudaSuccess);
    torch::Tensor torch_zero = torch::zeros(c10::IntArrayRef{N}, torch_weight_tensor_options);
    torch::Tensor mask = at::eq(torch_weight, torch_zero);
    at::masked_fill(torch_label, mask, (int64_t)-1);

    ORT_ENFORCE(cudaGetLastError() == cudaSuccess);
    torch::Tensor torch_log_prob = at::log_softmax(torch_logit, 1, c10::nullopt);
    ORT_ENFORCE(cudaGetLastError() == cudaSuccess);
    torch::Tensor torch_loss = at::nll_loss(torch_log_prob, torch_label, {}, at::Reduction::None);
    ORT_ENFORCE(cudaGetLastError() == cudaSuccess);

    cudaMemcpy(log_prob_data, torch_log_prob.data_ptr(), N * D * sizeof(T), cudaMemcpyDeviceToDevice);

    torch::Tensor torch_weighted_loss = at::mul(torch_loss, torch_weight);
    if (reduction_ == ReductionType::SUM) {
      torch::Tensor torch_reduced_loss = at::sum(torch_weighted_loss);
      cudaMemcpy(total_loss_data, torch_reduced_loss.data_ptr(), sizeof(T), cudaMemcpyDeviceToDevice);
    } else if (reduction_ == ReductionType::MEAN) {
      torch::Tensor torch_weight_sum = at::sum(torch_weight);
      torch::Tensor torch_reduced_loss = at::sum(torch_weighted_loss);
      torch::Tensor torch_final_loss = torch_reduced_loss / torch_weight_sum;
      cudaMemcpy(total_loss_data, torch_final_loss.data_ptr(), sizeof(T), cudaMemcpyDeviceToDevice);
    }
    return Status::OK();
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

  if (true) {
    // calculate  (label * log(softmax)) for each sample
    const T* weight_data = nullptr;
    if (OpKernel::Node().InputDefs().size() == 4) {
      const Tensor& weight = *ctx->Input<Tensor>(3);
      const TensorShape weight_shape{weight.Shape()};
      ORT_ENFORCE(weight_shape == label_shape, "The shape in weights and labels is different");
      weight_data = weight.template Data<T>();
    }

    int torch_device = 0;
    cudaGetDevice(&torch_device);

    // dY
    auto torch_dY_options = torch::TensorOptions().dtype(get_torch_type<T>()).device(torch::kCUDA, torch_device);
    torch::Tensor torch_dY = torch::zeros(c10::IntArrayRef{}, torch_dY_options);
    cudaMemcpy(torch_dY.data_ptr(), dY_data, sizeof(T), cudaMemcpyDeviceToDevice);
    ORT_ENFORCE(cudaGetLastError() == cudaSuccess);

    // log_prob
    auto torch_log_prob_options = torch::TensorOptions().dtype(get_torch_type<T>()).device(torch::kCUDA, torch_device);
    torch::Tensor torch_log_prob = torch::zeros(c10::IntArrayRef{N, D}, torch_log_prob_options);
    cudaMemcpy(torch_log_prob.data_ptr(), log_prob_data, N * D * sizeof(T), cudaMemcpyDeviceToDevice);
    ORT_ENFORCE(cudaGetLastError() == cudaSuccess);

    // label
    auto torch_label_tensor_options = torch::TensorOptions().dtype(get_torch_type<Tin>()).device(torch::kCUDA, torch_device);
    torch::Tensor torch_label = torch::zeros(c10::IntArrayRef{N}, torch_label_tensor_options);
    cudaMemcpy(torch_label.data_ptr(), label_data, N * sizeof(Tin), cudaMemcpyDeviceToDevice);
    ORT_ENFORCE(cudaGetLastError() == cudaSuccess);

    // weight
    auto torch_weight_tensor_options = torch::TensorOptions().dtype(get_torch_type<T>()).device(torch::kCUDA, torch_device);
    torch::Tensor torch_weight = torch::ones(c10::IntArrayRef{N, 1}, torch_weight_tensor_options);
    if (weight_data) {
      cudaMemcpy(torch_weight.data_ptr(), weight_data, N * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    torch::Tensor prob = at::exp(torch_log_prob);
    ORT_ENFORCE(cudaGetLastError() == cudaSuccess);

    torch::Tensor mask = at::one_hot(torch_label, D).toType(get_torch_type<T>());
    ORT_ENFORCE(cudaGetLastError() == cudaSuccess);

    torch::Tensor grad = torch_dY * (-mask + prob);
    ORT_ENFORCE(cudaGetLastError() == cudaSuccess);

    torch::Tensor weighted_grad = at::mul(grad, torch_weight);
    ORT_ENFORCE(cudaGetLastError() == cudaSuccess);

    if (reduction_ == ReductionType::SUM) {
      cudaMemcpy(d_logit_data, weighted_grad.data_ptr(), N * D * sizeof(T), cudaMemcpyDeviceToDevice);
    } else if (reduction_ == ReductionType::MEAN) {
      torch::Tensor torch_weight_sum = at::sum(torch_weight);
      torch::Tensor torch_final_loss = weighted_grad / torch_weight_sum;
      cudaMemcpy(d_logit_data, torch_final_loss.data_ptr(), N * D * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    return Status::OK();
  }

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

}  // namespace cuda
}  // namespace onnxruntime
