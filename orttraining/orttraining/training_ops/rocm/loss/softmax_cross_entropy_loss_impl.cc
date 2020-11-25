// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/math/softmax.h"
#include "core/providers/rocm/reduction/reduction_functions.h"
#include "core/providers/rocm/tensor/transpose.h"
#include "core/providers/cpu/controlflow/scan_utils.h"
#include "orttraining/training_ops/cpu/loss/softmax_cross_entropy_loss.h"
#include "orttraining/training_ops/rocm/loss/softmax_cross_entropy_loss_impl.h"

namespace onnxruntime {
namespace rocm {

#define REGISTER_KERNEL_VERSIONED_TYPED_TWO_TYPES(Class, T, Tin, domain, startver, endver)  \
  ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX(                                              \
      Class,                                                                                \
      domain,                                                                               \
      startver, endver,                                                                     \
      T, Tin,                                                                               \
      kRocmExecutionProvider,                                                               \
      KernelDefBuilder()                                                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                            \
          .TypeConstraint("Tin", DataTypeImpl::GetTensorType<Tin>()),                       \
      Class<T, Tin>);

#define REGISTER_KERNEL_TYPED_TWO_TYPES(Class, T, Tin, domain, version) \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                    \
      Class,                                                            \
      domain,                                                           \
      version,                                                          \
      T, Tin,                                                           \
      kRocmExecutionProvider,                                           \
      KernelDefBuilder()                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("Tin", DataTypeImpl::GetTensorType<Tin>()),   \
      Class<T, Tin>);

template <typename T, typename Tin>
Status SoftmaxCrossEntropyLoss<T, Tin>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& logit = *ctx->Input<Tensor>(0);
  const Tensor& label = *ctx->Input<Tensor>(1);
  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};
  onnxruntime::contrib::VerifyLogitWeightAndLabelShape(logit_shape, label_shape,
                                                       OpKernel::Node().InputDefs().size() == 3 ? &(*(ctx->Input<Tensor>(2))).Shape() : nullptr);

  // N_D = N * D1 * D2...D*K
  int64_t N_D;
  int64_t C;
  onnxruntime::contrib::GetNDCFromLogitAndLabelShape(logit_shape, label_shape, N_D, C);
  const TensorShape logit_reshape({N_D, C});
  const TensorShape label_reshape({N_D});
  Tensor* total_loss = ctx->Output(0, reduction_ == ReductionType::NONE ? TensorShape(label.Shape()) : TensorShape({}));
  T* total_loss_data = total_loss->template MutableData<T>();
  T* tmp_loss_sample_buffer = nullptr;
  IAllocatorUniquePtr<T> tmp_loss_sample;
  if (reduction_ == ReductionType::NONE) {
    tmp_loss_sample_buffer = total_loss_data;
  } else {
    tmp_loss_sample = GetScratchBuffer<T>(N_D);
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
    log_prob_scratch_buffer = GetScratchBuffer<T>(logit_shape.Size());
    log_prob_data = log_prob_scratch_buffer.get();
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
    onnxruntime::contrib::GetPermutationAndShape(true, logit_shape, new_shape, permutations);
    transpose_output = scan::detail::AllocateTensorInMLValue(logit.DataType(), new_shape, alloc);
    ORT_RETURN_IF_ERROR(rocm::Transpose::DoTranspose(rocm::Transpose(info), permutations, logit, *transpose_output.GetMutable<Tensor>()));
    logit_data = (*transpose_output.GetMutable<Tensor>()).template Data<T>();
  }

  // calculate logsoftmax
  auto status = SoftMaxComputeHelper<T, true>(logit_data,
                                              logit_reshape,
                                              log_prob_data,
                                              MiopenHandle(),
                                              1);
  ORT_RETURN_IF_ERROR(status);

  const T* weight_data = nullptr;
  if (OpKernel::Node().InputDefs().size() == 3) {
    const Tensor& weight = *ctx->Input<Tensor>(2);
    weight_data = weight.template Data<T>();
  }

  IAllocatorUniquePtr<T> weight_data_nd = GetScratchBuffer<T>(N_D);
  T* weight_data_nd_data = weight_data_nd.get();
  HIP_RETURN_IF_ERROR(hipMemsetAsync(weight_data_nd_data, 0, N_D * sizeof(T)));
  ComputeWeightsSoftmaxCrossEntropyImpl(label_data, weight_data, N_D, C, ignore_index_, weight_data_nd_data);

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
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(T), hipMemcpyHostToDevice));
  }

  SoftmaxCrossEntropyLossImpl(log_prob_data,
                              label_data,
                              weight_data_nd_data,
                              normalize_factor_data.get(),
                              N_D,
                              C,
                              ignore_index_,
                              tmp_loss_sample_buffer);

  // Transpose log probability from [N, D1, D2...Dk, C] to [N, C, D1, D2 .. Dk].
  if (logit_shape.NumDimensions() > 2 && log_prob != nullptr) {
    TensorShape log_prob_shape = new_shape;
    new_shape.clear();
    permutations.clear();
    onnxruntime::contrib::GetPermutationAndShape(false, log_prob_shape, new_shape, permutations);
    auto* transposed_data = (*transpose_output.GetMutable<Tensor>()).template MutableData<T>();
    transpose_output.GetMutable<Tensor>()->Reshape(log_prob->Shape());
    log_prob->Reshape(log_prob_shape);
    ORT_RETURN_IF_ERROR(rocm::Transpose::DoTranspose(rocm::Transpose(info), permutations, *log_prob, *transpose_output.GetMutable<Tensor>()));
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(log_prob_data, transposed_data, sizeof(T) * logit_shape.Size(), hipMemcpyDeviceToDevice));
    log_prob->Reshape(new_shape);
  }

  if (reduction_ != ReductionType::NONE) {
    // ReduceSum on loss_per_sample
    // std::vector<int64_t> output_dims(1, 1);
    // ReduceKernelShared<T, T>(
    //     tmp_loss_sample_buffer,
    //     label_reshape,
    //     total_loss_data,
    //     TensorShape({}),
    //     MIOPEN_REDUCE_TENSOR_ADD,
    //     output_dims);
    // Compute buffer size in byte for reduction APIs.
    const auto tmp_buffer_size = static_cast<size_t>(
        compute_reduction_buffer_size(
            static_cast<int>(sizeof(T)), static_cast<int>(N_D)));
    // Allocate reduction buffer whose size is buffer_size bytes.
    IAllocatorUniquePtr<void> tmp_reduction_buffer = GetScratchBuffer<void>(
        tmp_buffer_size);
    reduce_sum(tmp_loss_sample_buffer,
               total_loss_data,
               static_cast<int>(N_D),
               reinterpret_cast<T*>(tmp_reduction_buffer.get()));
    return Status::OK();
  }

  return Status::OK();
}

template <typename T, typename Tin>
Status SoftmaxCrossEntropyLossGrad<T, Tin>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& dY = *ctx->Input<Tensor>(0);
  const Tensor& log_prob = *ctx->Input<Tensor>(1);
  const Tensor& label = *ctx->Input<Tensor>(2);
  const TensorShape probability_shape{log_prob.Shape()};
  const TensorShape label_shape{label.Shape()};
  onnxruntime::contrib::VerifyLogitWeightAndLabelShape(probability_shape, label_shape,
                                                       OpKernel::Node().InputDefs().size() == 4 ? &(*(ctx->Input<Tensor>(3))).Shape() : nullptr);

  // N_D = N * D1 * D2...D*K
  int64_t N_D;
  int64_t C;
  onnxruntime::contrib::GetNDCFromLogitAndLabelShape(probability_shape, label_shape, N_D, C);
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
  if (probability_shape.NumDimensions() > 2) {
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
    onnxruntime::contrib::GetPermutationAndShape(true, probability_shape, new_shape, permutations);
    transpose_output = scan::detail::AllocateTensorInMLValue(log_prob.DataType(), new_shape, alloc);
    ORT_RETURN_IF_ERROR(rocm::Transpose::DoTranspose(rocm::Transpose(info), permutations, log_prob, *transpose_output.GetMutable<Tensor>()));
    log_prob_data = (*transpose_output.GetMutable<Tensor>()).template Data<T>();
  }

  if (OpKernel::Node().InputDefs().size() == 4) {
    const Tensor& weight = *ctx->Input<Tensor>(3);
    weight_data = weight.template Data<T>();
  }

  IAllocatorUniquePtr<T> weight_data_nd = GetScratchBuffer<T>(N_D);
  T* weight_data_nd_data = weight_data_nd.get();
  HIP_RETURN_IF_ERROR(hipMemsetAsync(weight_data_nd_data, 0, N_D * sizeof(T)));
  ComputeWeightsSoftmaxCrossEntropyImpl(label_data, weight_data, N_D, C, ignore_index_, weight_data_nd_data);
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
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(normalize_factor_data.get(), &normalize_factor, sizeof(T), hipMemcpyHostToDevice));
  }

  SoftmaxCrossEntropyLossGradImpl(dY_data,
                                  log_prob_data,
                                  label_data,
                                  weight_data_nd_data,
                                  normalize_factor_data.get(),
                                  N_D,
                                  C,
                                  ReductionType::NONE == reduction_,
                                  d_logit_data);

  // Transpose logit from [N, D1, D2...Dk, C] to [N, C, D1, D2 .. Dk]
  if (probability_shape.NumDimensions() > 2) {
    TensorShape logit_shape = new_shape;
    new_shape.clear();
    permutations.clear();
    onnxruntime::contrib::GetPermutationAndShape(false, logit_shape, new_shape, permutations);
    transpose_output.GetMutable<Tensor>()->Reshape(d_logit->Shape());
    d_logit->Reshape(logit_shape);
    ORT_RETURN_IF_ERROR(rocm::Transpose::DoTranspose(rocm::Transpose(info), permutations, *d_logit, *transpose_output.GetMutable<Tensor>()));
    auto* transposed_data = (*transpose_output.GetMutable<Tensor>()).template Data<T>();
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(d_logit_data, transposed_data, sizeof(T) * probability_shape.Size(), hipMemcpyDeviceToDevice));
    d_logit->Reshape(new_shape);
  }

  return Status::OK();
}

#define SPECIALIZED_VERSIONED_COMPUTE_SPARSE(Class, T, Tin, domain, startver, endvar) \
  REGISTER_KERNEL_VERSIONED_TYPED_TWO_TYPES(Class, T, Tin, domain, startver, endvar)

#define SPECIALIZED_COMPUTE_SPARSE(Class, T, Tin, domain, version) \
  REGISTER_KERNEL_TYPED_TWO_TYPES(Class, T, Tin, domain, version)  \
  template Status Class<T, Tin>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_VERSIONED_COMPUTE_SPARSE(SoftmaxCrossEntropyLoss, float, int64_t, kOnnxDomain, 12, 12)
SPECIALIZED_COMPUTE_SPARSE(SoftmaxCrossEntropyLoss, float, int64_t, kOnnxDomain, 13)
SPECIALIZED_COMPUTE_SPARSE(SoftmaxCrossEntropyLossGrad, float, int64_t, kMSDomain, 1)

}  // namespace rocm
}  // namespace onnxruntime
