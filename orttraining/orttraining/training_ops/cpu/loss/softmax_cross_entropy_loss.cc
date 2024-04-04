// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/cpu/controlflow/scan_utils.h"
#include "orttraining/training_ops/cpu/loss/cross_entropy.h"
#include "orttraining/training_ops/cpu/loss/softmax_cross_entropy_loss.h"
#include "core/common/gsl.h"

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_VERSIONED_TYPED(OpName, Domain, StartVer, EndVer, T1, T2) \
  ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX(                                    \
      OpName,                                                                     \
      Domain,                                                                     \
      StartVer,                                                                   \
      EndVer,                                                                     \
      T1,                                                                         \
      T2,                                                                         \
      kCpuExecutionProvider,                                                      \
      KernelDefBuilder()                                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())                 \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<T2>()),             \
      OpName<T1, T2>);

#define REGISTER_KERNEL_TYPED(OpName, Domain, VER, T1, T2)            \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                  \
      OpName,                                                         \
      Domain,                                                         \
      VER,                                                            \
      T1,                                                             \
      T2,                                                             \
      kCpuExecutionProvider,                                          \
      KernelDefBuilder()                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())     \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<T2>()), \
      OpName<T1, T2>);

REGISTER_KERNEL_VERSIONED_TYPED(SoftmaxCrossEntropyLoss, kOnnxDomain, 12, 12, float, int32_t)
REGISTER_KERNEL_VERSIONED_TYPED(SoftmaxCrossEntropyLoss, kOnnxDomain, 12, 12, float, int64_t)
REGISTER_KERNEL_TYPED(SoftmaxCrossEntropyLoss, kOnnxDomain, 13, float, int32_t)
REGISTER_KERNEL_TYPED(SoftmaxCrossEntropyLoss, kOnnxDomain, 13, float, int64_t)

void GetNDCFromLogitAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape, int64_t& N_D, int64_t& C) {
  // N_D = N * D1 * D2...D*K
  N_D = label_shape.Size();
  C = logit_shape.Size() / N_D;
}

void VerifyLogitWeightAndLabelShape(const TensorShape& logit_shape,
                                    const TensorShape& label_shape,
                                    const TensorShape* weight_shape) {
  ORT_ENFORCE(nullptr == weight_shape || 1 == weight_shape->NumDimensions(), "Weights tensor is not 1-D.");

  const size_t label_dims = label_shape.NumDimensions();
  ORT_ENFORCE(logit_shape.NumDimensions() == label_dims + 1,
              "logit_shape must be (1 + label_shape)");

  ORT_ENFORCE(label_shape[0] == logit_shape[0], "The shape of logit and label does not match");

  if (label_dims >= 2) {
    for (size_t i = 0; i < label_shape.NumDimensions() - 1; i++) {
      ORT_ENFORCE(label_shape[i + 1] == logit_shape[i + 2], "The shape of logit and label does not match");
    }
  }
}

void GetPermutationAndShape(bool ncd_to_ndc, const TensorShape& tensor_shape, TensorShapeVector& new_shape,
                            std::vector<size_t>& permutations) {
  if (ncd_to_ndc) {
    new_shape.emplace_back(tensor_shape[0]);
    permutations.emplace_back(0);
    for (size_t index = 2; index < tensor_shape.NumDimensions(); index += 1) {
      new_shape.emplace_back(tensor_shape[index]);
      permutations.emplace_back(index);
    }

    new_shape.emplace_back(tensor_shape[1]);
    permutations.emplace_back(1);
  } else {
    new_shape.emplace_back(tensor_shape[0]);
    permutations.emplace_back(0);
    new_shape.emplace_back(tensor_shape[tensor_shape.NumDimensions() - 1]);
    permutations.emplace_back(tensor_shape.NumDimensions() - 1);
    for (size_t index = 1; index < tensor_shape.NumDimensions() - 1; index += 1) {
      new_shape.emplace_back(tensor_shape[index]);
      permutations.emplace_back(index);
    }
  }
}

template <typename T1, typename T2>
Status SoftmaxCrossEntropyLoss<T1, T2>::Compute(OpKernelContext* context) const {
  const Tensor& logit = *context->Input<Tensor>(0);
  const Tensor& label = *context->Input<Tensor>(1);
  const Tensor* p_weight = context->Input<Tensor>(2);
  const Tensor* p_ignore_index = context->Input<Tensor>(3);
  int64_t ignore_index = ignore_index_;
  if (p_ignore_index) {
    ORT_ENFORCE(p_ignore_index->Shape().IsScalar(), "ignore_index should be a scalar.");
    ignore_index = *(p_ignore_index->template Data<int64_t>());
  }

  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};
  VerifyLogitWeightAndLabelShape(logit_shape, label_shape, p_weight ? &p_weight->Shape() : nullptr);

  // N_D = N * D1 * D2...D*K
  int64_t N_D = 0;
  int64_t C = 0;
  GetNDCFromLogitAndLabelShape(logit_shape, label_shape, N_D, C);
  const T1* logit_data = logit.template Data<T1>();
  OrtValue transpose_output;
  TensorShapeVector new_shape;
  std::vector<size_t> permutations;
  AllocatorPtr alloc;

  // Transpose logit from [N, C, D1, D2 .. Dk] to [N, D1, D2...Dk, C]
  if (logit_shape.NumDimensions() > 2) {
    GetPermutationAndShape(true, logit_shape, new_shape, permutations);
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
    transpose_output = scan::detail::AllocateTensorInMLValue(logit.DataType(), new_shape, alloc);
    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(permutations, logit, *transpose_output.GetMutable<Tensor>()));
    logit_data = (*transpose_output.GetMutable<Tensor>()).template Data<T1>();
  }

  const int n_d = gsl::narrow_cast<int>(N_D);
  const int c = gsl::narrow_cast<int>(C);
  const uint64_t n_d_c = N_D * C;
  Tensor* loss = context->Output(0, reduction_ == ReductionType::NONE ? TensorShape(label.Shape()) : TensorShape({}));
  T1* log_prob_data;
  std::vector<T1> log_prob_data_buffer(0);

  Tensor* log_prob = nullptr;
  if (context->OutputCount() > 1) {
    log_prob = context->Output(1, logit_shape);
    log_prob_data = log_prob->template MutableData<T1>();
  } else {
    log_prob_data_buffer.resize(narrow<size_t>(logit_shape.Size()));
    log_prob_data = log_prob_data_buffer.data();
  }

  const T2* label_data = label.template Data<T2>();
  T1* loss_data = loss->template MutableData<T1>();
  std::vector<T1> shifted_logit(narrow<size_t>(n_d_c));
  ORT_ENFORCE(n_d_c <= static_cast<uint64_t>(std::numeric_limits<Eigen::Index>::max()));
  ComputeShareSoftmaxCrossEntropyCPU(n_d, c, static_cast<Eigen::Index>(n_d_c), logit_data, shifted_logit.data(),
                                     log_prob_data);
  std::vector<T1> loss_sample_buffer(0);
  T1* loss_sample;
  if (reduction_ == ReductionType::NONE) {
    loss_sample = loss_data;
  } else {
    loss_sample_buffer.resize(n_d);
    loss_sample = loss_sample_buffer.data();
  }

  std::memset(loss_sample, 0, sizeof(T1) * n_d);
  // Case where weights are provided.
  // Review(codemzs):
  // 1) Can merge the two cases in one if we assume default weight values to be 1s but will need to
  // multiply even when weights are not provided.
  // 2) Use parallel for below everywhere.
  if (p_weight) {
    const Tensor& weight = *p_weight;
    const T1* weight_data = weight.template Data<T1>();
    T1 sum_weight = (T1)0;

    // Compute weighed loss for each sample while summing weights for unignored target/label values.
    if (reduction_ == ReductionType::MEAN) {
      for (ptrdiff_t i = 0; i < n_d; i++) {
        if (ignore_index == label_data[i]) {
          loss_sample[i] = 0;
        } else {
          loss_sample[i] = -log_prob_data[i * c + label_data[i]] * weight_data[label_data[i]];
          sum_weight += weight_data[label_data[i]];
        }
      }
    } else {
      for (ptrdiff_t i = 0; i < n_d; i++) {
        if (ignore_index == label_data[i]) {
          loss_sample[i] = 0;
        } else {
          loss_sample[i] = -log_prob_data[i * c + label_data[i]] * weight_data[label_data[i]];
        }
      }
    }

    // Return loss.
    if (reduction_ != ReductionType::NONE) {
      // Sum loss over n_d samples
      math::Sum<float, CPUMathUtil>(n_d, loss_sample, loss_data, nullptr);
      // Average sum_loss over sum_weights
      if ((reduction_ == ReductionType::MEAN) && (sum_weight != 0)) {
        *loss_data /= sum_weight;
      }
    }
  } else {
    // Compute loss for each sample while counting unignored target/label values.
    int unignored_samples = 0;
    for (ptrdiff_t i = 0; i < n_d; i++) {
      if (ignore_index == label_data[i]) {
        loss_sample[i] = 0;
      } else {
        loss_sample[i] = -log_prob_data[i * c + label_data[i]];
        unignored_samples += 1;
      }
    }

    // Return loss.
    if (reduction_ != ReductionType::NONE) {
      // Sum loss over n_d samples
      math::Sum<T1, CPUMathUtil>(n_d, loss_sample, loss_data, nullptr);
      if ((reduction_ == ReductionType::MEAN) && (unignored_samples != 0)) {
        *loss_data /= unignored_samples;
      }
    }
  }

  // Transpose log probabilities from [N, D1, D2...Dk, C] to [N, C, D1, D2 .. Dk].
  if (logit_shape.NumDimensions() > 2 && log_prob != nullptr) {
    TensorShape log_prob_shape = new_shape;
    new_shape.clear();
    permutations.clear();
    GetPermutationAndShape(false, log_prob_shape, new_shape, permutations);
    auto* transposed_data = (*transpose_output.GetMutable<Tensor>()).template Data<T1>();
    transpose_output.GetMutable<Tensor>()->Reshape(log_prob->Shape());
    log_prob->Reshape(log_prob_shape);
    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(permutations, *log_prob, *transpose_output.GetMutable<Tensor>()));
    memcpy(log_prob_data, transposed_data, narrow<size_t>(log_prob_shape.Size() * sizeof(T1)));
    log_prob->Reshape(new_shape);
  }

  return Status::OK();
}

REGISTER_KERNEL_TYPED(SoftmaxCrossEntropyLossGrad, kMSDomain, 1, float, int32_t)
REGISTER_KERNEL_TYPED(SoftmaxCrossEntropyLossGrad, kMSDomain, 1, float, int64_t)

template <typename T1, typename T2>
Status SoftmaxCrossEntropyLossGrad<T1, T2>::Compute(OpKernelContext* context) const {
  const Tensor& dY = *context->Input<Tensor>(0);
  const Tensor& log_prob = *context->Input<Tensor>(1);
  const Tensor& label = *context->Input<Tensor>(2);
  const Tensor* p_weight = context->Input<Tensor>(3);
  const Tensor* p_ignore_index = context->Input<Tensor>(4);
  int64_t ignore_index = ignore_index_;
  if (p_ignore_index) {
    ORT_ENFORCE(p_ignore_index->Shape().IsScalar(), "ignore_index should be a scalar.");
    ignore_index = *(p_ignore_index->template Data<int64_t>());
  }

  const TensorShape probability_shape{log_prob.Shape()};
  const TensorShape label_shape{label.Shape()};
  VerifyLogitWeightAndLabelShape(probability_shape, label_shape, p_weight ? &p_weight->Shape() : nullptr);

  // N_D = N * D1 * D2...D*K
  int64_t N_D = 0;
  int64_t C = 0;
  GetNDCFromLogitAndLabelShape(probability_shape, label_shape, N_D, C);
  const T1* dY_data = dY.template Data<T1>();
  const T1* log_prob_data = log_prob.template Data<T1>();
  const T2* label_data = label.template Data<T2>();
  Tensor* d_logit = context->Output(0, probability_shape);
  T1* d_logit_data = d_logit->template MutableData<T1>();
  std::memset(d_logit_data, 0, narrow<size_t>(sizeof(T1) * N_D));
  OrtValue transpose_output;
  TensorShapeVector new_shape;
  std::vector<size_t> permutations;
  AllocatorPtr alloc;

  // Transpose logit from [N, C, D1, D2 .. Dk] to [N, D1, D2...Dk, C]
  if (probability_shape.NumDimensions() > 2) {
    GetPermutationAndShape(true, probability_shape, new_shape, permutations);
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
    transpose_output = scan::detail::AllocateTensorInMLValue(log_prob.DataType(), new_shape, alloc);
    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(permutations, log_prob, *transpose_output.GetMutable<Tensor>()));
    log_prob_data = (*transpose_output.GetMutable<Tensor>()).template Data<T1>();
  }

  static constexpr double cost = 1.0;
  auto* tp = context->GetOperatorThreadPool();
  if (p_weight) {
    const Tensor& weight = *p_weight;
    const T1* weight_data = weight.template Data<T1>();

    if (reduction_ == ReductionType::NONE) {
      concurrency::ThreadPool::TryParallelFor(
          tp, narrow<ptrdiff_t>(N_D * C), cost,
          [&label_data, &weight_data, &d_logit_data, &log_prob_data, ignore_index, C, &dY_data](
              std::ptrdiff_t begin, std::ptrdiff_t end) {
            for (std::ptrdiff_t index = begin; index != end; ++index) {
              int64_t row = index / C;
              int64_t col = index % C;
              T2 label_sample = label_data[row];
              T1 weight_smaple = weight_data[label_sample] * dY_data[row];
              if (ignore_index == label_sample) {
                d_logit_data[index] = 0;
              } else {
                d_logit_data[index] = (exp(log_prob_data[index]) - (label_sample == col)) * weight_smaple;
              }
            }
          });
    } else {
      T1 dY_scaled = *dY_data;
      if (reduction_ == ReductionType::MEAN) {
        double sum_weight = (double)0;
        for (int64_t i = 0; i < N_D; i++) {
          if (ignore_index != label_data[i]) {
            sum_weight += weight_data[label_data[i]];
          }
        }

        if (sum_weight != 0) {
          dY_scaled = static_cast<T1>(*dY_data / sum_weight);
        }
      }

      concurrency::ThreadPool::TryParallelFor(
          tp, narrow<ptrdiff_t>(N_D * C), cost,
          [&label_data, &weight_data, dY_scaled, &d_logit_data, &log_prob_data, ignore_index, C](
              std::ptrdiff_t begin, std::ptrdiff_t end) {
            for (std::ptrdiff_t index = begin; index != end; ++index) {
              int64_t row = index / C;
              int64_t col = index % C;
              T2 label_sample = label_data[row];
              T1 weight_smaple = weight_data[label_sample] * dY_scaled;
              if (ignore_index == label_sample) {
                d_logit_data[index] = 0;
              } else {
                d_logit_data[index] = (exp(log_prob_data[index]) - (label_sample == col)) * weight_smaple;
              }
            }
          });
    }
  } else {
    if (reduction_ == ReductionType::NONE) {
      concurrency::ThreadPool::TryParallelFor(
          tp, narrow<ptrdiff_t>(N_D * C), cost,
          [&label_data, &d_logit_data, &log_prob_data, ignore_index, C, &dY_data](
              std::ptrdiff_t begin, std::ptrdiff_t end) {
            for (std::ptrdiff_t index = begin; index != end; ++index) {
              int64_t row = index / C;
              int64_t col = index % C;
              T2 label_sample = label_data[row];
              if (ignore_index == label_sample) {
                d_logit_data[index] = 0;
              } else {
                d_logit_data[index] = (exp(log_prob_data[index]) - (label_sample == col)) * dY_data[row];
              }
            }
          });

    } else {
      T1 dY_scaled = *dY_data;
      uint64_t unignored_sample_count = 0;
      for (int64_t i = 0; i < N_D; i++) {
        if (ignore_index != label_data[i]) {
          unignored_sample_count += 1;
        }
      }

      if ((reduction_ == ReductionType::MEAN) && (unignored_sample_count != 0)) {
        dY_scaled = static_cast<T1>(*dY_data / unignored_sample_count);
      }

      concurrency::ThreadPool::TryParallelFor(
          tp, narrow<ptrdiff_t>(N_D * C), cost,
          [&label_data, &d_logit_data, &log_prob_data, ignore_index, C, &dY_scaled](
              std::ptrdiff_t begin, std::ptrdiff_t end) {
            for (std::ptrdiff_t index = begin; index != end; ++index) {
              int64_t row = index / C;
              int64_t col = index % C;
              T2 label_sample = label_data[row];
              if (ignore_index == label_sample) {
                d_logit_data[index] = 0;
              } else {
                d_logit_data[index] = (exp(log_prob_data[index]) - (label_sample == col)) * dY_scaled;
              }
            }
          });
    }
  }

  // Transpose logit from [N, D1, D2...Dk, C] to [N, C, D1, D2 .. Dk]
  if (probability_shape.NumDimensions() > 2) {
    TensorShape logit_shape = new_shape;
    new_shape.clear();
    permutations.clear();
    GetPermutationAndShape(false, logit_shape, new_shape, permutations);
    transpose_output.GetMutable<Tensor>()->Reshape(d_logit->Shape());
    d_logit->Reshape(logit_shape);
    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(permutations, *d_logit, *transpose_output.GetMutable<Tensor>()));
    auto* transposed_data = (*transpose_output.GetMutable<Tensor>()).template Data<T1>();
    memcpy(d_logit_data, transposed_data, narrow<size_t>(probability_shape.Size() * sizeof(T1)));
    d_logit->Reshape(new_shape);
  }

  // Bias.
  const Tensor* p_bias = context->Input<Tensor>(5);
  if (p_bias) {
    ORT_ENFORCE(probability_shape.Size() == p_bias->Shape().Size());
    const T1* bias_data = p_bias->Data<T1>();
    concurrency::ThreadPool::TryParallelFor(
        tp, narrow<ptrdiff_t>(probability_shape.Size()), cost,
        [&d_logit_data, &bias_data](
            std::ptrdiff_t begin, std::ptrdiff_t end) {
          for (std::ptrdiff_t index = begin; index != end; ++index) {
            d_logit_data[index] += bias_data[index];
          }
        });
  }

  return Status::OK();
}

#define REGISTER_KERNEL_INTERNAL_TYPED(OpName, ClassName, T1, T2)                                     \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(OpName, kMSDomain, 1, T1, T2, kCpuExecutionProvider,              \
                                    KernelDefBuilder()                                                \
                                        .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())       \
                                        .TypeConstraint("TOut", DataTypeImpl::GetTensorType<T1>())    \
                                        .TypeConstraint("Tind", DataTypeImpl::GetTensorType<T2>())    \
                                        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
                                    ClassName<T1, T2>);

REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternal, SoftmaxCrossEntropyLoss, float, int32_t)
REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternal, SoftmaxCrossEntropyLoss, float, int64_t)
REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternalGrad, SoftmaxCrossEntropyLossGrad, float, int32_t)
REGISTER_KERNEL_INTERNAL_TYPED(SoftmaxCrossEntropyLossInternalGrad, SoftmaxCrossEntropyLossGrad, float, int64_t)

}  // namespace contrib
}  // namespace onnxruntime
