// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "softmax_cross_entropy_loss.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/util/math.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/cpu/controlflow/scan_utils.h"
#include "gsl/gsl"

namespace onnxruntime {
namespace contrib {

template <typename T>
void ComputeShareSoftmaxCrossEntropyCPU(const int n,
                                        const int d,
                                        const int nd,
                                        const T* logit_data,
                                        T* shifted_logit,
                                        T* log_prob_data) {
  // Find the max in each batch, resulting in a tensor of shape [batch]
  // logit_max = max(logit_data)
  std::vector<T> logit_max(n);
  math::RowwiseMax<T, CPUMathUtil>(n, d, logit_data, logit_max.data(), nullptr);

  // Subtract the max in batch b from every element in batch b.
  // Broadcasts along the batch dimension.
  // shifted_logit = logit_data - logit_max
  gsl::copy(gsl::make_span(logit_data, nd), gsl::make_span(shifted_logit, nd));
  math::SubToCol<T, CPUMathUtil>(n, d, logit_max.data(), shifted_logit, nullptr);

  // exp_shifted_logit = exp(shifted_logit)
  math::Exp<T, CPUMathUtil>(nd, shifted_logit, log_prob_data, nullptr);

  // sum_exp = sum_{class} (exp_shifted_logit)
  float* sum_exp = logit_max.data();
  math::RowwiseSum<T, CPUMathUtil>(n, d, log_prob_data, sum_exp, nullptr);

  // log_sum_exp = log(sum_exp)
  float* log_sum_exp = sum_exp;
  math::Log<T, CPUMathUtil>(n, sum_exp, log_sum_exp, nullptr);

  // log_prob = shifted_logit - log(sum_exp)
  // the subtraction broadcasts along the batch dimension
  gsl::copy(gsl::make_span(shifted_logit, nd), gsl::make_span(log_prob_data, nd));
  math::SubToCol<T, CPUMathUtil>(n, d, log_sum_exp, log_prob_data, nullptr);
}

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

REGISTER_KERNEL_TYPED(SoftmaxCrossEntropyLoss, kOnnxDomain, 12, float, int32_t)
REGISTER_KERNEL_TYPED(SoftmaxCrossEntropyLoss, kOnnxDomain, 12, float, int64_t)

template <typename T1, typename T2>
Status SoftmaxCrossEntropyLoss<T1, T2>::Compute(OpKernelContext* context) const {
  const Tensor& logit = *context->Input<Tensor>(0);
  const Tensor& label = *context->Input<Tensor>(1);

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
  const T1* logit_data = logit.template Data<T1>();
  OrtValue transpose_output;
  Tensor transpose_tensor;
  std::vector<int64_t> new_shape;
  std::vector<size_t> permutations;
  AllocatorPtr alloc;

  // Transpose logit from [N, C, D1, D2 .. Dk] to [N, D1, D2...Dk, C]
  if (logit_shape.NumDimensions() > 2) {
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

    new_shape.emplace_back(logit_shape[0]);
    permutations.emplace_back(0);
    for (int index = 2; index < logit_shape.NumDimensions(); index += 1) {
      new_shape.emplace_back(logit_shape[index]);
      permutations.emplace_back(index);
    }

    new_shape.emplace_back(logit_shape[1]);
    permutations.emplace_back(1);

    transpose_output = scan::detail::AllocateTensorInMLValue(logit.DataType(), new_shape, alloc);
    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(permutations, logit, *transpose_output.GetMutable<Tensor>()));
    logit_data = (*transpose_output.GetMutable<Tensor>()).template Data<T1>();
  }

  const int n_d = gsl::narrow_cast<int>(N_D);
  const int c = gsl::narrow_cast<int>(C);
  const int n_d_c = gsl::narrow_cast<int>(N_D * C);
  Tensor* loss = context->Output(0, reduction_ == ReductionType::NONE ? TensorShape({label_shape[0]}) : TensorShape({}));

  T1* log_prob_data;
  std::vector<T1> log_prob_data_buffer(0);
  if (context->OutputCount() > 1) {
    log_prob_data = context->Output(1, logit_shape)->template MutableData<T1>();
  } else {
    log_prob_data_buffer.resize(logit_shape.Size());
    log_prob_data = log_prob_data_buffer.data();
  }

  const T2* label_data = label.template Data<T2>();
  T1* loss_data = loss->template MutableData<T1>();

  // computation begins here
  std::vector<T1> shifted_logit(n_d_c);
  ComputeShareSoftmaxCrossEntropyCPU(n_d, c, n_d_c, logit_data, shifted_logit.data(), log_prob_data);
  std::vector<T1> loss_sample_buffer(0);
  T1* loss_sample;
  if (reduction_ == ReductionType::NONE) {
    loss_sample = loss_data;
  } else {
    loss_sample_buffer.resize(n_d);
    loss_sample = loss_sample_buffer.data();
  }

  if (OpKernel::Node().InputDefs().size() == 3) {
    const Tensor& weight = *context->Input<Tensor>(2);
    const TensorShape weight_shape{weight.Shape()};
    ORT_ENFORCE(1 == weight_shape.NumDimensions(), "Weights tensor is not 1-C.");
    const T1* weight_data = weight.template Data<T1>();
    T1 sum_weight = (T1)0;

    if (reduction_ == ReductionType::MEAN) {
      for (int i = 0; i < n_d; i++) {
        if (ignore_index_ == label_data[i]) {
          loss_sample[i] = 0;
        } else {
          loss_sample[i] = -log_prob_data[i * c + label_data[i]] * weight_data[label_data[i]];
          sum_weight += weight_data[label_data[i]];
        }
      }

    } else {
      for (int i = 0; i < n_d; i++) {
        if (ignore_index_ == label_data[i]) {
          loss_sample[i] = 0;
        } else {
          loss_sample[i] = -log_prob_data[i * c + label_data[i]] * weight_data[label_data[i]];
        }
      }
    }

    if (reduction_ == ReductionType::NONE) {
      return Status::OK();
    } else {
      // Sum loss over n_d samples
      math::Sum<float, CPUMathUtil>(n_d, loss_sample, loss_data, nullptr);

      // Average sum_loss over sum_weights
      if (reduction_ == ReductionType::MEAN) {
        *loss_data /= sum_weight;
      }
    }
  } else {
    for (int i = 0; i < n_d; i++) {
      loss_sample[i] = -log_prob_data[i * c + label_data[i]];
    }

    if (reduction_ == ReductionType::NONE) {
      return Status::OK();
    } else {
      // Sum loss over n_d samples
      math::Sum<T1, CPUMathUtil>(n_d, loss_sample, loss_data, nullptr);

      if (reduction_ == ReductionType::MEAN) {
        *loss_data /= n_d;
      }
    }
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

  const TensorShape probability_shape{log_prob.Shape()};
  const TensorShape label_shape{label.Shape()};
  auto label_dims = label_shape.NumDimensions();
  ORT_ENFORCE(probability_shape.NumDimensions() == label_shape.NumDimensions() + 1,
              "probability_shape must be (1 + label_shape)");

  ORT_ENFORCE(label_shape[0] == probability_shape[0], "The shape of logit and label does not match");

  if (label_dims >= 2) {
    for (size_t i = 0; i < label_shape.NumDimensions() - 1; i++) {
      ORT_ENFORCE(label_shape[i + 1] == probability_shape[i + 2], "The shape of log probabilities and label does not match");
    }
  }

  int64_t N = probability_shape[0];
  int64_t D = probability_shape.NumDimensions() > 2 ? label_shape.Size() / N : 1;
  int64_t N_D = N * D;
  int64_t C = probability_shape.Size() / N_D;
  const int n_d = gsl::narrow_cast<int>(N_D);
  const int c = gsl::narrow_cast<int>(C);

  Tensor* d_logit = context->Output(0, probability_shape);

  const T1* dY_data = dY.template Data<T1>();
  const T1* log_prob_data = log_prob.template Data<T1>();
  const T2* label_data = label.template Data<T2>();
  T1* d_logit_data = d_logit->template MutableData<T1>();

  // computation begins here
  if (OpKernel::Node().InputDefs().size() == 4) {
    const Tensor& weight = *context->Input<Tensor>(3);
    const TensorShape weight_shape{weight.Shape()};
    ORT_ENFORCE(1 == weight_shape.NumDimensions(), "Weights tensor is not 1-C.");
    const T1* weight_data = weight.template Data<T1>();

    T1 dY_scaled = *dY_data;
    if (reduction_ == ReductionType::MEAN) {
      T1 sum_weight = (T1)0;
      for (int i = 0; i < n_d; i++) {
        if (ignore_index_ != label_data[i])
          sum_weight += weight_data[label_data[i]];
      }

      dY_scaled = *dY_data / sum_weight;
    }

    for (int i = 0; i < n_d; i++) {
      T2 label_sample = label_data[i];
      T1 weight_smaple = weight_data[label_sample] * dY_scaled;
      for (int j = 0; j < c; j++) {
        int index = i * c + j;
        if (ignore_index_ == label_sample) {
          d_logit_data[index] = 0;
        } else {
          d_logit_data[index] = (exp(log_prob_data[index]) - (label_sample == j)) * weight_smaple;
        }
      }
    }
  } else {
    T1 dY_scaled = *dY_data;
    if (reduction_ == ReductionType::MEAN) {
      dY_scaled = *dY_data / n_d;
    }

    for (int i = 0; i < n_d; i++) {
      T2 label_sample = label_data[i];
      for (int j = 0; j < c; j++) {
        int index = i * c + j;
        d_logit_data[index] = (exp(log_prob_data[index]) - (label_sample == j)) * dY_scaled;
      }
    }
  }

  // Transpose logit from [N, D1, D2...Dk, C] to [N, C, D1, D2 .. Dk]
  OrtValue transpose_output;
  Tensor transpose_tensor;
  std::vector<size_t> permutations;
  AllocatorPtr alloc;

  if (probability_shape.NumDimensions() > 2) {
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

    permutations.emplace_back(0);
    permutations.emplace_back(probability_shape.NumDimensions() - 1);
    for (int index = 1; index < probability_shape.NumDimensions() - 1; index += 1) {
      permutations.emplace_back(index);
    }

    transpose_output = scan::detail::AllocateTensorInMLValue(log_prob.DataType(), probability_shape, alloc);
    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(permutations, *d_logit, *transpose_output.GetMutable<Tensor>()));
    auto transposed_data = (*transpose_output.GetMutable<Tensor>()).template Data<T1>();
    memcpy(d_logit_data, transposed_data, probability_shape.Size() * sizeof(T1));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime