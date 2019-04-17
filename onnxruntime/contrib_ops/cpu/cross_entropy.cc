// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cross_entropy.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/util/math.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "gsl/gsl_algorithm"
#include "gsl/gsl_util"

namespace onnxruntime {
namespace contrib {

void ComputeShareSoftmaxCrossEntropyCPU(const int n,
                                        const int d,
                                        const int nd,
                                        const float* logits_data,
                                        float* shifted_logits,
                                        float* exp_shifted_logits,
                                        float* sum_exp) {
  // Find the max in each batch, resulting in a tensor of shape [batch]
  // logits_max = max(logits_data)
  std::vector<float> logits_max(n);
  math::RowwiseMax<float, CPUMathUtil>(n, d, logits_data, logits_max.data(), nullptr);

  // Subtract the max in batch b from every element in batch b.
  // Broadcasts along the batch dimension.
  // shifted_logits = logits_data - logits_max
  gsl::copy(gsl::make_span(logits_data, nd), gsl::make_span(shifted_logits, nd));
  math::SubToCol<float, CPUMathUtil>(n, d, logits_max.data(), shifted_logits, nullptr);

  // exp_shifted_logits = exp(shifted_logits)
  math::Exp<float, CPUMathUtil>(nd, shifted_logits, exp_shifted_logits, nullptr);

  // sum_exp = sum_{class} (exp_shifted_logits)
  math::RowwiseSum<float, CPUMathUtil>(n, d, exp_shifted_logits, sum_exp, nullptr);
}

ONNX_OPERATOR_KERNEL_EX(
    SoftmaxCrossEntropy,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SoftmaxCrossEntropy<float>);

template <typename T>
Status SoftmaxCrossEntropy<T>::Compute(OpKernelContext* context) const {
  const Tensor& logits = *context->Input<Tensor>(0);
  const Tensor& lable = *context->Input<Tensor>(1);

  const TensorShape logits_shape{logits.Shape()};
  const TensorShape label_shape{lable.Shape()};

  ORT_ENFORCE(logits_shape.NumDimensions() == 2, "logits must be 2-dimensional");
  ORT_ENFORCE(label_shape == logits_shape, "The shape in logits and lable is not identical");

  int64_t N = logits_shape[0];
  int64_t D = logits_shape[1];

  const TensorShape output_shape({1});
  Tensor* loss = context->Output(0, output_shape);

  const float* logits_data = logits.template Data<float>();
  const float* labels_data = lable.template Data<float>();
  float* loss_data = loss->template MutableData<float>();

  // computation begins here
  const int n = gsl::narrow_cast<int>(N);
  const int d = gsl::narrow_cast<int>(D);
  const int nd = gsl::narrow_cast<int>(N * D);

  std::vector<float> shifted_logits(nd);
  std::vector<float> exp_shifted_logits(nd);
  std::vector<float> sum_exp(n);
  ComputeShareSoftmaxCrossEntropyCPU(n, d, nd, logits_data,
                                     shifted_logits.data(),
                                     exp_shifted_logits.data(),
                                     sum_exp.data());

  // log(sum(exp(logits - max_logits)))
  std::vector<float>& log_sum_exp = sum_exp;
  math::Log<float, CPUMathUtil>(n, sum_exp.data(), log_sum_exp.data(), nullptr);

  // -sum(labels *
  //    ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
  // along classes
  // (The subtraction broadcasts along the batch dimension.)
  auto& sub = shifted_logits;
  math::SubToCol<float, CPUMathUtil>(n, d, log_sum_exp.data(), sub.data(), nullptr);

  auto& mul = sub;
  math::Mul<float, CPUMathUtil>(nd, labels_data, sub.data(), mul.data(), nullptr);

  // Sum over batches and classes
  math::Sum<float, CPUMathUtil>(nd, mul.data(), loss_data, nullptr);
  *loss_data *= -1;

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    SoftmaxCrossEntropyGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SoftmaxCrossEntropyGrad<float>);

template <typename T>
Status SoftmaxCrossEntropyGrad<T>::Compute(OpKernelContext* context) const {
  const Tensor& dY = *context->Input<Tensor>(0);
  const Tensor& logits = *context->Input<Tensor>(1);
  const Tensor& lable = *context->Input<Tensor>(2);

  const TensorShape logits_shape{logits.Shape()};
  const TensorShape label_shape{lable.Shape()};

  ORT_ENFORCE(logits_shape.NumDimensions() == 2, "logits must be 2-dimensional");
  ORT_ENFORCE(label_shape == logits_shape, "The shape in logits and lable is not identical");

  int64_t N = logits_shape[0];
  int64_t D = logits_shape[1];

  Tensor* d_logits = context->Output(0, logits_shape);

  const float* logits_data = logits.template Data<float>();
  const float* labels_data = lable.template Data<float>();
  const float* dY_data = dY.template Data<float>();
  float* d_logits_data = d_logits->template MutableData<float>();

  // computation begins here
  const int n = gsl::narrow_cast<int>(N);
  const int d = gsl::narrow_cast<int>(D);
  const int nd = gsl::narrow_cast<int>(N * D);

  std::vector<float> shifted_logits(nd);
  std::vector<float> exp_shifted_logits(nd);
  std::vector<float> sum_exp(n);
  ComputeShareSoftmaxCrossEntropyCPU(n, d, nd, logits_data,
                                     shifted_logits.data(),
                                     exp_shifted_logits.data(),
                                     sum_exp.data());

  // backprop: prob - labels, where
  //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
  //     (where the division broadcasts along the batch dimension)
  auto& prob = exp_shifted_logits;
  math::DivToCol<float, CPUMathUtil>(n, d, sum_exp.data(), prob.data(), nullptr);
  math::Sub<float, CPUMathUtil>(nd, prob.data(), labels_data, d_logits_data, nullptr);

  // d_logits = dY * backprop, dY is a scalar
  math::Scale<float, CPUMathUtil>(nd, dY_data, d_logits_data, d_logits_data, nullptr);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
