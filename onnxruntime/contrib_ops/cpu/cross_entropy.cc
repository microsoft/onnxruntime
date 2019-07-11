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
    kOnnxDomain,
    9,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SoftmaxCrossEntropy<float>);

template <typename T>
Status SoftmaxCrossEntropy<T>::Compute(OpKernelContext* context) const {
  const Tensor& logits = *context->Input<Tensor>(0);
  const Tensor& label = *context->Input<Tensor>(1);

  const TensorShape logits_shape{logits.Shape()};
  const TensorShape label_shape{label.Shape()};

  ORT_ENFORCE(label_shape == logits_shape, "The shape in logits and labels is not identical");

  int64_t N = logits_shape.SizeToDimension(logits_shape.NumDimensions() - 1);
  int64_t D = logits_shape.SizeFromDimension(logits_shape.NumDimensions() - 1);

  const TensorShape output_shape({1});
  Tensor* loss = context->Output(0, output_shape);

  const float* logits_data = logits.template Data<float>();
  const float* labels_data = label.template Data<float>();
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
    kOnnxDomain,
    9,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SoftmaxCrossEntropyGrad<float>);

template <typename T>
Status SoftmaxCrossEntropyGrad<T>::Compute(OpKernelContext* context) const {
  const Tensor& dY = *context->Input<Tensor>(0);
  const Tensor& logits = *context->Input<Tensor>(1);
  const Tensor& label = *context->Input<Tensor>(2);

  const TensorShape logits_shape{logits.Shape()};
  const TensorShape label_shape{label.Shape()};

  ORT_ENFORCE(label_shape == logits_shape, "The shape in logits and labels is not identical");

  int64_t N = logits_shape.SizeToDimension(logits_shape.NumDimensions() - 1);
  int64_t D = logits_shape.SizeFromDimension(logits_shape.NumDimensions() - 1);

  Tensor* d_logits = context->Output(0, logits_shape);

  const float* logits_data = logits.template Data<float>();
  const float* labels_data = label.template Data<float>();
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

ONNX_OPERATOR_KERNEL_EX(
    SparseSoftmaxCrossEntropy,
    kOnnxDomain,
    9,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SparseSoftmaxCrossEntropy<float>);

template <typename T>
Status SparseSoftmaxCrossEntropy<T>::Compute(OpKernelContext* context) const {
  const Tensor& logit = *context->Input<Tensor>(0);
  const Tensor& label = *context->Input<Tensor>(1);

  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};
  ORT_ENFORCE(logit_shape.NumDimensions() == label_shape.NumDimensions() + 1,
              "logits_shape must be (1 + label_shape)");
  for (size_t i = 0; i < label_shape.NumDimensions(); i++) {
    ORT_ENFORCE(label_shape[i] == logit_shape[i], "The shape in logits and labels does not match");
  }

  int64_t N = label_shape.Size();
  int64_t D = logit_shape[logit_shape.NumDimensions() - 1];

  const TensorShape output_shape({1});
  Tensor* loss = context->Output(0, output_shape);

  const float* logit_data = logit.template Data<float>();
  const int64_t* label_data = label.template Data<int64_t>();
  float* loss_data = loss->template MutableData<float>();

  // computation begins here
  const int n = gsl::narrow_cast<int>(N);
  const int d = gsl::narrow_cast<int>(D);
  const int nd = gsl::narrow_cast<int>(N * D);

  std::vector<float> shifted_logits(nd);
  std::vector<float> exp_shifted_logits(nd);
  std::vector<float> sum_exp(n);

  ComputeShareSoftmaxCrossEntropyCPU(n, d, nd, logit_data,
                                     shifted_logits.data(),
                                     exp_shifted_logits.data(),
                                     sum_exp.data());

  // log(sum(exp(logits - max_logits)))
  std::vector<float>& log_sum_exp = sum_exp;
  math::Log<float, CPUMathUtil>(n, sum_exp.data(), log_sum_exp.data(), nullptr);

  std::vector<float> loss_sample(n);

  if (OpKernel::Node().InputDefs().size() == 3) {
    const Tensor& weight = *context->Input<Tensor>(2);
    const TensorShape weight_shape{weight.Shape()};
    ORT_ENFORCE(weight_shape == label_shape, "The shape in weights and labels is different");
    const float* weight_data = weight.template Data<float>();
    for (int i = 0; i < n; i++) {
      loss_sample[i] = (log_sum_exp[i] - shifted_logits[i * d + label_data[i]]) * weight_data[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      loss_sample[i] = log_sum_exp[i] - shifted_logits[i * d + label_data[i]];
    }
  }

  // Sum over batches and classes
  math::Sum<float, CPUMathUtil>(nd, loss_sample.data(), loss_data, nullptr);

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    SparseSoftmaxCrossEntropyGrad,
    kOnnxDomain,
    9,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SparseSoftmaxCrossEntropyGrad<float>);

template <typename T>
Status SparseSoftmaxCrossEntropyGrad<T>::Compute(OpKernelContext* context) const {
  const Tensor& dY = *context->Input<Tensor>(0);
  const Tensor& logit = *context->Input<Tensor>(1);
  const Tensor& label = *context->Input<Tensor>(2);

  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};
  ORT_ENFORCE(logit_shape.NumDimensions() == label_shape.NumDimensions() + 1,
              "logits_shape must be (1 + label_shape)");
  for (size_t i = 0; i < label_shape.NumDimensions(); i++) {
    ORT_ENFORCE(label_shape[i] == logit_shape[i], "The shape in logits and labels does not match");
  }

  int64_t N = label_shape.Size();
  int64_t D = logit_shape[logit_shape.NumDimensions() - 1];

  Tensor* d_logits = context->Output(0, logit_shape);

  const float* dY_data = dY.template Data<float>();
  const float* logit_data = logit.template Data<float>();
  const int64_t* label_data = label.template Data<int64_t>();
  float* d_logits_data = d_logits->template MutableData<float>();

  // computation begins here
  const int n = gsl::narrow_cast<int>(N);
  const int d = gsl::narrow_cast<int>(D);
  const int nd = gsl::narrow_cast<int>(N * D);

  std::vector<float> shifted_logits(nd);
  std::vector<float> exp_shifted_logits(nd);
  std::vector<float> sum_exp(n);
  ComputeShareSoftmaxCrossEntropyCPU(n, d, nd, logit_data,
                                     shifted_logits.data(),
                                     exp_shifted_logits.data(),
                                     sum_exp.data());

  // backprop: prob - label, where
  //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
  //     (where the division broadcasts along the batch dimension)
  auto& prob = exp_shifted_logits;
  math::DivToCol<float, CPUMathUtil>(n, d, sum_exp.data(), prob.data(), nullptr);

  if (OpKernel::Node().InputDefs().size() == 4) {
    const Tensor& weight = *context->Input<Tensor>(3);
    const TensorShape weight_shape{weight.Shape()};
    ORT_ENFORCE(weight_shape == label_shape, "The shape in weights and labels is different");
    const float* weight_data = weight.template Data<float>();
    for (int i = 0; i < n; i++) {
      int64_t label_sample = label_data[i];
      float weight_smaple = weight_data[i] * (*dY_data);
      for (int j = 0; j < d; j++) {
        int index = i * d + j;
        d_logits_data[index] = (prob[index] - (label_sample == j)) * weight_smaple;
      }
    }
  } else {
    for (int i = 0; i < n; i++) {
      int64_t idx = i * d + label_data[i];
      prob[idx] = prob[idx] - (float)1;
    }
    // d_logits = dY * backprop, dY is a scalar
    math::Scale<float, CPUMathUtil>(nd, dY_data, prob.data(), d_logits_data, nullptr);
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
