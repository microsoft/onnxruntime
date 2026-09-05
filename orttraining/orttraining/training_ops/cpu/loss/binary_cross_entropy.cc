// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/loss/binary_cross_entropy.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "gsl/gsl"

namespace onnxruntime {
namespace contrib {

constexpr float epsilon = 1e-8;

ONNX_OPERATOR_KERNEL_EX(
    BinaryCrossEntropy,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    BinaryCrossEntropy<float>);

template <typename T>
Status BinaryCrossEntropy<T>::Compute(OpKernelContext* context) const {
  const Tensor& logit = *context->Input<Tensor>(0);
  const Tensor& label = *context->Input<Tensor>(1);

  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};

  ORT_ENFORCE(label_shape == logit_shape, "The shape of logit and label is not identical");

  int64_t N = logit_shape.SizeToDimension(logit_shape.NumDimensions() - 1);
  int64_t D = logit_shape[logit_shape.NumDimensions() - 1];

  ORT_ENFORCE(D == 1, "For binary cross entropy, Dimension should be equal to 1");

  const int n = gsl::narrow_cast<int>(N);

  Tensor* loss = context->Output(0, TensorShape({}));
  Tensor* logit_out = context->Output(1, logit_shape);

  const float* logit_data = logit.template Data<float>();
  const float* label_data = label.template Data<float>();
  float* loss_data = loss->template MutableData<float>();
  float* logit_out_data = logit_out->template MutableData<float>();
  gsl::copy(gsl::make_span(logit_data, n), gsl::make_span(logit_out_data, n));

  std::vector<T> tmp_buffer(n);
  std::vector<T> eps(n, epsilon);
  std::vector<T> ones(n, 1);

  // Pytorch's implemention clamps its log function outputs to be greater than or equal to -100
  // (https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
  // For our implementation,add epsilon.
  auto tmp_map = EigenVectorMap<T>(tmp_buffer.data(), n);
  // tmp = t * log(s) where s >= epsilon for preventing log(0)
  tmp_map = EigenVectorMap<T>((T*)label_data, n).array() *
            EigenVectorMap<T>((T*)logit_data, n).cwiseMax(EigenVectorMap<T>(eps.data(), n)).array().log();
  // tmp = t * log(s) + (1 - t) * log(1 - s) where (1 - s >= epsilon) for preventing log(0)
  tmp_map = tmp_map.array() +
            (EigenVectorMap<T>(ones.data(), n) - EigenVectorMap<T>((T*)label_data, n)).array() *
                (EigenVectorMap<T>(ones.data(), n) - EigenVectorMap<T>((T*)logit_data, n)).cwiseMax(EigenVectorMap<T>(eps.data(), n)).array().log();

  auto loss_value = tmp_map.sum();
  *loss_data = loss_value;

  if (reduction_ == ReductionType::MEAN) {
    *loss_data /= -n;
  } else if (reduction_ == ReductionType::SUM) {
    *loss_data *= -1;
  }

  return Status::OK();
}


ONNX_OPERATOR_KERNEL_EX(
    BinaryCrossEntropyGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    BinaryCrossEntropyGrad<float>);

template <typename T>
Status BinaryCrossEntropyGrad<T>::Compute(OpKernelContext* context) const {
  const Tensor& dY = *context->Input<Tensor>(0);
  const Tensor& logit = *context->Input<Tensor>(1);
  const Tensor& label = *context->Input<Tensor>(2);

  const TensorShape probability_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};

  ORT_ENFORCE(label_shape == probability_shape, "The shape of probability and label is not identical");

  int64_t N = probability_shape.SizeToDimension(probability_shape.NumDimensions() - 1);
  const int n = gsl::narrow_cast<int>(N);
  const int nd = gsl::narrow_cast<int>(probability_shape.Size());

  Tensor* d_logit = context->Output(0, probability_shape);

  const float* dY_data = dY.template Data<float>();
  const float* logit_data = logit.template Data<float>();
  const float* label_data = label.template Data<float>();
  float* d_logit_data = d_logit->template MutableData<float>();


  std::vector<T> ones(n, 1);
  std::vector<T> eps(n, epsilon);

  auto temp = EigenVectorMap<T>(d_logit_data, n);
  // temp = (1 - t) / (1 - s) where 1 - s >= epsilon for preventing 0 division
  temp = (EigenVectorMap<T>(ones.data(), n) - EigenVectorMap<T>((T*)label_data, n)).array() /
         (EigenVectorMap<T>(ones.data(), n) - EigenVectorMap<T>((T*)logit_data, n)).cwiseMax(EigenVectorMap<T>(eps.data(), n)).array();
  // temp = ((1 - t) / (1 - s)) - (t / s) where s >= epsilon for preventing 0 division
  temp = temp.array() -
         (EigenVectorMap<T>((T*)label_data, n).array() /
          EigenVectorMap<T>((T*)logit_data, n).cwiseMax(EigenVectorMap<T>(eps.data(), n)).array());

  float dY_scaled;
  if (reduction_ == ReductionType::MEAN) {
    dY_scaled = *dY_data / n;
  } else if (reduction_ == ReductionType::SUM) {
    dY_scaled = *dY_data;
  }
  // d_logit = dY * backprop, dY is a scalar
  math::Scale<float, CPUMathUtil>(nd, &dY_scaled, d_logit_data, d_logit_data, nullptr);

  return Status::OK();
}



}  // namespace contrib
}  // namespace onnxruntime
