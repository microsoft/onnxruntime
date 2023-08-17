// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/optimizer/adamw/adamw.h"
#include "orttraining/training_ops/cpu/optimizer/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/TensorSeq.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace contrib {

Status AdamWOptimizerBase::PrepareForCompute(OpKernelContext* ctx, AdamWOptimizerBase::Prepare& prepare) const {
  prepare.learning_rate = ctx->Input<Tensor>(0);
  prepare.step = ctx->Input<Tensor>(1);
  prepare.weights = ctx->Input<TensorSeq>(2);
  prepare.gradients = ctx->Input<TensorSeq>(3);
  prepare.momentums_1 = ctx->Input<TensorSeq>(4);
  prepare.momentums_2 = ctx->Input<TensorSeq>(5);

  prepare.num_of_weights = prepare.weights->Size();
  size_t num_of_gradients = prepare.gradients->Size();
  size_t num_of_momentums_1 = prepare.momentums_1->Size();
  size_t num_of_momentums_2 = prepare.momentums_2->Size();

  // Check the number of weights, gradients, momentums matchs.
  ORT_RETURN_IF_NOT(prepare.num_of_weights > 0, "Invalid count of tensors in Seq<Tensor>.");
  ORT_RETURN_IF_NOT(prepare.num_of_weights == num_of_gradients, "Number of weights and gradients mismatch.");
  ORT_RETURN_IF_NOT(num_of_gradients == num_of_momentums_1, "Number of gradients and momentums_1 mismatch.");
  ORT_RETURN_IF_NOT(num_of_momentums_1 == num_of_momentums_2, "Number of momentums_1 and momentums_2 mismatch.");

  prepare.grouped_tensor_sizes.resize(prepare.num_of_weights);
  prepare.grouped_tensor_pointers.resize(prepare.num_of_weights);

  static constexpr double cost = 1.0;
  auto* tp = ctx->GetOperatorThreadPool();

  concurrency::ThreadPool::TryParallelFor(
      tp, prepare.num_of_weights, cost, [&prepare](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t index = begin; index != end; ++index) {
          int i = static_cast<int>(index);
          const Tensor& weight_tensor = prepare.weights->Get(i);
          const Tensor& gradient_tensor = prepare.gradients->Get(i);
          const Tensor& momentum_1_tensor = prepare.momentums_1->Get(i);
          const Tensor& momentum_2_tensor = prepare.momentums_2->Get(i);

          // Check the weight/gradient/momentums at the same index should have same shape.
          ORT_ENFORCE(weight_tensor.Shape() == gradient_tensor.Shape(),
                      "Shape of weight and gradient mismatch, weight index:", i);
          ORT_ENFORCE(gradient_tensor.Shape() == momentum_1_tensor.Shape(),
                      "Shape of gradient and momentum_1 mismatch, weight index:", i);
          ORT_ENFORCE(momentum_1_tensor.Shape() == momentum_2_tensor.Shape(),
                      "Shape of momentum_1 and momentum_2 mismatch, weight index:", i);

          prepare.grouped_tensor_sizes[i] = static_cast<int>(weight_tensor.Shape().Size());

          prepare.grouped_tensor_pointers[i] = {
              const_cast<float*>(weight_tensor.Data<float>()),
              const_cast<float*>(gradient_tensor.Data<float>()),
              const_cast<float*>(momentum_1_tensor.Data<float>()),
              const_cast<float*>(momentum_2_tensor.Data<float>())};
        }
      });

  prepare.updated_flag = ctx->Output(0, prepare.step->Shape());
  prepare.updated_weights = ctx->Output<TensorSeq>(1);
  prepare.updated_momentums_1 = ctx->Output<TensorSeq>(2);
  prepare.updated_momentums_2 = ctx->Output<TensorSeq>(3);

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    AdamWOptimizer,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(2, 1) /* Return updated weights in-place */
        .Alias(4, 2) /* Return updated moment-1 in-place */
        .Alias(5, 3) /* Return updated moment-2 in-place */
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("S_WEIGHT", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("S_GRAD", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("S_MOMENT", DataTypeImpl::AllFixedSizeSequenceTensorTypes()),
    AdamWOptimizer<float>);

template <typename T>
Status AdamWOptimizer<T>::AdamWComputeMode0(Tensor& weight, Tensor& gradient, Tensor& momentums_1, Tensor& momentums_2,
                                            float lr, float alpha_correction, float beta_correction) const {
  // Perform weight decay.
  MakeEigenArrayMap<T>(weight) = MakeEigenArrayMap<T>(weight) - (MakeEigenArrayMap<T>(weight) * lr * weight_decay_);

  // Compute exponentially-averaged historical gradient.
  MakeEigenArrayMap<T>(momentums_1) = alpha_ * MakeEigenArrayMap<T>(momentums_1) +
                                      (1.f - alpha_) * MakeEigenArrayMap<T>(gradient);

  // Compute exponentially-averaged historical squared gradient.
  MakeEigenArrayMap<T>(momentums_2) = beta_ * MakeEigenArrayMap<T>(momentums_2) +
                                      (1.f - beta_) * MakeEigenArrayMap<T>(gradient) * MakeEigenArrayMap<T>(gradient);

  // Compute the new weight.
  auto denom = (MakeEigenArrayMap<T>(momentums_2) / beta_correction).sqrt() + epsilon_;
  MakeEigenArrayMap<T>(weight) = MakeEigenArrayMap<T>(weight) -
                                 (lr * MakeEigenArrayMap<T>(momentums_1)) / (alpha_correction * denom);

  return Status::OK();
}

template <typename T>
Status AdamWOptimizer<T>::AdamWComputeMode1(Tensor& weight, Tensor& gradient, Tensor& momentums_1, Tensor& momentums_2,
                                            float lr, float lr_corrected) const {
  // Compute exponentially-averaged historical gradient.
  MakeEigenArrayMap<T>(momentums_1) = alpha_ * MakeEigenArrayMap<T>(momentums_1) +
                                      (1.f - alpha_) * MakeEigenArrayMap<T>(gradient);

  // Compute exponentially-averaged historical squared gradient.
  MakeEigenArrayMap<T>(momentums_2) = beta_ * MakeEigenArrayMap<T>(momentums_2) +
                                      (1.f - beta_) * MakeEigenArrayMap<T>(gradient) * MakeEigenArrayMap<T>(gradient);

  auto denom = MakeEigenArrayMap<T>(momentums_2).sqrt() + epsilon_;
  MakeEigenArrayMap<T>(weight) = MakeEigenArrayMap<T>(weight) -
                                 (lr_corrected * MakeEigenArrayMap<T>(momentums_1) / denom);

  // Perform weight decay.
  MakeEigenArrayMap<T>(weight) = MakeEigenArrayMap<T>(weight) - (lr * weight_decay_ * MakeEigenArrayMap<T>(weight));

  return Status::OK();
}

template <typename T>
Status AdamWOptimizer<T>::Compute(OpKernelContext* ctx) const {
  AdamWOptimizerBase::Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(ctx, p));

  bool* updated_flag_ptr = p.updated_flag->template MutableData<bool>();

  const Tensor* update_signal = ctx->Input<Tensor>(6);
  if (update_signal == nullptr || *update_signal->template Data<bool>()) {
    const float lr = *p.learning_rate->template Data<float>();
    const int64_t step = *p.step->template Data<int64_t>();

    float alpha_correction = 1.f, beta_correction = 1.f;
    float lr_corrected = lr;
    if (correct_bias_ == 1) {
      // Notes:
      // > there is a minor difference compared with Apex's implementation,
      //   which uses double storing corrections before casting to float passing to kernels.
      // > std::pow(float, int) return double since C++11, so we cast back to float.
      alpha_correction = 1.f - static_cast<float>(std::pow(alpha_, step));
      beta_correction = 1.f - static_cast<float>(std::pow(beta_, step));
      lr_corrected *= std::sqrt(beta_correction) / alpha_correction;
    }

    // Currently two kinds of AdamW supported:
    // Mode 0: Pytorch https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW,
    //         bias correction is applied on m and v individually,
    //         weight decay is applied before weight is updated.
    // Mode 1: Huggingface https://github.com/huggingface/transformers/blob/d91841315aab55cf1347f4eb59332858525fad0f/
    //         src/transformers/optimization.py,
    //         bias correction is applied on learning rate, then use lr_corrected for subsequent computations.
    //         weight decay is applied after weight is updated.

    for (size_t weight_index = 0; weight_index < p.num_of_weights; ++weight_index) {
      Tensor& weight = const_cast<Tensor&>(p.weights->Get(weight_index));
      Tensor& gradient = const_cast<Tensor&>(p.gradients->Get(weight_index));
      Tensor& momentums_1 = const_cast<Tensor&>(p.momentums_1->Get(weight_index));
      Tensor& momentums_2 = const_cast<Tensor&>(p.momentums_2->Get(weight_index));

      if (adam_mode_ == 0) {
        ORT_RETURN_IF_ERROR(
            AdamWComputeMode0(weight, gradient, momentums_1, momentums_2, lr, alpha_correction, beta_correction));
      } else if (adam_mode_ == 1) {
        ORT_RETURN_IF_ERROR(
            AdamWComputeMode1(weight, gradient, momentums_1, momentums_2, lr, lr_corrected));
      } else {
        ORT_THROW("Unsupported Adamw optimizer mode.");
      }
    }

    *updated_flag_ptr = true;
  } else {
    *updated_flag_ptr = false;
  }

  if (p.updated_weights != nullptr) {
    ORT_RETURN_IF_ERROR(CopyIfNotSameCPUBuffer(ctx, p.num_of_weights, p.weights, p.updated_weights));
  }
  if (p.updated_momentums_1 != nullptr) {
    ORT_RETURN_IF_ERROR(CopyIfNotSameCPUBuffer(ctx, p.num_of_weights, p.momentums_1, p.updated_momentums_1));
  }
  if (p.updated_momentums_2 != nullptr) {
    ORT_RETURN_IF_ERROR(CopyIfNotSameCPUBuffer(ctx, p.num_of_weights, p.momentums_2, p.updated_momentums_2));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
