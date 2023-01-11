// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/optimizer/sgd/sgd.h"
#include "orttraining/training_ops/cpu/optimizer/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {
namespace contrib {

Status SGDOptimizerV2Base::PrepareForCompute(OpKernelContext* ctx, SGDOptimizerV2Base::Prepare& prepare) const {
  prepare.learning_rate = ctx->Input<Tensor>(0);
  prepare.weights = ctx->Input<TensorSeq>(1);
  prepare.gradients = ctx->Input<TensorSeq>(2);

  prepare.num_of_weights = prepare.weights->Size();
  size_t num_of_gradients = prepare.gradients->Size();

  // Check the number of weights, gradients, momentums match.
  ORT_RETURN_IF_NOT(prepare.num_of_weights > 0, "Invalid count of tensors in Seq<Tensor>.");
  ORT_RETURN_IF_NOT(prepare.num_of_weights == num_of_gradients, "Number of weights and gradients mismatch.");

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

          // Check the weight/gradient/momentums at the same index should have same shape.
          ORT_ENFORCE(weight_tensor.Shape() == gradient_tensor.Shape(),
                      "Shape of weight and gradient mismatch, weight index:", i);

          prepare.grouped_tensor_sizes[i] = static_cast<int>(weight_tensor.Shape().Size());

          prepare.grouped_tensor_pointers[i] = {
              const_cast<float*>(weight_tensor.Data<float>()),
              const_cast<float*>(gradient_tensor.Data<float>())};
        }
      });

  prepare.update_completed = ctx->Output(0, prepare.learning_rate->Shape());
  prepare.updated_weights = ctx->Output<TensorSeq>(1);

  return Status::OK();
}

template <typename T>
Status SGDOptimizerV2<T>::Compute(OpKernelContext* ctx) const {
  SGDOptimizerV2Base::Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(ctx, p));

  bool* updated_flag_ptr = p.update_completed->template MutableData<bool>();
  const Tensor* update_signal = ctx->Input<Tensor>(3);
  if (update_signal == nullptr || *update_signal->template Data<bool>()) {
    const float lr = *p.learning_rate->template Data<float>();

    for (size_t weight_index = 0; weight_index < p.num_of_weights; ++weight_index) {
      Tensor& weight = const_cast<Tensor&>(p.weights->Get(weight_index));
      Tensor& gradient = const_cast<Tensor&>(p.gradients->Get(weight_index));

      // new_weight = weight - lr * gradient
      const auto& delta = -lr * MakeEigenArrayMap<T>(gradient);
      MakeEigenArrayMap<T>(weight) = MakeEigenArrayMap<T>(weight) + delta;
    }

    *updated_flag_ptr = true;
  } else {
    *updated_flag_ptr = false;
  }

  if (p.updated_weights != nullptr) {
    ORT_RETURN_IF_ERROR(CopyIfNotSameCPUBuffer(ctx, p.num_of_weights, p.weights, p.updated_weights));
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    SGDOptimizerV2,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .Alias(1, 1)  // Update weights in-place
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T_BOOL", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("S_WEIGHT", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("S_GRAD", DataTypeImpl::AllFixedSizeSequenceTensorTypes()),
    SGDOptimizerV2<float>);

}  // namespace contrib
}  // namespace onnxruntime
