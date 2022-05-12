// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/optimizer/adam/adam.h"
#include "orttraining/training_ops/cuda/optimizer/adam/adam_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Adam,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 4)
        .InputMemoryType(OrtMemTypeCPUInput, 5)
        .Alias(0, 0) /* Return updated weights in-place */
        .Alias(2, 1) /* Return updated moment-1 in-place */
        .Alias(3, 2) /* Return updated moment-2 in-place */
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("S_WEIGHT", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("S_GRAD", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("S_MOMENT", DataTypeImpl::AllFixedSizeSequenceTensorTypes()),
    Adam);

template <typename T>
bool IsSameBuffer(const TensorSeq& source_tensor, TensorSeq& target_tensor, size_t num_of_weights) {
  if (source_tensor.Size() != target_tensor.Size()) {
    return false;
  }

  for (size_t input_idx = 0; input_idx < num_of_weights; ++input_idx) {
    const Tensor& t = target_tensor.Get(input_idx);
    const Tensor& s = source_tensor.Get(input_idx);
    const T* source = s.template Data<T>();
    const T* target = t.template Data<T>();
    if (source != target) {
      return false;
    }
  }

  return true;
}

AllocatorPtr CreateAllocatorPtr(OpKernelContext* ctx) {
  AllocatorPtr alloc;
  ORT_ENFORCE(ctx->GetTempSpaceAllocator(&alloc).IsOK(),
              "Adam GPU: Unable to get an allocator.");
  return alloc;
}

AllocatorPtr& GetAllocatorPtr(OpKernelContext* ctx) {
  static AllocatorPtr alloc = CreateAllocatorPtr(ctx);
  return alloc;
}

Status GenerateOutputs(OpKernelContext* ctx, cudaStream_t steam,
                       const TensorSeq* values, TensorSeq* updated_values,
                       size_t number_of_values) {
  if (!IsSameBuffer<float>(*values, *updated_values, number_of_values)) {
    updated_values->SetType(values->Get(0).DataType());
    updated_values->Reserve(number_of_values);
    for (size_t input_idx = 0; input_idx < number_of_values; ++input_idx) {
      const Tensor& source_tensor = values->Get(input_idx);
      std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor.DataType(),
                                                             source_tensor.Shape(), GetAllocatorPtr(ctx));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target_tensor->MutableDataRaw(),
                                           source_tensor.DataRaw(),
                                           source_tensor.SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, steam));
      updated_values->Add(std::move(*target_tensor));  // Add will check for type consistency
    }
  }

  return Status::OK();
}

Status Adam::ComputeInternal(OpKernelContext* ctx) const {
  const TensorSeq* weights = ctx->Input<TensorSeq>(0);
  const TensorSeq* gradients = ctx->Input<TensorSeq>(1);
  const TensorSeq* momentums_1 = ctx->Input<TensorSeq>(2);
  const TensorSeq* momentums_2 = ctx->Input<TensorSeq>(3);
  const Tensor& learning_rate = *ctx->Input<Tensor>(4);
  const Tensor* step_count = ctx->Input<Tensor>(5);

  size_t num_of_weights = weights->Size();
  size_t num_of_gradients = gradients->Size();
  size_t num_of_momentums_1 = momentums_1->Size();
  size_t num_of_momentums_2 = momentums_2->Size();

  // Check the number of weights, gradients, momentums matchs.
  ORT_ENFORCE(num_of_weights > 0, "invalid count of tensors in Seq<Tensor>.");
  ORT_ENFORCE(num_of_weights == num_of_gradients, "number of weights and gradients mismatch.");
  ORT_ENFORCE(num_of_gradients == num_of_momentums_1, "number of gradients and momentums_1 mismatch.");
  ORT_ENFORCE(num_of_momentums_1 == num_of_momentums_2, "number of momentus_1 and momentums_2 mismatch.");

  std::vector<int> tensor_sizes(num_of_weights);
  std::vector<std::vector<void*>> grouped_tensor_pointers(num_of_weights);

  for (size_t i = 0; i < num_of_weights; ++i) {
    const Tensor& weight_tensor = weights->Get(i);
    const Tensor& gradient_tensor = gradients->Get(i);
    const Tensor& momentum_1_tensor = momentums_1->Get(i);
    const Tensor& momentum_2_tensor = momentums_2->Get(i);

    // Check the weight/gradient/momentums at the same index should have same shape.
    ORT_ENFORCE(weight_tensor.Shape() == gradient_tensor.Shape(), "shape of weight and gradient mismatch.");
    ORT_ENFORCE(gradient_tensor.Shape() == momentum_1_tensor.Shape(), "shape of gradient and momentum_1 mismatch.");
    ORT_ENFORCE(momentum_1_tensor.Shape() == momentum_2_tensor.Shape(), "shape of momentum_1 and momentum_2 mismatch.");

    // Currently we only support float data types.
    ORT_ENFORCE(weight_tensor.IsDataType<float>() &&
                    gradient_tensor.IsDataType<float>() &&
                    momentum_1_tensor.IsDataType<float>() &&
                    momentum_2_tensor.IsDataType<float>(),
                "Only float data type support for Adam cuda kernel.");

    tensor_sizes[i] = static_cast<int>(weight_tensor.Shape().Size());

    grouped_tensor_pointers[i] = {
        const_cast<float*>(weight_tensor.Data<float>()),
        const_cast<float*>(gradient_tensor.Data<float>()),
        const_cast<float*>(momentum_1_tensor.Data<float>()),
        const_cast<float*>(momentum_2_tensor.Data<float>())};
  }

  TensorSeq* updated_weights = ctx->Output<TensorSeq>(0);
  TensorSeq* updated_momentums_1 = ctx->Output<TensorSeq>(1);
  TensorSeq* updated_momentums_2 = ctx->Output<TensorSeq>(2);
  Tensor* is_updated = ctx->Output(3, step_count->Shape());

  ORT_ENFORCE(is_updated);
  ORT_ENFORCE(updated_weights);
  ORT_ENFORCE(updated_momentums_1);
  ORT_ENFORCE(updated_momentums_2);

  typedef typename ToCudaType<float>::MappedType CudaT_FLOAT;
  typedef AdamMTAFunctor<CudaT_FLOAT, CudaT_FLOAT, CudaT_FLOAT> TFunctor;
  TFunctor functor;

  const float* lr_ptr = learning_rate.template Data<float>();
  const int64_t* step_ptr = step_count->template Data<int64_t>();
  const float lr = *lr_ptr;
  const int64_t step = *step_ptr;

  ORT_UNUSED_PARAMETER(lr);
  ORT_UNUSED_PARAMETER(step);

  launch_multi_tensor_functor<MTA_ADAM_GROUP_SIZE, TFunctor>(
      Stream(), 2048 * 32, tensor_sizes, grouped_tensor_pointers, functor,
      alpha_, beta_, epsilon_, lr, weight_decay_, adam_mode_, correct_bias_, step);

  ORT_RETURN_IF_ERROR(GenerateOutputs(ctx, Stream(), weights, updated_weights, num_of_weights));
  ORT_RETURN_IF_ERROR(GenerateOutputs(ctx, Stream(), momentums_1, updated_momentums_1, num_of_weights));
  ORT_RETURN_IF_ERROR(GenerateOutputs(ctx, Stream(), momentums_2, updated_momentums_2, num_of_weights));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
