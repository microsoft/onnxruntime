// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <utility>

#include "orttraining/training_ops/cuda/optimizer/adamw/adamw.h"
#include "orttraining/training_ops/cuda/optimizer/adamw/adamw_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    AdamWOptimizer,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 6)
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .Alias(2, 1) /* Return updated weights in-place */
        .Alias(4, 2) /* Return updated moment-1 in-place */
        .Alias(5, 3) /* Return updated moment-2 in-place */
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("S_WEIGHT", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("S_GRAD", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("S_MOMENT", DataTypeImpl::AllFixedSizeSequenceTensorTypes()),
    AdamWOptimizer);

AllocatorPtr CreateAllocatorPtr(OpKernelContext* ctx) {
  AllocatorPtr alloc;
  ORT_ENFORCE(ctx->GetTempSpaceAllocator(&alloc).IsOK(),
              "AdamWOptimizer GPU: Unable to get an allocator.");
  return alloc;
}

AllocatorPtr& GetAllocatorPtr(OpKernelContext* ctx) {
  static AllocatorPtr alloc = CreateAllocatorPtr(ctx);
  return alloc;
}

Status GenerateOutputs(OpKernelContext* ctx, cudaStream_t stream,
                       const TensorSeq* values, TensorSeq* updated_values,
                       size_t number_of_values) {
  // Return if the output edge is not fetched.
  if (updated_values == nullptr) {
    return Status::OK();
  }

  bool is_same_buffer = const_cast<TensorSeq*>(values) == updated_values;
  if (!is_same_buffer) {
    updated_values->SetType(values->DataType());
    updated_values->Reserve(number_of_values);
    for (size_t input_idx = 0; input_idx < number_of_values; ++input_idx) {
      const Tensor& source_tensor = values->Get(input_idx);
      std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor.DataType(),
                                                             source_tensor.Shape(), GetAllocatorPtr(ctx));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target_tensor->MutableDataRaw(),
                                           source_tensor.DataRaw(),
                                           source_tensor.SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, stream));
      updated_values->Add(std::move(*target_tensor));  // Add will check for type consistency
    }
  }

  return Status::OK();
}

Status AdamWOptimizer::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* learning_rate = ctx->Input<Tensor>(0);
  const Tensor* step = ctx->Input<Tensor>(1);
  const TensorSeq* weights = ctx->Input<TensorSeq>(2);
  const TensorSeq* gradients = ctx->Input<TensorSeq>(3);
  const TensorSeq* momentums_1 = ctx->Input<TensorSeq>(4);
  const TensorSeq* momentums_2 = ctx->Input<TensorSeq>(5);

  size_t num_of_weights = weights->Size();
  size_t num_of_gradients = gradients->Size();
  size_t num_of_momentums_1 = momentums_1->Size();
  size_t num_of_momentums_2 = momentums_2->Size();

  // Check the number of weights, gradients, momentums matchs.
  ORT_ENFORCE(num_of_weights > 0, "Invalid count of tensors in Seq<Tensor>.");
  ORT_ENFORCE(num_of_weights == num_of_gradients, "Number of weights and gradients mismatch.");
  ORT_ENFORCE(num_of_gradients == num_of_momentums_1, "Number of gradients and momentums_1 mismatch.");
  ORT_ENFORCE(num_of_momentums_1 == num_of_momentums_2, "Number of momentums_1 and momentums_2 mismatch.");

  std::vector<int> tensor_sizes(num_of_weights);
  std::vector<std::vector<void*>> grouped_tensor_pointers(num_of_weights);

  for (size_t i = 0; i < num_of_weights; ++i) {
    const Tensor& weight_tensor = weights->Get(i);
    const Tensor& gradient_tensor = gradients->Get(i);
    const Tensor& momentum_1_tensor = momentums_1->Get(i);
    const Tensor& momentum_2_tensor = momentums_2->Get(i);

    // Check the weight/gradient/momentums at the same index should have same shape.
    ORT_ENFORCE(weight_tensor.Shape() == gradient_tensor.Shape(),
                "Shape of weight and gradient mismatch, weight index:", i);
    ORT_ENFORCE(gradient_tensor.Shape() == momentum_1_tensor.Shape(),
                "Shape of gradient and momentum_1 mismatch, weight index:", i);
    ORT_ENFORCE(momentum_1_tensor.Shape() == momentum_2_tensor.Shape(),
                "Shape of momentum_1 and momentum_2 mismatch, weight index:", i);

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

  Tensor* updated_flag = ctx->Output(0, step->Shape());
  TensorSeq* updated_weights = ctx->Output<TensorSeq>(1);
  TensorSeq* updated_momentums_1 = ctx->Output<TensorSeq>(2);
  TensorSeq* updated_momentums_2 = ctx->Output<TensorSeq>(3);

  int64_t* updated_flag_ptr = updated_flag->template MutableData<int64_t>();

  // Currently placed on CPU, need revisit when we had mixed precision training requirement.
  const Tensor* update_signal = ctx->Input<Tensor>(6);
  if (update_signal == nullptr || *update_signal->template Data<bool>()) {
    typedef typename ToCudaType<float>::MappedType CudaT_FLOAT;
    typedef AdamWMTAFunctor<CudaT_FLOAT, CudaT_FLOAT, CudaT_FLOAT> TFunctor;
    TFunctor functor;

    const float* lr_ptr = learning_rate->template Data<float>();
    const int64_t* step_ptr = step->template Data<int64_t>();
    ORT_ENFORCE(lr_ptr && step_ptr);

    launch_multi_tensor_functor<MTA_ADAMW_GROUP_SIZE, TFunctor>(
        Stream(), MTA_ADAMW_CHUNK_SIZE, tensor_sizes, grouped_tensor_pointers, functor,
        alpha_, beta_, epsilon_, *lr_ptr, weight_decay_, adam_mode_, correct_bias_, *step_ptr);
    *updated_flag_ptr = 1;
  } else {
    *updated_flag_ptr = 0;
  }

  ORT_RETURN_IF_ERROR(GenerateOutputs(ctx, Stream(), weights, updated_weights, num_of_weights));
  ORT_RETURN_IF_ERROR(GenerateOutputs(ctx, Stream(), momentums_1, updated_momentums_1, num_of_weights));
  ORT_RETURN_IF_ERROR(GenerateOutputs(ctx, Stream(), momentums_2, updated_momentums_2, num_of_weights));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
