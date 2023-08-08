// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/optimizer/common.h"
#include "orttraining/training_ops/cuda/optimizer/sgd/sgd.h"
#include "orttraining/training_ops/cuda/optimizer/sgd/sgd_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    SGDOptimizerV2,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .Alias(1, 1)  // Update weights in-place
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T_BOOL", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("S_WEIGHT", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("S_GRAD", DataTypeImpl::AllFixedSizeSequenceTensorTypes()),
    SGDOptimizerV2);

Status SGDOptimizerV2::ComputeInternal(OpKernelContext* ctx) const {
  SGDOptimizerV2Base::Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(ctx, p));

  bool* updated_flag_ptr = p.update_completed->template MutableData<bool>();

  // Currently placed on CPU, need revisit when we had mixed precision training requirement.
  const Tensor* update_signal = ctx->Input<Tensor>(3);
  if (update_signal == nullptr || *update_signal->template Data<bool>()) {
    typedef typename ToCudaType<float>::MappedType CudaT_FLOAT;
    typedef SGDMTAFunctor<CudaT_FLOAT, CudaT_FLOAT> TFunctor;
    TFunctor functor;

    const float* lr_ptr = p.learning_rate->template Data<float>();
    ORT_ENFORCE(lr_ptr);

    launch_multi_tensor_functor<MTA_SGD_GROUP_SIZE, TFunctor>(
        Stream(ctx), MTA_SGD_CHUNK_SIZE, p.grouped_tensor_sizes, p.grouped_tensor_pointers, functor,
        *lr_ptr);
    *updated_flag_ptr = true;
  } else {
    *updated_flag_ptr = false;
  }

  if (p.updated_weights != nullptr) {
    ORT_RETURN_IF_ERROR(CopyIfNotSameCUDABuffer(ctx, p.num_of_weights, p.weights, p.updated_weights));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
