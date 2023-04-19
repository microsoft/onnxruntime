// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/qordered_ops/qordered_unary_ops.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_unary_ops_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    QOrderedGelu,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)   // scale_input
        .InputMemoryType(OrtMemTypeCPUInput, 2),  // scale_output
    QOrderedGelu);

Status QOrderedGelu::ComputeInternal(OpKernelContext* context) const {
  const Tensor& input = *context->Input<Tensor>(0);

  const float* scale_input = context->Input<Tensor>(1)->Data<float>();
  const float* scale_output = context->Input<Tensor>(2)->Data<float>();

  const auto& shape = input.Shape();

  Tensor* output = context->Output(0, shape);

  return QOrderedUnarySharedMemory_Gelu(Stream(context), input.Data<int8_t>(), scale_input, output->MutableData<int8_t>(),
                                        scale_output, static_cast<size_t>(shape.Size()));
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
