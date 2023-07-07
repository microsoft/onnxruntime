// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/split_view.h"

#include "orttraining/training_ops/cpu/tensor/split_view.h"

using namespace onnxruntime::contrib;

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(SplitView, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .InputMemoryType(OrtMemTypeCPUInput, 1)
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                            .Alias(SplitViewAliasMapping()),  // all output tensors are sharing the same bffer as
                                                              // input[0], execept that the byte_offset are different.
                        SplitView);

Status SplitView::ComputeInternal(OpKernelContext* context) const {
  const Tensor& input_tensor = *context->Input<Tensor>(0);
  const Tensor* p_split_tensor = context->Input<Tensor>(1);
  size_t num_outputs =
      num_outputs_ == -1 ? static_cast<size_t>(context->OutputCount()) : static_cast<size_t>(num_outputs_);
  InlinedVector<TensorShape> output_shapes;
  InlinedVector<size_t> output_offsets;
  ORT_RETURN_IF_ERROR(PrepareForSplitView(input_tensor, num_outputs, p_split_tensor, output_shapes, output_offsets));
  ORT_ENFORCE(static_cast<size_t>(context->OutputCount()) <= output_shapes.size(), "Output count mismatch.");

  const void* input_data = input_tensor.DataRaw();
  for (size_t i = 0; i < static_cast<size_t>(context->OutputCount()); ++i) {
    // Outputs are allowed to be unused.
    Tensor* output_tensor = context->Output(i, output_shapes[i]);
    if (output_tensor != nullptr) {
      void* output_data = output_tensor->MutableDataRaw();
      if (input_data != output_data) {
        // View output is not sharing the underlaying buffer of input, copy instead.
        const void* source = static_cast<const char*>(input_data) + output_offsets[i];
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_data, source, output_tensor->SizeInBytes(),
                                             cudaMemcpyDeviceToDevice, Stream(context)));
      } else {
        output_tensor->SetByteOffset(output_offsets[i]);
      }
    }
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
