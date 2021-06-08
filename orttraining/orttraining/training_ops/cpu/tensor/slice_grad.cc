// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/tensor/slice_grad.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    SliceGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    SliceGrad);

Status SliceGrad::Compute(OpKernelContext* context) const {
  const Tensor& grad = *context->Input<Tensor>(0);
  const Tensor& shape = *context->Input<Tensor>(1);
  const TensorShape data_shape(shape.template Data<int64_t>(), shape.Shape().Size());
  Tensor& output = *context->Output(0, data_shape);
  memset(output.MutableDataRaw(), 0, output.SizeInBytes());
  // Initialize the starts & ends to the actual tensor shape
  std::vector<int64_t> input_starts;
  std::vector<int64_t> input_ends;
  std::vector<int64_t> input_axes;
  std::vector<int64_t> input_steps;
  FillVectorsFromInput(*context->Input<Tensor>(2), *context->Input<Tensor>(3), context->Input<Tensor>(4),
                       context->Input<Tensor>(5), input_starts, input_ends, input_axes, input_steps);

  SliceOp::PrepareForComputeMetadata compute_metadata(data_shape.GetDims());
  ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes, input_steps, compute_metadata));

  MLDataType T_type = grad.DataType();
  if (T_type == DataTypeImpl::GetType<float>()) {
    return ComputeImpl<float>(context, output, compute_metadata.output_dims_, compute_metadata.p_flattened_output_dims_,
                              compute_metadata.starts_, compute_metadata.steps_);
  }

  if (T_type == DataTypeImpl::GetType<double>()) {
    return ComputeImpl<double>(context, output, compute_metadata.output_dims_, compute_metadata.p_flattened_output_dims_,
                               compute_metadata.starts_, compute_metadata.steps_);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for T or Tind not supported yet in SliceGrad.");
}

template <typename T>
Status SliceGrad::ComputeImpl(OpKernelContext* ctx,
                              Tensor& output_grad_tensor,
                              const std::vector<int64_t>& output_dims,
                              std::vector<int64_t>* flattened_output_dims,
                              const std::vector<int64_t>& starts,
                              const std::vector<int64_t>& steps) const {
  TensorShape output_shape(output_dims);
  // output tensor's size is 0, nothing to fill - return
  if (output_shape.Size() == 0)
    return Status::OK();

  auto& grad_tensor = *const_cast<Tensor*>(ctx->Input<Tensor>(0));
  auto* grad_data = grad_tensor.template MutableData<T>();
  const auto* grad_data_end = grad_data + grad_tensor.Shape().Size();

  auto create_output = [&grad_data, &grad_data_end](WritableSliceIterator<T>& output_iterator) {
    if (output_iterator.SolitaryInnerStep()) {
      while (grad_data < grad_data_end) {
        grad_data = output_iterator.CopyFromInnermostAxisSolitaryInnerStep(grad_data);
      }
    } else {
      while (grad_data < grad_data_end) {
        grad_data = output_iterator.CopyFromInnermostAxisNonSolitaryInnerStep(grad_data);
      }
    }

    ORT_ENFORCE(grad_data == grad_data_end);
  };

  if (flattened_output_dims) {
    // if we have flattened output dims we need to also flatten the input dims.
    // as we're combining the innermost dims and keeping all values we can just copy the size of the last dim
    std::vector<int64_t> flattened_input_dims(output_grad_tensor.Shape().GetDims());
    flattened_input_dims.resize(flattened_output_dims->size());
    flattened_input_dims.back() = flattened_output_dims->back();
    TensorShape input_shape(std::move(flattened_input_dims));

    WritableSliceIterator<T> input_iterator(output_grad_tensor, input_shape, starts, *flattened_output_dims, steps);
    create_output(input_iterator);
  } else {
    WritableSliceIterator<T> input_iterator(output_grad_tensor, starts, output_dims, steps);
    create_output(input_iterator);
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
