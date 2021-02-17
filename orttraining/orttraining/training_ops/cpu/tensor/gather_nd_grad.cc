// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/tensor/gather_nd_grad.h"
#include "core/providers/cpu/tensor/gather_nd.h"
#include "core/common/common.h"
namespace onnxruntime {

#ifndef DISABLE_CONTRIB_OPS

namespace contrib {
ONNX_OPERATOR_KERNEL_EX(GatherNDGrad, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                  DataTypeImpl::GetTensorType<double>()})
                            .TypeConstraint("Tind", {DataTypeImpl::GetTensorType<int64_t>(),
                                                     DataTypeImpl::GetTensorType<int32_t>()}),
                        GatherNDGrad);

}  // namespace contrib

#endif

template <typename InputT>
struct GatherNDGradComputeImpl {
  void operator()(GatherNDBase::Prepare& p, const Tensor* update_tensor) const {
    const int64_t grad_size = update_tensor->Shape().Size();
    const int64_t slice_size = p.element_count_per_slice;
    const InputT* input_base_casted = reinterpret_cast<const InputT*>(p.input_base);
    InputT* output_base_casted = reinterpret_cast<InputT*>(p.output_base);

    for (int64_t i = 0; i < grad_size; i++) {
      uint64_t slice_offset = p.slice_offsets[i / slice_size];
      size_t j = i % slice_size;
      output_base_casted[slice_offset + j] += input_base_casted[i];
    }
  }
};

Status GatherNDGrad::Compute(OpKernelContext* context) const {
  const auto* shape_tensor = context->Input<Tensor>(0);
  const auto* indices_tensor = context->Input<Tensor>(1);
  const auto* update_tensor = context->Input<Tensor>(2);

  ORT_ENFORCE(shape_tensor != nullptr && indices_tensor != nullptr && update_tensor != nullptr,
              "GatherNDGrad::Compute : Input count mismatch");

  auto shape_data = shape_tensor->Data<int64_t>();
  auto input_shape = TensorShape(shape_data, shape_tensor->SizeInBytes() / sizeof(shape_tensor->DataType()));

  const auto& indices_shape = indices_tensor->Shape();

  int64_t last_indices_dimension = batch_dims_ + indices_shape[indices_shape.NumDimensions() - 1];
  if (last_indices_dimension > static_cast<int64_t>(input_shape.NumDimensions())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "last dimension of indices must not be larger than rank of input tensor");
  }

  auto* output_tensor = context->Output(0, input_shape);
  memset(output_tensor->MutableDataRaw(), 0, output_tensor->SizeInBytes());

  GatherNDBase::Prepare p;
  p.input_base = static_cast<const uint8_t*>(update_tensor->DataRaw());
  p.output_base = static_cast<uint8_t*>(output_tensor->MutableDataRaw());

  auto bytes_per_value = update_tensor->DataType()->Size();
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  if (indices_tensor->IsDataType<int32_t>()) {
    PrepareForCompute<int32_t>(input_shape, indices_tensor, bytes_per_value, p, tp);
  } else if (indices_tensor->IsDataType<int64_t>()) {
    PrepareForCompute<int64_t>(input_shape, indices_tensor, bytes_per_value, p, tp);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "indices tensor data type not supported");
  }

  ORT_RETURN_IF_NOT(nullptr == p.input_str_base, "nullptr != p.input_str_base");
  utils::MLTypeCallDispatcher<float, double> t_disp(update_tensor->GetElementType());
  t_disp.Invoke<GatherNDGradComputeImpl>(p, update_tensor);

  return Status::OK();
}

}  // namespace onnxruntime
