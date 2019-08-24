// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "expand.h"
#include "expand_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

Status Expand::ComputeInternal(OpKernelContext* ctx) const {
  const auto& input0 = *ctx->Input<Tensor>(0);
  const auto& input1 = *ctx->Input<Tensor>(1);
  int device_id = GetDeviceId();

  // new shape to be expanded to
  const auto* p_shape = input1.template Data<int64_t>();
  std::vector<int64_t> output_dims{p_shape, p_shape + input1.Shape().Size()};
  TensorShape output_shape(output_dims);

  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), input0.Shape(), output_dims, output_shape));
  auto rank = output_shape.NumDimensions();
  auto& output_tensor = *ctx->Output(0, output_shape);

  if (0 == output_shape.Size()) {
    return Status::OK();
  }

  auto input_shape = input0.Shape().GetDims();

  // pad input_dims with 1 to make ranks match
  for (int i = 0; i < rank - input_shape.size(); i++) {
    input_shape.insert(input_shape.begin(), 1);
  }

  // create fast_divmod using dimension values
  CudaAsyncBuffer<fast_divmod> fdm_input_dims(this, device_id, rank);
  CudaAsyncBuffer<fast_divmod> fdm_output_dims(this, device_id, rank);
  CudaAsyncBuffer<fast_divmod> fdm_output_subdim_size(this, device_id, rank);
  {
    auto in_span = fdm_input_dims.CpuSpan();
    auto out_span = fdm_output_dims.CpuSpan();
    auto sdm_span = fdm_output_subdim_size.CpuSpan();
    auto subdim_size = output_shape.Size();
    for (auto i = 0; i < rank; i++) {
      in_span[i] = fast_divmod(static_cast<int>(input_shape[i]));
      out_span[i] = fast_divmod(static_cast<int>(output_shape[i]));
      // output_shape[i] won't be 0 here, it's covered in (0 == output_shape.Size())
      // a null output will be returned for that case
      subdim_size /= output_shape[i];
      sdm_span[i] = static_cast<int>(subdim_size);
    }
  }
  ORT_RETURN_IF_ERROR(fdm_input_dims.CopyToGpu());
  ORT_RETURN_IF_ERROR(fdm_output_dims.CopyToGpu());
  ORT_RETURN_IF_ERROR(fdm_output_subdim_size.CopyToGpu());

  ExpandImpl(
      input0.DataType()->Size(),
      output_shape.NumDimensions(),
      output_shape.Size(),
      input0.Shape().Size(),
      input0.DataRaw(),
      output_tensor.MutableDataRaw(),
      fdm_input_dims.GpuPtr(),
      fdm_output_dims.GpuPtr(),
      fdm_output_subdim_size.GpuPtr());

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    Expand,
    kOnnxDomain,
    8,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .InputMemoryType<OrtMemTypeCPUInput>(1),
    Expand);

}  // namespace cuda
};  // namespace onnxruntime
