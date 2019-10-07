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

  // new shape to be expanded to
  const auto* p_shape = input1.template Data<int64_t>();
  std::vector<int64_t> output_dims{p_shape, p_shape + input1.Shape().Size()};
  TensorShape output_shape(output_dims);

  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), input0.Shape(), output_dims, output_shape));
  int32_t rank = gsl::narrow_cast<int32_t>(output_shape.NumDimensions());
  auto& output_tensor = *ctx->Output(0, output_shape);

  if (0 == output_shape.Size()) {
    return Status::OK();
  }

  auto input_shape = input0.Shape().GetDims();

  // pad input_dims with 1 to make ranks match
  for (auto i = 0; i < rank - input_shape.size(); i++) {
    input_shape.insert(input_shape.begin(), 1);
  }

  // create fast_divmod using dimension values
  TArray<fast_divmod> fdm_input_dims(gsl::narrow_cast<int32_t>(input_shape.size()));
  for (auto i = 0; i < input_shape.size(); ++i) {
    fdm_input_dims.data_[i] = fast_divmod(gsl::narrow_cast<int32_t>(input_shape[i]));
  }

  ORT_ENFORCE(rank <= MAX_ARRAY_SIZE);
  TArray<fast_divmod> fdm_output_dims(rank);
  for (auto i = 0; i < rank; ++i) {
    fdm_output_dims.data_[i] = fast_divmod(gsl::narrow_cast<int32_t>(output_shape.GetDims()[i]));
  }

  TArray<fast_divmod> fdm_output_subdim_size(rank);
  auto subdim_size = output_shape.Size();
  for (auto i = 0; i < rank; i++) {
    // output_shape[i] won't be 0 here, it's covered in (0 == output_shape.Size())
    // a null output will be returned for that case
    subdim_size /= output_shape[i];
    fdm_output_subdim_size.data_[i] = fast_divmod(static_cast<int>(subdim_size));
  }

  ExpandImpl(
      input0.DataType()->Size(),
      rank,
      output_shape.Size(),
      input0.Shape().Size(),
      input0.DataRaw(),
      output_tensor.MutableDataRaw(),
      &fdm_input_dims,
      &fdm_output_dims,
      &fdm_output_subdim_size);

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
