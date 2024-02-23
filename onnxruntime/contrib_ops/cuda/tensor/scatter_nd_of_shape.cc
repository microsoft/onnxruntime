// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/scatter_nd.h"
#include "core/providers/cuda/tensor/scatter_nd_impl.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "scatter_nd_of_shape.h"
#include <algorithm>

using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    ScatterNDOfShape,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    ScatterNDOfShape);

Status ScatterNDOfShape::ComputeInternal(OpKernelContext* context) const {
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto* indices_tensor = context->Input<Tensor>(1);
  const auto* updates_tensor = context->Input<Tensor>(2);

  const auto& input_shape_shape = input_tensor->Shape();
  ORT_ENFORCE(input_shape_shape.NumDimensions() == 1, "First input should have one dimension only.");
  ORT_ENFORCE(input_shape_shape.Size() > 0, "First input cannot be empty.");

  auto shape_type = input_tensor->DataType()->AsPrimitiveDataType()->GetDataType();
  ORT_ENFORCE(shape_type == 7, "Expect int64 as first input not", shape_type, ".");

  printf("ZZZ\n");
  printf("FFZ %d %d\n", (int)input_shape_shape.NumDimensions(), shape_type);
  const int64_t* shape_ptr = input_tensor->Data<int64_t>();
  ORT_ENFORCE(shape_ptr != nullptr, "First input is not a int64 pointer.");
  std::vector<int64_t> shape_v(input_shape_shape.NumDimensions());
  std::copy(shape_ptr, shape_ptr + input_shape_shape.Size(), shape_v.begin());
  TensorShape input_shape(shape_v);
  printf("BEGIN\n");

  const auto& indices_shape = indices_tensor->Shape();
  printf("Y\n");
  const auto& updates_shape = updates_tensor->Shape();
  printf("BEGIN2\n");

  // Validate input shapes
  ORT_RETURN_IF_ERROR(onnxruntime::ScatterND::ValidateShapes(input_shape, indices_shape, updates_shape));
  printf("BEGIN\n");

  auto* output_tensor = context->Output(0, input_shape);
  void* output_data = output_tensor->MutableDataRaw();
  printf("BEGIN\n");

  // Bail out early
  if (indices_shape.Size() == 0) {
    return Status::OK();
  }

  auto last_index_dimension = indices_shape[indices_shape.NumDimensions() - 1];

  // We need element counts for each dimension and the input dim value for each dimension
  // for the range [0, last_index_dimension).
  // To avoid multiple GPU data transfers, we combine this into one array and send it through
  TensorPitches input_strides(input_shape);
  std::vector<int64_t> element_counts_and_input_dims(last_index_dimension * 2, 0LL);
  for (int64_t i = 0; i < last_index_dimension; ++i) {
    element_counts_and_input_dims[i] = input_strides[i];
    element_counts_and_input_dims[i + last_index_dimension] = input_shape[i];
  }
  CudaAsyncBuffer<int64_t> element_counts_and_input_dims_gpu(this, element_counts_and_input_dims);
  ORT_RETURN_IF_ERROR(element_counts_and_input_dims_gpu.CopyToGpu(context->GetComputeStream()));

  switch (reduction_) {
    case Reduction::Add: {
      auto element_type = updates_tensor->DataType()->AsPrimitiveDataType()->GetDataType();
      printf("element_type %d\n", element_type);
      ORT_RETURN_IF_ERROR(onnxruntime::cuda::ScatterNDImplReduction(
          Stream(context),
          output_data,
          element_type,
          indices_shape.Size() / static_cast<size_t>(last_index_dimension),
          indices_tensor->Data<int64_t>(),  // only int64_t is supported for indices as per the onnx spec
          last_index_dimension,
          element_counts_and_input_dims_gpu.GpuPtr(),
          updates_tensor->DataRaw(),
          input_shape.SizeFromDimension(last_index_dimension),
          static_cast<int>(reduction_)));
    } break;
    default:
      ORT_THROW("ScatterNDOfShape not supported for other reduction than Add, None.");
      break;
    }
    printf("done\n");

    return Status::OK();
  }

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
