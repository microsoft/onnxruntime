// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "expand.h"
#include "expand_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

namespace {
// Logically expanded y could just be a view of x.
static void CalcEffectiveDims(TensorShapeVector& x_dims, TensorShapeVector& y_dims) {
  TensorShapeVector x_reverse;
  TensorShapeVector y_reverse;

  int64_t xi = gsl::narrow_cast<int64_t>(x_dims.size()) - 1;
  for (int64_t yi = gsl::narrow_cast<int64_t>(y_dims.size()) - 1; yi >= 0; --yi, --xi) {
    int64_t xdim = (xi >= 0) ? x_dims[xi] : 1;
    int64_t ydim = y_dims[yi];
    if (xdim == ydim || xdim == 1) {
      x_reverse.push_back(xdim);
      y_reverse.push_back(ydim);
    } else {  // xdim < ydim && xdim > 1, split
      ydim /= xdim;
      x_reverse.push_back(xdim);
      y_reverse.push_back(xdim);
      x_reverse.push_back(1);
      y_reverse.push_back(ydim);
    }
  }

  x_dims.clear();
  y_dims.clear();
  x_dims.push_back(1);
  y_dims.push_back(1);
  // compact the dims, remove (x=1, y=1), merge (x=1, y1*y2...)
  for (int64_t i = gsl::narrow_cast<int64_t>(y_reverse.size()) - 1; i >= 0; --i) {
    if (x_reverse[i] == 1) {
      if (y_reverse[i] == 1) {
        continue;
      }
      if (x_dims.back() == 1) {
        y_dims.back() *= y_reverse[i];
      } else {
        x_dims.push_back(1);
        y_dims.push_back(y_reverse[i]);
      }
    } else {  // x_reverse[i] == y_reverse[i]
      if (x_dims.back() == y_dims.back()) {
        x_dims.back() *= x_reverse[i];
        y_dims.back() *= y_reverse[i];
      } else {
        x_dims.push_back(x_reverse[i]);
        y_dims.push_back(y_reverse[i]);
      }
    }
  }
}

#ifdef ENABLE_STRIDED_TENSORS
TensorShapeVector ComputeOutputStrides(const TensorShape& input_shapes, const gsl::span<const int64_t>& input_strides,
                                       const TensorShape& output_shapes) {
  const size_t rank = output_shapes.NumDimensions();
  const size_t input_rank = input_shapes.NumDimensions();

  if (input_rank == 0 || input_shapes.Size() == 1) {
    return TensorShapeVector(rank, 0);
  }

  TensorShapeVector output_strides(rank);
  const size_t offset = rank - input_rank;
  for (size_t dim = rank - 1;; --dim) {
    int64_t stride = 0;
    int64_t input_dim_size = dim >= offset ? input_shapes[dim - offset] : 1;
    if (input_dim_size == output_shapes[dim]) {
      stride = dim >= offset ? input_strides[dim - offset] : output_shapes[dim + 1] * output_strides[dim + 1];
    }

    output_strides[dim] = stride;
    if (dim == 0) break;
  }

  return output_strides;
}
#endif
}  // namespace

Status Expand::ComputeInternal(OpKernelContext* ctx) const {
  const auto& input_data_tensor = *ctx->Input<Tensor>(0);
  const auto& input_shape_tensor = *ctx->Input<Tensor>(1);

  // new shape to be expanded to
  const auto* p_shape = input_shape_tensor.Data<int64_t>();
  TensorShapeVector output_dims{p_shape, p_shape + input_shape_tensor.Shape().Size()};
  TensorShape output_shape(output_dims);

  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), input_data_tensor.Shape(), output_dims, output_shape));
  auto& output_tensor = *ctx->Output(0, output_shape);
  if (0 == output_shape.Size()) {
    return Status::OK();
  }

#ifdef ENABLE_STRIDED_TENSORS
  // Strided output.
  if (input_data_tensor.DataRaw() == output_tensor.DataRaw()) {
    gsl::span<const int64_t> input_strides = input_data_tensor.Strides();
    TensorShapeVector output_strides =
        ComputeOutputStrides(input_data_tensor.Shape(), input_strides, output_shape);
    output_tensor.SetShapeAndStrides(output_shape, output_strides);
    return Status::OK();
  }
#endif

  output_dims = output_shape.AsShapeVector();
  auto input_dims = input_data_tensor.Shape().AsShapeVector();

  CalcEffectiveDims(input_dims, output_dims);
  int rank = gsl::narrow_cast<int>(output_dims.size());

  TensorPitches original_input_strides(input_dims);
  TensorPitches original_output_strides(output_dims);

  TArray<int64_t> input_strides(rank);
  for (auto i = 0; i < rank; i++) {
    input_strides[i] = input_dims[i] == 1 ? 0 : original_input_strides[i];
  }

  TArray<fast_divmod> output_strides(rank);
  for (auto i = 0; i < rank; i++) {
    output_strides[i] = fast_divmod(static_cast<int>(original_output_strides[i]));
  }

  return ExpandImpl(
      Stream(ctx),
      input_data_tensor.DataType()->Size(),
      gsl::narrow_cast<int>(output_shape.Size()),
      gsl::narrow_cast<int>(input_data_tensor.Shape().Size()),
      input_data_tensor.DataRaw(),
      output_tensor.MutableDataRaw(),
      output_strides,
      input_strides);
}

Status FuncExpand(
    const CudaKernel* cuda_kernel,
    OpKernelContext* ctx,
    const Tensor* input_data_tensor,
    const Tensor* /*input_shape_tensor*/,
    Tensor* output_tensor) {
  TensorShape output_shape = output_tensor->Shape();

#ifdef ENABLE_STRIDED_TENSORS
  // Strided output.
  if (input_data_tensor->DataRaw() == output_tensor->DataRaw()) {
    gsl::span<const int64_t> input_strides = input_data_tensor->Strides();
    TensorShapeVector output_strides =
        ComputeOutputStrides(input_data_tensor->Shape(), input_strides, output_shape);
    output_tensor->SetShapeAndStrides(output_shape, output_strides);
    return Status::OK();
  }
#endif

  auto output_dims = output_shape.AsShapeVector();
  auto input_dims = input_data_tensor->Shape().AsShapeVector();

  CalcEffectiveDims(input_dims, output_dims);
  int rank = gsl::narrow_cast<int>(output_dims.size());

  TensorPitches original_input_strides(input_dims);
  TensorPitches original_output_strides(output_dims);

  TArray<int64_t> input_strides(rank);
  for (auto i = 0; i < rank; i++) {
    input_strides[i] = input_dims[i] == 1 ? 0 : original_input_strides[i];
  }

  TArray<fast_divmod> output_strides(rank);
  for (auto i = 0; i < rank; i++) {
    output_strides[i] = fast_divmod(static_cast<int>(original_output_strides[i]));
  }

  return ExpandImpl(
      cuda_kernel->Stream(ctx),
      input_data_tensor->DataType()->Size(),
      gsl::narrow_cast<int>(output_shape.Size()),
      gsl::narrow_cast<int>(input_data_tensor->Shape().Size()),
      input_data_tensor->DataRaw(),
      output_tensor->MutableDataRaw(),
      output_strides,
      input_strides);
}

std::unique_ptr<Tensor> FuncExpand(
    const CudaKernel* cuda_kernel,
    OpKernelContext* ctx,
    const Tensor* input_data_tensor,
    const Tensor* input_shape_tensor) {
  // new shape to be expanded to
  const auto* p_shape = input_shape_tensor->Data<int64_t>();
  TensorShapeVector output_dims{p_shape, p_shape + input_shape_tensor->Shape().Size()};
  TensorShape output_shape(output_dims);

  ORT_ENFORCE(
      ComputeOutputShape(
          cuda_kernel->Node().Name(),
          input_data_tensor->Shape(),
          output_dims, output_shape)
          .IsOK());

  // Pre-allocate output.
  AllocatorPtr alloc;
  ORT_ENFORCE(ctx->GetTempSpaceAllocator(&alloc).IsOK());
  auto output_tensor = Tensor::Create(input_data_tensor->DataType(), output_shape, alloc);

  // Only assign output values when output tensor is non-empty
  // because empty tensor doesn't own any data.
  if (output_shape.Size() > 0) {
    ORT_ENFORCE(FuncExpand(cuda_kernel, ctx, input_data_tensor, input_shape_tensor, output_tensor.get()).IsOK());
  }

  return output_tensor;
}

#ifdef ENABLE_STRIDED_TENSORS
#define CREATE_EXPAND_KERNEL_DEF (*KernelDefBuilder::Create()).MayStridedOutput(0, 0)
#else
#define CREATE_EXPAND_KERNEL_DEF (*KernelDefBuilder::Create())
#endif

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Expand, kOnnxDomain, 8, 12, kCudaExecutionProvider,
                                  CREATE_EXPAND_KERNEL_DEF.TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                                      .InputMemoryType(OrtMemTypeCPUInput, 1),
                                  Expand);

ONNX_OPERATOR_KERNEL_EX(Expand, kOnnxDomain, 13, kCudaExecutionProvider,
                        CREATE_EXPAND_KERNEL_DEF.TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                            .InputMemoryType(OrtMemTypeCPUInput, 1),
                        Expand);

#undef CREATE_EXPAND_KERNEL_DEF

}  // namespace cuda
};  // namespace onnxruntime
