// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "expand.h"
#include "expand_impl.h"
#include "core/providers/cpu/tensor/utils.h"

using std::vector;

namespace onnxruntime {
namespace cuda {

// Logically expanded y could just be a view of x.
static void CalcEffectiveDims(vector<int64_t>& x_dims, vector<int64_t>& y_dims) {
  vector<int64_t> x_reverse;
  vector<int64_t> y_reverse;

  int xi = gsl::narrow_cast<int>(x_dims.size()) - 1;
  for (int yi = gsl::narrow_cast<int>(y_dims.size()) - 1; yi >= 0; --yi, --xi) {
    int64_t xdim = (xi >= 0) ? x_dims[xi] : 1;
    int64_t ydim = y_dims[yi];
    if (xdim == ydim || xdim == 1) {
      x_reverse.push_back(xdim);
      y_reverse.push_back(ydim);
    }
    else { // xdim < ydim && xdim > 1, split
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
  for (int i = gsl::narrow_cast<int>(y_reverse.size()) - 1; i >= 0; --i) {
    if (x_reverse[i] == 1) {
      if (y_reverse[i] == 1) {
        continue;
      }
      if (x_dims.back() == 1) {
        y_dims.back() *= y_reverse[i];
      }
      else {
        x_dims.push_back(1);
        y_dims.push_back(y_reverse[i]);
      }
    }
    else { // x_reverse[i] == y_reverse[i]
      if (x_dims.back() == y_dims.back()) {
        x_dims.back() *= x_reverse[i];
        y_dims.back() *= y_reverse[i];
      }
      else {
        x_dims.push_back(x_reverse[i]);
        y_dims.push_back(y_reverse[i]);
      }
    }
  }
}

Status Expand::ComputeInternal(OpKernelContext* ctx) const {
<<<<<<< HEAD
  const auto& input0 = *ctx->Input<Tensor>(0);
  const auto& input1 = *ctx->Input<Tensor>(1);
=======
  const auto& input_data_tensor = *ctx->Input<Tensor>(0);
  const auto& input_shape_tensor = *ctx->Input<Tensor>(1);
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677

  // new shape to be expanded to
  const auto* p_shape = input_shape_tensor.template Data<int64_t>();
  std::vector<int64_t> output_dims{p_shape, p_shape + input_shape_tensor.Shape().Size()};
  TensorShape output_shape(output_dims);

<<<<<<< HEAD
  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), input0.Shape(), output_dims, output_shape));
  int32_t rank = gsl::narrow_cast<int32_t>(output_shape.NumDimensions());
=======
  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), input_data_tensor.Shape(), output_dims, output_shape));
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677
  auto& output_tensor = *ctx->Output(0, output_shape);
  if (0 == output_shape.Size()) {
    return Status::OK();
  }

  output_dims = output_shape.GetDims();
  auto input_dims = input_data_tensor.Shape().GetDims();

<<<<<<< HEAD
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
=======
  CalcEffectiveDims(input_dims, output_dims);
  int rank = gsl::narrow_cast<int>(output_dims.size());

  CudaAsyncBuffer<fast_divmod> fdm_output_strides(this, rank);
  ORT_ENFORCE(CalculateFdmStrides(fdm_output_strides.CpuSpan(), output_dims));

  CudaAsyncBuffer<int64_t> input_view_strides(this, rank);
  TensorPitches::Calculate(input_view_strides.CpuSpan(), input_dims);
  for (int i = 0; i < rank; ++i) {
    if (input_dims[i] == 1) input_view_strides.CpuSpan()[i] = 0;
  }
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677

  return ExpandImpl(
      input_data_tensor.DataType()->Size(),
      gsl::narrow_cast<int>(output_shape.Size()),
      gsl::narrow_cast<int>(input_data_tensor.Shape().Size()),
      input_data_tensor.DataRaw(),
      output_tensor.MutableDataRaw(),
      fdm_output_strides,
      input_view_strides);
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
