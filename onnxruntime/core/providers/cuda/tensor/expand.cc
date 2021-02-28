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
  const auto& input_data_tensor = *ctx->Input<Tensor>(0);
  const auto& input_shape_tensor = *ctx->Input<Tensor>(1);

  // new shape to be expanded to
  const auto* p_shape = input_shape_tensor.template Data<int64_t>();
  std::vector<int64_t> output_dims{p_shape, p_shape + input_shape_tensor.Shape().Size()};
  TensorShape output_shape(output_dims);

  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), input_data_tensor.Shape(), output_dims, output_shape));
  auto& output_tensor = *ctx->Output(0, output_shape);
  if (0 == output_shape.Size()) {
    return Status::OK();
  }

  output_dims = output_shape.GetDims();
  auto input_dims = input_data_tensor.Shape().GetDims();

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
      Stream(),
      input_data_tensor.DataType()->Size(),
      gsl::narrow_cast<int>(output_shape.Size()),
      gsl::narrow_cast<int>(input_data_tensor.Shape().Size()),
      input_data_tensor.DataRaw(),
      output_tensor.MutableDataRaw(),
      output_strides,
      input_strides);
}


ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Expand,
    kOnnxDomain,
    8, 12,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .InputMemoryType<OrtMemTypeCPUInput>(1),
    Expand);

ONNX_OPERATOR_KERNEL_EX(
    Expand,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .InputMemoryType<OrtMemTypeCPUInput>(1),
    Expand);

}  // namespace cuda
};  // namespace onnxruntime
