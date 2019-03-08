// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// disable warning because std::copy is used by Sliceiterator
// std::copy_n is not an option for raw pointer destinations as used by gsl::copy.
#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif
#include "core/providers/cpu/nn/unpool.h"
#include "core/providers/cpu/tensor/utils.h"
#include <cmath>

using namespace ::onnxruntime::common;

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    MaxUnpool,
    9,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("Y", DataTypeImpl::GetTensorType<float>()),
    MaxUnpool);

Status MaxUnpool::Compute(OpKernelContext* context) const {
  // Get pooled values tensor
  const Tensor* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const TensorShape& X_shape = X->Shape();
  const float* X_data = X->template Data<float>();

  ORT_RETURN_IF_NOT(X_shape.NumDimensions() >= 3, "Input dimension cannot be less than 3.");

  // Supported sizes check
  size_t pooling_dims = X_shape.NumDimensions() - 2;
  if (pooling_dims > 3) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size.");
  }

  // Get pooled index tensor
  const Tensor* I = context->Input<Tensor>(1);
  const TensorShape& I_shape = I->Shape();
  const int64_t* I_data = I->template Data<int64_t>();

  ORT_RETURN_IF_NOT(I_shape == X_shape, "Index tensor shape should be same as that of the input data tensor to unpool.");

  // Calculate output tensor shape from attributes
  std::vector<int64_t> inferredOutputShape(X_shape.NumDimensions());

  // Copy batch and channel dims
  inferredOutputShape[0] = X_shape[0];
  inferredOutputShape[1] = X_shape[1];

  // For feature dims calculate reversing the formula used for Maxpool
  for (auto dim = 0; dim < kernel_shape_.size(); ++dim) {
    inferredOutputShape[dim + 2] = (X_shape[dim + 2] - 1) * strides_[dim] - (pads_[dim + 2] + pads_[kernel_shape_.size() + dim + 4]) + kernel_shape_[dim];
  }

  // If outputshape is provided use that to infer additional padding.
  std::vector<int64_t> inferredPads;
  std::vector<int64_t> givenOutputShape;
  bool padsInferred = false;

  if (num_inputs_ == 3) {
    auto tensor_shape = context->Input<Tensor>(2);
    if (tensor_shape == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    ORT_RETURN_IF_NOT(tensor_shape->Shape().GetDims().size() == 1, "Shape must be 1 dimensional as it's tensor data is a shape");

    // Turn the shape tensor data into an actual shape
    const int64_t* p_shape = tensor_shape->template Data<int64_t>();
    std::vector<int64_t> shape{p_shape, p_shape + tensor_shape->Shape().Size()};
    givenOutputShape = shape;

    inferredPads.resize(inferredOutputShape.size() * 2, 0);

    // calculate if output shape has any padding over the inferred shape for feature dims.
    for (auto dim = 2; dim < shape.size(); dim++) {
      ORT_RETURN_IF_NOT(inferredOutputShape[dim] <= shape[dim], "Incorrect output shape");

      int64_t inferredPad = shape[dim] - inferredOutputShape[dim];
      ORT_RETURN_IF_NOT(inferredPad <= kernel_shape_[dim - 2], "Incorrect output shape");

      if (inferredPad > 0) {
        padsInferred = true;
        if (inferredPad == kernel_shape_[dim - 2]) {
          inferredPads[dim] = 1;
          inferredPads[dim + inferredOutputShape.size()] = inferredPad - 1;
        } else {
          inferredPads[dim + inferredOutputShape.size()] = inferredPad;
        }
      }
    }
  }

  // unpool
  int64_t totalPooledElem = 1;
  int64_t totalOutputElem = 1;

  for (auto dim = 0; dim < X_shape.NumDimensions(); dim++) {
    totalPooledElem *= X_shape[dim];
    totalOutputElem *= inferredOutputShape[dim];
  }

  // if there are no pads inferred from outputshape simply create the new unpooled tensor
  if (!padsInferred) {
    TensorShape shape(inferredOutputShape);

    Tensor* Y = context->Output(0, shape);
    auto Y_data = Y->template MutableData<float>();
    auto out = gsl::make_span(Y_data, Y->Shape().Size());
    std::fill_n(out.data(), out.size(), 0.f);

    for (auto curElem = 0; curElem < totalPooledElem; ++curElem) {
      out[I_data[curElem]] = X_data[curElem];
    }
  } else {
    // If the output shape has pads over the inferred dims , first
    // create the tensor with the inferred dims and add the padding.

    // Generate tensor with inferred dims.
    TensorShape shape(inferredOutputShape);

    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
    auto element_type = DataTypeImpl::GetType<float>();

    std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                                shape,
                                                                alloc);

    float* p = p_tensor->template MutableData<float>();

    auto out = gsl::make_span(p, p_tensor->Shape().Size());
    std::fill_n(out.data(), out.size(), 0.f);

    for (auto curElem = 0; curElem < totalPooledElem; ++curElem) {
      out[I_data[curElem]] = X_data[curElem];
    }

    std::vector<int64_t> output_dims(inferredOutputShape);
    size_t dimension_count = output_dims.size();

    std::vector<int64_t> input_starts;
    std::vector<int64_t> input_extents;

    // Calculate output dimensions
    for (size_t i = 0; i < dimension_count; i++) {
      input_starts.push_back(slices_[i]);
      input_extents.push_back(output_dims[i] + slices_[i] + slices_[i + dimension_count]);
      output_dims[i] += inferredPads[i] + inferredPads[i + dimension_count] + slices_[i] + slices_[i + dimension_count];
    }

    // setup output object
    TensorShape output_shape(givenOutputShape);
    Tensor* Y = context->Output(0, output_shape);
    auto Y_data = Y->template MutableData<float>();

    auto outData = gsl::make_span(Y_data, Y->Shape().Size());

    std::fill_n(outData.data(), outData.size(), 0.f);

    // add padding
    TensorPitches output_pitches(*Y);
    size_t alignSkip = 0;  // Amount to skip to align to where the next input tensor data needs to be written

    // Initial skip, sum up the begin padding on each axis
    for (size_t i = 0; i < dimension_count; i++)
      alignSkip += inferredPads[i] * output_pitches[i];

    size_t inner_axis = dimension_count - 1;

    TensorAxisCounters input_counters(*p_tensor);
    SliceIterator<float> input(*p_tensor, input_starts, input_extents);

    while (input_counters) {
      Y_data += alignSkip;
      {
        Y_data = input.CopyInnermostAxis(Y_data);
        int64_t prePad = inferredPads[inner_axis];
        int64_t postPad = inferredPads[inner_axis + dimension_count];
        Y_data += postPad;
        alignSkip = prePad;
      }
      // Calculate the size of the next block of padding (skipping over the innermost axis since that's already done)
      while (input_counters.Increment()) {
        ptrdiff_t inner_pitch = output_pitches[input_counters.Axis()];
        int64_t prePad = inferredPads[input_counters.Axis()];
        int64_t postPad = inferredPads[input_counters.Axis() + dimension_count];
        Y_data += inner_pitch * postPad;
        alignSkip += inner_pitch * prePad;
      }
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
