// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// disable warning because std::copy is used by Sliceiterator
// std::copy_n is not an option for raw pointer destinations as used by gsl::copy.
#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif
#include "core/providers/cpu/nn/unpool.h"
#include "core/common/narrow.h"
#include "core/providers/cpu/tensor/utils.h"
#include <cmath>

using namespace ::onnxruntime::common;

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    MaxUnpool,
    9, 10,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()),
    MaxUnpool);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    MaxUnpool,
    11, 21,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()),
    MaxUnpool);

ONNX_CPU_OPERATOR_KERNEL(
    MaxUnpool,
    22,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()),
    MaxUnpool);

Status MaxUnpool::Compute(OpKernelContext* context) const {
  // Get pooled values tensor
  const auto* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const TensorShape& X_shape = X->Shape();
  const auto* X_data = X->Data<float>();

  // Spec: "Dimensions ... are in the form of (N x C x D1 x D2 ... Dn)" — minimum rank is 3.
  ORT_RETURN_IF_NOT(X_shape.NumDimensions() >= 3, "Input dimension cannot be less than 3.");

  // Implementation limitation: only 1D/2D/3D spatial pooling supported.
  size_t pooling_dims = X_shape.NumDimensions() - 2;
  if (pooling_dims > 3) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size.");
  }

  // Spec: "The size of the kernel along each axis" — must match number of spatial dims.
  ORT_RETURN_IF_NOT(kernel_shape_.size() == pooling_dims,
                    "kernel_shape rank mismatch: expected ", pooling_dims, " got ", kernel_shape_.size());

  // Get pooled index tensor
  const auto* I = context->Input<Tensor>(1);
  const TensorShape& I_shape = I->Shape();
  const auto* I_data = I->Data<int64_t>();

  // Spec: Input I "Dimensions must be the same as input tensor X."
  ORT_RETURN_IF_NOT(I_shape == X_shape, "Index tensor shape should be same as that of the input data tensor to unpool.");

  // Calculate output tensor shape from attributes
  TensorShapeVector inferred_output_dims(X_shape.NumDimensions());

  // Copy batch and channel dims
  inferred_output_dims[0] = X_shape[0];
  inferred_output_dims[1] = X_shape[1];

  // For feature dims calculate reversing the formula used for MaxPool
  for (size_t dim = 0; dim < kernel_shape_.size(); ++dim) {
    int64_t dim_value = (X_shape[dim + 2] - 1) * strides_[dim] -
                        (pads_[dim] + pads_[kernel_shape_.size() + dim]) + kernel_shape_[dim];
    // Each inferred spatial dim must be positive for a valid unpooling configuration.
    ORT_RETURN_IF_NOT(dim_value > 0,
                      "Computed output dimension is not positive for axis ", dim + 2,
                      ". Check kernel_shape, strides, and pads attributes.");
    inferred_output_dims[dim + 2] = dim_value;
  }

  TensorShape shape(inferred_output_dims);

  if (num_inputs_ == 3) {
    auto tensor_shape = context->Input<Tensor>(2);
    if (tensor_shape == nullptr)
      return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    // Spec: output_shape is a 1-D tensor of int64.
    ORT_RETURN_IF_NOT(tensor_shape->Shape().GetDims().size() == 1,
                      "Shape must be 1 dimensional as it's tensor data of a shape");

    // Spec: output_shape specifies the full output shape (N x C x D1 x ... x Dn) — same rank as X.
    ORT_RETURN_IF_NOT(
        static_cast<size_t>(tensor_shape->Shape().Size()) == X_shape.NumDimensions(),
        "output_shape must have the same number of elements as the rank of input tensor X."
        " Got ",
        tensor_shape->Shape().Size(), ", expected ", X_shape.NumDimensions());

    // Turn the shape tensor data into an actual shape
    auto output_shape_span = tensor_shape->DataAsSpan<int64_t>();
    TensorShape given_shape(output_shape_span);

    // Spec: output shape is (N x C x D1 x ... x Dn) — batch and channel must match input.
    ORT_RETURN_IF_NOT(given_shape[0] == X_shape[0] && given_shape[1] == X_shape[1],
                      "output_shape batch and channel dimensions must match input. "
                      "Expected [",
                      X_shape[0], ", ", X_shape[1], "], got [",
                      given_shape[0], ", ", given_shape[1], "].");

    // Spec: output_shape disambiguates size — must be at least as large as the inferred minimum.
    ORT_RETURN_IF_NOT(given_shape.Size() >= shape.Size(),
                      "output_shape is smaller than minimum required. output_shape:", given_shape,
                      " inferred output shape:", shape);

    shape = std::move(given_shape);
  }

  // unpool
  size_t total_elements = narrow<size_t>(X_shape.Size());

  Tensor* Y = context->Output(0, shape);
  auto out = Y->MutableDataAsSpan<float>();
  std::fill_n(out.data(), out.size(), 0.f);

  for (size_t cur_elem = 0; cur_elem < total_elements; ++cur_elem) {
    const int64_t idx = I_data[cur_elem];
    // Spec: "the values in indices are in the range [0, N x C x D1 x ... x Dn)."
    if (idx < 0 || idx >= static_cast<int64_t>(out.size())) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Index value out of bounds. Got: ", idx, ". Valid range is [0, ", out.size(), ").");
    }

    out[static_cast<size_t>(idx)] = X_data[cur_elem];
  }

  return Status::OK();
}

}  // namespace onnxruntime
