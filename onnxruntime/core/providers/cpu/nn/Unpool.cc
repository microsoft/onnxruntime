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

  ORT_RETURN_IF_NOT(X_shape.NumDimensions() >= 3, "Input dimension cannot be less than 3.");

  // Supported sizes check
  size_t pooling_dims = X_shape.NumDimensions() - 2;
  if (pooling_dims > 3) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size.");
  }

  // Get pooled index tensor
  const auto* I = context->Input<Tensor>(1);
  const TensorShape& I_shape = I->Shape();
  const auto* I_data = I->Data<int64_t>();

  ORT_RETURN_IF_NOT(I_shape == X_shape, "Index tensor shape should be same as that of the input data tensor to unpool.");

  // Calculate output tensor shape from attributes
  std::vector<int64_t> inferred_output_dims(X_shape.NumDimensions());

  // Copy batch and channel dims
  inferred_output_dims[0] = X_shape[0];
  inferred_output_dims[1] = X_shape[1];

  // For feature dims calculate reversing the formula used for MaxPool
  for (size_t dim = 0; dim < kernel_shape_.size(); ++dim) {
    inferred_output_dims[dim + 2] =
        (X_shape[dim + 2] - 1) * strides_[dim] - (pads_[dim] + pads_[kernel_shape_.size() + dim]) + kernel_shape_[dim];
  }

  TensorShape shape(inferred_output_dims);

  if (num_inputs_ == 3) {
    auto tensor_shape = context->Input<Tensor>(2);
    if (tensor_shape == nullptr)
      return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    ORT_RETURN_IF_NOT(tensor_shape->Shape().GetDims().size() == 1,
                      "Shape must be 1 dimensional as it's tensor data of a shape");

    // Turn the shape tensor data into an actual shape
    const auto* p_shape = tensor_shape->Data<int64_t>();
    std::vector<int64_t> given_output_dims(p_shape, p_shape + tensor_shape->Shape().Size());
    TensorShape given_shape(given_output_dims);

    ORT_RETURN_IF_NOT(given_shape.Size() >= shape.Size(),
                      "output_shape is smaller than minimum required. output_shape:", given_shape,
                      " inferred output shape:", shape);

    shape = std::move(given_shape);
  }

  // unpool
  int64_t total_elements = X_shape.Size();

  Tensor* Y = context->Output(0, shape);
  auto* Y_data = Y->MutableData<float>();
  auto out = gsl::make_span(Y_data, narrow<size_t>(Y->Shape().Size()));
  std::fill_n(out.data(), out.size(), 0.f);

  for (auto cur_elem = 0; cur_elem < total_elements; ++cur_elem) {
    out[narrow<size_t>(I_data[narrow<size_t>(cur_elem)])] = X_data[narrow<size_t>(cur_elem)];
  }

  return Status::OK();
}

}  // namespace onnxruntime
