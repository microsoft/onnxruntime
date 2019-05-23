// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif
#include "densetodensesetoperation.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_TYPED_KERNEL_EX(DenseToDenseSetOperation,
                              kMSDomain,
                              1,
                              int64_t,
                              kCpuExecutionProvider,
                              KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
                              DenseToDenseSetOperation<int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(DenseToDenseSetOperation,
                              kMSDomain,
                              1,
                              int32_t,
                              kCpuExecutionProvider,
                              KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
                              DenseToDenseSetOperation<int32_t>);

template <typename T>
Status DenseToDenseSetOperation<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* x = ctx->Input<Tensor>(0);
  const Tensor* y = ctx->Input<Tensor>(1);

  auto x_rank = static_cast<int64_t>(x->Shape().NumDimensions());
  auto y_rank = static_cast<int64_t>(y->Shape().NumDimensions());
  // make sure we have at least 1 dimension
  if (x_rank < 1)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensor x shape should be at least rank 1.");
  if (x_rank != y_rank)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensors x and y should have same rank.");

  // check if all but last dimensions are same
  for (int64_t i = 0; i < x_rank - 1; i++) {
    if (x->Shape()[i] != y->Shape()[i])
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensors x and y must have same shape, except for last dimension.");
  }

  // obtain raw input data
  const T* x_data = x->Data<T>();
  const T* y_data = y->Data<T>();
  auto x_size = x->Shape().Size();
  //auto y_size = y->Shape().Size();

  auto x_lastdim = x->Shape()[x_rank - 1];
  auto y_lastdim = y->Shape()[y_rank - 1];
  int64_t z_lastdim = 0;

  T dv = static_cast<T>(default_value_);
  std::vector<std::set<T>> z_sets;  // container for subset intersection values
  int64_t y_pos = 0;                // index in y_data array
  int64_t z_index = 0;              // subset index

  for (int64_t x_pos = 0; x_pos < x_size; x_pos += x_lastdim) {
    std::unordered_map<T, bool> x_set;
    std::vector<T> v(z_lastdim);
    z_sets.resize(z_index + 1);

    // store all elements of x subset in map
    for (int64_t k = 0; k < x_lastdim; k++)
      x_set[x_data[x_pos + k]] = true;

    // compute intersection
    for (int64_t k = 0; k < y_lastdim; k++) {
      auto y_val = y_data[y_pos + k];
      if (y_val == dv)  // skip default values
        continue;
      if (x_set.find(y_val) != x_set.end())
        z_sets[z_index].insert(y_val);
    }

    z_lastdim = std::max(z_lastdim, static_cast<int64_t>(z_sets[z_index].size()));
    y_pos += y_lastdim;
    z_index++;
  }

  // compute output shape
  TensorShape z_shape(x->Shape());
  z_shape[x_rank - 1] = z_lastdim;
  int64_t z_size = z_shape.Size();

  // create output tensor
  Tensor* z_tensor = ctx->Output(0, z_shape);
  T* z_data = z_tensor->template MutableData<T>();

  // fill tensor values
  if (z_size == 0)
    return Status::OK();

  for (int64_t i = 0, z_pos = 0; i < z_size / z_lastdim; i += 1) {
    // copy intersections
    for (auto it = z_sets[i].begin(); it != z_sets[i].end(); ++it)
      z_data[z_pos++] = *it;
    // pad with default value
    for (auto j = 0; j < z_lastdim - z_sets[i].size(); j++)
      z_data[z_pos++] = dv;
  }

  return Status::OK();
}

}  // namespace contrib
};  // namespace onnxruntime
