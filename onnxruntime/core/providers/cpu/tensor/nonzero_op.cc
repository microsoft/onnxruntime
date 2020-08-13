// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/nonzero_op.h"

#include <cassert>
#include <vector>

#include "core/util/math_cpuonly.h"

namespace onnxruntime {
// kernel builder functions
#define NONZERO_TYPED_KERNEL_WITH_TYPE_NAME(type, type_name)                       \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                  \
      NonZero,                                                                     \
      9,                                                                           \
      type_name,                                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), \
      NonZero<type>)

#define NONZERO_TYPED_KERNEL(type) \
  NONZERO_TYPED_KERNEL_WITH_TYPE_NAME(type, type)

// start with a subset of types, enable more as needed...
NONZERO_TYPED_KERNEL(bool)
NONZERO_TYPED_KERNEL(uint8_t)
//NONZERO_TYPED_KERNEL(uint16_t)
//NONZERO_TYPED_KERNEL(uint32_t)
//NONZERO_TYPED_KERNEL(uint64_t)
//NONZERO_TYPED_KERNEL(int8_t)
//NONZERO_TYPED_KERNEL(int16_t)
NONZERO_TYPED_KERNEL(int32_t)
NONZERO_TYPED_KERNEL(int64_t)
//NONZERO_TYPED_KERNEL(MLFloat16)
//NONZERO_TYPED_KERNEL(BFloat16)
NONZERO_TYPED_KERNEL(float)
//NONZERO_TYPED_KERNEL(double)
//NONZERO_TYPED_KERNEL_WITH_TYPE_NAME(std::string, string)

#undef NONZERO_TYPED_KERNEL_WITH_TYPE_NAME
#undef NONZERO_TYPED_KERNEL

template <typename T>
Status NonZero<T>::Compute(OpKernelContext* context) const {
  const auto X = context->Input<Tensor>(0);
  ORT_ENFORCE(X, "X input is required!");

  const auto& X_shape = X->Shape();
  assert(X_shape.Size() >= 0);

  const Eigen::Index coordinate_size = X_shape.IsScalar() ? 1 : X_shape.NumDimensions();
  std::vector<int64_t> non_zero_indices_buffer{};
  // reserve enough space for indices for every element of X
  non_zero_indices_buffer.reserve(X_shape.Size() * coordinate_size);

  const T* data = X->Data<T>();

  if (X_shape.IsScalar()) {
    const T& value = *data;
    if (value != T{}) {
      non_zero_indices_buffer.push_back(0);
    }
  } else {
    std::vector<int64_t> coordinate(coordinate_size, 0);

    // as we iterate the entries, increment the coordinate for the current entry
    // e.g. if shape is {2,2}, we start with 0,0 increment to 0,1 increment to 1,0 and finally 1,1
    auto increment_coordinate = [&coordinate, &coordinate_size, &X_shape]() {
      for (int64_t idx = coordinate_size - 1; idx >= 0; --idx) {
        int64_t& cur_coord = coordinate[idx];
        if (cur_coord != X_shape[idx] - 1) {
          ++cur_coord;
          break;
        }
        cur_coord = 0;
      }
    };

    for (size_t i = 0, end = X_shape.Size(); i < end; ++i) {
      const T& value = *data++;
      if (value != T{}) {
        non_zero_indices_buffer.insert(non_zero_indices_buffer.end(),
                                       coordinate.begin(), coordinate.end());
      }

      increment_coordinate();
    }
  }

  const Eigen::Index num_non_zero_values = non_zero_indices_buffer.size() / coordinate_size;

  // transpose result for output
  ConstEigenMatrixMapRowMajor<int64_t> non_zero_indices_matrix{
      non_zero_indices_buffer.data(),
      num_non_zero_values, coordinate_size};

  Tensor* const Y = context->Output(0, {coordinate_size, num_non_zero_values});
  ORT_ENFORCE(Y, "failed to get first output!");

  EigenMatrixMapRowMajor<int64_t> y_matrix{
      Y->MutableData<int64_t>(),
      coordinate_size, num_non_zero_values};
  y_matrix = non_zero_indices_matrix.transpose();

  return Status::OK();
}

}  // namespace onnxruntime
