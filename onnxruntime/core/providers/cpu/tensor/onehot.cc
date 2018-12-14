/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/* Modifications Copyright (c) Microsoft. */

#include "core/providers/cpu/tensor/onehot.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using namespace ::onnxruntime::common;

namespace onnxruntime {
// spec: https://github.com/onnx/onnx/blob/master/docs/Operators.md#OneHot

#define REG_ONE_HOT_OP(in_type, out_type)                      \
  ONNX_CPU_OPERATOR_KERNEL(                                    \
      OneHotOp,                                                \
      9,                                                       \
      KernelDefBuilder()                                       \
          .TypeConstraint("T1", OpSchema::all_numeric_types()) \
          .TypeConstraint("T2", OpSchema::all_numeric_types()) \
          .TypeConstraint("T3", OpSchema::all_tensor_types()), \
      OneHot<in_type, out_type>);

REG_ONE_HOT_OP(int64_t, int64_t)

Status ValidateInputs(const Tensor* indices,
                      const Tensor* depth,
                      const Tensor* values) {
  // validation scenarios
  // indices must be non-negative integers
  // depth should be scalar and > 0
  if (!depth->Shape().IsScalar()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid argument for depth; it's not a scalar.");
  }

  // values Rank 1 tensor containing exactly two elements
  if (!(values->Shape().NumDimensions() == 1 && values->Shape().Size() == 2)) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  "Invalid argument for values; either it's rank is more than 1"
                  " or it has more than 2 elements");
  }

  return Status::OK();
}

// Helper to define Tensor types given that the scalar is of type T.
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
struct EigenTensorTypes {
  using EigenTensorMap = Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned>;
  using ConstEigenTensorMap = Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned>;
  using Scalar = Eigen::TensorMap<Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>, Eigen::Aligned>;
  using ConstScalar = Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>,Eigen::Aligned>;
};

namespace generator {
template <typename TI, typename TO>
class OneGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  OneGenerator(const typename EigenTensorTypes<TI>::ConstTensor& indices,
               const typename EigenTensorTypes<TO>::ConstScalar& on_value,
               const typename EigenTensorTypes<TO>::ConstScalar& off_value)
      : indices_(indices), on_value_(on_value), off_value_(off_value) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, 3>& pre_depth_suff) const {
    return (indices_(pre_depth_suff[0], pre_depth_suff[2]) == pre_depth_suff[1])
               ? on_value_()
               : off_value_();
  }

 private:
  const typename EigenTensorTypes<TI>::ConstTensor indices_;
  const typename EigenTensorTypes<TO>::ConstScalar on_value_;
  const typename EigenTensorTypes<TO>::ConstScalar off_value_;
};
}  // namespace generator

template <typename TI, typename TO>
Status OneHotOp<TI, TO>::Compute(OpKernelContext* p_op_kernel_context) const {
  const Tensor* indices = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* depth = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* values = p_op_kernel_context->Input<Tensor>(2);

  ONNXRUNTIME_RETURN_IF_ERROR(ValidateInputs(indices, depth, values));

  // prepare output shape
  const auto* depth_data = depth->Data<T>();
  const int64_t depth = static_cast<int64_t>(*depth_data);  // As per spec in case 'depth' is of non-integer type, it will be casted to int64 before use.
  if (depth <= 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Depth is negative.");
  }

  const auto& indices_shape = indices->Shape();
  std::vector<int64_t> output_shape(indices_shape.GetDims());
  output_shape.insert(axis < 0 ? output_shape.end() : output_shape.begin() + axis,
                      depth);

  // allocate output
  const auto* value_data = values->Data<TO>();
  Tensor* Y = context->Output(0, TensorShape(output_shape));

  const int axis = (axis_ == -1) ? indices_dims : axis_;
  int64 prefix_dim_size = 1;
  for (int i = 0; i < axis; ++i) {
    prefix_dim_size *= indices_shape.dim_size(i);
  }
  TI suffix_dim_size = indices_shape.num_elements() / prefix_dim_size;

  // Split indices into matrix of size prefix_dim_size x suffix_dim_size
  Eigen::array<Eigen::DenseIndex, 2> indices_dims_eigen = {{prefix_dim_size, suffix_dim_size}};
  const auto* indices_data = indices->Data<TI>();
  auto indices_tensor_eigen = EigenTensorTypes<T, 2>::EigenTensorMap(indices_data, indices_dims_eigen);

  // Split output into 3-Tensor of size:
  //   prefix_dim_size x depth x suffix_dim_size.
  Eigen::array<Eigen::DenseIndex, 2> output_dims_eigen = {{prefix_dim_size, depth_v, suffix_dim_size}};
  const auto* output_data = output->MutableData<TO>();
  auto output_tensor_eigen = EigenTensorTypes<T, 3>::EigenTensorMap(output_data, output_dims_eigen);

  auto on_value_eigen = EigenTensorTypes<TO>::Scalar(values->Data<T>()[1]);
  auto off_value_eigen = EigenTensorTypes<TO>::Scalar(values->Data<T>()[0]);

  generator::OneGenerator<TI, TO> generator(indices_tensor_eigen, on_value_eigen, off_value_eigen);
  Eigen::ThreadPoolDevice eigen_threadpool_device(4 /* number of threads to use */);  // TODO make this configurable
  output_tensor_eigen.device(eigen_threadpool_device) = output_tensor_eigen->generate(generator);

  return Status::OK();
}
}  // namespace onnxruntime
