// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/space_depth_ops.h"
#include "core/util/eigen_common_wrapper.h"
#include <array>

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    SpaceToDepth,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SpaceToDepth<float>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    DepthToSpace,
    1,
    4,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    DepthToSpace<float>);

// intemediate tensor shapes are:
// (batch, blocksize, blocksize, input_depth / (blocksize * blocksize), input_height, input_width) for DepthToSpace
// (batch, input_depth, input_height / blocksize, blocksize, input_width / blocksize, blocksize) for SpaceToDepth
const int IntermediateTensorRank = 6;
typedef Eigen::TensorMap<Eigen::Tensor<float, IntermediateTensorRank, Eigen::RowMajor, int64_t>,
                         Eigen::Aligned>
    EigenTensorMap;

template <>
Status SpaceToDepth<float>::Compute(OpKernelContext* context) const {
  const Tensor* tensor_pointer = context->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& input = *tensor_pointer;
  ORT_ENFORCE(input.Shape().NumDimensions() == 4);
  const int64_t batch = input.Shape().GetDims().at(0);
  const int64_t input_depth = input.Shape().GetDims().at(1);
  const int64_t input_height = input.Shape().GetDims().at(2);
  const int64_t input_width = input.Shape().GetDims().at(3);
  ORT_ENFORCE(input_height % this->blocksize_ == 0);
  ORT_ENFORCE(input_width % this->blocksize_ == 0);

  const int64_t output_depth = input_depth * blocksize_ * blocksize_;
  const int64_t output_height = input_height / blocksize_;
  const int64_t output_width = input_width / blocksize_;
  Tensor& output = *context->Output(0, {batch, output_depth, output_height, output_width});

  std::array<int64_t, IntermediateTensorRank> permutation{{0, 3, 5, 1, 2, 4}};
  EigenTensorMap(output.template MutableData<float>(), batch, blocksize_, blocksize_,
                 input_depth, input_height / blocksize_, input_width / blocksize_) =
      EigenTensorMap(const_cast<float*>(input.template Data<float>()), batch,
                     input_depth, input_height / blocksize_, blocksize_,
                     input_width / blocksize_, blocksize_)
          .shuffle(permutation);

  return Status::OK();
}

template <>
Status DepthToSpace<float>::Compute(OpKernelContext* context) const {
  const Tensor* tensor_pointer = context->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& input = *tensor_pointer;
  ORT_ENFORCE(input.Shape().NumDimensions() == 4);

  const int64_t batch = input.Shape().GetDims().at(0);
  const int64_t input_depth = input.Shape().GetDims().at(1);
  const int64_t input_height = input.Shape().GetDims().at(2);
  const int64_t input_width = input.Shape().GetDims().at(3);
  ORT_ENFORCE(input_depth % (blocksize_ * blocksize_) == 0);

  const int64_t output_depth = input_depth / blocksize_ / blocksize_;
  const int64_t output_height = input_height * blocksize_;
  const int64_t output_width = input_width * blocksize_;

  Tensor& output = *context->Output(0, {batch, output_depth, output_height, output_width});

  std::array<int64_t, IntermediateTensorRank> permutation{{0, 3, 4, 1, 5, 2}};
  EigenTensorMap(output.template MutableData<float>(), batch, input_depth / blocksize_ / blocksize_,
                 input_height, blocksize_, input_width, blocksize_) =
      EigenTensorMap(const_cast<float*>(input.template Data<float>()), batch,
                     blocksize_, blocksize_, input_depth / blocksize_ / blocksize_,
                     input_height, input_width)
          .shuffle(permutation);

  return Status::OK();
}

}  // namespace onnxruntime
