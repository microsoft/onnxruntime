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
    1, 10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    DepthToSpace<float>);

ONNX_CPU_OPERATOR_KERNEL(
    DepthToSpace,
    11,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    DepthToSpace<float>);

// intermediate tensor shapes are:
// (batch, blocksize, blocksize, input_depth / (blocksize * blocksize), input_height, input_width) for DepthToSpace
// (batch, input_depth, input_height / blocksize, blocksize, input_width / blocksize, blocksize) for SpaceToDepth
const int IntermediateTensorRank = 6;
typedef Eigen::TensorMap<Eigen::Tensor<float, IntermediateTensorRank, Eigen::RowMajor, int64_t>,
                         Eigen::Aligned>
    EigenTensorMap;

typedef Eigen::TensorMap<Eigen::Tensor<const float, IntermediateTensorRank, Eigen::RowMajor, int64_t>,
                         Eigen::Aligned>
    ConstEigenTensorMap;

// helper to create output to minimize binary size
static void CreateOutput(const Tensor& input, Tensor& output,
                         const std::array<int64_t, IntermediateTensorRank>& permutation,
                         const int64_t batch_size,  // dim0 in both input and output
                         const int64_t in_dim1, const int64_t in_dim2, const int64_t in_dim3,
                         const int64_t in_dim4, const int64_t in_dim5,
                         const int64_t out_dim1, const int64_t out_dim2, const int64_t out_dim3,
                         const int64_t out_dim4, const int64_t out_dim5) {
  EigenTensorMap(output.template MutableData<float>(), batch_size, out_dim1, out_dim2, out_dim3, out_dim4, out_dim5) =
      ConstEigenTensorMap(input.template Data<float>(), batch_size,
                          in_dim1, in_dim2, in_dim3, in_dim4, in_dim5)
          .shuffle(permutation);
}

template <>
Status SpaceToDepth<float>::Compute(OpKernelContext* context) const {
  const auto* tensor_pointer = context->Input<Tensor>(0);
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
  CreateOutput(input, output, permutation, batch,
               input_depth, input_height / blocksize_, blocksize_, input_width / blocksize_, blocksize_,
               blocksize_, blocksize_, input_depth, input_height / blocksize_, input_width / blocksize_);

  return Status::OK();
}

template <>
Status DepthToSpace<float>::Compute(OpKernelContext* context) const {
  const auto* tensor_pointer = context->Input<Tensor>(0);
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

  // handle DCR and CRD format
  auto dim1 = is_dcr_ ? blocksize_ : input_depth / blocksize_ / blocksize_;
  auto dim3 = is_dcr_ ? input_depth / blocksize_ / blocksize_ : blocksize_;

  auto permutation = is_dcr_ ? std::array<int64_t, IntermediateTensorRank>{{0, 3, 4, 1, 5, 2}}
                             : std::array<int64_t, IntermediateTensorRank>{{0, 1, 4, 2, 5, 3}};

  CreateOutput(input, output, permutation, batch,
               dim1, blocksize_, dim3, input_height, input_width,
               input_depth / blocksize_ / blocksize_, input_height, blocksize_, input_width, blocksize_);

  return Status::OK();
}

}  // namespace onnxruntime
