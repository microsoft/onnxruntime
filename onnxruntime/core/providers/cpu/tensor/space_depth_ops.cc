// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// TODO: x86 release builds produce warning C4723: potential divide by 0
#ifdef _MSC_VER
#pragma warning(disable : 4723)
#endif

#include "core/providers/cpu/tensor/space_depth_ops.h"
#include "core/common/eigen_common_wrapper.h"
#include <array>

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    SpaceToDepth,
    1,
    12,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    SpaceToDepth);

ONNX_CPU_OPERATOR_KERNEL(
    SpaceToDepth,
    13,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    SpaceToDepth);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    DepthToSpace,
    1, 10,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    DepthToSpace);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    DepthToSpace,
    11,
    12,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    DepthToSpace);

ONNX_CPU_OPERATOR_KERNEL(
    DepthToSpace,
    13,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    DepthToSpace);

// intermediate tensor shapes are:
// (batch, blocksize, blocksize, input_depth / (blocksize * blocksize), input_height, input_width) for DepthToSpace
// (batch, input_depth, input_height / blocksize, blocksize, input_width / blocksize, blocksize) for SpaceToDepth
const int IntermediateTensorRank = 6;
typedef Eigen::TensorMap<Eigen::Tensor<float, IntermediateTensorRank, Eigen::RowMajor, Eigen::DenseIndex>,
                         Eigen::Aligned>
    EigenTensorMap;

typedef Eigen::TensorMap<Eigen::Tensor<const float, IntermediateTensorRank, Eigen::RowMajor, Eigen::DenseIndex>,
                         Eigen::Aligned>
    ConstEigenTensorMap;

// helper to create output to minimize binary size
template <typename T>
static void CreateOutput(const Tensor& input, Tensor& output,
                         const std::array<Eigen::DenseIndex, IntermediateTensorRank>& permutation,
                         const Eigen::DenseIndex batch_size,  // dim0 in both input and output
                         const Eigen::DenseIndex in_dim1, const Eigen::DenseIndex in_dim2, const Eigen::DenseIndex in_dim3,
                         const Eigen::DenseIndex in_dim4, const Eigen::DenseIndex in_dim5,
                         const Eigen::DenseIndex out_dim1, const Eigen::DenseIndex out_dim2, const Eigen::DenseIndex out_dim3,
                         const Eigen::DenseIndex out_dim4, const Eigen::DenseIndex out_dim5) {
  EigenTensorMap(output.template MutableData<float>(), batch_size, out_dim1, out_dim2, out_dim3, out_dim4, out_dim5) =
      ConstEigenTensorMap(input.template Data<float>(), batch_size,
                          in_dim1, in_dim2, in_dim3, in_dim4, in_dim5)
          .shuffle(permutation);
}

Status SpaceDepthBase::InputValidationsAndOutputDims(const Tensor& input,
                                                     int64_t& batch,
                                                     int64_t& input_depth, int64_t& input_height, int64_t& input_width,
                                                     int64_t& output_depth, int64_t& output_height, int64_t& output_width,
                                                     bool is_space_to_depth) const {
  const TensorShape& input_shape = input.Shape();
  if (input_shape.NumDimensions() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "SpaceDepth ops require a 4-D input. Provided rank: ",
                           input_shape.NumDimensions());
  }

  batch = input_shape[0];
  input_depth = input_shape[1];
  input_height = input_shape[2];
  input_width = input_shape[3];

  if (is_space_to_depth) {  // SpaceToDepth op
    if ((input_height % this->blocksize_) != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "SpaceToDepth requires input height to be a multiple of block_size");
    }

    if ((input_width % this->blocksize_) != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "SpaceToDepth requires input width to be a multiple of block_size");
    }

    output_depth = input_depth * blocksize_ * blocksize_;
    output_height = input_height / blocksize_;
    output_width = input_width / blocksize_;

  } else {  // DepthToSpace op
    if ((input_depth % (blocksize_ * blocksize_) != 0)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "DepthToSpace requires input depth to be a multiple of (block_size * blok_size)");
    }

    output_depth = input_depth / blocksize_ / blocksize_;
    output_height = input_height * blocksize_;
    output_width = input_width * blocksize_;
  }

  return Status::OK();
}

Status SpaceToDepth::Compute(OpKernelContext* context) const {
  const auto* tensor_pointer = context->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& input = *tensor_pointer;

  int64_t batch;

  int64_t input_depth;
  int64_t input_height;
  int64_t input_width;

  int64_t output_depth;
  int64_t output_height;
  int64_t output_width;

  ORT_RETURN_IF_ERROR(InputValidationsAndOutputDims(input,
                                                    batch,
                                                    input_depth, input_height, input_width,
                                                    output_depth, output_depth, output_depth,
                                                    true));

  Tensor& output = *context->Output(0, {batch, output_depth, output_height, output_width});

  std::array<Eigen::DenseIndex, IntermediateTensorRank> permutation{{0, 3, 5, 1, 2, 4}};

  if (input.IsDataType<float>()) {
    CreateOutput<float>(input, output, permutation, batch,
                        input_depth, input_height / blocksize_, blocksize_, input_width / blocksize_, blocksize_,
                        blocksize_, blocksize_, input_depth, input_height / blocksize_, input_width / blocksize_);
  } else if (input.IsDataType<double>()) {
    CreateOutput<double>(input, output, permutation, batch,
                         input_depth, input_height / blocksize_, blocksize_, input_width / blocksize_, blocksize_,
                         blocksize_, blocksize_, input_depth, input_height / blocksize_, input_width / blocksize_);
  } else {
    // user will not see this as the kernel doesn't claim support for types other than float and double
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported input type in SpaceToDepth op: ", input.DataType());
  }

  return Status::OK();
}

Status DepthToSpace::Compute(OpKernelContext* context) const {
  const auto* tensor_pointer = context->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& input = *tensor_pointer;

  int64_t batch;

  int64_t input_depth;
  int64_t input_height;
  int64_t input_width;

  int64_t output_depth;
  int64_t output_height;
  int64_t output_width;

  ORT_RETURN_IF_ERROR(InputValidationsAndOutputDims(input,
                                                    batch,
                                                    input_depth, input_height, input_width,
                                                    output_depth, output_depth, output_depth,
                                                    false));

  Tensor& output = *context->Output(0, {batch, output_depth, output_height, output_width});

  // handle DCR and CRD format
  auto dim1 = is_dcr_ ? blocksize_ : input_depth / blocksize_ / blocksize_;
  auto dim3 = is_dcr_ ? input_depth / blocksize_ / blocksize_ : blocksize_;

  auto permutation = is_dcr_ ? std::array<Eigen::DenseIndex, IntermediateTensorRank>{{0, 3, 4, 1, 5, 2}}
                             : std::array<Eigen::DenseIndex, IntermediateTensorRank>{{0, 1, 4, 2, 5, 3}};

  if (input.IsDataType<float>()) {
    CreateOutput<float>(input, output, permutation, batch,
                        dim1, blocksize_, dim3, input_height, input_width,
                        input_depth / blocksize_ / blocksize_, input_height, blocksize_, input_width, blocksize_);
  } else if (input.IsDataType<double>()) {
    CreateOutput<double>(input, output, permutation, batch,
                         dim1, blocksize_, dim3, input_height, input_width,
                         input_depth / blocksize_ / blocksize_, input_height, blocksize_, input_width, blocksize_);
  } else {
    // user will not see this as the kernel doesn't claim support for types other than float and double
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported input type in DepthToSpace op: ", input.DataType());
  }

  return Status::OK();
}

}  // namespace onnxruntime
