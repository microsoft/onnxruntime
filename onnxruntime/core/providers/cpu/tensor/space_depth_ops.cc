// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// TODO: x86 release builds produce warning C4723: potential divide by 0
#ifdef _MSC_VER
#pragma warning(disable : 4723)
#endif

#include "core/providers/cpu/tensor/space_depth_ops.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/common/eigen_common_wrapper.h"
#include <array>

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    SpaceToDepth,
    1,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>()}),
    SpaceToDepth);

ONNX_CPU_OPERATOR_KERNEL(
    SpaceToDepth,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>()}),
    SpaceToDepth);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    DepthToSpace,
    1, 10,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>()}),
    DepthToSpace);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    DepthToSpace,
    11,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>(),
                              DataTypeImpl::GetTensorType<uint8_t>()}),
    DepthToSpace);

ONNX_CPU_OPERATOR_KERNEL(
    DepthToSpace,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>(),
                              DataTypeImpl::GetTensorType<uint8_t>()}),
    DepthToSpace);

namespace contrib {
ONNX_OPERATOR_TYPED_KERNEL_EX(
    DepthToSpace,
    kMSDomain,
    1,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()),
    DepthToSpace);
}

// intermediate tensor shapes are:
// (batch, blocksize, blocksize, input_depth / (blocksize * blocksize), input_height, input_width) for DepthToSpace
// (batch, input_depth, input_height / blocksize, blocksize, input_width / blocksize, blocksize) for SpaceToDepth
constexpr int IntermediateTensorRank = 6;

template <typename T>
using EigenTensorMap = Eigen::TensorMap<Eigen::Tensor<T, IntermediateTensorRank, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>;

template <typename T>
using ConstEigenTensorMap = Eigen::TensorMap<Eigen::Tensor<const T, IntermediateTensorRank, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>;

// helper method to fill in output buffer
// only this portion is templated to minimize binary size
template <typename T>
static void SpaceDepthOpCpuImpl(const Tensor& input, Tensor& output,
                                const std::array<Eigen::DenseIndex, IntermediateTensorRank>& permutation,
                                const Eigen::DenseIndex batch_size,  // dim0 in both input and output
                                const Eigen::DenseIndex in_dim1, const Eigen::DenseIndex in_dim2, const Eigen::DenseIndex in_dim3,
                                const Eigen::DenseIndex in_dim4, const Eigen::DenseIndex in_dim5,
                                const Eigen::DenseIndex out_dim1, const Eigen::DenseIndex out_dim2, const Eigen::DenseIndex out_dim3,
                                const Eigen::DenseIndex out_dim4, const Eigen::DenseIndex out_dim5) {
  EigenTensorMap<T>(output.MutableData<T>(), batch_size, out_dim1, out_dim2, out_dim3, out_dim4, out_dim5) =
      ConstEigenTensorMap<T>(input.Data<T>(), batch_size,
                             in_dim1, in_dim2, in_dim3, in_dim4, in_dim5)
          .shuffle(permutation);
}

Status SpaceToDepth::Compute(OpKernelContext* context) const {
  const auto* tensor_pointer = context->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& input = *tensor_pointer;

  int64_t batch = -1;

  int64_t input_depth = -1;
  int64_t input_height = -1;
  int64_t input_width = -1;

  int64_t output_depth = -1;
  int64_t output_height = -1;
  int64_t output_width = -1;

  ORT_RETURN_IF_ERROR(InputValidationsAndOutputDimsCalc(input,
                                                        batch,
                                                        input_depth, input_height, input_width,
                                                        output_depth, output_height, output_width,
                                                        true));

  Tensor& output = *context->Output(0, {batch, output_depth, output_height, output_width});

  std::array<Eigen::DenseIndex, IntermediateTensorRank> permutation{{0, 3, 5, 1, 2, 4}};

  if (input.IsDataType<float>()) {
    SpaceDepthOpCpuImpl<float>(input, output, permutation,
                               onnxruntime::narrow<ptrdiff_t>(batch),
                               onnxruntime::narrow<std::ptrdiff_t>(input_depth),
                               onnxruntime::narrow<std::ptrdiff_t>(input_height / blocksize_),
                               onnxruntime::narrow<std::ptrdiff_t>(blocksize_),
                               onnxruntime::narrow<std::ptrdiff_t>(input_width / blocksize_),
                               onnxruntime::narrow<std::ptrdiff_t>(blocksize_),
                               onnxruntime::narrow<ptrdiff_t>(blocksize_),
                               onnxruntime::narrow<ptrdiff_t>(blocksize_),
                               onnxruntime::narrow<std::ptrdiff_t>(input_depth),
                               onnxruntime::narrow<std::ptrdiff_t>(input_height / blocksize_),
                               onnxruntime::narrow<std::ptrdiff_t>(input_width / blocksize_));
  } else if (input.IsDataType<double>()) {
    SpaceDepthOpCpuImpl<double>(input, output, permutation,
                                onnxruntime::narrow<ptrdiff_t>(batch),
                                onnxruntime::narrow<std::ptrdiff_t>(input_depth),
                                onnxruntime::narrow<std::ptrdiff_t>(input_height / blocksize_),
                                onnxruntime::narrow<std::ptrdiff_t>(blocksize_),
                                onnxruntime::narrow<std::ptrdiff_t>(input_width / blocksize_),
                                onnxruntime::narrow<std::ptrdiff_t>(blocksize_),
                                onnxruntime::narrow<ptrdiff_t>(blocksize_),
                                onnxruntime::narrow<ptrdiff_t>(blocksize_),
                                onnxruntime::narrow<std::ptrdiff_t>(input_depth),
                                onnxruntime::narrow<std::ptrdiff_t>(input_height / blocksize_),
                                onnxruntime::narrow<std::ptrdiff_t>(input_width / blocksize_));
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

  int64_t batch = -1;

  int64_t input_depth = -1;
  int64_t input_height = -1;
  int64_t input_width = -1;

  int64_t output_depth = -1;
  int64_t output_height = -1;
  int64_t output_width = -1;

  if (is_nhwc_) {
    ORT_RETURN_IF_ERROR(InputValidationsAndOutputDimsCalc<true>(input,
                                                          batch,
                                                          input_depth, input_height, input_width,
                                                          output_depth, output_height, output_width,
                                                          false));

    Tensor& output = *context->Output(0, {batch, output_height, output_width, output_depth});

    // handle DCR and CRD format
    auto dim3 = is_dcr_ ? blocksize_ : input_depth / blocksize_ / blocksize_;
    auto dim5 = is_dcr_ ? input_depth / blocksize_ / blocksize_ : blocksize_;

    int64_t virtual_input_depth = input_depth / blocksize_ / blocksize_;

    TensorShape virtual_input_shape;
    if (is_dcr_) {
      virtual_input_shape = TensorShape{batch, input_height, input_width,
                                        blocksize_, blocksize_, virtual_input_depth};
    } else {
      virtual_input_shape = TensorShape{batch, input_height, input_width,
                                        virtual_input_depth, blocksize_, blocksize_};
    }

    TensorShape virtual_output_shape = TensorShape{batch,
                                                   input_height, blocksize_,
                                                   input_width, blocksize_,
                                                   virtual_input_depth};

#if 0
    auto permutation = is_dcr_ ? std::array<Eigen::DenseIndex, IntermediateTensorRank>{{0, 1, 3, 2, 4, 5}}
                               : std::array<Eigen::DenseIndex, IntermediateTensorRank>{{0, 1, 4, 2, 5, 3}};
#else
    std::vector<size_t> permutation = is_dcr_ ? std::vector<size_t>{0, 1, 3, 2, 4, 5}
                                              : std::vector<size_t>{0, 1, 4, 2, 5, 3};
#endif

    if (input.IsDataType<uint8_t>()) {

      #if 0
      SpaceDepthOpCpuImpl<uint8_t>(input, output, permutation,
                                  onnxruntime::narrow<std::ptrdiff_t>(batch),
                                  onnxruntime::narrow<std::ptrdiff_t>(input_height),
                                  onnxruntime::narrow<std::ptrdiff_t>(input_width),
                                  onnxruntime::narrow<std::ptrdiff_t>(dim3),
                                  onnxruntime::narrow<std::ptrdiff_t>(blocksize_),
                                  onnxruntime::narrow<std::ptrdiff_t>(dim5),
                                  onnxruntime::narrow<std::ptrdiff_t>(input_height),
                                  onnxruntime::narrow<std::ptrdiff_t>(blocksize_),
                                  onnxruntime::narrow<std::ptrdiff_t>(input_width),
                                  onnxruntime::narrow<std::ptrdiff_t>(blocksize_),
                                  onnxruntime::narrow<std::ptrdiff_t>(input_depth / blocksize_ / blocksize_));
      #else
      return Transpose::DoTranspose(
        permutation, input, output, &virtual_input_shape, &virtual_output_shape, context->GetOperatorThreadPool());
      #endif

    } else {
      // user will not see this as the kernel doesn't claim support for types other than float and double
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported input type in DepthToSpace (channels_last = 1) op: ", input.DataType());
    }

    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(InputValidationsAndOutputDimsCalc(input,
                                                        batch,
                                                        input_depth, input_height, input_width,
                                                        output_depth, output_height, output_width,
                                                        false));

  Tensor& output = *context->Output(0, {batch, output_depth, output_height, output_width});

  // handle DCR and CRD format
  auto dim1 = is_dcr_ ? blocksize_ : input_depth / blocksize_ / blocksize_;
  auto dim3 = is_dcr_ ? input_depth / blocksize_ / blocksize_ : blocksize_;

  auto permutation = is_dcr_ ? std::array<Eigen::DenseIndex, IntermediateTensorRank>{{0, 3, 4, 1, 5, 2}}
                             : std::array<Eigen::DenseIndex, IntermediateTensorRank>{{0, 1, 4, 2, 5, 3}};

  if (input.IsDataType<float>()) {
    SpaceDepthOpCpuImpl<float>(input, output, permutation,
                               onnxruntime::narrow<std::ptrdiff_t>(batch),
                               onnxruntime::narrow<std::ptrdiff_t>(dim1),
                               onnxruntime::narrow<std::ptrdiff_t>(blocksize_),
                               onnxruntime::narrow<std::ptrdiff_t>(dim3),
                               onnxruntime::narrow<std::ptrdiff_t>(input_height),
                               onnxruntime::narrow<std::ptrdiff_t>(input_width),
                               onnxruntime::narrow<std::ptrdiff_t>(input_depth / blocksize_ / blocksize_),
                               onnxruntime::narrow<std::ptrdiff_t>(input_height),
                               onnxruntime::narrow<std::ptrdiff_t>(blocksize_),
                               onnxruntime::narrow<std::ptrdiff_t>(input_width),
                               onnxruntime::narrow<std::ptrdiff_t>(blocksize_));
  } else if (input.IsDataType<double>()) {
    SpaceDepthOpCpuImpl<double>(input, output, permutation,
                                onnxruntime::narrow<std::ptrdiff_t>(batch),
                                onnxruntime::narrow<std::ptrdiff_t>(dim1),
                                onnxruntime::narrow<std::ptrdiff_t>(blocksize_),
                                onnxruntime::narrow<std::ptrdiff_t>(dim3),
                                onnxruntime::narrow<std::ptrdiff_t>(input_height),
                                onnxruntime::narrow<std::ptrdiff_t>(input_width),
                                onnxruntime::narrow<std::ptrdiff_t>(input_depth / blocksize_ / blocksize_),
                                onnxruntime::narrow<std::ptrdiff_t>(input_height),
                                onnxruntime::narrow<std::ptrdiff_t>(blocksize_),
                                onnxruntime::narrow<std::ptrdiff_t>(input_width),
                                onnxruntime::narrow<std::ptrdiff_t>(blocksize_));
  } else if (input.IsDataType<uint8_t>()) {
    SpaceDepthOpCpuImpl<uint8_t>(input, output, permutation,
                                 onnxruntime::narrow<std::ptrdiff_t>(batch),
                                 onnxruntime::narrow<std::ptrdiff_t>(dim1),
                                 onnxruntime::narrow<std::ptrdiff_t>(blocksize_),
                                 onnxruntime::narrow<std::ptrdiff_t>(dim3),
                                 onnxruntime::narrow<std::ptrdiff_t>(input_height),
                                 onnxruntime::narrow<std::ptrdiff_t>(input_width),
                                 onnxruntime::narrow<std::ptrdiff_t>(input_depth / blocksize_ / blocksize_),
                                 onnxruntime::narrow<std::ptrdiff_t>(input_height),
                                 onnxruntime::narrow<std::ptrdiff_t>(blocksize_),
                                 onnxruntime::narrow<std::ptrdiff_t>(input_width),
                                 onnxruntime::narrow<std::ptrdiff_t>(blocksize_));
  } else {
    // user will not see this as the kernel doesn't claim support for types other than float and double
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported input type in DepthToSpace op: ", input.DataType());
  }

  return Status::OK();
}

}  // namespace onnxruntime
