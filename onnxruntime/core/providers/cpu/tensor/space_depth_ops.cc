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

template <typename T>
static Status DepthSpaceDCRBlock4Width8XOpCpuImpl(OpKernelContext* ctx,
                                                  const Tensor& input, Tensor& output,
                                                  const int64_t batch,
                                                  const int64_t input_depth,
                                                  const int64_t input_height,
                                                  const int64_t input_width) {
  std::vector<size_t> permutations = {0, 3, 4, 1, 5, 2};
  constexpr int blocksize = 4;
  constexpr int internal_rank = 6;
  int64_t internal_output_depth = input_depth / blocksize / blocksize;
  const TensorShape internal_input_shape = TensorShape{batch, blocksize, blocksize,
                                                       internal_output_depth, input_height, input_width};
  const TensorShape internal_output_shape = TensorShape{batch, internal_output_depth,
                                                        input_height, blocksize,
                                                        input_width, blocksize};
  const int64_t number_of_elements = internal_input_shape.Size();
  const auto& internal_output_dims = internal_output_shape.GetDims();

  InlinedVector<size_t> stride(internal_rank);
  for (size_t i = 0; i < internal_rank; i++) {
    size_t inpdim = permutations[i];
    if (inpdim + 1 < internal_rank)
      stride[i] = onnxruntime::narrow<size_t>(internal_input_shape.SizeFromDimension(inpdim + 1));
    else
      stride[i] = 1;
  }

  InlinedVector<size_t> internal_output_stride(internal_rank);
  internal_output_stride[internal_rank - 1] = 1;
  for (int64_t i = internal_rank - 2; i >= 0; --i) {
    internal_output_stride[i] = internal_output_stride[i + 1] * internal_output_dims[i + 1];
  }

  const auto* input_data = reinterpret_cast<const T*>(input.DataRaw());
  auto* output_data = reinterpret_cast<T*>(output.MutableDataRaw());

  Status status = Status::OK();

  concurrency::ThreadPool::TryParallelFor(
      ctx->GetOperatorThreadPool(), static_cast<std::ptrdiff_t>(number_of_elements),
      {static_cast<float>(sizeof(uint8_t)), static_cast<float>(sizeof(uint8_t)), 1.0F},
      [&internal_output_stride, input_data, &stride, output_data](std::ptrdiff_t first,
                                                                  std::ptrdiff_t last) {
        constexpr int chunk_size = 32;

        ORT_ENFORCE((first < last) && (first % chunk_size == 0) && (last % chunk_size == 0));

        /// The loop is unrolled by 32 for the code below:
        /// for (std::ptrdiff_t i = first; i < last; ++i) {
        ///   int d0 = static_cast<int>(i / internal_output_stride[0]);
        ///   int d1 = static_cast<int>((i % internal_output_stride[0]) / internal_output_stride[1]);
        ///   int d2 = static_cast<int>((i % internal_output_stride[1]) / internal_output_stride[2]);
        ///   int d3 = static_cast<int>((i % internal_output_stride[2]) / internal_output_stride[3]);
        ///   int d4 = static_cast<int>((i % internal_output_stride[3]) / internal_output_stride[4] /* blocksize = 4 */);
        ///   int d5 = static_cast<int>(i % internal_output_stride[4] /* blocksize = 4 */);
        ///   const T* source = input_data + (d0 * stride[0] +
        ///                                   d1 * stride[1] +
        ///                                   d2 * stride[2] +
        ///                                   d3 * stride[3] +
        ///                                   d4 * stride[4] /* 1 */ +
        ///                                   d5 * stride[5]);
        ///   T* target = output_data + i;
        ///   *target = *source;
        /// }
        for (std::ptrdiff_t i = first; i < last; i += chunk_size) {
          int d0 = static_cast<int>(i / internal_output_stride[0]);
          int d1 = static_cast<int>((i % internal_output_stride[0]) / internal_output_stride[1]);
          int d2 = static_cast<int>((i % internal_output_stride[1]) / internal_output_stride[2]);
          int d3 = static_cast<int>((i % internal_output_stride[2]) / internal_output_stride[3]);
          int d4 = static_cast<int>((i % internal_output_stride[3]) / 4 /* blocksize = internal_output_stride[4] */);
          const T* source = input_data + (d0 * stride[0] +
                                          d1 * stride[1] +
                                          d2 * stride[2] +
                                          d3 * stride[3] +
                                          d4 * 1 /* stride[4] */);
          T* target = output_data + i;
          *(target + 0) = *(source + ((0 / 4) * 1) + ((0 % 4) * stride[5]));
          *(target + 1) = *(source + ((1 / 4) * 1) + ((1 % 4) * stride[5]));
          *(target + 2) = *(source + ((2 / 4) * 1) + ((2 % 4) * stride[5]));
          *(target + 3) = *(source + ((3 / 4) * 1) + ((3 % 4) * stride[5]));
          *(target + 4) = *(source + ((4 / 4) * 1) + ((4 % 4) * stride[5]));
          *(target + 5) = *(source + ((5 / 4) * 1) + ((5 % 4) * stride[5]));
          *(target + 6) = *(source + ((6 / 4) * 1) + ((6 % 4) * stride[5]));
          *(target + 7) = *(source + ((7 / 4) * 1) + ((7 % 4) * stride[5]));
          *(target + 8) = *(source + ((8 / 4) * 1) + ((8 % 4) * stride[5]));
          *(target + 9) = *(source + ((9 / 4) * 1) + ((9 % 4) * stride[5]));
          *(target + 10) = *(source + ((10 / 4) * 1) + ((10 % 4) * stride[5]));
          *(target + 11) = *(source + ((11 / 4) * 1) + ((11 % 4) * stride[5]));
          *(target + 12) = *(source + ((12 / 4) * 1) + ((12 % 4) * stride[5]));
          *(target + 13) = *(source + ((13 / 4) * 1) + ((13 % 4) * stride[5]));
          *(target + 14) = *(source + ((14 / 4) * 1) + ((14 % 4) * stride[5]));
          *(target + 15) = *(source + ((15 / 4) * 1) + ((15 % 4) * stride[5]));
          *(target + 16) = *(source + ((16 / 4) * 1) + ((16 % 4) * stride[5]));
          *(target + 17) = *(source + ((17 / 4) * 1) + ((17 % 4) * stride[5]));
          *(target + 18) = *(source + ((18 / 4) * 1) + ((18 % 4) * stride[5]));
          *(target + 19) = *(source + ((19 / 4) * 1) + ((19 % 4) * stride[5]));
          *(target + 20) = *(source + ((20 / 4) * 1) + ((20 % 4) * stride[5]));
          *(target + 21) = *(source + ((21 / 4) * 1) + ((21 % 4) * stride[5]));
          *(target + 22) = *(source + ((22 / 4) * 1) + ((22 % 4) * stride[5]));
          *(target + 23) = *(source + ((23 / 4) * 1) + ((23 % 4) * stride[5]));
          *(target + 24) = *(source + ((24 / 4) * 1) + ((24 % 4) * stride[5]));
          *(target + 25) = *(source + ((25 / 4) * 1) + ((25 % 4) * stride[5]));
          *(target + 26) = *(source + ((26 / 4) * 1) + ((26 % 4) * stride[5]));
          *(target + 27) = *(source + ((27 / 4) * 1) + ((27 % 4) * stride[5]));
          *(target + 28) = *(source + ((28 / 4) * 1) + ((28 % 4) * stride[5]));
          *(target + 29) = *(source + ((29 / 4) * 1) + ((29 % 4) * stride[5]));
          *(target + 30) = *(source + ((30 / 4) * 1) + ((30 % 4) * stride[5]));
          *(target + 31) = *(source + ((31 / 4) * 1) + ((31 % 4) * stride[5]));
        }
      });

  return status;
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
    if (is_dcr_ && (blocksize_ == 4) && (input_width % 8 == 0)) {
      ORT_RETURN_IF_ERROR(DepthSpaceDCRBlock4Width8XOpCpuImpl<uint8_t>(context, input, output,
                                                                       batch, input_depth, input_height, input_width));
    } else {
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
    }
  } else {
    // user will not see this as the kernel doesn't claim support for types other than float and double
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported input type in DepthToSpace op: ", input.DataType());
  }

  return Status::OK();
}

}  // namespace onnxruntime
