// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"

#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/hip/math/unary_elementwise_ops_impl.h"
#include "core/providers/hip/math/binary_elementwise_ops_impl.h"
#include "core/providers/hip/math/binary_elementwise_ops.h"

#include "core/providers/hip/hip_common.h"
#include "core/providers/hip/reduction/reduction_ops.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace hip {

// opset 11 explicitly added support for negative axis. implementation already allowed it.
#define REGISTER_KERNEL_TYPED(name, T)                                          \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      name,                                                                     \
      kOnnxDomain,                                                              \
      1, 10,                                                                    \
      T,                                                                        \
      kHipExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      name,                                                                     \
      kOnnxDomain,                                                              \
      11,                                                                       \
      T,                                                                        \
      kHipExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

static bool is_matrix_row_reduction(
    const HipReduceTensorType reduce_type,
    const int m,
    const int n,
    const size_t rank,
    std::vector<int64_t> axes) {
  if (m < 1)
    return false;

  if (n < 1)
    return false;

  if (rank < 2)
    return false;

  if (reduce_type != HipReduceTensorType::HIP_REDUCE_TENSOR_ADD)
    return false;

  // Check if all but the last axis are reduced. For example, reducing
  // [N, C, H, W]-tensor to [W]-tensor can pass these two checks but reducing
  // [N, C]-tensor to [N, 1]-tensor cannot.
  if (axes.size() != rank - 1)
    return false;

  // The last reduced axis should be the second last axis. For
  // [N, C, H, W]-input, the sorted axes should be [0, 1, 2].
  std::sort(axes.begin(), axes.end());
  if (axes.back() != rank - 2)
    return false;

  return true;
}

// TODO ReduceKernel::ReduceKernelShared() is still used by some other training classes though it's not used here - this should be refactored.
// template <bool allow_multi_axes>
// template <typename T, typename OutT>
// Status ReduceKernel<allow_multi_axes>::ReduceKernelShared(
//     const T* X,
//     const TensorShape& input_shape,
//     OutT* Y,
//     const TensorShape& output_shape,
//     HipReduceTensorType reduce_type,
//     std::vector<int64_t> output_dims) const {
//   typedef typename ToHipType<T>::MappedType HipT;
//   const auto rank = input_shape.NumDimensions();

// // Block of fast matrix row reduction.
// // It relies on new atomicAdd for half type, so old hip can't use it.
//   const auto stride = input_shape[input_shape.NumDimensions() - 1];
//   const auto reduction_size = input_shape.Size() / stride;
//   if (fast_reduction_ && reduction_size <= std::numeric_limits<int>::max() && stride <= std::numeric_limits<int>::max() &&
//       is_matrix_row_reduction(reduce_type,
//         static_cast<int>(reduction_size),
//         static_cast<int>(stride), rank, axes_)) {

//     reduce_matrix_rows(
//       reinterpret_cast<const HipT*>(X),
//       reinterpret_cast<HipT*>(Y),
//       static_cast<int>(reduction_size),
//       static_cast<int>(stride));
//     return Status::OK();
//   }

//   return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "reduction is not supported");
// }

// template Status ReduceKernel<true>::ReduceKernelShared<double, double>(
//     const double* X,
//     const TensorShape& input_shape,
//     double* Y,
//     const TensorShape& output_shape,
//     HipReduceTensorType reduce_type,
//     std::vector<int64_t> output_dims) const;

// template Status ReduceKernel<true>::ReduceKernelShared<float, float>(
//     const float* X,
//     const TensorShape& input_shape,
//     float* Y,
//     const TensorShape& output_shape,
//     HipReduceTensorType reduce_type,
//     std::vector<int64_t> output_dims) const;

// template Status ReduceKernel<true>::ReduceKernelShared<MLFloat16, MLFloat16>(
//     const MLFloat16* X,
//     const TensorShape& input_shape,
//     MLFloat16* Y,
//     const TensorShape& output_shape,
//     HipReduceTensorType reduce_type,
//     std::vector<int64_t> output_dims) const;

static Status PrepareForReduce(OpKernelContext* ctx,
                               bool keepdims,
                               const std::vector<int64_t>& axes,
                               const Tensor** x_pp,
                               Tensor** y_pp,
                               int64_t& input_count,
                               int64_t& output_count,
                               std::vector<int64_t>& output_dims,
                               std::vector<int64_t>& input_dims_cudnn,
                               std::vector<int64_t>& output_dims_cudnn,
                               int64_t& rank,
                               int64_t& stride) {
  const Tensor* X = ctx->Input<Tensor>(0);
  ORT_ENFORCE(nullptr != X);
  *x_pp = X;

  const TensorShape input_shape{X->Shape()};
  rank = input_shape.NumDimensions();
  input_count = input_shape.Size();
  stride = input_shape[input_shape.NumDimensions() - 1];

  if (rank > 8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "cuDNN only supports up to 8-D tensors in reduction");
  }

  const auto& input_dims = input_shape.GetDims();
  std::vector<bool> reduced(rank, false);
  std::vector<int64_t> squeezed_output_dims;
  output_dims.reserve(input_dims.size());
  if (axes.size() > 0) {
    output_dims = input_dims;
    for (auto reduced_axis : axes) {
      const int64_t axis = HandleNegativeAxis(reduced_axis, rank);
      ORT_ENFORCE(input_dims[axis] != 0,
                  "Can't reduce on dim with value of 0 if 'keepdims' is false. "
                  "Invalid output shape would be produced. input_shape:",
                  input_shape);
      output_dims[axis] = 1;
      reduced[axis] = true;
    }
  } else {
    // no axes provided (i.e.) default axes  => reduce on all dims
    for (auto dim : input_dims) {
      ORT_ENFORCE(keepdims || dim != 0,
                  "Can't reduce on dim with value of 0 if 'keepdims' is false. "
                  "Invalid output shape would be produced. input_shape:",
                  input_shape);
      output_dims.push_back(dim == 0 ? 0 : 1);
    }
  }

  if (keepdims) {
    squeezed_output_dims = output_dims;
  } else if (axes.size() > 0) {
    // we are not going to keep the reduced dims, hence compute the final output dim accordingly
    squeezed_output_dims.reserve(rank);  // even though we won't use the full capacity, it is better to reserve for peak possible usage
    for (auto i = 0; i < rank; ++i) {
      if (!reduced[i])
        squeezed_output_dims.push_back(input_dims[i]);
    }
  } else {
    // 'axes' is empty and keepdims is false => we reduce on all axes AND drop all dims,
    // so the result is just a scalar, we keep 'squeezed_output_dims' empty (i.e.) no-op
  }

  Tensor* Y = ctx->Output(0, TensorShape(squeezed_output_dims));
  HIP_RETURN_IF_ERROR(hipMemset(Y->MutableDataRaw(), 0, Y->SizeInBytes()));
  *y_pp = Y;

  // CUDNN requires at least 3D input, so pad 1s if needed
  input_dims_cudnn = input_dims;
  output_dims_cudnn = output_dims;
  if (rank < 3) {
    std::vector<int64_t> pads(3 - rank, 1);
    input_dims_cudnn.insert(input_dims_cudnn.end(), pads.begin(), pads.end());
    output_dims_cudnn.insert(output_dims_cudnn.end(), pads.begin(), pads.end());
  }

  output_count = Y->Shape().Size();

  return Status::OK();
}

template <bool allow_multi_axes>
template <typename T>
Status ReduceKernel<allow_multi_axes>::ComputeImpl(OpKernelContext* ctx, HipReduceTensorType reduce_type) const {
  typedef typename ToHipType<T>::MappedType HipT;
  const Tensor* X = nullptr;
  Tensor* Y = nullptr;

  int64_t input_count = 0;
  int64_t output_count = 0;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> input_dims_cudnn;
  std::vector<int64_t> output_dims_cudnn;
  int64_t rank = 0;
  int64_t stride = 0;
  ORT_RETURN_IF_ERROR(PrepareForReduce(ctx,
                                       keepdims_,
                                       axes_,
                                       &X,
                                       &Y,
                                       input_count,
                                       output_count,
                                       output_dims,
                                       input_dims_cudnn,
                                       output_dims_cudnn,
                                       rank, stride));

  // special case when there is a dim value of 0 in the shape.
  if (input_count == 0) {
    assert(Y->Shape().Size() == 0);
    return Status::OK();
  }

  // Block of fast matrix row reduction.
  // It relies on new atomicAdd for half type, so old hip can't use it.
  const auto reduction_size = input_count / stride;
  if (fast_reduction_ && reduction_size <= std::numeric_limits<int>::max() && stride <= std::numeric_limits<int>::max() &&
      is_matrix_row_reduction(reduce_type,
        static_cast<int>(reduction_size),
        static_cast<int>(stride), rank, axes_)) {

    reduce_matrix_rows(
      reinterpret_cast<const HipT*>(X->template Data<T>()),
      reinterpret_cast<HipT*>(Y->template MutableData<T>()),
      static_cast<int>(reduction_size),
      static_cast<int>(stride));
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "reduction is not supported");
}

template <>
template <>
Status ReduceKernel<true>::ComputeImpl<int32_t>(OpKernelContext* /*ctx*/, HipReduceTensorType /*reduce_type*/) const {
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "reduction is not supported");
}

#define REGISTER_KERNEL_HFD(name)        \
  REGISTER_KERNEL_TYPED(name, MLFloat16) \
  REGISTER_KERNEL_TYPED(name, float)     \
  REGISTER_KERNEL_TYPED(name, double)

// REGISTER_KERNEL_HFD(ArgMax)
// REGISTER_KERNEL_HFD(ArgMin)
// REGISTER_KERNEL_HFD(ReduceL1)
// REGISTER_KERNEL_HFD(ReduceL2)
// REGISTER_KERNEL_HFD(ReduceMax)
REGISTER_KERNEL_HFD(ReduceMean)
// REGISTER_KERNEL_HFD(ReduceMin)
// REGISTER_KERNEL_HFD(ReduceProd)
REGISTER_KERNEL_HFD(ReduceSum)
// REGISTER_KERNEL_HFD(ReduceLogSum)
// REGISTER_KERNEL_HFD(ReduceSumSquare)
// REGISTER_KERNEL_HFD(ReduceLogSumExp)

#define REGISTER_KERNEL_INT32(name) \
  REGISTER_KERNEL_TYPED(name, int32_t)

// REGISTER_KERNEL_INT32(ReduceL1)
// REGISTER_KERNEL_INT32(ReduceL2)
// REGISTER_KERNEL_INT32(ReduceMax)
REGISTER_KERNEL_INT32(ReduceMean)
// REGISTER_KERNEL_INT32(ReduceMin)
// REGISTER_KERNEL_INT32(ReduceProd)
REGISTER_KERNEL_INT32(ReduceSum)

}  // namespace hip
}  // namespace onnxruntime