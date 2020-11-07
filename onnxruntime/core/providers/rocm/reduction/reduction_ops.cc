// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reduction_ops.h"
#include "core/providers/common.h"
#include "core/providers/rocm/miopen_common.h"
#include "core/providers/rocm/math/unary_elementwise_ops_impl.h"
#include "core/providers/rocm/math/binary_elementwise_ops_impl.h"
#include "core/providers/rocm/math/binary_elementwise_ops.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/framework/op_kernel_context_internal.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace rocm {

// opset 11 explicitly added support for negative axis. implementation already allowed it.
#define REGISTER_KERNEL_TYPED(name, T)                                          \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      name,                                                                     \
      kOnnxDomain,                                                              \
      1, 10,                                                                    \
      T,                                                                        \
      kRocmExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      name,                                                                     \
      kOnnxDomain,                                                              \
      11, 12,                                                                   \
      T,                                                                        \
      kRocmExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      name,                                                                     \
      kOnnxDomain,                                                              \
      13,                                                                       \
      T,                                                                        \
      kRocmExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

// Register those with changes in OpSet12.
#define REGISTER_KERNEL_TYPED_12(name, T)                                       \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      name,                                                                     \
      kOnnxDomain,                                                              \
      1, 10,                                                                    \
      T,                                                                        \
      kRocmExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      name,                                                                     \
      kOnnxDomain,                                                              \
      11, 11,                                                                   \
      T,                                                                        \
      kRocmExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      name,                                                                     \
      kOnnxDomain,                                                              \
      12, 12,                                                                   \
      T,                                                                        \
      kRocmExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      name,                                                                     \
      kOnnxDomain,                                                              \
      13,                                                                       \
      T,                                                                        \
      kRocmExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

// CUDA ArgMax/ArgMin doesn't have OpSet12 implementation (with select_last_index attr), keep it in OpSet11 for now.
#define REGISTER_KERNEL_TYPED_11(name, T)                                       \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      name,                                                                     \
      kOnnxDomain,                                                              \
      1, 10,                                                                    \
      T,                                                                        \
      kRocmExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      name,                                                                     \
      kOnnxDomain,                                                              \
      11,                                                                       \
      T,                                                                        \
      kRocmExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

static bool is_matrix_row_reduction(
    const miopenReduceTensorOp_t miopen_reduce_op,
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

  if (miopen_reduce_op != MIOPEN_REDUCE_TENSOR_ADD)
    return false;

  //empty axes, default reduction
  if (axes.size() < 1)
    return false;

  return true;
}

// TODO ReduceKernel::ReduceKernelShared() is still used by some other training classes though it's not used here - this should be refactored.
template <bool allow_multi_axes>
template <typename T, typename OutT>
Status ReduceKernel<allow_multi_axes>::ReduceKernelShared(
    const T* X,
    const TensorShape& input_shape,
    OutT* Y,
    const TensorShape& /*output_shape*/,
    miopenReduceTensorOp_t miopen_reduce_op,
    std::vector<int64_t> /*output_dims*/) const {
  typedef typename ToHipType<T>::MappedType HipT;
  const auto rank = input_shape.NumDimensions();

  // Block of fast matrix row reduction.
  // It relies on new atomicAdd for half type, so old hip can't use it.
  const auto stride = input_shape[input_shape.NumDimensions() - 1];
  const auto reduction_size = input_shape.Size() / stride;
  if (fast_reduction_ && reduction_size <= std::numeric_limits<int>::max() && stride <= std::numeric_limits<int>::max() &&
      is_matrix_row_reduction(miopen_reduce_op,
                              static_cast<int>(reduction_size),
                              static_cast<int>(stride), rank, axes_)) {
    reduce_matrix_rows(
        reinterpret_cast<const HipT*>(X),
        reinterpret_cast<HipT*>(Y),
        static_cast<int>(reduction_size),
        static_cast<int>(stride));
    return Status::OK();
  }

  // TODO: miOpen doesn't support reduction op as CUDNN. Two options:
  // 1) implement reduction ops by ourselves 2) ask AMD to support same reduction functionality as CUDNN.
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "reduction1 is not supported");
}

template Status ReduceKernel<true>::ReduceKernelShared<double, double>(
    const double* X,
    const TensorShape& input_shape,
    double* Y,
    const TensorShape& output_shape,
    miopenReduceTensorOp_t miopen_reduce_op,
    std::vector<int64_t> output_dims) const;

template Status ReduceKernel<true>::ReduceKernelShared<float, float>(
    const float* X,
    const TensorShape& input_shape,
    float* Y,
    const TensorShape& output_shape,
    miopenReduceTensorOp_t miopen_reduce_op,
    std::vector<int64_t> output_dims) const;

template Status ReduceKernel<true>::ReduceKernelShared<MLFloat16, MLFloat16>(
    const MLFloat16* X,
    const TensorShape& input_shape,
    MLFloat16* Y,
    const TensorShape& output_shape,
    miopenReduceTensorOp_t miopen_reduce_op,
    std::vector<int64_t> output_dims) const;

// `input_shape_override` (if provided) is the input shape for compute purposes
Status PrepareForReduce(const Tensor* X,
                        bool keepdims,
                        const std::vector<int64_t>& axes,
                        PrepareReduceMetadata& prepare_reduce_metadata,
                        const TensorShape* input_shape_override) {
  ORT_ENFORCE(nullptr != X);

  const TensorShape& input_shape = input_shape_override ? *input_shape_override : X->Shape();
  int64_t rank = static_cast<int64_t>(input_shape.NumDimensions());
  prepare_reduce_metadata.rank = rank;
  prepare_reduce_metadata.input_count = input_shape.Size();
  prepare_reduce_metadata.stride = (rank > 0) ? input_shape[input_shape.NumDimensions() - 1] : 1;
  prepare_reduce_metadata.contiguous_axes = false;

  if (rank > 8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "miopen only supports up to 8-D tensors in reduction");
  }

  const auto& input_dims = input_shape.GetDims();
  std::vector<bool> reduced(rank, false);
  prepare_reduce_metadata.output_dims.reserve(input_dims.size());
  if (axes.size() > 0) {
    int64_t reduced_axis;
    std::vector<uint64_t> reduced_axes(axes.size());
    prepare_reduce_metadata.output_dims = input_dims;
    for (size_t i = 0; i < axes.size(); i++) {
      reduced_axis = axes[i];
      const int64_t axis = HandleNegativeAxis(reduced_axis, rank);
      ORT_ENFORCE(input_dims[axis] != 0,
                  "Can't reduce on dim with value of 0 if 'keepdims' is false. "
                  "Invalid output shape would be produced. input_shape:",
                  input_shape);
      prepare_reduce_metadata.output_dims[axis] = 1;
      reduced[axis] = true;
      reduced_axes[i] = axis;
    }

    bool contiguous_axes = true;
    std::sort(reduced_axes.begin(), reduced_axes.end());
    for (size_t i = 0; i < reduced_axes.size(); i++) {
      if (reduced_axes[i] != i) {
        contiguous_axes = false;
        break;
      }
    }
    int64_t stride = 1;
    if (contiguous_axes) {
      for (size_t s = rank - 1; s >= reduced_axes.size(); s--) {
        stride *= input_dims[s];
      }
      prepare_reduce_metadata.stride = stride;
      prepare_reduce_metadata.contiguous_axes = true;
    }
  } else {
    // no axes provided (i.e.) default axes  => reduce on all dims
    for (auto dim : input_dims) {
      ORT_ENFORCE(keepdims || dim != 0,
                  "Can't reduce on dim with value of 0 if 'keepdims' is false. "
                  "Invalid output shape would be produced. input_shape:",
                  input_shape);
      prepare_reduce_metadata.output_dims.push_back(dim == 0 ? 0 : 1);
    }
  }

  if (keepdims) {
    prepare_reduce_metadata.squeezed_output_dims = prepare_reduce_metadata.output_dims;
  } else if (axes.size() > 0) {
    // we are not going to keep the reduced dims, hence compute the final output dim accordingly
    prepare_reduce_metadata.squeezed_output_dims.reserve(rank);  // even though we won't use the full capacity, it is better to reserve for peak possible usage
    for (auto i = 0; i < rank; ++i) {
      if (!reduced[i])
        prepare_reduce_metadata.squeezed_output_dims.push_back(input_dims[i]);
    }
  } else {
    // 'axes' is empty and keepdims is false => we reduce on all axes AND drop all dims,
    // so the result is just a scalar, we keep 'squeezed_output_dims' empty (i.e.) no-op
  }

  // miopen requires at least 3D input, so pad 1s if needed
  prepare_reduce_metadata.input_dims_miopen = input_dims;
  prepare_reduce_metadata.output_dims_miopen = prepare_reduce_metadata.output_dims;
  if (rank < 3) {
    std::vector<int64_t> pads(3 - rank, 1);
    prepare_reduce_metadata.input_dims_miopen.insert(prepare_reduce_metadata.input_dims_miopen.end(), pads.begin(), pads.end());
    prepare_reduce_metadata.output_dims_miopen.insert(prepare_reduce_metadata.output_dims_miopen.end(), pads.begin(), pads.end());
  }

  prepare_reduce_metadata.output_count = TensorShape(prepare_reduce_metadata.output_dims).Size();

  if (prepare_reduce_metadata.rank == 0) {
    prepare_reduce_metadata.rank = 1;
  }

  return Status::OK();
}

// `input_shape_override` is the input shape for compute purposes (if provided)
template <typename T>
Status ReduceComputeCore(const Tensor& input, PrepareReduceMetadata& prepare_reduce_metadata,
                         /*out*/ Tensor& output, miopenReduceTensorOp_t miopen_reduce_op,
                         const std::vector<int64_t>& axes,
                         bool calculate_log, bool calculate_sqt, bool log_sum_exp, bool fast_reduction,
                         const TensorShape* input_shape_override) {
  typedef typename ToHipType<T>::MappedType HipT;
  // const TensorShape& input_shape = input_shape_override ? *input_shape_override : input.Shape();

  int64_t input_count = prepare_reduce_metadata.input_count;
  int64_t output_count = prepare_reduce_metadata.output_count;
  // std::vector<int64_t>& output_dims = prepare_reduce_metadata.output_dims;
  // std::vector<int64_t>& input_dims_miopen = prepare_reduce_metadata.input_dims_miopen;
  // std::vector<int64_t>& output_dims_miopen = prepare_reduce_metadata.output_dims_miopen;
  int64_t rank = prepare_reduce_metadata.rank;
  int64_t stride = prepare_reduce_metadata.stride;

  // special case when there is a dim value of 0 in the shape.
  if (input_count == 0) {
    assert(output.Shape().Size() == 0);
    return Status::OK();
  }

  // This reduction keep adding values to this buffer. If a non-zero value, say 1000, is here, the sum will start with 1000.
  // Therefore zeroing out the memory is required
  HIP_RETURN_IF_ERROR(hipMemsetAsync(output.MutableDataRaw(), 0, output.SizeInBytes()));

  // Block of fast matrix row reduction.
  // It relies on new atomicAdd for half type, so old CUDA can't use it.
  const auto reduction_size = input_count / stride;
  if (!std::is_same<T, int8_t>::value && !std::is_same<T, uint8_t>::value) {
    if (fast_reduction && reduction_size <= std::numeric_limits<int>::max() && stride <= std::numeric_limits<int>::max() &&
        prepare_reduce_metadata.contiguous_axes &&
        is_matrix_row_reduction(miopen_reduce_op, static_cast<int>(reduction_size), static_cast<int>(stride), rank, axes)) {
      reduce_matrix_rows(
          reinterpret_cast<const HipT*>(input.template Data<T>()),
          reinterpret_cast<HipT*>(output.template MutableData<T>()),
          static_cast<int>(reduction_size),
          static_cast<int>(stride));
      return Status::OK();
    }
  }

  if (input_count == output_count) {
    if (output.template MutableData<T>() != input.template Data<T>()) {
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(output.template MutableData<T>(), input.template Data<T>(), input_count * sizeof(T), hipMemcpyDeviceToDevice));
    }
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "reduction2 is not supported");
}

template <bool allow_multi_axes>
template <typename T>
Status ReduceKernel<allow_multi_axes>::ComputeImpl(OpKernelContext* ctx, miopenReduceTensorOp_t miopen_reduce_op) const {
  const Tensor* X = ctx->Input<Tensor>(0);

  PrepareReduceMetadata prepare_reduce_metadata;
  ORT_RETURN_IF_ERROR(PrepareForReduce(X,
                                       keepdims_,
                                       axes_,
                                       prepare_reduce_metadata));
  Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);
  const bool fast_reduction = fast_reduction_ && !ctx->GetUseDeterministicCompute();

  return ReduceComputeCore<T>(*X, prepare_reduce_metadata, *Y, miopen_reduce_op, axes_,
                              calculate_log_, calculate_sqt_, log_sum_exp_, fast_reduction);
}

template <>
template <>
Status ReduceKernel<true>::ComputeImpl<int32_t>(OpKernelContext* /*ctx*/, miopenReduceTensorOp_t /*miopen_reduce_op*/) const {
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, Node().OpType(), " is not supported");
}

#define REGISTER_KERNEL_HFD(name)        \
  REGISTER_KERNEL_TYPED(name, MLFloat16) \
  REGISTER_KERNEL_TYPED(name, float)     \
  REGISTER_KERNEL_TYPED(name, double)

#define REGISTER_KERNEL_HFD_11(name)        \
  REGISTER_KERNEL_TYPED_11(name, MLFloat16) \
  REGISTER_KERNEL_TYPED_11(name, float)     \
  REGISTER_KERNEL_TYPED_11(name, double)

REGISTER_KERNEL_HFD_11(ArgMax)
REGISTER_KERNEL_HFD_11(ArgMin)
REGISTER_KERNEL_HFD(ReduceL1)
REGISTER_KERNEL_HFD(ReduceL2)

REGISTER_KERNEL_TYPED_12(ReduceMax, MLFloat16)
REGISTER_KERNEL_TYPED_12(ReduceMax, float)
REGISTER_KERNEL_TYPED_12(ReduceMax, double)
REGISTER_KERNEL_TYPED_12(ReduceMax, int32_t)
REGISTER_KERNEL_TYPED_12(ReduceMax, int8_t)
REGISTER_KERNEL_TYPED_12(ReduceMax, uint8_t)

REGISTER_KERNEL_HFD(ReduceMean)

REGISTER_KERNEL_TYPED_12(ReduceMin, MLFloat16)
REGISTER_KERNEL_TYPED_12(ReduceMin, float)
REGISTER_KERNEL_TYPED_12(ReduceMin, double)
REGISTER_KERNEL_TYPED_12(ReduceMin, int32_t)
REGISTER_KERNEL_TYPED_12(ReduceMin, int8_t)
REGISTER_KERNEL_TYPED_12(ReduceMin, uint8_t)

REGISTER_KERNEL_HFD(ReduceProd)
REGISTER_KERNEL_HFD(ReduceSum)
REGISTER_KERNEL_HFD(ReduceLogSum)
REGISTER_KERNEL_HFD(ReduceSumSquare)
REGISTER_KERNEL_HFD(ReduceLogSumExp)

#define REGISTER_KERNEL_INT32(name) \
  REGISTER_KERNEL_TYPED(name, int32_t)

REGISTER_KERNEL_INT32(ReduceL1)
REGISTER_KERNEL_INT32(ReduceL2)
REGISTER_KERNEL_INT32(ReduceMean)

REGISTER_KERNEL_INT32(ReduceProd)
REGISTER_KERNEL_INT32(ReduceSum)

}  // namespace rocm
}  // namespace onnxruntime