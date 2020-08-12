// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reduction_ops.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/framework/op_kernel_context_internal.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

// opset 11 explicitly added support for negative axis. implementation already allowed it.
#define REGISTER_KERNEL_TYPED(name, T)                                          \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      name,                                                                     \
      kOnnxDomain,                                                              \
      1, 10,                                                                    \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      name,                                                                     \
      kOnnxDomain,                                                              \
      11,                                                                       \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

// Register with the latest version 12
#define REGISTER_KERNEL_TYPED_12(name, T)                                       \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      name,                                                                     \
      kOnnxDomain,                                                              \
      1, 10,                                                                    \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      name,                                                                     \
      kOnnxDomain,                                                              \
      11, 11,                                                                   \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      name,                                                                     \
      kOnnxDomain,                                                              \
      12,                                                                       \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

// TODO ReduceKernel::ReduceKernelShared() is still used by some other training classes though it's not used here - this should be refactored.
template <bool allow_multi_axes>
template <typename T, typename OutT, cudnnReduceTensorIndices_t ReduceTensorIndices>
Status ReduceKernel<allow_multi_axes>::ReduceKernelShared(
    const T* X,
    const TensorShape& input_shape,
    OutT* Y,
    const TensorShape& output_shape,
    cudnnReduceTensorOp_t cudnn_reduce_op,
    std::vector<int64_t>& output_dims) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  cudnnDataType_t cudnn_type_X = CudnnTensor::GetDataType<CudaT>();
  const auto rank = input_shape.NumDimensions();

  // Block of fast matrix row reduction.
  const auto stride = input_shape[input_shape.NumDimensions() - 1];
  const auto reduction_size = input_shape.Size() / stride;
  if (fast_reduction_ && reduction_size <= std::numeric_limits<int>::max() && stride <= std::numeric_limits<int>::max() &&
      is_matrix_row_reduction(cudnn_reduce_op,
                              static_cast<int>(reduction_size),
                              static_cast<int>(stride), rank, axes_)) {
    reduce_matrix_rows(
        reinterpret_cast<const CudaT*>(X),
        reinterpret_cast<CudaT*>(Y),
        static_cast<int>(reduction_size),
        static_cast<int>(stride));
    return Status::OK();
  }

  const auto& input_dims = input_shape.GetDims();
  int64_t input_count = input_shape.Size();
  IAllocatorUniquePtr<float> temp_X;
  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_FLATTENED_INDICES && std::is_same<T, MLFloat16>::value) {
    // ArgMax/ArgMin with FP16 are not supported by cudnn, so convert input to fp32 then call cudnn
    temp_X = GetScratchBuffer<float>(input_count);
    cudnn_type_X = CUDNN_DATA_FLOAT;
    Impl_Cast<CudaT, float>(reinterpret_cast<const CudaT*>(X), temp_X.get(), input_shape.Size());
  }

  // CUDNN requires at least 3D input, so pad 1s if needed
  std::vector<int64_t> input_dims_cudnn = input_dims;
  std::vector<int64_t> output_dims_cudnn = output_dims;
  if (rank < 3) {
    std::vector<int64_t> pads(3 - rank, 1);
    input_dims_cudnn.insert(input_dims_cudnn.end(), pads.begin(), pads.end());
    output_dims_cudnn.insert(output_dims_cudnn.end(), pads.begin(), pads.end());
  }

  CudnnReduceDescriptor reduce_desc;
  if (std::is_same<T, MLFloat16>::value)
    ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnn_reduce_op, CudnnTensor::GetDataType<float>(), ReduceTensorIndices));
  else
    ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnn_reduce_op, cudnn_type_X, ReduceTensorIndices));
  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set(input_dims_cudnn, cudnn_type_X));
  ORT_RETURN_IF_ERROR(output_tensor.Set(output_dims_cudnn, cudnn_type_X));
  size_t workspace_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(CudnnHandle(), reduce_desc, input_tensor, output_tensor, &workspace_bytes));
  auto workspace_cuda = GetScratchBuffer<CudaT>(workspace_bytes);

  size_t indices_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(CudnnHandle(), reduce_desc, input_tensor, output_tensor, &indices_bytes));
  auto indices_cuda = GetScratchBuffer<uint32_t>(indices_bytes);

  // need to allocate a separate buffer for ArgMin/ArgMax comparsion output
  auto output_count = output_shape.Size();

  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_NO_INDICES) {
    CudaT* input_data = nullptr;
    if (calculate_sqt_) {
      input_data = reinterpret_cast<CudaT*>(GetScratchBuffer<T>(input_count).get());
      fast_divmod tmp_div;
      Impl_Mul<CudaT>(static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                      reinterpret_cast<const CudaT*>(X), nullptr,
                      reinterpret_cast<const CudaT*>(X), nullptr,
                      tmp_div, tmp_div,
                      input_data, input_count);
    } else if (log_sum_exp_) {
      // Reduce max -- Max/Min will output indices data
      CudnnReduceDescriptor reduce_max_desc;
      ORT_RETURN_IF_ERROR(reduce_max_desc.Set(CUDNN_REDUCE_TENSOR_MAX, cudnn_type_X, CUDNN_REDUCE_TENSOR_NO_INDICES));
      size_t indices_bytes_max = 0;
      CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(CudnnHandle(), reduce_max_desc, input_tensor, output_tensor, &indices_bytes_max));
      auto indices_cuda_max = GetScratchBuffer<uint32_t>(indices_bytes);
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          CudnnHandle(), reduce_max_desc, indices_cuda_max.get(), indices_bytes_max, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, reinterpret_cast<const CudaT*>(X),
          &zero, output_tensor, reinterpret_cast<CudaT*>(Y)));

      // Exp(X-ReduceMax)
      const TensorShape rhs_shape(output_dims);
      auto exp_result = GetScratchBuffer<T>(input_count).get();
      auto log_sum_result = GetScratchBuffer<T>(output_count).get();
      BinaryElementwisePreparation prepare;
      prepare.BinaryElementwiseBroadcastPrepareHelper(input_shape, rhs_shape, input_shape);
      Impl_Sub<CudaT>(prepare.output_rank_or_simple_broadcast,
                      &prepare.lhs_padded_strides,
                      reinterpret_cast<const CudaT*>(X),
                      &prepare.rhs_padded_strides,
                      reinterpret_cast<CudaT*>(Y),
                      &prepare.fdm_output_strides,
                      prepare.fdm_H, prepare.fdm_C,
                      reinterpret_cast<CudaT*>(exp_result), input_count);

      Impl_Exp<CudaT>(reinterpret_cast<CudaT*>(exp_result),
                      reinterpret_cast<CudaT*>(exp_result),
                      input_count);

      // ReduceSum
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          CudnnHandle(), reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, exp_result,
          &zero, output_tensor, reinterpret_cast<CudaT*>(log_sum_result)));

      // Log(Sum)
      Impl_Log<CudaT>(reinterpret_cast<CudaT*>(log_sum_result),
                      reinterpret_cast<CudaT*>(log_sum_result),
                      output_count);

      // Log + ReduceMax
      fast_divmod tmp_div;
      Impl_Add<CudaT>(static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                      reinterpret_cast<CudaT*>(log_sum_result), nullptr,
                      reinterpret_cast<CudaT*>(Y), nullptr,
                      tmp_div, tmp_div,
                      reinterpret_cast<CudaT*>(Y), output_count);

      return Status::OK();
    }
    if (calculate_sqt_) {
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          CudnnHandle(), reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, input_data,
          &zero, output_tensor, reinterpret_cast<CudaT*>(Y)));
    } else {
      // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      if (input_count == output_count) {
        if (reinterpret_cast<const void*>(Y) != reinterpret_cast<const void*>(X)) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y, X, input_count * sizeof(T), cudaMemcpyDeviceToDevice));
        }
      } else {
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            CudnnHandle(), reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, reinterpret_cast<const CudaT*>(X),
            &zero, output_tensor, reinterpret_cast<CudaT*>(Y)));
      }
    }
  } else {  // For ArgMax & ArgMin ops, use the indicies as the output with int64 type
    if (temp_X) {
      auto temp_output = GetScratchBuffer<float>(output_count);
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          CudnnHandle(), reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, temp_X.get(),
          &zero, output_tensor, temp_output.get()));
    } else {
      auto temp_output = GetScratchBuffer<CudaT>(output_count);
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          CudnnHandle(), reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, reinterpret_cast<const CudaT*>(X),
          &zero, output_tensor, temp_output.get()));
    }

    // CUDA reduction index is uint32_t for now, cast it to int64_t according to ONNX spec
    Impl_Cast<uint32_t, int64_t>(reinterpret_cast<uint32_t*>(indices_cuda.get()), reinterpret_cast<int64_t*>(Y), output_count);
  }

  if (calculate_log_) {
    Impl_Log<CudaT>(reinterpret_cast<CudaT*>(Y),
                    reinterpret_cast<CudaT*>(Y),
                    output_count);
  }

  return Status::OK();
}

template Status ReduceKernel<true>::ReduceKernelShared<double, double, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const double* X,
    const TensorShape& input_shape,
    double* Y,
    const TensorShape& output_shape,
    cudnnReduceTensorOp_t cudnn_reduce_op,
    std::vector<int64_t>& output_dims) const;

template Status ReduceKernel<true>::ReduceKernelShared<float, float, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const float* X,
    const TensorShape& input_shape,
    float* Y,
    const TensorShape& output_shape,
    cudnnReduceTensorOp_t cudnn_reduce_op,
    std::vector<int64_t>& output_dims) const;

template Status ReduceKernel<true>::ReduceKernelShared<MLFloat16, MLFloat16, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const MLFloat16* X,
    const TensorShape& input_shape,
    MLFloat16* Y,
    const TensorShape& output_shape,
    cudnnReduceTensorOp_t cudnn_reduce_op,
    std::vector<int64_t>& output_dims) const;

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
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "cuDNN only supports up to 8-D tensors in reduction");
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

  // CUDNN requires at least 3D input, so pad 1s if needed
  prepare_reduce_metadata.input_dims_cudnn = input_dims;
  prepare_reduce_metadata.output_dims_cudnn = prepare_reduce_metadata.output_dims;
  if (rank < 3) {
    std::vector<int64_t> pads(3 - rank, 1);
    prepare_reduce_metadata.input_dims_cudnn.insert(prepare_reduce_metadata.input_dims_cudnn.end(), pads.begin(), pads.end());
    prepare_reduce_metadata.output_dims_cudnn.insert(prepare_reduce_metadata.output_dims_cudnn.end(), pads.begin(), pads.end());
  }

  prepare_reduce_metadata.output_count = TensorShape(prepare_reduce_metadata.output_dims).Size();

  if (prepare_reduce_metadata.rank == 0) {
    prepare_reduce_metadata.rank = 1;
  }

  return Status::OK();
}

// `input_shape_override` is the input shape for compute purposes (if provided)
template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
Status ReduceComputeCore(CUDAExecutionProvider& cuda_ep, const Tensor& input, PrepareReduceMetadata& prepare_reduce_metadata,
                         /*out*/ Tensor& output, cudnnReduceTensorOp_t cudnn_reduce_op,
                         const std::vector<int64_t>& axes,
                         bool calculate_log, bool calculate_sqt, bool log_sum_exp, bool fast_reduction,
                         const TensorShape* input_shape_override) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const TensorShape& input_shape = input_shape_override ? *input_shape_override : input.Shape();

  int64_t input_count = prepare_reduce_metadata.input_count;
  int64_t output_count = prepare_reduce_metadata.output_count;
  std::vector<int64_t>& output_dims = prepare_reduce_metadata.output_dims;
  std::vector<int64_t>& input_dims_cudnn = prepare_reduce_metadata.input_dims_cudnn;
  std::vector<int64_t>& output_dims_cudnn = prepare_reduce_metadata.output_dims_cudnn;
  int64_t rank = prepare_reduce_metadata.rank;
  int64_t stride = prepare_reduce_metadata.stride;

  // special case when there is a dim value of 0 in the shape.
  if (input_count == 0) {
    assert(output.Shape().Size() == 0);
    return Status::OK();
  }

  // This reduction keep adding values to this buffer. If a non-zero value, say 1000, is here, the sum will start with 1000.
  // Therefore zeroing out the memory is required
  CUDA_RETURN_IF_ERROR(cudaMemset(output.MutableDataRaw(), 0, output.SizeInBytes()));

  IAllocatorUniquePtr<float> temp_X;
  cudnnDataType_t cudnn_type_X = CudnnTensor::GetDataType<CudaT>();

  // Block of fast matrix row reduction.
  // It relies on new atomicAdd for half type, so old CUDA can't use it.
  const auto reduction_size = input_count / stride;
  if (!std::is_same<T, int8_t>::value && !std::is_same<T, uint8_t>::value) {
    if (fast_reduction && reduction_size <= std::numeric_limits<int>::max() && stride <= std::numeric_limits<int>::max() &&
        prepare_reduce_metadata.contiguous_axes &&
        is_matrix_row_reduction(cudnn_reduce_op, static_cast<int>(reduction_size), static_cast<int>(stride), rank, axes)) {
      reduce_matrix_rows(
          reinterpret_cast<const CudaT*>(input.template Data<T>()),
          reinterpret_cast<CudaT*>(output.template MutableData<T>()),
          static_cast<int>(reduction_size),
          static_cast<int>(stride));
      return Status::OK();
    }
  }

  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_FLATTENED_INDICES && std::is_same<T, MLFloat16>::value) {
    // ArgMax/ArgMin with FP16 are not supported by cudnn, so convert input to fp32 then call cudnn
    temp_X = cuda_ep.GetScratchBuffer<float>(input_count);
    cudnn_type_X = CUDNN_DATA_FLOAT;
    Impl_Cast<CudaT, float>(reinterpret_cast<const CudaT*>(input.template Data<T>()), temp_X.get(), input_shape.Size());
  }

  CudnnReduceDescriptor reduce_desc;
  if (std::is_same<T, MLFloat16>::value) {
    ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnn_reduce_op, CudnnTensor::GetDataType<float>(), ReduceTensorIndices));
  } else {
    ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnn_reduce_op, cudnn_type_X, ReduceTensorIndices));
  }

  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set(input_dims_cudnn, cudnn_type_X));
  ORT_RETURN_IF_ERROR(output_tensor.Set(output_dims_cudnn, cudnn_type_X));
  size_t workspace_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(cuda_ep.PerThreadCudnnHandle(), reduce_desc,
                                                       input_tensor, output_tensor, &workspace_bytes));
  auto workspace_cuda = cuda_ep.GetScratchBuffer<CudaT>(workspace_bytes);

  size_t indices_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(cuda_ep.PerThreadCudnnHandle(), reduce_desc,
                                                     input_tensor, output_tensor, &indices_bytes));
  auto indices_cuda = cuda_ep.GetScratchBuffer<uint32_t>(indices_bytes);

  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_NO_INDICES) {
    IAllocatorUniquePtr<T> input_data_buffer(nullptr, [](T*) {});
    CudaT* input_data = nullptr;
    if (calculate_sqt) {
      input_data_buffer = cuda_ep.GetScratchBuffer<T>(input_count);
      input_data = reinterpret_cast<CudaT*>(input_data_buffer.get());
      fast_divmod tmp_div;
      Impl_Mul<CudaT>(static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                      reinterpret_cast<const CudaT*>(input.template Data<T>()), nullptr,
                      reinterpret_cast<const CudaT*>(input.template Data<T>()), nullptr,
                      tmp_div, tmp_div,
                      input_data, input_count);
    } else if (log_sum_exp) {
      // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      // This happens when the input is Scalar
      if (input_count == output_count) {
        if (output.template MutableData<T>() != input.template Data<T>()) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output.template MutableData<T>(), input.template Data<T>(), input_count * sizeof(T), cudaMemcpyDeviceToDevice));
        }
      } else {
        // Reduce max -- Max/Min will output indices data
        CudnnReduceDescriptor reduce_max_desc;
        cudnnDataType_t cudnn_reduce_max_type = cudnn_type_X;
        if ((std::is_same<T, MLFloat16>::value)) {
          cudnn_reduce_max_type = CUDNN_DATA_FLOAT;
        }
        ORT_RETURN_IF_ERROR(reduce_max_desc.Set(CUDNN_REDUCE_TENSOR_MAX, cudnn_reduce_max_type, CUDNN_REDUCE_TENSOR_NO_INDICES));
        size_t indices_bytes_max = 0;
        CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(cuda_ep.PerThreadCudnnHandle(), reduce_max_desc,
                                                           input_tensor, output_tensor, &indices_bytes_max));
        auto indices_cuda_max = cuda_ep.GetScratchBuffer<uint32_t>(indices_bytes);
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            cuda_ep.PerThreadCudnnHandle(), reduce_max_desc, indices_cuda_max.get(), indices_bytes_max,
            workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, reinterpret_cast<const CudaT*>(input.template Data<T>()),
            &zero, output_tensor, reinterpret_cast<CudaT*>(output.template MutableData<T>())));
      }

      // Exp(X-ReduceMax)
      const TensorShape output_shape(output_dims);
      auto exp_result_buffer = cuda_ep.GetScratchBuffer<T>(input_count);
      auto exp_result = exp_result_buffer.get();
      auto log_sum_result_buffer = cuda_ep.GetScratchBuffer<T>(output_count);
      auto log_sum_result = log_sum_result_buffer.get();
      BinaryElementwisePreparation prepare;
      prepare.BinaryElementwiseBroadcastPrepareHelper(input_shape, output_shape, input_shape);
      Impl_Sub<CudaT>(prepare.output_rank_or_simple_broadcast,
                      &prepare.lhs_padded_strides,
                      reinterpret_cast<const CudaT*>(input.template Data<T>()),
                      &prepare.rhs_padded_strides,
                      reinterpret_cast<CudaT*>(output.template MutableData<T>()),
                      &prepare.fdm_output_strides,
                      prepare.fdm_H, prepare.fdm_C,
                      reinterpret_cast<CudaT*>(exp_result), input_count);

      Impl_Exp<CudaT>(reinterpret_cast<CudaT*>(exp_result),
                      reinterpret_cast<CudaT*>(exp_result),
                      input_count);

      // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      // This happens when the input is Scalar. We do not need to add anything in this case.
      if (input_count == output_count) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(reinterpret_cast<CudaT*>(log_sum_result), exp_result, input_count * sizeof(T), cudaMemcpyDeviceToDevice));
      } else {
        // ReduceSum
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            cuda_ep.PerThreadCudnnHandle(), reduce_desc, indices_cuda.get(), indices_bytes,
            workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, exp_result,
            &zero, output_tensor, reinterpret_cast<CudaT*>(log_sum_result)));
      }

      // Log(Sum)
      Impl_Log<CudaT>(reinterpret_cast<CudaT*>(log_sum_result),
                      reinterpret_cast<CudaT*>(log_sum_result),
                      output_count);

      // Log + ReduceMax
      fast_divmod tmp_div;
      Impl_Add<CudaT>(static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                      reinterpret_cast<CudaT*>(log_sum_result), nullptr,
                      reinterpret_cast<CudaT*>(output.template MutableData<T>()), nullptr,
                      tmp_div, tmp_div,
                      reinterpret_cast<CudaT*>(output.template MutableData<T>()), output_count);

      return Status::OK();
    }
    if (calculate_sqt) {
      // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      // This happens when the input is Scalar. We do not need to add anything in this case.
      if (input_count == output_count) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(reinterpret_cast<CudaT*>(output.template MutableData<T>()), input_data, input_count * sizeof(T), cudaMemcpyDeviceToDevice));
      } else {
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            cuda_ep.PerThreadCudnnHandle(), reduce_desc, indices_cuda.get(), indices_bytes,
            workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, input_data,
            &zero, output_tensor, reinterpret_cast<CudaT*>(output.template MutableData<T>())));
      }
    } else {
      // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      if (input_count == output_count) {
        if (output.template MutableData<T>() != input.template Data<T>()) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output.template MutableData<T>(), input.template Data<T>(), input_count * sizeof(T), cudaMemcpyDeviceToDevice));
        }
      } else {
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            cuda_ep.PerThreadCudnnHandle(), reduce_desc, indices_cuda.get(), indices_bytes,
            workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, reinterpret_cast<const CudaT*>(input.template Data<T>()),
            &zero, output_tensor, reinterpret_cast<CudaT*>(output.template MutableData<T>())));
      }
    }
  } else {  // For ArgMax & ArgMin ops, use the indicies as the output with int64 type
    if (temp_X) {
      auto temp_output = cuda_ep.GetScratchBuffer<float>(output_count);
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          cuda_ep.PerThreadCudnnHandle(), reduce_desc, indices_cuda.get(), indices_bytes,
          workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, temp_X.get(),
          &zero, output_tensor, temp_output.get()));
    } else {
      auto temp_output = cuda_ep.GetScratchBuffer<CudaT>(output_count);
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          cuda_ep.PerThreadCudnnHandle(), reduce_desc, indices_cuda.get(), indices_bytes,
          workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, reinterpret_cast<const CudaT*>(input.template Data<T>()),
          &zero, output_tensor, temp_output.get()));
    }

    // CUDA reduction index is uint32_t for now, cast it to int64_t according to ONNX spec
    Impl_Cast<uint32_t, int64_t>(reinterpret_cast<uint32_t*>(indices_cuda.get()), output.template MutableData<int64_t>(), output_count);
  }

  if (calculate_log) {
    Impl_Log<CudaT>(reinterpret_cast<CudaT*>(output.template MutableData<T>()),
                    reinterpret_cast<CudaT*>(output.template MutableData<T>()),
                    output_count);
  }

  return Status::OK();
}

template <bool allow_multi_axes>
template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
Status ReduceKernel<allow_multi_axes>::ComputeImpl(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnn_reduce_op) const {
  const Tensor* X = ctx->Input<Tensor>(0);

  PrepareReduceMetadata prepare_reduce_metadata;
  ORT_RETURN_IF_ERROR(PrepareForReduce(X,
                                       keepdims_,
                                       axes_,
                                       prepare_reduce_metadata));
  Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);
  bool fast_reduction = fast_reduction_;
  if (fast_reduction) {
    auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
    if (ctx_internal && ctx_internal->GetUseDeterministicCompute())
      fast_reduction = false;
  }

  return ReduceComputeCore<T, ReduceTensorIndices>(*cuda_ep_, *X, prepare_reduce_metadata, *Y, cudnn_reduce_op, axes_,
                                                   calculate_log_, calculate_sqt_, log_sum_exp_, fast_reduction);
}

template <>
template <>
Status ReduceKernel<true>::ComputeImpl<int32_t, CUDNN_REDUCE_TENSOR_NO_INDICES>(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnn_reduce_op) const {
  typedef typename ToCudaType<int32_t>::MappedType CudaT;

  const Tensor* X = ctx->Input<Tensor>(0);

  PrepareReduceMetadata prepare_reduce_metadata;

  ORT_RETURN_IF_ERROR(PrepareForReduce(X,
                                       keepdims_,
                                       axes_,
                                       prepare_reduce_metadata));

  Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);

  int64_t input_count = prepare_reduce_metadata.input_count;
  int64_t output_count = prepare_reduce_metadata.output_count;
  std::vector<int64_t>& input_dims_cudnn = prepare_reduce_metadata.input_dims_cudnn;
  std::vector<int64_t>& output_dims_cudnn = prepare_reduce_metadata.output_dims_cudnn;

  // special case when there is a dim value of 0 in the shape.
  if (input_count == 0) {
    assert(Y->Shape().Size() == 0);
    return Status::OK();
  }

  // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
  if (input_count == output_count) {
    if (Y->template MutableData<int32_t>() != X->template Data<int32_t>()) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y->template MutableData<int32_t>(), X->template Data<int32_t>(), input_count * sizeof(int32_t), cudaMemcpyDeviceToDevice));
    }
    return Status::OK();
  }

  // This reduction keep adding values to this buffer. If a non-zero value, say 1000, is here, the sum will start with 1000.
  // Therefore zeroing out the memory is required
  CUDA_RETURN_IF_ERROR(cudaMemset(Y->MutableDataRaw(), 0, Y->SizeInBytes()));

  size_t indices_bytes = 0;
  size_t workspace_bytes = 0;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  CudnnReduceDescriptor reduce_desc;

  cudnnDataType_t cudnn_type_X = CUDNN_DATA_FLOAT;
  IAllocatorUniquePtr<float> temp_X = GetScratchBuffer<float>(input_count);
  Impl_Cast<CudaT, float>(reinterpret_cast<const CudaT*>(X->template Data<int32_t>()), temp_X.get(), X->Shape().Size());

  ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnn_reduce_op, cudnn_type_X, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES));
  ORT_RETURN_IF_ERROR(input_tensor.Set(input_dims_cudnn, cudnn_type_X));
  ORT_RETURN_IF_ERROR(output_tensor.Set(output_dims_cudnn, cudnn_type_X));
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(CudnnHandle(), reduce_desc, input_tensor, output_tensor, &indices_bytes));
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(CudnnHandle(), reduce_desc, input_tensor, output_tensor, &workspace_bytes));
  IAllocatorUniquePtr<uint32_t> indices_cuda = GetScratchBuffer<uint32_t>(indices_bytes);
  IAllocatorUniquePtr<CudaT> workspace_cuda = GetScratchBuffer<CudaT>(workspace_bytes);

  const auto one = Consts<float>::One;
  const auto zero = Consts<float>::Zero;
  auto temp_Y = GetScratchBuffer<float>(output_count);
  CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(CudnnHandle(),
                                          reduce_desc,
                                          indices_cuda.get(),
                                          indices_bytes,
                                          workspace_cuda.get(),
                                          workspace_bytes,
                                          &one,
                                          input_tensor,
                                          temp_X.get(),
                                          &zero,
                                          output_tensor,
                                          temp_Y.get()));

  Impl_Cast<float, int32_t>(temp_Y.get(), Y->template MutableData<int32_t>(), output_count);

  return Status::OK();
}

template <>
template <>
Status ReduceKernel<true>::ComputeImpl<int8_t, CUDNN_REDUCE_TENSOR_NO_INDICES>(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnn_reduce_op) const {
  typedef typename ToCudaType<int8_t>::MappedType CudaT;

  const Tensor* X = ctx->Input<Tensor>(0);
  PrepareReduceMetadata prepare_reduce_metadata;

  ORT_RETURN_IF_ERROR(PrepareForReduce(X,
                                       keepdims_,
                                       axes_,
                                       prepare_reduce_metadata));

  Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);

  int64_t input_count = prepare_reduce_metadata.input_count;
  int64_t output_count = prepare_reduce_metadata.output_count;
  std::vector<int64_t>& input_dims_cudnn = prepare_reduce_metadata.input_dims_cudnn;
  std::vector<int64_t>& output_dims_cudnn = prepare_reduce_metadata.output_dims_cudnn;

  // special case when there is a dim value of 0 in the shape.
  if (input_count == 0) {
    assert(Y->Shape().Size() == 0);
    return Status::OK();
  }

  // cudnnReduceTensor has issue if input and output has same size, we just need to copy the data for this case
  auto* const dst = Y->template MutableData<int8_t>();
  const auto* const src = X->template Data<int8_t>();
  if (input_count == output_count) {
    if (src != dst) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst, src, input_count * sizeof(int8_t), cudaMemcpyDeviceToDevice));
    }
    return Status::OK();
  }

  // This reduction keep adding values to this buffer. If a non-zero value, say 1000, is here, the sum will start with 1000.
  // Therefore zeroing out the memory is required
  CUDA_RETURN_IF_ERROR(cudaMemset(Y->MutableDataRaw(), 0, Y->SizeInBytes()));

  size_t indices_bytes = 0;
  size_t workspace_bytes = 0;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  CudnnReduceDescriptor reduce_desc;

  cudnnDataType_t cudnn_type_X = CUDNN_DATA_FLOAT;
  IAllocatorUniquePtr<float> temp_X = GetScratchBuffer<float>(input_count);
  Impl_Cast<CudaT, float>(reinterpret_cast<const CudaT*>(src), temp_X.get(), X->Shape().Size());

  ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnn_reduce_op, cudnn_type_X, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES));
  ORT_RETURN_IF_ERROR(input_tensor.Set(input_dims_cudnn, cudnn_type_X));
  ORT_RETURN_IF_ERROR(output_tensor.Set(output_dims_cudnn, cudnn_type_X));
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(CudnnHandle(), reduce_desc, input_tensor, output_tensor, &indices_bytes));
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(CudnnHandle(), reduce_desc, input_tensor, output_tensor, &workspace_bytes));
  IAllocatorUniquePtr<uint32_t> indices_cuda = GetScratchBuffer<uint32_t>(indices_bytes);
  IAllocatorUniquePtr<CudaT> workspace_cuda = GetScratchBuffer<CudaT>(workspace_bytes);

  const auto one = Consts<float>::One;
  const auto zero = Consts<float>::Zero;
  auto temp_Y = GetScratchBuffer<float>(output_count);
  CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(CudnnHandle(),
                                          reduce_desc,
                                          indices_cuda.get(),
                                          indices_bytes,
                                          workspace_cuda.get(),
                                          workspace_bytes,
                                          &one,
                                          input_tensor,
                                          temp_X.get(),
                                          &zero,
                                          output_tensor,
                                          temp_Y.get()));

  Impl_Cast<float, int8_t>(temp_Y.get(), dst, output_count);

  return Status::OK();
}

template <>
template <>
Status ReduceKernel<true>::ComputeImpl<uint8_t, CUDNN_REDUCE_TENSOR_NO_INDICES>(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnn_reduce_op) const {
  typedef typename ToCudaType<uint8_t>::MappedType CudaT;

  const Tensor* X = ctx->Input<Tensor>(0);
  PrepareReduceMetadata prepare_reduce_metadata;

  ORT_RETURN_IF_ERROR(PrepareForReduce(X,
                                       keepdims_,
                                       axes_,
                                       prepare_reduce_metadata));

  Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);

  int64_t input_count = prepare_reduce_metadata.input_count;
  int64_t output_count = prepare_reduce_metadata.output_count;
  std::vector<int64_t>& input_dims_cudnn = prepare_reduce_metadata.input_dims_cudnn;
  std::vector<int64_t>& output_dims_cudnn = prepare_reduce_metadata.output_dims_cudnn;

  // special case when there is a dim value of 0 in the shape.
  if (input_count == 0) {
    assert(Y->Shape().Size() == 0);
    return Status::OK();
  }

  // cudnnReduceTensor has issue if input and output has same size, we just need to copy the data for this case
  auto* const dst = Y->template MutableData<uint8_t>();
  const auto* const src = X->template Data<uint8_t>();
  if (input_count == output_count) {
    if (src != dst) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst, src, input_count * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
    }
    return Status::OK();
  }

  // This reduction keep adding values to this buffer. If a non-zero value, say 1000, is here, the sum will start with 1000.
  // Therefore zeroing out the memory is required
  CUDA_RETURN_IF_ERROR(cudaMemset(Y->MutableDataRaw(), 0, Y->SizeInBytes()));

  size_t indices_bytes = 0;
  size_t workspace_bytes = 0;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  CudnnReduceDescriptor reduce_desc;

  cudnnDataType_t cudnn_type_X = CUDNN_DATA_FLOAT;
  IAllocatorUniquePtr<float> temp_X = GetScratchBuffer<float>(input_count);
  Impl_Cast<CudaT, float>(reinterpret_cast<const CudaT*>(src), temp_X.get(), X->Shape().Size());

  ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnn_reduce_op, cudnn_type_X, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES));
  ORT_RETURN_IF_ERROR(input_tensor.Set(input_dims_cudnn, cudnn_type_X));
  ORT_RETURN_IF_ERROR(output_tensor.Set(output_dims_cudnn, cudnn_type_X));
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(CudnnHandle(), reduce_desc, input_tensor, output_tensor, &indices_bytes));
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(CudnnHandle(), reduce_desc, input_tensor, output_tensor, &workspace_bytes));
  IAllocatorUniquePtr<uint32_t> indices_cuda = GetScratchBuffer<uint32_t>(indices_bytes);
  IAllocatorUniquePtr<CudaT> workspace_cuda = GetScratchBuffer<CudaT>(workspace_bytes);

  const auto one = Consts<float>::One;
  const auto zero = Consts<float>::Zero;
  auto temp_Y = GetScratchBuffer<float>(output_count);
  CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(CudnnHandle(),
                                          reduce_desc,
                                          indices_cuda.get(),
                                          indices_bytes,
                                          workspace_cuda.get(),
                                          workspace_bytes,
                                          &one,
                                          input_tensor,
                                          temp_X.get(),
                                          &zero,
                                          output_tensor,
                                          temp_Y.get()));

  Impl_Cast<float, uint8_t>(temp_Y.get(), dst, output_count);

  return Status::OK();
}

namespace ReductionOps {

template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
Tensor ReduceCompute(CUDAExecutionProvider& cuda_ep, cudnnReduceTensorOp_t cudnn_reduce_op, AllocatorPtr allocator,
                     const Tensor& input, const std::vector<int64_t>& axes,
                     bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
                     bool fast_reduction, const TensorShape* input_shape_override) {
  PrepareReduceMetadata prepare_reduce_metadata;
  auto status = PrepareForReduce(&input,
                                 keep_dims,
                                 axes,
                                 prepare_reduce_metadata,
                                 input_shape_override);

  if (!status.IsOK()) {
    ORT_THROW(ONNXRUNTIME, FAIL, "Failed to perform reduce op: ", status.ErrorMessage());
  }

  Tensor output(input.DataType(), prepare_reduce_metadata.squeezed_output_dims, allocator);

  status = ReduceComputeCore<T, ReduceTensorIndices>(cuda_ep, input, prepare_reduce_metadata, output, cudnn_reduce_op, axes,
                                                     calculate_log, calculate_sqt, log_sum_exp, fast_reduction, input_shape_override);

  if (!status.IsOK()) {
    ORT_THROW(ONNXRUNTIME, FAIL, "Failed to perform reduce op: ", status.ErrorMessage());
  }

  return output;
}

// Explicit template instantiation (needed to be used in einsum_auxiliary_ops.cc)

template Tensor ReduceCompute<float, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    CUDAExecutionProvider& cuda_ep, cudnnReduceTensorOp_t cudnn_reduce_op,
    AllocatorPtr allocator,
    const Tensor& input, const std::vector<int64_t>& axes,
    bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
    bool fast_reduction, const TensorShape* input_shape_override);

template Tensor ReduceCompute<double, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    CUDAExecutionProvider& cuda_ep, cudnnReduceTensorOp_t cudnn_reduce_op,
    AllocatorPtr allocator,
    const Tensor& input, const std::vector<int64_t>& axes,
    bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
    bool fast_reduction, const TensorShape* input_shape_override);

}  // namespace ReductionOps

#define REGISTER_KERNEL_HFD(name)        \
  REGISTER_KERNEL_TYPED(name, MLFloat16) \
  REGISTER_KERNEL_TYPED(name, float)     \
  REGISTER_KERNEL_TYPED(name, double)

REGISTER_KERNEL_HFD(ArgMax)
REGISTER_KERNEL_HFD(ArgMin)
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

}  // namespace cuda
}  // namespace onnxruntime
