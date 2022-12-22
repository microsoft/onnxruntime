// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/reduction/reduction_ops.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#ifdef ENABLE_TRAINING
#include "contrib_ops/cpu/aten_ops/aten_op.h"
#endif

using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

// opset 11 explicitly added support for negative axis. implementation already allowed it.
#define REGISTER_KERNEL_TYPED(name, T)                                                     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      1, 10,                                                                               \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                            \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      11, 12,                                                                              \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      13,                                                                                  \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

#define REGISTER_KERNEL_VERSIONED_TYPED_12(name, T)                                        \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      1, 10,                                                                               \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                            \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      11, 11,                                                                              \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                            \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      12, 12,                                                                              \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

// Register those with changes in OpSet12.
#define REGISTER_KERNEL_TYPED_13_WITH_VERSIONED_12(name, T)                                \
  REGISTER_KERNEL_VERSIONED_TYPED_12(name, T)                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      13,                                                                                  \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

#define REGISTER_KERNEL_VERSIONED_TYPED_13(name, T)                                        \
  REGISTER_KERNEL_VERSIONED_TYPED_12(name, T)                                              \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      13, 13,                                                                              \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

// Register ReduceMin int64_t support in OpSet14.
#define REGISTER_KERNEL_TYPED_14(name, T)                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      14,                                                                                  \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

// CUDA ArgMax/ArgMin doesn't have OpSet12+ implementation (with select_last_index attr) yet
#define REGISTER_KERNEL_VERSIONED_TYPED_11(name, T)                                        \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      1, 10,                                                                               \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                            \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      11, 11,                                                                              \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

// Register with the latest version 13
#define REGISTER_KERNEL_TYPED_13(name, T)                                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      1, 10,                                                                               \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                            \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      11, 12,                                                                              \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);                                                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      13,                                                                                  \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create())                                                        \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),                          \
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
    cudnnHandle_t cudnn_handle,
    onnxruntime::Stream* stream,
    TensorShapeVector& output_dims) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<OutT>::MappedType CudaOutT;
  cudnnDataType_t cudnn_type_X = CudnnTensor::GetDataType<CudaT>();
  const auto rank = input_shape.NumDimensions();

  auto cuda_stream = stream ? static_cast<cudaStream_t>(stream->GetHandle()) : nullptr;
  // Block of fast matrix reduction.
  if (fast_reduction_) {
    int m{}, n{};
    const auto applicable_matrix_reduction = get_applicable_matrix_reduction(
        cudnn_reduce_op, input_shape.GetDims(), axes_, m, n);
    switch (applicable_matrix_reduction) {
      case ApplicableMatrixReduction::Rows: {
        return reduce_matrix_rows(
            cuda_stream,
            reinterpret_cast<const CudaT*>(X),
            reinterpret_cast<CudaOutT*>(Y),
            m, n, false);
      }
      case ApplicableMatrixReduction::Columns:
      // don't call reduce_matrix_columns() since it will reset initial output data
      default:
        break;
    }
  }

  int64_t input_count = input_shape.Size();
  IAllocatorUniquePtr<float> temp_X;
  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_FLATTENED_INDICES && std::is_same<T, MLFloat16>::value) {
    // ArgMax/ArgMin with FP16 are not supported by cudnn, so convert input to fp32 then call cudnn
    temp_X = GetScratchBuffer<float>(input_count, stream);
    cudnn_type_X = CUDNN_DATA_FLOAT;
    Impl_Cast<CudaT, float>(cuda_stream, reinterpret_cast<const CudaT*>(X), temp_X.get(), input_shape.Size());
  }

  // CUDNN requires at least 3D input, so pad 1s if needed
  auto input_dims_cudnn = input_shape.AsShapeVector();
  auto output_dims_cudnn = output_dims;
  if (rank < 3) {
    TensorShapeVector pads(3 - rank, 1);
    input_dims_cudnn.insert(input_dims_cudnn.end(), pads.begin(), pads.end());
    output_dims_cudnn.insert(output_dims_cudnn.end(), pads.begin(), pads.end());
  }

  CudnnReduceDescriptor reduce_desc;
  if constexpr (std::is_same<T, MLFloat16>::value)
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
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(cudnn_handle, reduce_desc, input_tensor, output_tensor, &workspace_bytes));
  auto workspace_cuda = GetScratchBuffer<CudaT>(workspace_bytes, stream);

  size_t indices_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(cudnn_handle, reduce_desc, input_tensor, output_tensor, &indices_bytes));
  auto indices_cuda = GetScratchBuffer<uint32_t>(indices_bytes, stream);

  // need to allocate a separate buffer for ArgMin/ArgMax comparsion output
  auto output_count = output_shape.Size();

  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_NO_INDICES) {
    IAllocatorUniquePtr<T> input_data_buffer(nullptr, [](T*) {});
    CudaT* input_data = nullptr;
    if (calculate_sqt_) {
      input_data_buffer = GetScratchBuffer<T>(input_count, stream);
      input_data = reinterpret_cast<CudaT*>(input_data_buffer.get());
      fast_divmod tmp_div;
      Impl_Mul<CudaT>(cuda_stream, static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                      reinterpret_cast<const CudaT*>(X), nullptr,
                      reinterpret_cast<const CudaT*>(X), nullptr,
                      tmp_div, tmp_div,
                      input_data, input_count);
    } else if (log_sum_exp_) {
      // Reduce max -- Max/Min will output indices data
      CudnnReduceDescriptor reduce_max_desc;
      ORT_RETURN_IF_ERROR(reduce_max_desc.Set(CUDNN_REDUCE_TENSOR_MAX, cudnn_type_X, CUDNN_REDUCE_TENSOR_NO_INDICES));
      size_t indices_bytes_max = 0;
      CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(cudnn_handle, reduce_max_desc, input_tensor, output_tensor, &indices_bytes_max));
      auto indices_cuda_max = GetScratchBuffer<uint32_t>(indices_bytes, stream);
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          cudnn_handle, reduce_max_desc, indices_cuda_max.get(), indices_bytes_max, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, reinterpret_cast<const CudaT*>(X),
          &zero, output_tensor, reinterpret_cast<CudaT*>(Y)));

      // Exp(X-ReduceMax)
      const TensorShape rhs_shape(output_dims);
      auto exp_result_buffer = GetScratchBuffer<T>(input_count, stream);
      auto exp_result = exp_result_buffer.get();
      auto log_sum_result_buffer = GetScratchBuffer<T>(output_count, stream);
      auto log_sum_result = log_sum_result_buffer.get();
      BinaryElementwisePreparation prepare;
      ORT_RETURN_IF_ERROR(prepare.BinaryElementwiseBroadcastPrepareHelper(input_shape, rhs_shape, input_shape));
      Impl_Sub<CudaT>(cuda_stream,
                      prepare.output_rank_or_simple_broadcast,
                      &prepare.lhs_padded_strides,
                      reinterpret_cast<const CudaT*>(X),
                      &prepare.rhs_padded_strides,
                      reinterpret_cast<CudaT*>(Y),
                      &prepare.fdm_output_strides,
                      prepare.fdm_H, prepare.fdm_C,
                      reinterpret_cast<CudaT*>(exp_result), input_count);

      Impl_Exp<CudaT>(cuda_stream, reinterpret_cast<CudaT*>(exp_result),
                      reinterpret_cast<CudaT*>(exp_result),
                      input_count);

      // ReduceSum
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, exp_result,
          &zero, output_tensor, reinterpret_cast<CudaT*>(log_sum_result)));

      // Log(Sum)
      Impl_Log<CudaT>(cuda_stream, reinterpret_cast<CudaT*>(log_sum_result),
                      reinterpret_cast<CudaT*>(log_sum_result),
                      output_count);

      // Log + ReduceMax
      fast_divmod tmp_div;
      Impl_Add<CudaT>(cuda_stream, static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                      reinterpret_cast<CudaT*>(log_sum_result), nullptr,
                      reinterpret_cast<CudaT*>(Y), nullptr,
                      tmp_div, tmp_div,
                      reinterpret_cast<CudaT*>(Y), output_count);

      return Status::OK();
    }
    if (calculate_sqt_) {
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, input_data,
          &zero, output_tensor, reinterpret_cast<CudaT*>(Y)));
    } else {
      // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      if (input_count == output_count) {
        if (reinterpret_cast<const void*>(Y) != reinterpret_cast<const void*>(X)) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y, X, input_count * sizeof(T), cudaMemcpyDeviceToDevice, cuda_stream));
        }
      } else {
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, reinterpret_cast<const CudaT*>(X),
            &zero, output_tensor, reinterpret_cast<CudaT*>(Y)));
      }
    }
  } else {  // For ArgMax & ArgMin ops, use the indicies as the output with int64 type
    if (temp_X) {
      auto temp_output = GetScratchBuffer<float>(output_count, stream);
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, temp_X.get(),
          &zero, output_tensor, temp_output.get()));
    } else {
      auto temp_output = GetScratchBuffer<CudaT>(output_count, stream);
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, reinterpret_cast<const CudaT*>(X),
          &zero, output_tensor, temp_output.get()));
    }

    // CUDA reduction index is uint32_t for now, cast it to int64_t according to ONNX spec
    Impl_Cast<uint32_t, int64_t>(cuda_stream, reinterpret_cast<uint32_t*>(indices_cuda.get()), reinterpret_cast<int64_t*>(Y), output_count);
  }

  if (calculate_log_) {
    Impl_Log<CudaT>(cuda_stream, reinterpret_cast<CudaT*>(Y),
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
    cudnnHandle_t cudnn_handle,
    onnxruntime::Stream* stream,
    TensorShapeVector& output_dims) const;

template Status ReduceKernel<true>::ReduceKernelShared<float, float, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const float* X,
    const TensorShape& input_shape,
    float* Y,
    const TensorShape& output_shape,
    cudnnReduceTensorOp_t cudnn_reduce_op,
    cudnnHandle_t cudnn_handle,
    onnxruntime::Stream* stream,
    TensorShapeVector& output_dims) const;

template Status ReduceKernel<true>::ReduceKernelShared<MLFloat16, MLFloat16, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const MLFloat16* X,
    const TensorShape& input_shape,
    MLFloat16* Y,
    const TensorShape& output_shape,
    cudnnReduceTensorOp_t cudnn_reduce_op,
    cudnnHandle_t cudnn_handle,
    onnxruntime::Stream* stream,
    TensorShapeVector& output_dims) const;

// `input_shape_override` (if provided) is the input shape for compute purposes
Status PrepareForReduce(const Tensor* X,
                        bool keepdims,
                        gsl::span<const int64_t> axes,
                        PrepareReduceMetadata& prepare_reduce_metadata,
                        const TensorShape* input_shape_override) {
  ORT_ENFORCE(nullptr != X);

  const TensorShape& input_shape = input_shape_override ? *input_shape_override : X->Shape();
  const int64_t rank = gsl::narrow<int64_t>(input_shape.NumDimensions());
  prepare_reduce_metadata.input_count = input_shape.Size();

  if (rank > 8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "cuDNN only supports up to 8-D tensors in reduction");
  }

  const auto input_dims = input_shape.GetDims();
  std::vector<bool> reduced(rank, false);
  if (axes.size() > 0) {
    prepare_reduce_metadata.output_dims = input_shape.AsShapeVector();
    for (auto axis : axes) {
      axis = HandleNegativeAxis(axis, rank);
      ORT_ENFORCE(input_dims[axis] != 0,
                  "Can't reduce on dim with value of 0 if 'keepdims' is false. "
                  "Invalid output shape would be produced. input_shape:",
                  input_shape);
      prepare_reduce_metadata.output_dims[axis] = 1;
      reduced[axis] = true;
    }
  } else {
    // no axes provided (i.e.) default axes  => reduce on all dims
    prepare_reduce_metadata.output_dims.reserve(input_dims.size());
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
  prepare_reduce_metadata.input_dims_cudnn = input_shape.AsShapeVector();
  prepare_reduce_metadata.output_dims_cudnn = prepare_reduce_metadata.output_dims;
  if (rank < 3) {
    TensorShapeVector pads(3 - rank, 1);
    prepare_reduce_metadata.input_dims_cudnn.insert(prepare_reduce_metadata.input_dims_cudnn.end(), pads.begin(), pads.end());
    prepare_reduce_metadata.output_dims_cudnn.insert(prepare_reduce_metadata.output_dims_cudnn.end(), pads.begin(), pads.end());
  }

  prepare_reduce_metadata.output_count = TensorShape(prepare_reduce_metadata.output_dims).Size();

  return Status::OK();
}

// `input_shape_override` is the input shape for compute purposes (if provided)
template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
Status ReduceComputeCore(const CUDAExecutionProvider& cuda_ep, const Tensor& input, PrepareReduceMetadata& prepare_reduce_metadata,
                         /*out*/ Tensor& output, cudnnReduceTensorOp_t cudnn_reduce_op,
                         gsl::span<const int64_t> axes,
                         bool calculate_log, bool calculate_sqt, bool log_sum_exp, bool fast_reduction,
                         Stream* ort_stream,
                         const TensorShape* input_shape_override) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const TensorShape& input_shape = input_shape_override ? *input_shape_override : input.Shape();
  cudaStream_t stream = ort_stream ? static_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;

  int64_t input_count = prepare_reduce_metadata.input_count;
  int64_t output_count = prepare_reduce_metadata.output_count;
  auto& output_dims = prepare_reduce_metadata.output_dims;
  auto& input_dims_cudnn = prepare_reduce_metadata.input_dims_cudnn;
  auto& output_dims_cudnn = prepare_reduce_metadata.output_dims_cudnn;
  // special case when there is a dim value of 0 in the shape.
  if (input_count == 0) {
    assert(output.Shape().Size() == 0);
    return Status::OK();
  }

  // Block of fast matrix reduction.
  if (fast_reduction) {
    int m{}, n{};
    const auto applicable_matrix_reduction =
        get_applicable_matrix_reduction(cudnn_reduce_op, input_shape.GetDims(), axes, m, n);
    if (applicable_matrix_reduction != ApplicableMatrixReduction::None) {
      IAllocatorUniquePtr<T> input_data_buffer(nullptr, [](T*) {});
      const CudaT* input_data = reinterpret_cast<const CudaT*>(input.Data<T>());
      if (calculate_sqt) {
        input_data_buffer = cuda_ep.GetScratchBuffer<T>(input_count, ort_stream, WaitCudaNotificationOnDevice);
        input_data = reinterpret_cast<CudaT*>(input_data_buffer.get());
        fast_divmod tmp_div;
        Impl_Mul<CudaT>(stream, static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                        reinterpret_cast<const CudaT*>(input.Data<T>()), nullptr,
                        reinterpret_cast<const CudaT*>(input.Data<T>()), nullptr, tmp_div, tmp_div,
                        reinterpret_cast<CudaT*>(input_data_buffer.get()), input_count);
        input_data = reinterpret_cast<const CudaT*>(input_data_buffer.get());
      }

      switch (applicable_matrix_reduction) {
        case ApplicableMatrixReduction::Rows: {
          ORT_RETURN_IF_ERROR(reduce_matrix_rows(
              stream, input_data, reinterpret_cast<CudaT*>(output.MutableData<T>()), m, n));
        } break;
        case ApplicableMatrixReduction::Columns: {
          const auto buffer_size_bytes = compute_reduce_matrix_columns_buffer_size<CudaT>(m, n);
          auto buffer = cuda_ep.GetScratchBuffer<void>(buffer_size_bytes, ort_stream, WaitCudaNotificationOnDevice);
          ORT_RETURN_IF_ERROR(reduce_matrix_columns(stream, input_data,
                                                    reinterpret_cast<CudaT*>(output.MutableData<T>()), m, n,
                                                    buffer.get(), buffer_size_bytes));
        } break;
        default: {
          ORT_ENFORCE(false, "Invild matrix reduction type.");
        }
      }

      if (calculate_log) {
        Impl_Log<CudaT>(stream, reinterpret_cast<const CudaT*>(output.Data<T>()),
                        reinterpret_cast<CudaT*>(output.MutableData<T>()), output_count);
      } else if (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_AVG) {
        float denominator_float = applicable_matrix_reduction == ApplicableMatrixReduction::Rows
                                      ? static_cast<float>(m)
                                      : static_cast<float>(n);
        CudaT denominator = ToCudaType<T>::FromFloat(denominator_float);
        UnaryDiv(stream, reinterpret_cast<const CudaT*>(output.Data<T>()),
                 reinterpret_cast<CudaT*>(output.MutableData<T>()), denominator, output_count);
      }

      return Status::OK();
    }
  }

  // This reduction keep adding values to this buffer. If a non-zero value, say 1000, is here, the sum will start with 1000.
  // Therefore zeroing out the memory is required
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(output.MutableDataRaw(), 0, output.SizeInBytes(), stream));

  IAllocatorUniquePtr<float> temp_X;
  cudnnDataType_t cudnn_type_X = CUDNN_DATA_FLOAT;

  // Reducesum with BFP16 is not supported by cudnn, so convert input to fp32 then call cudnn
  if ((ReduceTensorIndices == CUDNN_REDUCE_TENSOR_FLATTENED_INDICES && std::is_same<T, MLFloat16>::value) ||
      (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_NO_INDICES && std::is_same<T, BFloat16>::value)) {
    // ArgMax/ArgMin with FP16 are not supported by cudnn, so convert input to fp32 then call cudnn
    temp_X = cuda_ep.GetScratchBuffer<float>(input_count, ort_stream, WaitCudaNotificationOnDevice);
    Impl_Cast<CudaT, float>(stream, reinterpret_cast<const CudaT*>(input.Data<T>()), temp_X.get(), input_shape.Size());
  } else {
    cudnn_type_X = CudnnTensor::GetDataType<CudaT>();
  }

  CudnnReduceDescriptor reduce_desc;
  if constexpr (std::is_same<T, MLFloat16>::value || std::is_same<T, BFloat16>::value) {
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
  CudaStream* cuda_stream = static_cast<CudaStream*>(ort_stream);
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(CudaKernel::GetCudnnHandle(cuda_stream), reduce_desc,
                                                       input_tensor, output_tensor, &workspace_bytes));
  auto workspace_cuda = cuda_ep.GetScratchBuffer<CudaT>(workspace_bytes, ort_stream, WaitCudaNotificationOnDevice);

  size_t indices_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(CudaKernel::GetCudnnHandle(cuda_stream), reduce_desc,
                                                     input_tensor, output_tensor, &indices_bytes));
  auto indices_cuda = cuda_ep.GetScratchBuffer<uint32_t>(indices_bytes, ort_stream, WaitCudaNotificationOnDevice);

  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_NO_INDICES) {
    IAllocatorUniquePtr<T> input_data_buffer(nullptr, [](T*) {});
    CudaT* input_data = nullptr;
    if (calculate_sqt) {
      input_data_buffer = cuda_ep.GetScratchBuffer<T>(input_count, ort_stream, WaitCudaNotificationOnDevice);
      input_data = reinterpret_cast<CudaT*>(input_data_buffer.get());
      fast_divmod tmp_div;
      Impl_Mul<CudaT>(stream,
                      static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                      reinterpret_cast<const CudaT*>(input.Data<T>()), nullptr,
                      reinterpret_cast<const CudaT*>(input.Data<T>()), nullptr,
                      tmp_div, tmp_div,
                      input_data, input_count);
    } else if (log_sum_exp) {
      // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      // This happens when the input is Scalar
      if (input_count == output_count) {
        if (output.MutableData<T>() != input.Data<T>()) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output.MutableData<T>(), input.Data<T>(), input_count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
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
        CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(CudaKernel::GetCudnnHandle(cuda_stream), reduce_max_desc,
                                                           input_tensor, output_tensor, &indices_bytes_max));
        auto indices_cuda_max = cuda_ep.GetScratchBuffer<uint32_t>(indices_bytes, ort_stream, WaitCudaNotificationOnDevice);
        auto* p_output = reinterpret_cast<CudaT*>(output.template MutableData<T>());
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            CudaKernel::GetCudnnHandle(cuda_stream), reduce_max_desc, indices_cuda_max.get(), indices_bytes_max,
            workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, reinterpret_cast<const CudaT*>(input.Data<T>()),
            &zero, output_tensor, p_output));
      }

      // Exp(X-ReduceMax)
      const TensorShape output_shape(output_dims);
      auto exp_result_buffer = cuda_ep.GetScratchBuffer<T>(input_count, ort_stream, WaitCudaNotificationOnDevice);
      auto exp_result = exp_result_buffer.get();
      auto log_sum_result_buffer = cuda_ep.GetScratchBuffer<T>(output_count, ort_stream, WaitCudaNotificationOnDevice);
      auto log_sum_result = log_sum_result_buffer.get();
      BinaryElementwisePreparation prepare;
      ORT_RETURN_IF_ERROR(prepare.BinaryElementwiseBroadcastPrepareHelper(input_shape, output_shape, input_shape));
      Impl_Sub<CudaT>(stream,
                      prepare.output_rank_or_simple_broadcast,
                      &prepare.lhs_padded_strides,
                      reinterpret_cast<const CudaT*>(input.Data<T>()),
                      &prepare.rhs_padded_strides,
                      reinterpret_cast<CudaT*>(output.MutableData<T>()),
                      &prepare.fdm_output_strides,
                      prepare.fdm_H, prepare.fdm_C,
                      reinterpret_cast<CudaT*>(exp_result), input_count);

      Impl_Exp<CudaT>(stream,
                      reinterpret_cast<CudaT*>(exp_result),
                      reinterpret_cast<CudaT*>(exp_result),
                      input_count);

      // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      // This happens when the input is Scalar. We do not need to add anything in this case.
      if (input_count == output_count) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(reinterpret_cast<CudaT*>(log_sum_result), exp_result, input_count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
      } else {
        // ReduceSum
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            CudaKernel::GetCudnnHandle(cuda_stream), reduce_desc, indices_cuda.get(), indices_bytes,
            workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, exp_result,
            &zero, output_tensor, reinterpret_cast<CudaT*>(log_sum_result)));
      }

      // Log(Sum)
      Impl_Log<CudaT>(stream, reinterpret_cast<CudaT*>(log_sum_result),
                      reinterpret_cast<CudaT*>(log_sum_result),
                      output_count);

      // Log + ReduceMax
      fast_divmod tmp_div;
      Impl_Add<CudaT>(stream, static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                      reinterpret_cast<CudaT*>(log_sum_result), nullptr,
                      reinterpret_cast<CudaT*>(output.MutableData<T>()), nullptr,
                      tmp_div, tmp_div,
                      reinterpret_cast<CudaT*>(output.MutableData<T>()), output_count);

      return Status::OK();
    }
    if (calculate_sqt) {
      // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      // This happens when the input is Scalar. We do not need to add anything in this case.
      if (input_count == output_count) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(reinterpret_cast<CudaT*>(output.MutableData<T>()), input_data, input_count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
      } else {
        auto* p_output = reinterpret_cast<CudaT*>(output.template MutableData<T>());
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            CudaKernel::GetCudnnHandle(cuda_stream), reduce_desc, indices_cuda.get(), indices_bytes,
            workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, input_data,
            &zero, output_tensor, p_output));
      }
    } else {
      // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      if (input_count == output_count) {
        if (output.MutableData<T>() != input.Data<T>()) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output.MutableData<T>(), input.Data<T>(), input_count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        }
      } else {
        if (temp_X) {
          auto temp_output = cuda_ep.GetScratchBuffer<float>(output_count, ort_stream, WaitCudaNotificationOnDevice);
          CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
              CudaKernel::GetCudnnHandle(cuda_stream), reduce_desc, indices_cuda.get(), indices_bytes,
              workspace_cuda.get(), workspace_bytes,
              &one, input_tensor, temp_X.get(),
              &zero, output_tensor, temp_output.get()));

          Impl_Cast<float, CudaT>(stream, temp_output.get(), reinterpret_cast<CudaT*>(output.MutableData<T>()), output_count);
        } else {
          auto* p_output = reinterpret_cast<CudaT*>(output.template MutableData<T>());
          CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
              CudaKernel::GetCudnnHandle(cuda_stream), reduce_desc, indices_cuda.get(), indices_bytes,
              workspace_cuda.get(), workspace_bytes,
              &one, input_tensor, reinterpret_cast<const CudaT*>(input.Data<T>()),
              &zero, output_tensor, p_output));
        }
      }
    }
  } else {
    // For ArgMax & ArgMin ops, use the indicies as the output with int64 type
    // cudnnReduceTensor has issue if input and output has same size, which will happen if the axis to be reduced has dim value of 1.
    // the output is zeros of the output size
    if (input_count == output_count) {
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(output.MutableData<int64_t>(), static_cast<int64_t>(0), output_count * sizeof(int64_t), stream));
    } else {
      if (temp_X) {
        auto temp_output = cuda_ep.GetScratchBuffer<float>(output_count, ort_stream, WaitCudaNotificationOnDevice);
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            CudaKernel::GetCudnnHandle(cuda_stream), reduce_desc, indices_cuda.get(), indices_bytes,
            workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, temp_X.get(),
            &zero, output_tensor, temp_output.get()));
      } else {
        auto temp_output = cuda_ep.GetScratchBuffer<CudaT>(output_count, ort_stream, WaitCudaNotificationOnDevice);
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            CudaKernel::GetCudnnHandle(cuda_stream), reduce_desc, indices_cuda.get(), indices_bytes,
            workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, reinterpret_cast<const CudaT*>(input.Data<T>()),
            &zero, output_tensor, temp_output.get()));
      }

      // CUDA reduction index is uint32_t for now, cast it to int64_t according to ONNX spec
      Impl_Cast<uint32_t, int64_t>(stream, reinterpret_cast<uint32_t*>(indices_cuda.get()), output.MutableData<int64_t>(), output_count);
    }
  }

  if (calculate_log) {
    Impl_Log<CudaT>(stream,
                    reinterpret_cast<CudaT*>(output.MutableData<T>()),
                    reinterpret_cast<CudaT*>(output.MutableData<T>()),
                    output_count);
  }

  return Status::OK();
}

template <bool allow_multi_axes>
template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
Status ReduceKernel<allow_multi_axes>::ComputeImpl(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnn_reduce_op) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  TensorShapeVector axes;

  size_t num_inputs = ctx->InputCount();
  if (num_inputs == 2) {
    // override the attribute value with the input value for reduction_axes
    const Tensor* axes_tensor = ctx->Input<Tensor>(1);
    ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
    ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1, "An axes tensor must be a vector tensor.");
    auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
    const auto* data = axes_tensor->Data<int64_t>();
    axes.assign(data, data + nDims);
  } else {
    axes.assign(axes_.begin(), axes_.end());
  }

  // empty axes and no-op
  if (axes.empty() && noop_with_empty_axes_) {
    auto* Y = ctx->Output(0, X->Shape());
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y->MutableData<T>(), X->Data<T>(), X->SizeInBytes(),
                                         cudaMemcpyDeviceToDevice, Stream(ctx)));
    return Status::OK();
  }

#ifdef ENABLE_TRAINING
  // Use ATen for ReduceSum if possible.
  const TensorShape& input_shape = X->Shape();
  if (contrib::IsATenOperatorExecutorInitialized() && cudnn_reduce_op == CUDNN_REDUCE_TENSOR_ADD && !calculate_log_ &&
      !calculate_sqt_ && !log_sum_exp_ && input_shape.Size() > 0) {
    if (axes.empty()) {
      axes.resize(input_shape.NumDimensions());
      std::iota(axes.begin(), axes.end(), 0);
    }
    ORT_RETURN_IF_ERROR(contrib::ExecuteReduceSumATen(ctx, axes, keepdims_));
    return Status::OK();
  }
#endif

  PrepareReduceMetadata prepare_reduce_metadata;
  ORT_RETURN_IF_ERROR(PrepareForReduce(X, keepdims_, axes, prepare_reduce_metadata));
  Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);
  const bool fast_reduction = fast_reduction_ && !ctx->GetUseDeterministicCompute();
  return ReduceComputeCore<T, ReduceTensorIndices>(*cuda_ep_, *X, prepare_reduce_metadata, *Y, cudnn_reduce_op, axes,
                                                   calculate_log_, calculate_sqt_, log_sum_exp_, fast_reduction, ctx->GetComputeStream());
}

#define SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(T)                                                                           \
  template <>                                                                                                             \
  template <>                                                                                                             \
  Status ReduceKernel<true>::ComputeImpl<T, CUDNN_REDUCE_TENSOR_NO_INDICES>(                                              \
      OpKernelContext * ctx, cudnnReduceTensorOp_t cudnn_reduce_op) const {                                               \
    typedef typename ToCudaType<T>::MappedType CudaT;                                                                     \
    const Tensor* X = ctx->Input<Tensor>(0);                                                                              \
    TensorShapeVector axes;                                                                                               \
    size_t num_inputs = ctx->InputCount();                                                                                \
    if (num_inputs == 2) {                                                                                                \
      const Tensor* axes_tensor = ctx->Input<Tensor>(1);                                                                  \
      ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");                                                          \
      ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1, "An axes tensor must be a vector tensor.");                  \
      auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);                                                          \
      const auto* data = axes_tensor->Data<int64_t>();                                                                    \
      axes.assign(data, data + nDims);                                                                                    \
    } else {                                                                                                              \
      axes.assign(axes_.begin(), axes_.end());                                                                            \
    }                                                                                                                     \
                                                                                                                          \
    if (axes.empty() && noop_with_empty_axes_) {                                                                          \
      auto* Y = ctx->Output(0, X->Shape());                                                                               \
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y->MutableData<T>(), X->Data<T>(), X->SizeInBytes(),                           \
                                           cudaMemcpyDeviceToDevice, Stream(ctx)));                                       \
      return Status::OK();                                                                                                \
    }                                                                                                                     \
                                                                                                                          \
    PrepareReduceMetadata prepare_reduce_metadata;                                                                        \
    ORT_RETURN_IF_ERROR(PrepareForReduce(X, keepdims_, axes, prepare_reduce_metadata));                                   \
                                                                                                                          \
    Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);                                             \
                                                                                                                          \
    int64_t input_count = prepare_reduce_metadata.input_count;                                                            \
    int64_t output_count = prepare_reduce_metadata.output_count;                                                          \
    auto& input_dims_cudnn = prepare_reduce_metadata.input_dims_cudnn;                                                    \
    auto& output_dims_cudnn = prepare_reduce_metadata.output_dims_cudnn;                                                  \
                                                                                                                          \
    if (input_count == 0) {                                                                                               \
      assert(Y->Shape().Size() == 0);                                                                                     \
      return Status::OK();                                                                                                \
    }                                                                                                                     \
                                                                                                                          \
    if (input_count == output_count) {                                                                                    \
      if (Y->MutableData<T>() != X->Data<T>()) {                                                                          \
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y->MutableData<T>(), X->Data<T>(),                                           \
                                             input_count * sizeof(T), cudaMemcpyDeviceToDevice, Stream(ctx)));            \
      }                                                                                                                   \
      return Status::OK();                                                                                                \
    }                                                                                                                     \
                                                                                                                          \
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(Y->MutableDataRaw(), 0, Y->SizeInBytes(), Stream(ctx)));                         \
                                                                                                                          \
    size_t indices_bytes = 0;                                                                                             \
    size_t workspace_bytes = 0;                                                                                           \
    CudnnTensor input_tensor;                                                                                             \
    CudnnTensor output_tensor;                                                                                            \
    CudnnReduceDescriptor reduce_desc;                                                                                    \
                                                                                                                          \
    cudnnDataType_t cudnn_type_X = CUDNN_DATA_FLOAT;                                                                      \
    IAllocatorUniquePtr<float> temp_X = GetScratchBuffer<float>(input_count, ctx->GetComputeStream());                             \
    Impl_Cast<CudaT, float>(Stream(ctx), reinterpret_cast<const CudaT*>(X->Data<T>()), temp_X.get(), X->Shape().Size());  \
                                                                                                                          \
    ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnn_reduce_op, cudnn_type_X, CUDNN_REDUCE_TENSOR_NO_INDICES));                  \
    ORT_RETURN_IF_ERROR(input_tensor.Set(input_dims_cudnn, cudnn_type_X));                                                \
    ORT_RETURN_IF_ERROR(output_tensor.Set(output_dims_cudnn, cudnn_type_X));                                              \
    CUDNN_RETURN_IF_ERROR(                                                                                                \
        cudnnGetReductionIndicesSize(GetCudnnHandle(ctx), reduce_desc, input_tensor, output_tensor, &indices_bytes));     \
    CUDNN_RETURN_IF_ERROR(                                                                                                \
        cudnnGetReductionWorkspaceSize(GetCudnnHandle(ctx), reduce_desc, input_tensor, output_tensor, &workspace_bytes)); \
    IAllocatorUniquePtr<uint32_t> indices_cuda = GetScratchBuffer<uint32_t>(indices_bytes, ctx->GetComputeStream());               \
    IAllocatorUniquePtr<CudaT> workspace_cuda = GetScratchBuffer<CudaT>(workspace_bytes, ctx->GetComputeStream());                 \
                                                                                                                          \
    const auto one = Consts<float>::One;                                                                                  \
    const auto zero = Consts<float>::Zero;                                                                                \
    auto temp_Y = GetScratchBuffer<float>(output_count, ctx->GetComputeStream());                                                  \
    CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(GetCudnnHandle(ctx), reduce_desc, indices_cuda.get(), indices_bytes,          \
                                            workspace_cuda.get(), workspace_bytes, &one, input_tensor, temp_X.get(),      \
                                            &zero, output_tensor, temp_Y.get()));                                         \
    Impl_Cast<float, CudaT>(Stream(ctx), temp_Y.get(), reinterpret_cast<CudaT*>(Y->MutableData<T>()), output_count);      \
                                                                                                                          \
    return Status::OK();                                                                                                  \
  }

SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(int32_t)
SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(int64_t)
SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(int8_t)
SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(uint8_t)

namespace ReductionOps {

template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
std::unique_ptr<Tensor> ReduceCompute(const CUDAExecutionProvider& cuda_ep, cudnnReduceTensorOp_t cudnn_reduce_op, AllocatorPtr allocator,
                                      const Tensor& input, gsl::span<const int64_t> axes,
                                      bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
                                      bool fast_reduction, Stream* stream, const TensorShape* input_shape_override) {
  PrepareReduceMetadata prepare_reduce_metadata;
  auto status = PrepareForReduce(&input,
                                 keep_dims,
                                 axes,
                                 prepare_reduce_metadata,
                                 input_shape_override);

  if (!status.IsOK()) {
    ORT_THROW(ONNXRUNTIME, FAIL, "Failed to perform reduce op: ", status.ErrorMessage());
  }

  auto output = Tensor::Create(input.DataType(), prepare_reduce_metadata.squeezed_output_dims, allocator);

  status = ReduceComputeCore<T, ReduceTensorIndices>(cuda_ep, input, prepare_reduce_metadata, *output, cudnn_reduce_op, axes,
                                                     calculate_log, calculate_sqt, log_sum_exp, fast_reduction, stream, input_shape_override);

  if (!status.IsOK()) {
    ORT_THROW(ONNXRUNTIME, FAIL, "Failed to perform reduce op: ", status.ErrorMessage());
  }

  return output;
}

// Explicit template instantiation (needed to be used in einsum_auxiliary_ops.cc)

template std::unique_ptr<Tensor> ReduceCompute<float, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const CUDAExecutionProvider& cuda_ep, cudnnReduceTensorOp_t cudnn_reduce_op,
    AllocatorPtr allocator,
    const Tensor& input, gsl::span<const int64_t> axes,
    bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
    bool fast_reduction, Stream* stream, const TensorShape* input_shape_override);

template std::unique_ptr<Tensor> ReduceCompute<double, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const CUDAExecutionProvider& cuda_ep, cudnnReduceTensorOp_t cudnn_reduce_op,
    AllocatorPtr allocator,
    const Tensor& input, gsl::span<const int64_t> axes,
    bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
    bool fast_reduction, Stream* stream, const TensorShape* input_shape_override);

template std::unique_ptr<Tensor> ReduceCompute<MLFloat16, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const CUDAExecutionProvider& cuda_ep, cudnnReduceTensorOp_t cudnn_reduce_op,
    AllocatorPtr allocator,
    const Tensor& input, gsl::span<const int64_t> axes,
    bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
    bool fast_reduction, Stream* stream, const TensorShape* input_shape_override);

}  // namespace ReductionOps

#define REGISTER_KERNEL_HFD(name)        \
  REGISTER_KERNEL_TYPED(name, MLFloat16) \
  REGISTER_KERNEL_TYPED(name, float)     \
  REGISTER_KERNEL_TYPED(name, double)    \
  REGISTER_KERNEL_TYPED(name, BFloat16)

#define REGISTER_KERNEL_HFD_VERSIONED_11(name)        \
  REGISTER_KERNEL_VERSIONED_TYPED_11(name, MLFloat16) \
  REGISTER_KERNEL_VERSIONED_TYPED_11(name, float)     \
  REGISTER_KERNEL_VERSIONED_TYPED_11(name, double)

REGISTER_KERNEL_HFD_VERSIONED_11(ArgMax)
REGISTER_KERNEL_HFD_VERSIONED_11(ArgMin)
REGISTER_KERNEL_HFD(ReduceL1)
REGISTER_KERNEL_HFD(ReduceL2)

REGISTER_KERNEL_TYPED_13_WITH_VERSIONED_12(ReduceMax, MLFloat16)
REGISTER_KERNEL_TYPED_13_WITH_VERSIONED_12(ReduceMax, float)
REGISTER_KERNEL_TYPED_13_WITH_VERSIONED_12(ReduceMax, double)
REGISTER_KERNEL_TYPED_13_WITH_VERSIONED_12(ReduceMax, int32_t)
REGISTER_KERNEL_TYPED_13_WITH_VERSIONED_12(ReduceMax, int64_t)
REGISTER_KERNEL_TYPED_13_WITH_VERSIONED_12(ReduceMax, int8_t)
REGISTER_KERNEL_TYPED_13_WITH_VERSIONED_12(ReduceMax, uint8_t)

REGISTER_KERNEL_HFD(ReduceMean)

REGISTER_KERNEL_VERSIONED_TYPED_13(ReduceMin, MLFloat16)
REGISTER_KERNEL_VERSIONED_TYPED_13(ReduceMin, float)
REGISTER_KERNEL_VERSIONED_TYPED_13(ReduceMin, double)
REGISTER_KERNEL_VERSIONED_TYPED_13(ReduceMin, int32_t)
REGISTER_KERNEL_VERSIONED_TYPED_13(ReduceMin, int64_t)
REGISTER_KERNEL_VERSIONED_TYPED_13(ReduceMin, int8_t)
REGISTER_KERNEL_VERSIONED_TYPED_13(ReduceMin, uint8_t)

REGISTER_KERNEL_TYPED_14(ReduceMin, MLFloat16)
REGISTER_KERNEL_TYPED_14(ReduceMin, float)
REGISTER_KERNEL_TYPED_14(ReduceMin, double)
REGISTER_KERNEL_TYPED_14(ReduceMin, int32_t)
REGISTER_KERNEL_TYPED_14(ReduceMin, int8_t)
REGISTER_KERNEL_TYPED_14(ReduceMin, uint8_t)
REGISTER_KERNEL_TYPED_14(ReduceMin, int64_t)

REGISTER_KERNEL_HFD(ReduceProd)

REGISTER_KERNEL_TYPED_13(ReduceSum, MLFloat16)
REGISTER_KERNEL_TYPED_13(ReduceSum, float)
REGISTER_KERNEL_TYPED_13(ReduceSum, double)
REGISTER_KERNEL_TYPED_13(ReduceSum, int32_t)
REGISTER_KERNEL_TYPED_13(ReduceSum, int64_t)
REGISTER_KERNEL_TYPED_13(ReduceSum, BFloat16)

REGISTER_KERNEL_HFD(ReduceLogSum)
REGISTER_KERNEL_HFD(ReduceSumSquare)
REGISTER_KERNEL_HFD(ReduceLogSumExp)

#define REGISTER_KERNEL_INT32(name) \
  REGISTER_KERNEL_TYPED(name, int32_t)

REGISTER_KERNEL_INT32(ReduceL1)
REGISTER_KERNEL_INT32(ReduceL2)
REGISTER_KERNEL_INT32(ReduceMean)

REGISTER_KERNEL_INT32(ReduceProd)

}  // namespace cuda
}  // namespace onnxruntime
