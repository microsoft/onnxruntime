// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/reduction/reduction_ops.h"

#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/rocm/miopen_common.h"
#include "core/providers/rocm/math/binary_elementwise_ops_impl.h"
#include "core/providers/rocm/math/binary_elementwise_ops.h"
#include "core/providers/rocm/math/unary_elementwise_ops_impl.h"

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

// ROCM ArgMax/ArgMin doesn't have OpSet12 implementation (with select_last_index attr), keep it in OpSet11 for now.
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

// Register with the latest version 13
#define REGISTER_KERNEL_TYPED_13(name, T)                                       \
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
      KernelDefBuilder()                                                        \
          .InputMemoryType<OrtMemTypeCPUInput>(1)                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),               \
      name<T>);

// TODO ReduceKernel::ReduceKernelShared() is still used by some other training classes though it's not used here - this should be refactored.
template <bool allow_multi_axes>
template <typename T, typename OutT, miopenReduceTensorIndices_t ReduceTensorIndices>
Status ReduceKernel<allow_multi_axes>::ReduceKernelShared(
    const T* X,
    const TensorShape& input_shape,
    OutT* Y,
    const TensorShape& output_shape,
    miopenReduceTensorOp_t miopen_reduce_op,
    std::vector<int64_t>& output_dims) const {
  typedef typename ToHipType<T>::MappedType HipT;
  typedef typename ToHipType<OutT>::MappedType HipOutT;
  miopenDataType_t miopen_type_X = MiopenTensor::GetDataType<HipT>();
  const auto rank = input_shape.NumDimensions();

  // Block of fast matrix reduction.
  if (fast_reduction_) {
    int m{}, n{};
    const auto applicable_matrix_reduction = get_applicable_matrix_reduction(
        miopen_reduce_op, input_shape.GetDims(), axes_, m, n);
    switch (applicable_matrix_reduction) {
      case ApplicableMatrixReduction::Rows: {
        return reduce_matrix_rows(
            Stream(),
            reinterpret_cast<const HipT*>(X),
            reinterpret_cast<HipOutT*>(Y),
            m, n, false);
      }
      case ApplicableMatrixReduction::Columns:
      // don't call reduce_matrix_columns() since it will reset initial output data
      default:
        break;
    }
  }

  const auto& input_dims = input_shape.GetDims();
  int64_t input_count = input_shape.Size();
  IAllocatorUniquePtr<float> temp_X;
  if (ReduceTensorIndices == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES && std::is_same<T, MLFloat16>::value) {
    // ArgMax/ArgMin with FP16 are not supported by miopen, so convert input to fp32 then call miopen
    temp_X = GetScratchBuffer<float>(input_count);
    miopen_type_X = miopenFloat;
    Impl_Cast<HipT, float>(Stream(), reinterpret_cast<const HipT*>(X), temp_X.get(), input_shape.Size());
  }

  // MIOpen requires at least 3D input, so pad 1s if needed
  std::vector<int64_t> input_dims_miopen = input_dims;
  std::vector<int64_t> output_dims_miopen = output_dims;
  if (rank < 3) {
    std::vector<int64_t> pads(3 - rank, 1);
    input_dims_miopen.insert(input_dims_miopen.end(), pads.begin(), pads.end());
    output_dims_miopen.insert(output_dims_miopen.end(), pads.begin(), pads.end());
  }

  MiopenReduceDescriptor reduce_desc;
  if (std::is_same<T, MLFloat16>::value)
    ORT_RETURN_IF_ERROR(reduce_desc.Set(miopen_reduce_op, MiopenTensor::GetDataType<float>(), ReduceTensorIndices));
  else
    ORT_RETURN_IF_ERROR(reduce_desc.Set(miopen_reduce_op, miopen_type_X, ReduceTensorIndices));

  // As of ROCm 4.2, miopenReduceTensor() requires alpha/beta to be the same data
  // type as the input type. This differs from cudnnReduceTensor() and other
  // MIOpen/cuDNN APIs where alpha/beta are float when input type is half (float16).
  //
  // NOTE: this workaround can be removed in ROCm 4.3:
  //       https://github.com/ROCmSoftwarePlatform/MIOpen/pull/914
  const auto one = ReduceConsts<HipT>::One;
  const auto zero = ReduceConsts<HipT>::Zero;
  MiopenTensor input_tensor;
  MiopenTensor output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set(input_dims_miopen, miopen_type_X));
  ORT_RETURN_IF_ERROR(output_tensor.Set(output_dims_miopen, miopen_type_X));
  size_t workspace_bytes = 0;
  MIOPEN_RETURN_IF_ERROR(miopenGetReductionWorkspaceSize(MiopenHandle(), reduce_desc, input_tensor, output_tensor, &workspace_bytes));
  auto workspace_rocm = GetScratchBuffer<HipT>(workspace_bytes);

  size_t indices_bytes = 0;
  MIOPEN_RETURN_IF_ERROR(miopenGetReductionIndicesSize(MiopenHandle(), reduce_desc, input_tensor, output_tensor, &indices_bytes));
  auto indices_rocm = GetScratchBuffer<uint32_t>(indices_bytes);

  // need to allocate a separate buffer for ArgMin/ArgMax comparsion output
  auto output_count = output_shape.Size();

  if (ReduceTensorIndices == MIOPEN_REDUCE_TENSOR_NO_INDICES) {
    IAllocatorUniquePtr<T> input_data_buffer(nullptr, [](T*) {});
    HipT* input_data = nullptr;
    if (calculate_sqt_) {
      input_data_buffer = GetScratchBuffer<T>(input_count);
      input_data = reinterpret_cast<HipT*>(input_data_buffer.get());
      fast_divmod tmp_div;
      Impl_Mul<HipT>(Stream(),
                     static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                     reinterpret_cast<const HipT*>(X), nullptr,
                     reinterpret_cast<const HipT*>(X), nullptr,
                     tmp_div, tmp_div,
                     input_data, input_count);
    } else if (log_sum_exp_) {
      // Reduce max -- Max/Min will output indices data
      MiopenReduceDescriptor reduce_max_desc;
      ORT_RETURN_IF_ERROR(reduce_max_desc.Set(MIOPEN_REDUCE_TENSOR_MAX, miopen_type_X, MIOPEN_REDUCE_TENSOR_NO_INDICES));
      size_t indices_bytes_max = 0;
      MIOPEN_RETURN_IF_ERROR(miopenGetReductionIndicesSize(MiopenHandle(), reduce_max_desc, input_tensor, output_tensor, &indices_bytes_max));
      auto indices_rocm_max = GetScratchBuffer<uint32_t>(indices_bytes);
      MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(
          MiopenHandle(), reduce_max_desc, indices_rocm_max.get(), indices_bytes_max, workspace_rocm.get(), workspace_bytes,
          &one, input_tensor, reinterpret_cast<const HipT*>(X),
          &zero, output_tensor, reinterpret_cast<HipT*>(Y)));

      // Exp(X-ReduceMax)
      const TensorShape rhs_shape(output_dims);
      auto exp_result_buffer = GetScratchBuffer<T>(input_count);
      auto exp_result = exp_result_buffer.get();
      auto log_sum_result_buffer = GetScratchBuffer<T>(output_count);
      auto log_sum_result = log_sum_result_buffer.get();
      BinaryElementwisePreparation prepare;
      ORT_RETURN_IF_ERROR(prepare.BinaryElementwiseBroadcastPrepareHelper(input_shape, rhs_shape, input_shape));
      Impl_Sub<HipT>(Stream(),
                     prepare.output_rank_or_simple_broadcast,
                     &prepare.lhs_padded_strides,
                     reinterpret_cast<const HipT*>(X),
                     &prepare.rhs_padded_strides,
                     reinterpret_cast<HipT*>(Y),
                     &prepare.fdm_output_strides,
                     prepare.fdm_H, prepare.fdm_C,
                     reinterpret_cast<HipT*>(exp_result), input_count);

      Impl_Exp<HipT>(Stream(),
                     reinterpret_cast<HipT*>(exp_result),
                     reinterpret_cast<HipT*>(exp_result),
                     input_count);

      // ReduceSum
      MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(
          MiopenHandle(), reduce_desc, indices_rocm.get(), indices_bytes, workspace_rocm.get(), workspace_bytes,
          &one, input_tensor, exp_result,
          &zero, output_tensor, reinterpret_cast<HipT*>(log_sum_result)));

      // Log(Sum)
      Impl_Log<HipT>(Stream(),
                     reinterpret_cast<HipT*>(log_sum_result),
                     reinterpret_cast<HipT*>(log_sum_result),
                     output_count);

      // Log + ReduceMax
      fast_divmod tmp_div;
      Impl_Add<HipT>(Stream(),
                     static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                     reinterpret_cast<HipT*>(log_sum_result), nullptr,
                     reinterpret_cast<HipT*>(Y), nullptr,
                     tmp_div, tmp_div,
                     reinterpret_cast<HipT*>(Y), output_count);

      return Status::OK();
    }
    if (calculate_sqt_) {
      MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(
          MiopenHandle(), reduce_desc, indices_rocm.get(), indices_bytes, workspace_rocm.get(), workspace_bytes,
          &one, input_tensor, input_data,
          &zero, output_tensor, reinterpret_cast<HipT*>(Y)));
    } else {
      // miopenReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      if (input_count == output_count) {
        if (reinterpret_cast<const void*>(Y) != reinterpret_cast<const void*>(X)) {
          HIP_RETURN_IF_ERROR(hipMemcpyAsync(Y, X, input_count * sizeof(T), hipMemcpyDeviceToDevice, Stream()));
        }
      } else {
        MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(
            MiopenHandle(), reduce_desc, indices_rocm.get(), indices_bytes, workspace_rocm.get(), workspace_bytes,
            &one, input_tensor, reinterpret_cast<const HipT*>(X),
            &zero, output_tensor, reinterpret_cast<HipT*>(Y)));
      }
    }
  } else {  // For ArgMax & ArgMin ops, use the indicies as the output with int64 type
    if (temp_X) {
      auto temp_output = GetScratchBuffer<float>(output_count);
      MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(
          MiopenHandle(), reduce_desc, indices_rocm.get(), indices_bytes, workspace_rocm.get(), workspace_bytes,
          &one, input_tensor, temp_X.get(),
          &zero, output_tensor, temp_output.get()));
    } else {
      auto temp_output = GetScratchBuffer<HipT>(output_count);
      MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(
          MiopenHandle(), reduce_desc, indices_rocm.get(), indices_bytes, workspace_rocm.get(), workspace_bytes,
          &one, input_tensor, reinterpret_cast<const HipT*>(X),
          &zero, output_tensor, temp_output.get()));
    }

    // MIOpen reduction index is uint32_t for now, cast it to int64_t according to ONNX spec
    Impl_Cast<uint32_t, int64_t>(Stream(), reinterpret_cast<uint32_t*>(indices_rocm.get()), reinterpret_cast<int64_t*>(Y), output_count);
  }

  if (calculate_log_) {
    Impl_Log<HipT>(Stream(),
                   reinterpret_cast<HipT*>(Y),
                   reinterpret_cast<HipT*>(Y),
                   output_count);
  }

  return Status::OK();
}

// template Status ReduceKernel<true>::ReduceKernelShared<double, double, MIOPEN_REDUCE_TENSOR_NO_INDICES>(
//     const double* X,
//     const TensorShape& input_shape,
//     double* Y,
//     const TensorShape& output_shape,
//     miopenReduceTensorOp_t miopen_reduce_op,
//     std::vector<int64_t>& output_dims) const;

template Status ReduceKernel<true>::ReduceKernelShared<float, float, MIOPEN_REDUCE_TENSOR_NO_INDICES>(
    const float* X,
    const TensorShape& input_shape,
    float* Y,
    const TensorShape& output_shape,
    miopenReduceTensorOp_t miopen_reduce_op,
    std::vector<int64_t>& output_dims) const;

template Status ReduceKernel<true>::ReduceKernelShared<MLFloat16, MLFloat16, MIOPEN_REDUCE_TENSOR_NO_INDICES>(
    const MLFloat16* X,
    const TensorShape& input_shape,
    MLFloat16* Y,
    const TensorShape& output_shape,
    miopenReduceTensorOp_t miopen_reduce_op,
    std::vector<int64_t>& output_dims) const;

// `input_shape_override` (if provided) is the input shape for compute purposes
Status PrepareForReduce(const Tensor* X,
                        bool keepdims,
                        const std::vector<int64_t>& axes,
                        PrepareReduceMetadata& prepare_reduce_metadata,
                        const TensorShape* input_shape_override) {
  ORT_ENFORCE(nullptr != X);

  const TensorShape& input_shape = input_shape_override ? *input_shape_override : X->Shape();
  const int64_t rank = gsl::narrow<int64_t>(input_shape.NumDimensions());
  prepare_reduce_metadata.input_count = input_shape.Size();

  if (rank > 8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MIOpen only supports up to 8-D tensors in reduction");
  }

  const auto& input_dims = input_shape.GetDims();
  std::vector<bool> reduced(rank, false);
  prepare_reduce_metadata.output_dims.reserve(input_dims.size());
  if (axes.size() > 0) {
    prepare_reduce_metadata.output_dims = input_dims;
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

  // MIOpen requires at least 3D input, so pad 1s if needed
  prepare_reduce_metadata.input_dims_miopen = input_dims;
  prepare_reduce_metadata.output_dims_miopen = prepare_reduce_metadata.output_dims;
  if (rank < 3) {
    std::vector<int64_t> pads(3 - rank, 1);
    prepare_reduce_metadata.input_dims_miopen.insert(prepare_reduce_metadata.input_dims_miopen.end(), pads.begin(), pads.end());
    prepare_reduce_metadata.output_dims_miopen.insert(prepare_reduce_metadata.output_dims_miopen.end(), pads.begin(), pads.end());
  }

  prepare_reduce_metadata.output_count = TensorShape(prepare_reduce_metadata.output_dims).Size();

  return Status::OK();
}

// `input_shape_override` is the input shape for compute purposes (if provided)
template <typename T, miopenReduceTensorIndices_t ReduceTensorIndices>
Status ReduceComputeCore(ROCMExecutionProvider& rocm_ep, const Tensor& input, PrepareReduceMetadata& prepare_reduce_metadata,
                         /*out*/ Tensor& output, miopenReduceTensorOp_t miopen_reduce_op,
                         const std::vector<int64_t>& axes,
                         bool calculate_log, bool calculate_sqt, bool log_sum_exp, bool fast_reduction,
                         const TensorShape* input_shape_override) {
  typedef typename ToHipType<T>::MappedType HipT;
  const TensorShape& input_shape = input_shape_override ? *input_shape_override : input.Shape();

  int64_t input_count = prepare_reduce_metadata.input_count;
  int64_t output_count = prepare_reduce_metadata.output_count;
  std::vector<int64_t>& output_dims = prepare_reduce_metadata.output_dims;
  std::vector<int64_t>& input_dims_miopen = prepare_reduce_metadata.input_dims_miopen;
  std::vector<int64_t>& output_dims_miopen = prepare_reduce_metadata.output_dims_miopen;
  hipStream_t stream = static_cast<hipStream_t>(rocm_ep.GetComputeStream());
  // special case when there is a dim value of 0 in the shape.
  if (input_count == 0) {
    assert(output.Shape().Size() == 0);
    return Status::OK();
  }

  // Block of fast matrix reduction.
  if (fast_reduction) {
    int m{}, n{};
    const auto applicable_matrix_reduction = get_applicable_matrix_reduction(
        miopen_reduce_op, input_shape.GetDims(), axes, m, n);
    switch (applicable_matrix_reduction) {
      case ApplicableMatrixReduction::Rows: {
        return reduce_matrix_rows(
            stream,
            reinterpret_cast<const HipT*>(input.template Data<T>()),
            reinterpret_cast<HipT*>(output.template MutableData<T>()),
            m, n);
      }
      case ApplicableMatrixReduction::Columns: {
        const auto buffer_size_bytes = compute_reduce_matrix_columns_buffer_size<HipT>(m, n);
        auto buffer = rocm_ep.GetScratchBuffer<void>(buffer_size_bytes);
        return reduce_matrix_columns(
            stream,
            reinterpret_cast<const HipT*>(input.template Data<T>()),
            reinterpret_cast<HipT*>(output.template MutableData<T>()),
            m, n, buffer.get(), buffer_size_bytes);
      }
      default:
        break;
    }
  }

  // This reduction keep adding values to this buffer. If a non-zero value, say 1000, is here, the sum will start with 1000.
  // Therefore zeroing out the memory is required
  HIP_RETURN_IF_ERROR(hipMemsetAsync(output.MutableDataRaw(), 0, output.SizeInBytes(), stream));

  IAllocatorUniquePtr<float> temp_X;
  miopenDataType_t miopen_type_X = MiopenTensor::GetDataType<HipT>();

  if (ReduceTensorIndices == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES && std::is_same<T, MLFloat16>::value) {
    // ArgMax/ArgMin with FP16 are not supported by miopen, so convert input to fp32 then call miopen
    temp_X = rocm_ep.GetScratchBuffer<float>(input_count);
    miopen_type_X = miopenFloat;
    Impl_Cast<HipT, float>(stream, reinterpret_cast<const HipT*>(input.template Data<T>()), temp_X.get(), input_shape.Size());
  }

  MiopenReduceDescriptor reduce_desc;
  if (std::is_same<T, MLFloat16>::value) {
    ORT_RETURN_IF_ERROR(reduce_desc.Set(miopen_reduce_op, MiopenTensor::GetDataType<float>(), ReduceTensorIndices));
  } else {
    ORT_RETURN_IF_ERROR(reduce_desc.Set(miopen_reduce_op, miopen_type_X, ReduceTensorIndices));
  }

  // As of ROCm 4.2, miopenReduceTensor() requires alpha/beta to be the same data
  // type as the input type. This differs from cudnnReduceTensor() and other
  // MIOpen/cuDNN APIs where alpha/beta are float when input type is half (float16).
  //
  // NOTE: this workaround can be removed in ROCm 4.3:
  //       https://github.com/ROCmSoftwarePlatform/MIOpen/pull/914
  const auto one = ReduceConsts<HipT>::One;
  const auto zero = ReduceConsts<HipT>::Zero;
  MiopenTensor input_tensor;
  MiopenTensor output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set(input_dims_miopen, miopen_type_X));
  ORT_RETURN_IF_ERROR(output_tensor.Set(output_dims_miopen, miopen_type_X));
  size_t workspace_bytes = 0;
  MIOPEN_RETURN_IF_ERROR(miopenGetReductionWorkspaceSize(rocm_ep.PerThreadMiopenHandle(), reduce_desc,
                                                         input_tensor, output_tensor, &workspace_bytes));
  auto workspace_rocm = rocm_ep.GetScratchBuffer<HipT>(workspace_bytes);

  size_t indices_bytes = 0;
  MIOPEN_RETURN_IF_ERROR(miopenGetReductionIndicesSize(rocm_ep.PerThreadMiopenHandle(), reduce_desc,
                                                       input_tensor, output_tensor, &indices_bytes));
  auto indices_rocm = rocm_ep.GetScratchBuffer<uint32_t>(indices_bytes);

  if (ReduceTensorIndices == MIOPEN_REDUCE_TENSOR_NO_INDICES) {
    IAllocatorUniquePtr<T> input_data_buffer(nullptr, [](T*) {});
    HipT* input_data = nullptr;
    if (calculate_sqt) {
      input_data_buffer = rocm_ep.GetScratchBuffer<T>(input_count);
      input_data = reinterpret_cast<HipT*>(input_data_buffer.get());
      fast_divmod tmp_div;
      Impl_Mul<HipT>(stream,
                     static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                     reinterpret_cast<const HipT*>(input.template Data<T>()), nullptr,
                     reinterpret_cast<const HipT*>(input.template Data<T>()), nullptr,
                     tmp_div, tmp_div,
                     input_data, input_count);
    } else if (log_sum_exp) {
      // miopenReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      // This happens when the input is Scalar
      if (input_count == output_count) {
        if (output.template MutableData<T>() != input.template Data<T>()) {
          HIP_RETURN_IF_ERROR(hipMemcpyAsync(output.template MutableData<T>(), input.template Data<T>(), input_count * sizeof(T), hipMemcpyDeviceToDevice, stream));
        }
      } else {
        // Reduce max -- Max/Min will output indices data
        MiopenReduceDescriptor reduce_max_desc;
        miopenDataType_t miopen_reduce_max_type = miopen_type_X;
        if ((std::is_same<T, MLFloat16>::value)) {
          miopen_reduce_max_type = miopenFloat;
        }
        ORT_RETURN_IF_ERROR(reduce_max_desc.Set(MIOPEN_REDUCE_TENSOR_MAX, miopen_reduce_max_type, MIOPEN_REDUCE_TENSOR_NO_INDICES));
        size_t indices_bytes_max = 0;
        MIOPEN_RETURN_IF_ERROR(miopenGetReductionIndicesSize(rocm_ep.PerThreadMiopenHandle(), reduce_max_desc,
                                                             input_tensor, output_tensor, &indices_bytes_max));
        auto indices_rocm_max = rocm_ep.GetScratchBuffer<uint32_t>(indices_bytes);
        MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(
            rocm_ep.PerThreadMiopenHandle(), reduce_max_desc, indices_rocm_max.get(), indices_bytes_max,
            workspace_rocm.get(), workspace_bytes,
            &one, input_tensor, reinterpret_cast<const HipT*>(input.template Data<T>()),
            &zero, output_tensor, reinterpret_cast<HipT*>(output.template MutableData<T>())));
      }

      // Exp(X-ReduceMax)
      const TensorShape output_shape(output_dims);
      auto exp_result_buffer = rocm_ep.GetScratchBuffer<T>(input_count);
      auto exp_result = exp_result_buffer.get();
      auto log_sum_result_buffer = rocm_ep.GetScratchBuffer<T>(output_count);
      auto log_sum_result = log_sum_result_buffer.get();
      BinaryElementwisePreparation prepare;
      ORT_RETURN_IF_ERROR(prepare.BinaryElementwiseBroadcastPrepareHelper(input_shape, output_shape, input_shape));
      Impl_Sub<HipT>(stream,
                     prepare.output_rank_or_simple_broadcast,
                     &prepare.lhs_padded_strides,
                     reinterpret_cast<const HipT*>(input.template Data<T>()),
                     &prepare.rhs_padded_strides,
                     reinterpret_cast<HipT*>(output.template MutableData<T>()),
                     &prepare.fdm_output_strides,
                     prepare.fdm_H, prepare.fdm_C,
                     reinterpret_cast<HipT*>(exp_result), input_count);

      Impl_Exp<HipT>(stream,
                     reinterpret_cast<HipT*>(exp_result),
                     reinterpret_cast<HipT*>(exp_result),
                     input_count);

      // miopenReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      // This happens when the input is Scalar. We do not need to add anything in this case.
      if (input_count == output_count) {
        HIP_RETURN_IF_ERROR(hipMemcpyAsync(reinterpret_cast<HipT*>(log_sum_result), exp_result, input_count * sizeof(T), hipMemcpyDeviceToDevice, stream));
      } else {
        // ReduceSum
        MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(
            rocm_ep.PerThreadMiopenHandle(), reduce_desc, indices_rocm.get(), indices_bytes,
            workspace_rocm.get(), workspace_bytes,
            &one, input_tensor, exp_result,
            &zero, output_tensor, reinterpret_cast<HipT*>(log_sum_result)));
      }

      // Log(Sum)
      Impl_Log<HipT>(stream,
                     reinterpret_cast<HipT*>(log_sum_result),
                     reinterpret_cast<HipT*>(log_sum_result),
                     output_count);

      // Log + ReduceMax
      fast_divmod tmp_div;
      Impl_Add<HipT>(stream,
                     static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                     reinterpret_cast<HipT*>(log_sum_result), nullptr,
                     reinterpret_cast<HipT*>(output.template MutableData<T>()), nullptr,
                     tmp_div, tmp_div,
                     reinterpret_cast<HipT*>(output.template MutableData<T>()), output_count);

      return Status::OK();
    }
    if (calculate_sqt) {
      // miopenReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      // This happens when the input is Scalar. We do not need to add anything in this case.
      if (input_count == output_count) {
        HIP_RETURN_IF_ERROR(hipMemcpyAsync(reinterpret_cast<HipT*>(output.template MutableData<T>()), input_data, input_count * sizeof(T), hipMemcpyDeviceToDevice, stream));
      } else {
        MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(
            rocm_ep.PerThreadMiopenHandle(), reduce_desc, indices_rocm.get(), indices_bytes,
            workspace_rocm.get(), workspace_bytes,
            &one, input_tensor, input_data,
            &zero, output_tensor, reinterpret_cast<HipT*>(output.template MutableData<T>())));
      }
    } else {
      // miopenReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      if (input_count == output_count) {
        if (output.template MutableData<T>() != input.template Data<T>()) {
          HIP_RETURN_IF_ERROR(hipMemcpyAsync(output.template MutableData<T>(), input.template Data<T>(), input_count * sizeof(T), hipMemcpyDeviceToDevice, stream));
        }
      } else {
        MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(
            rocm_ep.PerThreadMiopenHandle(), reduce_desc, indices_rocm.get(), indices_bytes,
            workspace_rocm.get(), workspace_bytes,
            &one, input_tensor, reinterpret_cast<const HipT*>(input.template Data<T>()),
            &zero, output_tensor, reinterpret_cast<HipT*>(output.template MutableData<T>())));
      }
    }
  } else {
    // For ArgMax & ArgMin ops, use the indicies as the output with int64 type
    // miopenReduceTensor has issue if input and output has same size, which will happen if the axis to be reduced has dim value of 1.
    // the output is zeros of the output size
    if (input_count == output_count) {
      HIP_RETURN_IF_ERROR(hipMemsetAsync(output.template MutableData<int64_t>(), static_cast<int64_t>(0), output_count * sizeof(int64_t), stream));
    } else {
      if (temp_X) {
        auto temp_output = rocm_ep.GetScratchBuffer<float>(output_count);
        MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(
            rocm_ep.PerThreadMiopenHandle(), reduce_desc, indices_rocm.get(), indices_bytes,
            workspace_rocm.get(), workspace_bytes,
            &one, input_tensor, temp_X.get(),
            &zero, output_tensor, temp_output.get()));
      } else {
        auto temp_output = rocm_ep.GetScratchBuffer<HipT>(output_count);
        MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(
            rocm_ep.PerThreadMiopenHandle(), reduce_desc, indices_rocm.get(), indices_bytes,
            workspace_rocm.get(), workspace_bytes,
            &one, input_tensor, reinterpret_cast<const HipT*>(input.template Data<T>()),
            &zero, output_tensor, temp_output.get()));
      }

      // MIOpen reduction index is uint32_t for now, cast it to int64_t according to ONNX spec
      Impl_Cast<uint32_t, int64_t>(stream, reinterpret_cast<uint32_t*>(indices_rocm.get()), output.template MutableData<int64_t>(), output_count);
    }
  }

  if (calculate_log) {
    Impl_Log<HipT>(stream,
                   reinterpret_cast<HipT*>(output.template MutableData<T>()),
                   reinterpret_cast<HipT*>(output.template MutableData<T>()),
                   output_count);
  }

  return Status::OK();
}

template <bool allow_multi_axes>
template <typename T, miopenReduceTensorIndices_t ReduceTensorIndices>
Status ReduceKernel<allow_multi_axes>::ComputeImpl(OpKernelContext* ctx, miopenReduceTensorOp_t miopen_reduce_op) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  std::vector<int64_t> axes;

  size_t num_inputs = ctx->InputCount();
  if (num_inputs == 2) {
    //override the attribute value with the input value for reduction_axes
    const Tensor* axes_tensor = ctx->Input<Tensor>(1);
    ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
    ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1, "An axes tensor must be a vector tensor.");
    auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
    const auto* data = axes_tensor->template Data<int64_t>();
    axes.assign(data, data + nDims);
  } else {
    axes.assign(axes_.begin(), axes_.end());
  }

  // empty axes and no-op
  if (axes.empty() && noop_with_empty_axes_) {
    auto* Y = ctx->Output(0, X->Shape());
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(Y->template MutableData<T>(), X->template Data<T>(), X->SizeInBytes(), hipMemcpyDeviceToDevice, Stream()));
    return Status::OK();
  }

  PrepareReduceMetadata prepare_reduce_metadata;
  ORT_RETURN_IF_ERROR(PrepareForReduce(X,
                                       keepdims_,
                                       axes,
                                       prepare_reduce_metadata));
  Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);
  const bool fast_reduction = fast_reduction_ && !ctx->GetUseDeterministicCompute();

  return ReduceComputeCore<T, ReduceTensorIndices>(*rocm_ep_, *X, prepare_reduce_metadata, *Y, miopen_reduce_op, axes,
                                                   calculate_log_, calculate_sqt_, log_sum_exp_, fast_reduction);
}

#define SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(T)                                                                              \
  template <>                                                                                                                \
  template <>                                                                                                                \
  Status ReduceKernel<true>::ComputeImpl<T, MIOPEN_REDUCE_TENSOR_NO_INDICES>(                                                \
      OpKernelContext * ctx, miopenReduceTensorOp_t miopen_reduce_op) const {                                                \
    typedef typename ToHipType<T>::MappedType HipT;                                                                          \
    const Tensor* X = ctx->Input<Tensor>(0);                                                                                 \
    std::vector<int64_t> axes;                                                                                               \
    size_t num_inputs = ctx->InputCount();                                                                                   \
    if (num_inputs == 2) {                                                                                                   \
      const Tensor* axes_tensor = ctx->Input<Tensor>(1);                                                                     \
      ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");                                                             \
      ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1, "An axes tensor must be a vector tensor.");                     \
      auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);                                                             \
      const auto* data = axes_tensor->template Data<int64_t>();                                                              \
      axes.assign(data, data + nDims);                                                                                       \
    } else {                                                                                                                 \
      axes.assign(axes_.begin(), axes_.end());                                                                               \
    }                                                                                                                        \
                                                                                                                             \
    if (axes.empty() && noop_with_empty_axes_) {                                                                             \
      auto* Y = ctx->Output(0, X->Shape());                                                                                  \
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(Y->template MutableData<T>(), X->template Data<T>(), X->SizeInBytes(),              \
                                         hipMemcpyDeviceToDevice, Stream()));                                                \
      return Status::OK();                                                                                                   \
    }                                                                                                                        \
                                                                                                                             \
    PrepareReduceMetadata prepare_reduce_metadata;                                                                           \
    ORT_RETURN_IF_ERROR(PrepareForReduce(X, keepdims_, axes, prepare_reduce_metadata));                                      \
                                                                                                                             \
    Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);                                                \
                                                                                                                             \
    int64_t input_count = prepare_reduce_metadata.input_count;                                                               \
    int64_t output_count = prepare_reduce_metadata.output_count;                                                             \
    std::vector<int64_t>& input_dims_miopen = prepare_reduce_metadata.input_dims_miopen;                                     \
    std::vector<int64_t>& output_dims_miopen = prepare_reduce_metadata.output_dims_miopen;                                   \
                                                                                                                             \
    if (input_count == 0) {                                                                                                  \
      assert(Y->Shape().Size() == 0);                                                                                        \
      return Status::OK();                                                                                                   \
    }                                                                                                                        \
                                                                                                                             \
    if (input_count == output_count) {                                                                                       \
      if (Y->template MutableData<T>() != X->template Data<T>()) {                                                           \
        HIP_RETURN_IF_ERROR(hipMemcpyAsync(Y->template MutableData<T>(), X->template Data<T>(),                              \
                                           input_count * sizeof(T), hipMemcpyDeviceToDevice, Stream()));                     \
      }                                                                                                                      \
      return Status::OK();                                                                                                   \
    }                                                                                                                        \
                                                                                                                             \
    HIP_RETURN_IF_ERROR(hipMemsetAsync(Y->MutableDataRaw(), 0, Y->SizeInBytes(), Stream()));                                 \
                                                                                                                             \
    size_t indices_bytes = 0;                                                                                                \
    size_t workspace_bytes = 0;                                                                                              \
    MiopenTensor input_tensor;                                                                                               \
    MiopenTensor output_tensor;                                                                                              \
    MiopenReduceDescriptor reduce_desc;                                                                                      \
                                                                                                                             \
    miopenDataType_t miopen_type_X = miopenFloat;                                                                            \
    IAllocatorUniquePtr<float> temp_X = GetScratchBuffer<float>(input_count);                                                \
    Impl_Cast<HipT, float>(Stream(), reinterpret_cast<const HipT*>(X->template Data<T>()), temp_X.get(), X->Shape().Size()); \
                                                                                                                             \
    ORT_RETURN_IF_ERROR(reduce_desc.Set(miopen_reduce_op, miopen_type_X, MIOPEN_REDUCE_TENSOR_NO_INDICES));                  \
    ORT_RETURN_IF_ERROR(input_tensor.Set(input_dims_miopen, miopen_type_X));                                                 \
    ORT_RETURN_IF_ERROR(output_tensor.Set(output_dims_miopen, miopen_type_X));                                               \
    MIOPEN_RETURN_IF_ERROR(                                                                                                  \
        miopenGetReductionIndicesSize(MiopenHandle(), reduce_desc, input_tensor, output_tensor, &indices_bytes));            \
    MIOPEN_RETURN_IF_ERROR(                                                                                                  \
        miopenGetReductionWorkspaceSize(MiopenHandle(), reduce_desc, input_tensor, output_tensor, &workspace_bytes));        \
    IAllocatorUniquePtr<uint32_t> indices_rocm = GetScratchBuffer<uint32_t>(indices_bytes);                                  \
    IAllocatorUniquePtr<HipT> workspace_rocm = GetScratchBuffer<HipT>(workspace_bytes);                                      \
                                                                                                                             \
    const auto one = Consts<float>::One;                                                                                     \
    const auto zero = Consts<float>::Zero;                                                                                   \
    auto temp_Y = GetScratchBuffer<float>(output_count);                                                                     \
    MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(MiopenHandle(), reduce_desc, indices_rocm.get(), indices_bytes,                \
                                              workspace_rocm.get(), workspace_bytes, &one, input_tensor, temp_X.get(),       \
                                              &zero, output_tensor, temp_Y.get()));                                          \
                                                                                                                             \
    Impl_Cast<float, HipT>(Stream(), temp_Y.get(), reinterpret_cast<HipT*>(Y->template MutableData<T>()), output_count);     \
                                                                                                                             \
    return Status::OK();                                                                                                     \
  }

SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(int32_t)
SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(int64_t)
SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(int8_t)
SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(uint8_t)

namespace ReductionOps {

template <typename T, miopenReduceTensorIndices_t ReduceTensorIndices>
Tensor ReduceCompute(ROCMExecutionProvider& rocm_ep, miopenReduceTensorOp_t miopen_reduce_op, AllocatorPtr allocator,
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

  status = ReduceComputeCore<T, ReduceTensorIndices>(rocm_ep, input, prepare_reduce_metadata, output, miopen_reduce_op, axes,
                                                     calculate_log, calculate_sqt, log_sum_exp, fast_reduction, input_shape_override);

  if (!status.IsOK()) {
    ORT_THROW(ONNXRUNTIME, FAIL, "Failed to perform reduce op: ", status.ErrorMessage());
  }

  return output;
}

// Explicit template instantiation (needed to be used in einsum_auxiliary_ops.cc)

template Tensor ReduceCompute<float, MIOPEN_REDUCE_TENSOR_NO_INDICES>(
    ROCMExecutionProvider& rocm_ep, miopenReduceTensorOp_t miopen_reduce_op,
    AllocatorPtr allocator,
    const Tensor& input, const std::vector<int64_t>& axes,
    bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
    bool fast_reduction, const TensorShape* input_shape_override);

// template Tensor ReduceCompute<double, MIOPEN_REDUCE_TENSOR_NO_INDICES>(
//     ROCMExecutionProvider& rocm_ep, miopenReduceTensorOp_t miopen_reduce_op,
//     AllocatorPtr allocator,
//     const Tensor& input, const std::vector<int64_t>& axes,
//     bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
//     bool fast_reduction, const TensorShape* input_shape_override);

template Tensor ReduceCompute<MLFloat16, MIOPEN_REDUCE_TENSOR_NO_INDICES>(
    ROCMExecutionProvider& rocm_ep, miopenReduceTensorOp_t miopen_reduce_op,
    AllocatorPtr allocator,
    const Tensor& input, const std::vector<int64_t>& axes,
    bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
    bool fast_reduction, const TensorShape* input_shape_override);

}  // namespace ReductionOps

#define REGISTER_KERNEL_HFD(name)        \
  REGISTER_KERNEL_TYPED(name, MLFloat16) \
  REGISTER_KERNEL_TYPED(name, float)
// REGISTER_KERNEL_TYPED(name, double)

#define REGISTER_KERNEL_HFD_11(name)        \
  REGISTER_KERNEL_TYPED_11(name, MLFloat16) \
  REGISTER_KERNEL_TYPED_11(name, float)
// REGISTER_KERNEL_TYPED_11(name, double)

REGISTER_KERNEL_HFD_11(ArgMax)
REGISTER_KERNEL_HFD_11(ArgMin)
REGISTER_KERNEL_HFD(ReduceL1)
REGISTER_KERNEL_HFD(ReduceL2)

REGISTER_KERNEL_TYPED_12(ReduceMax, MLFloat16)
REGISTER_KERNEL_TYPED_12(ReduceMax, float)
// REGISTER_KERNEL_TYPED_12(ReduceMax, double)
REGISTER_KERNEL_TYPED_12(ReduceMax, int32_t)
REGISTER_KERNEL_TYPED_12(ReduceMax, int64_t)
REGISTER_KERNEL_TYPED_12(ReduceMax, int8_t)
REGISTER_KERNEL_TYPED_12(ReduceMax, uint8_t)

REGISTER_KERNEL_HFD(ReduceMean)

REGISTER_KERNEL_TYPED_12(ReduceMin, MLFloat16)
REGISTER_KERNEL_TYPED_12(ReduceMin, float)
// REGISTER_KERNEL_TYPED_12(ReduceMin, double)
REGISTER_KERNEL_TYPED_12(ReduceMin, int32_t)
REGISTER_KERNEL_TYPED_12(ReduceMin, int8_t)
REGISTER_KERNEL_TYPED_12(ReduceMin, uint8_t)

REGISTER_KERNEL_HFD(ReduceProd)

REGISTER_KERNEL_TYPED_13(ReduceSum, MLFloat16)
REGISTER_KERNEL_TYPED_13(ReduceSum, float)
// REGISTER_KERNEL_TYPED_13(ReduceSum, double)
REGISTER_KERNEL_TYPED_13(ReduceSum, int32_t)
REGISTER_KERNEL_TYPED_13(ReduceSum, int64_t)

REGISTER_KERNEL_HFD(ReduceLogSum)
REGISTER_KERNEL_HFD(ReduceSumSquare)
REGISTER_KERNEL_HFD(ReduceLogSumExp)

#define REGISTER_KERNEL_INT32(name) \
  REGISTER_KERNEL_TYPED(name, int32_t)

REGISTER_KERNEL_INT32(ReduceL1)
REGISTER_KERNEL_INT32(ReduceL2)
REGISTER_KERNEL_INT32(ReduceMean)

REGISTER_KERNEL_INT32(ReduceProd)

}  // namespace rocm
}  // namespace onnxruntime
