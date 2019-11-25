// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reduction_ops.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cpu/tensor/utils.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(name, T)                                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      name,                                                                     \
      kOnnxDomain,                                                              \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

class CudnnReduceDescriptor final {
 public:
  CudnnReduceDescriptor() : desc_(nullptr) {
  }

  ~CudnnReduceDescriptor() {
    if (desc_ != nullptr) {
      cudnnDestroyReduceTensorDescriptor(desc_);
      desc_ = nullptr;
    }
  }

  Status Set(cudnnReduceTensorOp_t op, cudnnDataType_t type, cudnnReduceTensorIndices_t indices) {
    if (!desc_)
      CUDNN_RETURN_IF_ERROR(cudnnCreateReduceTensorDescriptor(&desc_));

    CUDNN_RETURN_IF_ERROR(cudnnSetReduceTensorDescriptor(
        desc_,
        op,
        type,
        CUDNN_PROPAGATE_NAN,
        indices,
        CUDNN_32BIT_INDICES));  // currently only the 32-bit (unsigned int) type is supported.
    return Status::OK();
  }

  operator cudnnReduceTensorDescriptor_t() const { return desc_; }

 private:
  cudnnReduceTensorDescriptor_t desc_;
};

static Status PrepareForReduce(OpKernelContext* ctx,
                               bool keepdims,
                               const std::vector<int64_t>& axes,
                               const Tensor** x_pp,
                               Tensor** y_pp,
                               int64_t& input_count,
                               int64_t& output_count,
                               std::vector<int64_t>& output_dims,
                               std::vector<int64_t>& input_dims_cudnn,
                               std::vector<int64_t>& output_dims_cudnn) {
  const Tensor* X = ctx->Input<Tensor>(0);
  ORT_ENFORCE(nullptr != X);
  *x_pp = X;

  const TensorShape input_shape{X->Shape()};
  const auto rank = input_shape.NumDimensions();
  input_count = input_shape.Size();

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
    for (size_t i = 0; i < rank; ++i) {
      if (!reduced[i])
        squeezed_output_dims.push_back(input_dims[i]);
    }
  } else {
    // 'axes' is empty and keepdims is false => we reduce on all axes AND drop all dims,
    // so the result is just a scalar, we keep 'squeezed_output_dims' empty (i.e.) no-op
  }

  Tensor* Y = ctx->Output(0, TensorShape(squeezed_output_dims));
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
template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
Status ReduceKernel<allow_multi_axes>::ComputeImpl(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnnReduceOp) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const Tensor* X = nullptr;
  Tensor* Y = nullptr;

  int64_t input_count = 0;
  int64_t output_count = 0;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> input_dims_cudnn;
  std::vector<int64_t> output_dims_cudnn;
  ORT_RETURN_IF_ERROR(PrepareForReduce(ctx,
                                       keepdims_,
                                       axes_,
                                       &X,
                                       &Y,
                                       input_count,
                                       output_count,
                                       output_dims,
                                       input_dims_cudnn,
                                       output_dims_cudnn));

  // special case when there is a dim value of 0 in the shape.
  if (input_count == 0) {
    assert(Y->Shape().Size() == 0);
    return Status::OK();
  }

  IAllocatorUniquePtr<float> temp_X;
  cudnnDataType_t cudnn_type_X = CudnnTensor::GetDataType<CudaT>();
  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_FLATTENED_INDICES && std::is_same<T, MLFloat16>::value) {
    // ArgMax/ArgMin with FP16 are not supported by cudnn, so convert input to fp32 then call cudnn
    temp_X = GetScratchBuffer<float>(input_count);
    cudnn_type_X = CUDNN_DATA_FLOAT;
    Impl_Cast<CudaT, float>(reinterpret_cast<const CudaT*>(X->template Data<T>()), temp_X.get(), X->Shape().Size());
  }

  CudnnReduceDescriptor reduce_desc;
  ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnnReduceOp, cudnn_type_X, ReduceTensorIndices));
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

  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_NO_INDICES) {
    IAllocatorUniquePtr<T> input_data_buffer(nullptr, [](T*) {});
    CudaT* input_data = nullptr;
    if (calculate_sqt_) {
      input_data_buffer = GetScratchBuffer<T>(input_count);
      input_data = reinterpret_cast<CudaT*>(input_data_buffer.get());
      fast_divmod tmp_div;
      Impl_Mul<CudaT>(static_cast<size_t>(SimpleBroadcast::NoBroadcast), nullptr,
                      reinterpret_cast<const CudaT*>(X->template Data<T>()), nullptr,
                      reinterpret_cast<const CudaT*>(X->template Data<T>()), nullptr,
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
          &one, input_tensor, reinterpret_cast<const CudaT*>(X->template Data<T>()),
          &zero, output_tensor, reinterpret_cast<CudaT*>(Y->template MutableData<T>())));

      // Exp(X-ReduceMax)
      const TensorShape output_shape(output_dims);
      auto exp_result_buffer = GetScratchBuffer<T>(input_count);
      auto exp_result = exp_result_buffer.get();
      auto log_sum_result_buffer = GetScratchBuffer<T>(output_count);
      auto log_sum_result = log_sum_result_buffer.get();
      BinaryElementwisePreparation prepare(this);
      prepare.BinaryElementwiseBroadcastPrepareHelper(X->Shape(), output_shape, X->Shape());
      prepare.CopyToGpu();
      Impl_Sub<CudaT>(prepare.output_rank_or_simple_broadcast,
                      prepare.lhs_padded_strides.GpuPtr(),
                      reinterpret_cast<const CudaT*>(X->template Data<T>()),
                      prepare.rhs_padded_strides.GpuPtr(),
                      reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
                      prepare.fdm_output_strides.GpuPtr(),
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
      Impl_Add<CudaT>(static_cast<size_t>(SimpleBroadcast::NoBroadcast), nullptr,
                      reinterpret_cast<CudaT*>(log_sum_result), nullptr,
                      reinterpret_cast<CudaT*>(Y->template MutableData<T>()), nullptr,
                      tmp_div, tmp_div,
                      reinterpret_cast<CudaT*>(Y->template MutableData<T>()), output_count);

      return Status::OK();
    }
    if (calculate_sqt_) {
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          CudnnHandle(), reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, input_data,
          &zero, output_tensor, reinterpret_cast<CudaT*>(Y->template MutableData<T>())));
    } else {
      // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      if (input_count == output_count) {
        if (Y->template MutableData<T>() != X->template Data<T>()) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y->template MutableData<T>(), X->template Data<T>(), input_count * sizeof(T), cudaMemcpyDeviceToDevice));
        }
      } else {
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            CudnnHandle(), reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, reinterpret_cast<const CudaT*>(X->template Data<T>()),
            &zero, output_tensor, reinterpret_cast<CudaT*>(Y->template MutableData<T>())));
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
          &one, input_tensor, reinterpret_cast<const CudaT*>(X->template Data<T>()),
          &zero, output_tensor, temp_output.get()));
    }

    // CUDA reduction index is uint32_t for now, cast it to int64_t according to ONNX spec
    Impl_Cast<uint32_t, int64_t>(reinterpret_cast<uint32_t*>(indices_cuda.get()), Y->template MutableData<int64_t>(), output_count);
  }

  if (calculate_log_) {
    Impl_Log<CudaT>(reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
                    reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
                    output_count);
  }

  return Status::OK();
}

template <>
template <>
Status ReduceKernel<true>::ComputeImpl<int32_t, CUDNN_REDUCE_TENSOR_NO_INDICES>(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnnReduceOp) const {
  typedef typename ToCudaType<int32_t>::MappedType CudaT;

  const Tensor* X = nullptr;
  Tensor* Y = nullptr;

  int64_t input_count = 0;
  int64_t output_count = 0;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> input_dims_cudnn;
  std::vector<int64_t> output_dims_cudnn;
  ORT_RETURN_IF_ERROR(PrepareForReduce(ctx,
                                       keepdims_,
                                       axes_,
                                       &X,
                                       &Y,
                                       input_count,
                                       output_count,
                                       output_dims,
                                       input_dims_cudnn,
                                       output_dims_cudnn));

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

  size_t indices_bytes = 0;
  size_t workspace_bytes = 0;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  CudnnReduceDescriptor reduce_desc;

  cudnnDataType_t cudnn_type_X = CUDNN_DATA_FLOAT;
  IAllocatorUniquePtr<float> temp_X = GetScratchBuffer<float>(input_count);
  Impl_Cast<CudaT, float>(reinterpret_cast<const CudaT*>(X->template Data<int32_t>()), temp_X.get(), X->Shape().Size());

  ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnnReduceOp, cudnn_type_X, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES));
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

#define REGISTER_KERNEL_HFD(name)        \
  REGISTER_KERNEL_TYPED(name, MLFloat16) \
  REGISTER_KERNEL_TYPED(name, float)     \
  REGISTER_KERNEL_TYPED(name, double)

REGISTER_KERNEL_HFD(ArgMax)
REGISTER_KERNEL_HFD(ArgMin)
REGISTER_KERNEL_HFD(ReduceL1)
REGISTER_KERNEL_HFD(ReduceL2)
REGISTER_KERNEL_HFD(ReduceMax)
REGISTER_KERNEL_HFD(ReduceMean)
REGISTER_KERNEL_HFD(ReduceMin)
REGISTER_KERNEL_HFD(ReduceProd)
REGISTER_KERNEL_HFD(ReduceSum)
REGISTER_KERNEL_HFD(ReduceLogSum)
REGISTER_KERNEL_HFD(ReduceSumSquare)
REGISTER_KERNEL_HFD(ReduceLogSumExp)

#define REGISTER_KERNEL_INT32(name) \
  REGISTER_KERNEL_TYPED(name, int32_t)

REGISTER_KERNEL_INT32(ReduceL1)
REGISTER_KERNEL_INT32(ReduceL2)
REGISTER_KERNEL_INT32(ReduceMax)
REGISTER_KERNEL_INT32(ReduceMean)
REGISTER_KERNEL_INT32(ReduceMin)
REGISTER_KERNEL_INT32(ReduceProd)
REGISTER_KERNEL_INT32(ReduceSum)

}  // namespace cuda
}  // namespace onnxruntime
