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

template <bool allow_multi_axes>
template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
Status ReduceKernel<allow_multi_axes>::ComputeImpl(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnnReduceOp) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const Tensor* X = ctx->Input<Tensor>(0);
  ORT_ENFORCE(nullptr != X);
  const TensorShape input_shape{X->Shape()};
  const auto rank = input_shape.NumDimensions();

  if (rank > 8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "cuDNN only supports up to 8-D tensors in reduction");
  }

  const auto& input_dims = input_shape.GetDims();
  std::vector<int64_t> output_dims;
  std::vector<bool> reduced(rank, false);
  std::vector<int64_t> squeezed_output_dims;
  if (axes_.size() > 0) {
    output_dims = input_dims;
    for (auto reduced_axis : axes_) {
      const int64_t axis = HandleNegativeAxis(reduced_axis, rank);
      output_dims[axis] = 1;
      reduced[axis] = true;
    }
  } else {
    output_dims = std::vector<int64_t>(rank, 1);
  }

  if (keepdims_) {
    squeezed_output_dims = output_dims;
  } else {
    for (size_t i = 0; i < rank; ++i) {
      if (!reduced[i])
        squeezed_output_dims.push_back(input_dims[i]);
    }
  }

  Tensor* Y = ctx->Output(0, TensorShape(squeezed_output_dims));

  int64_t input_count = input_shape.Size();
  IAllocatorUniquePtr<float> temp_X;
  cudnnDataType_t cudnn_type_X = CudnnTensor::GetDataType<CudaT>();
  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_FLATTENED_INDICES && std::is_same<T, MLFloat16>::value) {
    // ArgMax/ArgMin with FP16 are not supported by cudnn, so convert input to fp32 then call cudnn
    temp_X = GetScratchBuffer<float>(input_count);
    cudnn_type_X = CUDNN_DATA_FLOAT;
    Impl_Cast<CudaT, float>(reinterpret_cast<const CudaT*>(X->template Data<T>()), temp_X.get(), X->Shape().Size());
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

  // need to allocate a separate buffer for ArgMin/ArgMax comparsion output
  auto output_count = Y->Shape().Size();

  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_NO_INDICES) {
    CudaT* input_data = nullptr;
    if (calculate_sqt_) {
      input_data = reinterpret_cast<CudaT*>(GetScratchBuffer<T>(input_count).get());
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
      auto exp_result = GetScratchBuffer<T>(input_count).get();
      auto log_sum_result = GetScratchBuffer<T>(output_count).get();
      BinaryElementwisePreparation prepare(this);
      prepare.BinaryElementwiseBroadcastPrepareHelper(0, input_shape, output_shape, input_shape);
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

}  // namespace cuda
}  // namespace onnxruntime
