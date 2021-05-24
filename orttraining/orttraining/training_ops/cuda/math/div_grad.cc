// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/math/div_grad.h"
#include "orttraining/training_ops/cuda/math/div_grad_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#define DIVGRAD_REGISTER_KERNEL_TYPED(T)                                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      DivGrad,                                                                             \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DivGrad<T>);

DIVGRAD_REGISTER_KERNEL_TYPED(MLFloat16)
DIVGRAD_REGISTER_KERNEL_TYPED(float)
DIVGRAD_REGISTER_KERNEL_TYPED(double)

std::vector<int64_t> prepended_dimension_1(const TensorShape& shape, size_t total_rank) {
  size_t input_rank = shape.NumDimensions();
  if (input_rank == total_rank)
    return shape.GetDims();

  std::vector<int64_t> dims(total_rank, 1);

  // https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
  // for property 3 of Multidirectional Broadcasting, we need to prepended with a dimension of length 1.
  if (input_rank > 0)
    std::copy(shape.GetDims().begin(), shape.GetDims().end(), &dims[total_rank - input_rank]);
  return dims;
}

template <typename T>
Status DivGrad<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* dy_tensor = context->Input<Tensor>(0);
  const Tensor* a_tensor = context->Input<Tensor>(1);
  const Tensor* b_tensor = context->Input<Tensor>(2);
  const TensorShape& a_shape = a_tensor->Shape();
  const TensorShape& b_shape = b_tensor->Shape();
  const TensorShape& dy_shape = dy_tensor->Shape();

  // output shapes shall match its corresponding inputs
  Tensor* da_output_tensor = context->Output(0, a_shape);
  Tensor* db_output_tensor = context->Output(1, b_shape);
  if (!da_output_tensor && !db_output_tensor)
    return Status::OK();

  BinaryElementwisePreparation prepare;
  ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(a_tensor, b_tensor,
                                                        // TODO: BinaryElementwiseBroadcastPrepare shall take dy_tensor as const Tensor*.
                                                        const_cast<Tensor*>(dy_tensor), &prepare));
  const CudaT* prepare_a_data = reinterpret_cast<const CudaT*>(prepare.lhs_tensor->template Data<T>());
  const CudaT* prepare_b_data = reinterpret_cast<const CudaT*>(prepare.rhs_tensor->template Data<T>());
  const CudaT* prepare_dy_data = reinterpret_cast<const CudaT*>(prepare.output_tensor->template Data<T>());
  T* da_data = da_output_tensor ? da_output_tensor->template MutableData<T>() : nullptr;
  T* db_data = db_output_tensor ? db_output_tensor->template MutableData<T>() : nullptr;

  switch (prepare.output_rank_or_simple_broadcast) {
    case static_cast<int32_t>(SimpleBroadcast::NoBroadcast):
      ImplDivGradSimple<CudaT>(
          Stream(),
          SimpleBroadcast::NoBroadcast,
          prepare_a_data,
          prepare_b_data,
          prepare_dy_data,
          dy_shape.Size(),
          reinterpret_cast<CudaT*>(da_data),
          reinterpret_cast<CudaT*>(db_data));
      break;
    case static_cast<int32_t>(SimpleBroadcast::LeftScalar): {
      T* temp_da_data = nullptr;
      IAllocatorUniquePtr<T> temp_da_allocator;
      if (da_output_tensor) {
        temp_da_allocator = GetScratchBuffer<T>(dy_shape.Size());
        temp_da_data = temp_da_allocator.get();
      }

      ImplDivGradSimple<CudaT>(
          Stream(),
          SimpleBroadcast::LeftScalar,
          prepare_a_data,
          prepare_b_data,
          prepare_dy_data,
          dy_shape.Size(),
          reinterpret_cast<CudaT*>(temp_da_data),
          reinterpret_cast<CudaT*>(db_data));

      if (da_output_tensor) {
        std::vector<int64_t> a_output_dims = prepended_dimension_1(a_shape, dy_shape.NumDimensions());
        ReduceKernelShared<T, T, CUDNN_REDUCE_TENSOR_NO_INDICES>(
            temp_da_data,
            dy_shape,
            da_data,
            TensorShape({}),
            CUDNN_REDUCE_TENSOR_ADD,
            a_output_dims);
      }
      break;
    }
    case static_cast<int32_t>(SimpleBroadcast::RightScalar): {
      T* temp_db_data = nullptr;
      IAllocatorUniquePtr<T> temp_db_allocator;
      if (db_output_tensor) {
        temp_db_allocator = GetScratchBuffer<T>(dy_shape.Size());
        temp_db_data = temp_db_allocator.get();
      }
      ImplDivGradSimple<CudaT>(
          Stream(),
          SimpleBroadcast::RightScalar,
          prepare_a_data,
          prepare_b_data,
          prepare_dy_data,
          dy_shape.Size(),
          reinterpret_cast<CudaT*>(da_data),
          reinterpret_cast<CudaT*>(temp_db_data));

      if (db_output_tensor) {
        std::vector<int64_t> b_output_dims = prepended_dimension_1(b_shape, dy_shape.NumDimensions());
        ReduceKernelShared<T, T, CUDNN_REDUCE_TENSOR_NO_INDICES>(
            temp_db_data,
            dy_shape,
            db_data,
            TensorShape({}),
            CUDNN_REDUCE_TENSOR_ADD,
            b_output_dims);
      }
      break;
    }
    case static_cast<int32_t>(SimpleBroadcast::RightPerChannelBatch1):
    case static_cast<int32_t>(SimpleBroadcast::RightPerChannelBatchN): {
      T* temp_db_data = nullptr;
      IAllocatorUniquePtr<T> temp_db_allocator;
      if (db_output_tensor) {
        temp_db_allocator = GetScratchBuffer<T>(dy_shape.Size());
        temp_db_data = temp_db_allocator.get();
      }
      if (prepare.output_rank_or_simple_broadcast == static_cast<int32_t>(SimpleBroadcast::RightPerChannelBatch1)) {
        // lhs(1,C,H) and rhs (C,1)
        ImplDivGradRhsPerChannelBatch1<CudaT>(
            Stream(),
            prepare_a_data,
            prepare_b_data,
            prepare_dy_data,
            dy_shape.Size(),
            prepare.fdm_H,
            reinterpret_cast<CudaT*>(da_data),
            reinterpret_cast<CudaT*>(temp_db_data));
      } else {
        // lhs(N,C,H) and rhs (C,1)
        ImplDivGradRhsPerChannelBatchN<CudaT>(
            Stream(),
            prepare_a_data,
            prepare_b_data,
            prepare_dy_data,
            dy_shape.Size(),
            prepare.fdm_H,
            prepare.fdm_C,
            reinterpret_cast<CudaT*>(da_data),
            reinterpret_cast<CudaT*>(temp_db_data));
      }

      if (db_output_tensor) {
        std::vector<int64_t> b_output_dims = prepended_dimension_1(b_shape, dy_shape.NumDimensions());
        ReduceKernelShared<T, T, CUDNN_REDUCE_TENSOR_NO_INDICES>(
            temp_db_data,
            dy_shape,
            db_data,
            b_shape,
            CUDNN_REDUCE_TENSOR_ADD,
            b_output_dims);
      }
      break;
    }
    default: {
      bool need_reduce_da = da_output_tensor && a_shape.Size() != dy_shape.Size();
      bool need_reduce_db = db_output_tensor && b_shape.Size() != dy_shape.Size();
      IAllocatorUniquePtr<T> temp_da_allocator, temp_db_allocator;
      T* da_data_ref = nullptr;
      if (da_output_tensor)
        if (need_reduce_da) {
          temp_da_allocator = GetScratchBuffer<T>(dy_shape.Size());
          da_data_ref = temp_da_allocator.get();
        } else {
          da_data_ref = da_data;
        }
      T* db_data_ref = nullptr;
      if (db_output_tensor)
        if (need_reduce_db) {
          temp_db_allocator = GetScratchBuffer<T>(dy_shape.Size());
          db_data_ref = temp_db_allocator.get();
        } else {
          db_data_ref = db_data;
        }

      ImplDivGrad<CudaT>(
          Stream(),
          prepare.output_rank_or_simple_broadcast,
          &prepare.lhs_padded_strides,
          prepare_a_data,
          &prepare.rhs_padded_strides,
          prepare_b_data,
          prepare_dy_data,
          dy_shape.Size(),
          &prepare.fdm_output_strides,
          reinterpret_cast<CudaT*>(da_data_ref),
          reinterpret_cast<CudaT*>(db_data_ref));

      if (need_reduce_da) {
        std::vector<int64_t> a_output_dims = prepended_dimension_1(a_shape, dy_shape.NumDimensions());
        ReduceKernelShared<T, T, CUDNN_REDUCE_TENSOR_NO_INDICES>(
            da_data_ref,
            dy_shape,
            da_data,
            a_shape,
            CUDNN_REDUCE_TENSOR_ADD,
            a_output_dims);
      }

      if (need_reduce_db) {
        std::vector<int64_t> b_output_dims = prepended_dimension_1(b_shape, dy_shape.NumDimensions());
        ReduceKernelShared<T, T, CUDNN_REDUCE_TENSOR_NO_INDICES>(
            db_data_ref,
            dy_shape,
            db_data,
            b_shape,
            CUDNN_REDUCE_TENSOR_ADD,
            b_output_dims);
      }
    }
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
