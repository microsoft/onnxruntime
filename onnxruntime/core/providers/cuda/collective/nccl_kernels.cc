// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/collective/nccl_kernels.h"

namespace onnxruntime {
namespace cuda {

NcclAllReduce::NcclAllReduce(const OpKernelInfo& info) : NcclKernel(info) {
}

Status NcclAllReduce::ComputeInternal(OpKernelContext* context) const {
  cudaStream_t stream = nullptr;  // Default stream
  ncclComm_t comm = nccl_->Comm();

  for (int i = 0; i < context->InputCount(); i++) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    auto onnx_type = input_tensor->DataType();
    const void* input_data = input_tensor->DataRaw();
    size_t input_count = input_tensor->Shape().Size();

    Tensor* output_tensor = context->Output(i, input_tensor->Shape());
    void* output_data = output_tensor->MutableDataRaw();

    ncclDataType_t dtype = GetNcclDataType(onnx_type);
    NCCL_RETURN_IF_ERROR(ncclAllReduce(input_data, output_data, input_count, dtype, ncclSum, comm, stream));
  }

  return Status::OK();
}

NcclAllGather::NcclAllGather(const OpKernelInfo& info) : NcclKernel(info) {
}

Status NcclAllGather::ComputeInternal(OpKernelContext* context) const {
  cudaStream_t stream = nullptr;  // Default stream
  ncclComm_t comm = nccl_->Comm();
  const int rank = nccl_->Rank();
  const int size = nccl_->Size();

  for (int i = 0; i < context->InputCount(); i++) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    const void* input_data = input_tensor->DataRaw();
    TensorShape input_shape = input_tensor->Shape();
    auto onnx_type = input_tensor->DataType();
    ncclDataType_t dtype = GetNcclDataType(onnx_type);

    Tensor* output_tensor = context->Output(i, input_tensor->Shape());
    void* output_data = output_tensor->MutableDataRaw();

    // AllGather on as many elements as possible (NCCL requires all ranks receive the same amount of data).
    size_t allgather_count = AllGatherCount(input_shape);
    size_t allgather_bytes = allgather_count * onnx_type->Size();
    if (allgather_count > 0) {
      const void* input_data_at_offset = (const int8_t*)input_data + allgather_bytes * rank;
      NCCL_RETURN_IF_ERROR(ncclAllGather(input_data_at_offset, output_data, allgather_count, dtype, comm, stream));
    }

    // Broadcast the remaining elements to the last rank.
    size_t broadcast_count = BroadcastCount(input_shape);
    if (broadcast_count > 0) {
      const void* input_data_at_offset = (const int8_t*)input_data + allgather_bytes * size;
      void* output_data_at_offset = (int8_t*)output_data + allgather_bytes * size;
      NCCL_RETURN_IF_ERROR(ncclBroadcast(input_data_at_offset, output_data_at_offset, broadcast_count, dtype, size - 1, comm, stream));
    }
  }

  return Status::OK();
}

size_t NcclAllGather::AllGatherCount(const TensorShape& input_shape) const {
  if (input_shape.Size() == 0 || input_shape.NumDimensions() == 0)
    return 0;

  TensorShape allgather_shape = input_shape;
  allgather_shape[0] /= nccl_->Size();
  return allgather_shape.Size();
}

size_t NcclAllGather::BroadcastCount(const TensorShape& input_shape) const {
  if (input_shape.Size() == 0 || input_shape.NumDimensions() == 0)
    return 0;

  TensorShape broadcast_shape = input_shape;
  broadcast_shape[0] %= nccl_->Size();
  return broadcast_shape.Size();
}

NcclReduceScatter::NcclReduceScatter(const OpKernelInfo& info) : NcclKernel(info) {
}

Status NcclReduceScatter::ComputeInternal(OpKernelContext* context) const {
  cudaStream_t stream = nullptr;  // Default stream
  ncclComm_t comm = nccl_->Comm();
  const int size = nccl_->Size();

  for (int i = 0; i < context->InputCount(); i++) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    const void* input_data = input_tensor->DataRaw();
    TensorShape input_shape = input_tensor->Shape();
    auto onnx_type = input_tensor->DataType();
    ncclDataType_t dtype = GetNcclDataType(onnx_type);

    TensorShape output_shape = OutputShape(input_shape);
    Tensor* output_tensor = context->Output(i, output_shape);
    void* output_data = output_tensor->MutableDataRaw();

    // ReduceScatter as many elements as possible (NCCL requires all ranks receive the same amount of data).
    size_t reducescatter_count = ReduceScatterCount(input_shape);
    if (reducescatter_count > 0) {
      NCCL_RETURN_IF_ERROR(ncclReduceScatter(input_data, output_data, reducescatter_count, dtype, ncclSum, comm, stream));
    }

    // Reduce the remaining elements to the last rank.
    size_t reduce_count = ReduceCount(input_shape);
    if (reduce_count > 0) {
      size_t reducescatter_bytes = reducescatter_count * onnx_type->Size();
      const void* input_data_at_offset = (const int8_t*)input_data + reducescatter_bytes * size;
      void* output_data_at_offset = (int8_t*)output_data + reducescatter_bytes;
      NCCL_RETURN_IF_ERROR(ncclReduce(input_data_at_offset, output_data_at_offset, reduce_count, dtype, ncclSum, size - 1, comm, stream));
    }
  }

  return Status::OK();
}

TensorShape NcclReduceScatter::OutputShape(const TensorShape& input_shape) const {
  if (input_shape.Size() == 0 || input_shape.NumDimensions() == 0)
    return input_shape;

  const int rank = nccl_->Rank();
  const int size = nccl_->Size();

  TensorShape output_shape = input_shape;
  if (rank == size - 1) {
    output_shape[0] = input_shape[0] / size + input_shape[0] % size;
  } else {
    output_shape[0] = input_shape[0] / size;
  }

  return output_shape;
}

size_t NcclReduceScatter::ReduceScatterCount(const TensorShape& input_shape) const {
  if (input_shape.Size() == 0 || input_shape.NumDimensions() == 0)
    return 0;

  TensorShape reducescatter_shape = input_shape;
  reducescatter_shape[0] /= nccl_->Size();
  return reducescatter_shape.Size();
}

size_t NcclReduceScatter::ReduceCount(const TensorShape& input_shape) const {
  if (input_shape.Size() == 0 || input_shape.NumDimensions() == 0)
    return 0;

  TensorShape reduce_shape = input_shape;
  reduce_shape[0] %= nccl_->Size();
  return reduce_shape.Size();
}

static std::vector<std::pair<int, int>> AliasRange(int start, int end) {
  std::vector<std::pair<int, int>> aliases;
  for (int i = start; i < end; i++) {
    aliases.push_back(std::pair<int, int>(i, i));
  }
  return aliases;
}

ONNX_OPERATOR_KERNEL_EX(
    NcclAllReduce,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .Alias(AliasRange(0, 1024))
        .TypeConstraint("T", DataTypeImpl::AllFloatingPointTensorTypes()),
    NcclAllReduce);

ONNX_OPERATOR_KERNEL_EX(
    NcclAllGather,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .Alias(AliasRange(0, 1024))
        .TypeConstraint("T", DataTypeImpl::AllFloatingPointTensorTypes()),
    NcclAllGather);

ONNX_OPERATOR_KERNEL_EX(
    NcclReduceScatter,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFloatingPointTensorTypes()),
    NcclReduceScatter);

}  // namespace cuda
}  // namespace onnxruntime
