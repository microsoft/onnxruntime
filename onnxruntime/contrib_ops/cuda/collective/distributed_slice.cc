// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Distributed computation.
#include "distributed_slice.h"
#include "mpi_include.h"

// ORT system.
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cuda/tensor/slice.h"
#include "core/providers/cuda/math/matmul.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cuda/cuda_check_memory.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)
template <typename T, typename Tind>
DistributedSlice<T, Tind>::DistributedSlice(const OpKernelInfo& info) : DistributedKernel(info) {
}

template <typename T, typename Tind>
Status DistributedSlice<T, Tind>::ComputeInternal(OpKernelContext* context) const {
  const auto tensor_shard_data = context->Input<Tensor>(0);
  const auto tensor_shard_starts = context->Input<Tensor>(1);
  const auto tensor_shard_ends = context->Input<Tensor>(2);

  const TensorPartitionSpec& spec_data = input_shard_specs_[0];
  const TensorPartitionSpec& spec_starts = input_shard_specs_[1];
  const TensorPartitionSpec& spec_ends = input_shard_specs_[2];
  const TensorPartitionSpec& spec_Y = output_shard_specs_[0];

  const auto tensor_shard_axes = context->Input<Tensor>(3);
  const TensorPartitionSpec& spec_axes = input_shard_specs_[3];

  if (spec_starts.HasShard() ||
      spec_ends.HasShard() ||
      spec_axes.HasShard() ||
      (input_shard_specs_.size() > 4 && input_shard_specs_[4].HasShard()))
    ORT_THROW("DistributedSlice: shard on starts / ends / axes / steps are not supported yet.");

  std::vector<int64_t> input_starts;
  std::vector<int64_t> input_ends;
  auto starts_data = tensor_shard_starts->DataAsSpan<Tind>();
  input_starts.resize(starts_data.size());
  std::copy(starts_data.begin(), starts_data.end(), input_starts.begin());
  auto ends_data = tensor_shard_ends->DataAsSpan<Tind>();
  input_ends.resize(ends_data.size());
  std::copy(ends_data.begin(), ends_data.end(), input_ends.begin());

  std::vector<int64_t> input_axes;
  if (tensor_shard_axes) {
    auto axes_data = tensor_shard_axes->DataAsSpan<Tind>();
    input_axes.resize(axes_data.size());
    std::copy(axes_data.begin(), axes_data.end(), input_axes.begin());
  }

  std::vector<int64_t> input_steps;
  const auto tensor_shard_steps = context->Input<Tensor>(4);
  if (tensor_shard_steps) {
    const TensorPartitionSpec& spec_steps = input_shard_specs_[4];
    if (spec_steps.HasShard())
      ORT_THROW("Not supported yet.");

    auto steps_data = tensor_shard_steps->DataAsSpan<Tind>();
    input_steps.resize(steps_data.size());
    std::copy(steps_data.begin(), steps_data.end(), input_steps.begin());
  }

  if (spec_data.GetPartitionAxis() != -1 &&
      std::find(input_axes.begin(), input_axes.end(), spec_data.GetPartitionAxis()) != input_axes.end()) {
    // shard on slice axes, reshard first
    auto tmp_spec_data = TensorPartitionSpec::CreateAllReplica(spec_data);
    auto tensor_data = ReshardTensor(this, context, spec_data, tmp_spec_data, nccl_->Rank(), tensor_shard_data);

    const auto& input_shape = tensor_data->Shape();
    const auto input_dimensions = input_shape.GetDims();
    if (input_dimensions.empty()) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot slice scalars");

    SliceOp::PrepareForComputeMetadata compute_metadata(input_dimensions);
    ORT_RETURN_IF_ERROR(SliceBase::PrepareForCompute(input_starts, input_ends, input_axes, input_steps, compute_metadata));
    TensorShape output_shape(compute_metadata.output_dims_);

    if (spec_Y.HasNoShard()) {
      ORT_RETURN_IF_ERROR(FuncSlice(this,
                                    context,
                                    tensor_data.get(),
                                    input_starts,
                                    input_ends,
                                    input_axes,
                                    input_steps,
                                    context->Output(0, output_shape)));
    } else {
      AllocatorPtr alloc;
      ORT_ENFORCE(context->GetTempSpaceAllocator(&alloc) == Status::OK());
      auto dst_tensor = Tensor::Create(tensor_data->DataType(), output_shape, alloc);
      ORT_RETURN_IF_ERROR(FuncSlice(this,
                                    context,
                                    tensor_data.get(),
                                    input_starts,
                                    input_ends,
                                    input_axes,
                                    input_steps,
                                    dst_tensor.get()));
      auto tmp_spec_output = TensorPartitionSpec::CreateAllReplica(spec_Y);
      ReshardTensor(this, context, tmp_spec_output, spec_Y, nccl_->Rank(), dst_tensor.get(), 0);
    }
  } else {
    const auto& input_shape = tensor_shard_data->Shape();
    const auto input_dimensions = input_shape.GetDims();
    if (input_dimensions.empty()) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot slice scalars");

    SliceOp::PrepareForComputeMetadata compute_metadata(input_dimensions);
    ORT_RETURN_IF_ERROR(SliceBase::PrepareForCompute(input_starts, input_ends, input_axes, input_steps, compute_metadata));
    TensorShape output_shape(compute_metadata.output_dims_);

    if (spec_Y.GetPartitionAxis() == spec_data.GetPartitionAxis()) {
      ORT_RETURN_IF_ERROR(FuncSlice(this,
                                    context,
                                    tensor_shard_data,
                                    input_starts,
                                    input_ends,
                                    input_axes,
                                    input_steps,
                                    context->Output(0, output_shape)));
    } else {
      AllocatorPtr alloc;
      ORT_ENFORCE(context->GetTempSpaceAllocator(&alloc) == Status::OK());
      auto dst_tensor = Tensor::Create(tensor_shard_data->DataType(), output_shape, alloc);
      ORT_RETURN_IF_ERROR(FuncSlice(this,
                                    context,
                                    tensor_shard_data,
                                    input_starts,
                                    input_ends,
                                    input_axes,
                                    input_steps,
                                    dst_tensor.get()));
      ReshardTensor(this, context, spec_data, spec_Y, nccl_->Rank(), dst_tensor.get(), 0);
    }
  }

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedSlice,
    kMSDomain,
    1,
    bool,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .InputMemoryType(OrtMemTypeCPUInput, 4)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int64_t>()),
    DistributedSlice<bool, int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedSlice,
    kMSDomain,
    1,
    int64_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .InputMemoryType(OrtMemTypeCPUInput, 4)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int64_t>()),
    DistributedSlice<int64_t, int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedSlice,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .InputMemoryType(OrtMemTypeCPUInput, 4)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int64_t>()),
    DistributedSlice<float, int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedSlice,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .InputMemoryType(OrtMemTypeCPUInput, 4)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int64_t>()),
    DistributedSlice<MLFloat16, int64_t>);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
