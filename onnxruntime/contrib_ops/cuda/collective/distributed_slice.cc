// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Distributed computation.
#include "sharding.h"
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
DistributedSliice<T>::DistributedSliice(const OpKernelInfo& info) : DistributedKernel(info) {
}

template <typename T, typename Tind>
Status DistributedSliice<T>::ComputeInternal(OpKernelContext* context) const {
  const auto tensor_shard_data = context->Input<Tensor>(0);
  const auto tensor_shard_starts = context->Input<Tensor>(1);
  const auto tensor_shard_ends = context->Input<Tensor>(2);

  const TensorPartitionSpec& spec_data = input_shard_specs_[0];
  const TensorPartitionSpec& spec_starts = input_shard_specs_[1];
  const TensorPartitionSpec& spec_ends = input_shard_specs_[2];
  const TensorPartitionSpec& spec_Y = output_shard_specs_[0];

  if (spec_starts.HasShard() || spec_ends.HasShard())
    ORT_THROW("Not supported yet.");

  TensorShapeVector input_starts;
  TensorShapeVector input_ends;
  auto starts_data = tensor_shard_starts->DataAsSpan<Tind>();
  std::copy(starts_data.begin(), starts_data.end(), std::back_inserter(input_starts));
  auto ends_data = tensor_shard_ends->DataAsSpan<Tind>();
  std::copy(ends_data.begin(), ends_data.end(), std::back_inserter(input_ends));

  const auto tensor_shard_axes = context->Input<Tensor>(3);
  const TensorPartitionSpec& spec_axes = input_shard_specs_[3];

  TensorShapeVector input_axes;

  if (spec_axes.HasShard()){
    auto tmp_spec_axes = CreateAllReplica(spec_axes);
    auto tensor_axes = ReshardTensor(this, context, spec_axes, tmp_spec_axes, nccl_->Rank(), tensor_shard_axes);
    auto axes_data = tensor_axes->DataAsSpan<Tind>();
    std::copy(axes_data.begin(), axes_data.end(), std::back_inserter(input_axes));
  } else if (tensor_shard_axes){
    auto axes_data = tensor_shard_axes->DataAsSpan<Tind>();
    std::copy(axes_data.begin(), axes_data.end(), std::back_inserter(input_axes));
  }

  const auto tensor_shard_steps = context->Input<Tensor>(4);
  const TensorPartitionSpec& spec_steps = input_shard_specs_[4];
  if (spec_steps.HasShard())
    ORT_THROW("Not supported yet.");

  TensorShapeVector input_steps;
  auto steps_data = tensor_shard_steps->DataAsSpan<Tind>();
  std::copy(steps_data.begin(), steps_data.end(), std::back_inserter(input_steps));

  if (spec_data.GetPartitionAxis() != -1 &&
      std::find(input_axes.begin(), input_axes.end(), spec_data.GetPartitionAxis()) != input_axes.end()){
    // shard on slice axes, reshard first
    auto tmp_spec_data = CreateAllReplica(spec_data);
    auto tensor_data = ReshardTensor(this, context, spec_data, tmp_spec_data, nccl_->Rank(), tensor_shard_data);

    const auto& input_shape = tensor_data->Shape();
    const auto input_dimensions = input_shape.GetDims();
    if (input_dimensions.empty()) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot slice scalars");

    SliceOp::PrepareForComputeMetadata compute_metadata(input_dimensions);
    TensorShape output_shape(compute_metadata.output_dims_);

    ORT_RETURN_IF_ERROR(SliceBase::PrepareForCompute(input_starts, input_ends, input_axes, input_steps, compute_metadata));

    if (spec_Y.HasNoShard()){
      ORT_RETURN_IF_ERROR(FuncSlice(this,
                                    context,
                                    tensor_data.get(),
                                    input_starts,
                                    input_ends,
                                    input_axes,
                                    input_steps,
                                    context->Output(0, output_shape)));
    } else{
      ORT_THROW("Not Implemented yet.");
    }
  } else{
    if (spec_Y.GetPartitionAxis() == spec_data.GetPartitionAxis()){
      const auto& input_shape = tensor_data->Shape();
      const auto input_dimensions = input_shape.GetDims();
      if (input_dimensions.empty()) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot slice scalars");

      SliceOp::PrepareForComputeMetadata compute_metadata(input_dimensions);
      TensorShape output_shape(compute_metadata.output_dims_);

      ORT_RETURN_IF_ERROR(SliceBase::PrepareForCompute(input_starts, input_ends, input_axes, input_steps, compute_metadata));
      ORT_RETURN_IF_ERROR(FuncSlice(this,
                                    context,
                                    tensor_shard_data.get(),
                                    input_starts,
                                    input_ends,
                                    input_axes,
                                    input_steps,
                                    context->Output(0, output_shape)));
    } else{
      ORT_THROW("Not Implemented yet.")
    }
  }

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedSliice,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .AllocateInputsContiguously()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int64_t>()),
    DistributedSliice<float, int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedSliice,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .AllocateInputsContiguously()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int64_t>()),
    DistributedSliice<MLFloat16, int64_t>);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
