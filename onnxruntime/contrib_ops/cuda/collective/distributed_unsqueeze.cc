// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Distributed computation.
#include "distributed_unsqueeze.h"
#include "mpi_include.h"

// ORT system.
#include "core/providers/cuda/cuda_check_memory.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)
template <typename T, typename Tind>
DistributedUnsqueeze<T, Tind>::DistributedUnsqueeze(const OpKernelInfo& info) : DistributedKernel(info) {
}

template <typename T, typename Tind>
Status DistributedUnsqueeze<T, Tind>::ComputeInternal(OpKernelContext* context) const {
  auto input_tensor = context->Input<Tensor>(0);
  auto axes_tensor = context->Input<Tensor>(1);
  auto axes_span = axes_tensor->DataAsSpan<Tind>();

  const TensorPartitionSpec& input_spec = input_shard_specs_[0];
  const TensorPartitionSpec& axes_spec = input_shard_specs_[1];
  const TensorPartitionSpec& output_spec = output_shard_specs_[0];

  ORT_ENFORCE(axes_spec.HasNoShard(), "Axes tensor cannot be sharded.");

  std::vector<int64_t> axes(axes_span.begin(), axes_span.end());
  std::sort(axes.begin(), axes.end());
  auto dims = input_tensor->Shape().AsShapeVector();
  auto native_output_spec = input_spec;
  for (auto axis : axes) {
    if (axis < 0) {
      axis += input_tensor->Shape().NumDimensions() + 1;
    }
    dims.insert(dims.begin() + axis, 1);
    native_output_spec = TensorPartitionSpec::CreateByInsertOneAxis(
        native_output_spec,
        axis);
  }
  ORT_ENFORCE(
      output_spec == native_output_spec,
      "Re-sharding is required but not supported yet for this case.");
  auto output_tensor = context->Output(0, dims);
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
      output_tensor->MutableDataRaw(),
      input_tensor->DataRaw(),
      input_tensor->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream(context)));
  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedUnsqueeze,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    DistributedUnsqueeze<float, int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedUnsqueeze,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    DistributedUnsqueeze<MLFloat16, int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedUnsqueeze,
    kMSDomain,
    1,
    int64_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    DistributedUnsqueeze<int64_t, int64_t>);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
