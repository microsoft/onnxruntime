// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Distributed computation.
#include "distributed_squeeze.h"
#include "mpi_include.h"

// ORT system.
#include "core/providers/cuda/cuda_check_memory.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)
template <typename T, typename Tind>
DistributedSqueeze<T, Tind>::DistributedSqueeze(const OpKernelInfo& info) : DistributedKernel(info) {
}

template <typename T, typename Tind>
Status DistributedSqueeze<T, Tind>::ComputeInternal(OpKernelContext* context) const {
  auto input_tensor = context->Input<Tensor>(0);
  auto axes_tensor = context->Input<Tensor>(1);
  auto axes_span = axes_tensor->DataAsSpan<Tind>();

  const TensorPartitionSpec& input_spec = input_shard_specs_[0];
  const TensorPartitionSpec& axes_spec = input_shard_specs_[1];
  const TensorPartitionSpec& output_spec = output_shard_specs_[0];

  ORT_ENFORCE(axes_spec.HasNoShard(), "Axes tensor cannot be sharded.");

  // Non-negative collection of axes to drop.
  std::vector<Tind> axes;
  for (const auto axis : axes_span) {
    axes.push_back(axis >= 0 ? axis : axis + input_tensor->Shape().NumDimensions());
  }
  // Shape after dropping axes.
  auto dims = input_tensor->Shape().AsShapeVector();
  // Sort in descending order so that we can drop axes from the end.
  std::sort(axes.begin(), axes.end(), [](Tind a, Tind b) { return a > b; });
  for (const auto axis : axes) {
    ORT_ENFORCE(input_tensor->Shape()[axis] == 1, "Cannot squeeze non-singleton dimension.");
    dims.erase(dims.begin() + axis);
  }
  auto native_output_spec = TensorPartitionSpec::CreateByDropAxes(
      input_spec,
      axes);
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
    DistributedSqueeze,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    DistributedSqueeze<float, int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedSqueeze,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    DistributedSqueeze<MLFloat16, int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedSqueeze,
    kMSDomain,
    1,
    bool,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    DistributedSqueeze<bool, int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedSqueeze,
    kMSDomain,
    1,
    int8_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()),
    DistributedSqueeze<int8_t, int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedSqueeze,
    kMSDomain,
    1,
    uint8_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()),
    DistributedSqueeze<uint8_t, int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedSqueeze,
    kMSDomain,
    1,
    int64_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    DistributedSqueeze<int64_t, int64_t>);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
