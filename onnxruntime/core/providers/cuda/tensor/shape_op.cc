// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#ifndef BUILD_CUDA_EP_AS_PLUGIN
#include "core/providers/cpu/tensor/shape_op.h"
#endif
#include "core/providers/cuda/cuda_fwd.h"

#ifdef BUILD_CUDA_EP_AS_PLUGIN
#include <limits>
#endif

namespace onnxruntime {
namespace cuda {

#ifdef BUILD_CUDA_EP_AS_PLUGIN
// The bundled CUDA EP registers the CPU `onnxruntime::Shape` kernel (which
// derives from the framework `OpKernel`) and only marks its output as CPU
// memory. That class cannot be registered through the plugin EP's adapter
// kernel machinery, so the plugin build provides an adapter-based Shape kernel
// with identical semantics. Shape only reads the input's shape metadata (never
// its data) and writes the dims to a CPU output, so registering it on the CUDA
// EP keeps the node inside the device partition and avoids the device->host
// Memcpy node that the framework would otherwise insert to feed an isolated CPU
// Shape node -- a Memcpy that would prevent CUDA Graph capture. The output is
// still CPU memory, so a downstream device consumer may still need a copy; this
// removes the graph-breaking input-side Memcpy, it does not eliminate all copies.
class Shape final : public CudaKernel {
 public:
  explicit Shape(const OpKernelInfo& info) : CudaKernel(info) {
    info.GetAttrOrDefault<int64_t>("start", &start_index_, 0);

    if (start_index_ != 0) {
      // "start" is provided and is non-default (default is 0)
      needs_slicing_ = true;
    }

    if (info.GetAttr<int64_t>("end", &end_index_).IsOK()) {
      needs_slicing_ = true;
    }
  }

  // Takes a tensor as input and outputs a 1D int64 tensor (on CPU memory)
  // containing the shape of the input tensor.
  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* input = context->Input<Tensor>(0);
    const TensorShape& input_shape = input->Shape();

    int64_t rank = static_cast<int64_t>(input_shape.NumDimensions());

    if (!needs_slicing_) {  // vanilla use of Shape (no slicing)
      Tensor* output = context->Output(0, {rank});
      input_shape.CopyDims(output->MutableData<int64_t>(), static_cast<size_t>(rank));
    } else {  // slicing is needed
      int64_t true_start = start_index_;
      int64_t true_end = end_index_;

      // Deal with negative(s) and clamp
      true_start = true_start < 0 ? true_start + rank : true_start;
      true_start = true_start < 0 ? 0 : ((true_start > rank) ? rank : true_start);

      true_end = true_end < 0 ? true_end + rank : true_end;
      true_end = true_end < 0 ? 0 : ((true_end > rank) ? rank : true_end);

      auto slice_length = true_end - true_start;
      Tensor* output = context->Output(0, {slice_length < 0 ? 0 : slice_length});

      if (slice_length > 0) {
        input_shape.CopyDims(output->MutableData<int64_t>(),
                             static_cast<size_t>(true_start),
                             static_cast<size_t>(slice_length));
      }
    }

    return Status::OK();
  }

 private:
  bool needs_slicing_ = false;
  int64_t start_index_ = 0;
  int64_t end_index_ = std::numeric_limits<int64_t>::max();
};
#endif  // BUILD_CUDA_EP_AS_PLUGIN

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    1, 12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        // properly force CPU/GPU synch inside the kernel
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    13, 14,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        // properly force CPU/GPU synch inside the kernel
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    15, 18,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        // properly force CPU/GPU synch inside the kernel
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    19, 20,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        // properly force CPU/GPU synch inside the kernel
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypesIRv9())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    21, 22,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        // properly force CPU/GPU synch inside the kernel
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypesIRv9())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    23, 24,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        // properly force CPU/GPU synch inside the kernel
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypesIRv9())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_OPERATOR_KERNEL_EX(
    Shape,
    kOnnxDomain,
    25,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        // properly force CPU/GPU synch inside the kernel
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypesIRv9())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

}  // namespace cuda
}  // namespace onnxruntime
