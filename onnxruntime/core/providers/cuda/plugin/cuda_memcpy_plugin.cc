// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Plugin-side MemcpyFromHost / MemcpyToHost kernels.
// These handle the common Tensor case using cudaMemcpyAsync directly.

#include "core/providers/cuda/plugin/cuda_kernel_adapter.h"

#include <cuda_runtime_api.h>

namespace onnxruntime {
namespace cuda {

class PluginMemcpy final : public CudaKernel {
 public:
  PluginMemcpy(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    const Tensor* X = ctx->Input<Tensor>(0);
    ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
    Tensor* Y = ctx->Output(0, X->Shape());
    ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor.");

    if (X->SizeInBytes() == 0) {
      return Status::OK();
    }

    const void* src = X->DataRaw();
    void* dst = Y->MutableDataRaw();
    if (src == dst) {
      return Status::OK();
    }

    // Determine copy direction from device placement.
    const auto& src_loc = X->Location();
    const auto& dst_loc = Y->Location();
    cudaMemcpyKind kind;
    if (src_loc.device.Type() == OrtDevice::CPU && dst_loc.device.Type() == OrtDevice::GPU) {
      kind = cudaMemcpyHostToDevice;
    } else if (src_loc.device.Type() == OrtDevice::GPU && dst_loc.device.Type() == OrtDevice::CPU) {
      kind = cudaMemcpyDeviceToHost;
    } else if (src_loc.device.Type() == OrtDevice::GPU && dst_loc.device.Type() == OrtDevice::GPU) {
      kind = cudaMemcpyDeviceToDevice;
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "PluginMemcpy: unsupported copy direction");
    }

    cudaStream_t stream = Stream(ctx);
    if (stream != nullptr) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst, src, X->SizeInBytes(), kind, stream));
    } else {
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst, src, X->SizeInBytes(), kind));
    }
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypesIRv9()),
    PluginMemcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypesIRv9()),
    PluginMemcpy);

}  // namespace cuda
}  // namespace onnxruntime
