// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_execution_provider.h"

#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#endif

#include "core/framework/compute_capability.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/fallback_cpu_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/function_utils.h"
#include "core/graph/indexed_sub_graph.h"

namespace onnxruntime {

namespace webgpu {
template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto* X = ctx->Input<Tensor>(0);
    Tensor* Y = ctx->Output(0, X->Shape());
    return Info().GetDataTransferManager().CopyTensor(*X, *Y);
  }
};

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPU, 0)
        .ExecQueueId(0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPU, 0)
        .ExecQueueId(1)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

std::unique_ptr<KernelRegistry> RegisterKernels() {
  auto kernel_registry = std::make_unique<onnxruntime::KernelRegistry>();

  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list becoming empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_THROW_IF_ERROR(kernel_registry->Register(std::move(info)));
    }
  }

#ifndef DISABLE_CONTRIB_OPS
  Status status = ::onnxruntime::contrib::webgpu::RegisterWebGpuContribKernels(*kernel_registry);
  ORT_ENFORCE(status.IsOK(), "Failed to register WebGPU contrib kernels: " + status.ErrorMessage());
#endif

  return kernel_registry;
}

}  // namespace webgpu

using namespace webgpu;

WebGpuExecutionProvider::WebGpuExecutionProvider()
    : IExecutionProvider{kWebGpuExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0)} {}

std::shared_ptr<KernelRegistry> WebGpuExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> registry = webgpu::RegisterKernels();

  return registry;
}

WebGpuExecutionProvider::~WebGpuExecutionProvider() {
}

}  // namespace onnxruntime
