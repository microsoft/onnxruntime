// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "xnnpack_execution_provider.h"
#include "core/framework/allocator.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "contrib_ops/cpu/cpu_contrib_kernels.h"
#include "xnnpack_fwd.h"

#include <xnnpack.h>

namespace onnxruntime {

constexpr const char* XNNPack = "XNNPack";
constexpr const char* XNNPack_Cpu = "XNNPackCpu";

// Forward declarations of op kernels
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kXNNPackExecutionProvider, kMSDomain, 1, QLinearConv);

namespace xnnpack_ep {
Status RegisterKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kXNNPackExecutionProvider, kMSDomain, 1, QLinearConv)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> GetKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  ORT_THROW_IF_ERROR(RegisterKernels(*kernel_registry));

  return kernel_registry;
}

}  // namespace xnnpack_ep

XNNPackExecutionProvider::XNNPackExecutionProvider()
    : IExecutionProvider{onnxruntime::kXNNPackExecutionProvider} {
  AllocatorCreationInfo default_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo(XNNPack, OrtAllocatorType::OrtDeviceAllocator));
      },
      0};

  InsertAllocator(CreateAllocator(default_memory_info));

  AllocatorCreationInfo cpu_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(XNNPack_Cpu, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      }};

  InsertAllocator(CreateAllocator(cpu_memory_info));

  ORT_ENFORCE(xnn_status_success == xnn_initialize(nullptr));
}

XNNPackExecutionProvider::~XNNPackExecutionProvider() {
}

std::shared_ptr<KernelRegistry> XNNPackExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::xnnpack_ep::GetKernelRegistry();
  return kernel_registry;
}

std::vector<std::unique_ptr<ComputeCapability>>
XNNPackExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                        const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputeCapability>>
      result = IExecutionProvider::GetCapability(graph, kernel_registries);

  return result;
}

}  // namespace onnxruntime
