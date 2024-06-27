// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #include <array>
// #include <utility>

#include "core/common/logging/logging.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/memcpy.h"
#include "core/framework/op_kernel.h"

#include "core/providers/vulkan/vulkan_execution_provider.h"
#include "core/providers/vulkan/vulkan_allocator.h"
#include "core/providers/vulkan/vulkan_data_transfer.h"

// Add includes of kernel implementations
// #include "core/providers/vulkan/math/mul.h"

namespace onnxruntime {

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kVulkanExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kVulkanExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kVulkanExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kVulkanExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

Status RegisterVulkanKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kVulkanExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kVulkanExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,

  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> GetVulkanKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  ORT_THROW_IF_ERROR(RegisterVulkanKernels(*kernel_registry));
  return kernel_registry;
}

VulkanExecutionProvider::VulkanExecutionProvider(const VulkanExecutionProviderInfo& /*info*/)
    : IExecutionProvider(kVulkanExecutionProvider) {
  // TODO: Find device/queues/etc
}

VulkanExecutionProvider::~VulkanExecutionProvider() {
}

std::shared_ptr<KernelRegistry> VulkanExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = GetVulkanKernelRegistry();
  return kernel_registry;
}

std::vector<AllocatorPtr> VulkanExecutionProvider::CreatePreferredAllocators() {
  // TODO: Need pinned memory handling
  AllocatorCreationInfo device_memory_info(
      [](OrtDevice::DeviceId) {  // ignoring device id for now. may need to plugin the Vulkan device info somewhere
        return std::make_unique<vulkan::VulkanBufferAllocator>();
      });

  return std::vector<AllocatorPtr>{
      CreateAllocator(device_memory_info),  // this plugs in arena usage. assuming we want that.
  };
}

}  // namespace onnxruntime
