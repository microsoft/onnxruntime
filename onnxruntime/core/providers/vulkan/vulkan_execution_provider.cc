// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_execution_provider.h"
#include "vulkan_data_transfer.h"

#include "core/common/common.h"
//#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"

namespace {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::KernelRegistry> kernel_registry = std::make_shared<onnxruntime::KernelRegistry>();
  onnxruntime::Status st;
};
}  // namespace

namespace onnxruntime {

VulkanInstance::VulkanInstance() {
  // A boilerplate struct to create the Vulkan engine instance
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Ort";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "Ort";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

#ifndef NDEBUG
  // TODO: Learn how to use this to setup the debugging framework
  const std::vector<const char*> validation_layers = {
      "VK_LAYER_KHRONOS_validation"};
#endif

  // Create the Vulkan engine instance - a boilerplate struct is
  // required to create the engine instance
  VkInstanceCreateInfo instanceCreateInfo{
      VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      nullptr,
      0,
      &appInfo,
#ifndef NDEBUG
      1,
      validation_layers.data(),
#else
      0,
      nullptr,
#endif
      0,
      nullptr,
  };

  VK_CALL(vkCreateInstance(&instanceCreateInfo, nullptr, &vulkan_instance_));
}

VulkanInstance::~VulkanInstance() {
  if (VK_NULL_HANDLE != vulkan_instance_) {
    vkDestroyInstance(vulkan_instance_, nullptr);
    vulkan_instance_ = VK_NULL_HANDLE;
  }
}

VkInstance VulkanInstance::Get() const {
  return vulkan_instance_;
}

AllocatorPtr VulkanExecutionProvider::CreateVulkanAllocator() {
  VulkanMemoryAllocationHelper memory_alloc_helper(vulkan_logical_device_, vulkan_device_memory_properties_, vulkan_queue_family_index_);
  AllocatorCreationInfo device_info{[&memory_alloc_helper, this](int) { return std::make_unique<VulkanAllocator>(info_, memory_alloc_helper, vulkan_device_properties_.limits); },
                                    info_.device_id, /*no arena*/ false};

  return CreateAllocator(device_info);
}

VulkanExecutionProvider::VulkanExecutionProvider(const VulkanExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kVulkanExecutionProvider}, info_(info) {
  vulkan_instance_ = std::make_shared<VulkanInstance>();

  // Look for physical devices that Vulkan can be used against
  // and ensure that it can perform compute (i.e.) it has a compute queue family

  uint32_t device_count = 0;
  VK_CALL(vkEnumeratePhysicalDevices(vulkan_instance_->Get(), &device_count, nullptr));

  if (device_count == 0) {
    ORT_THROW("No Vulkan compatible device found");
  }

  VkPhysicalDevice vulkan_physical_devices[1] = {nullptr};
  VK_CALL(vkEnumeratePhysicalDevices(vulkan_instance_->Get(), &device_count, vulkan_physical_devices));

  if (vulkan_physical_devices[0] != nullptr) {
    ORT_THROW("Received nullptr for Vulkan Physical device");
  }

  vulkan_physical_device_ = vulkan_physical_devices[info_.device_id];

  // Poll the physical device to see if it contains a compute queue family
  uint32_t total_queue_families = 0;

  VK_CALL_RETURNS_VOID(vkGetPhysicalDeviceQueueFamilyProperties(vulkan_physical_device_, &total_queue_families, nullptr));

  if (total_queue_families == 0) {
    ORT_THROW("No queue families on the device at all !");
  }

  std::vector<VkQueueFamilyProperties> queue_family_properties(total_queue_families);

  VK_CALL_RETURNS_VOID(vkGetPhysicalDeviceQueueFamilyProperties(vulkan_physical_device_, &total_queue_families, queue_family_properties.data()));

  for (vulkan_queue_family_index_ = 0; vulkan_queue_family_index_ < total_queue_families; vulkan_queue_family_index_++) {
    if (queue_family_properties[vulkan_queue_family_index_].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      break;
    }
  }

  if (vulkan_queue_family_index_ == total_queue_families) {
    ORT_THROW("The Vulkan device does not have a compute queue family !");
  }

  // Create a logical device (vulkan device)
  // There is only one queue and its priority is going to be 1.0
  float queue_priorities[] = {
      1.0f,
  };

  VkDeviceQueueCreateInfo queue_create_info{
      VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      nullptr,
      0,
      vulkan_queue_family_index_,
      1,
      queue_priorities,
  };

  VkPhysicalDeviceFeatures vulkan_device_features;
  ::memset(&vulkan_device_features, 0, sizeof(vulkan_device_features));

  // TODO: Understand the significance of this
  vulkan_device_features.shaderStorageImageWriteWithoutFormat = VK_TRUE;

  VkDeviceCreateInfo vulkan_device_creation_info{
      VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      nullptr,
      0,
      1,
      &queue_create_info,
      0,
      nullptr,
      static_cast<uint32_t>(0),  // TODO: Should device extensions be 0 ?
      nullptr,                   // TODO: Should device extensions be 0 ?
      &vulkan_device_features,
  };

  VK_CALL(vkCreateDevice(vulkan_physical_device_, &vulkan_device_creation_info,
                         nullptr, &vulkan_logical_device_));
  VK_CALL_RETURNS_VOID(vkGetPhysicalDeviceProperties(vulkan_physical_device_, &vulkan_device_properties_));
  VK_CALL_RETURNS_VOID(vkGetPhysicalDeviceMemoryProperties(vulkan_physical_device_, &vulkan_device_memory_properties_));
  VK_CALL_RETURNS_VOID(vkGetDeviceQueue(vulkan_logical_device_, vulkan_queue_family_index_, 0, &vulkan_queue_));

  // Insert the allocator
  InsertAllocator(CreateVulkanAllocator());

  // Initialize other resources
  vulkan_sampler_ = std::make_shared<VulkanSampler>(vulkan_logical_device_, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
  vulkan_clamp_sampler_ = std::make_shared<VulkanSampler>(vulkan_logical_device_, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
  vulkan_memory_alloc_ = std::make_shared<VulkanMemoryAllocationHelper>(vulkan_logical_device_, vulkan_device_memory_properties_, vulkan_queue_family_index_);
  vulkan_command_pool_ = std::make_shared<VulkanCommandPool>(vulkan_logical_device_, vulkan_queue_family_index_);
  vulkan_pipeline_factory_ = std::make_shared<VulkanPipelineFactory>(vulkan_logical_device_);
}

VulkanExecutionProvider::~VulkanExecutionProvider() {
  vulkan_instance_ = nullptr;
  vulkan_sampler_ = nullptr;
  vulkan_clamp_sampler_ = nullptr;
  vulkan_memory_alloc_ = nullptr;
  vulkan_command_pool_ = nullptr;
  vulkan_pipeline_factory_ = nullptr;

  // NOTE:
  // (1) Physical device is implicitly destroyed when the Vulkan instance is destroyed
  // finally, so there is nothing to add here wrt to that

  // (2) Device queues are implicitly destroyed when logical device is destroyed
  if (VK_NULL_HANDLE != vulkan_logical_device_) {
    vkDestroyDevice(vulkan_logical_device_, nullptr);
    vulkan_logical_device_ = VK_NULL_HANDLE;
  }
}

void VulkanExecutionProvider::RegisterAllocator(AllocatorManager& allocator_manager) {
  OrtDevice vulkan_device{OrtDevice::GPU, OrtDevice::MemType::DEFAULT, info_.device_id};

  auto vulkan_alloc = GetAllocator(vulkan_device.Id(), OrtMemTypeDefault);

  if (!vulkan_alloc) {
    vulkan_alloc = allocator_manager.GetAllocator(OrtMemTypeDefault, vulkan_device);

    if (!vulkan_alloc) {
      vulkan_alloc = CreateVulkanAllocator();

      allocator_manager.InsertAllocator(vulkan_alloc);
    }
  } else {
    // enable sharing of our allocator from the EP
    allocator_manager.InsertAllocator(vulkan_alloc);
  }
}

const VulkanSampler& VulkanExecutionProvider::GetCommonSampler(bool clamp) const {
  if (clamp) {
    return *vulkan_clamp_sampler_;
  }

  return *vulkan_sampler_;
}

VulkanMemoryAllocationHelper& VulkanExecutionProvider::GetMemoryPool() const {
  return *vulkan_memory_alloc_;
}

VulkanCommandPool& VulkanExecutionProvider::GetCommandPool() const {
  return *vulkan_command_pool_;
}

VulkanPipeline& VulkanExecutionProvider::GetPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                                                     const std::vector<uint32_t>& local_sizes) const {
  return *vulkan_pipeline_factory_->GetPipeline(key, types, local_sizes);
}

const VkPhysicalDeviceLimits& VulkanExecutionProvider::GetMemoryLimits() const {
  return vulkan_device_properties_.limits;
}

std::unique_ptr<IDataTransfer> VulkanExecutionProvider::GetDataTransfer() const {
  return std::make_unique<VulkanDataTransfer>(*this);
}

Status VulkanExecutionProvider::QueueCommand(VkCommandBuffer cmd_buffer) const {
  vulkan_command_buffers_.emplace_back(cmd_buffer);
  return Status::OK();
}

Status VulkanExecutionProvider::Sync() const {
  return Status::OK();
}

// Forward declarations of op kernels
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kVulkanExecutionProvider, kOnnxDomain, 6, 12, float, Relu);

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  return {};
}

Status RegisterVulkanOnnxOperatorKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kVulkanExecutionProvider, kOnnxDomain, 6, 12, float, Relu)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

Status RegisterVulkanKernels(KernelRegistry& kernel_registry) {
  return RegisterVulkanOnnxOperatorKernels(kernel_registry);
}

KernelRegistryAndStatus GetVulkanKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = RegisterVulkanKernels(*ret.kernel_registry);
  return ret;
}

std::shared_ptr<KernelRegistry> VulkanExecutionProvider::GetKernelRegistry() const {
  static KernelRegistryAndStatus k = GetVulkanKernelRegistry();
  // throw if the registry failed to initialize
  ORT_THROW_IF_ERROR(k.st);
  return k.kernel_registry;
}
}  // namespace onnxruntime
