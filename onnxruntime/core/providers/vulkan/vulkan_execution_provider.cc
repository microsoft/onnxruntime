// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_execution_provider.h"

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

VulkanExecutionProvider::VulkanExecutionProvider() : IExecutionProvider{onnxruntime::kVulkanExecutionProvider} {
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

  // TODO: Should the device id be made configurable ?
  // Currently, we are picking the first one returned back in the enumeration
  vulkan_physical_device_ = vulkan_physical_devices[0];

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
}

VulkanExecutionProvider::~VulkanExecutionProvider() {
  // NOTES:
  // Physical device is implicitly destroyed when the vulkan instance is destroyed
  // finally, so there is nothing to add here wrt to that

  // Device queues are implicitly destroyed when logical device is destroyed

  if (VK_NULL_HANDLE != vulkan_logical_device_) {
    vkDestroyDevice(vulkan_logical_device_, nullptr);
    vulkan_logical_device_ = VK_NULL_HANDLE;
  }
}

}  // namespace onnxruntime
