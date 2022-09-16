// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_instance.h"

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

}  // namespace onnxruntime
