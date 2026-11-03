#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  VkInstance instance = VK_NULL_HANDLE;
  try {
    if (argc != 2) throw std::runtime_error("Usage: vulkan_uuid_probe expected-NVIDIA-UUID");
    const VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "physical-device-only",
                                       1, nullptr, 0, VK_API_VERSION_1_1};
    const VkInstanceCreateInfo descriptor{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0,
                                         &application, 0, nullptr, 0, nullptr};
    if (vkCreateInstance(&descriptor, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateInstance failed");
    }
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count != 1) {
      throw std::runtime_error("Require exactly one exposed Vulkan physical device, got" + std::to_string(count));
    }
    VkPhysicalDevice device;
    if (vkEnumeratePhysicalDevices(instance, &count, &device) != VK_SUCCESS) {
      throw std::runtime_error("vkEnumeratePhysicalDevices failed");
    }
    VkPhysicalDeviceIDProperties id{};
    id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &id;
    vkGetPhysicalDeviceProperties2(device, &properties);
    char text[37];
    std::snprintf(text, sizeof(text),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  id.deviceUUID[0], id.deviceUUID[1], id.deviceUUID[2], id.deviceUUID[3],
                  id.deviceUUID[4], id.deviceUUID[5], id.deviceUUID[6], id.deviceUUID[7],
                  id.deviceUUID[8], id.deviceUUID[9], id.deviceUUID[10], id.deviceUUID[11],
                  id.deviceUUID[12], id.deviceUUID[13], id.deviceUUID[14], id.deviceUUID[15]);
    const std::string uuid = std::string("GPU-") + text;
    std::cout << "VULKAN_PHYSICAL_DEVICE uuid=" << uuid
              << " name=" << properties.properties.deviceName << std::endl;
    if (uuid != argv[1]) throw std::runtime_error("Exposed Vulkan UUID is not the reserved GPU");
    uint32_t extension_count = 0;
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr) != VK_SUCCESS) {
      throw std::runtime_error("Cannot count device extensions");
    }
    std::vector<VkExtensionProperties> extensions(extension_count);
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, extensions.data()) != VK_SUCCESS) {
      throw std::runtime_error("Cannot enumerate device extensions");
    }
    for (const char* requested : {"VK_EXT_shader_float8", "VK_KHR_cooperative_matrix", "VK_NV_cooperative_vector"}) {
      bool found = false;
      for (const auto& extension : extensions) found |= std::strcmp(extension.extensionName, requested) == 0;
      std::cout << requested << "=" << (found ? "present" : "absent") << std::endl;
    }
    vkDestroyInstance(instance, nullptr);
    std::cout << "PASS physical-device isolation; no logical device or GPU work was created" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    if (instance) vkDestroyInstance(instance, nullptr);
    std::cerr << "FAIL: " << error.what() << std::endl;
    return 1;
  }
}
