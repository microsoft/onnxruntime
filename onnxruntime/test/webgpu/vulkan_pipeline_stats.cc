// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Compiler-statistics interposer only; pipeline capture changes compilation and timing.
#include <vulkan/vulkan.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

static PFN_vkGetInstanceProcAddr real_gipa;
static PFN_vkGetDeviceProcAddr real_gdpa;
static PFN_vkCreateDevice real_create_device;
static std::mutex output_mutex;
static unsigned pipeline_index;

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance, const char*);
extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice, const char*);

static std::ofstream Output(const std::string& suffix) {
  const char* prefix = std::getenv("PIPELINE_STATS_PREFIX");
  if (!prefix || !*prefix) {
    std::fprintf(stderr, "PIPELINE_STATS_PREFIX must be set.\n");
    std::abort();
  }
  std::ofstream output(std::string(prefix) + suffix, std::ios::app);
  if (!output) {
    std::fprintf(stderr, "Cannot open pipeline-statistics output: %s%s\n", prefix, suffix.c_str());
    std::abort();
  }
  return output;
}

static void Capture(VkDevice device, VkPipeline pipeline, VkShaderModule shader) {
  auto get_properties = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
      real_gdpa(device, "vkGetPipelineExecutablePropertiesKHR"));
  auto get_statistics = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
      real_gdpa(device, "vkGetPipelineExecutableStatisticsKHR"));
  auto get_representations = reinterpret_cast<PFN_vkGetPipelineExecutableInternalRepresentationsKHR>(
      real_gdpa(device, "vkGetPipelineExecutableInternalRepresentationsKHR"));
  std::lock_guard<std::mutex> lock(output_mutex);
  auto output = Output(".txt");
  const unsigned index = pipeline_index++;
  output << "PIPELINE " << index << " handle=" << pipeline << " shader=" << shader << "\n";
  if (!get_properties || !get_statistics || !get_representations) {
    output << "Executable query functions unavailable.\n";
    return;
  }
  VkPipelineInfoKHR info{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, nullptr, pipeline};
  uint32_t count = 0;
  VkResult result = get_properties(device, &info, &count, nullptr);
  output << "properties_result=" << result << " count=" << count << "\n";
  if (result != VK_SUCCESS) {
    return;
  }
  std::vector<VkPipelineExecutablePropertiesKHR> properties(count);
  for (auto& value : properties) {
    value.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
  }
  result = get_properties(device, &info, &count, properties.data());
  if (result != VK_SUCCESS) {
    output << "properties_error=" << result << "\n";
    return;
  }
  for (uint32_t executable = 0; executable < count; ++executable) {
    const auto& property = properties[executable];
    output << "EXECUTABLE " << property.name << " subgroup=" << property.subgroupSize
           << " description=" << property.description << "\n";
    VkPipelineExecutableInfoKHR executable_info{
        VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, nullptr, pipeline, executable};
    uint32_t num_stats = 0;
    result = get_statistics(device, &executable_info, &num_stats, nullptr);
    output << "statistics_result=" << result << " count=" << num_stats << "\n";
    if (result == VK_SUCCESS) {
      std::vector<VkPipelineExecutableStatisticKHR> stats(num_stats);
      for (auto& value : stats) {
        value.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
      }
      result = get_statistics(device, &executable_info, &num_stats, stats.data());
      if (result != VK_SUCCESS) {
        output << "statistics_error=" << result << "\n";
      } else {
        for (const auto& stat : stats) {
          output << "STAT " << stat.name << " format=" << stat.format << " = ";
          switch (stat.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
              output << stat.value.b32;
              break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
              output << stat.value.i64;
              break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
              output << stat.value.u64;
              break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
              output << stat.value.f64;
              break;
            default:
              output << "unknown format";
              break;
          }
          output << " raw_hex=" << std::hex << stat.value.u64 << std::dec
                 << " description=" << stat.description << "\n";
        }
      }
    }
    uint32_t num_reps = 0;
    result = get_representations(device, &executable_info, &num_reps, nullptr);
    output << "representations_result=" << result << " count=" << num_reps << "\n";
    if (result != VK_SUCCESS || num_reps == 0) {
      continue;
    }
    std::vector<VkPipelineExecutableInternalRepresentationKHR> reps(num_reps);
    for (auto& value : reps) {
      value.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
    }
    result = get_representations(device, &executable_info, &num_reps, reps.data());
    if (result != VK_SUCCESS) {
      output << "representations_error=" << result << "\n";
      continue;
    }
    std::vector<std::vector<char>> buffers(num_reps);
    for (uint32_t i = 0; i < num_reps; ++i) {
      buffers[i].resize(reps[i].dataSize);
      reps[i].pData = buffers[i].data();
    }
    result = get_representations(device, &executable_info, &num_reps, reps.data());
    output << "representations_data_result=" << result << "\n";
    if (result == VK_SUCCESS) {
      for (uint32_t i = 0; i < num_reps; ++i) {
        output << "REP " << reps[i].name << " bytes=" << reps[i].dataSize
               << " text=" << reps[i].isText << " description=" << reps[i].description << "\n";
        std::string suffix = "-" + std::to_string(index) + "-" + std::to_string(i) +
                             (reps[i].isText ? ".asm" : ".bin");
        auto dump = Output(suffix);
        dump.write(buffers[i].data(), reps[i].dataSize);
        if (!dump) {
          std::fprintf(stderr, "Cannot write pipeline representation.\n");
          std::abort();
        }
      }
    }
  }
}

static VkResult VKAPI_CALL CreatePipelines(
    VkDevice device, VkPipelineCache cache, uint32_t count, const VkComputePipelineCreateInfo* infos,
    const VkAllocationCallbacks* allocator, VkPipeline* pipelines) {
  auto real = reinterpret_cast<PFN_vkCreateComputePipelines>(real_gdpa(device, "vkCreateComputePipelines"));
  std::vector<VkComputePipelineCreateInfo> capture(infos, infos + count);
  for (auto& info : capture) {
    info.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  }
  const VkResult result = real(device, cache, count, capture.data(), allocator, pipelines);
  if (result == VK_SUCCESS) {
    for (uint32_t i = 0; i < count; ++i) {
      Capture(device, pipelines[i], infos[i].stage.module);
    }
  }
  return result;
}

static VkResult VKAPI_CALL SetName(VkDevice device, const VkDebugUtilsObjectNameInfoEXT* info) {
  {
    std::lock_guard<std::mutex> lock(output_mutex);
    auto output = Output(".txt");
    output << "NAME type=" << info->objectType << " handle=" << reinterpret_cast<void*>(info->objectHandle)
           << " name=" << (info->pObjectName ? info->pObjectName : "") << "\n";
  }
  auto real = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(real_gdpa(device, "vkSetDebugUtilsObjectNameEXT"));
  return real(device, info);
}

static VkResult VKAPI_CALL CreateDevice(
    VkPhysicalDevice physical, const VkDeviceCreateInfo* info,
    const VkAllocationCallbacks* allocator, VkDevice* device) {
  VkDeviceCreateInfo capture = *info;
  std::vector<const char*> extensions(info->ppEnabledExtensionNames,
                                      info->ppEnabledExtensionNames + info->enabledExtensionCount);
  bool has_extension = false;
  for (auto name : extensions) {
    has_extension |= std::strcmp(name, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME) == 0;
  }
  if (!has_extension) {
    extensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
  }
  capture.enabledExtensionCount = extensions.size();
  capture.ppEnabledExtensionNames = extensions.data();
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
      const_cast<void*>(info->pNext), VK_TRUE};
  capture.pNext = &features;
  const VkResult result = real_create_device(physical, &capture, allocator, device);
  auto output = Output(".txt");
  output << "CREATE_DEVICE result=" << result << "\n";
  return result;
}

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* name) {
  if (std::strcmp(name, "vkCreateComputePipelines") == 0) {
    return reinterpret_cast<PFN_vkVoidFunction>(CreatePipelines);
  }
  if (std::strcmp(name, "vkSetDebugUtilsObjectNameEXT") == 0) {
    return reinterpret_cast<PFN_vkVoidFunction>(SetName);
  }
  return real_gdpa(device, name);
}

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* name) {
  auto proc = real_gipa(instance, name);
  if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) {
    real_gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(proc);
    return reinterpret_cast<PFN_vkVoidFunction>(vkGetDeviceProcAddr);
  }
  if (std::strcmp(name, "vkCreateDevice") == 0) {
    real_create_device = reinterpret_cast<PFN_vkCreateDevice>(proc);
    return reinterpret_cast<PFN_vkVoidFunction>(CreateDevice);
  }
  return proc;
}

extern "C" void* dlsym(void* handle, const char* name) noexcept {
  static const auto real = reinterpret_cast<void* (*)(void*, const char*)>(
      dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));
  void* proc = real(handle, name);
  if (std::strcmp(name, "vkGetInstanceProcAddr") == 0 && proc) {
    real_gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(proc);
    return reinterpret_cast<void*>(vkGetInstanceProcAddr);
  }
  return proc;
}
