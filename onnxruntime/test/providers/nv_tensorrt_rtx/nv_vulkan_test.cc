#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include "core/session/onnxruntime_cxx_api.h"
#include "core/graph/model.h"
#if __has_include(<vulkan/vulkan.hpp>)
#include <cuda.h>
#include <cuda_runtime.h>
#include <core/providers/nv_tensorrt_rtx/nv_provider_options.h>
#include <gtest/gtest.h>
#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_NO_SMART_HANDLE
#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>

#include "test/providers/nv_tensorrt_rtx/test_nv_trt_rtx_ep_util.h"
#include "test/providers/provider_test_utils.h"

#if __linux__
#include <dlfcn.h>
#include <unistd.h>
#define HMODULE void*
#define CUDALIB "libcuda.so.1"
#define DLOPENOPTIONS , RTLD_LAZY
#else
#define STRINGIFY(a) _STRINGIFY(a)
#define _STRINGIFY(a) #a
#define NOMINMAX
#include <windows.h>
#define dlopen LoadLibraryA
#define dlclose FreeLibrary
#define dlsym GetProcAddress
#define RTLD_LAZY
#define CUDALIB "nvcuda.dll"
#define DLOPENOPTIONS
#endif

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime::test {

// Dynamic CUDA driver function loader
class CudaDriverLoader {
 private:
  HMODULE cuda_driver_dll_ = nullptr;

  // CUDA Driver API function pointers
  using cuCtxCreate_v4_t = CUresult (*)(CUcontext*, CUctxCreateParams*, unsigned int, CUdevice);
  using cuCtxDestroy_t = CUresult (*)(CUcontext);
  using cuCtxGetCurrent_t = CUresult (*)(CUcontext*);
  using cuCtxSetCurrent_t = CUresult (*)(CUcontext);
  using cuDeviceGetAttribute_t = CUresult (*)(int*, CUdevice_attribute attrib, CUdevice dev);

 public:
  cuCtxCreate_v4_t cuCtxCreate_v4_fn = nullptr;
  cuCtxDestroy_t cuCtxDestroy_fn = nullptr;
  cuCtxSetCurrent_t cuCtxSetCurrent_fn = nullptr;
  cuCtxGetCurrent_t cuCtxGetCurrent_fn = nullptr;
  cuDeviceGetAttribute_t cuDeviceGetAttribute_fn = nullptr;

  CudaDriverLoader() {
    cuda_driver_dll_ = dlopen(CUDALIB DLOPENOPTIONS);
    if (cuda_driver_dll_) {
      cuCtxCreate_v4_fn = reinterpret_cast<cuCtxCreate_v4_t>(
          dlsym(cuda_driver_dll_, "cuCtxCreate_v4"));
      cuCtxDestroy_fn = reinterpret_cast<cuCtxDestroy_t>(
          dlsym(cuda_driver_dll_, "cuCtxDestroy"));
      cuCtxSetCurrent_fn = reinterpret_cast<cuCtxSetCurrent_t>(
          dlsym(cuda_driver_dll_, "cuCtxSetCurrent"));
      cuCtxGetCurrent_fn = reinterpret_cast<cuCtxGetCurrent_t>(
          dlsym(cuda_driver_dll_, "cuCtxGetCurrent"));
      cuDeviceGetAttribute_fn = reinterpret_cast<cuDeviceGetAttribute_t>(
          dlsym(cuda_driver_dll_, "cuDeviceGetAttribute"));
    }
  }

  ~CudaDriverLoader() {
    if (cuda_driver_dll_) {
      dlclose(cuda_driver_dll_);
    }
  }

  bool IsLoaded() const {
    return cuda_driver_dll_ != nullptr &&
           cuCtxCreate_v4_fn != nullptr &&
           cuCtxSetCurrent_fn != nullptr &&
           cuCtxDestroy_fn != nullptr &&
           cuCtxGetCurrent_fn != nullptr &&
           cuDeviceGetAttribute_fn != nullptr;
  }
};

struct NvDevice {
  VkPhysicalDevice phys_dev{};
  VkPhysicalDeviceProperties props{};
  VkPhysicalDeviceVulkan11Properties id_props{};
  std::vector<Ort::ConstEpDevice> ep_device_candidates;
};

struct ExportableTimelineSemaphore {
  VkSemaphore vk_handle{};
  void* native_handle{};  // fd (Linux) or HANDLE (Windows), owned by us; close after ReleaseExternalSemaphoreHandle
  OrtExternalSemaphoreHandle* ort_handle{};
};

struct ExportableBuffer {
  VkBuffer buffer{};
  VkBufferView view{};
  VkDeviceMemory memory{};
  void* native_handle{};
  OrtExternalMemoryHandle* ort_handle{};
};

struct VkResources {
  vk::detail::DispatchLoaderDynamic loader;
  VkInstance instance;
  VkDevice device;
  VkPhysicalDevice phys_device;
  std::vector<ExportableBuffer> buffers;
  std::vector<ExportableTimelineSemaphore> semaphores;
  std::vector<NvDevice> nv_devices;
  VkQueue queue{};
  VkExternalComputeQueueNV ex_compute_queue{};
  OrtExternalResourceImporter* importer{};
  VkCommandPool cmd_pool{};
  VkCommandBuffer upload_cmd_buf{};
  VkCommandBuffer download_cmd_buf{};

  ~VkResources() {
    auto& interop = Ort::GetInteropApi();
    // Release ORT external handles first (they reference Vulkan objects)
    for (auto& sem : semaphores) {
      if (sem.ort_handle) {
        interop.ReleaseExternalSemaphoreHandle(sem.ort_handle);
        sem.ort_handle = nullptr;
      }
#if defined(_WIN32)
      if (sem.native_handle) {
        CloseHandle(static_cast<HANDLE>(sem.native_handle));
        sem.native_handle = nullptr;
      }
#else
      if (sem.native_handle) {
        close(static_cast<int>(reinterpret_cast<intptr_t>(sem.native_handle)));
        sem.native_handle = nullptr;
      }
#endif
    }
    for (auto& buf : buffers) {
      if (buf.ort_handle) {
        interop.ReleaseExternalMemoryHandle(buf.ort_handle);
        buf.ort_handle = nullptr;
      }
#if defined(_WIN32)
      if (buf.native_handle) {
        CloseHandle(static_cast<HANDLE>(buf.native_handle));
        buf.native_handle = nullptr;
      }
#else
      if (buf.native_handle) {
        close(static_cast<int>(reinterpret_cast<intptr_t>(buf.native_handle)));
        buf.native_handle = nullptr;
      }
#endif
    }
    if (importer) {
      interop.ReleaseExternalResourceImporter(importer);
      importer = nullptr;
    }
    // Destroy Vulkan objects in correct order: buffer views -> buffers -> memory, then semaphores, queue, device, instance
    if (device != VK_NULL_HANDLE) {
      loader.vkDeviceWaitIdle(device);
      for (auto& buf : buffers) {
        if (buf.view != VK_NULL_HANDLE) {
          loader.vkDestroyBufferView(device, buf.view, nullptr);
          buf.view = VK_NULL_HANDLE;
        }
        if (buf.buffer != VK_NULL_HANDLE) {
          loader.vkDestroyBuffer(device, buf.buffer, nullptr);
          buf.buffer = VK_NULL_HANDLE;
        }
        if (buf.memory != VK_NULL_HANDLE) {
          loader.vkFreeMemory(device, buf.memory, nullptr);
          buf.memory = VK_NULL_HANDLE;
        }
      }
      for (auto& sem : semaphores) {
        if (sem.vk_handle != VK_NULL_HANDLE) {
          loader.vkDestroySemaphore(device, sem.vk_handle, nullptr);
          sem.vk_handle = VK_NULL_HANDLE;
        }
      }
      if (ex_compute_queue != VK_NULL_HANDLE) {
        loader.vkDestroyExternalComputeQueueNV(device, ex_compute_queue, nullptr);
        ex_compute_queue = VK_NULL_HANDLE;
      }
      if (upload_cmd_buf != VK_NULL_HANDLE || download_cmd_buf != VK_NULL_HANDLE) {
        std::vector<VkCommandBuffer> to_free;
        if (upload_cmd_buf != VK_NULL_HANDLE) to_free.push_back(upload_cmd_buf);
        if (download_cmd_buf != VK_NULL_HANDLE) to_free.push_back(download_cmd_buf);
        loader.vkFreeCommandBuffers(device, cmd_pool, static_cast<uint32_t>(to_free.size()), to_free.data());
        upload_cmd_buf = VK_NULL_HANDLE;
        download_cmd_buf = VK_NULL_HANDLE;
      }
      if (cmd_pool != VK_NULL_HANDLE) {
        loader.vkDestroyCommandPool(device, cmd_pool, nullptr);
        cmd_pool = VK_NULL_HANDLE;
      }
      loader.vkDestroyDevice(device, nullptr);
      device = VK_NULL_HANDLE;
    }
    if (instance != VK_NULL_HANDLE) {
      loader.vkDestroyInstance(instance, nullptr);
      instance = VK_NULL_HANDLE;
    }
  }
};

void init_vulkan_interop(VkResources& resources) {
  resources.loader.init();

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.pApplicationName = "ORT";
  app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.apiVersion = VK_API_VERSION_1_4;
  app_info.pEngineName = "ORT";

  std::vector<const char*> instance_extensions{};
  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pApplicationInfo = &app_info;

  EXPECT_EQ(VK_SUCCESS, resources.loader.vkCreateInstance(&instance_info, nullptr, &resources.instance));
  resources.loader.init(vk::Instance(resources.instance));

  std::vector<VkPhysicalDevice> physical_devices;
  uint32_t count{};
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkEnumeratePhysicalDevices(resources.instance, &count, nullptr));
  physical_devices.resize(count);
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkEnumeratePhysicalDevices(resources.instance, &count, physical_devices.data()));

  auto ep_devices = ort_env->GetEpDevices();

  for (auto& p : physical_devices) {
    VkPhysicalDeviceVulkan11Properties id_props{};
    id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
    VkPhysicalDeviceProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &id_props;
    resources.loader.vkGetPhysicalDeviceProperties2(p, &props);
    if (props.properties.vendorID == 0x10DE) {
      NvDevice dev;
      dev.phys_dev = p;
      std::memcpy(&dev.props, &props, sizeof(props));
      std::memcpy(&dev.id_props, &id_props, sizeof(id_props));

      for (const auto& d : ep_devices) {
        if (d.Device().VendorId() == props.properties.vendorID && d.Device().DeviceId() == props.properties.deviceID) {
#if defined(_WIN32)
          // verify the real device with LUID, but on Linux we only have UUID which we don't know for EpDevice
          auto luid = d.Device().Metadata().GetValue("LUID");
          if (id_props.deviceLUIDValid && luid) {
            LUID vk_luid;
            std::memcpy(&vk_luid, dev.id_props.deviceLUID, sizeof(LUID));
            uint64_t ep_luid = std::stoull(luid);
            uint64_t vk = (uint64_t(vk_luid.HighPart) << 32) | uint64_t(vk_luid.LowPart);
            if (ep_luid != vk) {
              continue;
            }
          }
#endif
          dev.ep_device_candidates.push_back(d);
        }
      }
      if (!dev.ep_device_candidates.empty()) {
        resources.nv_devices.push_back(dev);
      }
    }
  }

  if (resources.nv_devices.empty()) {
    GTEST_SKIP() << "No Nv VK devices with EpDevices found";
  }

  // Try to run the test for first NV device
  resources.phys_device = resources.nv_devices[0].phys_dev;

  VkPhysicalDeviceVulkan11Features vk11{};
  vk11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  VkPhysicalDeviceVulkan12Features vk12{};
  vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  vk12.bufferDeviceAddress = true;
  vk12.timelineSemaphore = true;
  vk12.pNext = &vk11;

  VkExternalComputeQueueDeviceCreateInfoNV cig_create_info{};
  cig_create_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DEVICE_CREATE_INFO_NV;
  cig_create_info.reservedExternalQueues = 1;
  cig_create_info.pNext = &vk12;

  float priority = 1.f;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = 0;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;

  std::vector<const char*> device_extensions{};
#if defined(_WIN32)
  device_extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
  device_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
  device_extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  device_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  device_extensions.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
#endif
  device_extensions.push_back(VK_NV_EXTERNAL_COMPUTE_QUEUE_EXTENSION_NAME);

  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.enabledExtensionCount = uint32_t(device_extensions.size());
  device_info.ppEnabledExtensionNames = device_extensions.data();
  device_info.pNext = &cig_create_info;
  // TODO: skip if device with right extension fails?
  EXPECT_EQ(VK_SUCCESS,
            resources.loader.vkCreateDevice(resources.phys_device,
                                            &device_info,
                                            nullptr,
                                            &resources.device));
  resources.loader.init(vk::Device(resources.device));

  // External compute queues are used in CiG mode
  resources.loader.vkGetDeviceQueue(resources.device, 0, 0, &resources.queue);
  VkExternalComputeQueueCreateInfoNV ex_compute_queue_info{};
  ex_compute_queue_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_CREATE_INFO_NV;
  ex_compute_queue_info.preferredQueue = resources.queue;
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkCreateExternalComputeQueueNV(resources.device, &ex_compute_queue_info, nullptr, &resources.ex_compute_queue));
}

void create_timeline_semaphore(VkResources& resources, ExportableTimelineSemaphore& semaphore) {
  // VK creation
  VkSemaphoreCreateInfo sem_info{};
  sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkSemaphoreTypeCreateInfo timeline_info{};
  timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_info.initialValue = 0;
  timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkExportSemaphoreCreateInfoKHR export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
#if defined(_WIN32)
  export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
  sem_info.pNext = &timeline_info;
  timeline_info.pNext = &export_info;

  EXPECT_EQ(VK_SUCCESS, resources.loader.vkCreateSemaphore(resources.device, &sem_info, nullptr, &semaphore.vk_handle));

  // ORT logic
  OrtExternalSemaphoreDescriptor sem_desc = {};
  sem_desc.version = ORT_API_VERSION;
#if defined(_WIN32)
  VkSemaphoreGetWin32HandleInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
  info.semaphore = semaphore.vk_handle;
  info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  HANDLE fd;
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkGetSemaphoreWin32HandleKHR(resources.device, &info, &fd));
  sem_desc.type = ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_WIN32;
  semaphore.native_handle = (void*)(size_t)fd;
#else
  VkSemaphoreGetFdInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  info.semaphore = semaphore.vk_handle;
  info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
  int fd;
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkGetSemaphoreFdKHR(resources.device, &info, &fd));
  sem_desc.type = ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_OPAQUE_FD;
  semaphore.native_handle = (void*)(size_t)fd;
#endif
  sem_desc.native_handle = semaphore.native_handle;

  EXPECT_TRUE(Ort::Status(Ort::GetInteropApi().ImportSemaphore(resources.importer, &sem_desc, &semaphore.ort_handle)).IsOK());
}

void allocate_buffer(VkResources& resources,
                     ExportableBuffer& export_buffer,
                     VkDeviceSize size,
                     VkMemoryPropertyFlags mem_prop_flags,
                     bool do_export,
                     [[maybe_unused]] bool use_dma,
                     const Ort::ConstEpDevice& ep_device) {
  VkPhysicalDeviceMemoryProperties mem_props = {};
  resources.loader.vkGetPhysicalDeviceMemoryProperties(resources.phys_device,
                                                       &mem_props);
#if defined(_WIN32)
  auto handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  auto handle_type = use_dma ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
  uint32_t index = 0;
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.queueFamilyIndexCount = 1;
  buffer_info.pQueueFamilyIndices = &index;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_info.size = size;
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_STORAGE_TEXEL_BUFFER_BIT;
  VkExternalMemoryBufferCreateInfo external_create_info{};
  external_create_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  external_create_info.handleTypes = handle_type;
  if (do_export) {
    buffer_info.pNext = &external_create_info;
  }
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkCreateBuffer(resources.device, &buffer_info,
                                                        nullptr, &export_buffer.buffer));
  VkMemoryRequirements mem_reqs = {};
  resources.loader.vkGetBufferMemoryRequirements(resources.device,
                                                 export_buffer.buffer, &mem_reqs);
  int mem_idx = -1;
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((mem_reqs.memoryTypeBits & (1 << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & mem_prop_flags) ==
            mem_prop_flags) {
      mem_idx = i;
      break;
    }
  }
  EXPECT_NE(mem_idx, -1);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.memoryTypeIndex = mem_idx;
  alloc_info.allocationSize = mem_reqs.size;
  VkExportMemoryAllocateInfo export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
  export_info.handleTypes = handle_type;

  if (do_export) {
    alloc_info.pNext = &export_info;
  }
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkAllocateMemory(resources.device, &alloc_info,
                                                          nullptr, &export_buffer.memory));
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkBindBufferMemory(resources.device,
                                                            export_buffer.buffer, export_buffer.memory, 0));
  VkBufferViewCreateInfo buffer_view_info{};
  buffer_view_info.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
  buffer_view_info.buffer = export_buffer.buffer;
  buffer_view_info.format = VK_FORMAT_R32_SFLOAT;
  buffer_view_info.offset = 0;
  buffer_view_info.range = size;
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkCreateBufferView(resources.device, &buffer_view_info,
                                                            nullptr, &export_buffer.view));
  if (!do_export) {
    return;
  }
#if defined(_WIN32)
  VkMemoryGetWin32HandleInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
  info.memory = export_buffer.memory;
  info.handleType = handle_type;
  HANDLE fd;
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkGetMemoryWin32HandleKHR(resources.device, &info, &fd));
  export_buffer.native_handle = (void*)fd;
#else
  VkMemoryGetFdInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  info.memory = export_buffer.memory;
  info.handleType = handle_type;
  int fd;
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkGetMemoryFdKHR(resources.device, &info, &fd));
  export_buffer.native_handle = (void*)(size_t)fd;
#endif

  OrtExternalMemoryDescriptor mem_desc = {};
  mem_desc.version = ORT_API_VERSION;
#if defined(_WIN32)
  mem_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_BUFFER_WIN32;
#else
  EXPECT_FALSE(use_dma);  // would need to use ORT_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD
  mem_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_BUFFER_OPAQUE_FD;
#endif
  mem_desc.native_handle = export_buffer.native_handle;
  mem_desc.size_bytes = size;
  mem_desc.offset_bytes = 0;

  auto& interop_api = Ort::GetInteropApi();
  EXPECT_TRUE(Ort::Status(interop_api.ImportMemory(resources.importer, &mem_desc, &export_buffer.ort_handle)).IsOK());
}

struct TestParameters {
  bool force_cig{};
  bool allow_cig{};
  bool use_dmabuf{};
};

Ort::Session configure_session(const PathString& model_path, Ort::SyncStream& ort_stream, const Ort::ConstEpDevice& ep_device, size_t* aux_streams_array) {
  Ort::SessionOptions session_options;
  session_options.SetExecutionMode(ORT_SEQUENTIAL);
  session_options.DisableMemPattern();
  session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
  Ort::KeyValuePairs ep_options;
  ep_options.Add(onnxruntime::nv::provider_option_names::kUserComputeStream, std::to_string(size_t(ort_stream.GetHandle())).c_str());
  ep_options.Add(onnxruntime::nv::provider_option_names::kHasUserComputeStream, "1");
  ep_options.Add(onnxruntime::nv::provider_option_names::kMaxSharedMemSize, std::to_string(1024 * 28).c_str());
  ep_options.Add(onnxruntime::nv::provider_option_names::kUserAuxStreamArray, std::to_string((size_t)aux_streams_array).c_str());
  ep_options.Add(onnxruntime::nv::provider_option_names::kLengthAuxStreamArray, "1");
  ep_options.Add(onnxruntime::nv::provider_option_names::kCudaGraphEnable, "0");
  session_options.AppendExecutionProvider_V2(*ort_env, std::vector{ep_device}, ep_options);

  return Ort::Session(*ort_env, model_path.c_str(), session_options);
}

void test_vulkan_interop(TestParameters& test_params) {
  RegisteredEpDeviceUniquePtr nv_tensorrt_rtx_ep;
  Utils::RegisterAndGetNvTensorRtRtxEp(*ort_env, nv_tensorrt_rtx_ep);

  VkResources resources;
  init_vulkan_interop(resources);

  // Create a simple model: Y = X + 1 (Add with constant 1)
  PathString model_path = ORT_TSTR("external_mem_add_one_test.onnx");
  {
    onnxruntime::Model model("add_one_test", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();

    ONNX_NAMESPACE::TypeProto tensor_type;
    tensor_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
    tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(64);
    tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(64);

    auto& input_arg = graph.GetOrCreateNodeArg("X", &tensor_type);
    auto& output_arg = graph.GetOrCreateNodeArg("Y", &tensor_type);
    ONNX_NAMESPACE::TypeProto scalar_float;
    scalar_float.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    scalar_float.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    auto& one_arg = graph.GetOrCreateNodeArg("one", &scalar_float);
    ONNX_NAMESPACE::TensorProto one_proto;
    one_proto.set_name("one");
    one_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    one_proto.add_dims(1);
    one_proto.add_float_data(1.f);
    graph.AddInitializedTensor(one_proto);
    std::vector<onnxruntime::NodeArg*> add_inputs = {&input_arg, &one_arg};
    graph.AddNode("add_one", "Add", "Y = X + 1", add_inputs, {&output_arg});

    EXPECT_STATUS_OK(graph.Resolve());
    EXPECT_STATUS_OK(onnxruntime::Model::Save(model, model_path));
  }

  const int64_t batch = 1, channels = 3, dim = 64;
  const int64_t shape[] = {batch, channels, dim, dim};
  const size_t num_elements = batch * channels * dim * dim;
  const size_t buffer_size = num_elements * sizeof(float);

  if (resources.nv_devices.empty()) {
    GTEST_SKIP() << "No NV devices found";
  }
  const auto& ep_device = resources.nv_devices[0].ep_device_candidates[0];
  auto& interop_api = Ort::GetInteropApi();

  // Create external resource importer
  Ort::Status status(interop_api.CreateExternalResourceImporterForDevice(ep_device, &resources.importer));
  if (!status.IsOK() || resources.importer == nullptr) {
    clearFileIfExists(model_path);
    GTEST_SKIP() << "External resource import not supported";
  }

  // Check VK external memory (buffer) support
  bool can_import_vk_buffer = false;
#if defined(_WIN32)
  const auto vk_mem_handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_BUFFER_WIN32;
#else
  EXPECT_FALSE(test_params.use_dmabuf);  // would need to use ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_BUFFER_OPAQUE_FD
  const auto vk_mem_handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_BUFFER_OPAQUE_FD;
#endif
  status = Ort::Status(interop_api.CanImportMemory(resources.importer, vk_mem_handle_type, &can_import_vk_buffer));
  if (!status.IsOK() || !can_import_vk_buffer) {
    clearFileIfExists(model_path);
    GTEST_FAIL() << "VK external buffer import not supported";
  }

  // Check VK external semaphore (timeline) support
  bool can_import_vk_semaphore = false;
#if defined(_WIN32)
  const auto vk_sem_type = ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_WIN32;
#else
  const auto vk_sem_type = ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_OPAQUE_FD;
#endif
  status = Ort::Status(interop_api.CanImportSemaphore(resources.importer, vk_sem_type, &can_import_vk_semaphore));
  if (!status.IsOK() || !can_import_vk_semaphore) {
    clearFileIfExists(model_path);
    GTEST_FAIL() << "VK external timeline semaphore import not supported";
  }

  std::vector<uint8_t> external_compute_queue_data(64);
  VkExternalComputeQueueDataParamsNV ex_compute_queue_params{};
  ex_compute_queue_params.sType = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DATA_PARAMS_NV;
  ex_compute_queue_params.deviceIndex = 0;  // device index within device group, so 0 for us
  resources.loader.vkGetExternalComputeQueueDataNV(resources.ex_compute_queue, &ex_compute_queue_params, external_compute_queue_data.data());

  int device_count;
  CUdevice selected_device = -1;
  EXPECT_EQ(cudaSuccess, cudaGetDeviceCount(&device_count));
  EXPECT_GT(device_count, 0);
  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp props{};
    EXPECT_EQ(cudaSuccess, cudaGetDeviceProperties(&props, i));
    if (0 == std::memcmp(&props.uuid, resources.nv_devices[0].id_props.deviceUUID, sizeof(resources.nv_devices[0].id_props.deviceUUID))) {
      selected_device = i;
      break;
    }
  }
  EXPECT_GT(selected_device, -1);
  EXPECT_EQ(cudaSuccess, cudaSetDevice(selected_device));
  int cig_supported{false};
  EXPECT_EQ(cudaSuccess, cudaDeviceGetAttribute(&cig_supported, cudaDevAttrVulkanCigSupported, selected_device));
  CudaDriverLoader driver;
  // int dma_supported{false};
  // EXPECT_EQ(CUDA_SUCCESS, driver.cuDeviceGetAttribute_fn(&dma_supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, selected_device));

  if (test_params.allow_cig) {
    EXPECT_EQ(driver.IsLoaded(), true);
    // At least on Windows we would like to test CiG
    EXPECT_TRUE(cig_supported || !test_params.force_cig);

    if (cig_supported) {
      CUcontext ctx{};
      CUctxCigParam cig_params{};
      cig_params.sharedDataType = CIG_DATA_TYPE_NV_BLOB;
      cig_params.sharedData = external_compute_queue_data.data();
      CUctxCreateParams params{};
      params.cigParams = &cig_params;
      EXPECT_EQ(CUDA_SUCCESS, driver.cuCtxCreate_v4_fn(&ctx, &params, 0, selected_device));
      EXPECT_EQ(CUDA_SUCCESS, driver.cuCtxSetCurrent_fn(ctx));
    }
  }

  ExportableTimelineSemaphore input_ready{};
  create_timeline_semaphore(resources, input_ready);
  resources.semaphores.push_back(input_ready);
  ExportableTimelineSemaphore inference_done{};
  create_timeline_semaphore(resources, inference_done);
  resources.semaphores.push_back(inference_done);
  ExportableTimelineSemaphore download_done{};
  create_timeline_semaphore(resources, download_done);
  resources.semaphores.push_back(download_done);

  ExportableBuffer upload_buffer{};
  allocate_buffer(resources, upload_buffer, buffer_size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, false, test_params.use_dmabuf, ep_device);
  resources.buffers.push_back(upload_buffer);
  ExportableBuffer input_buffer{};
  allocate_buffer(resources, input_buffer, buffer_size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true, test_params.use_dmabuf, ep_device);
  resources.buffers.push_back(input_buffer);
  ExportableBuffer output_buffer{};
  allocate_buffer(resources, output_buffer, buffer_size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true, test_params.use_dmabuf, ep_device);
  resources.buffers.push_back(output_buffer);

  // Create command pool and command buffers for upload/download
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = 0;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkCreateCommandPool(resources.device, &pool_info, nullptr, &resources.cmd_pool));
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = resources.cmd_pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 2;
  VkCommandBuffer cmd_bufs[2];
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkAllocateCommandBuffers(resources.device, &alloc_info, cmd_bufs));
  resources.upload_cmd_buf = cmd_bufs[0];
  resources.download_cmd_buf = cmd_bufs[1];

  ExportableBuffer& upload_buf = resources.buffers[0];
  ExportableBuffer& input_buf = resources.buffers[1];
  ExportableBuffer& output_buf = resources.buffers[2];
  ExportableTimelineSemaphore& input_ready_sem = resources.semaphores[0];
  ExportableTimelineSemaphore& inference_done_sem = resources.semaphores[1];
  ExportableTimelineSemaphore& download_done_sem = resources.semaphores[2];

  // Fill upload buffer with known values: 0.f, 1.f, 2.f, 3.f, ...
  void* mapped = nullptr;
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkMapMemory(resources.device, upload_buf.memory, 0, buffer_size, 0, &mapped));
  float* host_data = static_cast<float*>(mapped);
  for (size_t i = 0; i < num_elements; ++i) {
    host_data[i] = static_cast<float>(i);
  }
  resources.loader.vkUnmapMemory(resources.device, upload_buf.memory);

  // Record upload command buffer: upload_buffer -> input_buffer (do not submit yet)
  VkCommandBufferBeginInfo cmd_buf_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkBeginCommandBuffer(resources.upload_cmd_buf, &cmd_buf_info));
  VkBufferCopy copy_region{};
  copy_region.size = buffer_size;
  copy_region.srcOffset = 0;
  copy_region.dstOffset = 0;
  resources.loader.vkCmdCopyBuffer(resources.upload_cmd_buf, upload_buf.buffer, input_buf.buffer, 1, &copy_region);
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkEndCommandBuffer(resources.upload_cmd_buf));

  // Create ORT tensors from imported memory (input and output buffers)
  OrtExternalTensorDescriptor tensor_desc = {};
  tensor_desc.version = ORT_API_VERSION;
  tensor_desc.element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  tensor_desc.shape = shape;
  tensor_desc.rank = 4;
  tensor_desc.offset_bytes = 0;
  OrtValue* input_tensor = nullptr;
  OrtValue* output_tensor = nullptr;
  EXPECT_TRUE(Ort::Status(interop_api.CreateTensorFromMemory(resources.importer, input_buf.ort_handle, &tensor_desc, &input_tensor)).IsOK());
  EXPECT_TRUE(Ort::Status(interop_api.CreateTensorFromMemory(resources.importer, output_buf.ort_handle, &tensor_desc, &output_tensor)).IsOK());
  void* input_data_ptr = nullptr;
  void* output_data_ptr = nullptr;
  EXPECT_TRUE(Ort::Status(Ort::GetApi().GetTensorMutableData(input_tensor, &input_data_ptr)).IsOK());
  EXPECT_TRUE(Ort::Status(Ort::GetApi().GetTensorMutableData(output_tensor, &output_data_ptr)).IsOK());
  cudaPointerAttributes input_attrs, output_attrs;
  ASSERT_EQ(cudaPointerGetAttributes(&input_attrs, input_data_ptr), cudaSuccess);
  ASSERT_EQ(cudaPointerGetAttributes(&output_attrs, output_data_ptr), cudaSuccess);
  EXPECT_EQ(input_attrs.type, cudaMemoryTypeDevice) << "Input tensor must be CUDA device memory";
  EXPECT_EQ(output_attrs.type, cudaMemoryTypeDevice) << "Output tensor must be CUDA device memory";

  // Add the NvTensorRtRtx EP
  // Configure to use our CUDA stream
  auto ort_stream = ep_device.CreateSyncStream();
  size_t stream_addr_val = reinterpret_cast<size_t>(ort_stream.GetHandle());
  size_t aux_streams[] = {stream_addr_val};
  auto session = configure_session(model_path, ort_stream, ep_device, aux_streams);

  Ort::IoBinding io_binding(session);
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
  Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
  io_binding.BindInput(input_name.get(), Ort::Value(input_tensor));
  io_binding.BindOutput(output_name.get(), Ort::Value(output_tensor));
  io_binding.SynchronizeInputs();

  // Semaphores: upload signals input_ready=1; ORT waits input_ready 1, runs, signals inference_done=2; download signals download_done=3
  const uint64_t input_ready_value = 1;
  const uint64_t inference_done_value = 1;
  const uint64_t download_done_value = 1;

  // Record download command buffer: output_buffer -> upload_buffer (do not submit yet)
  auto begin_info = VkCommandBufferBeginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkBeginCommandBuffer(resources.download_cmd_buf, &begin_info));
  resources.loader.vkCmdCopyBuffer(resources.download_cmd_buf, output_buf.buffer, upload_buf.buffer, 1, &copy_region);
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkEndCommandBuffer(resources.download_cmd_buf));

  // Submit upload first (signal input_ready = 1 when copy completes)
  uint64_t signal_input_ready = input_ready_value;
  VkTimelineSemaphoreSubmitInfo timeline_info = {};
  timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timeline_info.signalSemaphoreValueCount = 1;
  timeline_info.pSignalSemaphoreValues = &signal_input_ready;
  VkSubmitInfo submit_upload = {};
  submit_upload.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_upload.pNext = &timeline_info;
  submit_upload.commandBufferCount = 1;
  submit_upload.pCommandBuffers = &resources.upload_cmd_buf;
  submit_upload.signalSemaphoreCount = 1;
  submit_upload.pSignalSemaphores = &input_ready_sem.vk_handle;
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkQueueSubmit(resources.queue, 1, &submit_upload, VK_NULL_HANDLE));

  EXPECT_TRUE(Ort::Status(interop_api.WaitSemaphore(resources.importer, input_ready_sem.ort_handle, ort_stream, input_ready_value)).IsOK());
  Ort::RunOptions run_options;
  run_options.SetSyncStream(ort_stream);
  run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
  session.Run(run_options, io_binding);
  EXPECT_TRUE(Ort::Status(interop_api.SignalSemaphore(resources.importer, inference_done_sem.ort_handle, ort_stream, inference_done_value)).IsOK());

  uint64_t wait_inference_done = inference_done_value;
  uint64_t signal_download_done = download_done_value;
  timeline_info.waitSemaphoreValueCount = 1;
  timeline_info.pWaitSemaphoreValues = &wait_inference_done;
  timeline_info.signalSemaphoreValueCount = 1;
  timeline_info.pSignalSemaphoreValues = &signal_download_done;
  VkPipelineStageFlags wait_dst_stage_mask = VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkSubmitInfo submit_download = {};
  submit_download.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_download.pNext = &timeline_info;
  submit_download.waitSemaphoreCount = 1;
  submit_download.pWaitDstStageMask = &wait_dst_stage_mask;
  submit_download.pWaitSemaphores = &inference_done_sem.vk_handle;
  submit_download.commandBufferCount = 1;
  submit_download.pCommandBuffers = &resources.download_cmd_buf;
  submit_download.signalSemaphoreCount = 1;
  submit_download.pSignalSemaphores = &download_done_sem.vk_handle;
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkQueueSubmit(resources.queue, 1, &submit_download, VK_NULL_HANDLE));

  VkSemaphoreWaitInfo wait_info = {};
  wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  wait_info.semaphoreCount = 1;
  wait_info.pSemaphores = &download_done_sem.vk_handle;
  wait_info.pValues = &download_done_value;
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkWaitSemaphores(resources.device, &wait_info, UINT64_MAX));

  // Read back and verify: result should be input + 1 (0+1, 1+1, 2+1, 3+1, ...)
  mapped = nullptr;
  EXPECT_EQ(VK_SUCCESS, resources.loader.vkMapMemory(resources.device, upload_buf.memory, 0, buffer_size, 0, &mapped));
  host_data = static_cast<float*>(mapped);
  for (size_t i = 0; i < num_elements; ++i) {
    float expected = static_cast<float>(i) + 1.f;
    EXPECT_FLOAT_EQ(host_data[i], expected) << "index " << i;
  }
  resources.loader.vkUnmapMemory(resources.device, upload_buf.memory);

  clearFileIfExists(model_path);
}

TEST(NvExecutionProviderVulkanTest, VkCigDisabled) {
  TestParameters params;
  params.allow_cig = false;
  test_vulkan_interop(params);
}

#if defined(_WIN32)
TEST(NvExecutionProviderVulkanTest, VkForceCig) {
  TestParameters params;
  params.allow_cig = true;
  params.force_cig = true;
  test_vulkan_interop(params);
}
#endif

}  // namespace onnxruntime::test
#endif
