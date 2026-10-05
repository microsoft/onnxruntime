// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/session/onnxruntime_cxx_api.h"

#include "dawn/dawn_proc.h"
#include "dawn/native/DawnNative.h"

namespace {
struct HostCache {
  std::mutex mutex;
  std::unordered_map<std::string, std::vector<std::byte>> blobs;
  size_t load_count = 0;
  size_t hit_count = 0;
  size_t store_count = 0;

  size_t Load(std::span<const std::byte> key, std::span<std::byte> value) {
    std::lock_guard<std::mutex> lock(mutex);
    ++load_count;
    const auto found = blobs.find(std::string(reinterpret_cast<const char*>(key.data()), key.size()));
    if (found == blobs.end()) {
      return 0;
    }
    if (value.size() >= found->second.size() && !value.empty()) {
      std::memcpy(value.data(), found->second.data(), found->second.size());
      ++hit_count;
    }
    return found->second.size();
  }

  void Store(std::span<const std::byte> key, std::span<const std::byte> value) {
    std::lock_guard<std::mutex> lock(mutex);
    ++store_count;
    blobs[std::string(reinterpret_cast<const char*>(key.data()), key.size())] =
        std::vector<std::byte>(value.begin(), value.end());
  }
};
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  bool no_proc_table = false;
  bool host_device_mode = false;
  bool no_implicit_sync = false;
  std::basic_string<ORTCHAR_T> plugin_path;
  int retval = 0;
  HostCache host_cache;
  std::unique_ptr<dawn::native::Instance> host_instance;
  wgpu::Device host_device;
  Ort::Env env{nullptr};
  try {
    for (int argument_index = 1; argument_index < argc; ++argument_index) {
      const std::basic_string<ORTCHAR_T> argument{argv[argument_index]};
      if (argument == ORT_TSTR("--no_proc_table")) {
        no_proc_table = true;
      } else if (argument == ORT_TSTR("--host_device")) {
        host_device_mode = true;
      } else if (argument == ORT_TSTR("--no_implicit_sync")) {
        no_implicit_sync = true;
      } else if (argument == ORT_TSTR("--plugin") && argument_index + 1 < argc) {
        plugin_path = argv[++argument_index];
      } else {
        throw std::runtime_error("Invalid external Dawn test argument.");
      }
    }
    for (size_t cache_pass = 0; cache_pass < (host_device_mode ? 3u : 1u); ++cache_pass) {
      size_t hits_before = 0;
      size_t stores_before = 0;
      {
        std::lock_guard<std::mutex> cache_lock(host_cache.mutex);
        hits_before = host_cache.hit_count;
        stores_before = host_cache.store_count;
      }
      env = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "Default"};

      if (host_device_mode) {
        dawnProcSetProcs(&dawn::native::GetProcs());
        if (!host_instance) {
          const wgpu::InstanceFeatureName instance_feature = wgpu::InstanceFeatureName::TimedWaitAny;
          wgpu::InstanceDescriptor instance_descriptor{};
          instance_descriptor.requiredFeatureCount = 1;
          instance_descriptor.requiredFeatures = &instance_feature;
          host_instance = std::make_unique<dawn::native::Instance>(&instance_descriptor);
        }
        auto adapters = host_instance->EnumerateAdapters();
        if (adapters.empty()) {
          throw std::runtime_error("No native Dawn adapter was found for the host device.");
        }
        const wgpu::FeatureName device_feature = wgpu::FeatureName::ImplicitDeviceSynchronization;
        wgpu::DawnCacheDeviceDescriptor cache_descriptor{};
        cache_descriptor.isolationKey = cache_pass == 2 ? "ort-external-host-isolated-cache" : "ort-external-host-cache";
        cache_descriptor.SetDawnLoadCacheDataCallback(
            [](std::span<const std::byte> key, std::span<std::byte> value, HostCache* cache) noexcept {
              return cache->Load(key, value);
            },
            &host_cache);
        cache_descriptor.SetDawnStoreCacheDataCallback(
            [](std::span<const std::byte> key, std::span<const std::byte> value, HostCache* cache) noexcept {
              cache->Store(key, value);
            },
            &host_cache);
        wgpu::DeviceDescriptor device_descriptor{};
        device_descriptor.nextInChain = &cache_descriptor;
        device_descriptor.requiredFeatureCount = no_implicit_sync ? 0 : 1;
        device_descriptor.requiredFeatures = no_implicit_sync ? nullptr : &device_feature;
        host_device = wgpu::Device::Acquire(adapters.front().CreateDevice(&device_descriptor));
        if (!host_device) {
          throw std::runtime_error("Native Dawn host device creation failed.");
        }
      }

      // model is https://github.com/onnx/onnx/blob/v1.15.0/onnx/backend/test/data/node/test_abs/model.onnx
      constexpr uint8_t MODEL_DATA[] = {8, 7, 18, 12, 98, 97, 99, 107, 101, 110,
                                        100, 45, 116, 101, 115, 116, 58, 73, 10, 11,
                                        10, 1, 120, 18, 1, 121, 34, 3, 65, 98,
                                        115, 18, 8, 116, 101, 115, 116, 95, 97, 98,
                                        115, 90, 23, 10, 1, 120, 18, 18, 10, 16,
                                        8, 1, 18, 12, 10, 2, 8, 3, 10, 2,
                                        8, 4, 10, 2, 8, 5, 98, 23, 10, 1,
                                        121, 18, 18, 10, 16, 8, 1, 18, 12, 10,
                                        2, 8, 3, 10, 2, 8, 4, 10, 2, 8,
                                        5, 66, 4, 10, 0, 16, 13};

      Ort::SessionOptions session_options;
      session_options.DisableMemPattern();
      session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      std::unordered_map<std::string, std::string> provider_options;
      if (!no_proc_table) {
        provider_options["dawnProcTable"] = std::to_string(reinterpret_cast<size_t>(&dawn::native::GetProcs()));
      }
      if (host_device_mode) {
        provider_options["deviceId"] = std::to_string(cache_pass + 1);
        provider_options["webgpuInstance"] = std::to_string(reinterpret_cast<size_t>(host_instance->Get()));
        provider_options["webgpuDevice"] = std::to_string(reinterpret_cast<size_t>(host_device.Get()));
        provider_options["preserveDevice"] = "1";
      }
      if (plugin_path.empty()) {
        session_options.AppendExecutionProvider("WebGPU", provider_options);
      } else {
        env.RegisterExecutionProviderLibrary("external_dawn_webgpu", plugin_path);
        Ort::ConstEpDevice webgpu_device{nullptr};
        for (const auto& ep_device : env.GetEpDevices()) {
          if (std::string(ep_device.EpName()) == "WebGpuExecutionProvider") {
            webgpu_device = ep_device;
            break;
          }
        }
        if (!webgpu_device) {
          throw std::runtime_error("External Dawn WebGPU plugin device was not found.");
        }
        session_options.AppendExecutionProvider_V2(env, {webgpu_device}, provider_options);
      }
      Ort::Session session{env, MODEL_DATA, sizeof(MODEL_DATA), session_options};

      if (no_proc_table || no_implicit_sync) {
        std::cerr << "Expected external-host initialization failure was not reported." << std::endl;
        retval = -1;
      } else {
        const std::array<int64_t, 3> shape{3, 4, 5};
        std::array<float, 60> input_data;
        for (size_t index = 0; index < input_data.size(); ++index) {
          input_data[index] = static_cast<float>(index) - 30.0f;
        }
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(), shape.data(), shape.size());
        const char* input_names[] = {"x"};
        const char* output_names[] = {"y"};
        auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
        if (outputs.size() != 1 ||
            outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != input_data.size()) {
          throw std::runtime_error("Unexpected Abs output shape.");
        }
        const float* output_data = outputs[0].GetTensorData<float>();
        for (size_t index = 0; index < input_data.size(); ++index) {
          if (output_data[index] != std::abs(input_data[index])) {
            throw std::runtime_error("Unexpected Abs output value.");
          }
        }
        std::cout << "WebGPU Abs inference passed with CPU fallback disabled." << std::endl;
        if (!plugin_path.empty()) {
          std::cout << "WebGPU plugin EP was registered and selected explicitly." << std::endl;
        }
        if (host_device_mode) {
          session = Ort::Session{nullptr};
          session_options = Ort::SessionOptions{nullptr};
          env = Ort::Env{nullptr};
          host_device.PushErrorScope(wgpu::ErrorFilter::Validation);
          wgpu::BufferDescriptor buffer_descriptor{};
          buffer_descriptor.size = sizeof(uint32_t);
          buffer_descriptor.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
          auto buffer = host_device.CreateBuffer(&buffer_descriptor);
          bool host_device_usable = false;
          auto error_scope = host_device.PopErrorScope(
              wgpu::CallbackMode::WaitAnyOnly,
              [](wgpu::PopErrorScopeStatus status, wgpu::ErrorType error_type, wgpu::StringView, bool* usable) {
                *usable = status == wgpu::PopErrorScopeStatus::Success && error_type == wgpu::ErrorType::NoError;
              },
              &host_device_usable);
          const auto wait_status = wgpu::Instance{host_instance->Get()}.WaitAny(
              error_scope, std::numeric_limits<uint64_t>::max());
          if (wait_status != wgpu::WaitStatus::Success || !host_device_usable || !buffer) {
            throw std::runtime_error("Host device was not usable after ORT session and environment teardown.");
          }
          std::cout << "Host-created WebGPU device remained usable after ORT teardown." << std::endl;
          std::lock_guard<std::mutex> cache_lock(host_cache.mutex);
          if (host_cache.load_count == 0 || host_cache.store_count == 0 || host_cache.blobs.empty()) {
            throw std::runtime_error("Host shader-cache callbacks were not exercised by GPU inference.");
          }
          if (cache_pass == 1 && host_cache.hit_count <= hits_before) {
            throw std::runtime_error("A second host device did not reuse the populated shader cache.");
          }
          if (cache_pass == 2 && (host_cache.hit_count != hits_before || host_cache.store_count <= stores_before)) {
            throw std::runtime_error("A different isolation key did not produce an isolated shader-cache miss.");
          }
          std::cout << "Host shader-cache callbacks passed: loads=" << host_cache.load_count
                    << ", hits=" << host_cache.hit_count << ", stores=" << host_cache.store_count << std::endl;
        }
        retval = 0;
      }
    }
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;

    if (no_proc_table && std::string(ex.what()).find("DawnProcTable must be provided") != std::string::npos) {
      std::cout << "DawnProcTable is not passing to ONNX Runtime, so an exception is thrown as expected." << std::endl;
      retval = 0;
    } else if (host_device_mode && no_implicit_sync &&
               std::string(ex.what()).find("must enable ImplicitDeviceSynchronization") != std::string::npos) {
      std::cout << "A host device without ImplicitDeviceSynchronization was rejected as expected." << std::endl;
      retval = 0;
    } else {
      std::cerr << "Unexpected exception." << std::endl;
      retval = -1;
    }
  }

  return retval;
}
