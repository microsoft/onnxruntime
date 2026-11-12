// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/webgpu/webgpu_provider_factory.h"

#include "dawn/dawn_proc.h"
#include "dawn/native/DawnNative.h"

#if defined(_WIN32) && defined(DAWN_ENABLE_D3D12)
#include "dawn/native/D3D12Backend.h"
#endif

namespace {
// Instrument the host proc table, so these checks also exercise an external ORT DLL.
std::atomic<size_t> timestamp_queries{0};
std::atomic<size_t> timestamp_writes{0};
std::atomic<size_t> timestamp_resolves{0};
std::atomic<size_t> query_buffers{0};

struct CalibrationState {
  size_t calls = 0;
  bool fail = false;
  bool invalid = false;
  bool profiling = false;
  std::atomic<bool> delay_failed{false};
  std::vector<std::thread> release_threads;
#if defined(_WIN32) && defined(DAWN_ENABLE_D3D12)
  Microsoft::WRL::ComPtr<ID3D12Device> device;
  Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
#endif

  void Join() {
    for (auto& thread : release_threads) thread.join();
    release_threads.clear();
  }
  ~CalibrationState() { Join(); }
};

CalibrationState* calibration_state = nullptr;

int ORT_API_CALL CalibrateHostDevice(void* device, OrtWebGpuClockCalibration* result, void* user_data) noexcept {
  try {
    auto& state = *static_cast<CalibrationState*>(user_data);
    ++state.calls;
    if (state.fail) return 0;
    if (state.invalid) {
      result->gpu_to_cpu_scale = std::numeric_limits<double>::quiet_NaN();
      return 1;
    }
#if defined(_WIN32) && defined(DAWN_ENABLE_D3D12)
    const auto handle = static_cast<WGPUDevice>(device);
    const auto toggles = dawn::native::GetTogglesUsed(handle);
    for (const auto& toggle : toggles) {
      if (std::string_view{toggle} == "disable_timestamp_query_conversion" ||
          std::string_view{toggle} == "timestamp_quantization") return 0;
    }
    state.device = dawn::native::d3d12::GetD3D12Device(handle);
    state.queue = dawn::native::d3d12::GetD3D12CommandQueue(handle);
    UINT64 frequency = 0;
    UINT64 gpu_ticks = 0;
    UINT64 cpu_ticks = 0;
    if (!state.queue || FAILED(state.queue->GetTimestampFrequency(&frequency)) ||
        frequency < 15259 || frequency > 1000000000 ||
        FAILED(state.queue->GetClockCalibration(&gpu_ticks, &cpu_ticks))) return 0;
    // Match this Dawn pin's resolved query conversion, including its rounded rate.
    const float period = static_cast<float>(1e9 / static_cast<double>(frequency));
    const auto shift = 16u - std::min(static_cast<uint32_t>(std::ceil(std::log2(period))), 16u);
    const auto multiplier = static_cast<uint32_t>(period * static_cast<float>(1u << shift));
    result->gpu_timestamp_ns = (gpu_ticks >> shift) * multiplier +
                               (((gpu_ticks & ((uint64_t{1} << shift) - 1)) * multiplier) >> shift);
    result->cpu_qpc_ticks = cpu_ticks;
    result->gpu_to_cpu_scale = 1e9 * static_cast<double>(uint64_t{1} << shift) /
                               (static_cast<double>(frequency) * multiplier);
    return 1;
#else
    (void)device;
    (void)result;
    return 0;
#endif
  } catch (...) {
    return 0;
  }
}

void DelaySubmission() noexcept {
#if defined(_WIN32) && defined(DAWN_ENABLE_D3D12)
  auto* state = calibration_state;
  if (!state || !state->profiling || !state->queue) return;
  try {
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    if (FAILED(state->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
      state->delay_failed = true;
      return;
    }
    // Test-only delay: submission anchoring would hide this GPU queue bubble.
    // Start the releaser first so a thread-allocation failure cannot block the GPU.
    state->release_threads.emplace_back([fence, state]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(25));
      if (FAILED(fence->Signal(1))) state->delay_failed = true;
    });
    if (FAILED(state->queue->Wait(fence.Get(), 1))) state->delay_failed = true;
  } catch (...) {
    state->delay_failed = true;
  }
#endif
}

DawnProcTable ProfilingProcs() {
  auto procs = dawn::native::GetProcs();
  procs.deviceCreateQuerySet = [](WGPUDevice device, const WGPUQuerySetDescriptor* descriptor) {
    if (descriptor->type == WGPUQueryType_Timestamp) {
      ++timestamp_queries;
    }
    return dawn::native::GetProcs().deviceCreateQuerySet(device, descriptor);
  };
  procs.commandEncoderBeginComputePass = [](WGPUCommandEncoder encoder, const WGPUComputePassDescriptor* descriptor) {
    if (descriptor && descriptor->timestampWrites) {
      ++timestamp_writes;
    }
    return dawn::native::GetProcs().commandEncoderBeginComputePass(encoder, descriptor);
  };
  procs.computePassEncoderWriteTimestamp = [](WGPUComputePassEncoder encoder, WGPUQuerySet queries, uint32_t index) {
    ++timestamp_writes;
    dawn::native::GetProcs().computePassEncoderWriteTimestamp(encoder, queries, index);
  };
  procs.commandEncoderResolveQuerySet = [](WGPUCommandEncoder encoder, WGPUQuerySet queries, uint32_t first,
                                           uint32_t count, WGPUBuffer destination, uint64_t offset) {
    ++timestamp_resolves;
    dawn::native::GetProcs().commandEncoderResolveQuerySet(encoder, queries, first, count, destination, offset);
  };
  procs.deviceCreateBuffer = [](WGPUDevice device, const WGPUBufferDescriptor* descriptor) {
    if ((descriptor->usage & WGPUBufferUsage_QueryResolve) != 0) {
      ++query_buffers;
    }
    return dawn::native::GetProcs().deviceCreateBuffer(device, descriptor);
  };
  procs.queueSubmit = [](WGPUQueue queue, size_t count, const WGPUCommandBuffer* commands) {
    DelaySubmission();
    dawn::native::GetProcs().queueSubmit(queue, count, commands);
  };
  return procs;
}

void TestProfiling(Ort::Env& env, const uint8_t* model, size_t model_size,
                   std::unordered_map<std::string, std::string> provider_options,
                   bool use_calibration, bool calibration_failure, bool calibration_invalid) {
  CalibrationState state;
  state.fail = calibration_failure;
  state.invalid = calibration_invalid;
  calibration_state = &state;
  struct ClearState {
    Ort::Env& env;
    CalibrationState& state;
    ~ClearState() {
      state.Join();
#if defined(_WIN32) && defined(DAWN_ENABLE_D3D12)
      // Native COM references must not outlive Dawn's D3D12 runtime loader.
      state.queue.Reset();
      state.device.Reset();
#endif
      // Preserved contexts must release the copied callback before its state dies.
      env = Ort::Env{nullptr};
      calibration_state = nullptr;
    }
  } clear_state{env, state};
  const OrtWebGpuClockCalibrationCallbackInfo callback{1, CalibrateHostDevice, &state};
  if (use_calibration) {
    provider_options["clockCalibration"] = std::to_string(reinterpret_cast<uintptr_t>(&callback));
  }
#if defined(_WIN32) && defined(DAWN_ENABLE_D3D12)
  const bool expect_calibration = use_calibration && !calibration_failure && !calibration_invalid;
#else
  const bool expect_calibration = false;
#endif
  const auto procs = ProfilingProcs();
  provider_options["dawnProcTable"] = std::to_string(reinterpret_cast<uintptr_t>(&procs));
  const std::array<int64_t, 3> shape{3, 4, 5};
  std::array<float, 60> data{};
  data.fill(-2.0f);
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input = Ort::Value::CreateTensor<float>(memory, data.data(), data.size(), shape.data(), shape.size());
  const char* input_names[] = {"x"};
  const char* output_names[] = {"y"};
  for (const bool profile : {false, true, false}) {
    state.profiling = profile;
    const auto calls_before = state.calls;
    const std::array counts_before{timestamp_queries.load(), timestamp_writes.load(),
                                   timestamp_resolves.load(), query_buffers.load()};
    Ort::SessionOptions options;
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    if (profile) {
      options.EnableProfiling(ORT_TSTR("external_dawn_profile"));
    }
    options.AppendExecutionProvider("WebGPU", provider_options);
    Ort::Session session{env, model, model_size, options};
    if (use_calibration && profile) {
      auto check_rejection = [&](const std::string& value, const char* message) {
        auto rejected_options = provider_options;
        rejected_options["clockCalibration"] = value;
        try {
          Ort::SessionOptions rejected_session_options;
          rejected_session_options.AppendExecutionProvider("WebGPU", rejected_options);
          Ort::Session rejected_session{env, model, model_size, rejected_session_options};
        } catch (const Ort::Exception& error) {
          if (std::string{error.what()}.find(message) != std::string::npos) return;
          throw;
        }
        throw std::runtime_error("Invalid clockCalibration configuration was accepted.");
      };
      check_rejection("0", "Invalid clockCalibration address");
      check_rejection("123junk", "Invalid clockCalibration address");
      auto rejected = callback;
      rejected.version = 2;
      const auto address = std::to_string(reinterpret_cast<uintptr_t>(&rejected));
      check_rejection(address, "Unsupported clockCalibration version");
      rejected.version = 1;
      rejected.callback = nullptr;
      check_rejection(address, "clockCalibration requires a callback");
      rejected = callback;
      rejected.user_data = nullptr;
      check_rejection(address, "different clockCalibration callback");
      std::cout << "Invalid and conflicting calibration options rejected." << std::endl;
    }
    for (size_t run = 0; run < 2; ++run) {
      auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
      state.Join();
      const auto* result = outputs.at(0).GetTensorData<float>();
      for (size_t index = 0; index < data.size(); ++index) {
        if (result[index] != 2.0f) {
          throw std::runtime_error("Profiling test did not produce the expected GPU output.");
        }
      }
    }
    if (!profile) {
      const std::array counts_after{timestamp_queries.load(), timestamp_writes.load(),
                                    timestamp_resolves.load(), query_buffers.load()};
      if (counts_before != counts_after || state.calls != calls_before) {
        throw std::runtime_error("Timestamp work was performed with profiling disabled.");
      }
      std::cout << "Profiling off: zero additional timestamp queries, writes, resolves and query buffers." << std::endl;
    } else {
      if (timestamp_queries == 0 || timestamp_writes == 0 || timestamp_resolves < 2 || query_buffers == 0) {
        throw std::runtime_error("Profiling did not exercise timestamp queries on both runs.");
      }
      Ort::AllocatorWithDefaultOptions allocator;
      auto path = session.EndProfilingAllocated(allocator);
      std::ifstream file{path.get()};
      const std::string trace{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
      if (trace.find(expect_calibration ? "d3d12_clock_calibration" : "first_submit") == std::string::npos ||
          trace.find(expect_calibration ? "first_submit" : "d3d12_clock_calibration") != std::string::npos ||
          trace.find("WebGPU::Submit") == std::string::npos || trace.find("WebGPU::Wait") == std::string::npos) {
        throw std::runtime_error("External profiling trace did not identify the expected clock alignment.");
      }
#ifdef _WIN32
      const size_t expected_calls = use_calibration ? 2 : 0;
#else
      const size_t expected_calls = 0;
#endif
      if (state.delay_failed || state.calls - calls_before != expected_calls) {
        throw std::runtime_error("External calibration callback or queue-delay checks failed.");
      }
      std::cout << "Profiling on: queries=" << timestamp_queries << ", writes=" << timestamp_writes
                << ", resolves=" << timestamp_resolves << ", query buffers=" << query_buffers
                << "; trace=" << path.get() << std::endl;
      std::cout << "Calibration: calls=" << state.calls - calls_before
                << ", alignment=" << (expect_calibration ? "d3d12_clock_calibration" : "first_submit") << std::endl;
    }
  }
}

struct HostCache {
  std::mutex mutex;
  std::unordered_map<std::string, std::vector<std::byte>> blobs;
  size_t load_count = 0;
  size_t hit_count = 0;
  size_t store_count = 0;
  std::atomic<bool> callback_failed{false};
  bool simulate_allocation_failure = false;

  size_t Load(std::span<const std::byte> key, std::span<std::byte> value) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex);
      ++load_count;
      if (simulate_allocation_failure) {
        throw std::bad_alloc{};
      }
      const auto found = blobs.find(std::string(reinterpret_cast<const char*>(key.data()), key.size()));
      if (found == blobs.end()) {
        return 0;
      }
      if (!value.empty()) {
        if (value.size() != found->second.size()) {
          return 0;
        }
        std::memcpy(value.data(), found->second.data(), found->second.size());
        ++hit_count;
      }
      return found->second.size();
    } catch (...) {
      callback_failed.store(true);
      return 0;
    }
  }

  void Store(std::span<const std::byte> key, std::span<const std::byte> value) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex);
      ++store_count;
      if (simulate_allocation_failure) {
        throw std::bad_alloc{};
      }
      blobs[std::string(reinterpret_cast<const char*>(key.data()), key.size())] =
          std::vector<std::byte>(value.begin(), value.end());
    } catch (...) {
      callback_failed.store(true);
    }
  }

  void ValidatePass(size_t pass, size_t hits_before, size_t stores_before) {
    if (callback_failed.load()) {
      throw std::runtime_error("A host shader-cache callback failed before completing its operation.");
    }
    std::lock_guard<std::mutex> cache_lock(mutex);
    if (load_count == 0 || store_count == 0 || blobs.empty()) {
      throw std::runtime_error("Host shader-cache callbacks were not exercised by GPU inference.");
    }
    if (pass == 0 && (hit_count != hits_before || store_count <= stores_before)) {
      throw std::runtime_error("A cold device did not populate the empty shader cache.");
    }
    if (pass == 1 && hit_count <= hits_before) {
      throw std::runtime_error("A second device did not reuse the populated shader cache.");
    }
    if (pass == 2 && (hit_count != hits_before || store_count <= stores_before)) {
      throw std::runtime_error("A different isolation key did not produce an isolated shader-cache miss.");
    }
    std::cout << "Shader-cache pass " << pass << " passed: loads=" << load_count
              << ", hits=" << hit_count << ", stores=" << store_count << std::endl;
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
  bool ort_owned_cache_mode = false;
  bool static_plugin_mode = false;
  bool no_implicit_sync = false;
  bool cache_callback_failure = false;
  bool profiling_test = false;
  bool use_calibration = false;
  bool calibration_failure = false;
  bool calibration_invalid = false;
  std::basic_string<ORTCHAR_T> plugin_path;
  int retval = 0;
  HostCache host_cache;
  OrtDawnCacheInterface cache_interface{};
  cache_interface.version = ORT_DAWN_CACHE_INTERFACE_VERSION;
  cache_interface.userdata = &host_cache;
  cache_interface.load = [](const void* key, size_t key_size, void* value, size_t value_size, void* userdata) noexcept {
    return static_cast<HostCache*>(userdata)->Load(
        {static_cast<const std::byte*>(key), key_size}, {static_cast<std::byte*>(value), value_size});
  };
  cache_interface.store = [](const void* key, size_t key_size, const void* value, size_t value_size, void* userdata) noexcept {
    static_cast<HostCache*>(userdata)->Store(
        {static_cast<const std::byte*>(key), key_size}, {static_cast<const std::byte*>(value), value_size});
  };
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
      } else if (argument == ORT_TSTR("--profiling")) {
        profiling_test = true;
      } else if (argument == ORT_TSTR("--clock_calibration")) {
        use_calibration = true;
      } else if (argument == ORT_TSTR("--calibration_failure")) {
        calibration_failure = true;
      } else if (argument == ORT_TSTR("--calibration_invalid")) {
        calibration_invalid = true;
      } else if (argument == ORT_TSTR("--ort_owned_cache")) {
        ort_owned_cache_mode = true;
      } else if (argument == ORT_TSTR("--static_plugin")) {
        static_plugin_mode = true;
      } else if (argument == ORT_TSTR("--no_implicit_sync")) {
        no_implicit_sync = true;
      } else if (argument == ORT_TSTR("--cache_callback_failure")) {
        host_device_mode = true;
        cache_callback_failure = true;
        host_cache.simulate_allocation_failure = true;
      } else if (argument == ORT_TSTR("--plugin") && argument_index + 1 < argc) {
        plugin_path = argv[++argument_index];
      } else {
        throw std::runtime_error("Invalid external Dawn test argument.");
      }
    }
    if ((use_calibration && !profiling_test) || ((calibration_failure || calibration_invalid) && !use_calibration)) {
      throw std::runtime_error("Calibration flags require --profiling --clock_calibration.");
    }
#if !defined(_WIN32) || !defined(DAWN_ENABLE_D3D12)
    if (use_calibration) throw std::runtime_error("The calibration regression requires Windows D3D12 Dawn.");
#endif
    if (profiling_test && (no_proc_table || no_implicit_sync || cache_callback_failure || !plugin_path.empty() ||
                           ort_owned_cache_mode || static_plugin_mode)) {
      throw std::runtime_error("--profiling supports only device and clock-calibration arguments.");
    }
    if (ort_owned_cache_mode && (host_device_mode || no_proc_table || no_implicit_sync)) {
      throw std::runtime_error("ORT-owned cache validation requires an ORT-owned device and a proc table.");
    }
    for (size_t cache_pass = 0; cache_pass < (host_device_mode || ort_owned_cache_mode ? 3u : 1u); ++cache_pass) {
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
        wgpu::RequestAdapterOptions adapter_options{};
        if (use_calibration) adapter_options.backendType = wgpu::BackendType::D3D12;
        auto adapters = host_instance->EnumerateAdapters(use_calibration ? &adapter_options : nullptr);
        if (adapters.empty()) {
          throw std::runtime_error("No native Dawn adapter was found for the host device.");
        }
        const std::array device_features{wgpu::FeatureName::ImplicitDeviceSynchronization,
                                         wgpu::FeatureName::TimestampQuery};
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
        const char* disabled_toggle = "timestamp_quantization";
        wgpu::DawnTogglesDescriptor calibration_toggles{};
        if (use_calibration) {
          calibration_toggles.disabledToggleCount = 1;
          calibration_toggles.disabledToggles = &disabled_toggle;
          calibration_toggles.nextInChain = &cache_descriptor;
          device_descriptor.nextInChain = &calibration_toggles;
        }
        device_descriptor.requiredFeatureCount = no_implicit_sync ? 0 : (profiling_test ? 2 : 1);
        device_descriptor.requiredFeatures = no_implicit_sync ? nullptr : device_features.data();
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
        provider_options["dawnProcTable"] = std::to_string(reinterpret_cast<uintptr_t>(&dawn::native::GetProcs()));
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
        provider_options["weightLoadAcceleration"] = "preferred";
#endif
      }
      const int custom_id = cache_pass == 2 ? std::numeric_limits<int16_t>::max() : static_cast<int>(cache_pass + 1);
      if (host_device_mode) {
        provider_options["deviceId"] = std::to_string(custom_id);
        provider_options["webgpuInstance"] = std::to_string(reinterpret_cast<uintptr_t>(host_instance->Get()));
        provider_options["webgpuDevice"] = std::to_string(reinterpret_cast<uintptr_t>(host_device.Get()));
        provider_options["preserveDevice"] = "1";
      }
      if (profiling_test) {
        TestProfiling(env, MODEL_DATA, sizeof(MODEL_DATA), provider_options,
                      use_calibration, calibration_failure, calibration_invalid);
        return 0;
      }
      if (ort_owned_cache_mode) {
        cache_interface.isolation_key = cache_pass == 2 ? "ort-owned-isolated-cache" : "ort-owned-cache";
        provider_options[ORT_WEBGPU_DAWN_CACHE_INTERFACE_OPTION] =
            std::to_string(reinterpret_cast<uintptr_t>(&cache_interface));
      }
      Ort::ConstEpDevice webgpu_device{nullptr};
      if (!plugin_path.empty()) {
        env.RegisterExecutionProviderLibrary("external_dawn_webgpu", plugin_path);
      }
      if (!plugin_path.empty() || static_plugin_mode) {
        for (const auto& ep_device : env.GetEpDevices()) {
          if (std::string(ep_device.EpName()) == "WebGpuExecutionProvider") {
            webgpu_device = ep_device;
            break;
          }
        }
        if (!webgpu_device) {
          throw std::runtime_error("External Dawn WebGPU plugin device was not found.");
        }
      }
      auto append_provider = [&](Ort::SessionOptions& options,
                                 const std::unordered_map<std::string, std::string>& settings) {
        if (webgpu_device) {
          options.AppendExecutionProvider_V2(env, {webgpu_device}, settings);
        } else {
          options.AppendExecutionProvider("WebGPU", settings);
        }
      };
      append_provider(session_options, provider_options);
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
        if (ort_owned_cache_mode) {
          Ort::SessionOptions auxiliary_options;
          auxiliary_options.DisableMemPattern();
          auxiliary_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
          auto auxiliary_provider_options = provider_options;
          auxiliary_provider_options.erase(ORT_WEBGPU_DAWN_CACHE_INTERFACE_OPTION);
          auxiliary_provider_options["enableGraphCapture"] = "0";
          append_provider(auxiliary_options, auxiliary_provider_options);
          Ort::Session auxiliary{env, MODEL_DATA, sizeof(MODEL_DATA), auxiliary_options};
          session = Ort::Session{nullptr};
          session_options = Ort::SessionOptions{nullptr};
          auto auxiliary_outputs = auxiliary.Run(
              Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
          if (auxiliary_outputs.size() != 1 ||
              auxiliary_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != input_data.size()) {
            throw std::runtime_error("Unexpected auxiliary Abs output shape.");
          }
          const float* auxiliary_data = auxiliary_outputs[0].GetTensorData<float>();
          for (size_t index = 0; index < input_data.size(); ++index) {
            if (auxiliary_data[index] != std::abs(input_data[index])) {
              throw std::runtime_error("Unexpected auxiliary Abs output value.");
            }
          }
          std::cout << "Auxiliary WebGPU session inherited the cache after the configuring session was released."
                    << std::endl;
        }
        if (!plugin_path.empty()) {
          std::cout << "WebGPU plugin EP was registered and selected explicitly." << std::endl;
        }
        if (host_device_mode) {
          Ort::SessionOptions default_options;
          default_options.DisableMemPattern();
          default_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
          append_provider(default_options, {{"dawnProcTable", provider_options.at("dawnProcTable")}});
          Ort::Session default_session{env, MODEL_DATA, sizeof(MODEL_DATA), default_options};
          Ort::MemoryInfo default_memory{"WebGPU_Buffer", OrtDeviceAllocator, 0, OrtMemTypeDefault};
          Ort::MemoryInfo custom_memory{"WebGPU_Buffer", OrtDeviceAllocator, custom_id, OrtMemTypeDefault};
          Ort::Allocator default_allocator{default_session, default_memory};
          Ort::Allocator custom_allocator{session, custom_memory};
          if (default_allocator.GetInfo().GetDeviceId() != 0 ||
              custom_allocator.GetInfo().GetDeviceId() != custom_id) {
            throw std::runtime_error("Default and host-owned WebGPU allocators did not retain distinct context IDs.");
          }
          auto default_tensor = Ort::Value::CreateTensor<float>(default_allocator, shape.data(), shape.size());
          auto custom_tensor = Ort::Value::CreateTensor<float>(custom_allocator, shape.data(), shape.size());
          for (const bool custom : {false, true}) {
            auto& current_session = custom ? session : default_session;
            auto& allocator = custom ? custom_allocator : default_allocator;
            auto& gpu_input = custom ? custom_tensor : default_tensor;
            auto& foreign_tensor = custom ? default_tensor : custom_tensor;
            auto& gpu_memory = custom ? custom_memory : default_memory;
            auto gpu_copy = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
            Ort::ThrowOnError(env.CopyTensor(input, gpu_input, nullptr));
            Ort::ThrowOnError(env.CopyTensor(gpu_input, gpu_copy, nullptr));
            if (env.CopyTensor(gpu_input, foreign_tensor, nullptr).IsOK()) {
              throw std::runtime_error("A WebGPU copy between distinct context IDs was incorrectly accepted.");
            }
            Ort::IoBinding binding{current_session};
            binding.BindInput("x", gpu_copy);
            binding.BindOutput("y", gpu_memory);
            current_session.Run(Ort::RunOptions{nullptr}, binding);
            binding.SynchronizeOutputs();
            auto gpu_outputs = binding.GetOutputValues();
            if (gpu_outputs.size() != 1 ||
                gpu_outputs[0].GetTensorMemoryInfo().GetDeviceId() != (custom ? custom_id : 0)) {
              throw std::runtime_error("A WebGPU output was allocated in the wrong context.");
            }
            std::array<float, 60> copied_output{};
            auto cpu_output = Ort::Value::CreateTensor<float>(
                memory_info, copied_output.data(), copied_output.size(), shape.data(), shape.size());
            Ort::ThrowOnError(env.CopyTensor(gpu_outputs[0], cpu_output, nullptr));
            for (size_t index = 0; index < copied_output.size(); ++index) {
              if (copied_output[index] != std::abs(input_data[index])) {
                throw std::runtime_error("Concurrent WebGPU contexts produced an incorrect GPU-bound output.");
              }
            }
          }
          std::cout << "Concurrent default and host-owned WebGPU context allocation and copies passed." << std::endl;
        }
        if (ort_owned_cache_mode) {
          session = Ort::Session{nullptr};
          session_options = Ort::SessionOptions{nullptr};
          env = Ort::Env{nullptr};
          host_cache.ValidatePass(cache_pass, hits_before, stores_before);
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
          host_cache.ValidatePass(cache_pass, hits_before, stores_before);
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
    } else if (cache_callback_failure &&
               std::string(ex.what()).find("A host shader-cache callback failed") != std::string::npos) {
      std::lock_guard<std::mutex> cache_lock(host_cache.mutex);
      if (host_cache.callback_failed.load() && host_cache.load_count > 0 && host_cache.store_count > 0) {
        std::cout << "Host cache load/store allocation failures were contained and reported as expected." << std::endl;
        retval = 0;
      } else {
        std::cerr << "The expected load/store callback failures were not both observed." << std::endl;
        retval = -1;
      }
    } else {
      std::cerr << "Unexpected exception." << std::endl;
      retval = -1;
    }
  }

  return retval;
}
