// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include "core/session/onnxruntime_cxx_api.h"

namespace {

int ParseCount(const char* text, bool positive) {
  size_t end = 0;
  const int value = std::stoi(text, &end);
  if (text[end] != '\0' || value < (positive ? 1 : 0)) {
    throw std::invalid_argument("Expected a nonnegative warmup / positive repeat count.");
  }
  return value;
}

int64_t Timestamp() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void Copy(Ort::Env& env, const Ort::Value& from, Ort::Value& to) {
  const OrtValue* source = from;
  OrtValue* target = to;
  Ort::ThrowOnError(Ort::GetApi().CopyTensors(env, &source, &target, nullptr, 1));
}

#ifdef _WIN32
void PrintLoadedModule(const char* label, const wchar_t* name) {
  const HMODULE module = GetModuleHandleW(name);
  std::wstring path(32768, L'\0');
  const DWORD length = module ? GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size())) : 0;
  if (length == 0 || length == path.size()) {
    throw std::runtime_error("Cannot identify the loaded benchmark binary.");
  }
  path.resize(length);
  std::cout << label << '=' << std::filesystem::path(path).string() << '\n';
}
#endif

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 9) {
      throw std::invalid_argument(
          "usage: diagnostic model input_prefix output_prefix warmup repeat validation fp32_acc mode");
    }
    const std::string input_prefix = argv[2], prefix = argv[3], mode = argv[8];
    if (mode != "auto" && mode != "regular" && mode != "simt") {
      throw std::invalid_argument("mode must be auto, regular, or simt");
    }
#ifdef _WIN32
    if (!SetEnvironmentVariableA("ORT_WEBGPU_SMALL_M_DIAGNOSTIC", mode.c_str())) {
#else
    if (setenv("ORT_WEBGPU_SMALL_M_DIAGNOSTIC", mode.c_str(), 1) != 0) {
#endif
      throw std::runtime_error("Cannot set the explicit diagnostic selection.");
    }
    const int warmup = ParseCount(argv[4], false), repeat = ParseCount(argv[5], true);
    std::cout << "ORT_BUILD=" << Ort::GetBuildInfoString() << '\n';
    Ort::Env env(ORT_LOGGING_LEVEL_INFO, "small-m-simt");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    const auto profile_prefix = std::filesystem::path(prefix);
    options.EnableProfiling(profile_prefix.c_str());
#ifdef _WIN32
    constexpr const char* backend = "D3D12";
#else
    constexpr const char* backend = "Vulkan";
#endif
    const std::unordered_map<std::string, std::string> provider_options{{"dawnBackendType", backend},
                                                                        {"enableGraphCapture", "0"},
                                                                        {"enableRobustness", "0"},
                                                                        {"validationMode", argv[6]},
                                                                        {"enableMatmulFp32Accumulation", argv[7]}};
#ifdef ORT_WEBGPU_BENCHMARK_PLUGIN_PATH
    std::cout << "ORT_PROVIDER_MODE=shared-plugin\n";
    env.RegisterExecutionProviderLibrary("small_m_simt", ORT_TSTR_ON_MACRO(ORT_WEBGPU_BENCHMARK_PLUGIN_PATH));
#elif defined(ORT_WEBGPU_BENCHMARK_STATIC_PLUGIN)
    std::cout << "ORT_PROVIDER_MODE=static-plugin\n";
#else
  std::cout << "ORT_PROVIDER_MODE=in-tree\n";
#endif
#if defined(ORT_WEBGPU_BENCHMARK_PLUGIN_PATH) || defined(ORT_WEBGPU_BENCHMARK_STATIC_PLUGIN)
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string_view{device.EpName()} == "WebGpuExecutionProvider" &&
          device.Device().Type() == OrtHardwareDeviceType_GPU) {
        devices.push_back(device);
      }
    }
    if (devices.size() != 1) {
      throw std::runtime_error("Benchmark requires exactly one WebGPU hardware GPU; refuse ambiguous/software selection.");
    }
    options.AppendExecutionProvider_V2(env, devices, provider_options);
    std::cout << "ADAPTER_VENDOR_ID=" << devices[0].Device().VendorId()
              << "\nADAPTER_DEVICE_ID=" << devices[0].Device().DeviceId() << '\n';
#else
    options.AppendExecutionProvider("WebGPU", provider_options);
#endif
    const auto model_path = std::filesystem::path(argv[1]);
    Ort::Session session(env, model_path.c_str(), options);
#ifdef _WIN32
    PrintLoadedModule("ORT_CORE_PATH", L"onnxruntime.dll");
#ifdef ORT_WEBGPU_BENCHMARK_PLUGIN_PATH
    PrintLoadedModule("ORT_PLUGIN_PATH", ORT_TSTR_ON_MACRO(ORT_WEBGPU_BENCHMARK_PLUGIN_PATH));
#endif
#endif
    Ort::AllocatorWithDefaultOptions cpu_allocator;
    Ort::MemoryInfo gpu_info("WebGPU_Buf", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    Ort::Allocator gpu_allocator(session, gpu_info);
    Ort::IoBinding binding(session);
    std::vector<Ort::Value> inputs, outputs;
    std::vector<std::string> output_names;
    inputs.reserve(session.GetInputCount());
    outputs.reserve(session.GetOutputCount());
    output_names.reserve(session.GetOutputCount());
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
      const auto name = session.GetInputNameAllocated(i, cpu_allocator);
      const auto type = session.GetInputTypeInfo(i);
      const auto info = type.GetTensorTypeAndShapeInfo();
      const auto shape = info.GetShape();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        throw std::invalid_argument("Diagnostic runner expects FP16 activation inputs.");
      }
      auto cpu = Ort::Value::CreateTensor<Ort::Float16_t>(cpu_allocator, shape.data(), shape.size());
      const size_t bytes = info.GetElementCount() * sizeof(Ort::Float16_t);
      std::ifstream stream(input_prefix + "-" + name.get() + ".bin", std::ios::binary);
      stream.read(reinterpret_cast<char*>(cpu.GetTensorMutableData<Ort::Float16_t>()),
                  static_cast<std::streamsize>(bytes));
      if (!stream || stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Incorrect input file size for " + std::string{name.get()});
      }
      inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(gpu_allocator, shape.data(), shape.size()));
      Copy(env, cpu, inputs.back());
      binding.BindInput(name.get(), inputs.back());
    }
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
      output_names.emplace_back(session.GetOutputNameAllocated(i, cpu_allocator).get());
      const auto type = session.GetOutputTypeInfo(i);
      const auto info = type.GetTensorTypeAndShapeInfo();
      const auto shape = info.GetShape();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        throw std::invalid_argument("Diagnostic runner expects FP16 outputs.");
      }
      outputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(gpu_allocator, shape.data(), shape.size()));
      binding.BindOutput(output_names.back().c_str(), outputs.back());
    }
    if (outputs.empty()) {
      throw std::invalid_argument("Diagnostic model needs at least one output.");
    }
    const auto last_shape = outputs.back().GetTensorTypeAndShapeInfo().GetShape();
    auto fence = Ort::Value::CreateTensor<Ort::Float16_t>(cpu_allocator, last_shape.data(), last_shape.size());
    Ort::RunOptions run;
    run.SetRunLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);
    const auto run_and_wait = [&]() {
      session.Run(run, binding);
      run.SetRunLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
      Copy(env, outputs.back(), fence);
    };
    for (int i = 0; i < warmup; ++i) {
      run_and_wait();
    }
    std::cout << "RESIDENT_INPUTS=" << inputs.size() << "\nRESIDENT_OUTPUTS=" << outputs.size() << '\n';
    std::cout << "MEASURE_START=" << Timestamp() << '\n';
    for (int i = 0; i < repeat; ++i) {
      run_and_wait();
    }
    std::cout << "MEASURE_END=" << Timestamp() << '\n';
    const auto profile_path = session.EndProfilingAllocated(cpu_allocator);
    std::cout << "PROFILE=" << std::filesystem::path(profile_path.get()).string() << '\n';
    std::ofstream output(prefix + "-outputs.bin", std::ios::binary);
    std::ofstream index(prefix + "-outputs.tsv");
    size_t offset = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto shape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
      auto cpu = Ort::Value::CreateTensor<Ort::Float16_t>(cpu_allocator, shape.data(), shape.size());
      Copy(env, outputs[i], cpu);
      const size_t bytes = cpu.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(Ort::Float16_t);
      output.write(reinterpret_cast<const char*>(cpu.GetTensorData<Ort::Float16_t>()),
                   static_cast<std::streamsize>(bytes));
      index << output_names[i] << '\t' << offset << '\t' << bytes << '\n';
      offset += bytes;
    }
    if (!output || !index) {
      throw std::runtime_error("Cannot write diagnostic outputs/index.");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
