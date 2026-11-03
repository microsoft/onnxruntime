#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("Usage: benchmark provider fixture report profiling-prefix-or-dash");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "fc2-private-a-benchmark"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[1]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty()) throw std::runtime_error("No WebGPU device");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    const bool profiling = std::string(argv[4]) != "-";
    if (profiling) options.EnableProfiling(argv[4]);
    options.AppendExecutionProvider_V2(env, devices,
        {{"enableInt64", "1"}, {"validationMode", "full"}, {"maxNumPendingDispatches", "64"}, {"enableGraphCapture", "0"}});
    Ort::Session session{env, argv[2], options};
    const char* input_names[] = {"x", "router"};
    const char* output_names[] = {"y"};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<std::vector<Ort::Float16_t>> data(2);
    std::vector<Ort::Value> inputs;
    for (size_t i = 0; i < 2; ++i) {
      auto type_info = session.GetInputTypeInfo(i);
      auto info = type_info.GetTensorTypeAndShapeInfo();
      const auto shape = info.GetShape();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
          info.GetElementCount() != (i == 0 ? 2560u : 512u))
        throw std::runtime_error("Expected one-token FP16 QMoE fixture");
      data[i].resize(info.GetElementCount());
      inputs.emplace_back(Ort::Value::CreateTensor<Ort::Float16_t>(
          memory, data[i].data(), data[i].size(), shape.data(), shape.size()));
    }
    std::vector<double> seconds;
    uint64_t hash = 14695981039346656037ull;
    rusage before{};
    for (int round = -16; round < 128; ++round) {
      const int input_round = round < 0 ? round + 16 : round;
      for (size_t i = 0; i < data[0].size(); ++i)
        data[0][i] = Ort::Float16_t(0.07f * std::sin(float((i + 1) * (input_round + 1)) * 0.07f));
      for (size_t i = 0; i < data[1].size(); ++i)
        data[1][i] = Ort::Float16_t(float(int((i * 73 + input_round * 37) % 509) - 254) * 0.02f);
      if (round == 0 && getrusage(RUSAGE_SELF, &before) != 0)
        throw std::runtime_error("Cannot read initial resource usage");
      const auto start = std::chrono::steady_clock::now();
      auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs.data(), 2, output_names, 1);
      const auto end = std::chrono::steady_clock::now();
      auto info = outputs[0].GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 || info.GetElementCount() != 2560)
        throw std::runtime_error("Wrong benchmark output shape/type");
      if (round >= 0) {
        seconds.push_back(std::chrono::duration<double>(end - start).count());
        const auto* bytes = reinterpret_cast<const uint8_t*>(outputs[0].GetTensorData<Ort::Float16_t>());
        for (size_t i = 0; i < 5120; ++i) {
          hash ^= bytes[i];
          hash *= 1099511628211ull;
        }
      }
    }
    rusage after{};
    if (getrusage(RUSAGE_SELF, &after) != 0)
      throw std::runtime_error("Cannot read final resource usage");
    std::string profile;
    if (profiling) {
      Ort::AllocatorWithDefaultOptions allocator;
      auto name = session.EndProfilingAllocated(allocator);
      profile = name.get();
    }
    std::ofstream report(argv[3]);
    report << std::setprecision(17)
           << "{\"status\":\"qmoe_operator_benchmark_completed\",\"warmups\":16,\"measured_runs\":128,"
           << "\"output_hash\":\"" << hash << "\",\"profiling\":" << (profiling ? "true" : "false")
           << ",\"profile_path\":\"" << profile << "\",\"run_seconds\":[";
    for (size_t i = 0; i < seconds.size(); ++i) report << (i ? "," : "") << seconds[i];
    report << "],\"measured_major_faults\":" << after.ru_majflt - before.ru_majflt << "}\n";
    if (!report) throw std::runtime_error("Cannot persist native benchmark report");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
