#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

struct InputStorage {
  std::vector<Ort::Float16_t> half;
  std::vector<float> single;
  std::vector<int64_t> integer;
};

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("Usage: benchmark provider fixture report profiling-prefix-or-dash");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "hc-small-cta-benchmark"};
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
        {{"enableInt64", "1"}, {"validationMode", "full"}, {"maxNumPendingDispatches", "64"},
         {"enableGraphCapture", "0"}});
    Ort::Session session{env, argv[2], options};
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_strings, output_strings;
    for (size_t i = 0; i < session.GetInputCount(); ++i)
      input_strings.emplace_back(session.GetInputNameAllocated(i, allocator).get());
    for (size_t i = 0; i < session.GetOutputCount(); ++i)
      output_strings.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
    if (output_strings.size() != 4) throw std::runtime_error("Expected four HC outputs");
    std::vector<const char*> names, output_names;
    for (const auto& name : input_strings) names.push_back(name.c_str());
    for (const auto& name : output_strings) output_names.push_back(name.c_str());
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<InputStorage> data(names.size());
    std::vector<Ort::Value> inputs;
    for (size_t i = 0; i < names.size(); ++i) {
      auto type_info = session.GetInputTypeInfo(i);
      auto info = type_info.GetTensorTypeAndShapeInfo();
      const auto shape = info.GetShape();
      const size_t count = info.GetElementCount();
      if (i == 0 && (count != 10240 || info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16))
        throw std::runtime_error("Expected canonical one-row HC input");
      switch (info.GetElementType()) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
          data[i].half.resize(count);
          inputs.emplace_back(Ort::Value::CreateTensor<Ort::Float16_t>(
              memory, data[i].half.data(), count, shape.data(), shape.size()));
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
          data[i].single.resize(count);
          inputs.emplace_back(Ort::Value::CreateTensor<float>(
              memory, data[i].single.data(), count, shape.data(), shape.size()));
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
          if (count != 2) throw std::runtime_error("Expected two split sizes");
          data[i].integer = {320, 4};
          inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
              memory, data[i].integer.data(), count, shape.data(), shape.size()));
          break;
        default: throw std::runtime_error("Unexpected HC fixture input type");
      }
    }
    std::vector<double> seconds;
    uint64_t hash = 14695981039346656037ull;
    rusage before{};
    for (int round = -16; round < 128; ++round) {
      const int input_round = round < 0 ? round + 16 : round;
      for (size_t input = 0; input < data.size(); ++input) {
        auto value = [&](size_t i) {
          const float v = 0.125f * std::sin(float((i + 1) * (input_round + 1)) * 0.07f);
          return input_strings[input] == "scale" ? 0.75f + v : v;
        };
        for (size_t i = 0; i < data[input].half.size(); ++i)
          data[input].half[i] = Ort::Float16_t(value(i));
        for (size_t i = 0; i < data[input].single.size(); ++i) data[input].single[i] = value(i);
      }
      if (round == 0 && getrusage(RUSAGE_SELF, &before) != 0)
        throw std::runtime_error("Cannot read initial resource usage");
      const auto start = std::chrono::steady_clock::now();
      auto outputs = session.Run(Ort::RunOptions{}, names.data(), inputs.data(), inputs.size(),
                                 output_names.data(), output_names.size());
      const auto end = std::chrono::steady_clock::now();
      size_t elements = 0;
      for (const auto& output : outputs) {
        auto info = output.GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
          throw std::runtime_error("Expected FP16 HC output");
        elements += info.GetElementCount();
        const auto* values = output.GetTensorData<Ort::Float16_t>();
        for (size_t i = 0; i < info.GetElementCount(); ++i)
          if (!std::isfinite(values[i].ToFloat())) throw std::runtime_error("Nonfinite HC output");
        if (round >= 0) {
          const auto* bytes = reinterpret_cast<const uint8_t*>(values);
          for (size_t i = 0; i < info.GetElementCount() * sizeof(Ort::Float16_t); ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
          }
        }
      }
      if (elements != 20804) throw std::runtime_error("Unexpected HC output element count");
      if (round >= 0) seconds.push_back(std::chrono::duration<double>(end - start).count());
    }
    rusage after{};
    if (getrusage(RUSAGE_SELF, &after) != 0) throw std::runtime_error("Cannot read final resource usage");
    std::string profile;
    if (profiling) {
      auto name = session.EndProfilingAllocated(allocator);
      profile = name.get();
    }
    std::ofstream report(argv[3]);
    report << std::setprecision(17)
           << "{\"status\":\"hc_operator_benchmark_completed\",\"warmups\":16,\"measured_runs\":128,"
           << "\"output_hash\":\"" << hash << "\",\"profiling\":" << (profiling ? "true" : "false")
           << ",\"profile_path\":\"" << profile << "\",\"run_seconds\":[";
    for (size_t i = 0; i < seconds.size(); ++i) report << (i ? "," : "") << seconds[i];
    report << "],\"measured_major_faults\":" << after.ru_majflt - before.ru_majflt << "}\n";
    if (!report) throw std::runtime_error("Cannot persist HC benchmark report");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
