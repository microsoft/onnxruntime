#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 4) throw std::runtime_error("Usage: qmoe_subgroup_compare plugin profile-prefix fixture...");
    int first_fixture = 3;
    const char* flag = "ORT_WEBGPU_QMOE_SUBGROUP_GATE";
    std::string candidate = "1";
    if (argc > 5 && (std::string(argv[3]) == "--bind-group-cache" ||
                     std::string(argv[3]) == "--dynamic-uniforms")) {
      flag = std::string(argv[3]) == "--bind-group-cache"
                 ? "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES"
                 : "ORT_WEBGPU_DYNAMIC_UNIFORMS";
      candidate = argv[4];
      first_fixture = 5;
    }
    const bool retain_sessions = first_fixture < argc && std::string(argv[first_fixture]) == "--retain-sessions";
    if (retain_sessions) ++first_fixture;
    if (first_fixture >= argc) throw std::runtime_error("At least one reference fixture is required");
    Ort::Env env{first_fixture == 5 ? ORT_LOGGING_LEVEL_INFO : ORT_LOGGING_LEVEL_WARNING, "qmoe-subgroup"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[1]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty()) throw std::runtime_error("No WebGPU adapter");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    options.AppendExecutionProvider_V2(env, devices,
        {{"validationMode", "full"}, {"enableGraphCapture", "0"}});
    const char* names[] = {"Y", "expected"};
    std::vector<Ort::Session> retained;
    for (int arg = first_fixture; arg < argc; ++arg) {
      const std::filesystem::path path{argv[arg]};
      auto configured = options.Clone();
      if (!retain_sessions)
        configured.EnableProfiling((std::string(argv[2]) + "-" + path.stem().string()).c_str());
      Ort::Session session{env, path.c_str(), configured};
      double maximum_error = 0;
      for (int round = 0; round < 4; ++round) {
        std::vector<unsigned char> control;
        for (int mode = 0; mode <= 1; ++mode) {
          if (setenv(flag, mode ? candidate.c_str() : "0", 1))
            throw std::runtime_error("Cannot set requested experiment");
          if (round == 0)
            std::cout << "RUN " << path.stem().string() << " mode=" << mode << std::endl;
          auto output = session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, names, 2);
          const auto actual_info = output[0].GetTensorTypeAndShapeInfo();
          const auto expected_info = output[1].GetTensorTypeAndShapeInfo();
          if (actual_info.GetShape() != expected_info.GetShape() ||
              actual_info.GetElementType() != expected_info.GetElementType())
            throw std::runtime_error("Output contract mismatch");
          const bool half = actual_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
          if (!half && actual_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Unexpected fixture type");
          const size_t count = actual_info.GetElementCount();
          for (size_t i = 0; i < count; ++i) {
            const float actual = half ? output[0].GetTensorData<Ort::Float16_t>()[i].ToFloat()
                                      : output[0].GetTensorData<float>()[i];
            const float expected = half ? output[1].GetTensorData<Ort::Float16_t>()[i].ToFloat()
                                        : output[1].GetTensorData<float>()[i];
            const double error = std::abs(double(actual) - expected);
            if (!std::isfinite(actual) || !std::isfinite(expected) ||
                error > 0.001 + 0.001 * std::abs(double(expected)))
              throw std::runtime_error(path.stem().string() + ": independent reference mismatch");
            maximum_error = std::max(maximum_error, error);
          }
          const auto* raw = static_cast<const unsigned char*>(output[0].GetTensorRawData());
          const std::vector<unsigned char> bytes(raw, raw + count * (half ? 2 : 4));
          if (!mode) control = bytes;
          else if (control != bytes)
            throw std::runtime_error(path.stem().string() + ": exact off/on output parity failed");
        }
      }
      std::cout << "PASS " << path.stem().string() << " comparisons=8 max_abs=" << maximum_error
                << " exact_off_on=true" << std::endl;
      if (retain_sessions) retained.push_back(std::move(session));
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << std::endl;
    return 1;
  }
}
