#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("Expected provider and fixture");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "hc-up-mismatch"};
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
    options.AppendExecutionProvider_V2(env, devices, {{"enableInt64", "1"},
        {"validationMode", "full"}, {"enableGraphCapture", "0"}, {"maxNumPendingDispatches", "64"}});
    const std::filesystem::path fixture{argv[2]};
    const auto reference = fixture.parent_path() / (fixture.stem().string() + "_reference.onnx");
    auto other = options.Clone();
    Ort::Session baseline{env, reference.c_str(), options};
    Ort::Session candidate{env, fixture.c_str(), other};
    const char* names[] = {"y"};
    auto expected = baseline.Run(Ort::RunOptions{}, nullptr, nullptr, 0, names, 1);
    auto actual = candidate.Run(Ort::RunOptions{}, nullptr, nullptr, 0, names, 1);
    const auto* a = actual[0].GetTensorData<Ort::Float16_t>();
    const auto* b = expected[0].GetTensorData<Ort::Float16_t>();
    size_t mismatches = 0;
    for (size_t i = 0; i < actual[0].GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
      if (std::memcmp(a + i, b + i, sizeof(*a)) == 0) continue;
      uint16_t abits, bbits;
      std::memcpy(&abits, a + i, 2);
      std::memcpy(&bbits, b + i, 2);
      if (mismatches < 10)
        std::cout << "element=" << i << " actual=" << std::hexfloat << a[i].ToFloat()
                  << " expected=" << b[i].ToFloat() << " bits=" << std::hex << abits
                  << "/" << bbits << std::defaultfloat << std::dec << "\n";
      ++mismatches;
    }
    std::cout << "mismatches=" << mismatches << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
