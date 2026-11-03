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
    if (argc < 4) throw std::runtime_error("Usage: matmul_n96_compare plugin profile-prefix fixture...");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "matmul-n96"};
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
    Ort::AllocatorWithDefaultOptions allocator;
    for (int arg = 3; arg < argc; ++arg) {
      const std::filesystem::path path{argv[arg]};
      auto configured = options.Clone();
      configured.EnableProfiling((std::string(argv[2]) + "-" + path.stem().string()).c_str());
      Ort::Session session{env, path.c_str(), configured};
      const auto check = session.GetModelMetadata().LookupCustomMetadataMapAllocated("reference_check", allocator);
      if (!check || (std::string(check.get()) != "exact" && std::string(check.get()) != "candidate_close"))
        throw std::runtime_error("Missing or invalid reference-check contract");
      const bool exact = std::string(check.get()) == "exact";
      double maximum_error[2]{};
      size_t nonfinite[2]{};
      int comparisons = 0;
      for (int round = 0; round < 4; ++round) {
        std::vector<unsigned char> control;
        for (int mode = 0; mode <= 1; ++mode) {
          if (setenv("ORT_WEBGPU_MATMUL_GEMV_N96", mode ? "1" : "0", 1))
            throw std::runtime_error("Cannot set GEMV experiment");
          auto output = session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, names, 2);
          const auto actual_info = output[0].GetTensorTypeAndShapeInfo();
          const auto expected_info = output[1].GetTensorTypeAndShapeInfo();
          if (actual_info.GetShape() != expected_info.GetShape() ||
              actual_info.GetElementType() != expected_info.GetElementType())
            throw std::runtime_error("Output contract mismatch");
          const bool half = actual_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
          if (!half && actual_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Unsupported fixture data type");
          const size_t count = actual_info.GetElementCount();
          for (size_t i = 0; i < count; ++i) {
            const float actual = half ? output[0].GetTensorData<Ort::Float16_t>()[i].ToFloat()
                                      : output[0].GetTensorData<float>()[i];
            const float expected = half ? output[1].GetTensorData<Ort::Float16_t>()[i].ToFloat()
                                        : output[1].GetTensorData<float>()[i];
            if (!std::isfinite(expected)) throw std::runtime_error("Nonfinite independent reference");
            if (!std::isfinite(actual)) {
              ++nonfinite[mode];
              if (mode || exact) throw std::runtime_error("Nonfinite qualified output");
              continue;
            }
            const double error = std::abs(double(actual) - expected);
            maximum_error[mode] = std::max(maximum_error[mode], error);
            if ((exact && error != 0) ||
                (mode && error > 0.001 + 0.001 * std::abs(double(expected))))
              throw std::runtime_error(path.stem().string() + ": independent reference mismatch");
          }
          const auto* raw = static_cast<const unsigned char*>(output[0].GetTensorRawData());
          const std::vector<unsigned char> bytes(raw, raw + count * (half ? 2 : 4));
          if (!mode) control = bytes;
          else if (exact && control != bytes) throw std::runtime_error("Fallback/exact output changed");
          ++comparisons;
        }
      }
      std::cout << "PASS " << path.stem().string() << " comparisons=" << comparisons
                << " contract=" << check.get() << " control_max_abs=" << maximum_error[0]
                << " candidate_max_abs=" << maximum_error[1]
                << " control_nonfinite=" << nonfinite[0] << " candidate_nonfinite=" << nonfinite[1]
                << std::endl;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << std::endl;
    return 1;
  }
}
