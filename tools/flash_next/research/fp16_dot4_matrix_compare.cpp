#include <onnxruntime_cxx_api.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 3) throw std::runtime_error("Usage: dot4_compare provider fixture...");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dot4-matrix"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[1]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty()) throw std::runtime_error("No WebGPU device");
    size_t comparisons = 0, mismatches = 0;
    for (int file = 2; file < argc; ++file) {
      Ort::SessionOptions options;
      options.SetIntraOpNumThreads(1);
      options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
      options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      options.AppendExecutionProvider_V2(env, devices,
          {{"enableInt64", "1"}, {"validationMode", "full"},
           {"maxNumPendingDispatches", "1"}, {"enableGraphCapture", "0"}});
      Ort::Session session{env, argv[file], options};
      const char* names[] = {"y"};
      for (int round = 0; round < 4; ++round) {
        auto output = session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, names, 1);
        auto info = output[0].GetTensorTypeAndShapeInfo();
        if (info.GetShape() != std::vector<int64_t>{16, 32} ||
            info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
          throw std::runtime_error("Unexpected diagnostic output");
        const auto* values = output[0].GetTensorData<Ort::Float16_t>();
        for (int row = 0; row < 16; ++row) {
          for (int col = 0; col < 16; ++col) {
            const auto& matrix = values[row * 32 + col];
            const auto& scalar = values[row * 32 + 16 + col];
            if (!std::isfinite(matrix.ToFloat()) || !std::isfinite(scalar.ToFloat()))
              throw std::runtime_error("Non-finite dot4 result");
            ++comparisons;
            if (std::memcmp(&matrix, &scalar, sizeof(matrix)) != 0) {
              if (mismatches < 8)
                std::cerr << "MISMATCH file=" << argv[file] << " row=" << row << " col=" << col
                          << " matrix=" << matrix.ToFloat() << " dot4=" << scalar.ToFloat() << '\n';
              ++mismatches;
            }
          }
        }
      }
    }
    std::cout << "DOT4 comparisons=" << comparisons << " mismatches=" << mismatches << std::endl;
    if (mismatches != 0) return 2;
    std::cout << "PASS finite bitwise padded matrix and WGSL dot4" << std::endl;
    return 0;
  } catch (const Ort::Exception& error) {
    std::cerr << "FAIL ORT " << error.what() << std::endl;
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
