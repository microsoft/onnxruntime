#include <onnxruntime_cxx_api.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 10) throw std::runtime_error("Usage: pair_matrix_compare provider eight-fixtures...");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dot4-pair-matrix"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[1]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty()) throw std::runtime_error("No WebGPU device");
    size_t comparisons = 0, both_native = 0, corrected_pairs = 0, small_pairs = 0, boundary_pairs = 0;
    std::array<size_t, 2> mismatches{};
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
        if (info.GetShape() != std::vector<int64_t>{16, 64} ||
            info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
          throw std::runtime_error("Unexpected diagnostic output");
        const auto* values = output[0].GetTensorData<Ort::Float16_t>();
        for (int row = 0; row < 16; ++row) {
          for (int col = 0; col < 16; ++col) {
            const auto& scalar = values[row * 64 + 32 + col];
            if (!std::isfinite(scalar.ToFloat())) throw std::runtime_error("Non-finite dot4 reference");
            const float mask_value = values[row * 64 + 48 + col].ToFloat();
            if (!std::isfinite(mask_value) || mask_value < 0 || mask_value > 63 ||
                std::trunc(mask_value) != mask_value)
              throw std::runtime_error("Invalid correction mask");
            const auto mask = static_cast<unsigned>(mask_value);
            if ((mask & 3u) != (((mask >> 2u) | (mask >> 4u)) & 3u))
              throw std::runtime_error("Correction mask disagrees with reasons");
            ++comparisons;
            both_native += (mask & 3u) == 0u;
            corrected_pairs += (mask & 1u) != 0u;
            corrected_pairs += (mask & 2u) != 0u;
            small_pairs += (mask & 4u) != 0u;
            small_pairs += (mask & 8u) != 0u;
            boundary_pairs += (mask & 16u) != 0u;
            boundary_pairs += (mask & 32u) != 0u;
            for (int pattern = 0; pattern < 2; ++pattern) {
              const auto& candidate = values[row * 64 + pattern * 16 + col];
              if (!std::isfinite(candidate.ToFloat())) throw std::runtime_error("Non-finite matrix result");
              if (std::memcmp(&candidate, &scalar, sizeof(candidate)) != 0) {
                if (mismatches[pattern] < 4)
                  std::cerr << "MISMATCH pattern=" << pattern << " row=" << row << " col=" << col
                            << " candidate=" << candidate.ToFloat() << " dot4=" << scalar.ToFloat() << '\n';
                ++mismatches[pattern];
              }
            }
          }
        }
      }
    }
    for (int pattern = 0; pattern < 2; ++pattern)
      std::cout << "PAIR pattern=" << pattern << " comparisons=" << comparisons
                << " mismatches=" << mismatches[pattern] << std::endl;
    std::cout << "PATH both_native=" << both_native << " corrected_pairs=" << corrected_pairs
              << " small_pairs=" << small_pairs << " boundary_pairs=" << boundary_pairs << std::endl;
    return mismatches[1] == 0 ? 0 : 2;
  } catch (const Ort::Exception& error) {
    std::cerr << "FAIL ORT " << error.what() << std::endl;
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
