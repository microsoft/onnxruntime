#include <onnxruntime_cxx_api.h>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 12) throw std::runtime_error("Usage: comparator capture|compare provider reference eight-fixtures...");
    const std::string mode = argv[1];
    if (mode != "capture" && mode != "compare") throw std::runtime_error("Invalid comparator mode");
    const bool capture = mode == "capture";
    static_assert(sizeof(Ort::Float16_t) == 2);
    std::vector<Ort::Float16_t> reference(2048);
    if (!capture) {
      std::ifstream input(argv[3], std::ios::binary);
      if (!input.read(reinterpret_cast<char*>(reference.data()), reference.size() * sizeof(reference[0])) ||
          input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Missing or wrong-size reference capture");
      for (const auto& value : reference)
        if (!std::isfinite(value.ToFloat())) throw std::runtime_error("Non-finite captured reference");
    }
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dot4-cross-shader"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[2]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty()) throw std::runtime_error("No WebGPU device");
    size_t comparisons = 0, both_native = 0, corrected_pairs = 0;
    std::array<size_t, 5> mismatches{};
    for (int file = 4; file < argc; ++file) {
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
        const auto info = output[0].GetTensorTypeAndShapeInfo();
        const int columns = capture ? 1552 : 160;
        if (info.GetShape() != std::vector<int64_t>{16, columns} ||
            info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
          throw std::runtime_error("Unexpected output shape or dtype");
        const auto* values = output[0].GetTensorData<Ort::Float16_t>();
        for (int row = 0; row < 16; ++row) {
          for (int col = 0; col < 16; ++col) {
            auto& expected = reference[(file - 4) * 256 + row * 16 + col];
            if (capture) {
              const auto& scalar = values[row * columns + 1536 + col];
              if (!std::isfinite(scalar.ToFloat())) throw std::runtime_error("Non-finite reference");
              if (round == 0) expected = scalar;
              if (std::memcmp(&expected, &scalar, sizeof(scalar)) != 0)
                throw std::runtime_error("Reference changed between repeats");
              for (int pattern : {47, 95}) {
                const auto& paired = values[row * columns + pattern * 16 + col];
                if (std::memcmp(&paired, &scalar, sizeof(scalar)) != 0)
                  throw std::runtime_error("Original enumeration exact pattern no longer matches");
              }
            } else {
              const float mask_value = values[row * columns + 48 + col].ToFloat();
              if (!std::isfinite(mask_value) || mask_value < 0 || mask_value > 63 ||
                  std::trunc(mask_value) != mask_value)
                throw std::runtime_error("Invalid correction mask");
              const auto mask = static_cast<unsigned>(mask_value);
              if ((mask & 3u) != (((mask >> 2u) | (mask >> 4u)) & 3u))
                throw std::runtime_error("Correction mask disagrees with reasons");
              both_native += (mask & 3u) == 0u;
              corrected_pairs += (mask & 1u) != 0u;
              corrected_pairs += (mask & 2u) != 0u;
              constexpr std::array<int, 5> planes{0, 1, 8, 9, 2};
              for (size_t kind = 0; kind < planes.size(); ++kind) {
                const auto& value = values[row * columns + planes[kind] * 16 + col];
                if (!std::isfinite(value.ToFloat())) throw std::runtime_error("Non-finite candidate");
                if (std::memcmp(&value, &expected, sizeof(value)) != 0) {
                  if (mismatches[kind] < 4)
                    std::cerr << "CROSS mismatch kind=" << kind << " seed=" << file - 4
                              << " row=" << row << " col=" << col << " value=" << value.ToFloat()
                              << " original=" << expected.ToFloat() << '\n';
                  ++mismatches[kind];
                }
              }
            }
            ++comparisons;
          }
        }
      }
    }
    if (capture) {
      std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
      output.write(reinterpret_cast<const char*>(reference.data()), reference.size() * sizeof(reference[0]));
      output.flush();
      if (!output.good()) throw std::runtime_error("Failed to persist reference");
      std::cout << "CAPTURE comparisons=" << comparisons << " elements=" << reference.size() << std::endl;
      return 0;
    }
    for (size_t kind = 0; kind < mismatches.size(); ++kind)
      std::cout << "CROSS kind=" << kind << " comparisons=" << comparisons
                << " mismatches=" << mismatches[kind] << std::endl;
    std::cout << "PATH both_native=" << both_native << " corrected_pairs=" << corrected_pairs << std::endl;
    return mismatches[1] == 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
