#include <onnxruntime_cxx_api.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 10) throw std::runtime_error("Usage: dot4_tree_compare provider eight-fixtures...");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dot4-tree"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[1]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty()) throw std::runtime_error("No WebGPU device");
    size_t comparisons = 0;
    std::array<size_t, 4> mismatches{};
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
        if (info.GetShape() != std::vector<int64_t>{16, 80} ||
            info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
          throw std::runtime_error("Unexpected diagnostic output");
        const auto* values = output[0].GetTensorData<Ort::Float16_t>();
        for (int row = 0; row < 16; ++row) {
          for (int col = 0; col < 16; ++col) {
            const auto& scalar = values[row * 80 + 64 + col];
            if (!std::isfinite(scalar.ToFloat())) throw std::runtime_error("Non-finite dot4");
            ++comparisons;
            for (int pattern = 0; pattern < 4; ++pattern) {
              const auto& tree = values[row * 80 + pattern * 16 + col];
              if (!std::isfinite(tree.ToFloat())) throw std::runtime_error("Non-finite tree result");
              if (std::memcmp(&tree, &scalar, sizeof(tree)) != 0) {
                if (mismatches[pattern] < 4)
                  std::cerr << "MISMATCH pattern=" << pattern << " file=" << argv[file]
                            << " row=" << row << " col=" << col << " tree=" << tree.ToFloat()
                            << " dot4=" << scalar.ToFloat() << '\n';
                ++mismatches[pattern];
              }
            }
          }
        }
      }
    }
    bool exact = false;
    for (int pattern = 0; pattern < 4; ++pattern) {
      std::cout << "TREE pattern=" << pattern << " comparisons=" << comparisons
                << " mismatches=" << mismatches[pattern] << std::endl;
      exact |= mismatches[pattern] == 0;
    }
    if (!exact) return 2;
    std::cout << "PASS at least one tree matches all diagnostic finite bits; not model qualification"
              << std::endl;
    return 0;
  } catch (const Ort::Exception& error) {
    std::cerr << "FAIL ORT " << error.what() << std::endl;
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
