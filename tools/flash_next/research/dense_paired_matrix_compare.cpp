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
    if (argc != 6) throw std::runtime_error("Usage: capture|compare provider fixture reference profile-prefix");
    const std::string mode = argv[1];
    if (mode != "capture" && mode != "compare") throw std::runtime_error("Invalid mode");
    const bool capture = mode == "capture";
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dense-paired-matrix"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[2]);
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
    options.AppendExecutionProvider_V2(env, devices,
        {{"enableInt64", "1"}, {"validationMode", "full"},
         {"maxNumPendingDispatches", "1"}, {"enableGraphCapture", "0"}});
    options.EnableProfiling(argv[5]);
    Ort::Session session{env, argv[3], options};
    const auto shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const auto output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 2 || shape[0] != 1 || shape[1] <= 0 ||
        output_shape.size() != 2 || output_shape[0] != 1 || output_shape[1] <= 0)
      throw std::runtime_error("Unexpected fixture shape");
    const size_t k = shape[1], n = output_shape[1];
    constexpr size_t rounds = 16;
    std::vector<Ort::Float16_t> reference(n * rounds), input(k);
    if (!capture) {
      std::ifstream file(argv[4], std::ios::binary);
      if (!file.read(reinterpret_cast<char*>(reference.data()), reference.size() * sizeof(reference[0])) ||
          file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Invalid reference payload");
    }
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* names[]{"a"};
    const char* outputs[]{"y"};
    size_t mismatches = 0;
    for (size_t round = 0; round < rounds; ++round) {
      const float amplitude = std::array<float, 4>{0.125f, 0.03125f, 0.000030517578125f, 1.0f}[round % 4];
      for (size_t i = 0; i < k; ++i)
        input[i] = Ort::Float16_t(amplitude * std::sin(float((i + 1) * (round + 1)) * 0.071f));
      auto value = Ort::Value::CreateTensor<Ort::Float16_t>(memory, input.data(), input.size(),
                                                         shape.data(), shape.size());
      for (int repeat = 0; repeat < 4; ++repeat) {
        auto actual = session.Run(Ort::RunOptions{}, names, &value, 1, outputs, 1);
        const auto info = actual[0].GetTensorTypeAndShapeInfo();
        if (info.GetShape() != output_shape || info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
          throw std::runtime_error("Unexpected output");
        const auto* observed = actual[0].GetTensorData<Ort::Float16_t>();
        for (size_t i = 0; i < n; ++i) {
          auto& expected = reference[round * n + i];
          if (capture && repeat == 0) expected = observed[i];
          if (!std::isfinite(expected.ToFloat()) || !std::isfinite(observed[i].ToFloat()))
            throw std::runtime_error("Non-finite result");
          if (std::memcmp(&expected, &observed[i], sizeof(expected)) != 0) {
            if (capture) throw std::runtime_error("Original reference is not repeatable");
            if (mismatches < 8)
              std::cerr << "MISMATCH round=" << round << " repeat=" << repeat << " col=" << i
                        << " expected_bits=" << expected.val << " actual_bits=" << observed[i].val << '\n';
            ++mismatches;
          }
        }
      }
    }
    if (capture) {
      std::ofstream file(argv[4], std::ios::binary | std::ios::trunc);
      file.write(reinterpret_cast<const char*>(reference.data()), reference.size() * sizeof(reference[0]));
      file.flush();
      if (!file.good()) throw std::runtime_error("Failed to save reference");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    auto profile = session.EndProfilingAllocated(allocator);
    std::cout << "PROFILE " << profile.get() << '\n';
    std::cout << "DENSE comparisons=" << n * rounds * 4 << " mismatches=" << mismatches
              << " K=" << k << " N=" << n << std::endl;
    return mismatches == 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
