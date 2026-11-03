#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "hc_original_reference_io.inc"

int main(int argc, char** argv) {
  try {
    if (argc != 7) throw std::runtime_error("Usage: capture|compare provider fixture reference profile-prefix pending");
    const std::string mode = argv[1], pending = argv[6];
    if ((mode != "capture" && mode != "compare") || (pending != "1" && pending != "64"))
      throw std::runtime_error("Invalid reference mode or pending window");
    const bool capture = mode == "capture";
    if (capture && std::filesystem::exists(argv[4])) throw std::runtime_error("Existing original reference");
    std::fstream reference(argv[4], std::ios::binary | (capture ? std::ios::out : std::ios::in));
    if (!reference) throw std::runtime_error("Cannot open original reference");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dsa-original-reference"};
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
         {"maxNumPendingDispatches", pending}, {"enableGraphCapture", "0"}});
    options.EnableProfiling(argv[5]);
    Ort::Session session{env, argv[3], options};
    if (session.GetInputCount() != 0 || session.GetOutputCount() != 1)
      throw std::runtime_error("Expected constant-input single-output fixture");
    const char* outputs[]{"y"};
    size_t comparisons = 0;
    std::vector<char> first;
    for (size_t repeat = 0; repeat < 9; ++repeat) {
      auto actual = session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, outputs, 1);
      const auto info = actual[0].GetTensorTypeAndShapeInfo();
      const auto type = info.GetElementType();
      const bool half = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
      if (!half && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("Unexpected output type");
      const size_t count = info.GetElementCount(), bytes = count * (half ? 2 : 4);
      const auto* payload = static_cast<const char*>(actual[0].GetTensorRawData());
      if (repeat == 0) first.assign(payload, payload + bytes);
      if (first.size() != bytes || std::memcmp(first.data(), payload, bytes) != 0)
        throw NumericMismatch("Output is not repeatable");
      for (size_t i = 0; i < count; ++i) {
        const float value = half ? actual[0].GetTensorData<Ort::Float16_t>()[i].ToFloat() :
                                  actual[0].GetTensorData<float>()[i];
        if (!std::isfinite(value)) throw NumericMismatch("Non-finite output");
      }
      TransferOriginalOutput(reference, capture, actual[0], std::string(argv[3]) + " repeat=" + std::to_string(repeat));
      comparisons += count;
    }
    if (!capture && reference.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("Trailing original reference bytes");
    reference.close();
    if (reference.fail()) throw std::runtime_error("Cannot close original reference");
    Ort::AllocatorWithDefaultOptions allocator;
    auto profile = session.EndProfilingAllocated(allocator);
    std::cout << "PROFILE " << profile.get() << '\n';
    std::cout << "DSA comparisons=" << comparisons << " mismatches=0 repeats=9" << std::endl;
    return 0;
  } catch (const NumericMismatch& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
