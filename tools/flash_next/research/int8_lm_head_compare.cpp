#include <onnxruntime_cxx_api.h>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "hc_original_reference_io.inc"

int main(int argc, char** argv) {
  try {
    if (argc != 6) throw std::runtime_error("Usage: capture|compare provider fixture reference profile-prefix");
    const std::string mode = argv[1];
    if (mode != "capture" && mode != "compare") throw std::runtime_error("Invalid reference mode");
    const bool capture = mode == "capture";
    if (capture && std::filesystem::exists(argv[4])) throw std::runtime_error("Existing original reference");
    std::fstream reference(argv[4], std::ios::binary | (capture ? std::ios::out : std::ios::in));
    if (!reference) throw std::runtime_error("Cannot open original reference");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "int8-lm-head"};
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
         {"maxNumPendingDispatches", "64"}, {"enableGraphCapture", "0"}});
    options.EnableProfiling(argv[5]);
    Ort::Session session{env, argv[3], options};
    const auto input_type = session.GetInputTypeInfo(0);
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    const auto shape = input_info.GetShape();
    const auto output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const bool half = input_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    if ((!half && input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
        shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0 ||
        output_shape.size() != 2 || output_shape[0] != shape[0] || output_shape[1] <= 0)
      throw std::runtime_error("Unexpected fixture shape/type");
    const size_t count = shape[0] * shape[1], output_count = output_shape[0] * output_shape[1];
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* names[]{"a"};
    const char* outputs[]{"y"};
    std::vector<Ort::Float16_t> h_input(count);
    std::vector<float> f_input(count);
    size_t comparisons = 0;
    for (size_t round = 0; round < 16; ++round) {
      const float amplitude = std::array<float, 4>{0.125f, 0.03125f, 0.000030517578125f, 1.0f}[round % 4];
      for (size_t i = 0; i < count; ++i) {
        const float value = i % 29 == 0 ? ((i + round) % 2 ? -0.0f : 0.0f) :
            amplitude * std::sin(float((i + 1) * (round + 1)) * 0.071f);
        h_input[i] = Ort::Float16_t(value);
        f_input[i] = value;
      }
      auto input = half ? Ort::Value::CreateTensor<Ort::Float16_t>(
          memory, h_input.data(), count, shape.data(), shape.size()) :
          Ort::Value::CreateTensor<float>(memory, f_input.data(), count, shape.data(), shape.size());
      std::vector<char> first;
      for (int repeat = 0; repeat < 4; ++repeat) {
        auto actual = session.Run(Ort::RunOptions{}, names, &input, 1, outputs, 1);
        const auto info = actual[0].GetTensorTypeAndShapeInfo();
        if (info.GetShape() != output_shape || info.GetElementType() != input_info.GetElementType())
          throw std::runtime_error("Unexpected output shape/type");
        const size_t bytes = output_count * (half ? 2 : 4);
        const auto* payload = static_cast<const char*>(actual[0].GetTensorRawData());
        if (repeat == 0) first.assign(payload, payload + bytes);
        if (std::memcmp(first.data(), payload, bytes) != 0)
          throw NumericMismatch("Output is not repeatable");
        for (size_t i = 0; i < output_count; ++i) {
          const float value = half ? actual[0].GetTensorData<Ort::Float16_t>()[i].ToFloat() :
                                     actual[0].GetTensorData<float>()[i];
          if (!std::isfinite(value)) throw NumericMismatch("Non-finite output");
        }
        TransferOriginalOutput(reference, capture, actual[0],
            std::string(argv[3]) + " round=" + std::to_string(round) + " repeat=" + std::to_string(repeat));
        comparisons += output_count;
      }
    }
    if (!capture && reference.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("Trailing original reference bytes");
    reference.close();
    if (reference.fail()) throw std::runtime_error("Cannot close original reference");
    Ort::AllocatorWithDefaultOptions allocator;
    auto profile = session.EndProfilingAllocated(allocator);
    std::cout << "PROFILE " << profile.get() << '\n';
    std::cout << "INT8_HEAD comparisons=" << comparisons << " mismatches=0"
              << " M=" << shape[0] << " K=" << shape[1] << " N=" << output_shape[1] << std::endl;
    return 0;
  } catch (const NumericMismatch& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
