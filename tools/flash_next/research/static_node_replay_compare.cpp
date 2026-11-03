#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct InputStorage {
  std::vector<Ort::Float16_t> half;
  std::vector<float> single;
  std::vector<int64_t> integer;
};

int main(int argc, char** argv) {
  try {
    if (argc < 3) throw std::runtime_error("Usage: native_compare provider fixture...");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "static-node-replay"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[1]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty()) throw std::runtime_error("No WebGPU device");
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;
    auto flag = [](const char* value) {
      if (setenv("ORT_WEBGPU_STATIC_NODE_REPLAY", value, 1) != 0)
        throw std::runtime_error("Cannot set static node replay flag");
    };
    size_t comparisons = 0;
    for (int file = 2; file < argc; ++file) {
      Ort::SessionOptions options;
      options.SetIntraOpNumThreads(1);
      options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
      options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      options.AppendExecutionProvider_V2(
          env, devices, {{"enableInt64", "1"}, {"validationMode", "full"},
                         {"maxNumPendingDispatches", file % 2 ? "1" : "64"},
                         {"enableGraphCapture", "0"}});
      Ort::Session session{env, argv[file], options};
      std::vector<std::string> input_strings, output_strings;
      for (size_t i = 0; i < session.GetInputCount(); ++i)
        input_strings.emplace_back(session.GetInputNameAllocated(i, allocator).get());
      for (size_t i = 0; i < session.GetOutputCount(); ++i)
        output_strings.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
      std::vector<const char*> names, outputs;
      for (const auto& name : input_strings) names.push_back(name.c_str());
      for (const auto& name : output_strings) outputs.push_back(name.c_str());
      std::vector<InputStorage> data(names.size());
      std::vector<Ort::Value> inputs;
      int64_t width = 0;
      for (size_t i = 0; i < names.size(); ++i) {
        auto type_info = session.GetInputTypeInfo(i);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        if (i == 0) width = shape.back();
        const size_t count = info.GetElementCount();
        switch (info.GetElementType()) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            data[i].half.resize(count);
            inputs.emplace_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                memory, data[i].half.data(), count, shape.data(), shape.size()));
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            data[i].single.resize(count);
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory, data[i].single.data(), count, shape.data(), shape.size()));
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            if (count != 2) throw std::runtime_error("Expected two split sizes");
            data[i].integer.resize(count);
            inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
                memory, data[i].integer.data(), count, shape.data(), shape.size()));
            break;
          default: throw std::runtime_error("Unexpected fixture input type");
        }
      }
      for (int round = 0; round < 128; ++round) {
        const char* exact_mode = round >= 64 && round < 96 ? "0" : "1";
        if (setenv("ORT_WEBGPU_EXACT_DENSE_MID", exact_mode, 1) != 0 ||
            setenv("ORT_WEBGPU_HC_EXACT_DENSE", exact_mode, 1) != 0)
          throw std::runtime_error("Cannot set exact dense selector flags");
        for (size_t index = 0; index < data.size(); ++index) {
          auto value = [&](size_t i) {
            const float v = 0.125f * std::sin(float((i + 1) * (round + 1)) * 0.07f);
            return input_strings[index] == "scale" ? 0.75f + v : v;
          };
          for (size_t i = 0; i < data[index].half.size(); ++i)
            data[index].half[i] = Ort::Float16_t(value(i));
          for (size_t i = 0; i < data[index].single.size(); ++i) data[index].single[i] = value(i);
          if (!data[index].integer.empty()) {
            const int64_t prefix = round < 24 ? width * 2 / 3 :
                1 + ((round < 120 ? (round - 24) % 96 : (round - 120) % 2) % (width - 1));
            data[index].integer[0] = prefix;
            data[index].integer[1] = width - prefix;
          }
        }
        flag("0");
        auto reference = session.Run(Ort::RunOptions{}, names.data(), inputs.data(), inputs.size(),
                                     outputs.data(), outputs.size());
        flag("1");
        auto actual = session.Run(Ort::RunOptions{}, names.data(), inputs.data(), inputs.size(),
                                  outputs.data(), outputs.size());
        for (size_t output = 0; output < outputs.size(); ++output) {
          auto expected_info = reference[output].GetTensorTypeAndShapeInfo();
          auto actual_info = actual[output].GetTensorTypeAndShapeInfo();
          if (actual_info.GetShape() != expected_info.GetShape() ||
              actual_info.GetElementType() != expected_info.GetElementType())
            throw std::runtime_error("Mismatching output shape/type");
          const bool half = expected_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
          if (!half && expected_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Unexpected output type");
          for (size_t i = 0; i < expected_info.GetElementCount(); ++i) {
            const bool finite = half ?
                std::isfinite(reference[output].GetTensorData<Ort::Float16_t>()[i].ToFloat()) &&
                std::isfinite(actual[output].GetTensorData<Ort::Float16_t>()[i].ToFloat()) :
                std::isfinite(reference[output].GetTensorData<float>()[i]) &&
                std::isfinite(actual[output].GetTensorData<float>()[i]);
            const size_t bytes = half ? sizeof(Ort::Float16_t) : sizeof(float);
            const auto* expected = static_cast<const unsigned char*>(reference[output].GetTensorRawData());
            const auto* observed = static_cast<const unsigned char*>(actual[output].GetTensorRawData());
            if (!finite || std::memcmp(expected + i * bytes, observed + i * bytes, bytes) != 0)
              throw std::runtime_error(std::string(argv[file]) + " finite/bitwise mismatch round=" +
                                       std::to_string(round) + " element=" + std::to_string(i));
            ++comparisons;
          }
        }
      }
      flag("2");
      bool rejected = false;
      try {
        session.Run(Ort::RunOptions{}, names.data(), inputs.data(), inputs.size(), outputs.data(), outputs.size());
      } catch (const Ort::Exception& error) {
        rejected = std::string(error.what()).find("must be 0 or 1") != std::string::npos;
      }
      if (!rejected) throw std::runtime_error("Invalid replay flag was not explicitly rejected");
      flag("1");
      std::cout << "PASS " << std::filesystem::path(argv[file]).stem().string()
                << " changing_rounds=128 invalid_flag_rejected=1" << std::endl;
    }
    std::cout << "PASS total_exact_elements=" << comparisons << "; CPU fallback disabled" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
