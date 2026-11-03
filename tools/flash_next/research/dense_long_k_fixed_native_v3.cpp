#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 6)
      throw std::runtime_error("Usage: provider capture|compare reference-dir profile-prefix fixture...");
    const std::string mode = argv[2];
    if (mode != "capture" && mode != "compare") throw std::runtime_error("Invalid native mode");
    const bool capture = mode == "capture";
    const std::filesystem::path directory = argv[3];
    if (!std::filesystem::is_directory(directory)) throw std::runtime_error("Missing reference directory");
    const char* fp32_env = std::getenv("ORT_TEST_DENSE_FP32_ACCUMULATION");
    const std::string fp32 = fp32_env == nullptr ? "0" : fp32_env;
    if (fp32 != "0" && fp32 != "1")
      throw std::runtime_error("FP32 accumulation selector must be 0 or 1");
    const char* profiling_env = std::getenv("ORT_TEST_DENSE_NATIVE_PROFILE");
    const std::string profiling = profiling_env == nullptr ? "" : profiling_env;
    if (profiling != "0" && profiling != "1")
      throw std::runtime_error("Explicit native profiling selector must be 0 or 1");
    const auto private_value = []() {
      const char* value = std::getenv("ORT_WEBGPU_DENSE_TILE_PARALLEL3_LONG_K");
      return std::string(value == nullptr ? "unset" : value);
    };
    const std::string initial_private = private_value();
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dense-long-k-fixed-native"};
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
    size_t total = 0;
    for (int file = 5; file < argc; ++file) {
      const auto fixture = std::filesystem::path(argv[file]);
      const auto reference = directory / (fixture.filename().string() + ".bin");
      if (capture && std::filesystem::exists(reference)) throw std::runtime_error("Reference already exists");
      std::ifstream input_file;
      std::ofstream output_file;
      if (capture) {
        output_file.open(reference, std::ios::binary | std::ios::out);
        if (!output_file.is_open()) throw std::runtime_error("Cannot create reference");
      } else {
        input_file.open(reference, std::ios::binary);
        if (!input_file.is_open()) throw std::runtime_error("Cannot open original reference");
      }
      Ort::SessionOptions options;
      options.SetIntraOpNumThreads(1);
      options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
      options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      const char* pending = file % 2 ? "1" : "64";
      options.AppendExecutionProvider_V2(
          env, devices, {{"enableInt64", "1"}, {"validationMode", "full"},
                         {"enableMatmulFp32Accumulation", fp32}, {"maxNumPendingDispatches", pending},
                         {"enableGraphCapture", "0"}});
      const std::string prefix = std::string(argv[4]) + "-" + fixture.stem().string();
      if (profiling == "1") options.EnableProfiling(prefix.c_str());
      Ort::Session session{env, argv[file], options};
      if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
        throw std::runtime_error("Expected one input and one output");
      const auto input_type = session.GetInputTypeInfo(0);
      const auto output_type = session.GetOutputTypeInfo(0);
      const auto input_info = input_type.GetTensorTypeAndShapeInfo();
      const auto output_info = output_type.GetTensorTypeAndShapeInfo();
      const auto shape = input_info.GetShape(), output_shape = output_info.GetShape();
      if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
          output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
          shape.size() < 2 || output_shape.size() != shape.size() ||
          !std::equal(shape.begin(), shape.end() - 1, output_shape.begin()) ||
          std::any_of(shape.begin(), shape.end(), [](int64_t dimension) { return dimension <= 0; }) ||
          std::any_of(output_shape.begin(), output_shape.end(), [](int64_t dimension) { return dimension <= 0; }))
        throw std::runtime_error("Unexpected finite native fixture type/rank/shape");
      std::vector<Ort::Float16_t> input(input_info.GetElementCount());
      std::vector<Ort::Float16_t> expected(output_info.GetElementCount());
      const size_t bytes = expected.size() * sizeof(expected[0]);
      auto tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
          memory, input.data(), input.size(), shape.data(), shape.size());
      const std::string input_name = session.GetInputNameAllocated(0, allocator).get();
      const std::string output_name = session.GetOutputNameAllocated(0, allocator).get();
      const char* inputs[]{input_name.c_str()};
      const char* outputs[]{output_name.c_str()};
      uint64_t fingerprint = 1469598103934665603ULL;
      for (int round = 0; round < 128; ++round) {
        const char* exact_mode = round >= 64 && round < 96 ? "0" : "1";
        if (setenv("ORT_WEBGPU_EXACT_DENSE_MID", exact_mode, 1) != 0 ||
            setenv("ORT_WEBGPU_HC_EXACT_DENSE", exact_mode, 1) != 0)
          throw std::runtime_error("Cannot set legacy exact-dense selector");
        for (size_t i = 0; i < input.size(); ++i)
          input[i] = Ort::Float16_t(0.125f * std::sin(float((i + 1) * (round + 1)) * 0.07f));
        auto actual = session.Run(Ort::RunOptions{}, inputs, &tensor, 1, outputs, 1);
        auto info = actual[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
            info.GetShape() != output_shape || info.GetElementCount() != expected.size())
          throw std::runtime_error("Actual output type/rank/shape/count mismatch");
        if (capture) {
          std::copy_n(actual[0].GetTensorData<Ort::Float16_t>(), expected.size(), expected.data());
          actual = session.Run(Ort::RunOptions{}, inputs, &tensor, 1, outputs, 1);
          info = actual[0].GetTensorTypeAndShapeInfo();
          if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
              info.GetShape() != output_shape || info.GetElementCount() != expected.size())
            throw std::runtime_error("Repeated original output type/rank/shape/count mismatch");
        } else if (!input_file.read(reinterpret_cast<char*>(expected.data()), bytes)) {
          throw std::runtime_error("Truncated original reference");
        }
        const auto* observed = actual[0].GetTensorData<Ort::Float16_t>();
        for (size_t i = 0; i < expected.size(); ++i) {
          if (!std::isfinite(expected[i].ToFloat()) || !std::isfinite(observed[i].ToFloat()) ||
              std::memcmp(&expected[i], &observed[i], sizeof(expected[i])) != 0)
            throw std::runtime_error(fixture.string() + " finite/bitwise mismatch round=" +
                                     std::to_string(round) + " element=" + std::to_string(i));
          fingerprint ^= expected[i].val;
          fingerprint *= 1099511628211ULL;
        }
        if (capture) {
          output_file.write(reinterpret_cast<const char*>(expected.data()), bytes);
          if (!output_file.good()) throw std::runtime_error("Failed to write original reference");
        }
      }
      if (capture) {
        output_file.flush();
        if (!output_file.good()) throw std::runtime_error("Failed to flush original reference");
        output_file.close();
        if (output_file.fail()) throw std::runtime_error("Failed to close original reference");
      } else if (input_file.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Trailing original reference bytes");
      }
      if (std::filesystem::file_size(reference) != bytes * 128)
        throw std::runtime_error("Wrong original reference size");
      if (profiling == "1") {
        auto profile = session.EndProfilingAllocated(allocator);
        std::cout << "PROFILE " << profile.get() << '\n';
      }
      const size_t count = expected.size() * 128;
      total += count;
      std::cout << "PASS " << fixture.filename().string() << " changing_rounds=128 finite_bitwise_elements="
                << count << " fingerprint=" << fingerprint << " pending=" << pending
                << " fp32=" << fp32 << " profiling=" << profiling << std::endl;
    }
    if (private_value() != initial_private) throw std::runtime_error("Private selector changed in-process");
    std::cout << "PASS total_exact_elements=" << total << "; CPU fallback disabled; fixed_process=1"
              << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
