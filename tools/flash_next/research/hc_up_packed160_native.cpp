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
#include <sys/resource.h>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 6) throw std::runtime_error("Usage: provider capture|compare reference-dir profile-prefix fixture...");
    const std::string mode = argv[2];
    if (mode != "capture" && mode != "compare") throw std::runtime_error("Invalid native mode");
    const bool capture = mode == "capture";
    const std::filesystem::path directory = argv[3];
    if (!std::filesystem::is_directory(directory)) throw std::runtime_error("Missing reference directory");
    const auto setting = [](const char* name) {
      const char* value = std::getenv(name);
      return std::string(value == nullptr ? "unset" : value);
    };
    const std::string selected = setting("ORT_WEBGPU_HC_UP_PACKED160");
    const std::string profiling = setting("ORT_TEST_HC_UP_NATIVE_PROFILE");
    const std::string fp32 = setting("ORT_TEST_HC_UP_FP32_ACCUMULATION");
    if ((profiling != "0" && profiling != "1") || (fp32 != "0" && fp32 != "1"))
      throw std::runtime_error("Explicit profile and FP32 selectors must be 0 or 1");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "hc-up-packed160-native"};
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
      const char* pending = file % 2 ? "64" : "1";
      options.AppendExecutionProvider_V2(
          env, devices, {{"enableInt64", "1"}, {"validationMode", "full"},
                         {"enableMatmulFp32Accumulation", fp32}, {"maxNumPendingDispatches", pending},
                         {"enableGraphCapture", "0"}});
      const std::string prefix = std::string(argv[4]) + "-" + fixture.stem().string();
      if (profiling == "1") options.EnableProfiling(prefix.c_str());
      Ort::Session session{env, argv[file], options};
      if (session.GetInputCount() != 2 || session.GetOutputCount() != 1)
        throw std::runtime_error("Expected streams, activations and one output");
      const auto stream_type = session.GetInputTypeInfo(0);
      const auto activation_type = session.GetInputTypeInfo(1);
      const auto output_type = session.GetOutputTypeInfo(0);
      const auto stream_info = stream_type.GetTensorTypeAndShapeInfo();
      const auto activation_info = activation_type.GetTensorTypeAndShapeInfo();
      const auto output_info = output_type.GetTensorTypeAndShapeInfo();
      const auto shape = stream_info.GetShape();
      const auto activation_shape = activation_info.GetShape();
      const auto output_shape = output_info.GetShape();
      if (stream_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
          activation_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
          output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
          shape.size() < 2 || activation_shape.size() != shape.size() ||
          output_shape.size() != shape.size() || shape.back() != 10240 ||
          activation_shape.back() != 320 || output_shape.back() != 2560 ||
          !std::equal(shape.begin(), shape.end() - 1, activation_shape.begin()) ||
          !std::equal(shape.begin(), shape.end() - 1, output_shape.begin()) ||
          std::any_of(shape.begin(), shape.end(), [](int64_t dimension) { return dimension <= 0; }))
        throw std::runtime_error("Unexpected native input/output type/rank/shape");
      std::vector<Ort::Float16_t> streams(stream_info.GetElementCount());
      std::vector<Ort::Float16_t> activations(activation_info.GetElementCount());
      std::vector<Ort::Float16_t> expected(output_info.GetElementCount());
      const size_t bytes = expected.size() * sizeof(expected[0]);
      Ort::Value tensors[]{
          Ort::Value::CreateTensor<Ort::Float16_t>(
              memory, streams.data(), streams.size(), shape.data(), shape.size()),
          Ort::Value::CreateTensor<Ort::Float16_t>(
              memory, activations.data(), activations.size(), activation_shape.data(), activation_shape.size())};
      const std::string stream_name = session.GetInputNameAllocated(0, allocator).get();
      const std::string activation_name = session.GetInputNameAllocated(1, allocator).get();
      const std::string output_name = session.GetOutputNameAllocated(0, allocator).get();
      if (stream_name != "streams" || activation_name != "x" || output_name != "y")
        throw std::runtime_error("Unexpected native input/output names");
      const char* inputs[]{stream_name.c_str(), activation_name.c_str()};
      const char* outputs[]{output_name.c_str()};
      uint64_t fingerprint = 1469598103934665603ULL;
      struct rusage measured_start{}, measured_end{};
      size_t comparisons = 0;
      for (int round = 0; round < 144; ++round) {
        const char* exact_mode = round >= 64 && round < 96 ? "0" : "1";
        const char* parallel2 = profiling == "0" && round >= 64 && round < 96 ? "0" : "1";
        if (setenv("ORT_WEBGPU_EXACT_DENSE_MID", exact_mode, 1) != 0 ||
            setenv("ORT_WEBGPU_HC_EXACT_DENSE", exact_mode, 1) != 0 ||
            setenv("ORT_WEBGPU_HC_UP_PARALLEL2", parallel2, 1) != 0)
          throw std::runtime_error("Cannot set legacy selectors");
        for (size_t i = 0; i < streams.size(); ++i) {
          streams[i] = Ort::Float16_t(0.125f * std::sin(float((i + 1) * (round + 1)) * 0.07f));
          if (round % 32 == 0) streams[i].val = i % 2 ? 0x8000 : 0;
        }
        for (size_t i = 0; i < activations.size(); ++i) {
          activations[i] = Ort::Float16_t(0.125f * std::sin(float((i + 3) * (round + 1)) * 0.09f));
          if (round % 32 == 1) activations[i].val = i % 2 ? 0x8001 : 1;
        }
        if (!capture && !input_file.read(reinterpret_cast<char*>(expected.data()), bytes))
          throw std::runtime_error("Truncated original reference");
        if (round == 16 && getrusage(RUSAGE_SELF, &measured_start) != 0)
          throw std::runtime_error("getrusage start failed");
        for (int repeat = 0; repeat < 2; ++repeat) {
          auto actual = session.Run(Ort::RunOptions{}, inputs, tensors, 2, outputs, 1);
          const auto info = actual[0].GetTensorTypeAndShapeInfo();
          if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
              info.GetShape() != output_shape || info.GetElementCount() != expected.size())
            throw std::runtime_error("Actual output type/rank/shape/count mismatch");
          const auto* observed = actual[0].GetTensorData<Ort::Float16_t>();
          if (capture && repeat == 0) std::copy_n(observed, expected.size(), expected.data());
          for (size_t i = 0; i < expected.size(); ++i) {
            if (!std::isfinite(expected[i].ToFloat()) || !std::isfinite(observed[i].ToFloat()) ||
                std::memcmp(&expected[i], &observed[i], sizeof(expected[i])) != 0)
              throw std::runtime_error(fixture.string() + " finite/bitwise mismatch round=" +
                                       std::to_string(round) + " repeat=" + std::to_string(repeat) +
                                       " element=" + std::to_string(i));
            fingerprint ^= observed[i].val;
            fingerprint *= 1099511628211ULL;
            ++comparisons;
          }
          if (setting("ORT_WEBGPU_HC_UP_PACKED160") != selected)
            throw std::runtime_error("Private selector changed during process");
        }
        if (capture) {
          output_file.write(reinterpret_cast<const char*>(expected.data()), bytes);
          if (!output_file.good()) throw std::runtime_error("Failed to write original reference");
        }
      }
      if (getrusage(RUSAGE_SELF, &measured_end) != 0)
        throw std::runtime_error("getrusage end failed");
      if (measured_end.ru_majflt != measured_start.ru_majflt)
        throw std::runtime_error("Measured major faults");
      if (capture) {
        output_file.flush();
        if (!output_file.good()) throw std::runtime_error("Failed to flush original reference");
        output_file.close();
        if (output_file.fail()) throw std::runtime_error("Failed to close original reference");
      } else if (input_file.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Trailing original reference bytes");
      }
      if (profiling == "1") {
        const auto profile = session.EndProfilingAllocated(allocator);
        std::cout << "PROFILE " << profile.get() << std::endl;
      }
      std::cout << "PASS " << fixture.filename().string() << " comparisons=" << comparisons
                << " fingerprint=" << fingerprint << " profiling=" << profiling
                << " fp32=" << fp32 << " pending=" << pending
                << " warmup_pairs=16 measured_pairs=128 major_faults=0" << std::endl;
      total += comparisons;
    }
    std::cout << "PASS total_exact_elements=" << total << "; CPU fallback disabled" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
