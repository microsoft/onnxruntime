#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
  try {
    if (argc != 4)
      throw std::runtime_error(
          "Usage: upload_compare plugin fixture_directory profile_prefix");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ordered-small-upload"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[1]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto &device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty())
      throw std::runtime_error("No WebGPU adapter");
    auto memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char *inputs[] = {"x"};
    const char *outputs[] = {"y"};
    for (bool half : {false, true}) {
      const std::string name = half ? "fp16" : "fp32";
      Ort::SessionOptions options;
      options.SetIntraOpNumThreads(1);
      options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_DISABLE_ALL);
      options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      options.AppendExecutionProvider_V2(env, devices,
                                         {{"enableInt64", "1"},
                                          {"validationMode", "full"},
                                          {"enableGraphCapture", "0"}});
      options.EnableProfiling((std::string(argv[3]) + "-" + name).c_str());
      Ort::Session session{
          env, (std::filesystem::path(argv[2]) / (name + ".onnx")).c_str(),
          options};
      for (int64_t size : {1, 3, 17, 257, 262143, 262144, 262145, 17, 1}) {
        if (half && size > 257)
          continue;
        std::vector<float> floats(static_cast<size_t>(size));
        std::vector<Ort::Float16_t> halves(static_cast<size_t>(size));
        const std::array<int64_t, 1> shape{size};
        for (int iteration = 0; iteration < 20; ++iteration) {
          const bool enabled = (iteration % 4 == 0 || iteration % 4 == 3);
          if (setenv("ORT_WEBGPU_SMALL_UPLOAD_WRITE_BUFFER",
                     enabled ? "1" : "0", 1)) {
            throw std::runtime_error("Cannot set upload experiment");
          }
          if (setenv("ORT_WEBGPU_SMALL_READBACK_CACHE",
                     (iteration % 4 < 2) ? "1" : "0", 1)) {
            throw std::runtime_error("Cannot set readback experiment");
          }
          for (int64_t i = 0; i < size; ++i) {
            floats[static_cast<size_t>(i)] =
                static_cast<float>((i + iteration) % 31) / 8;
            halves[static_cast<size_t>(i)] =
                Ort::Float16_t{floats[static_cast<size_t>(i)]};
          }
          auto input = half ? Ort::Value::CreateTensor<Ort::Float16_t>(
                                  memory, halves.data(), halves.size(),
                                  shape.data(), shape.size())
                            : Ort::Value::CreateTensor<float>(
                                  memory, floats.data(), floats.size(),
                                  shape.data(), shape.size());
          auto result =
              session.Run(Ort::RunOptions{}, inputs, &input, 1, outputs, 1);
          const auto info = result[0].GetTensorTypeAndShapeInfo();
          if (info.GetShape() != std::vector<int64_t>{size})
            throw std::runtime_error("Output shape mismatch");
          for (size_t i = 0; i < floats.size(); ++i) {
            const float expected = 2 * floats[i] + static_cast<float>(size);
            const float actual =
                half ? result[0].GetTensorData<Ort::Float16_t>()[i].ToFloat()
                     : result[0].GetTensorData<float>()[i];
            const float rounded =
                half ? Ort::Float16_t{expected}.ToFloat() : expected;
            if (actual != rounded) {
              throw std::runtime_error(
                  name + " size=" + std::to_string(size) +
                  " iteration=" + std::to_string(iteration) + " mismatch");
            }
          }
        }
        std::cout << "PASS " << name << " elements=" << size
                  << " alternating_upload_and_readback_modes=20 exact_outputs"
                  << std::endl;
      }
    }
    if (setenv("ORT_WEBGPU_SMALL_UPLOAD_WRITE_BUFFER", "1", 1) ||
        setenv("ORT_WEBGPU_SMALL_READBACK_CACHE", "1", 1)) {
      throw std::runtime_error("Cannot enable concurrent transfer experiments");
    }
    std::vector<std::future<void>> workers;
    for (int worker = 0; worker < 2; ++worker) {
      workers.push_back(std::async(std::launch::async, [&, worker] {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_DISABLE_ALL);
        options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        options.AppendExecutionProvider_V2(env, devices,
                                           {{"enableInt64", "1"},
                                            {"validationMode", "full"},
                                            {"enableGraphCapture", "0"}});
        Ort::Session session{
            env, (std::filesystem::path(argv[2]) / "fp32.onnx").c_str(),
            options};
        for (int iteration = 0; iteration < 8; ++iteration) {
          const int64_t count = (iteration % 2) ? 257 : 3;
          const std::array<int64_t, 1> shape{count};
          std::vector<float> values(static_cast<size_t>(count));
          for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>((i + iteration + worker) % 31) / 8;
          }
          auto input = Ort::Value::CreateTensor<float>(
              memory, values.data(), values.size(), shape.data(), shape.size());
          auto result =
              session.Run(Ort::RunOptions{}, inputs, &input, 1, outputs, 1);
          if (result[0].GetTensorTypeAndShapeInfo().GetShape() !=
              std::vector<int64_t>{count}) {
            throw std::runtime_error("Concurrent output shape mismatch");
          }
          for (size_t i = 0; i < values.size(); ++i) {
            if (result[0].GetTensorData<float>()[i] !=
                2 * values[i] + static_cast<float>(count)) {
              throw std::runtime_error(
                  "Concurrent staging reuse corrupted output");
            }
          }
        }
      }));
    }
    for (auto &worker : workers)
      worker.get();
    std::cout << "PASS concurrent_sessions=2 exact_outputs=16" << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << std::endl;
    return 1;
  }
}
