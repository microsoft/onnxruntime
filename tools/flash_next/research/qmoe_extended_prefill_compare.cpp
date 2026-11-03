#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 3) throw std::runtime_error("Usage: native_compare provider fixture...");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "extended-prefill"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[1]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty()) throw std::runtime_error("No WebGPU device");
    const char* names[] = {"x", "router"};
    const char* outputs[] = {"y"};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto flag = [](const char* value) {
      if (setenv("ORT_WEBGPU_QMOE_EXTENDED_PREFILL", value, 1) != 0)
        throw std::runtime_error("Cannot set extended prefill flag");
    };
    size_t comparisons = 0;
    Ort::AllocatorWithDefaultOptions allocator;
    for (int file = 2; file < argc; ++file) {
      Ort::SessionOptions options;
      options.SetIntraOpNumThreads(1);
      options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
      options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      options.AppendExecutionProvider_V2(
          env, devices, {{"enableInt64", "1"}, {"validationMode", "full"},
                         {"maxNumPendingDispatches", file % 2 ? "1" : "64"},
                         {"enableGraphCapture", "0"}});
      const auto profile_prefix = std::string("/evidence/webgpu-perf-oct20/extended-prefill-v1-native-") +
          std::filesystem::path(argv[file]).stem().string();
      options.EnableProfiling(profile_prefix.c_str());
      Ort::Session session{env, argv[file], options};
      std::vector<Ort::Float16_t> last_x, last_router;
      auto run = [&](Ort::Float16_t* x, Ort::Float16_t* router, int64_t rows) {
        const std::array<int64_t, 2> x_shape{rows, 2560}, r_shape{rows, 512};
        std::array<Ort::Value, 2> inputs{
            Ort::Value::CreateTensor<Ort::Float16_t>(memory, x, rows * 2560, x_shape.data(), 2),
            Ort::Value::CreateTensor<Ort::Float16_t>(memory, router, rows * 512, r_shape.data(), 2)};
        return session.Run(Ort::RunOptions{}, names, inputs.data(), 2, outputs, 1);
      };
      for (int64_t rows : {16, 32, 64, 128, 129}) {
        const int rounds = rows == 129 ? 2 : 8;
        std::vector<Ort::Float16_t> x(rows * 2560), router(rows * 512), expected(rows * 2560);
        for (int round = 0; round < rounds; ++round) {
          for (size_t i = 0; i < x.size(); ++i)
            x[i] = Ort::Float16_t(0.07f * std::sin(float((i + 1) * (round + 1)) * 0.07f));
          for (size_t i = 0; i < router.size(); ++i)
            router[i] = Ort::Float16_t(float(int((i * 73 + round * 37) % 509) - 254) * 0.02f);
          flag("0");
          const int64_t shard = rows == 129 ? rows : 16;
          for (int64_t first = 0; first < rows; first += shard) {
            auto reference = run(x.data() + first * 2560, router.data() + first * 512, shard);
            const auto info = reference[0].GetTensorTypeAndShapeInfo();
            if (info.GetShape() != std::vector<int64_t>{shard, 2560} ||
                info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
              throw std::runtime_error("Unexpected reference shape/type");
            std::copy_n(reference[0].GetTensorData<Ort::Float16_t>(), shard * 2560,
                        expected.data() + first * 2560);
          }
          flag("1");
          auto actual = run(x.data(), router.data(), rows);
          const auto info = actual[0].GetTensorTypeAndShapeInfo();
          if (info.GetShape() != std::vector<int64_t>{rows, 2560} ||
              info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
            throw std::runtime_error("Unexpected actual shape/type");
          const auto* observed = actual[0].GetTensorData<Ort::Float16_t>();
          for (size_t i = 0; i < expected.size(); ++i) {
            if (!std::isfinite(expected[i].ToFloat()) || !std::isfinite(observed[i].ToFloat()) ||
                std::memcmp(&expected[i], &observed[i], sizeof(observed[i])) != 0)
              throw std::runtime_error(std::string(argv[file]) + " finite/bitwise mismatch rows=" +
                  std::to_string(rows) + " round=" + std::to_string(round) + " element=" + std::to_string(i));
            ++comparisons;
          }
        }
        last_x = std::move(x);
        last_router = std::move(router);
        std::cout << "PASS rows=" << rows << " rounds=" << rounds << std::endl;
      }
      flag("2");
      bool rejected = false;
      try {
        run(last_x.data(), last_router.data(), 129);
      } catch (const Ort::Exception& error) {
        rejected = std::string(error.what()).find("must be 0 or 1") != std::string::npos;
      }
      if (!rejected) throw std::runtime_error("Invalid flag was not explicitly rejected");
      flag("1");
      auto profile_path = session.EndProfilingAllocated(allocator);
      std::cout << "PROFILE " << profile_path.get() << std::endl;
      std::cout << "PASS " << std::filesystem::path(argv[file]).stem().string()
                << " retained_chunk16_reference=1 invalid_flag_rejected=1" << std::endl;
    }
    std::cout << "PASS total_exact_elements=" << comparisons << "; CPU fallback disabled" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
