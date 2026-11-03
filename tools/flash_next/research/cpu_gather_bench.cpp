#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("Usage: cpu_gather_bench model.onnx");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "original-cpu-gather"};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* inputs[] = {"indices"};
    const char* outputs[] = {"y"};
    for (int threads : {1, 4, 16}) {
      Ort::SessionOptions options;
      options.SetIntraOpNumThreads(threads);
      options.SetInterOpNumThreads(1);
      options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
      options.AddConfigEntry("session.intra_op.allow_spinning", "0");
      Ort::Session session{env, argv[1], options};
      for (int64_t rows : {1, 16}) {
        for (bool changing : {false, true}) {
          std::mt19937_64 random{4321};
          std::uniform_int_distribution<int64_t> distribution{0, 320001535};
          std::vector<int64_t> indices(static_cast<size_t>(rows * 16));
          std::array<int64_t, 2> shape{rows, 16};
          auto input = Ort::Value::CreateTensor<int64_t>(
              memory, indices.data(), indices.size(), shape.data(), shape.size());
          std::vector<double> milliseconds;
          rusage before{}, after{};
          for (int iteration = 0; iteration < 45; ++iteration) {
            if (iteration == 5 && getrusage(RUSAGE_SELF, &before) != 0) {
              throw std::runtime_error("Cannot read initial page-fault counters");
            }
            if (changing || iteration == 0) {
              for (auto& index : indices) index = distribution(random);
            }
            if (getrusage(RUSAGE_SELF, &after) != 0) {
              throw std::runtime_error("Cannot read final page-fault counters");
            }
            const auto start = std::chrono::steady_clock::now();
            auto result = session.Run(Ort::RunOptions{}, inputs, &input, 1, outputs, 1);
            const auto end = std::chrono::steady_clock::now();
            const auto info = result[0].GetTensorTypeAndShapeInfo();
            if (info.GetShape() != std::vector<int64_t>{rows, 16, 160} ||
                info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
              throw std::runtime_error("Unexpected original PLE output contract");
            }
            for (size_t i = 0; i < info.GetElementCount(); ++i) {
              if (!std::isfinite(result[0].GetTensorData<Ort::Float16_t>()[i].ToFloat())) {
                throw std::runtime_error("Nonfinite original PLE output");
              }
            }
            if (iteration >= 5) {
              milliseconds.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            }
          }
          std::sort(milliseconds.begin(), milliseconds.end());
          std::cout << "{\"threads\":" << threads << ",\"rows\":" << rows
                    << ",\"changing_indices\":" << (changing ? "true" : "false")
                    << ",\"median_ms\":" << (milliseconds[19] + milliseconds[20]) / 2
                    << ",\"min_ms\":" << milliseconds.front()
                    << ",\"max_ms\":" << milliseconds.back()
                    << ",\"major_faults\":" << after.ru_majflt - before.ru_majflt
                    << ",\"minor_faults\":" << after.ru_minflt - before.ru_minflt << "}" << std::endl;
        }
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << std::endl;
    return 1;
  }
}
