#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
float as_float(T value) {
  if constexpr (std::is_same_v<T, Ort::Float16_t>) return value.ToFloat();
  else return value;
}

template <typename T>
void compare(Ort::Session& session, const std::vector<int64_t>& shape, bool timing, bool exclude_subnormals) {
  const size_t rows = shape[0], n = shape[1];
  std::vector<T> input(rows * n);
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const char* inputs[]{"x"};
  const char* outputs[]{"y"};
  std::vector<double> durations;
  size_t comparisons = 0;
  const int rounds = timing ? 1032 : 16;
  for (int round = 0; round < rounds; ++round) {
    const int pattern = timing || (exclude_subnormals && round == 12) ? 0 : round;
    for (size_t row = 0; row < rows; ++row) {
      auto* x = input.data() + row * n;
      for (size_t i = 0; i < n; ++i)
        x[i] = T(std::sin(float(i + 1) * 0.071f + float(row)) * 0.125f);
      const float inf = std::numeric_limits<float>::infinity();
      const float nan = std::numeric_limits<float>::quiet_NaN();
      if (pattern == 1) std::fill_n(x, n, T(-inf));
      if (pattern == 2) std::fill_n(x, n, T(0.0f));
      if (pattern == 3) { std::fill_n(x, n, T(-0.0f)); x[n - 1] = T(0.0f); }
      if (pattern == 4) x[0] = T(nan);
      if (pattern == 5) x[n / 2] = T(nan);
      if (pattern == 6) { x[n / 2] = T(inf); x[n - 1] = T(inf); }
      if (pattern == 7) { std::fill_n(x, n, T(-inf)); x[n - 1] = T(-1.0f); }
      if (pattern == 8) { x[0] = T(inf); x[n - 1] = T(inf); }
      if (pattern == 9) { x[std::min<size_t>(127, n - 1)] = T(2.0f); x[n - 1] = T(2.0f); }
      if (pattern == 10) { x[std::min<size_t>(2047, n - 1)] = T(2.0f); x[n - 1] = T(2.0f); }
      if (pattern == 11) { std::fill_n(x, n, T(nan)); x[n - 1] = T(inf); }
      if (pattern == 12) {
        std::fill_n(x, n, T(0.0f));
        x[n - 1] = T(std::is_same_v<T, float> ? std::numeric_limits<float>::denorm_min() : 0.000000059604644775390625f);
      }
      if (pattern == 13) std::fill_n(x, n, T(-2.0f));
      if (pattern == 14) { x[0] = T(-inf); x[n - 1] = T(nan); }
      if (pattern == 15) x[n - 1] = T(3.0f + float(row));
    }
    std::vector<int64_t> expected(rows);
    for (size_t row = 0; row < rows; ++row) {
      float best = as_float(input[row * n]);
      for (size_t i = 1; i < n; ++i) {
        const float candidate = as_float(input[row * n + i]);
        if (candidate > best) { best = candidate; expected[row] = static_cast<int64_t>(i); }
      }
    }
    auto tensor = Ort::Value::CreateTensor<T>(memory, input.data(), input.size(), shape.data(), shape.size());
    const int repeats = timing ? 1 : 4;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      const auto start = std::chrono::steady_clock::now();
      auto result = session.Run(Ort::RunOptions{}, inputs, &tensor, 1, outputs, 1);
      const auto stop = std::chrono::steady_clock::now();
      const auto info = result[0].GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 || info.GetElementCount() != rows)
        throw std::runtime_error("Wrong ArgMax output shape/type");
      const auto* observed = result[0].template GetTensorData<int64_t>();
      for (size_t row = 0; row < rows; ++row) {
        if (observed[row] != expected[row])
          throw std::runtime_error("ArgMax mismatch pattern=" + std::to_string(pattern) +
                                   " row=" + std::to_string(row) + " expected=" + std::to_string(expected[row]) +
                                   " actual=" + std::to_string(observed[row]));
        ++comparisons;
      }
      if (timing && round >= 8)
        durations.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
    }
  }
  std::cout << "ARGMAX rows=" << rows << " N=" << n << " comparisons=" << comparisons << " mismatches=0";
  if (timing) {
    double sum = 0;
    for (double duration : durations) sum += duration;
    std::sort(durations.begin(), durations.end());
    std::cout << " iterations=" << durations.size() << " warmup=8 mean_us=" << sum / durations.size()
              << " median_us=" << (durations[511] + durations[512]) * 0.5;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  try {
    if (argc != 5 || (std::string(argv[1]) != "compare" && std::string(argv[1]) != "timing" && std::string(argv[1]) != "compare-normal"))
      throw std::runtime_error("Usage: compare|compare-normal|timing provider model profile-prefix");
    const bool timing = std::string(argv[1]) == "timing";
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "argmax-parallel"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[2]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") { devices.push_back(device); break; }
    }
    if (devices.empty()) throw std::runtime_error("No WebGPU device");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    options.AppendExecutionProvider_V2(env, devices, {{"enableInt64", "1"}, {"validationMode", "full"},
        {"maxNumPendingDispatches", "64"}, {"enableGraphCapture", "0"}});
    if (!timing) options.EnableProfiling(argv[4]);
    Ort::Session session{env, argv[3], options};
    const auto input_type = session.GetInputTypeInfo(0);
    const auto info = input_type.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) throw std::runtime_error("Wrong fixture shape");
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      compare<float>(session, shape, timing, std::string(argv[1]) == "compare-normal");
    else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
      compare<Ort::Float16_t>(session, shape, timing, std::string(argv[1]) == "compare-normal");
    else throw std::runtime_error("Unsupported fixture type");
    if (!timing) {
      Ort::AllocatorWithDefaultOptions allocator;
      auto profile = session.EndProfilingAllocated(allocator);
      std::cout << "PROFILE " << profile.get() << std::endl;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
