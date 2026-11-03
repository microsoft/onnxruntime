#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 3) throw std::runtime_error("Usage: qmoe_compare plugin model...");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "qmoe-compare"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[1]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty()) throw std::runtime_error("No WebGPU device");
    Ort::SessionOptions cpu_options, gpu_options;
    cpu_options.SetIntraOpNumThreads(1);
    gpu_options.SetIntraOpNumThreads(1);
    cpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    gpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    if (const char* prefix = std::getenv("ORT_TEST_PROFILE_PREFIX")) {
      gpu_options.EnableProfiling(prefix);
    }
    gpu_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    std::unordered_map<std::string, std::string> provider_options{
        {"enableInt64", "1"}, {"validationMode", "full"}, {"enableGraphCapture", "0"}};
    if (const char* value = std::getenv("ORT_TEST_MAX_PENDING_DISPATCHES")) {
      provider_options["maxNumPendingDispatches"] = value;
      std::cout << "maxNumPendingDispatches=" << value << std::endl;
    }
    if (const char* value = std::getenv("ORT_TEST_VALIDATION_MODE")) {
      provider_options["validationMode"] = value;
      std::cout << "validationMode=" << value << std::endl;
    }
    if (const char* value = std::getenv("ORT_TEST_ENABLE_GRAPH_CAPTURE")) {
      provider_options["enableGraphCapture"] = value;
      std::cout << "enableGraphCapture=" << value << std::endl;
    }
    if (const char* value = std::getenv("ORT_TEST_FP32_ACCUMULATION")) {
      provider_options["enableMatmulFp32Accumulation"] = value;
      std::cout << "enableMatmulFp32Accumulation=" << value << std::endl;
    }
    gpu_options.AppendExecutionProvider_V2(env, devices, provider_options);
    const char* outputs[] = {"y"};
    auto tolerance_value = [](const char* name, float fallback) {
      const char* text = std::getenv(name);
      if (!text) return fallback;
      char* end = nullptr;
      const float value = std::strtof(text, &end);
      if (end == text || *end || !std::isfinite(value) || value < 0)
        throw std::runtime_error(std::string("Invalid tolerance: ") + name);
      return value;
    };
    int selected_mode = -1;
    if (const char* text = std::getenv("ORT_TEST_ONLY_MODE")) {
      char* end = nullptr;
      const long value = std::strtol(text, &end, 10);
      if (end == text || *end || value < 0 || value > 3)
        throw std::runtime_error("ORT_TEST_ONLY_MODE must be an integer from0 through3");
      selected_mode = static_cast<int>(value);
    }
    for (int index = 2; index < argc; ++index) {
      const std::filesystem::path path{argv[index]};
      if (path.stem().string().ends_with("_reference")) continue;
      const auto reference_path = path.parent_path() / (path.stem().string() + "_reference.onnx");
      Ort::Session reference{env, reference_path.c_str(), cpu_options};
      Ort::Session session{env, path.c_str(), gpu_options};
      auto expected = reference.Run(Ort::RunOptions{}, nullptr, nullptr, 0, outputs, 1);
      const bool reference_half = expected[0].GetTensorTypeAndShapeInfo().GetElementType() ==
                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
      Ort::Value captured_output{nullptr};
      std::unique_ptr<Ort::IoBinding> capture_binding;
      if (provider_options.at("enableGraphCapture") == "1") {
        Ort::AllocatorWithDefaultOptions allocator;
        const auto info = expected[0].GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();
        captured_output = Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), info.GetElementType());
        capture_binding = std::make_unique<Ort::IoBinding>(session);
        capture_binding->BindOutput("y", captured_output);
      }
      const float abs_tolerance = tolerance_value("ORT_TEST_ATOL", reference_half ? 0.0005f : 0.000002f);
      const float rel_tolerance = tolerance_value("ORT_TEST_RTOL", reference_half ? 0.005f : 0.0005f);
      std::unique_ptr<Ort::Session> other;
      if (std::getenv("ORT_TEST_PROFILE_INTERLEAVE")) {
        const char* prefix = std::getenv("ORT_TEST_PROFILE_PREFIX");
        if (!prefix) throw std::runtime_error("Interleaved profiling requires ORT_TEST_PROFILE_PREFIX");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        auto other_options = gpu_options.Clone();
        other_options.EnableProfiling((std::string(prefix) + "-other").c_str());
        other = std::make_unique<Ort::Session>(env, path.c_str(), other_options);
      }
      int current_mode = 0;
      double max_errors[4]{};
      double squared_errors[4]{}, squared_references[4]{};
      size_t checked_elements[4]{};
      auto check = [&](const std::vector<Ort::Value>& actual) {
        const auto info = actual[0].GetTensorTypeAndShapeInfo();
        const auto ref_info = expected[0].GetTensorTypeAndShapeInfo();
        if (info.GetShape() != ref_info.GetShape() || info.GetElementType() != ref_info.GetElementType())
          throw std::runtime_error("Shape/type mismatch");
        const bool half = info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        for (size_t i = 0; i < info.GetElementCount(); ++i) {
          const float a = half ? actual[0].GetTensorData<Ort::Float16_t>()[i].ToFloat()
                               : actual[0].GetTensorData<float>()[i];
          const float e = half ? expected[0].GetTensorData<Ort::Float16_t>()[i].ToFloat()
                               : expected[0].GetTensorData<float>()[i];
          // Baseline mixes in expert-index order; indirect routing uses rank order.
          const float tolerance = abs_tolerance + rel_tolerance * std::abs(e);
          max_errors[current_mode] = std::max(max_errors[current_mode], static_cast<double>(std::abs(a - e)));
          if (!std::isfinite(a) || !std::isfinite(e) || std::abs(a - e) > tolerance)
            throw std::runtime_error(path.stem().string() + " mode=" + std::to_string(current_mode) +
                                     " numerical mismatch at " + std::to_string(i) +
                                     ": " + std::to_string(a) + " vs " + std::to_string(e));
          const double difference = static_cast<double>(a) - e;
          squared_errors[current_mode] += difference * difference;
          squared_references[current_mode] += static_cast<double>(e) * e;
          ++checked_elements[current_mode];
        }
      };
      std::vector<double> times[4];
      const bool compare_skinny = std::getenv("ORT_TEST_SKINNY_COMPARE") != nullptr;
      const bool compare_uniform_arena = std::getenv("ORT_TEST_UNIFORM_ARENA_COMPARE") != nullptr;
      const bool compare_attention_subgroup = std::getenv("ORT_TEST_ATTENTION_SUBGROUP_COMPARE") != nullptr;
      const bool compare_attention_split = std::getenv("ORT_TEST_ATTENTION_SPLIT_COMPARE") != nullptr;
      if (int(compare_skinny) + int(compare_uniform_arena) + int(compare_attention_subgroup) +
              int(compare_attention_split) > 1)
        throw std::runtime_error("Select only one experimental comparison");
      if (compare_attention_subgroup || compare_attention_split)
        std::cout << "comparison=" << (compare_attention_split ? "attention_split_kv" : "attention_subgroup")
                  << " mode0=legacy mode1=candidate" << std::endl;
      for (int round = 0; round < 9; ++round) {
        for (int step = 0; step < 4; ++step) {
          const int mode = (round + step) % 4;
          if ((compare_skinny || compare_uniform_arena || compare_attention_subgroup || compare_attention_split) &&
              mode >= 2) continue;
          if (selected_mode >= 0 && mode != selected_mode) continue;
          current_mode = mode;
          if (compare_attention_split) {
            if (setenv("ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SPLIT_KV", mode & 1 ? "1" : "0", 1))
              throw std::runtime_error("setenv failed");
          } else if (compare_attention_subgroup) {
            if (setenv("ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SUBGROUP", mode & 1 ? "1" : "0", 1))
              throw std::runtime_error("setenv failed");
          } else if (compare_uniform_arena) {
            if (setenv("ORT_WEBGPU_UNIFORM_ARENA", mode & 1 ? "1" : "0", 1))
              throw std::runtime_error("setenv failed");
          } else if (setenv("ORT_WEBGPU_QMOE_FP8_COOPERATIVE", mode & 1 ? "1" : "0", 1) ||
              setenv("ORT_WEBGPU_MATMUL_NBITS_SGM_N_TAIL", compare_skinny || (mode & 1) ? "1" : "0", 1) ||
              (compare_skinny && setenv("ORT_WEBGPU_MATMUL_NBITS_SKINNY", mode & 1 ? "1" : "0", 1)) ||
              setenv("ORT_WEBGPU_QMOE_INT4_COOPERATIVE", mode & 1 ? "1" : "0", 1) ||
              setenv("ORT_WEBGPU_QMOE_SMALL_BATCH", mode & 2 ? "1" : "0", 1) ||
              setenv("ORT_WEBGPU_QMOE_NVFP4_PREFILL", mode & 2 ? "1" : "0", 1) ||
              setenv("ORT_WEBGPU_QMOE_INT4_PREFILL", mode & 2 ? "1" : "0", 1))
            throw std::runtime_error("setenv failed");
          const auto start = std::chrono::steady_clock::now();
          auto actual = [&]() {
            if (!capture_binding)
              return session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, outputs, 1);
            session.Run(Ort::RunOptions{}, *capture_binding);
            capture_binding->SynchronizeOutputs();
            return capture_binding->GetOutputValues();
          }();
          const auto end = std::chrono::steady_clock::now();
          check(actual);
          if (other) check(other->Run(Ort::RunOptions{}, nullptr, nullptr, 0, outputs, 1));
          if (round >= 2)
            times[mode].push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
      }
      for (auto& samples : times) std::sort(samples.begin(), samples.end());
      const auto median = [&](int mode) {
        return times[mode].empty() ? std::numeric_limits<double>::quiet_NaN() : times[mode][3];
      };
      for (int mode = 0; mode < 4; ++mode)
        if (times[mode].empty()) max_errors[mode] = std::numeric_limits<double>::quiet_NaN();
      std::cout << "PASS " << path.stem().string() << " scalar_ms=" << median(0)
                << " cooperative_ms=" << median(1) << " small_batch_ms=" << median(2)
                << " combined_ms=" << median(3) << " combined_speedup=" << median(0) / median(3)
                << " max_abs_errors=" << max_errors[0] << "," << max_errors[1] << ","
                << max_errors[2] << "," << max_errors[3];
      std::cout << " rmse=";
      for (int mode = 0; mode < 4; ++mode)
        std::cout << (mode ? "," : "") << (checked_elements[mode] ?
            std::sqrt(squared_errors[mode] / checked_elements[mode]) : std::numeric_limits<double>::quiet_NaN());
      std::cout << " relative_rmse=";
      for (int mode = 0; mode < 4; ++mode)
        std::cout << (mode ? "," : "") << (checked_elements[mode] == 0 ?
            std::numeric_limits<double>::quiet_NaN() : squared_references[mode] > 0 ?
            std::sqrt(squared_errors[mode] / squared_references[mode]) : squared_errors[mode] == 0 ? 0. : INFINITY);
      std::cout << " samples_per_mode=7 selected_mode=" << selected_mode
                << " (host Run incl readback, validationMode=" << provider_options.at("validationMode")
                << ")" << std::endl;
      if (std::getenv("ORT_TEST_PROFILE_PREFIX")) {
        Ort::AllocatorWithDefaultOptions allocator;
        const auto profile = session.EndProfilingAllocated(allocator);
        std::cout << "PROFILE " << profile.get() << std::endl;
        if (other) {
          const auto other_profile = other->EndProfilingAllocated(allocator);
          std::cout << "PROFILE_OTHER " << other_profile.get() << std::endl;
        }
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
