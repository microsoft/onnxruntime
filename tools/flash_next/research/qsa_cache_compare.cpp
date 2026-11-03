#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 3) throw std::runtime_error("Usage: qsa_cache_compare plugin fixture...");
    const char* skip = std::getenv("ORT_TEST_QSA_CANDIDATE_ONLY");
    if (skip && std::string(skip) != "0" && std::string(skip) != "1")
      throw std::runtime_error("ORT_TEST_QSA_CANDIDATE_ONLY must be0 or1");
    const bool candidate_only = skip && std::string(skip) == "1";
    const char* rank_setting = std::getenv("ORT_TEST_QSA_RANK_COMPARE");
    if (rank_setting && std::string(rank_setting) != "0" && std::string(rank_setting) != "1")
      throw std::runtime_error("ORT_TEST_QSA_RANK_COMPARE must be0 or1");
    const bool rank_compare = rank_setting && std::string(rank_setting) == "1";
    const char* validation_setting = std::getenv("ORT_TEST_VALIDATION_MODE");
    const std::string validation = validation_setting ? validation_setting : "full";
    if (validation != "full" && validation != "disabled")
      throw std::runtime_error("ORT_TEST_VALIDATION_MODE must be full or disabled");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "qsa-global-cache"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[1]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty()) throw std::runtime_error("No WebGPU adapter");
    Ort::SessionOptions cpu_options, gpu_options;
    cpu_options.SetIntraOpNumThreads(1);
    gpu_options.SetIntraOpNumThreads(1);
    cpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    gpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    gpu_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    gpu_options.AppendExecutionProvider_V2(env, devices,
        {{"enableInt64", "1"}, {"validationMode", validation}, {"enableGraphCapture", "0"}});
    const char* names[] = {"indices", "present"};
    for (int arg = 2; arg < argc; ++arg) {
      const std::filesystem::path path{argv[arg]};
      if (path.stem().string().ends_with("_reference")) continue;
      auto options = gpu_options.Clone();
      if (const char* prefix = std::getenv("ORT_TEST_PROFILE_PREFIX")) {
        options.EnableProfiling((std::string(prefix) + "-" + path.stem().string()).c_str());
      }
      Ort::Session actual_session{env, path.c_str(), options};
      const auto reference_path = path.parent_path() / (path.stem().string() + "_reference.onnx");
      Ort::Session reference{env, reference_path.c_str(), cpu_options};
      auto expected = reference.Run(Ort::RunOptions{}, nullptr, nullptr, 0, names, 2);
      int checks = 0;
      std::array<std::vector<double>, 2> timings;
      for (int round = 0; round < 9; ++round) {
        for (int step = 0; step < 2; ++step) {
          const bool enabled = (round + step) % 2;
          if (candidate_only && !enabled) continue;
          if (setenv("ORT_WEBGPU_QSA_GLOBAL_SCORE_CACHE", rank_compare || enabled ? "1" : "0", 1))
            throw std::runtime_error("Cannot set QSA experiment");
          if (rank_compare && setenv("ORT_WEBGPU_QSA_PARALLEL_RANK", enabled ? "1" : "0", 1))
            throw std::runtime_error("Cannot set QSA parallel-rank experiment");
          const auto start = std::chrono::steady_clock::now();
          auto actual = actual_session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, names, 2);
          const auto elapsed = std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start).count();
          if (round >= 2) timings[enabled].push_back(elapsed);
          for (size_t output = 0; output < 2; ++output) {
            const auto info = actual[output].GetTensorTypeAndShapeInfo();
            const auto reference_info = expected[output].GetTensorTypeAndShapeInfo();
            if (info.GetShape() != reference_info.GetShape() ||
                info.GetElementType() != reference_info.GetElementType()) {
              throw std::runtime_error("Output contract mismatch");
            }
            const size_t element_bytes = info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? 2 : 4;
            if (std::memcmp(actual[output].GetTensorRawData(), expected[output].GetTensorRawData(),
                            info.GetElementCount() * element_bytes) != 0) {
              throw std::runtime_error(path.stem().string() + " mode=" + std::to_string(enabled) +
                                       " output=" + std::to_string(output) + " exact reference mismatch");
            }
          }
          ++checks;
        }
      }
      std::cout << "PASS " << path.stem().string() << " exact_checks=" << checks
                << " candidate_only=" << candidate_only << " rank_compare=" << rank_compare
                << " validation=" << validation;
      for (size_t mode = 0; mode < timings.size(); ++mode) {
        auto& values = timings[mode];
        if (!values.empty()) {
          std::sort(values.begin(), values.end());
          const size_t middle = values.size() / 2;
          const double median = values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
          std::cout << " mode" << mode << "_host_readback_ms=" << median;
        }
      }
      std::cout << std::endl;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << std::endl;
    return 1;
  }
}
