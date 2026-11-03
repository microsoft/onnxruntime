#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 3) throw std::runtime_error("Usage: hc_fusion_compare plugin model...");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "hc-fusion-exact"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[1]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty()) throw std::runtime_error("No isolated WebGPU device");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    options.AppendExecutionProvider_V2(
        env, devices, std::unordered_map<std::string, std::string>{
                          {"enableInt64", "1"}, {"validationMode", "full"},
                          {"enableGraphCapture", "0"}, {"maxNumPendingDispatches", "64"}});
    size_t total = 0;
    for (int arg = 2; arg < argc; ++arg) {
      const std::filesystem::path path{argv[arg]};
      const auto reference = path.parent_path() / (path.stem().string() + "_reference.onnx");
      auto reference_options = options.Clone();
      auto fused_options = options.Clone();
      if (const char* prefix = std::getenv("ORT_TEST_PROFILE_PREFIX")) {
        reference_options.EnableProfiling((std::string(prefix) + "-" + path.stem().string() + "-reference").c_str());
        fused_options.EnableProfiling((std::string(prefix) + "-" + path.stem().string() + "-fused").c_str());
      }
      Ort::Session baseline{env, reference.c_str(), reference_options};
      Ort::Session fused{env, path.c_str(), fused_options};
      if (fused.GetOutputCount() != baseline.GetOutputCount())
        throw std::runtime_error("Mismatching output count");
      Ort::AllocatorWithDefaultOptions allocator;
      std::vector<std::string> output_names;
      std::vector<const char*> names;
      output_names.reserve(fused.GetOutputCount());
      for (size_t i = 0; i < fused.GetOutputCount(); ++i)
        output_names.emplace_back(fused.GetOutputNameAllocated(i, allocator).get());
      for (const auto& name : output_names) names.push_back(name.c_str());
      const bool toggle_mid = std::getenv("ORT_TEST_TOGGLE_EXACT_MID") != nullptr;
      const char* original_mid = std::getenv("ORT_WEBGPU_EXACT_DENSE_MID");
      const bool had_mid = original_mid != nullptr;
      const std::string saved_mid = had_mid ? original_mid : "";
      auto set_mid = [](const char* value) {
        if (setenv("ORT_WEBGPU_EXACT_DENSE_MID", value, 1) != 0)
          throw std::runtime_error("Cannot set exact-mid test flag");
      };
      for (int run = 0; run < 3; ++run) {
        if (toggle_mid) set_mid("0");
        auto expected = baseline.Run(Ort::RunOptions{}, nullptr, nullptr, 0, names.data(), names.size());
        if (toggle_mid) set_mid(run == 1 ? "1" : "0");
        auto actual = fused.Run(Ort::RunOptions{}, nullptr, nullptr, 0, names.data(), names.size());
        for (size_t output = 0; output < names.size(); ++output) {
        const auto info = actual[output].GetTensorTypeAndShapeInfo();
        const auto ref_info = expected[output].GetTensorTypeAndShapeInfo();
        if (info.GetShape() != ref_info.GetShape() || info.GetElementType() != ref_info.GetElementType())
          throw std::runtime_error("Expected matching output shapes/types");
        const auto type = info.GetElementType();
        if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
          throw std::runtime_error("Expected FP16 or FP32 outputs");
        const size_t element_size = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? 2 : 4;
        const auto* a = static_cast<const unsigned char*>(actual[output].GetTensorRawData());
        const auto* b = static_cast<const unsigned char*>(expected[output].GetTensorRawData());
        for (size_t i = 0; i < info.GetElementCount(); ++i) {
          const float av = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                               ? actual[output].GetTensorData<Ort::Float16_t>()[i].ToFloat()
                               : actual[output].GetTensorData<float>()[i];
          const float bv = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                               ? expected[output].GetTensorData<Ort::Float16_t>()[i].ToFloat()
                               : expected[output].GetTensorData<float>()[i];
          if (!std::isfinite(av) || !std::isfinite(bv) ||
              std::memcmp(a + i * element_size, b + i * element_size, element_size) != 0)
            throw std::runtime_error(path.stem().string() + " output=" + names[output] + " bitwise/finite mismatch at " +
                                     std::to_string(i) + ": " + std::to_string(av) +
                                     " vs " + std::to_string(bv));
        }
        if (toggle_mid) {
          if (had_mid) {
            set_mid(saved_mid.c_str());
          } else if (unsetenv("ORT_WEBGPU_EXACT_DENSE_MID") != 0) {
            throw std::runtime_error("Cannot restore exact-mid test flag");
          }
        }
        total += info.GetElementCount();
        }
      }
      std::cout << "PASS exact finite outputs " << path.stem().string() << " runs=3" << std::endl;
    }
    std::cout << "PASS total_exact_elements=" << total << "; CPU fallback disabled" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
