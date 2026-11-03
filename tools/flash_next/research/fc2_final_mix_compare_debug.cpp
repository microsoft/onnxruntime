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
    if (argc < 3) throw std::runtime_error("Usage: native_compare provider fixture...");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "qmoe-node-replay"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[1]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty()) throw std::runtime_error("No WebGPU device");
    const char* input_names[] = {"x", "router"};
    const char* output_names[] = {"y"};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto set_flag = [](const char* name, const char* value) {
      if (setenv(name, value, 1) != 0) throw std::runtime_error(std::string("Cannot set ") + name);
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
      std::vector<std::vector<Ort::Float16_t>> data(2);
      std::vector<Ort::Value> inputs;
      for (size_t index = 0; index < 2; ++index) {
        auto type_info = session.GetInputTypeInfo(index);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
          throw std::runtime_error("Expected FP16 fixture inputs");
        auto shape = info.GetShape();
        data[index].resize(info.GetElementCount());
        inputs.emplace_back(Ort::Value::CreateTensor<Ort::Float16_t>(
            memory, data[index].data(), data[index].size(), shape.data(), shape.size()));
      }
      for (int round = 0; round < 32; ++round) {
        for (size_t i = 0; i < data[0].size(); ++i)
          data[0][i] = Ort::Float16_t(0.07f * std::sin(float((i + 1) * (round + 1)) * 0.07f));
        for (size_t i = 0; i < data[1].size(); ++i)
          data[1][i] = Ort::Float16_t(std::getenv("ORT_TEST_FC2_ONE_HOT") ?
              (i == 0 ? 20.0f : -20.0f) : float(int((i * 73 + round * 37) % 509) - 254) * 0.02f);
        const char* gate_mode = (round / 16) % 2 ? "1" : "0";
        set_flag("ORT_WEBGPU_QMOE_SUBGROUP_GATE",
                 std::getenv("ORT_TEST_COMPARE_GATE") ? "0" : gate_mode);
        set_flag("ORT_WEBGPU_QMOE_INT4_PREFILL", data[1].size() != 512 || (round / 2) % 2 ? "1" : "0");
        set_flag("ORT_WEBGPU_MATMUL_NBITS_SKINNY", "0");
        set_flag("ORT_WEBGPU_MATMUL_GEMV_N96", (round / 8) % 2 ? "1" : "0");
        set_flag("ORT_WEBGPU_QMOE_FC2_FINAL_MIX", "0");
        auto reference = session.Run(Ort::RunOptions{}, input_names, inputs.data(), 2, output_names, 1);
        set_flag("ORT_WEBGPU_QMOE_SUBGROUP_GATE", gate_mode);
        set_flag("ORT_WEBGPU_QMOE_FC2_FINAL_MIX", "1");
        auto actual = session.Run(Ort::RunOptions{}, input_names, inputs.data(), 2, output_names, 1);
        const auto info = reference[0].GetTensorTypeAndShapeInfo();
        const auto actual_info = actual[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
            actual_info.GetElementType() != info.GetElementType() || actual_info.GetShape() != info.GetShape())
          throw std::runtime_error("Mismatching output shape/type");
        for (size_t i = 0; i < info.GetElementCount(); ++i) {
          const auto& expected = reference[0].GetTensorData<Ort::Float16_t>()[i];
          const auto& value = actual[0].GetTensorData<Ort::Float16_t>()[i];
          if (!std::isfinite(expected.ToFloat()) || !std::isfinite(value.ToFloat()) ||
              std::memcmp(&expected, &value, sizeof(value)) != 0)
            throw std::runtime_error(std::string(argv[file]) + " finite/bitwise mismatch round=" +
                                     std::to_string(round) + " element=" + std::to_string(i) +
                                     " expected=" + std::to_string(expected.ToFloat()) +
                                     " actual=" + std::to_string(value.ToFloat()));
          ++comparisons;
        }
      }
      set_flag("ORT_WEBGPU_QMOE_FC2_FINAL_MIX", "2");
      bool rejected = false;
      try {
        session.Run(Ort::RunOptions{}, input_names, inputs.data(), 2, output_names, 1);
      } catch (const Ort::Exception& error) {
        rejected = std::string(error.what()).find("must be 0 or 1") != std::string::npos;
      }
      if (!rejected) throw std::runtime_error("Invalid replay flag was not explicitly rejected");
      set_flag("ORT_WEBGPU_QMOE_FC2_FINAL_MIX", "1");
      std::cout << "PASS changing_inputs_and_routes " << std::filesystem::path(argv[file]).stem().string()
                << " rounds=32 invalid_flag_rejected=1" << std::endl;
    }
    std::cout << "PASS total_exact_elements=" << comparisons << "; CPU fallback disabled" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
