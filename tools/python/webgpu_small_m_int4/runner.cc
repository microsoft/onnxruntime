// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace {

int ParseCount(const char* text, bool positive) {
  size_t end = 0;
  const int value = std::stoi(text, &end);
  if (text[end] != '\0' || value < (positive ? 1 : 0)) {
    throw std::invalid_argument("Expected a nonnegative warmup / positive repeat count.");
  }
  return value;
}

int64_t Timestamp() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void Copy(Ort::Env& env, const Ort::Value& from, Ort::Value& to) {
  const OrtValue* source = from;
  OrtValue* target = to;
  Ort::ThrowOnError(Ort::GetApi().CopyTensors(env, &source, &target, nullptr, 1));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 9) {
      throw std::invalid_argument(
          "usage: diagnostic model input_prefix output_prefix warmup repeat validation fp32_acc mode");
    }
    const std::string input_prefix = argv[2], prefix = argv[3], mode = argv[8];
    if (mode != "auto" && mode != "regular" && mode != "simt") {
      throw std::invalid_argument("mode must be auto, regular, or simt");
    }
    if (setenv("ORT_WEBGPU_SMALL_M_DIAGNOSTIC", mode.c_str(), 1) != 0) {
      throw std::runtime_error("Cannot set the explicit diagnostic selection.");
    }
    const int warmup = ParseCount(argv[4], false), repeat = ParseCount(argv[5], true);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "small-m-simt");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    options.EnableProfiling(prefix.c_str());
    options.AppendExecutionProvider("WebGPU", {{"dawnBackendType", "Vulkan"},
                                               {"enableGraphCapture", "0"},
                                               {"enableRobustness", "0"},
                                               {"validationMode", argv[6]},
                                               {"enableMatmulFp32Accumulation", argv[7]}});
    Ort::Session session(env, argv[1], options);
    Ort::AllocatorWithDefaultOptions cpu_allocator;
    Ort::MemoryInfo gpu_info("WebGPU_Buf", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    Ort::Allocator gpu_allocator(session, gpu_info);
    Ort::IoBinding binding(session);
    std::vector<Ort::Value> inputs, outputs;
    std::vector<std::string> output_names;
    inputs.reserve(session.GetInputCount());
    outputs.reserve(session.GetOutputCount());
    output_names.reserve(session.GetOutputCount());
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
      const auto name = session.GetInputNameAllocated(i, cpu_allocator);
      const auto type = session.GetInputTypeInfo(i);
      const auto info = type.GetTensorTypeAndShapeInfo();
      const auto shape = info.GetShape();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        throw std::invalid_argument("Diagnostic runner expects FP16 activation inputs.");
      }
      auto cpu = Ort::Value::CreateTensor<Ort::Float16_t>(cpu_allocator, shape.data(), shape.size());
      const size_t bytes = info.GetElementCount() * sizeof(Ort::Float16_t);
      std::ifstream stream(input_prefix + "-" + name.get() + ".bin", std::ios::binary);
      stream.read(reinterpret_cast<char*>(cpu.GetTensorMutableData<Ort::Float16_t>()),
                  static_cast<std::streamsize>(bytes));
      if (!stream || stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Incorrect input file size for " + std::string{name.get()});
      }
      inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(gpu_allocator, shape.data(), shape.size()));
      Copy(env, cpu, inputs.back());
      binding.BindInput(name.get(), inputs.back());
    }
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
      output_names.emplace_back(session.GetOutputNameAllocated(i, cpu_allocator).get());
      const auto type = session.GetOutputTypeInfo(i);
      const auto info = type.GetTensorTypeAndShapeInfo();
      const auto shape = info.GetShape();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        throw std::invalid_argument("Diagnostic runner expects FP16 outputs.");
      }
      outputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(gpu_allocator, shape.data(), shape.size()));
      binding.BindOutput(output_names.back().c_str(), outputs.back());
    }
    if (outputs.empty()) {
      throw std::invalid_argument("Diagnostic model needs at least one output.");
    }
    const auto last_shape = outputs.back().GetTensorTypeAndShapeInfo().GetShape();
    auto fence = Ort::Value::CreateTensor<Ort::Float16_t>(cpu_allocator, last_shape.data(), last_shape.size());
    Ort::RunOptions run;
    const auto run_and_wait = [&]() {
      session.Run(run, binding);
      Copy(env, outputs.back(), fence);
    };
    for (int i = 0; i < warmup; ++i) {
      run_and_wait();
    }
    std::cout << "RESIDENT_INPUTS=" << inputs.size() << "\nRESIDENT_OUTPUTS=" << outputs.size() << '\n';
    std::cout << "MEASURE_START=" << Timestamp() << '\n';
    for (int i = 0; i < repeat; ++i) {
      run_and_wait();
    }
    std::cout << "MEASURE_END=" << Timestamp() << '\n';
    std::cout << "PROFILE=" << session.EndProfilingAllocated(cpu_allocator).get() << '\n';
    std::ofstream output(prefix + "-outputs.bin", std::ios::binary);
    std::ofstream index(prefix + "-outputs.tsv");
    size_t offset = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto shape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
      auto cpu = Ort::Value::CreateTensor<Ort::Float16_t>(cpu_allocator, shape.data(), shape.size());
      Copy(env, outputs[i], cpu);
      const size_t bytes = cpu.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(Ort::Float16_t);
      output.write(reinterpret_cast<const char*>(cpu.GetTensorData<Ort::Float16_t>()),
                   static_cast<std::streamsize>(bytes));
      index << output_names[i] << '\t' << offset << '\t' << bytes << '\n';
      offset += bytes;
    }
    if (!output || !index) {
      throw std::runtime_error("Cannot write diagnostic outputs/index.");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
