// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "onnxruntime_cxx_api.h"

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error("Usage: sdk_inference_example <path-to-add_f32.onnx>");
    }
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bazel-sdk");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    const std::filesystem::path model(argv[1]);
    Ort::Session session(env, model.c_str(), options);
    const std::array<int64_t, 2> shape{2, 3};
    std::array<float, 6> a{1, 2, 3, 4, 5, 6};
    std::array<float, 6> b{10, 20, 30, 40, 50, 60};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<Ort::Value, 2> inputs{
        Ort::Value::CreateTensor<float>(memory, a.data(), a.size(), shape.data(), shape.size()),
        Ort::Value::CreateTensor<float>(memory, b.data(), b.size(), shape.data(), shape.size())};
    const std::array<const char*, 2> input_names{"A", "B"};
    const std::array<const char*, 1> output_names{"C"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                               output_names.data(), output_names.size());
    if (outputs[0].GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>({2, 3})) {
      throw std::runtime_error("Packaged SDK inference returned an unexpected shape; expected [2, 3].");
    }
    const float* values = outputs[0].GetTensorData<float>();
    for (size_t i = 0; i < a.size(); ++i) {
      if (values[i] != a[i] + b[i]) {
        throw std::runtime_error("Packaged SDK Add inference returned incorrect values.");
      }
    }
    std::cout << "Packaged SDK Add inference passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
