// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Sample program demonstrating basic ONNX Runtime C++ API usage.
// Loads a simple ONNX model (C = A + B), runs inference, and prints the result.
//
// Generate the model first:  python generate_model.py

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

// Throw std::runtime_error if `condition` is false. Includes file and line info.
#define THROW_IF_NOT(condition)                                  \
  do {                                                           \
    if (!(condition)) {                                          \
      throw std::runtime_error(std::string(__FILE__) + ":" +     \
                               std::to_string(__LINE__) + ": " + \
                               "check failed: " #condition);     \
    }                                                            \
  } while (0)

int main(int argc, char* argv[]) {
  try {
    // -----------------------------------------------------------------------
    // 1. Initialize the ONNX Runtime environment
    // -----------------------------------------------------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_sample");
    std::cout << "ONNX Runtime version: " << Ort::GetVersionString() << "\n\n";

    // -----------------------------------------------------------------------
    // 2. Create session options (could add execution providers here)
    // -----------------------------------------------------------------------
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    // -----------------------------------------------------------------------
    // 3. Load the ONNX model from a file
    //    Generate with:  python generate_model.py
    // -----------------------------------------------------------------------
    const std::filesystem::path model_path = (argc > 1) ? argv[1] : "add_model.onnx";
    std::cout << "Loading model: " << model_path.string() << "\n";

    Ort::Session session(env, model_path.native().c_str(), session_options);

    // -----------------------------------------------------------------------
    // 4. Query model metadata: input/output names and shapes
    // -----------------------------------------------------------------------
    Ort::AllocatorWithDefaultOptions allocator;

    const size_t num_inputs = session.GetInputCount();
    const size_t num_outputs = session.GetOutputCount();
    std::cout << "Model inputs:  " << num_inputs << "\n";
    std::cout << "Model outputs: " << num_outputs << "\n";

    // Collect input/output names
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    for (size_t i = 0; i < num_inputs; ++i) {
      auto name = session.GetInputNameAllocated(i, allocator);
      std::cout << "  Input  " << i << ": " << name.get() << "\n";
      input_names.emplace_back(name.get());
    }
    for (size_t i = 0; i < num_outputs; ++i) {
      auto name = session.GetOutputNameAllocated(i, allocator);
      std::cout << "  Output " << i << ": " << name.get() << "\n";
      output_names.emplace_back(name.get());
    }
    std::cout << "\n";

    // -----------------------------------------------------------------------
    // 5. Prepare input tensors
    // -----------------------------------------------------------------------
    // Our model expects two float tensors of shape [1, 3].
    constexpr int64_t batch_size = 1;
    constexpr int64_t num_elements = 3;
    const std::array<int64_t, 2> input_shape = {batch_size, num_elements};

    std::array<float, num_elements> input_a = {1.0f, 2.0f, 3.0f};
    std::array<float, num_elements> input_b = {4.0f, 5.0f, 6.0f};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto tensor_a = Ort::Value::CreateTensor<float>(
        memory_info, input_a.data(), input_a.size(),
        input_shape.data(), input_shape.size());

    auto tensor_b = Ort::Value::CreateTensor<float>(
        memory_info, input_b.data(), input_b.size(),
        input_shape.data(), input_shape.size());

    THROW_IF_NOT(tensor_a.IsTensor());
    THROW_IF_NOT(tensor_b.IsTensor());

    // The Run() API expects arrays of C strings for input/output names.
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
    for (const auto& n : input_names) input_name_ptrs.push_back(n.c_str());
    for (const auto& n : output_names) output_name_ptrs.push_back(n.c_str());

    std::array<Ort::Value, 2> input_tensors{std::move(tensor_a), std::move(tensor_b)};

    // -----------------------------------------------------------------------
    // 6. Run inference
    // -----------------------------------------------------------------------
    std::cout << "Running inference...\n";

    Ort::RunOptions run_options;
    auto output_tensors = session.Run(
        run_options,
        input_name_ptrs.data(), input_tensors.data(), input_tensors.size(),
        output_name_ptrs.data(), output_name_ptrs.size());

    // -----------------------------------------------------------------------
    // 7. Process output
    // -----------------------------------------------------------------------
    THROW_IF_NOT(!output_tensors.empty() && output_tensors[0].IsTensor());

    const float* output_data = output_tensors[0].GetTensorData<float>();
    auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t output_count = type_info.GetElementCount();

    std::cout << "\nInputs:\n";
    std::cout << "  A = [";
    for (size_t i = 0; i < input_a.size(); ++i) {
      std::cout << (i ? ", " : "") << input_a[i];
    }
    std::cout << "]\n";

    std::cout << "  B = [";
    for (size_t i = 0; i < input_b.size(); ++i) {
      std::cout << (i ? ", " : "") << input_b[i];
    }
    std::cout << "]\n";

    std::cout << "\nOutput (A + B):\n";
    std::cout << "  C = [";
    for (size_t i = 0; i < output_count; ++i) {
      std::cout << (i ? ", " : "") << output_data[i];
    }
    std::cout << "]\n";

    // Verify correctness
    bool correct = true;
    for (size_t i = 0; i < num_elements; ++i) {
      if (output_data[i] != input_a[i] + input_b[i]) {
        correct = false;
        break;
      }
    }
    std::cout << "\nResult: " << (correct ? "PASS" : "FAIL") << "\n";

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime error: " << e.what() << "\n";
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
