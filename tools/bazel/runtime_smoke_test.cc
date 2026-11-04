// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "onnx/onnx_pb.h"

namespace {

void SetTensorType(ONNX_NAMESPACE::ValueInfoProto& value, const char* name,
                   std::initializer_list<int64_t> shape) {
  value.set_name(name);
  auto* tensor = value.mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
  for (int64_t dim : shape) {
    tensor->mutable_shape()->add_dim()->set_dim_value(dim);
  }
}

void TestMatMul(Ort::Env& env, const Ort::SessionOptions& options) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(8);
  model.add_opset_import()->set_version(13);
  auto* graph = model.mutable_graph();
  graph->set_name("bazel_matmul");
  auto* node = graph->add_node();
  node->set_op_type("MatMul");
  node->add_input("A");
  node->add_input("B");
  node->add_output("C");
  SetTensorType(*graph->add_input(), "A", {2, 3});
  SetTensorType(*graph->add_input(), "B", {3, 2});
  SetTensorType(*graph->add_output(), "C", {2, 2});
  const std::string bytes = model.SerializeAsString();
  Ort::Session session(env, bytes.data(), bytes.size(), options);
  const std::array<int64_t, 2> a_shape{2, 3};
  const std::array<int64_t, 2> b_shape{3, 2};
  std::array<float, 6> a{1, 2, 3, 4, 5, 6};
  std::array<float, 6> b{1, 2, 3, 4, 5, 6};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<Ort::Value, 2> inputs{
      Ort::Value::CreateTensor<float>(memory, a.data(), a.size(), a_shape.data(), a_shape.size()),
      Ort::Value::CreateTensor<float>(memory, b.data(), b.size(), b_shape.data(), b_shape.size())};
  const std::array<const char*, 2> input_names{"A", "B"};
  const std::array<const char*, 1> output_names{"C"};
  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                             output_names.data(), output_names.size());
  const std::array<float, 4> expected{22, 28, 49, 64};
  if (outputs[0].GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>({2, 2}) ||
      !std::equal(expected.begin(), expected.end(), outputs[0].GetTensorData<float>())) {
    throw std::runtime_error("MatMul inference failed; expected shape [2, 2] and values [22, 28, 49, 64].");
  }
}

void TestConv(Ort::Env& env, const Ort::SessionOptions& options) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(8);
  model.add_opset_import()->set_version(13);
  auto* graph = model.mutable_graph();
  graph->set_name("bazel_conv");
  auto* node = graph->add_node();
  node->set_op_type("Conv");
  node->add_input("X");
  node->add_input("W");
  node->add_output("Y");
  SetTensorType(*graph->add_input(), "X", {1, 1, 3, 3});
  SetTensorType(*graph->add_input(), "W", {1, 1, 2, 2});
  SetTensorType(*graph->add_output(), "Y", {1, 1, 2, 2});
  const std::string bytes = model.SerializeAsString();
  Ort::Session session(env, bytes.data(), bytes.size(), options);
  const std::array<int64_t, 4> x_shape{1, 1, 3, 3};
  const std::array<int64_t, 4> w_shape{1, 1, 2, 2};
  std::array<float, 9> x{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<float, 4> w{1, 1, 1, 1};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<Ort::Value, 2> inputs{
      Ort::Value::CreateTensor<float>(memory, x.data(), x.size(), x_shape.data(), x_shape.size()),
      Ort::Value::CreateTensor<float>(memory, w.data(), w.size(), w_shape.data(), w_shape.size())};
  const std::array<const char*, 2> input_names{"X", "W"};
  const std::array<const char*, 1> output_names{"Y"};
  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                             output_names.data(), output_names.size());
  const std::array<float, 4> expected{12, 16, 24, 28};
  if (outputs[0].GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>({1, 1, 2, 2}) ||
      !std::equal(expected.begin(), expected.end(), outputs[0].GetTensorData<float>())) {
    throw std::runtime_error("Conv inference failed; expected shape [1, 1, 2, 2] and values [12, 16, 24, 28].");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2 && argc != 3) {
      throw std::runtime_error("Expected a model path and optional 'cuda' provider argument.");
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bazel_smoke");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    if (argc == 3) {
      if (std::string(argv[2]) != "cuda") {
        throw std::runtime_error("Unknown provider argument; use 'cuda' or omit it for CPU.");
      }
      Ort::CUDAProviderOptions cuda_options;
      options.AppendExecutionProvider_CUDA_V2(*cuda_options);
      options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    }

    Ort::Session session(env, std::filesystem::path(argv[1]).c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name_a = session.GetInputNameAllocated(0, allocator);
    const auto input_name_b = session.GetInputNameAllocated(1, allocator);
    const auto output_name = session.GetOutputNameAllocated(0, allocator);
    const std::array<const char*, 2> input_names{input_name_a.get(), input_name_b.get()};
    const std::array<const char*, 1> output_names{output_name.get()};
    const std::array<int64_t, 2> shape{2, 3};
    std::array<float, 6> a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::array<float, 6> b{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<Ort::Value, 2> inputs{
        Ort::Value::CreateTensor<float>(memory, a.data(), a.size(), shape.data(), shape.size()),
        Ort::Value::CreateTensor<float>(memory, b.data(), b.size(), shape.data(), shape.size())};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                               output_names.data(), output_names.size());
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto* values = outputs[0].GetTensorData<float>();
    if (info.GetShape() != std::vector<int64_t>({2, 3})) {
      throw std::runtime_error("Add inference returned an unexpected shape; expected [2, 3].");
    }
    for (size_t i = 0; i < a.size(); ++i) {
      if (values[i] != a[i] + b[i]) {
        throw std::runtime_error("Add inference returned unexpected values; expected [11, 22, 33, 44, 55, 66].");
      }
    }
    TestMatMul(env, options);
    TestConv(env, options);
    std::cout << "ORT " << OrtGetApiBase()->GetVersionString() << ": "
              << (argc == 3 ? "CUDA (CPU fallback disabled)" : "CPU") << " Add, MatMul, and Conv inference passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ORT Bazel inference failed: " << error.what() << '\n';
    return 1;
  }
}
