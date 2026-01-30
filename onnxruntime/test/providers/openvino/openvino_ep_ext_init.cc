// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <map>
#include <string>
#include <optional>

#include "core/session/onnxruntime_cxx_api.h"

#include "test/util/include/test/test_environment.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "onnxruntime_session_options_config_keys.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

extern std::unique_ptr<Ort::Env> ort_env;

class OVEP_ExtInit_Tests : public ::testing::TestWithParam<std::string> {};

namespace {

std::optional<std::vector<uint8_t>> LoadFileToMemory(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return std::nullopt;
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    return std::nullopt;
  }
  return buffer;
}

auto ProbeDevice(const std::string& device) {
  static std::map<std::string, bool> is_present;
  if (is_present.find(device) == is_present.end()) {
    Ort::SessionOptions sessionOptions;
    std::unordered_map<std::string, std::string> ov_options;
    ov_options["device_type"] = device;
    try {
      sessionOptions.AppendExecutionProvider_OpenVINO_V2(ov_options);
      is_present[device] = true;
    } catch (...) {
      is_present[device] = false;
    }
  }
  return is_present[device];
}
}  // namespace

namespace onnxruntime {
namespace test {

TEST_P(OVEP_ExtInit_Tests, ModelFromExtInit) {
  const auto& device = GetParam();
  if (!ProbeDevice(device))
    GTEST_SKIP() << device + " is not available on this machine";

  // Model and weights file paths
  const std::string model_path = "ovep_ext_init_test.onnx";
  const std::string weights_path = "ovep_ext_init_test.onnx.data";
  const size_t num_initializers = 8;
  const size_t floats_per_initializer = 64 * 1024 * 1024;  // 64 millions floats per initializer, 256MB
  const size_t total_floats = num_initializers * floats_per_initializer;
  const size_t total_bytes = total_floats * sizeof(float);
  // min size threshold for new logic with ext initializers
  ASSERT_GE(total_bytes, 32 * 1024 * 1024);

  // 1. Create initializers
  std::vector<std::vector<float>> initializer_data;
  for (size_t i = 0; i < num_initializers; ++i)
    initializer_data.emplace_back(floats_per_initializer, static_cast<float>(i + 1));  // W0:1, W1:2...

  // 2. Build ONNX model with 4 external initializers, and 4 ADD nodes
  {
    ModelProto model_proto;
    model_proto.set_ir_version(7);
    model_proto.set_producer_name("openvino_extinit_test");
    model_proto.set_producer_version("1.0");
    model_proto.set_domain("");
    model_proto.set_model_version(1);

    auto* graph = model_proto.mutable_graph();
    graph->set_name("TestGraph");

    // Input: shape [floats_per_initializer]
    auto* input = graph->add_input();
    input->set_name("X");
    auto* input_type = input->mutable_type()->mutable_tensor_type();
    input_type->set_elem_type(TensorProto_DataType_FLOAT);
    input_type->mutable_shape()->add_dim()->set_dim_value(floats_per_initializer);

    // Output: shape [floats_per_initializer]
    auto* output = graph->add_output();
    output->set_name("Y");
    auto* output_type = output->mutable_type()->mutable_tensor_type();
    output_type->set_elem_type(TensorProto_DataType_FLOAT);
    output_type->mutable_shape()->add_dim()->set_dim_value(floats_per_initializer);

    auto* opset_import = model_proto.add_opset_import();
    opset_import->set_domain("");
    opset_import->set_version(19);

    // Add initializers as external data
    size_t offset = 0;
    std::vector<std::string> initializer_names;
    for (size_t i = 0; i < num_initializers; ++i) {
      std::string name = "W" + std::to_string(i);
      initializer_names.push_back(name);
      TensorProto* initializer = graph->add_initializer();
      initializer->set_name(name);
      initializer->set_data_type(TensorProto_DataType_FLOAT);
      initializer->add_dims(floats_per_initializer);
      initializer->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
      auto* ext = initializer->add_external_data();
      ext->set_key("location");
      ext->set_value(weights_path);
      ext = initializer->add_external_data();
      ext->set_key("offset");
      ext->set_value(std::to_string(offset));
      ext = initializer->add_external_data();
      ext->set_key("length");
      ext->set_value(std::to_string(floats_per_initializer * sizeof(float)));
      offset += floats_per_initializer * sizeof(float);
    }

    // nodes: X -> Add with Init[0] -> ... -> output Y
    std::string prev_output = "X";
    std::string node_output;
    for (size_t i = 0; i < num_initializers; ++i) {
      node_output = (i == num_initializers - 1) ? "Y" : "A" + std::to_string(i);
      auto* add_node = graph->add_node();
      add_node->set_op_type("Add");
      add_node->add_input(prev_output);
      add_node->add_input(initializer_names[i]);
      add_node->add_output(node_output);
      prev_output = node_output;
    }

    // Save model
    std::ofstream model_file(model_path, std::ios::binary);
    ASSERT_TRUE(model_proto.SerializeToOstream(&model_file));
    model_file.close();
  }

  // 3. Save weights file (concatenate all initializers)
  {
    std::ofstream weights_file(weights_path, std::ios::binary);
    ASSERT_TRUE(weights_file.is_open());
    for (const auto& w : initializer_data) {
      weights_file.write(reinterpret_cast<const char*>(w.data()), w.size() * sizeof(float));
    }
    weights_file.close();
  }

  // 4. Load model and weights into memory
  auto model_data = LoadFileToMemory(model_path);
  auto weights_data = LoadFileToMemory(weights_path);
  ASSERT_TRUE(model_data.has_value() && weights_data.has_value());

  // 5. Prepare external initializer info
  PathString weights_name_path(weights_path.begin(), weights_path.end());
  std::vector<PathString> names_path = {weights_name_path};
  std::vector<char*> buffers = {reinterpret_cast<char*>(weights_data.value().data())};
  std::vector<size_t> buffer_sizes = {weights_data.value().size()};

  // 6. Set up session options with OpenVINO
  Ort::SessionOptions session_options;
  session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  session_options.SetIntraOpNumThreads(1);
  std::unordered_map<std::string, std::string> ov_options = {{"device_type", device}};
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);
  session_options.AddExternalInitializersFromFilesInMemory(names_path, buffers, buffer_sizes);

  // 7. Create session from memory
  Ort::Session session(*ort_env, model_data.value().data(), model_data.value().size(), session_options);

  // 8. Run inference to verify weights are loaded
  std::vector<float> input_data(floats_per_initializer, 2.0f);
  std::vector<int64_t> input_shape = {static_cast<int64_t>(floats_per_initializer)};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

  std::vector<const char*> input_names = {"X"};
  std::vector<const char*> output_names = {"Y"};
  std::vector<Ort::Value> output_tensors(1);

  session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), output_tensors.data(), 1);

  // Check output: should be input + W0 + W1 + W2...
  auto* out_data = output_tensors[0].GetTensorMutableData<float>();
  float expected = input_data[0];
  for (size_t i = 0; i < num_initializers; ++i) {
    expected += initializer_data[i][0];
  }

  for (size_t i = 0; i < floats_per_initializer; ++i)
    ASSERT_FLOAT_EQ(out_data[i], expected);

  // Cleanup
  std::filesystem::remove(model_path);
  std::filesystem::remove(weights_path);
}
INSTANTIATE_TEST_SUITE_P(OVEP_Tests,
                         OVEP_ExtInit_Tests,
                         ::testing::Values("CPU", "GPU", "NPU"));

}  // namespace test
}  // namespace onnxruntime
