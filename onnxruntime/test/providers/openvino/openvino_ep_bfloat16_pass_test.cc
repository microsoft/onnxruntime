// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <map>
#include <string>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/float16.h"

#include "test/util/include/test/test_environment.h"
#include "test/optimizer/qdq_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

extern std::unique_ptr<Ort::Env> ort_env;

class OVEP_BF16_Tests : public ::testing::TestWithParam<std::string> {};

namespace detail {
auto ConstructModel() {
  using namespace onnxruntime;
  using namespace test;

  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 19;
  Model model("Bfloat16Tester", true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, {}, DefaultLoggingManager().DefaultLogger());

  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);
  auto dim = 4;
  std::vector<float> input_data(dim, 1.0f);
  auto* input = builder.MakeInput<float>({dim}, input_data);
  builder.graph_.SetInputs({input});

  auto* cast_to_bf16 = builder.MakeIntermediate();
  Node& cast_node = builder.AddNode("Cast", {input}, {cast_to_bf16}, "");
  cast_node.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16));

  std::vector<onnxruntime::BFloat16> weight_data(dim * dim);
  for (std::size_t i = 0; i < weight_data.size(); ++i)
    weight_data[i] = onnxruntime::BFloat16(static_cast<float>(i % dim) / dim);
  auto* weights = builder.MakeInitializer<onnxruntime::BFloat16>({dim, dim}, weight_data);

  auto* matmul_out = builder.MakeIntermediate();
  builder.AddNode("MatMul", {cast_to_bf16, weights}, {matmul_out});

  std::vector<onnxruntime::BFloat16> weight_data_2(dim * dim);
  for (std::size_t i = 0; i < weight_data_2.size(); ++i)
    weight_data_2[i] = onnxruntime::BFloat16(static_cast<float>(i % dim) / dim);
  auto* weights_2 = builder.MakeInitializer<onnxruntime::BFloat16>({dim, dim}, weight_data_2);

  auto* matmul_out_2 = builder.MakeIntermediate();
  builder.AddNode("MatMul", {matmul_out, weights_2}, {matmul_out_2});

  auto* output = builder.MakeOutput();
  Node& cast2_node = builder.AddNode("Cast", {matmul_out_2}, {output});
  cast2_node.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));

  builder.SetGraphOutputs();
  auto st = model.MainGraph().Resolve();
  if (st != Status::OK())
    throw std::runtime_error(st.ErrorMessage());
  return model;
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
}  // namespace detail

namespace onnxruntime {
namespace test {

TEST_P(OVEP_BF16_Tests, TestModelConversion) {
  Ort::SessionOptions sessionOptions;
  std::unordered_map<std::string, std::string> ov_options;
  const auto& device = GetParam();
  if (!::detail::ProbeDevice(device))
    GTEST_SKIP() << device + " is not available on this machine";

  ov_options["device_type"] = device;
  auto model = ::detail::ConstructModel();
  sessionOptions.AppendExecutionProvider_OpenVINO_V2(ov_options);

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  auto model_data_span = AsByteSpan(model_data.data(), model_data.size());
  try {
    Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), sessionOptions);
  } catch (...) {
    FAIL();
  }
}
INSTANTIATE_TEST_SUITE_P(OVEP_Tests,
                         OVEP_BF16_Tests,
                         ::testing::Values("CPU", "GPU", "NPU"));
}  // namespace test
}  // namespace onnxruntime
