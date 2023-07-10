// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/op_tester.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "test/util/include/test_environment.h"
namespace onnxruntime {
namespace test {

void OpTester::AddNodes(onnxruntime::Graph& graph,
                        std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                        std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                        std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) {
  // default behavior is to create a single Node for the op being tested,
  // with node inputs/outputs being 1:1 with graph inputs/outputs.
  auto& node = graph.AddNode("node1", op_, op_, graph_input_defs, graph_output_defs, nullptr, Domain());

  // Add the attributes if any
  for (auto& add_attribute_fn : add_attribute_funcs)
    add_attribute_fn(node);
}

onnxruntime::Model& OpTester::BuildModel(const std::unordered_map<std::string, int>& extra_domain_to_version,
                                         const ModelOptions& model_options) {
  const auto get_defs = [](std::vector<BaseTester::Data>& data) {
    std::vector<onnxruntime::NodeArg*> defs;
    defs.reserve(data.size());
    std::transform(data.begin(), data.end(), std::back_inserter(defs),
                   [](Data& data) { return &data.def; });
    return defs;
  };

  // Generate the input & output def lists
  std::vector<onnxruntime::NodeArg*> input_defs = get_defs(GetInputData());
  std::vector<onnxruntime::NodeArg*> output_defs = get_defs(GetOutputData());

  // Create a simple model
  std::unordered_map<std::string, int> domain_to_version(extra_domain_to_version);
  const auto& domain = Domain();
  if (domain_to_version.count(domain) == 0) {
    domain_to_version.insert({domain, Opset()});
  } else {
    auto key_val = extra_domain_to_version.find(domain);

    ORT_ENFORCE(key_val->second <= Opset());

    if (key_val->second < Opset()) {
      domain_to_version[domain] = Opset();
    }
  }

  model_ = std::make_unique<onnxruntime::Model>(
      "test", false, ModelMetaData(), PathString(), CustomSchemaRegistries(),
      domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>{},
      DefaultLoggingManager().DefaultLogger(),
      model_options);

  onnxruntime::Graph& graph = model_->MainGraph();

  AddNodes(graph, input_defs, output_defs, add_attribute_funcs_);
  AddInitializers(graph);

  return *model_;
}

void OpTester::CreateModelToTest(const ModelOptions& model_options, Model*& model) {
  Status status = Status::OK();

  model = &BuildModel({}, model_options);
  auto& graph = model->MainGraph();

  const auto& ctx = GetRunContext();
  if (GetAddShapeToTensorData() && ctx.expect_result == ExpectResult::kExpectFailure) {
    // capture possible exceptions from shape inference for invalid testcase
    ORT_TRY {
      status = graph.Resolve(ctx.resolve_options);
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what());
      });
    }
  } else {
    status = graph.Resolve(ctx.resolve_options);
  }

  if (!status.IsOK()) {
    model = nullptr;

    if (ctx.expect_result == ExpectResult::kExpectFailure) {
      ASSERT_THAT(status.ErrorMessage(), testing::HasSubstr(ctx.expected_failure_string));
    } else {
      ASSERT_TRUE(status.IsOK()) << "Resolve failed with status: " << status.ErrorMessage();
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
