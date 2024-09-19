// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/optimizer/graph_transform_test_builder.h"

#include <functional>
#include <string>
#include <vector>
#include <memory>

#include "core/common/inlined_containers_fwd.h"
#include "core/common/span_utils.h"
#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

// enable to dump model for debugging
#define SAVE_TEST_GRAPH 0

namespace onnxruntime {
namespace test {

static InlinedVector<std::byte> GetZeroPointBytes(int64_t zero_point, ONNX_NAMESPACE::TensorProto_DataType type) {
  switch (type) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      int8_t val = static_cast<int8_t>(zero_point);
      auto span = gsl::as_bytes(gsl::make_span(&val, 1));
      return InlinedVector<std::byte>(span.begin(), span.end());
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      uint8_t val = static_cast<uint8_t>(zero_point);
      auto span = gsl::as_bytes(gsl::make_span(&val, 1));
      return InlinedVector<std::byte>(span.begin(), span.end());
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
      int16_t val = static_cast<int16_t>(zero_point);
      auto span = gsl::as_bytes(gsl::make_span(&val, 1));
      return InlinedVector<std::byte>(span.begin(), span.end());
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
      uint16_t val = static_cast<uint16_t>(zero_point);
      auto span = gsl::as_bytes(gsl::make_span(&val, 1));
      return InlinedVector<std::byte>(span.begin(), span.end());
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      int32_t val = static_cast<int32_t>(zero_point);
      auto span = gsl::as_bytes(gsl::make_span(&val, 1));
      return InlinedVector<std::byte>(span.begin(), span.end());
    }
    default:
      ORT_THROW("Unhandled zero-point type ", type, ".");
  }
}

NodeArg* ModelTestBuilder::MakeInitializer(gsl::span<const int64_t> shape,
                                           ONNX_NAMESPACE::TensorProto_DataType elem_type,
                                           gsl::span<const std::byte> raw_data) {
  std::string name = graph_.GenerateNodeArgName("constant");
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_name(name);
  tensor_proto.set_data_type(elem_type);
  utils::SetRawDataInTensorProto(tensor_proto, raw_data.data(), raw_data.size());

  for (auto& dim : shape) {
    tensor_proto.add_dims(dim);
  }

  graph_.AddInitializedTensor(tensor_proto);

  return &graph_.GetOrCreateNodeArg(name, nullptr);
}

Node& ModelTestBuilder::AddQuantizeLinearNode(NodeArg* input_arg,
                                              float input_scale,
                                              int64_t input_zero_point,
                                              ONNX_NAMESPACE::TensorProto_DataType zero_point_type,
                                              NodeArg* output_arg,
                                              bool use_ms_domain) {
  std::vector<NodeArg*> input_args;
  input_args.push_back(input_arg);
  input_args.push_back(MakeScalarInitializer<float>(input_scale));

  InlinedVector<std::byte> zp_bytes = GetZeroPointBytes(input_zero_point, zero_point_type);
  input_args.push_back(MakeInitializer({}, zero_point_type, zp_bytes));

  std::string domain = use_ms_domain ? kMSDomain : "";
  return AddNode("QuantizeLinear", input_args, {output_arg}, domain);
}

Node& ModelTestBuilder::AddDequantizeLinearNode(NodeArg* input_arg,
                                                float input_scale,
                                                int64_t input_zero_point,
                                                ONNX_NAMESPACE::TensorProto_DataType zero_point_type,
                                                NodeArg* output_arg,
                                                bool use_ms_domain) {
  std::vector<NodeArg*> input_args;
  input_args.push_back(input_arg);
  input_args.push_back(MakeScalarInitializer<float>(input_scale));

  InlinedVector<std::byte> zp_bytes = GetZeroPointBytes(input_zero_point, zero_point_type);
  input_args.push_back(MakeInitializer({}, zero_point_type, zp_bytes));

  std::string domain = use_ms_domain ? kMSDomain : "";
  return AddNode("DequantizeLinear", input_args, {output_arg}, domain);
}

void TransformerTester(const std::function<void(ModelTestBuilder& helper)>& build_test_case,
                       const std::function<void(InferenceSessionWrapper& session)>& check_transformed_graph,
                       TransformerLevel baseline_level,
                       TransformerLevel target_level,
                       const std::vector<int>& opset_versions,
                       double per_sample_tolerance,
                       double relative_per_sample_tolerance,
                       std::unique_ptr<GraphTransformer> transformer,
                       const std::function<void(SessionOptions&)>& add_session_options,
                       const InlinedHashSet<std::string>& disabled_optimizers) {
  ASSERT_TRUE(transformer == nullptr);
  for (auto opset_version : opset_versions) {
    TransformerTester(build_test_case,
                      check_transformed_graph,
                      baseline_level,
                      target_level,
                      opset_version,
                      per_sample_tolerance,
                      relative_per_sample_tolerance,
                      nullptr,
                      add_session_options,
                      disabled_optimizers);
  }
}

void TransformerTester(const std::function<void(ModelTestBuilder& helper)>& build_test_case,
                       const std::function<void(InferenceSessionWrapper& session)>& check_transformed_graph,
                       TransformerLevel baseline_level,
                       TransformerLevel target_level,
                       int opset_version,
                       double per_sample_tolerance,
                       double relative_per_sample_tolerance,
                       std::unique_ptr<GraphTransformer> transformer,
                       const std::function<void(SessionOptions&)>& add_session_options,
                       const InlinedHashSet<std::string>& disabled_optimizers,
                       std::unique_ptr<IExecutionProvider> ep) {
  // Build the model for this test.
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = opset_version;
  domain_to_version[kMSDomain] = 1;
  Model model("TransformerTester", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, {}, DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  ASSERT_TRUE(build_test_case);
  build_test_case(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  std::shared_ptr<IExecutionProvider> ep_shared = ep ? std::move(ep) : nullptr;

  auto run_model = [&](TransformerLevel level, std::vector<OrtValue>& fetches,
                       std::unique_ptr<GraphTransformer> transformer = nullptr) {
    SessionOptions session_options;
    session_options.graph_optimization_level = transformer ? baseline_level : level;
#if SAVE_TEST_GRAPH
    session_options.optimized_model_filepath =
        ToPathString("model" + std::to_string(static_cast<int>(level)) + ".onnx");
#endif
    if (add_session_options) {
      add_session_options(session_options);
    }
    InferenceSessionWrapper session{session_options, GetEnvironment()};
    if (ep_shared) {
      ASSERT_STATUS_OK(session.RegisterExecutionProvider(ep_shared));
    }

    ASSERT_STATUS_OK(session.Load(model_data.data(), static_cast<int>(model_data.size())));
    if (transformer) {
      ASSERT_STATUS_OK(session.RegisterGraphTransformer(std::move(transformer), level));
    } else if (!disabled_optimizers.empty()) {
      ASSERT_STATUS_OK(session.FilterEnabledOptimizers(InlinedHashSet<std::string>{disabled_optimizers}));
    }

    ASSERT_STATUS_OK(session.Initialize());

    RunOptions run_options;
    ASSERT_STATUS_OK(session.Run(run_options,
                                 helper.feeds_,
                                 helper.output_names_,
                                 &fetches));

    if (level == target_level) {
      if (check_transformed_graph) {
        check_transformed_graph(session);
      }
    }
  };

  std::vector<OrtValue> baseline_fetches;
  ASSERT_NO_FATAL_FAILURE(run_model(baseline_level, baseline_fetches));

  std::vector<OrtValue> target_fetches;
  ASSERT_NO_FATAL_FAILURE(run_model(target_level, target_fetches, std::move(transformer)));

  size_t num_outputs = baseline_fetches.size();
  ASSERT_EQ(num_outputs, target_fetches.size());

  for (size_t i = 0; i < num_outputs; i++) {
    std::pair<COMPARE_RESULT, std::string> ret =
        CompareOrtValue(target_fetches[i],
                        baseline_fetches[i],
                        per_sample_tolerance,
                        relative_per_sample_tolerance,
                        false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
  }
}

Status TestGraphTransformer(const std::function<void(ModelTestBuilder& helper)>& build_test_case, int opset_version,
                            const logging::Logger& logger, std::unique_ptr<GraphTransformer> transformer,
                            TransformerLevel level, unsigned steps, const std::function<Status(Graph&)>& pre_graph_checker,
                            const std::function<Status(Graph&)>& post_graph_checker) {
  const std::vector<int> opset_versions{opset_version};
  return TestGraphTransformer(build_test_case, opset_versions, logger, std::move(transformer),
                              level, steps, pre_graph_checker, post_graph_checker);
}

Status TestGraphTransformer(const std::function<void(ModelTestBuilder& helper)>& build_test_case,
                            const std::vector<int>& opset_versions,
                            const logging::Logger& logger, std::unique_ptr<GraphTransformer> transformer,
                            TransformerLevel level, unsigned steps, const std::function<Status(Graph&)>& pre_graph_checker,
                            const std::function<Status(Graph&)>& post_graph_checker) {
  onnxruntime::GraphTransformerManager graph_transformation_mgr{steps};
  ORT_RETURN_IF_ERROR(graph_transformation_mgr.Register(std::move(transformer), level));

  for (auto opset : opset_versions) {
    // Build the model for this test.
    std::unordered_map<std::string, int> domain_to_version;
    domain_to_version[kOnnxDomain] = opset;
    domain_to_version[kMSDomain] = 1;
    Model model("TransformerTester", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                domain_to_version, {}, logger);
    Graph& graph = model.MainGraph();
    ModelTestBuilder helper(graph);
    ORT_RETURN_IF_NOT(build_test_case, "build_test_case must be provided");
    build_test_case(helper);
    helper.SetGraphOutputs();
    ORT_RETURN_IF_ERROR(graph.Resolve());
    if (pre_graph_checker) {
      ORT_RETURN_IF_ERROR(pre_graph_checker(graph));
    }
#if SAVE_TEST_GRAPH
    ORT_RETURN_IF_ERROR(Model::Save(model, ToPathString("model_original.onnx")));
#endif
    ORT_RETURN_IF_ERROR(graph_transformation_mgr.ApplyTransformers(graph, level, logger));
    if (post_graph_checker) {
      ORT_RETURN_IF_ERROR(post_graph_checker(graph));
    }
#if SAVE_TEST_GRAPH
    ORT_RETURN_IF_ERROR(Model::Save(model, ToPathString("model_optimized.onnx")));
#endif
  };

  return Status::OK();
}

}  // namespace test
}  // namespace onnxruntime
