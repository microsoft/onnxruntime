// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "test/util/include/default_providers.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/test_environment.h"
#include "test/compare_ortvalue.h"

using namespace std;

namespace onnxruntime {
namespace test {

std::unique_ptr<IExecutionProvider> GetExecutionProvider(const std::string& provider_type) {
  std::unique_ptr<IExecutionProvider> execution_provider;
  if (provider_type == onnxruntime::kCpuExecutionProvider)
    execution_provider = DefaultCpuExecutionProvider();
  else if (provider_type == onnxruntime::kCudaExecutionProvider)
    execution_provider = DefaultCudaExecutionProvider();
  else if (provider_type == onnxruntime::kDnnlExecutionProvider)
    execution_provider = DefaultDnnlExecutionProvider();
  else if (provider_type == onnxruntime::kNupharExecutionProvider)
    execution_provider = DefaultNupharExecutionProvider();
  else if (provider_type == onnxruntime::kTensorrtExecutionProvider)
    execution_provider = DefaultTensorrtExecutionProvider();
  else if (provider_type == onnxruntime::kOpenVINOExecutionProvider)
    execution_provider = DefaultOpenVINOExecutionProvider();
  else if (provider_type == onnxruntime::kNnapiExecutionProvider)
    execution_provider = DefaultNnapiExecutionProvider();
  else if (provider_type == onnxruntime::kAclExecutionProvider)
    execution_provider = DefaultAclExecutionProvider();
  else if (provider_type == onnxruntime::kRocmExecutionProvider)
    execution_provider = DefaultRocmExecutionProvider();
  // skip if execution provider is disabled
  if (execution_provider == nullptr) {
    return nullptr;
  }
  return execution_provider;
}

void CompareOpTester::CompareWithCPU(const std::string& target_provider_type,
                                     double per_sample_tolerance,
                                     double relative_per_sample_tolerance,
                                     const bool need_cpu_cast,
                                     const std::unordered_map<std::string, int>& extra_domain_to_version) {
#ifndef NDEBUG
  run_called_ = true;
#endif

  std::unique_ptr<IExecutionProvider> target_execution_provider = GetExecutionProvider(target_provider_type);
  ASSERT_TRUE(target_execution_provider != nullptr) << "provider_type " << target_provider_type << " is not supported.";

  auto p_model = BuildGraph(extra_domain_to_version);
  auto& graph = p_model->MainGraph();

  Status status;

  // In InferenceSession::Initialize(), the call to graph partitioner, which is responsible
  // for Inlining function bodies for ops whose kernel is missing happens before the 
  // Cast Transformer. As a result, for MLFloat16 tests where the node is missing a CPU kernel,
  // the function body is instead used for CPU pass. This option allows the comparison with 
  // the CPU kernel by adding the input/output casts before looking for a registered CPU kernel.
  if (need_cpu_cast) {
    InsertCastTransformer transformer("Test");
    bool modified = false;
    status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
    ASSERT_TRUE(status.IsOK());
  }

  status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    return;
  }

  // Hookup the inputs and outputs
  std::unordered_map<std::string, MLValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(feeds, output_names);

  // Run the model
  SessionOptions so;
  so.session_logid = op_;
  so.session_log_verbosity_level = 1;

  InferenceSession cpu_session_object{so, GetEnvironment()};

  // first run with cpu
  std::string s1;
  p_model->ToProto().SerializeToString(&s1);
  std::istringstream model_proto_str(s1);

  status = cpu_session_object.Load(model_proto_str);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Load failed with status: " << status.ErrorMessage();
    return;
  }

  status = cpu_session_object.Initialize();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Initialize failed with status: " << status.ErrorMessage();
    return;
  }

  RunOptions run_options;
  run_options.run_tag = op_;
  run_options.run_log_verbosity_level = 1;

  std::vector<MLValue> cpu_fetches;
  status = cpu_session_object.Run(run_options, feeds, output_names, &cpu_fetches);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Run failed with status: " << status.ErrorMessage();
    return;
  }

  // run with target provider
  // build the graph again as the cpu graph may be with casts
  auto p_tp_model = BuildGraph(extra_domain_to_version);
  auto& tp_graph = p_tp_model->MainGraph();

  status = tp_graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    return;
  }

  InferenceSession target_session_object{so, GetEnvironment()};
  EXPECT_TRUE(target_session_object.RegisterExecutionProvider(std::move(target_execution_provider)).IsOK());

  std::string s2;
  p_tp_model->ToProto().SerializeToString(&s2);
  std::istringstream model_proto_str1(s2);
  status = target_session_object.Load(model_proto_str1);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Load failed with status: " << status.ErrorMessage();
    return;
  }

  status = target_session_object.Initialize();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Initialize failed with status: " << status.ErrorMessage();
    return;
  }

  std::vector<MLValue> target_fetches;
  status = target_session_object.Run(run_options, feeds, output_names, &target_fetches);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  //compare
  ASSERT_TRUE(cpu_fetches.size() == target_fetches.size());
  for (size_t i = 0; i < cpu_fetches.size(); i++) {
    auto ret = CompareOrtValue(target_fetches[i], cpu_fetches[i], per_sample_tolerance, relative_per_sample_tolerance, false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
  }
}

}  // namespace test
}  // namespace onnxruntime
