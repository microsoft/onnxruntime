// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"

#include "core/optimizer/insert_cast_transformer.h"
#include "core/session/inference_session.h"

#include "test/util/include/asserts.h"
#include "test/util/include/compare_ortvalue.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_environment.h"

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
  else if (provider_type == onnxruntime::kDmlExecutionProvider)
    execution_provider = DefaultDmlExecutionProvider();
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
  SetTestFunctionCalled();

  std::unique_ptr<IExecutionProvider> target_execution_provider = GetExecutionProvider(target_provider_type);
  ASSERT_TRUE(target_execution_provider != nullptr) << "provider_type " << target_provider_type
                                                    << " is not supported.";

  auto& model = BuildModel(extra_domain_to_version);
  auto& graph = model.MainGraph();

  // In InferenceSession::Initialize(), the call to graph partitioner, which is responsible
  // for Inlining function bodies for ops whose kernel is missing happens before the
  // Cast Transformer. As a result, for MLFloat16 tests where the node is missing a CPU kernel,
  // the function body is instead used for CPU pass. This option allows the comparison with
  // the CPU kernel by adding the input/output casts before looking for a registered CPU kernel.
  if (need_cpu_cast) {
    InsertCastTransformer transformer("Test", GetExecutionProvider(kCpuExecutionProvider)->GetKernelRegistry().get());
    bool modified = false;
    ASSERT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  }

  ASSERT_STATUS_OK(graph.Resolve());

  // Hookup the inputs and outputs
  std::unordered_map<std::string, OrtValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(feeds, output_names);

  // Run the model
  SessionOptions so;
  so.session_logid = Op();

  InferenceSession cpu_session_object{so, GetEnvironment()};

  // first run with cpu
  std::string s1;
  model.ToProto().SerializeToString(&s1);
  std::istringstream model_proto_str(s1);

  ASSERT_STATUS_OK(cpu_session_object.Load(model_proto_str));

  ASSERT_STATUS_OK(cpu_session_object.Initialize());

  std::vector<OrtValue> cpu_fetches;
  ASSERT_STATUS_OK(cpu_session_object.Run({}, feeds, output_names, &cpu_fetches));

  // run with target provider
  // build the graph again as the cpu graph may be with casts
  auto& tp_model = BuildModel(extra_domain_to_version);
  auto& tp_graph = tp_model.MainGraph();

  ASSERT_STATUS_OK(tp_graph.Resolve());

  InferenceSession target_session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(target_session_object.RegisterExecutionProvider(std::move(target_execution_provider)));

  std::string s2;
  tp_model.ToProto().SerializeToString(&s2);
  std::istringstream model_proto_str1(s2);
  ASSERT_STATUS_OK(target_session_object.Load(model_proto_str1));

  ASSERT_STATUS_OK(target_session_object.Initialize());

  std::vector<OrtValue> target_fetches;
  ASSERT_STATUS_OK(target_session_object.Run({}, feeds, output_names, &target_fetches));

  // compare
  ASSERT_TRUE(cpu_fetches.size() == target_fetches.size());
  for (size_t i = 0; i < cpu_fetches.size(); i++) {
    auto ret = CompareOrtValue(target_fetches[i], cpu_fetches[i], per_sample_tolerance, relative_per_sample_tolerance, false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
  }
}

}  // namespace test
}  // namespace onnxruntime
