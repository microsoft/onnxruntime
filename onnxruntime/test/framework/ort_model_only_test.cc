// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test_utils.h"
#include "test/util/include/asserts.h"

#include "gtest/gtest.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

namespace onnxruntime {

// InferenceSession wrapper to expose loaded graph.
class InferenceSessionGetGraphWrapper : public InferenceSession {
 public:
  explicit InferenceSessionGetGraphWrapper(const SessionOptions& session_options,
                                           const Environment& env) : InferenceSession(session_options, env) {
  }

  const Graph& GetGraph() {
    return model_->MainGraph();
  }

  const SessionState& GetSessionState() {
    return InferenceSession::GetSessionState();
  }
};

namespace test {

#if !defined(ORT_MINIMAL_BUILD)
// Same Tensor from ONNX and ORT format will have different binary representation, need to compare value by value
static void CompareTensors(const OrtValue& left_value, const OrtValue& right_value) {
  const Tensor& left = left_value.Get<Tensor>();
  const Tensor& right = right_value.Get<Tensor>();

  ASSERT_EQ(left.Shape().GetDims(), right.Shape().GetDims());
  ASSERT_EQ(left.GetElementType(), right.GetElementType());

  if (left.IsDataTypeString()) {
    auto size = left.Shape().Size();
    const auto* left_strings = left.Data<std::string>();
    const auto* right_strings = right.Data<std::string>();

    for (int i = 0; i < size; ++i) {
      EXPECT_EQ(left_strings[i], right_strings[i]) << "Mismatch index:" << i;
    }
  } else {
    ASSERT_EQ(memcmp(left.DataRaw(), right.DataRaw(), left.SizeInBytes()), 0);
  }
}

static void CompareValueInfos(const ValueInfoProto& left, const ValueInfoProto& right) {
  ASSERT_EQ(left.name(), right.name());
  ASSERT_EQ(left.doc_string(), right.doc_string());

  std::string left_data;
  std::string right_data;

  const auto& left_type_proto = left.type();
  const auto& right_type_proto = right.type();

  ASSERT_EQ(left_type_proto.denotation(), right_type_proto.denotation());
  ASSERT_TRUE(left_type_proto.has_tensor_type());
  ASSERT_TRUE(right_type_proto.has_tensor_type());

  const auto& left_tensor_type = left_type_proto.tensor_type();
  const auto& right_tensor_type = right_type_proto.tensor_type();

  ASSERT_EQ(left_tensor_type.elem_type(), right_tensor_type.elem_type());

  const auto& left_shape = left_tensor_type.shape();
  const auto& right_shape = right_tensor_type.shape();

  ASSERT_EQ(left_shape.dim_size(), right_shape.dim_size());
  for (int i = 0; i < left_shape.dim_size(); i++) {
    const auto& left_dim = left_shape.dim(i);
    const auto& right_dim = right_shape.dim(i);
    ASSERT_EQ(left_dim.has_dim_value(), right_dim.has_dim_value());
    ASSERT_EQ(left_dim.dim_value(), right_dim.dim_value());
    ASSERT_EQ(left_dim.has_dim_param(), right_dim.has_dim_param());
    ASSERT_EQ(left_dim.dim_param(), right_dim.dim_param());
  }
}

TEST(OrtModelOnlyTests, SerializeToOrtFormat) {
  const auto output_file = ORT_TSTR("ort_github_issue_4031.onnx.ort");
  SessionOptions so;
  so.session_logid = "SerializeToOrtFormat";
  so.optimized_model_filepath = output_file;
  // not strictly necessary - type should be inferred from the filename
  so.AddConfigEntry(ORT_SESSION_OPTIONS_CONFIG_SAVE_MODEL_FORMAT, "ORT");

  InferenceSessionGetGraphWrapper session_object{so, GetEnvironment()};

  // create .ort file during Initialize due to values in SessionOptions
  ASSERT_STATUS_OK(session_object.Load(ORT_TSTR("testdata/ort_github_issue_4031.onnx")));
  ASSERT_STATUS_OK(session_object.Initialize());

  // create inputs
  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, {123.f},
                       &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("state_var_in", ml_value));

  // prepare outputs
  std::vector<std::string> output_names{"state_var_out"};
  std::vector<OrtValue> fetches;

  ASSERT_STATUS_OK(session_object.Run(feeds, output_names, &fetches));

  SessionOptions so2;
  so.session_logid = "LoadOrtFormat";
  // not strictly necessary - type should be inferred from the filename, but to be sure we're testing what we
  // think we're testing set it.
  so.AddConfigEntry(ORT_SESSION_OPTIONS_CONFIG_LOAD_MODEL_FORMAT, "ORT");

  // load serialized version
  InferenceSessionGetGraphWrapper session_object2{so2, GetEnvironment()};
  ASSERT_STATUS_OK(session_object2.Load(output_file));
  ASSERT_STATUS_OK(session_object2.Initialize());

  // compare contents on Graph instances
  const auto& graph = session_object.GetGraph();
  const auto& graph2 = session_object2.GetGraph();

  const auto& session_state = session_object.GetSessionState();
  const auto& session_state2 = session_object2.GetSessionState();

  const auto& name_idx_map = session_state.GetOrtValueNameIdxMap();
  const auto& name_idx_map2 = session_state2.GetOrtValueNameIdxMap();

  const auto& i1 = session_state.GetInitializedTensors();
  const auto& i2 = session_state2.GetInitializedTensors();
  ASSERT_EQ(i1.size(), i2.size());

  for (const auto& pair : i1) {
    auto iter = i2.find(pair.first);
    ASSERT_NE(iter, i2.cend());

    const OrtValue& left = pair.second;
    const OrtValue& right = iter->second;
    CompareTensors(left, right);

    // check NodeArgs for both initializers. need to get name from map as we store the initialized tensors against
    // their OrtValueIdx in SessionState.
    std::string name;
    std::string name2;
    ASSERT_STATUS_OK(name_idx_map.GetName(pair.first, name));
    ASSERT_STATUS_OK(name_idx_map2.GetName(pair.first, name2));
    ASSERT_EQ(name, name2);

    const auto& left_nodearg = *graph.GetNodeArg(name);
    const auto& right_nodearg = *graph2.GetNodeArg(name2);
    CompareValueInfos(left_nodearg.ToProto(), right_nodearg.ToProto());
  }

  // check all node args are fine
  for (const auto& input : graph.GetInputs()) {
    const auto& left = *graph.GetNodeArg(input->Name());
    const auto* right = graph2.GetNodeArg(input->Name());
    ASSERT_TRUE(right != nullptr);

    const auto& left_proto = left.ToProto();
    const auto& right_proto = right->ToProto();
    CompareValueInfos(left_proto, right_proto);
  }

  for (const auto& left : graph.Nodes()) {
    const auto* right = graph2.GetNode(left.Index());
    ASSERT_TRUE(right != nullptr);
    const auto& left_outputs = left.OutputDefs();
    const auto& right_outputs = right->OutputDefs();
    ASSERT_EQ(left_outputs.size(), right_outputs.size());

    for (size_t i = 0, end = left_outputs.size(); i < end; ++i) {
      const auto& left_nodearg = *left_outputs[i];
      const auto& right_nodearg = *right_outputs[i];

      if (left_nodearg.Exists()) {
        EXPECT_EQ(left_nodearg.Name(), right_nodearg.Name());
        CompareValueInfos(left_nodearg.ToProto(), right_nodearg.ToProto());
      } else {
        EXPECT_FALSE(right_nodearg.Exists());
      }
    }
  }

  // check results match
  std::vector<OrtValue> fetches2;
  ASSERT_STATUS_OK(session_object2.Run(feeds, output_names, &fetches2));

  const auto& output = fetches[0].Get<Tensor>();
  ASSERT_TRUE(output.Shape().Size() == 1);
  ASSERT_TRUE(output.Data<float>()[0] == 125.f);

  const auto& output2 = fetches2[0].Get<Tensor>();
  ASSERT_TRUE(output2.Shape().Size() == 1);
  ASSERT_TRUE(output2.Data<float>()[0] == 125.f);
}
#endif

// test that we can deserialize and run a previously saved ORT format model
// TEMPORARY
// This works locally when loading the model produced by SerializeToOrtFormat but fails to find the kernel
// for Loop if using the pre-saved model in testdata despite there being no binary difference between the two.
// The hash for Loop is correct (14070537928877630320) according the the CI failure error message.
TEST(OrtModelOnlyTests, DISABLED_LoadOrtFormatModel) {
  const auto model_filename = ORT_TSTR("testdata/ort_github_issue_4031.onnx.ort");
  SessionOptions so;
  so.session_logid = "LoadOrtFormatModel";

  InferenceSessionGetGraphWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_filename));  // infer type from filename
  ASSERT_STATUS_OK(session_object.Initialize());

  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, {123.f},
                       &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("state_var_in", ml_value));

  // prepare outputs
  std::vector<std::string> output_names{"state_var_out"};
  std::vector<OrtValue> fetches;

  ASSERT_STATUS_OK(session_object.Run(feeds, output_names, &fetches));

  const auto& output = fetches[0].Get<Tensor>();
  ASSERT_TRUE(output.Shape().Size() == 1);
  ASSERT_TRUE(output.Data<float>()[0] == 125.f);
}

}  // namespace test
}  // namespace onnxruntime
