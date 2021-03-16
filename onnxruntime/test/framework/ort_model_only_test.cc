// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// if we can't load an ORT format model we can't really test anything
#if defined(ENABLE_ORT_FORMAT_LOAD)

#include "core/common/make_unique.h"
#include "core/framework/data_types.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

#include "core/session/onnxruntime_cxx_api.h"

#include "gtest/gtest.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

namespace onnxruntime {
namespace test {

struct OrtModelTestInfo {
  std::basic_string<ORTCHAR_T> model_filename;
  std::string logid;
  NameMLValMap inputs;
  std::vector<std::string> output_names;
  std::function<void(const std::vector<OrtValue>&)> output_verifier;
  std::vector<std::pair<std::string, std::string>> configs;
  bool run_use_buffer{false};
};

static void RunOrtModel(const OrtModelTestInfo& test_info) {
  SessionOptions so;
  so.session_logid = test_info.logid;
  for (const auto& config : test_info.configs)
    so.AddConfigEntry(config.first.c_str(), config.second.c_str());

  std::vector<char> model_data;
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  if (test_info.run_use_buffer) {
    // Load the file into a buffer and use the buffer to create inference session
    size_t num_bytes = 0;
    ASSERT_STATUS_OK(Env::Default().GetFileLength(test_info.model_filename.c_str(), num_bytes));
    model_data.resize(num_bytes);
    std::ifstream bytes_stream(test_info.model_filename, std::ifstream::in | std::ifstream::binary);
    bytes_stream.read(model_data.data(), num_bytes);
    bytes_stream.close();
    ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(num_bytes)));
  } else {
    ASSERT_STATUS_OK(session_object.Load(test_info.model_filename));  // infer type from filename
  }

  ASSERT_STATUS_OK(session_object.Initialize());

  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session_object.Run(test_info.inputs, test_info.output_names, &fetches));
  test_info.output_verifier(fetches);
}

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

// Keep the CompareTypeProtos in case we need debug the difference
/*
static void CompareTypeProtos(const TypeProto& left_type_proto, const TypeProto& right_type_proto) {
  ASSERT_EQ(left_type_proto.denotation(), right_type_proto.denotation());

  ASSERT_EQ(left_type_proto.has_tensor_type(), right_type_proto.has_tensor_type());
  ASSERT_EQ(left_type_proto.has_sequence_type(), right_type_proto.has_sequence_type());
  ASSERT_EQ(left_type_proto.has_map_type(), right_type_proto.has_map_type());

  if (left_type_proto.has_tensor_type()) {
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
  } else if (left_type_proto.has_sequence_type()) {
    CompareTypeProtos(left_type_proto.sequence_type().elem_type(), right_type_proto.sequence_type().elem_type());
  } else if (left_type_proto.has_map_type()) {
    const auto& left_map = left_type_proto.map_type();
    const auto& right_map = right_type_proto.map_type();
    ASSERT_EQ(left_map.key_type(), right_map.key_type());
    CompareTypeProtos(left_map.value_type(), right_map.value_type());
  } else {
    FAIL();  // We do not support SparseTensor and Opaque for now
  }
}
*/

static void CompareValueInfos(const ValueInfoProto& left, const ValueInfoProto& right) {
  const auto str_left = left.SerializeAsString();
  const auto str_right = right.SerializeAsString();
  ASSERT_EQ(str_left, str_right);

  // Keep the ValueInfoProto content comparison in case we need debug the difference
  // ASSERT_EQ(left.name(), right.name());
  // ASSERT_EQ(left.doc_string(), right.doc_string());
  // CompareTypeProtos(left.type(), right.type());
}

static void CompareGraphAndSessionState(const InferenceSessionWrapper& session_object_1,
                                        const InferenceSessionWrapper& session_object_2) {
  const auto& graph_1 = session_object_1.GetGraph();
  const auto& graph_2 = session_object_2.GetGraph();

  const auto& session_state_1 = session_object_1.GetSessionState();
  const auto& session_state_2 = session_object_2.GetSessionState();

  const auto& i1 = session_state_1.GetInitializedTensors();
  const auto& i2 = session_state_2.GetInitializedTensors();
  ASSERT_EQ(i1.size(), i2.size());

  for (const auto& pair : i1) {
    auto iter = i2.find(pair.first);
    ASSERT_NE(iter, i2.cend());

    const OrtValue& left = pair.second;
    const OrtValue& right = iter->second;
    CompareTensors(left, right);
  }

  // check all node args are fine
  for (const auto& input : graph_1.GetInputsIncludingInitializers()) {
    const auto& left = *graph_1.GetNodeArg(input->Name());
    const auto* right = graph_2.GetNodeArg(input->Name());
    ASSERT_TRUE(right != nullptr);

    const auto& left_proto = left.ToProto();
    const auto& right_proto = right->ToProto();
    CompareValueInfos(left_proto, right_proto);
  }

  for (const auto& left : graph_1.Nodes()) {
    const auto* right = graph_2.GetNode(left.Index());
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
}

static void SaveAndCompareModels(const std::string& onnx_file, const std::basic_string<ORTCHAR_T>& ort_file) {
  SessionOptions so;
  so.session_logid = "SerializeToOrtFormat";
  so.optimized_model_filepath = ort_file;
  // not strictly necessary - type should be inferred from the filename
  so.AddConfigEntry(kOrtSessionOptionsConfigSaveModelFormat, "ORT");
  InferenceSessionWrapper session_object{so, GetEnvironment()};

  // create .ort file during Initialize due to values in SessionOptions
  ASSERT_STATUS_OK(session_object.Load(onnx_file));
  ASSERT_STATUS_OK(session_object.Initialize());

  SessionOptions so2;
  so2.session_logid = "LoadOrtFormat";
  // not strictly necessary - type should be inferred from the filename, but to be sure we're testing what we
  // think we're testing set it.
  so2.AddConfigEntry(kOrtSessionOptionsConfigLoadModelFormat, "ORT");

  // load serialized version
  InferenceSessionWrapper session_object2{so2, GetEnvironment()};
  ASSERT_STATUS_OK(session_object2.Load(ort_file));
  ASSERT_STATUS_OK(session_object2.Initialize());

  CompareGraphAndSessionState(session_object, session_object2);
}

/*
static void DumpOrtModelAsJson(const std::string& model_uri) {
  std::string ort_repo_root("path to your ORT repo root");
  std::string ort_flatbuffers_dir(ort_repo_root + "onnxruntime/core/flatbuffers/");
  std::string schemafile(ort_flatbuffers_dir + "ort.fbs");
  std::string jsonfile;

  ORT_ENFORCE(flatbuffers::LoadFile(schemafile.c_str(), false, &schemafile));
  flatbuffers::Parser parser;
  const char* include_directories[] = {ort_flatbuffers_dir.c_str(), nullptr};

  ORT_ENFORCE(parser.Parse(schemafile.c_str(), include_directories));

  std::string flatbuffer;
  std::string json;
  flatbuffers::LoadFile(model_uri.c_str(), true, &flatbuffer);
  flatbuffers::GenerateText(parser, flatbuffer.data(), &json);
  std::ofstream(model_uri + ".json") << json;
}
*/

/* 
Validate we don't run optimizers on an ORT format model in a full build. The optimizers will remove nodes,
which will create a mismatch with the saved kernel information and result in a runtime error. 
We could take steps to handle this scenario in a full build, but for consistency we choose to not run optimizers 
on any ORT format model.
*/
TEST(OrtModelOnlyTests, ValidateOrtFormatModelDoesNotRunOptimizersInFullBuild) {
  const std::basic_string<ORTCHAR_T> ort_file = ORT_TSTR("testdata/mnist.onnx.test_output.ort");
  SaveAndCompareModels("testdata/mnist.onnx", ort_file);

  // DumpOrtModelAsJson(ToMBString(ort_file));

  OrtModelTestInfo test_info;
  test_info.model_filename = ort_file;
  test_info.logid = "ValidateOrtFormatModelDoesNotRunOptimizersInFullBuild";
  test_info.configs.push_back(std::make_pair(kOrtSessionOptionsConfigLoadModelFormat, "ORT"));

  OrtValue ml_value;
  vector<float> data(28 * 28, 0.0);
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1, 1, 28, 28}, data,
                       &ml_value);
  test_info.inputs.insert(std::make_pair("Input3", ml_value));

  // prepare outputs
  test_info.output_names = {"Plus214_Output_0"};
  test_info.output_verifier = [](const std::vector<OrtValue>& fetches) {
    const auto& output = fetches[0].Get<Tensor>();
    ASSERT_TRUE(output.Shape().NumDimensions() == 2);
  };

  RunOrtModel(test_info);
}

TEST(OrtModelOnlyTests, SerializeToOrtFormat) {
  const std::basic_string<ORTCHAR_T> ort_file = ORT_TSTR("testdata/ort_github_issue_4031.onnx.test_output.ort");
  SaveAndCompareModels("testdata/ort_github_issue_4031.onnx", ort_file);

  // DumpOrtModelAsJson(ToMBString(ort_file));

  OrtModelTestInfo test_info;
  test_info.model_filename = ort_file;
  test_info.logid = "SerializeToOrtFormat";
  test_info.configs.push_back(std::make_pair(kOrtSessionOptionsConfigLoadModelFormat, "ORT"));

  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, {123.f},
                       &ml_value);
  test_info.inputs.insert(std::make_pair("state_var_in", ml_value));

  // prepare outputs
  test_info.output_names = {"state_var_out"};
  test_info.output_verifier = [](const std::vector<OrtValue>& fetches) {
    const auto& output = fetches[0].Get<Tensor>();
    ASSERT_TRUE(output.Shape().Size() == 1);
    ASSERT_TRUE(output.Data<float>()[0] == 125.f);
  };

  RunOrtModel(test_info);
}

TEST(OrtModelOnlyTests, SparseInitializerHandling) {
  const std::basic_string<ORTCHAR_T> ort_file =
      ORT_TSTR("testdata/ort_minimal_test_models/sparse_initializer_handling.onnx.test_output.ort");
  SaveAndCompareModels("testdata/ort_minimal_test_models/sparse_initializer_handling.onnx", ort_file);

  SessionOptions so;
  so.session_logid = "SparseInitializerHandling";
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(ort_file));
  ASSERT_STATUS_OK(session_object.Initialize());

  // Check that there are no duplicates for initializers
  const auto* init_list = session_object.GetOverridableInitializers().second;
  ASSERT_EQ(init_list->size(), 1U);
  const auto& init_def = *init_list->front();
  ASSERT_EQ(init_def.Name(), "x");
}

// regression test to make sure the model path is correctly passed through when serializing a tensor attribute
TEST(OrtModelOnlyTests, TensorAttributeSerialization) {
  const std::basic_string<ORTCHAR_T> ort_file =
      ORT_TSTR("testdata/ort_minimal_test_models/tensor_attribute.onnx.test_output.ort");
  SaveAndCompareModels("testdata/ort_minimal_test_models/tensor_attribute.onnx", ort_file);
}

#if !defined(DISABLE_ML_OPS)
TEST(OrtModelOnlyTests, SerializeToOrtFormatMLOps) {
  const std::basic_string<ORTCHAR_T> ort_file =
      ORT_TSTR("testdata/sklearn_bin_voting_classifier_soft_converted.test_output.ort");
  SaveAndCompareModels("testdata/sklearn_bin_voting_classifier_soft.onnx", ort_file);

  OrtModelTestInfo test_info;
  test_info.model_filename = ort_file;
  test_info.logid = "SerializeToOrtFormatMLOps";
  test_info.configs.push_back(std::make_pair(kOrtSessionOptionsConfigLoadModelFormat, "ORT"));

  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {3, 2},
                       {0.f, 1.f, 1.f, 1.f, 2.f, 0.f}, &ml_value);
  test_info.inputs.insert(std::make_pair("input", ml_value));

  // prepare outputs
  test_info.output_names = {"output_label", "output_probability"};
  test_info.output_verifier = [](const std::vector<OrtValue>& fetches) {
    const auto& output_0 = fetches[0].Get<Tensor>();
    int64_t tensor_size = 3;
    ASSERT_EQ(tensor_size, output_0.Shape().Size());
    const auto& output_0_data = output_0.Data<std::string>();
    for (int64_t i = 0; i < tensor_size; i++)
      ASSERT_TRUE(output_0_data[i] == "A");

    VectorMapStringToFloat expected_output_1 = {{{"A", 0.572734f}, {"B", 0.427266f}},
                                                {{"A", 0.596016f}, {"B", 0.403984f}},
                                                {{"A", 0.656315f}, {"B", 0.343685f}}};
    const auto& actual_output_1 = fetches[1].Get<VectorMapStringToFloat>();
    ASSERT_EQ(actual_output_1.size(), size_t(3));
    for (size_t i = 0; i < 3; i++) {
      const auto& expected = expected_output_1[i];
      const auto& actual = actual_output_1[i];
      ASSERT_EQ(actual.size(), size_t(2));
      ASSERT_NEAR(expected.at("A"), actual.at("A"), 1e-6);
      ASSERT_NEAR(expected.at("B"), actual.at("B"), 1e-6);
    }
  };

  RunOrtModel(test_info);
}

#endif  // #if !defined(DISABLE_ML_OPS)
#endif  // #if !defined(ORT_MINIMAL_BUILD)

// test loading ORT format model with sparse initializers
TEST(OrtModelOnlyTests, LoadSparseInitializersOrtFormat) {
  const std::basic_string<ORTCHAR_T> ort_file = ORT_TSTR("testdata/ort_minimal_test_models/sparse_initializer_handling.onnx.ort");
  SessionOptions so;
  so.session_logid = "LoadOrtFormat";
  so.AddConfigEntry(kOrtSessionOptionsConfigLoadModelFormat, "ORT");
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(ort_file));
  ASSERT_STATUS_OK(session_object.Initialize());
}

OrtModelTestInfo GetTestInfoForLoadOrtFormatModel() {
  OrtModelTestInfo test_info;
  test_info.model_filename = ORT_TSTR("testdata/ort_github_issue_4031.onnx.ort");
  test_info.logid = "LoadOrtFormatModel";

  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, {123.f},
                       &ml_value);
  test_info.inputs.insert(std::make_pair("state_var_in", ml_value));

  // prepare outputs
  test_info.output_names = {"state_var_out"};
  test_info.output_verifier = [](const std::vector<OrtValue>& fetches) {
    const auto& output = fetches[0].Get<Tensor>();
    ASSERT_TRUE(output.Shape().Size() == 1);
    ASSERT_TRUE(output.Data<float>()[0] == 125.f);
  };

  return test_info;
}

// test that we can deserialize and run a previously saved ORT format model
TEST(OrtModelOnlyTests, LoadOrtFormatModel) {
  OrtModelTestInfo test_info = GetTestInfoForLoadOrtFormatModel();
  RunOrtModel(test_info);
}

// Load the model from a buffer instead of a file path
TEST(OrtModelOnlyTests, LoadOrtFormatModelFromBuffer) {
  OrtModelTestInfo test_info = GetTestInfoForLoadOrtFormatModel();
  test_info.run_use_buffer = true;
  RunOrtModel(test_info);
}

#if !defined(DISABLE_ML_OPS)
// test that we can deserialize and run a previously saved ORT format model
// for a model with sequence and map outputs
OrtModelTestInfo GetTestInfoForLoadOrtFormatModelMLOps() {
  OrtModelTestInfo test_info;
  test_info.model_filename = ORT_TSTR("testdata/sklearn_bin_voting_classifier_soft.ort");
  test_info.logid = "LoadOrtFormatModelMLOps";

  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {3, 2},
                       {0.f, 1.f, 1.f, 1.f, 2.f, 0.f}, &ml_value);
  test_info.inputs.insert(std::make_pair("input", ml_value));

  // prepare outputs
  test_info.output_names = {"output_label", "output_probability"};
  test_info.output_verifier = [](const std::vector<OrtValue>& fetches) {
    const auto& output_0 = fetches[0].Get<Tensor>();
    int64_t tensor_size = 3;
    ASSERT_EQ(tensor_size, output_0.Shape().Size());
    const auto& output_0_data = output_0.Data<std::string>();
    for (int64_t i = 0; i < tensor_size; i++)
      ASSERT_TRUE(output_0_data[i] == "A");

    VectorMapStringToFloat expected_output_1 = {{{"A", 0.572734f}, {"B", 0.427266f}},
                                                {{"A", 0.596016f}, {"B", 0.403984f}},
                                                {{"A", 0.656315f}, {"B", 0.343685f}}};
    const auto& actual_output_1 = fetches[1].Get<VectorMapStringToFloat>();
    ASSERT_EQ(actual_output_1.size(), size_t(3));
    for (size_t i = 0; i < 3; i++) {
      const auto& expected = expected_output_1[i];
      const auto& actual = actual_output_1[i];
      ASSERT_EQ(actual.size(), size_t(2));
      ASSERT_NEAR(expected.at("A"), actual.at("A"), 1e-6);
      ASSERT_NEAR(expected.at("B"), actual.at("B"), 1e-6);
    }
  };

  return test_info;
}

// test that we can deserialize and run a previously saved ORT format model
// for a model with sequence and map outputs
TEST(OrtModelOnlyTests, LoadOrtFormatModelMLOps) {
  OrtModelTestInfo test_info = GetTestInfoForLoadOrtFormatModelMLOps();
  RunOrtModel(test_info);
}

// Load the model from a buffer instead of a file path
TEST(OrtModelOnlyTests, LoadOrtFormatModelMLOpsFromBuffer) {
  OrtModelTestInfo test_info = GetTestInfoForLoadOrtFormatModelMLOps();
  test_info.run_use_buffer = true;
  RunOrtModel(test_info);
}

#endif  // !defined(DISABLE_ML_OPS)

}  // namespace test
}  // namespace onnxruntime

#endif  //  defined(ENABLE_ORT_FORMAT_LOAD)
