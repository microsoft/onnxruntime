// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/provider_options.h"
#include "core/framework/tensor_shape.h"
#include "core/common/float16.h"
#include "core/common/cpuid_info.h"

#include "test/util/include/test_utils.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/default_providers.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/inference_session.h"
#include "core/graph/model_saving_options.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

extern std::unique_ptr<Ort::Env> ort_env;

namespace {

using onnxruntime::MLFloat16;

// Returns true only on Intel CPUs.
//
// The OVIR EP context tests are gated on this because that path is currently
// validated only on Intel silicon. In particular, embed_mode = 0 dumps the
// compiled model to a separate .bin file and memory-maps it back on reload;
// this round-trip is unsupported on non-Intel CPUs (e.g. AMD), where it can
// crash. On those CPUs the OVIR EP context tests are skipped.
bool IsIntelCPU() {
  return onnxruntime::CPUIDInfo::GetCPUIDInfo().GetCPUVendor() == "Intel";
}

// Runs a mul_1-style model (X[3,2] -> Y[3,2], Y = X * {1,2,3,4,5,6}) with
// X = all 2.0f and validates that Y == {2,4,6,8,10,12}.
void RunAndValidate(Ort::Session& session) {
  const std::array<int64_t, 2> input_shape = {3, 2};
  std::vector<float> input_data(6, 2.0f);
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

  const std::array<const char*, 1> input_names = {"X"};
  const std::array<const char*, 1> output_names = {"Y"};
  std::vector<Ort::Value> output_tensors(1);

  session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
              output_names.data(), output_tensors.data(), 1);

  ASSERT_TRUE(output_tensors[0].IsTensor());
  ASSERT_EQ(output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount(), 6u);

  const float* out_data = output_tensors[0].GetTensorData<float>();
  EXPECT_THAT(std::vector<float>(out_data, out_data + 6),
              ::testing::ElementsAre(2.f, 4.f, 6.f, 8.f, 10.f, 12.f));
}

void AddFloat16Initializer(ONNX_NAMESPACE::GraphProto* graph,
                           const std::string& name,
                           std::span<const int64_t> shape,
                           std::span<const float> values) {
  auto* init = graph->add_initializer();
  init->set_name(name);
  init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  for (int64_t d : shape) {
    init->add_dims(d);
  }
  std::vector<uint16_t> bits;
  bits.reserve(values.size());
  for (float v : values) {
    bits.push_back(MLFloat16(v).val);
  }
  init->set_raw_data(bits.data(), bits.size() * sizeof(uint16_t));
}

void AddValueInfo(ONNX_NAMESPACE::ValueInfoProto* vi,
                  const std::string& name,
                  ONNX_NAMESPACE::TensorProto_DataType elem_type,
                  std::span<const int64_t> shape) {
  vi->set_name(name);
  auto* type = vi->mutable_type()->mutable_tensor_type();
  type->set_elem_type(elem_type);
  auto* vi_shape = type->mutable_shape();
  for (int64_t d : shape) {
    vi_shape->add_dim()->set_dim_value(d);
  }
}

ONNX_NAMESPACE::NodeProto* AddNode(ONNX_NAMESPACE::GraphProto* graph,
                                   const std::string& op_type,
                                   const std::string& name,
                                   const std::vector<std::string>& inputs,
                                   const std::string& output) {
  auto* n = graph->add_node();
  n->set_op_type(op_type);
  n->set_name(name);
  for (const auto& in : inputs) {
    n->add_input(in);
  }
  n->add_output(output);
  return n;
}

// Adds a RandomUniformLike sink so the graph is not wholly supported by OVEP;
// this forces partitioning, which is what runs the constant folding pass that
// populates the constant-output map exercised by tests.
void AddUnsupportedSink(ONNX_NAMESPACE::GraphProto* graph, const std::string& input) {
  auto* n = AddNode(graph, "RandomUniformLike", "unsupported_sink", {input}, "Z");
  auto* attr = n->add_attribute();
  attr->set_name("dtype");
  attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr->set_i(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
}

// Runs an OVEP constant-output model and returns the requested f16 output as
// floats. Uses "AUTO:CPU" (not plain "CPU"): plain "CPU" takes OVEP's unified-
// compile fast path and never reaches CreateOVModel where the folding lives.
// ORT-side optimizations are disabled so OVEP (not ORT) folds the constants,
// which is what routes them through the constant-output fill path under test.
// The runtime input's values never affect the constant outputs, so it is fed
// as an all-zero tensor of the given shape. Asserts the output is f16.
std::vector<float> RunConstantOutput(const std::string& model_data,
                                     const std::string& input_name,
                                     std::span<const int64_t> input_shape,
                                     const char* output_name,
                                     size_t& out_element_count) {
  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
  std::unordered_map<std::string, std::string> ov_options = {{"device_type", "AUTO:CPU"}};
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);
  Ort::Session session(*ort_env, model_data.data(), model_data.size(), session_options);

  int64_t input_count = 1;
  for (int64_t d : input_shape) {
    input_count *= d;
  }
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemTypeDefault);
  std::vector<uint16_t> input_bits(static_cast<size_t>(input_count), MLFloat16(0.0f).val);
  Ort::Value input_tensor = Ort::Value::CreateTensor(
      mem_info, input_bits.data(), input_bits.size() * sizeof(uint16_t),
      input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

  const std::array<const char*, 1> input_names = {input_name.c_str()};
  const std::array<const char*, 1> output_names = {output_name};
  std::vector<Ort::Value> output_tensors(1);
  session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
              output_names.data(), output_tensors.data(), 1);

  EXPECT_TRUE(output_tensors[0].IsTensor());
  auto type_shape = output_tensors[0].GetTensorTypeAndShapeInfo();
  EXPECT_EQ(type_shape.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  out_element_count = type_shape.GetElementCount();
  const MLFloat16* out_data = output_tensors[0].GetTensorData<MLFloat16>();
  std::vector<float> result;
  result.reserve(out_element_count);
  for (size_t i = 0; i < out_element_count; ++i) {
    result.push_back(out_data[i].ToFloat());
  }
  return result;
}

}  // namespace

class OVEPEPContextTests : public ::testing::Test {
};

namespace onnxruntime {
namespace test {

// Test if folder path given to ep_context_file_path throws an error
TEST_F(OVEPEPContextTests, OVEPEPContextFolderPath) {
  Ort::SessionOptions sessionOptions;
  std::unordered_map<std::string, std::string> ov_options;

  // The below line could fail the test in non NPU platforms.Commenting it out so that the device used for building OVEP will be used.
  // ov_options["device_type"] = "NPU";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("OVEP_Test_Model", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());

  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ep_context_file_path = "./ep_context_folder_path/";

  sessionOptions.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  sessionOptions.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_file_path.c_str());
  sessionOptions.AppendExecutionProvider_OpenVINO_V2(ov_options);

  try {
    Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), sessionOptions);
    FAIL();  // Should not get here!
  } catch (const Ort::Exception& excpt) {
    ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(excpt.what(), testing::HasSubstr("context_file_path should not point to a folder."));
  }
}

// Runs an existing OVIR-encapsulated EP context model: "mul_1_ep_ctx_ovir.onnx"
// wraps a single EPContext node whose "ep_cache_context" points to a sibling
// OpenVINO IR (".xml" + ".bin"), so OVEP imports it via read_model()/
// compile_model() instead of a pre-compiled blob.
//
// OVIR detection is filename-based (".onnx" -> ".xml"), so the model must be
// loaded from a path with the ".xml"/".bin" siblings next to it.
//
// CPU only.
class OVEPEPContextOVIRTests : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!IsIntelCPU()) {
      GTEST_SKIP() << "OVIR EP context is only validated on Intel CPUs; skipping on non-Intel silicon.";
    }
  }

  static constexpr const char* kDevice = "CPU";
  static constexpr const ORTCHAR_T* kOvirModelPath = ORT_TSTR("testdata/mul_1_ep_ctx_ovir.onnx");
};

TEST_F(OVEPEPContextOVIRTests, RunEpCtxOvirModel) {
  ASSERT_TRUE(std::filesystem::exists(kOvirModelPath))
      << "Missing OVIR EP context model. Expected testdata/mul_1_ep_ctx_ovir.onnx "
         "(with sibling .xml and .bin files).";

  // Set up session options targeting CPU.
  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

  // Load the OVIR EP context model from disk (path-based load is required for
  // OVIR encapsulation detection).
  Ort::Session session(*ort_env, kOvirModelPath, session_options);

  RunAndValidate(session);
}

// Negative / security test: an OVIR-encapsulated EP context model whose
// "ep_cache_context" attribute points outside the model directory via "../"
// traversal (e.g. "../../../etc/evil.xml") must be rejected at session-creation
// time rather than silently reading an arbitrary file off disk.

TEST_F(OVEPEPContextOVIRTests, RejectsEpCacheContextPathTraversal) {
  ASSERT_TRUE(std::filesystem::exists(kOvirModelPath))
      << "Missing OVIR EP context model. Expected testdata/mul_1_ep_ctx_ovir.onnx "
         "(with sibling .xml and .bin files).";

  // Load the known-good OVIR EP context model and rewrite its EPContext node so
  // that ep_cache_context escapes the model directory.
  ONNX_NAMESPACE::ModelProto model_proto;
  ASSERT_STATUS_OK(Model::Load(kOvirModelPath, model_proto));

  // Malicious relative path that escapes the model directory. The ".xml"
  // extension routes validation through the OVIR ".xml" branch in
  // EPCtxHandler::Initialize() (validated against the input model's directory),
  // and "evil.xml" matches the "evil.onnx" output stem below so the node is also
  // recognized as OVIR-encapsulated.
  const std::string malicious_xml_path = "../../../etc/evil.xml";

  bool patched = false;
  for (auto& node : *model_proto.mutable_graph()->mutable_node()) {
    if (node.op_type() != "EPContext") {
      continue;
    }
    for (auto& attr : *node.mutable_attribute()) {
      if (attr.name() == "embed_mode") {
        attr.set_i(0);  // force non-embed so the path (not an inline blob) is validated
      } else if (attr.name() == "ep_cache_context") {
        attr.set_s(malicious_xml_path);
        patched = true;
      }
    }
  }
  ASSERT_TRUE(patched) << "Test model did not contain an EPContext ep_cache_context attribute to patch.";

  // Write the tampered model to a dedicated subfolder. The malicious ".xml" is
  // intentionally never created on disk: validation must reject the path before
  // any attempt to read it.
  const std::filesystem::path out_dir = std::filesystem::path("testdata") / "ovir_epctx_path_traversal";
  std::filesystem::remove_all(out_dir);
  std::filesystem::create_directories(out_dir);
  const std::filesystem::path malicious_model = out_dir / "evil.onnx";
  {
    std::ofstream ofs(malicious_model, std::ios::binary);
    ASSERT_TRUE(ofs.is_open()) << "Failed to open " << malicious_model;
    ASSERT_TRUE(model_proto.SerializeToOstream(&ofs)) << "Failed to serialize tampered model.";
  }

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

  bool threw = false;
  std::string error_message;
  try {
    Ort::Session session(*ort_env, malicious_model.c_str(), session_options);
  } catch (const Ort::Exception& ex) {
    threw = true;
    error_message = ex.what();
  }

  std::filesystem::remove_all(out_dir);

  ASSERT_TRUE(threw)
      << "Session creation should have rejected the path-traversal ep_cache_context, but it succeeded.";
  EXPECT_THAT(error_message, ::testing::HasSubstr("escapes model directory"))
      << "Expected a path-escape rejection. Actual error: " << error_message;
}

// Generates an EP context model from the OVIR-encapsulated source model and
// then loads + runs the generated model, covering both EP context embed modes:
//   embed_mode = 1: the compiled context is serialized INLINE into the .onnx.
//   embed_mode = 0: the compiled context is dumped to a separate file and only
//                   its filename is stored in the .onnx EPContext node.
//
// embed_mode is only honored during generation (it is written into the
// EPContext node), so it must be exercised via a generate-then-run flow rather
// than by setting the option on a run-only session.
//
// CPU only. Parameter: embed_mode_enabled.
class OVEPOVIRModelsExportEPContextTests : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    if (!IsIntelCPU()) {
      GTEST_SKIP() << "OVIR EP context export is only validated on Intel CPUs; skipping on non-Intel silicon.";
    }
  }

  static constexpr const char* kDevice = "CPU";
  static constexpr const ORTCHAR_T* kOvirModelPath = ORT_TSTR("testdata/mul_1_ep_ctx_ovir.onnx");
};

TEST_P(OVEPOVIRModelsExportEPContextTests, ExportEpCtxFromOVIRModel) {
  const bool embed_mode = GetParam();

  ASSERT_TRUE(std::filesystem::exists(kOvirModelPath))
      << "Missing OVIR EP context model. Expected testdata/mul_1_ep_ctx_ovir.onnx "
         "(with sibling .xml and .bin files).";

  // Generate the EP context model into a dedicated subfolder so that the
  // separately-dumped blob (embed_mode = 0) doesn't collide with testdata.
  const std::filesystem::path out_dir =
      std::filesystem::path("testdata") / (std::string("ovir_epctx_export_embed_") + (embed_mode ? "on" : "off"));
  std::filesystem::remove_all(out_dir);
  std::filesystem::create_directories(out_dir);
  const std::filesystem::path epctx_model = out_dir / "mul_1_ovir_epctx_export.onnx";

  // --- Generate EP context model ---
  {
    Ort::SessionOptions session_options;
    session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, epctx_model.string().c_str());
    session_options.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, embed_mode ? "1" : "0");
    std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
    session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

    // Creating the session triggers EP context export to epctx_model.
    Ort::Session session(*ort_env, kOvirModelPath, session_options);
  }

  ASSERT_TRUE(std::filesystem::exists(epctx_model))
      << "EP context model was not generated at " << epctx_model;

  // --- Load + run the generated EP context model ---
  {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
    session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

    Ort::Session session(*ort_env, epctx_model.c_str(), session_options);

    RunAndValidate(session);
  }

  if (!embed_mode) {
    const std::filesystem::path external_initializers_dir = out_dir / "external_initializers";
    std::filesystem::create_directories(external_initializers_dir);

    {
      Ort::SessionOptions session_options;
      session_options.AddConfigEntry(kOrtSessionOptionsModelExternalInitializersFileFolderPath,
                                     external_initializers_dir.string().c_str());
      std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
      session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

      try {
        Ort::Session session(*ort_env, epctx_model.c_str(), session_options);
        FAIL() << "Session creation should fail when the EP context binary is resolved from the initializer folder.";
      } catch (const Ort::Exception& ex) {
        EXPECT_THAT(ex.what(), ::testing::HasSubstr("External data path does not exist"));
        EXPECT_THAT(ex.what(), ::testing::Not(::testing::HasSubstr("validate_status.IsOK()")));
        EXPECT_THAT(ex.what(), ::testing::HasSubstr("session.model_external_initializers_file_folder_path"));
        EXPECT_THAT(ex.what(), ::testing::HasSubstr("ep.context_file_path"));
      }
    }

    {
      Ort::SessionOptions session_options;
      session_options.AddConfigEntry(kOrtSessionOptionsModelExternalInitializersFileFolderPath,
                                     external_initializers_dir.string().c_str());
      session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, epctx_model.string().c_str());
      std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
      session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

      Ort::Session session(*ort_env, epctx_model.c_str(), session_options);
      RunAndValidate(session);
    }
  }

  std::filesystem::remove_all(out_dir);
}

INSTANTIATE_TEST_SUITE_P(
    OVEP_Tests,
    OVEPOVIRModelsExportEPContextTests,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<OVEPOVIRModelsExportEPContextTests::ParamType>& info) {
      return std::string("embed_") + (info.param ? "on" : "off");
    });

// Regression test for the float16 branch of FillOutputsWithConstantData
// (backend_utils.cc), which used to call FillOutputHelper<float> and so wrote
// 4-byte floats into a 2-byte-per-element f16 output tensor, corrupting the
// values and overrunning the buffer by 2x.
//
// That path (OVEP filling a constant graph output) only runs when all of:
//   - device is "AUTO:CPU": a plain "CPU" device takes OVEP's unified-compile
//     fast path and never reaches CreateOVModel, where the folding lives;
//   - the graph is not wholly supported, so partitioning runs the folding pass
//     (RandomUniformLike is the unsupported op that forces this);
//   - the constant branch C = Add(A, B) shares a cluster with a runtime input,
//     so it isn't dropped as an all-initializer cluster (via Y = Add(X, C));
//   - ORT optimizations are off, so OpenVINO (not ORT) folds Add(A, B).
// C is then emitted as an f16 Constant output filled by the code under test.
// Only C is asserted on; Y and Z exist only to shape the partition.
TEST(OVEPConstantOutputTests, FillsFloat16ConstantOutput) {
  const std::vector<int64_t> shape = {3, 2};
  const std::vector<float> a_f = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> b_f = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

  std::vector<float> expected;
  for (size_t i = 0; i < a_f.size(); ++i) {
    expected.push_back(a_f[i] + b_f[i]);
  }

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph = model.mutable_graph();
  graph->set_name("const_f16_add");

  AddFloat16Initializer(graph, "A", shape, a_f);
  AddFloat16Initializer(graph, "B", shape, b_f);

  // Runtime input; only exists so the constant branch shares a cluster with a
  // non-initializer input. Its values do not affect C.
  AddValueInfo(graph->add_input(), "X", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, shape);

  // Constant branch: OpenVINO folds Add(A, B) into the f16 Constant output C.
  AddNode(graph, "Add", "add_const", {"A", "B"}, "C");
  // Pulls C into a cluster that has a runtime input (X), so it isn't dropped.
  AddNode(graph, "Add", "add_runtime", {"X", "C"}, "Y");
  // Unsupported by OVEP -> forces partitioning so the folding pass runs.
  AddUnsupportedSink(graph, "X");

  AddValueInfo(graph->add_output(), "C", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, shape);
  AddValueInfo(graph->add_output(), "Y", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, shape);
  AddValueInfo(graph->add_output(), "Z", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape);

  std::string model_data;
  model.SerializeToString(&model_data);

  size_t count_c = 0;
  std::vector<float> actual = RunConstantOutput(model_data, "X", shape, "C", count_c);
  EXPECT_EQ(count_c, expected.size());
  EXPECT_THAT(actual, ::testing::ElementsAreArray(expected));
}

// Two same-shape constant outputs "D" and "D/x" whose names collide under
// '/'-truncation. Before the fix, "D/x"'s constant landed in "D"'s slot, so "D"
// returned Mul(A, B) instead of Add(A, B).
TEST(OVEPConstantOutputTests, RoutesConstantToCorrectOutputOnNameCollision) {
  const std::vector<int64_t> shape = {3};
  const std::vector<float> a = {1.0f, 2.0f, 3.0f};
  const std::vector<float> b = {10.0f, 20.0f, 30.0f};
  const std::vector<float> expected_d = {11.0f, 22.0f, 33.0f};   // Add(A, B)
  const std::vector<float> expected_dx = {10.0f, 40.0f, 90.0f};  // Mul(A, B)

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph = model.mutable_graph();
  graph->set_name("const_name_collision");

  AddFloat16Initializer(graph, "A", shape, a);
  AddFloat16Initializer(graph, "B", shape, b);
  AddValueInfo(graph->add_input(), "X", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, shape);

  // Two constants whose output names collide once truncated at the first '/'.
  AddNode(graph, "Add", "add_const", {"A", "B"}, "D");
  AddNode(graph, "Mul", "mul_const", {"A", "B"}, "D/x");
  // Pull both constants into a cluster that has a runtime input (X) so they are
  // not dropped as an all-initializer cluster.
  AddNode(graph, "Add", "add_runtime_d", {"X", "D"}, "XD");
  AddNode(graph, "Add", "add_runtime_dx", {"XD", "D/x"}, "Y");
  AddUnsupportedSink(graph, "X");

  AddValueInfo(graph->add_output(), "D", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, shape);
  AddValueInfo(graph->add_output(), "D/x", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, shape);
  AddValueInfo(graph->add_output(), "Y", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, shape);
  AddValueInfo(graph->add_output(), "Z", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape);

  std::string model_data;
  model.SerializeToString(&model_data);

  size_t count_d = 0;
  std::vector<float> actual_d = RunConstantOutput(model_data, "X", shape, "D", count_d);
  EXPECT_EQ(count_d, expected_d.size());
  // This is the assertion that fails before the fix: output "D" carries the
  // "D/x" (Mul) constant instead of its own (Add) constant.
  EXPECT_THAT(actual_d, ::testing::ElementsAreArray(expected_d));

  size_t count_dx = 0;
  std::vector<float> actual_dx = RunConstantOutput(model_data, "X", shape, "D/x", count_dx);
  EXPECT_EQ(count_dx, expected_dx.size());
  EXPECT_THAT(actual_dx, ::testing::ElementsAreArray(expected_dx));
}

// Same name collision but with DIFFERENT shapes ("D" is [1], "D/x" is [8]).
// Before the fix, the 8-element "D/x" constant was routed to the 1-element "D"
// slot; with a pre-bound fixed-size buffer that is a heap overrun, caught by
// ASan. With the fix, "D" keeps its own [1] value. The FillOutputHelper size
// check is the in-EP backstop for any residual mis-route, turning an overrun
// into a thrown error.
TEST(OVEPConstantOutputTests, KeepsConstantOutputShapeOnNameCollision) {
  const std::vector<int64_t> small_shape = {1};
  const std::vector<int64_t> large_shape = {8};
  const std::vector<float> a1 = {1.0f};
  const std::vector<float> b1 = {10.0f};
  const std::vector<float> expected_d = {11.0f};  // Add(A1, B1)

  const std::vector<float> a8 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> b8 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
  std::vector<float> expected_dx;  // Mul(A8, B8)
  for (size_t i = 0; i < a8.size(); ++i) {
    expected_dx.push_back(a8[i] * b8[i]);
  }

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph = model.mutable_graph();
  graph->set_name("const_name_collision_mismatched");

  AddFloat16Initializer(graph, "A1", small_shape, a1);
  AddFloat16Initializer(graph, "B1", small_shape, b1);
  AddFloat16Initializer(graph, "A8", large_shape, a8);
  AddFloat16Initializer(graph, "B8", large_shape, b8);
  AddValueInfo(graph->add_input(), "X8", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, large_shape);

  // Small [1] constant named "D"; large [8] constant named "D/x".
  AddNode(graph, "Add", "add_const", {"A1", "B1"}, "D");
  AddNode(graph, "Mul", "mul_const", {"A8", "B8"}, "D/x");
  // Keep both constants in a cluster that has the runtime input X8. "D" is
  // broadcast-added to X8 only to anchor it to the runtime cluster; its own
  // output value is unaffected.
  AddNode(graph, "Add", "add_runtime_dx", {"X8", "D/x"}, "Y8");
  AddNode(graph, "Add", "add_runtime_d", {"Y8", "D"}, "Y");
  AddUnsupportedSink(graph, "X8");

  AddValueInfo(graph->add_output(), "D", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, small_shape);
  AddValueInfo(graph->add_output(), "D/x", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, large_shape);
  AddValueInfo(graph->add_output(), "Y", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, large_shape);
  AddValueInfo(graph->add_output(), "Z", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, large_shape);

  std::string model_data;
  model.SerializeToString(&model_data);

  size_t count_d = 0;
  std::vector<float> actual_d = RunConstantOutput(model_data, "X8", large_shape, "D", count_d);
  // Before the fix, output "D" is overrun / reshaped by the [8] "D/x" constant.
  EXPECT_EQ(count_d, expected_d.size());
  EXPECT_THAT(actual_d, ::testing::ElementsAreArray(expected_d));

  size_t count_dx = 0;
  std::vector<float> actual_dx = RunConstantOutput(model_data, "X8", large_shape, "D/x", count_dx);
  EXPECT_EQ(count_dx, expected_dx.size());
  EXPECT_THAT(actual_dx, ::testing::ElementsAreArray(expected_dx));
}

}  // namespace test
}  // namespace onnxruntime
