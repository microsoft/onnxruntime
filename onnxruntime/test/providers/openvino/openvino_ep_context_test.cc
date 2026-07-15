// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <filesystem>
#include <fstream>
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

  std::filesystem::remove_all(out_dir);
}

INSTANTIATE_TEST_SUITE_P(
    OVEP_Tests,
    OVEPOVIRModelsExportEPContextTests,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<OVEPOVIRModelsExportEPContextTests::ParamType>& info) {
      return std::string("embed_") + (info.param ? "on" : "off");
    });

}  // namespace test
}  // namespace onnxruntime
