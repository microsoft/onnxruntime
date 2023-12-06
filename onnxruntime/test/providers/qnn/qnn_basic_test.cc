// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <filesystem>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/providers/cpu/cpu_provider_factory.h"  // For OrtSessionOptionsAppendExecutionProvider_CPU
#include "core/session/inference_session.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

// in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

// test uses ONNX model so can't be run in a minimal build.
// TODO: When we need QNN in a minimal build we should add an ORT format version of the model
#if !defined(ORT_MINIMAL_BUILD)

// Tests that the QNN EP is registered when added via the public C++ API.
// Loads a simple ONNX model that adds floats.
TEST(QnnEP, TestAddEpUsingPublicApi) {
  {
    Ort::SessionOptions so;

    // Can only enforce that model runs on QNN in linux CI machines
    // because they support the CPU backend and emulate the HPT backend.
    // TODO: Remove #ifdef when Windows Arm64 machines support the CPU backend.
#if defined(__linux__)
    so.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Disable fallback to the CPU EP.
#endif

    onnxruntime::ProviderOptions options;

#if defined(_WIN32)
    options["backend_path"] = "QnnCpu.dll";
#else
    options["backend_path"] = "libQnnCpu.so";
#endif

    so.AppendExecutionProvider("QNN", options);

    const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "constant_floats.onnx";
    Ort::Session session(*ort_env, ort_model_path, so);

    // Access the underlying InferenceSession.
    const OrtSession* ort_session = session;
    const InferenceSession* s = reinterpret_cast<const InferenceSession*>(ort_session);

    bool have_qnn_ep = false;

    for (const auto& provider : s->GetRegisteredProviderTypes()) {
      if (provider == kQnnExecutionProvider) {
        have_qnn_ep = true;
        break;
      }
    }

    ASSERT_TRUE(have_qnn_ep) << "QNN EP was not found in registered providers for session.";
  }
}

// Tests the `session.disable_cpu_ep_fallback` configuration option when the backend cannot be loaded.
// When the option is enabled, session creation throws an exception because the backend cannot be found.
TEST(QnnEP, TestDisableCPUFallback_BackendNotFound) {
  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Disable fallback to the CPU EP.

    onnxruntime::ProviderOptions options;
#if defined(_WIN32)
    options["backend_path"] = "DoesNotExist.dll";  // Invalid backend path!
#else
    options["backend_path"] = "libDoesNotExist.so";  // Invalid backend path!
#endif

    so.AppendExecutionProvider("QNN", options);

    const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "constant_floats.onnx";

    try {
      Ort::Session session(*ort_env, ort_model_path, so);
      FAIL();  // Should not get here!
    } catch (const Ort::Exception& excpt) {
      ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_FAIL);
      ASSERT_THAT(excpt.what(), testing::HasSubstr("This session contains graph nodes that are assigned to the default "
                                                   "CPU EP, but fallback to CPU EP has been explicitly disabled by "
                                                   "the user."));
    }
  }
}

// Tests the `session.disable_cpu_ep_fallback` configuration option when the entire model cannot be assigned to QNN EP.
// When the option is enabled, Session creation should throw an exception.
TEST(QnnEP, TestDisableCPUFallback_ModelNotFullySupported) {
  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Disable fallback to the CPU EP.

    onnxruntime::ProviderOptions options;
#if defined(_WIN32)
    options["backend_path"] = "QnnCpu.dll";
#else
    options["backend_path"] = "libQnnCpu.so";
#endif

    so.AppendExecutionProvider("QNN", options);

    // QNN EP doesn't support MatMulInteger.
    const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "qnn_ep_partial_support.onnx";

    try {
      Ort::Session session(*ort_env, ort_model_path, so);
      FAIL();  // Should not get here!
    } catch (const Ort::Exception& excpt) {
      ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_FAIL);
      ASSERT_THAT(excpt.what(), testing::HasSubstr("This session contains graph nodes that are assigned to the default "
                                                   "CPU EP, but fallback to CPU EP has been explicitly disabled by "
                                                   "the user."));
    }
  }
}

// Tests invalid use of the `session.disable_cpu_ep_fallback` configuration option.
// It is invalid to set the option and explicitly add the CPU EP to the session.
TEST(QnnEP, TestDisableCPUFallback_ConflictingConfig) {
  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Disable fallback to the CPU EP.

    onnxruntime::ProviderOptions options;
#if defined(_WIN32)
    options["backend_path"] = "QnnCpu.dll";
#else
    options["backend_path"] = "libQnnCpu.so";
#endif

    so.AppendExecutionProvider("QNN", options);

    // Invalid! Adds CPU EP to session, but also disables CPU fallback.
    Ort::Status status(OrtSessionOptionsAppendExecutionProvider_CPU(so, 1));

    const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "constant_floats.onnx";

    try {
      Ort::Session session(*ort_env, ort_model_path, so);
      FAIL();  // Should not get here!
    } catch (const Ort::Exception& excpt) {
      ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
      ASSERT_THAT(excpt.what(), testing::HasSubstr("Conflicting session configuration: explicitly added the CPU EP to the "
                                                   "session, but also disabled fallback to the CPU EP via session "
                                                   "configuration options."));
    }
  }
}

// Helper function that runs an ONNX model with a NHWC Resize operator to test that
// type/shape inference succeeds during layout transformation.
// Refer to onnxruntime/core/graph/contrib_ops/nhwc_inference_context.h.
//
// The models passed to this function are subgraphs extracted from a larger model that exhibited
// shape inferencing issues on QNN. Thus, the models are expected to have a specific input/output
// types and shapes.
static void RunNHWCResizeModel(const ORTCHAR_T* ort_model_path, bool use_htp, bool enable_qnn_saver = false,
                               std::string htp_graph_finalization_opt_mode = "",
                               std::string qnn_context_priority = "") {
  Ort::SessionOptions so;

  // Ensure all type/shape inference warnings result in errors!
  so.AddConfigEntry(kOrtSessionOptionsConfigStrictShapeTypeInference, "1");
  so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

  onnxruntime::ProviderOptions options;

#if defined(_WIN32)
  options["backend_path"] = use_htp ? "QnnHtp.dll" : "QnnCpu.dll";
  if (enable_qnn_saver) {
    options["qnn_saver_path"] = "QnnSaver.dll";
  }
#else
  options["backend_path"] = use_htp ? "libQnnHtp.so" : "libQnnCpu.so";
  if (enable_qnn_saver) {
    options["qnn_saver_path"] = "libQnnSaver.so";
  }
#endif

  if (!htp_graph_finalization_opt_mode.empty()) {
    options["htp_graph_finalization_optimization_mode"] = std::move(htp_graph_finalization_opt_mode);
  }

  if (!qnn_context_priority.empty()) {
    options["qnn_context_priority"] = std::move(qnn_context_priority);
  }

  so.AppendExecutionProvider("QNN", options);

  Ort::Session session(*ort_env, ort_model_path, so);

  // Input can be all zeros since we're testing for correct shape inference.
  std::array<float, 1 * 3 * 4 * 5> input0_data = {};
  std::array<float, 1 * 3 * 4 * 5> input1_data = {};
  std::array<float, 1 * 3 * 4 * 5> input2_data = {};

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> ort_input_names;

  // Add input0
  std::array<int64_t, 4> inputs_shape{1, 3, 4, 5};
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, input0_data.data(), input0_data.size(), inputs_shape.data(), inputs_shape.size()));
  ort_input_names.push_back("input0");

  // Add input1
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, input1_data.data(), input1_data.size(), inputs_shape.data(), inputs_shape.size()));
  ort_input_names.push_back("input1");

  // Add input2
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, input2_data.data(), input2_data.size(), inputs_shape.data(), inputs_shape.size()));
  ort_input_names.push_back("input2");

  // Run session and get outputs
  std::array<const char*, 2> output_names{"output0", "output1"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

  // Check output shape.
  Ort::Value& ort_output = ort_outputs[1];
  auto typeshape = ort_output.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> output_shape = typeshape.GetShape();

  EXPECT_THAT(output_shape, ::testing::ElementsAre(1, 6, 7, 10));
}

// Test shape inference of NHWC Resize operator (opset 11) that uses
// the scales input. Use the QNN CPU backend.
TEST_F(QnnCPUBackendTests, TestNHWCResizeShapeInference_scales_opset11) {
  RunNHWCResizeModel(ORT_MODEL_FOLDER "nhwc_resize_scales_opset11.onnx", false);
}

// Test shape inference of NHWC Resize operator (opset 18) that uses
// the scales input. Use the QNN CPU backend.
TEST_F(QnnCPUBackendTests, TestNHWCResizeShapeInference_scales_opset18) {
  RunNHWCResizeModel(ORT_MODEL_FOLDER "nhwc_resize_scales_opset18.onnx", false);
}

// Test shape inference of NHWC Resize operator (opset 11) that uses
// the sizes input. Use the QNN CPU backend.
TEST_F(QnnCPUBackendTests, TestNHWCResizeShapeInference_sizes_opset11) {
  RunNHWCResizeModel(ORT_MODEL_FOLDER "nhwc_resize_sizes_opset11.onnx", false);
}

// Test shape inference of NHWC Resize operator (opset 18) that uses
// the sizes input. Use the QNN CPU backend.
TEST_F(QnnCPUBackendTests, TestNHWCResizeShapeInference_sizes_opset18) {
  RunNHWCResizeModel(ORT_MODEL_FOLDER "nhwc_resize_sizes_opset18.onnx", false);
}

// Test that QNN Saver generates the expected files for a model meant to run on the QNN CPU backend.
TEST_F(QnnCPUBackendTests, QnnSaver_OutputFiles) {
  const std::filesystem::path qnn_saver_output_dir = "saver_output";

  // Remove pre-existing QNN Saver output files. Note that fs::remove_all() can handle non-existing paths.
  std::filesystem::remove_all(qnn_saver_output_dir);
  ASSERT_FALSE(std::filesystem::exists(qnn_saver_output_dir));

  RunNHWCResizeModel(ORT_MODEL_FOLDER "nhwc_resize_sizes_opset18.onnx",
                     false,  // use_htp
                     true);  // enable_qnn_saver

  // Check that QNN Saver output files exist.
  EXPECT_TRUE(std::filesystem::exists(qnn_saver_output_dir / "saver_output.c"));
  EXPECT_TRUE(std::filesystem::exists(qnn_saver_output_dir / "params.bin"));
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Test shape inference of QDQ NHWC Resize operator (opset 18) that uses
// the sizes input. Use the QNN HTP backend.
TEST_F(QnnHTPBackendTests, TestNHWCResizeShapeInference_qdq_sizes_opset18) {
  RunNHWCResizeModel(ORT_MODEL_FOLDER "nhwc_resize_sizes_opset18.quant.onnx", true);
}

// Test that QNN Saver generates the expected files for a model meant to run on the QNN HTP backend.
TEST_F(QnnHTPBackendTests, QnnSaver_OutputFiles) {
  const std::filesystem::path qnn_saver_output_dir = "saver_output";

  // Remove pre-existing QNN Saver output files. Note that fs::remove_all() can handle non-existing paths.
  std::filesystem::remove_all(qnn_saver_output_dir);
  ASSERT_FALSE(std::filesystem::exists(qnn_saver_output_dir));

  RunNHWCResizeModel(ORT_MODEL_FOLDER "nhwc_resize_sizes_opset18.onnx",
                     true,   // use_htp
                     true);  // enable_qnn_saver

  // Check that QNN Saver output files exist.
  EXPECT_TRUE(std::filesystem::exists(qnn_saver_output_dir / "saver_output.c"));
  EXPECT_TRUE(std::filesystem::exists(qnn_saver_output_dir / "params.bin"));
}

// Test that models run with various HTP graph finalization optimization modes.
TEST_F(QnnHTPBackendTests, HTPGraphFinalizationOptimizationModes) {
  constexpr std::array<const char*, 5> graph_opt_modes = {"",    // No explicit mode specified
                                                          "0",   // Explicit default mode
                                                          "1",   // Mode 1
                                                          "2",   // Mode 2
                                                          "3"};  // Mode 3
  for (auto mode : graph_opt_modes) {
    RunNHWCResizeModel(ORT_MODEL_FOLDER "nhwc_resize_sizes_opset18.quant.onnx",
                       true,   // use_htp
                       false,  // enable_qnn_saver
                       mode);  // htp_graph_finalization_opt_mode
  }
}

// Test that models run with high QNN context priority.
TEST_F(QnnHTPBackendTests, QnnContextPriorityHigh) {
  RunNHWCResizeModel(ORT_MODEL_FOLDER "nhwc_resize_sizes_opset18.quant.onnx",
                     true,     // use_htp
                     false,    // enable_qnn_saver
                     "",       // htp_graph_finalization_opt_mode
                     "high");  // qnn_context_priority
}

// Create a model with Case + Add (quantized)
// cast_input -> Cast -> Q -> DQ \
//                                Add -> Q -> DQ -> output
//             input2 -> Q -> DQ /
static GetTestModelFn BuildCastAddTestCase() {
  return [](ModelTestBuilder& builder) {
    // Creat Cast node int32 -> float32
    NodeArg* cast_input = MakeTestInput(builder, TestInputDef<int32_t>({2, 3}, false, {0, 1, 0, 1, 0, 1}));

    auto* cast_output = builder.MakeIntermediate();
    Node& cast_node = builder.AddNode("Cast", {cast_input}, {cast_output});
    cast_node.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));

    // Create Add node
    std::vector<float> data = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    gsl::span<float> data_range = gsl::make_span(data);
    QuantParams<uint8_t> q_parameter = GetDataQuantParams<uint8_t>(data_range);
    auto* add_input1_qdq = AddQDQNodePair<uint8_t>(builder, cast_output, q_parameter.scale, q_parameter.zero_point);

    NodeArg* add_input2 = MakeTestInput(builder, TestInputDef<float>({2, 3}, false, data));
    auto* add_input2_qdq = AddQDQNodePair<uint8_t>(builder, add_input2, q_parameter.scale, q_parameter.zero_point);

    auto* add_output = builder.MakeIntermediate();

    builder.AddNode("Add", {add_input1_qdq, add_input2_qdq}, {add_output});

    // add_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, add_output, q_parameter.scale, q_parameter.zero_point);
  };
}

// Test that models with 2 inputs which has different data type can still generate the context binary
TEST_F(QnnHTPBackendTests, QnnContextBinaryGeneration2InputTypes) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  provider_options["qnn_context_cache_enable"] = "1";
  const std::string context_binary_file = "./qnn_context_binary_int32_fp32_inputs_test.onnx";
  provider_options["qnn_context_cache_path"] = context_binary_file;

  RunQnnModelTest(BuildCastAddTestCase(),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All,
                  1e-5f,
                  logging::Severity::kERROR,
                  false);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));
}

// A repro of QC case 06838696, accuracy issue for Cast + Op (quantized)
// the value pair(1, 0.00392156886) at index #1 don't match,
// which is -0.996078 from 1
TEST_F(QnnHTPBackendTests, DISABLED_CastAddHTPAccuracyTest) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildCastAddTestCase(),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace test
}  // namespace onnxruntime
