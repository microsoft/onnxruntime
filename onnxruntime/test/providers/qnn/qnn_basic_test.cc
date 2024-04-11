// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <string>
#include <thread>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
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

// Conv node `Conv` is not supported: GetFileLength for conv_qdq_external_ini.bin failed:open file conv_qdq_external_ini.bin fail,
// errcode = 2 - The system cannot find the file specified.
TEST_F(QnnHTPBackendTests, TestConvWithExternalData) {
  Ort::SessionOptions so;
  onnxruntime::ProviderOptions options;
#if defined(_WIN32)
  options["backend_path"] = "QnnHtp.dll";
#else
  options["backend_path"] = "libQnnHtp.so";
#endif

  so.AppendExecutionProvider("QNN", options);

  Ort::Status status(OrtSessionOptionsAppendExecutionProvider_CPU(so, 1));

  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "conv_qdq_external_ini.onnx";

  Ort::Session session(*ort_env, ort_model_path, so);
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
                               std::string qnn_context_priority = "",
                               std::string soc_model = "",
                               std::string htp_arch = "",
                               std::string device_id = "") {
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

  if (!soc_model.empty()) {
    options["soc_model"] = std::move(soc_model);
  }

  if (!htp_arch.empty()) {
    options["htp_arch"] = std::move(htp_arch);
  }

  if (!device_id.empty()) {
    options["device_id"] = std::move(device_id);
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

struct ModelAndBuilder {
  ModelAndBuilder(Graph& graph) : builder(graph) {}
  std::string model_data;
  ModelTestBuilder builder;
};

// Creates a model in memory. Input feeds and output names can be accessed from result.builder.
static void CreateModelInMemory(std::unique_ptr<ModelAndBuilder>& result,
                                const GetTestModelFn& model_build_fn,
                                const std::string& model_name,
                                int opset_version = 18) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}, {kMSDomain, 1}};
  auto& logging_manager = DefaultLoggingManager();

  // Create float model and serialize it to a string.
  onnxruntime::Model model(model_name, false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  result = std::make_unique<ModelAndBuilder>(model.MainGraph());
  model_build_fn(result->builder);
  result->builder.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  model.ToProto().SerializeToString(&result->model_data);
}

// Runs a session and verifies the outputs. Can be run by individual threads.
static void RunSessionAndVerify(InferenceSession& session, const RunOptions& run_options, const NameMLValMap& feeds,
                                const std::vector<std::string>& output_names,
                                const std::vector<std::vector<int64_t>>& output_shapes,
                                const std::vector<std::vector<float>>& expected_values,
                                int loop_count = 10) {
  // Let it run for a while
  for (int it = 0; it < loop_count; ++it) {
    std::vector<OrtValue> fetches;
    auto status = session.Run(run_options, feeds, output_names, &fetches);
    ASSERT_TRUE(status.IsOK());

    for (size_t i = 0; i < fetches.size(); i++) {
      auto& tensor = fetches[i].Get<Tensor>();
      TensorShape expected_shape(output_shapes[i]);
      ASSERT_EQ(expected_shape, tensor.Shape());

      gsl::span<const float> actual = tensor.DataAsSpan<float>();
      gsl::span<const float> expected(expected_values[i].data(), expected_values[i].size());
      ASSERT_EQ(expected, actual);
    }
  }
}

// Returns a function that builds a float32 model that adds 3 tensors.
static GetTestModelFn F32BuildAdd3Tensors(const TestInputDef<float>& input0_def,
                                          const TestInputDef<float>& input1_def,
                                          const TestInputDef<float>& input2_def) {
  return [input0_def, input1_def, input2_def](ModelTestBuilder& builder) {
    NodeArg* input0 = MakeTestInput<float>(builder, input0_def);
    NodeArg* input1 = MakeTestInput<float>(builder, input1_def);
    NodeArg* input2 = MakeTestInput<float>(builder, input1_def);

    auto* add0_out = builder.MakeIntermediate();
    builder.AddNode("Add", {input0, input1}, {add0_out});

    auto* output = builder.MakeOutput();
    builder.AddNode("Add", {add0_out, input2}, {output});
  };
}

// Tests running a single session in multiple threads on the CPU backend.
TEST_F(QnnCPUBackendTests, MultithreadSessionRun) {
  std::unique_ptr<ModelAndBuilder> model;
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<int64_t> shape = {1, 3, 2};
  std::vector<std::vector<int64_t>> output_shapes = {shape};
  std::vector<std::vector<float>> output_values = {{3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f}};

  CreateModelInMemory(model,
                      F32BuildAdd3Tensors(TestInputDef<float>(shape, false, input_data),
                                          TestInputDef<float>(shape, false, input_data),
                                          TestInputDef<float>(shape, false, input_data)),
                      "add3.f32");

  SessionOptions session_opts;
  session_opts.session_logid = "logger0";

  RunOptions run_opts;
  run_opts.run_tag = session_opts.session_logid;

  InferenceSession session_obj{session_opts, GetEnvironment()};
  onnxruntime::ProviderOptions options;

#if defined(_WIN32)
  options["backend_path"] = "QnnCpu.dll";
#else
  options["backend_path"] = "libQnnCpu.so";
#endif

  auto qnn_ep = QnnExecutionProviderWithOptions(options, &session_opts);
  EXPECT_TRUE(session_obj.RegisterExecutionProvider(std::move(qnn_ep)).IsOK());

  auto status = session_obj.Load(model->model_data.data(), static_cast<int>(model->model_data.size()));
  ASSERT_TRUE(status.IsOK());
  status = session_obj.Initialize();
  ASSERT_TRUE(status.IsOK());

  std::vector<std::thread> threads;
  constexpr int num_threads = 5;
  constexpr int loop_count = 10;
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(RunSessionAndVerify, std::ref(session_obj), run_opts,
                                  model->builder.feeds_, model->builder.output_names_,
                                  output_shapes, output_values, loop_count));
  }

  for (auto& th : threads) {
    th.join();
  }
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Returns a function that builds a QDQ model that adds 3 tensors. Forces all scales and zero-points to be (1.0f, 0),
// so it is only accurate when using non-fractional positive inputs.
template <typename QuantType>
static GetTestModelFn QDQBuildAdd3Tensors(const TestInputDef<float>& input0_def,
                                          const TestInputDef<float>& input1_def,
                                          const TestInputDef<float>& input2_def) {
  return [input0_def, input1_def, input2_def](ModelTestBuilder& builder) {
    NodeArg* input0 = MakeTestInput<float>(builder, input0_def);
    NodeArg* input0_after_qdq = AddQDQNodePair<QuantType>(builder, input0, 1.0f, 0);
    NodeArg* input1 = MakeTestInput<float>(builder, input1_def);
    NodeArg* input1_after_qdq = AddQDQNodePair<QuantType>(builder, input1, 1.0f, 0);
    NodeArg* input2 = MakeTestInput<float>(builder, input1_def);
    NodeArg* input2_after_qdq = AddQDQNodePair<QuantType>(builder, input2, 1.0f, 0);

    auto* add0_out = builder.MakeIntermediate();
    builder.AddNode("Add", {input0_after_qdq, input1_after_qdq}, {add0_out});

    auto* add0_out_dq = AddQDQNodePair<QuantType>(builder, add0_out, 1.0f, 0);

    auto* add1_out = builder.MakeIntermediate();
    builder.AddNode("Add", {add0_out_dq, input2_after_qdq}, {add1_out});

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, add1_out, 1.0f, 0);
  };
}

// Tests running a single session in multiple threads on the HTP backend.
TEST_F(QnnHTPBackendTests, MultithreadSessionRun) {
  std::unique_ptr<ModelAndBuilder> model;
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<int64_t> shape = {1, 3, 2};
  std::vector<std::vector<int64_t>> output_shapes = {shape};
  std::vector<std::vector<float>> output_values = {{3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f}};

  CreateModelInMemory(model,
                      QDQBuildAdd3Tensors<uint8_t>(TestInputDef<float>(shape, false, input_data),
                                                   TestInputDef<float>(shape, false, input_data),
                                                   TestInputDef<float>(shape, false, input_data)),
                      "add3.qdq");

  SessionOptions session_opts;
  session_opts.session_logid = "logger0";

  RunOptions run_opts;
  run_opts.run_tag = session_opts.session_logid;

  InferenceSession session_obj{session_opts, GetEnvironment()};
  onnxruntime::ProviderOptions options;

#if defined(_WIN32)
  options["backend_path"] = "QnnHtp.dll";
#else
  options["backend_path"] = "libQnnHtp.so";
#endif

  auto qnn_ep = QnnExecutionProviderWithOptions(options, &session_opts);
  EXPECT_TRUE(session_obj.RegisterExecutionProvider(std::move(qnn_ep)).IsOK());

  auto status = session_obj.Load(model->model_data.data(), static_cast<int>(model->model_data.size()));
  ASSERT_TRUE(status.IsOK());
  status = session_obj.Initialize();
  ASSERT_TRUE(status.IsOK());

  std::vector<std::thread> threads;
  constexpr int num_threads = 5;
  constexpr int loop_count = 10;

  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(RunSessionAndVerify, std::ref(session_obj), run_opts,
                                  model->builder.feeds_, model->builder.output_names_,
                                  output_shapes, output_values, loop_count));
  }

  for (auto& th : threads) {
    th.join();
  }
}

// Tests running a single session in multiple threads on the HTP backend with run option to set power config
TEST_F(QnnHTPBackendTests, MultithreadHtpPowerCfgSessionRunOption) {
  std::unique_ptr<ModelAndBuilder> model;
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<int64_t> shape = {1, 3, 2};
  std::vector<std::vector<int64_t>> output_shapes = {shape};
  std::vector<std::vector<float>> output_values = {{3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f}};

  CreateModelInMemory(model,
                      QDQBuildAdd3Tensors<uint8_t>(TestInputDef<float>(shape, false, input_data),
                                                   TestInputDef<float>(shape, false, input_data),
                                                   TestInputDef<float>(shape, false, input_data)),
                      "add3.qdq");

  SessionOptions session_opts;
  session_opts.session_logid = "logger0";

  InferenceSession session_obj{session_opts, GetEnvironment()};
  onnxruntime::ProviderOptions options;

#if defined(_WIN32)
  options["backend_path"] = "QnnHtp.dll";
#else
  options["backend_path"] = "libQnnHtp.so";
#endif

  auto qnn_ep = QnnExecutionProviderWithOptions(options, &session_opts);
  EXPECT_TRUE(session_obj.RegisterExecutionProvider(std::move(qnn_ep)).IsOK());

  auto status = session_obj.Load(model->model_data.data(), static_cast<int>(model->model_data.size()));
  ASSERT_TRUE(status.IsOK());
  status = session_obj.Initialize();
  ASSERT_TRUE(status.IsOK());

  std::vector<std::thread> threads;
  constexpr int num_threads = 5;
  constexpr int loop_count = 10;

  std::vector<std::string> perf_modes{
      "burst", "balanced", "default", "high_performance", "high_power_saver",
      "low_balanced", "extreme_power_saver", "low_power_saver", "power_saver"};

  size_t post_i = perf_modes.size() - 1;
  ASSERT_TRUE(post_i > num_threads);
  for (int i = 0; i < num_threads; ++i, --post_i) {
    RunOptions run_opts;
    run_opts.run_tag = session_opts.session_logid;
    auto rt = run_opts.config_options.AddConfigEntry(kOrtRunOptionsConfigQnnPerfMode, perf_modes[i].c_str());
    ASSERT_TRUE(rt.IsOK());
    rt = run_opts.config_options.AddConfigEntry(kOrtRunOptionsConfigQnnPerfModePostRun, perf_modes[post_i].c_str());
    ASSERT_TRUE(rt.IsOK());

    threads.push_back(std::thread(RunSessionAndVerify, std::ref(session_obj), run_opts,
                                  model->builder.feeds_, model->builder.output_names_,
                                  output_shapes, output_values, loop_count));
  }

  for (auto& th : threads) {
    th.join();
  }
}

// Tests running a single session in multiple threads on the HTP backend with EP option to set default power config
TEST_F(QnnHTPBackendTests, MultithreadDefaultHtpPowerCfgFromEpOption) {
  std::unique_ptr<ModelAndBuilder> model;
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<int64_t> shape = {1, 3, 2};
  std::vector<std::vector<int64_t>> output_shapes = {shape};
  std::vector<std::vector<float>> output_values = {{3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f}};

  CreateModelInMemory(model,
                      QDQBuildAdd3Tensors<uint8_t>(TestInputDef<float>(shape, false, input_data),
                                                   TestInputDef<float>(shape, false, input_data),
                                                   TestInputDef<float>(shape, false, input_data)),
                      "add3.qdq");

  SessionOptions session_opts;
  session_opts.session_logid = "logger0";

  RunOptions run_opts;
  run_opts.run_tag = session_opts.session_logid;

  InferenceSession session_obj{session_opts, GetEnvironment()};
  onnxruntime::ProviderOptions options;

#if defined(_WIN32)
  options["backend_path"] = "QnnHtp.dll";
#else
  options["backend_path"] = "libQnnHtp.so";
#endif
  options["htp_performance_mode"] = "burst";

  auto qnn_ep = QnnExecutionProviderWithOptions(options, &session_opts);
  EXPECT_TRUE(session_obj.RegisterExecutionProvider(std::move(qnn_ep)).IsOK());

  auto status = session_obj.Load(model->model_data.data(), static_cast<int>(model->model_data.size()));
  ASSERT_TRUE(status.IsOK());
  status = session_obj.Initialize();
  ASSERT_TRUE(status.IsOK());

  std::vector<std::thread> threads;
  constexpr int num_threads = 5;
  constexpr int loop_count = 10;

  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(RunSessionAndVerify, std::ref(session_obj), run_opts,
                                  model->builder.feeds_, model->builder.output_names_,
                                  output_shapes, output_values, loop_count));
  }

  for (auto& th : threads) {
    th.join();
  }
}

// Tests running a single session in multiple threads on the HTP backend with
// EP option to set default power config + run option to set power config for each run
TEST_F(QnnHTPBackendTests, MultithreadHtpPowerCfgDefaultAndRunOption) {
  std::unique_ptr<ModelAndBuilder> model;
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<int64_t> shape = {1, 3, 2};
  std::vector<std::vector<int64_t>> output_shapes = {shape};
  std::vector<std::vector<float>> output_values = {{3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f}};

  CreateModelInMemory(model,
                      QDQBuildAdd3Tensors<uint8_t>(TestInputDef<float>(shape, false, input_data),
                                                   TestInputDef<float>(shape, false, input_data),
                                                   TestInputDef<float>(shape, false, input_data)),
                      "add3.qdq");

  SessionOptions session_opts;
  session_opts.session_logid = "logger0";

  InferenceSession session_obj{session_opts, GetEnvironment()};
  onnxruntime::ProviderOptions options;

#if defined(_WIN32)
  options["backend_path"] = "QnnHtp.dll";
#else
  options["backend_path"] = "libQnnHtp.so";
#endif
  options["htp_performance_mode"] = "burst";

  auto qnn_ep = QnnExecutionProviderWithOptions(options, &session_opts);
  EXPECT_TRUE(session_obj.RegisterExecutionProvider(std::move(qnn_ep)).IsOK());

  auto status = session_obj.Load(model->model_data.data(), static_cast<int>(model->model_data.size()));
  ASSERT_TRUE(status.IsOK());
  status = session_obj.Initialize();
  ASSERT_TRUE(status.IsOK());

  std::vector<std::thread> threads;
  constexpr int num_threads = 5;
  constexpr int loop_count = 10;

  std::vector<std::string> perf_modes{
      "burst", "balanced", "default", "high_performance", "high_power_saver",
      "low_balanced", "extreme_power_saver", "low_power_saver", "power_saver"};

  size_t post_i = perf_modes.size() - 1;
  ASSERT_TRUE(post_i > num_threads);
  for (int i = 0; i < num_threads; ++i, --post_i) {
    RunOptions run_opts;
    run_opts.run_tag = session_opts.session_logid;
    auto rt = run_opts.config_options.AddConfigEntry(kOrtRunOptionsConfigQnnPerfMode, perf_modes[i].c_str());
    ASSERT_TRUE(rt.IsOK());
    rt = run_opts.config_options.AddConfigEntry(kOrtRunOptionsConfigQnnPerfModePostRun, perf_modes[post_i].c_str());
    ASSERT_TRUE(rt.IsOK());

    threads.push_back(std::thread(RunSessionAndVerify, std::ref(session_obj), run_opts,
                                  model->builder.feeds_, model->builder.output_names_,
                                  output_shapes, output_values, loop_count));
  }

  for (auto& th : threads) {
    th.join();
  }
}

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

// Test that models run with various SoC model values
TEST_F(QnnHTPBackendTests, HTPSocModels) {
  constexpr std::array<const char*, 3> soc_models = { "",   // No explicit SoC model specified
                                                      "0",  // "Unknown"
#if defined(_M_ARM64)
                                                      "37" };  // SC8280X
#elif defined(__linux__)
                                                      "30" };  // SM8350
#else
                                                      "" };
#endif

  for (auto soc_model : soc_models) {
    RunNHWCResizeModel(ORT_MODEL_FOLDER "nhwc_resize_sizes_opset18.quant.onnx",
                       true,   // use_htp
                       false,  // enable_qnn_saver
                       "",     // htp_graph_finalization_opt_mode
                       "",     // qnn_context_priority
                       soc_model);
  }
}

// Test that models run with various HTP architecture values (and set device_id)
TEST_F(QnnHTPBackendTests, HTPArchValues) {
  constexpr std::array<const char*, 3> htp_archs = {"",     // No explicit arch specified
                                                    "0",    // "None"
                                                    "68"};  // v68
  for (auto htp_arch : htp_archs) {
    RunNHWCResizeModel(ORT_MODEL_FOLDER "nhwc_resize_sizes_opset18.quant.onnx",
                       true,      // use_htp
                       false,     // enable_qnn_saver
                       "",        // htp_graph_finalization_opt_mode
                       "",        // qnn_context_priority
                       "",        // soc_model
                       htp_arch,  // htp_arch
                       "0");      // device_id
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

// Test float32 model with FP16 precision
TEST_F(QnnHTPBackendTests, Float32ModelWithFP16PrecisionTest) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  provider_options["enable_htp_fp16_precision"] = "1";

  auto input_defs = {TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f),
                     TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f)};
  RunQnnModelTest(BuildOpTestCase<float>("Add", input_defs, {}, {}, kOnnxDomain),
                  provider_options,
                  13,
                  ExpectedEPNodeAssignment::All,
                  0.008f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace test
}  // namespace onnxruntime
