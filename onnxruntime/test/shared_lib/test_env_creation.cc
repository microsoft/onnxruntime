// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/graph/constants.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "test/util/include/api_asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;
extern "C" void ortenv_setup();
extern "C" void ortenv_teardown();

TEST(EnvCreation, CreateEnvWithOptions) {
  const OrtApi& ort_api = Ort::GetApi();

  // Basic error checking when user passes an invalid version for OrtEnvCreationOptions
  {
    OrtEnv* test_env = nullptr;
    OrtEnvCreationOptions options{};
    options.version = 0;  // Invalid!
    options.logging_severity_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
    options.log_id = "test logger";

    Ort::Status status{ort_api.CreateEnvWithOptions(&options, &test_env)};

    ASSERT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(status.GetErrorMessage(), testing::HasSubstr("version set equal to ORT_API_VERSION"));
  }

  // Basic error checking when user passes an invalid log identifier to the API function
  {
    OrtEnv* test_env = nullptr;
    OrtEnvCreationOptions options{};
    options.version = ORT_API_VERSION;
    options.logging_severity_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
    options.log_id = nullptr;  // Invalid!

    Ort::Status status{ort_api.CreateEnvWithOptions(&options, &test_env)};

    ASSERT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(status.GetErrorMessage(), testing::HasSubstr("valid (non-null) log identifier string"));
  }

  // Basic error checking when user passes an invalid logging severity level
  {
    OrtEnv* test_env = nullptr;
    OrtEnvCreationOptions options{};
    options.version = ORT_API_VERSION;
    options.logging_severity_level = 100;  // Invalid!
    options.log_id = "EnvCreation.CreateEnvWithOptions";

    Ort::Status status{ort_api.CreateEnvWithOptions(&options, &test_env)};

    ASSERT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(status.GetErrorMessage(), testing::HasSubstr("valid logging severity level value from "
                                                             "the OrtLoggingLevel enumeration"));
  }

  // Create an OrtEnv with configuration entries. Use the CXX API.

  ortenv_teardown();  // Release current OrtEnv as we need to recreate it.

  auto run_test = [&]() -> void {
    // Create OrtEnv with some dummy config entry.
    Ort::KeyValuePairs env_configs;
    env_configs.Add("some_key", "some_val");

    OrtEnvCreationOptions options{};
    options.version = ORT_API_VERSION;
    options.logging_severity_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE;
    options.log_id = "EnvCreation.CreateEnvWithOptions_2";
    options.config_entries = env_configs.GetConst();

    Ort::Env tmp_env(&options);

    // Use EP API to retrieve environment configs and check contents
    Ort::KeyValuePairs env_configs_2 = Ort::GetEnvConfigEntries();

    auto configs_expected = env_configs.GetKeyValuePairs();
    auto configs_actual = env_configs_2.GetKeyValuePairs();
    ASSERT_EQ(configs_actual, configs_expected);
  };

  EXPECT_NO_FATAL_FAILURE(run_test());
  ortenv_setup();  // Restore OrtEnv
}

#ifdef ORT_ENABLE_SESSION_THREADPOOL_CALLBACKS
// End-to-end test: SetPerSessionThreadPoolCallbacks -> session creation -> inference -> callbacks invoked.
TEST(EnvCreation, SetPerSessionThreadPoolCallbacks) {
  struct CallbackState {
    std::atomic<int> enqueue_count{0};
    std::atomic<int> start_count{0};
    std::atomic<int> stop_count{0};
    std::atomic<int> abandon_count{0};
  };

  auto on_enqueue = [](void* ctx) noexcept -> void* {
    static_cast<CallbackState*>(ctx)->enqueue_count++;
    return nullptr;
  };
  auto on_start = [](void* ctx, void*) noexcept {
    static_cast<CallbackState*>(ctx)->start_count++;
  };
  auto on_stop = [](void* ctx, void*) noexcept {
    static_cast<CallbackState*>(ctx)->stop_count++;
  };
  auto on_abandon = [](void* ctx, void*) noexcept {
    static_cast<CallbackState*>(ctx)->abandon_count++;
  };

  ortenv_teardown();

  auto run_test = [&]() {
    CallbackState state;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ThreadPoolCallbacksTest");
    OrtThreadPoolCallbacksConfig cb_config = {0};
    cb_config.version = ORT_API_VERSION;
    cb_config.on_enqueue = on_enqueue;
    cb_config.on_start_work = on_start;
    cb_config.on_stop_work = on_stop;
    cb_config.on_abandon = on_abandon;
    cb_config.user_context = &state;
    env.SetPerSessionThreadPoolCallbacks(cb_config);

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);

    // Build a Mul model with dynamic input shapes using ModelBuilder so we can
    // pass a tensor large enough to trigger thread pool parallelism.
    // Y = X * X  (element-wise)
    Ort::Graph graph;

    std::vector<Ort::ValueInfo> graph_inputs;
    std::vector<Ort::ValueInfo> graph_outputs;

    std::vector<int64_t> dims({-1});  // dynamic dimension
    Ort::TensorTypeAndShapeInfo tensor_info(
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dims);
    auto type_info = Ort::TypeInfo::CreateTensorInfo(tensor_info.GetConst());

    graph_inputs.emplace_back("X", type_info.GetConst());
    graph_outputs.emplace_back("Y", type_info.GetConst());
    graph.SetInputs(graph_inputs);
    graph.SetOutputs(graph_outputs);

    Ort::Node node("Mul", onnxruntime::kOnnxDomain, "mul_node", {"X", "X"}, {"Y"});
    graph.AddNode(node);

    std::vector<Ort::Model::DomainOpsetPair> opsets{{onnxruntime::kOnnxDomain, 18}};
    Ort::Model model(opsets);
    model.AddGraph(graph);

    Ort::Session session(env, model, session_options);

    // Use a large input (1M elements) to ensure the cost model dispatches
    // work to pool threads, which triggers callbacks.
    constexpr int64_t num_elements = 1024 * 1024;
    std::vector<int64_t> shape = {num_elements};
    std::vector<float> input_data(num_elements, 2.0f);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                        shape.data(), shape.size());

    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};
    auto outputs = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);

    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(outputs[0].IsTensor());

    // With 1M elements and 4 threads, the thread pool must have dispatched work.
    EXPECT_GT(state.enqueue_count.load(), 0) << "on_enqueue should have been called";
    EXPECT_GT(state.start_count.load(), 0) << "on_start should have been called";
    EXPECT_GT(state.stop_count.load(), 0) << "on_stop should have been called";
    // start and stop must be balanced
    EXPECT_EQ(state.start_count.load(), state.stop_count.load());
    // every enqueued item must either start+stop or be abandoned
    EXPECT_EQ(state.enqueue_count.load(), state.start_count.load() + state.abandon_count.load());
  };

  EXPECT_NO_FATAL_FAILURE(run_test());
  ortenv_setup();
}
#endif  // ORT_ENABLE_SESSION_THREADPOOL_CALLBACKS
