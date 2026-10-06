// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <barrier>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/platform/env.h"
#include "core/providers/webgpu/program_manager.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/inference_session.h"

#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/temp_dir.h"

namespace onnxruntime {
namespace test {

TEST(WebGpuConcurrentContextTestStandalone, ShaderDumpWritesRemainComplete) {
  TemporaryDirectory temp_dir{ORT_TSTR("webgpu_shader_dump_test")};
  const auto dump_path = std::filesystem::path(temp_dir.Path()) / ORT_TSTR("shaders.txt");
  constexpr int kThreads = 8;
  constexpr int kRecordsPerThread = 100;
  constexpr size_t kPayloadSize = 4096;

  {
    auto dump_shader = webgpu::detail::CreateShaderDumpFunction(ToUTF8String(dump_path.native()));
    std::barrier start{kThreads};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int thread_id = 0; thread_id < kThreads; ++thread_id) {
      threads.emplace_back([&, thread_id] {
        start.arrive_and_wait();
        for (int record_id = 0; record_id < kRecordsPerThread; ++record_id) {
          const std::string marker = "record-" + std::to_string(thread_id) + "-" + std::to_string(record_id);
          dump_shader(marker + "-start\n" + std::string(kPayloadSize, static_cast<char>('a' + thread_id)) +
                      "\n" + marker + "-end");
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }

  std::ifstream dump_file(dump_path);
  ASSERT_TRUE(dump_file.is_open());
  const std::string contents{std::istreambuf_iterator<char>{dump_file}, std::istreambuf_iterator<char>{}};
  for (int thread_id = 0; thread_id < kThreads; ++thread_id) {
    for (int record_id = 0; record_id < kRecordsPerThread; ++record_id) {
      const std::string marker = "record-" + std::to_string(thread_id) + "-" + std::to_string(record_id);
      const std::string record = marker + "-start\n" +
                                 std::string(kPayloadSize, static_cast<char>('a' + thread_id)) +
                                 "\n" + marker + "-end\n";
      const size_t position = contents.find(record);
      ASSERT_NE(position, std::string::npos) << "Missing or corrupted shader dump record " << marker;
      EXPECT_EQ(contents.find(record, position + 1), std::string::npos)
          << "Duplicate shader dump record " << marker;
    }
  }
}

// DIAGNOSTIC (disabled by default): keeps N sessions over an identical model alive and runs them
// round-robin on one thread, then holds so an external sampler can read steady-state GPU memory.
// This is the shape that distinguishes a context-wide buffer pool (steady state ~= one session's
// high-water mark) from per-session pools (~= N times that).
//
//   onnxruntime_provider_test.exe --gtest_also_run_disabled_tests \
//     --gtest_filter=WebGpuPoolMemory.DISABLED_MultiSessionSameShape
//
// Tunable via environment variables: ORT_DIAG_SESSIONS, ORT_DIAG_MB, ORT_DIAG_ITERS,
// ORT_DIAG_HOLD_MS, ORT_DIAG_VARY (1 = give each session a different tensor size),
// ORT_DIAG_CACHE (storage buffer cache mode; defaults to "bucket", the product default -
// note DefaultWebGpuExecutionProvider() disables the storage cache, which would hide the
// very behavior this test measures).
TEST(WebGpuPoolMemory, DISABLED_MultiSessionSameShape) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  auto env_int = [](const char* name, int fallback) {
    const std::string value = Env::Default().GetEnvironmentVar(name);
    return value.empty() ? fallback : std::stoi(value);
  };

  const int num_sessions = env_int("ORT_DIAG_SESSIONS", 4);
  const int mb = env_int("ORT_DIAG_MB", 64);
  const int iters = env_int("ORT_DIAG_ITERS", 10);
  const int hold_ms = env_int("ORT_DIAG_HOLD_MS", 5000);
  const bool vary = env_int("ORT_DIAG_VARY", 0) != 0;
  std::string cache_mode = Env::Default().GetEnvironmentVar("ORT_DIAG_CACHE");
  if (cache_mode.empty()) {
    cache_mode = webgpu::options::kBufferCacheMode_Bucket;
  }
  constexpr int kChainLen = 8;

  auto make_ep = [&cache_mode]() {
    ConfigOptions config_options;
    ORT_THROW_IF_ERROR(config_options.AddConfigEntry(webgpu::options::kStorageBufferCacheMode,
                                                     cache_mode.c_str()));
    return WebGpuExecutionProviderWithOptions(config_options);
  };
  std::cout << "DIAG_POOL storage_cache=" << cache_mode << "\n";

  std::vector<std::unique_ptr<InferenceSession>> sessions;
  std::vector<NameMLValMap> feeds;
  std::vector<std::string> output_names{"Y"};

  // Scalar weights keep GPU memory dominated by intermediate tensors rather than initializers.
  auto build_scalar_weight_chain = [](int chain_len, int64_t num_elements, std::string& bytes) {
    const std::unordered_map<std::string, int> domain_to_version{{"", 13}};
    Model model("webgpu_pool_memory", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
                std::vector<ONNX_NAMESPACE::FunctionProto>(),
                DefaultLoggingManager().DefaultLogger());
    Graph& graph = model.MainGraph();

    ONNX_NAMESPACE::TypeProto float_1d;
    float_1d.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    float_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(num_elements);

    ONNX_NAMESPACE::TypeProto float_scalar;
    float_scalar.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    float_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    NodeArg* prev = &graph.GetOrCreateNodeArg("X", &float_1d);
    for (int i = 0; i < chain_len; ++i) {
      const std::string w_name = "w" + std::to_string(i);
      ONNX_NAMESPACE::TensorProto w_tensor;
      w_tensor.set_name(w_name);
      w_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      w_tensor.add_dims(1);
      const float scalar = 1.0f + 0.01f * i;
      w_tensor.set_raw_data(&scalar, sizeof(float));
      graph.AddInitializedTensor(w_tensor);

      NodeArg* w_arg = &graph.GetOrCreateNodeArg(w_name, &float_scalar);
      const std::string out_name = (i == chain_len - 1) ? "Y" : ("h" + std::to_string(i));
      NodeArg* out_arg = &graph.GetOrCreateNodeArg(out_name, &float_1d);
      std::vector<NodeArg*> inputs{prev, w_arg};
      std::vector<NodeArg*> outputs{out_arg};
      graph.AddNode("op" + std::to_string(i), (i % 2 == 0) ? "Mul" : "Add", "", inputs, outputs);
      prev = out_arg;
    }

    graph.SetOutputs(std::vector<const NodeArg*>{prev});
    ASSERT_STATUS_OK(graph.Resolve());
    ASSERT_TRUE(model.ToProto().SerializeToString(&bytes));
  };

  for (int s = 0; s < num_sessions; ++s) {
    const int64_t num_elements =
        static_cast<int64_t>(mb) * 1024 * 1024 / 4 + (vary ? s * 4 * 1024 * 1024 : 0);

    std::string model_bytes;
    ASSERT_NO_FATAL_FAILURE(build_scalar_weight_chain(kChainLen, num_elements, model_bytes));

    SessionOptions so;
    so.session_logid = "webgpu_pool_memory";
    auto session = std::make_unique<InferenceSession>(so, GetEnvironment());
    ASSERT_STATUS_OK(session->RegisterExecutionProvider(make_ep()));
    ASSERT_STATUS_OK(session->Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
    ASSERT_STATUS_OK(session->Initialize());

    std::vector<float> x_values(static_cast<size_t>(num_elements), 1.0f);
    OrtValue x_value;
    CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                         std::vector<int64_t>{num_elements}, x_values, &x_value);

    sessions.push_back(std::move(session));
    feeds.push_back(NameMLValMap{{"X", x_value}});
  }

  for (int i = 0; i < iters; ++i) {
    for (int s = 0; s < num_sessions; ++s) {
      std::vector<OrtValue> fetches;
      ASSERT_STATUS_OK(sessions[s]->Run(RunOptions{}, feeds[s], output_names, &fetches));
    }
  }

  std::cout << "DIAG_POOL sessions=" << num_sessions << " mb=" << mb
            << " vary=" << (vary ? 1 : 0) << " iters=" << iters << " holding...\n"
            << std::flush;
  std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
}

}  // namespace test
}  // namespace onnxruntime
