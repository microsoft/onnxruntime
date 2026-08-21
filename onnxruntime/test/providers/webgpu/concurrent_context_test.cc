// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Concurrency regression tests for the shared default WebGpuContext (context_id=0) used by
// multiple InferenceSessions on different threads.
//
// Background:
//   InferenceSession only serializes a single session's Run via its own session_mutex_ (the
//   WebGPU EP reports ConcurrentRunSupported() == false). It does NOT serialize across
//   sessions. Multiple sessions with the default WebGPU provider share one WebGpuContext, so
//   their Run / allocation / initializer-upload paths run concurrently and mutate the
//   context's single command encoder (current_command_encoder_ / current_compute_pass_encoder_
//   / num_pending_dispatches_) AND the shared BufferManager cache maps.
//
//   Before the fix this produced a data race and Dawn errors such as:
//     "[CommandEncoder] is already finished. While encoding CopyBufferToBuffer(...)"
//     "WebGPU validation failed. Command encoding already finished."
//   a corrupted buffer cache -> "[Device] is lost", or - worst of all - a silently wrong result
//   when a buffer was recycled before the work referencing it had been submitted.
//
//   The fix partitions state by ownership rather than wrapping shared state in locks:
//     - Command recording state (encoder, compute pass, pending dispatch count) moved from the
//       shared context onto the EP, i.e. one per session. Dawn's ImplicitDeviceSynchronization
//       explicitly does not cover command encoding, and InferenceSession already serializes a
//       single session's Run via session_mutex_, so per-session ownership needs no lock at all.
//     - Each session owns its BufferManager and buffer caches, so allocation and release never
//       mutate another session's cache state.
//
// The tests cover several distinct multithreaded shapes:
//   A. one session, run() concurrently from many threads
//   B. many threads, each with its own pre-created session, running concurrently
//   C. mixed: some threads create+Initialize+run new sessions while others run existing ones
//   D. churn: many threads each repeatedly create + run + destroy their own session
//   E. a cold session compiling many pipelines must not block a warm session's inference

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/onnx_protobuf.h"
#include "core/graph/model.h"
#include "core/platform/env.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/inference_session.h"

#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

// Builds an in-memory model: Y = ((X + W0) + W1) + ... + W(chain_len-1)
// where each Wi is a constant initializer of `num_elements` floats. The constant initializers
// exercise the BufferManager::Upload (CopyBufferToBuffer) path during Initialize, and each Add
// node produces a compute dispatch during Run.
void BuildAddChainModel(int chain_len, int64_t num_elements, std::string& model_bytes) {
  const std::unordered_map<std::string, int> domain_to_version{{"", 13}};
  Model model("webgpu_concurrent_ctx", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
              std::vector<ONNX_NAMESPACE::FunctionProto>(),
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto float_1d;
  float_1d.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(num_elements);

  const std::vector<float> weight_values(static_cast<size_t>(num_elements), 0.5f);

  NodeArg* prev = &graph.GetOrCreateNodeArg("X", &float_1d);
  for (int i = 0; i < chain_len; ++i) {
    const std::string w_name = "W" + std::to_string(i);
    ONNX_NAMESPACE::TensorProto w_tensor;
    w_tensor.set_name(w_name);
    w_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    w_tensor.add_dims(num_elements);
    w_tensor.set_raw_data(weight_values.data(), weight_values.size() * sizeof(float));
    graph.AddInitializedTensor(w_tensor);

    NodeArg* w_arg = &graph.GetOrCreateNodeArg(w_name, &float_1d);
    const std::string out_name = (i == chain_len - 1) ? "Y" : ("H" + std::to_string(i));
    NodeArg* out_arg = &graph.GetOrCreateNodeArg(out_name, &float_1d);
    std::vector<NodeArg*> inputs{prev, w_arg};
    std::vector<NodeArg*> outputs{out_arg};
    graph.AddNode("add" + std::to_string(i), "Add", "", inputs, outputs);
    prev = out_arg;
  }

  graph.SetOutputs(std::vector<const NodeArg*>{prev});
  ASSERT_STATUS_OK(graph.Resolve());
  ASSERT_TRUE(model.ToProto().SerializeToString(&model_bytes));
}

// The distinct unary op types used by BuildUnaryFanOutModel. Each op type maps to its own WebGPU
// program, so a session using this model has to compile one pipeline per entry.
constexpr const char* kUnaryOps[] = {
    "Abs", "Neg", "Floor", "Ceil", "Reciprocal", "Sqrt", "Exp", "Erf", "Sigmoid",
    "Sin", "Cos", "Tan", "Atan", "Sinh", "Cosh", "Tanh", "HardSigmoid", "HardSwish"};

// Builds an in-memory model that fans one input out to every op in kUnaryOps:
//   Y0 = Abs(X), Y1 = Neg(X), ...
//
// Distinct op types mean distinct programs, so the first Run of a session built from this model
// compiles std::size(kUnaryOps) pipelines. A fan-out rather than a chain keeps every op inside
// its valid input domain no matter how many are used.
void BuildUnaryFanOutModel(int64_t num_elements, std::string& model_bytes) {
  const std::unordered_map<std::string, int> domain_to_version{{"", 14}};
  Model model("webgpu_concurrent_ctx_cold", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
              std::vector<ONNX_NAMESPACE::FunctionProto>(),
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto float_1d;
  float_1d.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(num_elements);

  NodeArg* x_arg = &graph.GetOrCreateNodeArg("X", &float_1d);
  std::vector<const NodeArg*> outputs;
  outputs.reserve(std::size(kUnaryOps));

  for (size_t i = 0; i < std::size(kUnaryOps); ++i) {
    NodeArg* out_arg = &graph.GetOrCreateNodeArg("Y" + std::to_string(i), &float_1d);
    std::vector<NodeArg*> node_inputs{x_arg};
    std::vector<NodeArg*> node_outputs{out_arg};
    graph.AddNode("op" + std::to_string(i), kUnaryOps[i], "", node_inputs, node_outputs);
    outputs.push_back(out_arg);
  }

  graph.SetOutputs(outputs);
  ASSERT_STATUS_OK(graph.Resolve());
  ASSERT_TRUE(model.ToProto().SerializeToString(&model_bytes));
}

// Thread-safe first-error recorder that also acts as a stop flag for the worker loops.
class ErrorSink {
 public:
  void Record(const std::string& message) {
    bool expected = false;
    if (failed_.compare_exchange_strong(expected, true)) {
      std::lock_guard<std::mutex> lock(mutex_);
      first_error_ = message;
    }
  }

  bool Failed() const { return failed_.load(); }

  std::string FirstError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return first_error_;
  }

 private:
  std::atomic<bool> failed_{false};
  mutable std::mutex mutex_;
  std::string first_error_;
};

}  // namespace

// Fixture: builds the shared model once and provides session/feed factories. A long-lived
// keepalive session pins the shared WebGPU context (ref-count > 0) for the whole test so that
// churn/destroy in one thread never tears the context down under the others.
class WebGpuConcurrentContextTest : public ::testing::Test {
 protected:
  static constexpr int64_t kNumElements = 256 * 1024;  // 1 MB per initializer
  static constexpr int kChainLen = 8;                  // 8 uploads + 8 dispatches per session
  static constexpr float kExpected = 1.0f + 0.5f * kChainLen;

  void SetUp() override {
    if (DefaultWebGpuExecutionProvider() == nullptr) {
      GTEST_SKIP() << "WebGPU execution provider is not available.";
    }
    ASSERT_NO_FATAL_FAILURE(BuildAddChainModel(kChainLen, kNumElements, model_bytes_));
    keepalive_ = MakeSession();
  }

  std::unique_ptr<InferenceSession> MakeSession() {
    SessionOptions so;
    so.session_logid = "webgpu_concurrent_ctx";
    auto session = std::make_unique<InferenceSession>(so, GetEnvironment());
    ORT_THROW_IF_ERROR(session->RegisterExecutionProvider(DefaultWebGpuExecutionProvider()));
    ORT_THROW_IF_ERROR(session->Load(model_bytes_.data(), static_cast<int>(model_bytes_.size())));
    ORT_THROW_IF_ERROR(session->Initialize());
    return session;
  }

  NameMLValMap MakeFeeds() const {
    std::vector<float> x_values(static_cast<size_t>(kNumElements), 1.0f);
    OrtValue x_value;
    CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                         std::vector<int64_t>{kNumElements}, x_values, &x_value);
    return NameMLValMap{{"X", x_value}};
  }

  // Runs one inference and validates the numerical result. Records into `sink` on failure.
  void RunOnce(InferenceSession& session, ErrorSink& sink, const std::string& tag) {
    std::vector<std::string> output_names{"Y"};
    std::vector<OrtValue> fetches;
    Status s = session.Run(RunOptions{}, MakeFeeds(), output_names, &fetches);
    if (!s.IsOK()) {
      sink.Record(tag + " Run failed: " + s.ErrorMessage());
      return;
    }
    const Tensor& out = fetches[0].Get<Tensor>();
    const float* data = out.Data<float>();
    const int64_t n = out.Shape().Size();
    for (int64_t i = 0; i < n; i += (n / 8) + 1) {
      if (std::abs(data[i] - kExpected) > 1e-3f) {
        sink.Record(tag + " wrong output: " + std::to_string(data[i]));
        return;
      }
    }
  }

  // Repeatedly runs an existing session until `iters` reached or a failure is recorded.
  void RunLoop(InferenceSession& session, int iters, ErrorSink& sink, const std::string& tag) {
    for (int i = 0; i < iters && !sink.Failed(); ++i) {
      try {
        RunOnce(session, sink, tag);
      } catch (const std::exception& e) {
        sink.Record(tag + " threw: " + e.what());
        return;
      }
    }
  }

  static void JoinAll(std::vector<std::thread>& threads) {
    for (auto& t : threads) {
      t.join();
    }
  }

  std::string model_bytes_;
  std::unique_ptr<InferenceSession> keepalive_;
};

// Case A: one session, run() concurrently from many threads. InferenceSession serializes these
// via session_mutex_, but this still must never crash or deadlock on the shared context.
TEST_F(WebGpuConcurrentContextTest, SingleSessionMultiThreadRun) {
  constexpr int kThreads = 4;
  constexpr int kIters = 30;

  auto session = MakeSession();
  ErrorSink sink;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() { RunLoop(*session, kIters, sink, "A.run" + std::to_string(t)); });
  }
  JoinAll(threads);

  ASSERT_FALSE(sink.Failed()) << sink.FirstError();
}

// Case B: many threads, each with its own pre-created session. All sessions share the default
// context, so their Run paths execute concurrently against it. Before the fix they also shared a
// single command encoder, which is what this case exposed.
TEST_F(WebGpuConcurrentContextTest, PerThreadSessionRun) {
  constexpr int kThreads = 4;
  constexpr int kIters = 30;

  std::vector<std::unique_ptr<InferenceSession>> sessions(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    sessions[t] = MakeSession();
  }

  ErrorSink sink;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() { RunLoop(*sessions[t], kIters, sink, "B.sess" + std::to_string(t)); });
  }
  JoinAll(threads);

  ASSERT_FALSE(sink.Failed()) << sink.FirstError();
}

// Case C: mixed. Runner threads dispatch on pre-initialized sessions while builder threads keep
// creating + Initializing (initializer-upload path) + running fresh sessions. This is the
// closest shape to the original WebNN crash (dispatch racing initializer upload).
TEST_F(WebGpuConcurrentContextTest, MixedCreateAndRun) {
  constexpr int kRunners = 3;
  constexpr int kBuilders = 3;
  constexpr int kIters = 30;

  std::vector<std::unique_ptr<InferenceSession>> runner_sessions(kRunners);
  for (int t = 0; t < kRunners; ++t) {
    runner_sessions[t] = MakeSession();
  }

  ErrorSink sink;
  std::vector<std::thread> threads;
  for (int t = 0; t < kRunners; ++t) {
    threads.emplace_back([&, t]() { RunLoop(*runner_sessions[t], kIters, sink, "C.runner" + std::to_string(t)); });
  }
  for (int t = 0; t < kBuilders; ++t) {
    threads.emplace_back([&, t]() {
      const std::string tag = "C.builder" + std::to_string(t);
      for (int i = 0; i < kIters && !sink.Failed(); ++i) {
        try {
          auto session = MakeSession();  // Initialize -> initializer upload
          RunOnce(*session, sink, tag);
        } catch (const std::exception& e) {
          sink.Record(tag + " threw: " + e.what());
          return;
        }
      }
    });
  }
  JoinAll(threads);

  ASSERT_FALSE(sink.Failed()) << sink.FirstError();
}

// Case D: churn. Many threads each repeatedly create + run + destroy their own session,
// exercising concurrent allocation and release against independent per-session buffer managers.
TEST_F(WebGpuConcurrentContextTest, ChurnCreateRunDestroy) {
  constexpr int kThreads = 4;
  constexpr int kIters = 15;

  ErrorSink sink;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      const std::string tag = "D.churn" + std::to_string(t);
      for (int i = 0; i < kIters && !sink.Failed(); ++i) {
        try {
          auto session = MakeSession();
          RunOnce(*session, sink, tag);
          session.reset();  // drop session -> release its resources while others run
        } catch (const std::exception& e) {
          sink.Record(tag + " threw: " + e.what());
          return;
        }
      }
    });
  }
  JoinAll(threads);

  ASSERT_FALSE(sink.Failed()) << sink.FirstError();
}

// Case E: the review concern behind this design - a session that is warming up (compiling
// pipelines) must not block inference in other sessions sharing the context.
//
// A builder thread creates a session over a model with many distinct op types and runs it once,
// which compiles one pipeline per op. Meanwhile a runner thread keeps dispatching on an already
// warm session.
//
// If shader compilation were performed while holding a shared context lock, the runner would be
// stalled for most of the builder's window. The assertion is therefore on throughput retained
// relative to the runner's uncontended rate, which separates the two designs cleanly; worst-case
// latency is reported as well but is too noisy to assert on.
TEST_F(WebGpuConcurrentContextTest, ColdSessionDoesNotBlockWarmInference) {
  using Clock = std::chrono::steady_clock;
  using Ms = std::chrono::duration<double, std::milli>;

  std::string cold_model_bytes;
  ASSERT_NO_FATAL_FAILURE(BuildUnaryFanOutModel(kNumElements, cold_model_bytes));

  // Warm session: every pipeline it needs is already compiled and cached on the context.
  auto warm_session = MakeSession();
  ErrorSink sink;
  RunOnce(*warm_session, sink, "E.warmup");
  ASSERT_FALSE(sink.Failed()) << sink.FirstError();

  // Baseline latency with no other session active.
  constexpr int kBaselineIters = 20;
  std::vector<double> baseline_ms;
  baseline_ms.reserve(kBaselineIters);
  for (int i = 0; i < kBaselineIters; ++i) {
    const auto start = Clock::now();
    RunOnce(*warm_session, sink, "E.baseline");
    baseline_ms.push_back(Ms(Clock::now() - start).count());
  }
  ASSERT_FALSE(sink.Failed()) << sink.FirstError();
  std::sort(baseline_ms.begin(), baseline_ms.end());
  const double baseline_median = baseline_ms[baseline_ms.size() / 2];

  std::atomic<bool> builder_done{false};
  std::vector<double> contended_ms;
  double builder_ms = 0.0;

  std::thread builder([&]() {
    const auto start = Clock::now();
    try {
      SessionOptions so;
      so.session_logid = "webgpu_concurrent_ctx_cold";
      InferenceSession cold_session(so, GetEnvironment());
      ORT_THROW_IF_ERROR(cold_session.RegisterExecutionProvider(DefaultWebGpuExecutionProvider()));
      ORT_THROW_IF_ERROR(cold_session.Load(cold_model_bytes.data(), static_cast<int>(cold_model_bytes.size())));
      ORT_THROW_IF_ERROR(cold_session.Initialize());

      std::vector<std::string> output_names;
      for (size_t i = 0; i < std::size(kUnaryOps); ++i) {
        output_names.push_back("Y" + std::to_string(i));
      }
      std::vector<OrtValue> fetches;
      // This Run is what triggers compilation of one pipeline per op type.
      ORT_THROW_IF_ERROR(cold_session.Run(RunOptions{}, MakeFeeds(), output_names, &fetches));
    } catch (const std::exception& e) {
      sink.Record(std::string("E.builder threw: ") + e.what());
    }
    builder_ms = Ms(Clock::now() - start).count();
    builder_done.store(true);
  });

  // Always take at least one measurement so the loop cannot produce an empty sample set if the
  // builder happens to finish first.
  const auto contended_start = Clock::now();
  do {
    const auto start = Clock::now();
    RunOnce(*warm_session, sink, "E.contended");
    contended_ms.push_back(Ms(Clock::now() - start).count());
  } while (!builder_done.load() && !sink.Failed());
  const double contended_window_ms = Ms(Clock::now() - contended_start).count();
  builder.join();

  ASSERT_FALSE(sink.Failed()) << sink.FirstError();

  const double contended_max = *std::max_element(contended_ms.begin(), contended_ms.end());
  std::sort(contended_ms.begin(), contended_ms.end());
  const double contended_median = contended_ms[contended_ms.size() / 2];

  // Fraction of the contended window during which the warm session kept doing useful work, using
  // its uncontended latency as the reference. 1.0 means the cold session cost it nothing.
  //
  // This aggregate is a far better regression signal than worst-case latency, which is dominated
  // by scheduler noise: when compilation holds a shared lock the warm session loses most of the
  // window, whereas it stays high once compilation is off the shared path entirely.
  const double throughput_efficiency =
      (static_cast<double>(contended_ms.size()) * baseline_median) / contended_window_ms;

  std::cout << "[ WebGPU  ] cold-session warmup: " << builder_ms << " ms for "
            << std::size(kUnaryOps) << " pipelines\n"
            << "[ WebGPU  ] warm-session latency: baseline median " << baseline_median
            << " ms, contended median " << contended_median
            << " ms, contended max " << contended_max
            << " ms over " << contended_ms.size() << " runs\n"
            << "[ WebGPU  ] warm-session throughput efficiency while cold session warmed up: "
            << throughput_efficiency << std::endl;

  // Only assert when the compile window was long enough relative to a warm run for interference to
  // be observable; otherwise there is nothing meaningful to measure and asserting would be flaky.
  //
  // Measured on Windows/D3D12 (RTX 5080), five runs each: a design that compiles under a shared
  // context lock yields 0.22-0.42, whereas the per-session design yields 0.81-0.93 - and its
  // contended median latency is indistinguishable from the uncontended baseline (~1.95 ms both),
  // i.e. the residual gap is CPU contention from the compiling thread, not blocking. The threshold
  // sits in the gap so the test fails on a real regression without flaking on a busy CI machine.
  if (builder_ms > 10.0 * std::max(baseline_median, 1.0)) {
    EXPECT_GT(throughput_efficiency, 0.5)
        << "warm inference lost most of the window to the cold session's shader compilation";
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

  // Scalar weights, so GPU memory is dominated by intermediate tensors (which flow through the
  // buffer pool) rather than by per-session initializers (which never share across sessions and
  // would otherwise swamp the signal).
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
