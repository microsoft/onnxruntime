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
//   or a corrupted buffer cache -> "[Device] is lost", often taking down the GPU process.
//
//   The fix serializes all shared-context mutations (command encoder AND buffer caches) with a
//   per-context recursive mutex (WebGpuContext::context_mutex_ / AcquireContextLock()). With
//   the fix, the concurrent scenarios below complete cleanly with correct output.
//
// The tests cover several distinct multithreaded shapes:
//   A. one session, run() concurrently from many threads
//   B. many threads, each with its own pre-created session, running concurrently
//   C. mixed: some threads create+Initialize+run new sessions while others run existing ones
//   D. churn: many threads each repeatedly create + run + destroy their own session

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/onnx_protobuf.h"
#include "core/graph/model.h"
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
// context, so their Run paths hammer the shared command encoder / buffer cache concurrently.
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
// exercising concurrent allocation and release on the shared buffer caches.
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

}  // namespace test
}  // namespace onnxruntime
