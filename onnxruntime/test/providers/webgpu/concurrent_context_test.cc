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
//   Context-level BufferManagers protect only their shared buffer-cache containers. Command
//   recording remains per session and is passed independently to context operations. Graph capture
//   retains a separate manager per graph so captured resources remain isolated.
//
// The tests cover several distinct multithreaded shapes:
//   A. one session, run() concurrently from many threads
//   B. many threads, each with its own pre-created session, running concurrently
//   C. mixed: some threads create+Initialize+run new sessions while others run existing ones
//   D. churn: many threads each repeatedly create + run + destroy their own session
//   E. a cold and a warm session remain correct while running concurrently

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/onnx_protobuf.h"
#include "core/graph/model.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_external_header.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

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
    if (MakeProvider() == nullptr) {
      GTEST_SKIP() << "WebGPU execution provider is not available.";
    }
    ASSERT_NO_FATAL_FAILURE(BuildAddChainModel(kChainLen, kNumElements, model_bytes_));
    keepalive_ = MakeSession();
  }

  std::unique_ptr<IExecutionProvider> MakeProvider() const {
    ConfigOptions config_options;
    ORT_THROW_IF_ERROR(config_options.AddConfigEntry(webgpu::options::kStorageBufferCacheMode,
                                                     webgpu::options::kBufferCacheMode_Bucket));
    return WebGpuExecutionProviderWithOptions(config_options);
  }

  std::unique_ptr<InferenceSession> MakeSession() {
    SessionOptions so;
    so.session_logid = "webgpu_concurrent_ctx";
    ORT_THROW_IF_ERROR(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    auto session = std::make_unique<InferenceSession>(so, GetEnvironment());
    ORT_THROW_IF_ERROR(session->RegisterExecutionProvider(MakeProvider()));
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
// exercising concurrent allocation and release against the shared context-level buffer manager.
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

// Case E: a session warming up pipelines concurrently with a warm session must remain correct.
//
// A builder thread creates a session over a model with many distinct op types and runs it once,
// which compiles one pipeline per op. Meanwhile a runner thread keeps dispatching on an already
// warm session.
TEST_F(WebGpuConcurrentContextTest, ColdAndWarmSessionsRunConcurrently) {
  std::string cold_model_bytes;
  ASSERT_NO_FATAL_FAILURE(BuildUnaryFanOutModel(kNumElements, cold_model_bytes));

  // Warm session: every pipeline it needs is already compiled and cached on the context.
  auto warm_session = MakeSession();
  ErrorSink sink;
  RunOnce(*warm_session, sink, "E.warmup");
  ASSERT_FALSE(sink.Failed()) << sink.FirstError();

  std::atomic<bool> builder_done{false};

  std::thread builder([&]() {
    try {
      SessionOptions so;
      so.session_logid = "webgpu_concurrent_ctx_cold";
      ORT_THROW_IF_ERROR(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
      InferenceSession cold_session(so, GetEnvironment());
      ORT_THROW_IF_ERROR(cold_session.RegisterExecutionProvider(MakeProvider()));
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
    builder_done.store(true);
  });

  do {
    RunOnce(*warm_session, sink, "E.contended");
  } while (!builder_done.load() && !sink.Failed());
  builder.join();

  ASSERT_FALSE(sink.Failed()) << sink.FirstError();
}

// Case F (future support): the public session allocator shares Run's recording. Concurrent access
// to that recording is unsupported without caller serialization.
TEST_F(WebGpuConcurrentContextTest, DISABLED_SessionAllocatorAndRunConcurrently) {
  constexpr int kIters = 40;
  auto session = MakeSession();
  auto allocator = session->GetAllocator(OrtMemoryInfo(WEBGPU_BUFFER,
                                                       OrtAllocatorType::OrtDeviceAllocator,
                                                       webgpu::WebGpuDevice,
                                                       OrtMemTypeDefault));
  ASSERT_NE(allocator, nullptr);

  ErrorSink sink;
  std::barrier start{2};
  std::thread runner([&]() {
    start.arrive_and_wait();
    RunLoop(*session, kIters, sink, "F.run");
  });
  std::thread allocator_user([&]() {
    start.arrive_and_wait();
    try {
      for (int i = 0; i < kIters && !sink.Failed(); ++i) {
        Tensor tensor(DataTypeImpl::GetType<float>(), TensorShape{kNumElements}, allocator);
        ASSERT_NE(tensor.MutableDataRaw(), nullptr);
      }
    } catch (const std::exception& e) {
      sink.Record(std::string("F.allocator threw: ") + e.what());
    }
  });
  runner.join();
  allocator_user.join();

  ASSERT_FALSE(sink.Failed()) << sink.FirstError();
}

// Case G (future support): the shared cached allocator requires caller serialization of the
// Env recording and allocator statistics. Shared cache locking alone does not protect these.
TEST_F(WebGpuConcurrentContextTest, DISABLED_SharedAllocatorMultiThreadCreateTensor) {
  constexpr int kThreads = 4;
  constexpr int kIters = 60;
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto context_ref = std::shared_ptr<webgpu::WebGpuContext>(&context, [](webgpu::WebGpuContext*) {});
  auto allocator = webgpu::CreateSharedWebGpuAllocator(std::move(context_ref));

  ErrorSink sink;
  std::barrier start{kThreads};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      start.arrive_and_wait();
      try {
        for (int i = 0; i < kIters && !sink.Failed(); ++i) {
          Tensor tensor(DataTypeImpl::GetType<float>(), TensorShape{1024 + t * 16}, allocator);
          ASSERT_NE(tensor.MutableDataRaw(), nullptr);
        }
      } catch (const std::exception& e) {
        sink.Record("G.thread" + std::to_string(t) + " threw: " + e.what());
      }
    });
  }
  JoinAll(threads);

  ASSERT_FALSE(sink.Failed()) << sink.FirstError();
}

// Case H (future support): concurrent use of one recording is unsupported, including the
// context-owned recording shared by Env allocators and transfers.
TEST_F(WebGpuConcurrentContextTest, DISABLED_SharedDataTransferMultiThreadCopy) {
  constexpr int kThreads = 4;
  constexpr int kIters = 30;
  constexpr size_t kElements = 4096;
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::CommandRecordingState recording;
  webgpu::DataTransferImpl data_transfer(context.BufferManager(), recording);

  std::array<wgpu::Buffer, kThreads> gpu_buffers;
  for (auto& buffer : gpu_buffers) {
    wgpu::BufferDescriptor desc{};
    desc.size = kElements * sizeof(float);
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
    buffer = context.Device().CreateBuffer(&desc);
  }

  ErrorSink sink;
  std::barrier start{kThreads};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      std::vector<float> input(kElements, static_cast<float>(t + 1));
      std::vector<float> output(kElements);
      start.arrive_and_wait();
      try {
        for (int i = 0; i < kIters && !sink.Failed(); ++i) {
          ORT_THROW_IF_ERROR(data_transfer.CopyTensor(input.data(), false, gpu_buffers[t].Get(), true,
                                                      input.size() * sizeof(float)));
          ORT_THROW_IF_ERROR(data_transfer.CopyTensor(gpu_buffers[t].Get(), true, output.data(), false,
                                                      output.size() * sizeof(float)));
          if (!std::all_of(output.begin(), output.end(), [&](float value) { return value == input[0]; })) {
            sink.Record("H.thread" + std::to_string(t) + " copied incorrect data");
          }
        }
      } catch (const std::exception& e) {
        sink.Record("H.thread" + std::to_string(t) + " threw: " + e.what());
      }
    });
  }
  JoinAll(threads);

  ASSERT_FALSE(sink.Failed()) << sink.FirstError();
}

// Case I: separate data-transfer objects share the context-level BufferManager but own distinct
// command recording timelines. Their command encoders and pending buffers must remain isolated.
TEST_F(WebGpuConcurrentContextTest, IndependentDataTransfersMultiThreadCopy) {
  constexpr int kThreads = 4;
  constexpr int kIters = 30;
  constexpr size_t kElements = 4096;
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);

  std::array<webgpu::CommandRecordingState, kThreads> recordings;
  std::array<std::unique_ptr<webgpu::DataTransferImpl>, kThreads> data_transfers;
  std::array<wgpu::Buffer, kThreads> gpu_buffers;
  for (int t = 0; t < kThreads; ++t) {
    data_transfers[t] = std::make_unique<webgpu::DataTransferImpl>(context.BufferManager(), recordings[t]);
    wgpu::BufferDescriptor desc{};
    desc.size = kElements * sizeof(float);
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
    gpu_buffers[t] = context.Device().CreateBuffer(&desc);
  }

  ErrorSink sink;
  std::barrier start{kThreads};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      std::vector<float> input(kElements, static_cast<float>(t + 1));
      std::vector<float> output(kElements);
      start.arrive_and_wait();
      try {
        for (int i = 0; i < kIters && !sink.Failed(); ++i) {
          ORT_THROW_IF_ERROR(data_transfers[t]->CopyTensor(input.data(), false, gpu_buffers[t].Get(), true,
                                                           input.size() * sizeof(float)));
          ORT_THROW_IF_ERROR(data_transfers[t]->CopyTensor(gpu_buffers[t].Get(), true, output.data(), false,
                                                           output.size() * sizeof(float)));
          if (!std::all_of(output.begin(), output.end(), [&](float value) { return value == input[0]; })) {
            sink.Record("I.thread" + std::to_string(t) + " copied incorrect data");
          }
        }
      } catch (const std::exception& e) {
        sink.Record("I.thread" + std::to_string(t) + " threw: " + e.what());
      }
    });
  }
  JoinAll(threads);

  ASSERT_FALSE(sink.Failed()) << sink.FirstError();
}

}  // namespace test
}  // namespace onnxruntime
