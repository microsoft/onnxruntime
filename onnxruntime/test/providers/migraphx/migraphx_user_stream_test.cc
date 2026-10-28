// Copyright (c) 2026 CERN.
// Licensed under the MIT License.

// Tests for the "has_user_compute_stream" and "user_compute_stream" options of the MIGraphX execution provider: the
// options must be parsed and reported back by the execution provider, the inference must run in the HIP stream
// provided by the user, asynchronously with respect to the host when the synchronisation of the execution providers
// is disabled, the stream must remain owned by the user, and a stream from a different device must be rejected.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "gtest/gtest.h"

#include "core/framework/execution_provider.h"
#include "core/framework/provider_options.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/migraphx/migraphx_provider_factory_creator.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

// Y[M, N] = A[M, K] @ ones[K, N] + arange(N), with a dynamic M: with A filled with the value v, Y[m][n] == K * v + n.
constexpr int64_t kK = 4;
constexpr int64_t kN = 4;
constexpr int64_t kM = 256;

std::string BuildGemmModel() {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  model.add_opset_import()->set_version(17);
  auto& graph = *model.mutable_graph();
  graph.set_name("migraphx_user_stream_test");

  auto set_value = [](ONNX_NAMESPACE::ValueInfoProto* value, const char* name, const char* rows, int64_t columns) {
    value->set_name(name);
    auto* type = value->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_param(rows);
    type->mutable_shape()->add_dim()->set_dim_value(columns);
  };
  set_value(graph.add_input(), "A", "M", kK);
  set_value(graph.add_output(), "Y", "M", kN);

  auto* b = graph.add_initializer();
  b->set_name("B");
  b->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  b->add_dims(kK);
  b->add_dims(kN);
  for (int64_t i = 0; i < kK * kN; ++i) b->add_float_data(1.f);

  auto* c = graph.add_initializer();
  c->set_name("C");
  c->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  c->add_dims(kN);
  for (int64_t i = 0; i < kN; ++i) c->add_float_data(static_cast<float>(i));

  auto* node = graph.add_node();
  node->set_op_type("Gemm");
  node->add_input("A");
  node->add_input("B");
  node->add_input("C");
  node->add_output("Y");

  return model.SerializeAsString();
}

std::string StreamAddress(hipStream_t stream) {
  return std::to_string(reinterpret_cast<std::uintptr_t>(stream));
}

// Create a MIGraphX execution provider with the given options, and return the options it reports.
ProviderOptions CreateProviderAndGetOptions(const ProviderOptions& options) {
  auto factory = MIGraphXProviderFactoryCreator::Create(options);
  EXPECT_NE(factory, nullptr);
  if (factory == nullptr) {
    return {};
  }
  auto provider = factory->CreateProvider();
  EXPECT_NE(provider, nullptr);
  if (provider == nullptr) {
    return {};
  }
  return provider->GetProviderOptions();
}

uint32_t FloatBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// Block the stream from the host for the number of milliseconds pointed to by data.
void SleepHostFunction(void* data) {
  std::this_thread::sleep_for(std::chrono::milliseconds(*static_cast<const int*>(data)));
}

class MIGraphXUserStreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No HIP device available.";
    }
    ASSERT_EQ(hipSuccess, hipSetDevice(0));
    ASSERT_EQ(hipSuccess, hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_a_, kM * kK * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_y_, kM * kN * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipHostMalloc(&h_y_, kM * kN * sizeof(float), hipHostMallocDefault));
    model_ = BuildGemmModel();
  }

  void TearDown() override {
    if (h_y_ != nullptr) (void)hipHostFree(h_y_);
    if (d_y_ != nullptr) (void)hipFree(d_y_);
    if (d_a_ != nullptr) (void)hipFree(d_a_);
    if (stream_ != nullptr) (void)hipStreamDestroy(stream_);
  }

  Ort::Session CreateSession(hipStream_t stream, int device_id = 0) {
    Ort::SessionOptions options;
    options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    std::unordered_map<std::string, std::string> provider_options{{"device_id", std::to_string(device_id)}};
    if (stream != nullptr) {
      provider_options["user_compute_stream"] = StreamAddress(stream);
    }
    options.AppendExecutionProvider("MIGraphX", provider_options);
    return Ort::Session(*ort_env, model_.data(), model_.size(), options);
  }

  // Wrap device memory in a tensor on the device of the MIGraphX execution provider, so that it is used in place.
  static Ort::Value DeviceTensor(float* data, int64_t rows, int64_t columns) {
    // The MIGraphX execution provider names its device allocator "Cuda", with the AMD vendor id.
    Ort::MemoryInfo memory_info("Cuda", OrtMemoryInfoDeviceType_GPU, 0x1002, 0, OrtDeviceMemoryType_DEFAULT, 0,
                                OrtDeviceAllocator);
    const int64_t shape[] = {rows, columns};
    return Ort::Value::CreateTensor<float>(memory_info, data, static_cast<size_t>(rows * columns), shape, 2);
  }

  // Queue in the user stream: reset A to 0, block the stream for delay_ms, then set A to v; then run the session.
  // If the inference is not ordered after the last write in the user stream, it reads A = 0 (or the value of a
  // previous iteration) and the result is wrong. Returns the time spent in Run(), in milliseconds.
  double EnqueueAndRun(Ort::Session& session, Ort::RunOptions& run_options, float v, int* delay_ms) {
    EXPECT_EQ(hipSuccess, hipMemsetD32Async(d_a_, static_cast<int>(FloatBits(0.f)), kM * kK, stream_));
    EXPECT_EQ(hipSuccess, hipLaunchHostFunc(stream_, SleepHostFunction, delay_ms));
    EXPECT_EQ(hipSuccess, hipMemsetD32Async(d_a_, static_cast<int>(FloatBits(v)), kM * kK, stream_));

    Ort::Value input = DeviceTensor(d_a_, kM, kK);
    Ort::Value output = DeviceTensor(d_y_, kM, kN);
    const char* input_names[] = {"A"};
    const char* output_names[] = {"Y"};
    const auto start = std::chrono::steady_clock::now();
    session.Run(run_options, input_names, &input, 1, output_names, &output, 1);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }

  // Copy the result to the host in the user stream, wait for it, and check it.
  void CheckResult(float v) {
    ASSERT_EQ(hipSuccess, hipMemcpyAsync(h_y_, d_y_, kM * kN * sizeof(float), hipMemcpyDeviceToHost, stream_));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream_));
    for (int64_t m = 0; m < kM; ++m) {
      for (int64_t n = 0; n < kN; ++n) {
        ASSERT_EQ(static_cast<float>(kK) * v + static_cast<float>(n), h_y_[m * kN + n])
            << "v = " << v << ", Y[" << m << "][" << n << "]";
      }
    }
  }

  std::string model_;
  hipStream_t stream_ = nullptr;
  float* d_a_ = nullptr;
  float* d_y_ = nullptr;
  float* h_y_ = nullptr;
};

// The user compute stream options are parsed, reported back by the execution provider, and round-trip through the
// reported options.
TEST_F(MIGraphXUserStreamTest, ProviderOptionsWithUserStream) {
  const ProviderOptions options{{"device_id", "0"}, {"user_compute_stream", StreamAddress(stream_)}};
  const ProviderOptions reported = CreateProviderAndGetOptions(options);
  ASSERT_EQ(1u, reported.count("has_user_compute_stream"));
  ASSERT_EQ(1u, reported.count("user_compute_stream"));
  EXPECT_EQ("1", reported.at("has_user_compute_stream"));
  EXPECT_EQ(StreamAddress(stream_), reported.at("user_compute_stream"));

  // create a second execution provider from the options reported by the first one
  const ProviderOptions round_trip{{"device_id", reported.at("device_id")},
                                   {"has_user_compute_stream", reported.at("has_user_compute_stream")},
                                   {"user_compute_stream", reported.at("user_compute_stream")}};
  const ProviderOptions reported_again = CreateProviderAndGetOptions(round_trip);
  EXPECT_EQ(reported.at("device_id"), reported_again.at("device_id"));
  EXPECT_EQ(reported.at("has_user_compute_stream"), reported_again.at("has_user_compute_stream"));
  EXPECT_EQ(reported.at("user_compute_stream"), reported_again.at("user_compute_stream"));
}

// Without a user compute stream the execution provider creates its own streams; like in the CUDA execution provider,
// "has_user_compute_stream" alone does not enable a user stream, the address of the stream is what matters.
TEST_F(MIGraphXUserStreamTest, ProviderOptionsWithoutUserStream) {
  for (const ProviderOptions& options : {ProviderOptions{{"device_id", "0"}},
                                         ProviderOptions{{"device_id", "0"}, {"has_user_compute_stream", "1"}},
                                         ProviderOptions{{"device_id", "0"}, {"user_compute_stream", "0"}}}) {
    const ProviderOptions reported = CreateProviderAndGetOptions(options);
    ASSERT_EQ(1u, reported.count("has_user_compute_stream"));
    ASSERT_EQ(1u, reported.count("user_compute_stream"));
    EXPECT_EQ("0", reported.at("has_user_compute_stream"));
    EXPECT_EQ("0", reported.at("user_compute_stream"));
  }
}

// A user compute stream that is not a valid address is rejected.
TEST_F(MIGraphXUserStreamTest, ProviderOptionsInvalidUserStream) {
  EXPECT_ANY_THROW(CreateProviderAndGetOptions({{"device_id", "0"}, {"user_compute_stream", "not a stream"}}));
}

// The inference runs in the user stream, ordered after the work already queued in it, and Run() does not wait for
// the stream when the synchronisation of the execution providers is disabled.
TEST_F(MIGraphXUserStreamTest, AsynchronousRunInUserStream) {
  auto session = CreateSession(stream_);
  Ort::RunOptions run_options;
  run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");

  int delay_ms = 200;
  for (int i = 0; i < 5; ++i) {
    const float v = static_cast<float>(i + 1);
    const double elapsed = EnqueueAndRun(session, run_options, v, &delay_ms);
    // The first run compiles the MIGraphX program, and may synchronise the device.
    if (i > 0) {
      EXPECT_LT(elapsed, delay_ms / 2.) << "Run() waited for the user stream";
    }
    CheckResult(v);
  }
}

// With the default run options, Run() synchronises the user stream before returning.
TEST_F(MIGraphXUserStreamTest, SynchronousRunInUserStream) {
  auto session = CreateSession(stream_);
  Ort::RunOptions run_options;

  int delay_ms = 50;
  for (int i = 0; i < 3; ++i) {
    const float v = static_cast<float>(i + 1);
    EnqueueAndRun(session, run_options, v, &delay_ms);
    EXPECT_EQ(hipSuccess, hipStreamQuery(stream_)) << "the user stream is still busy after Run()";
    CheckResult(v);
  }
}

// IoBinding::SynchronizeInputs() and SynchronizeOutputs() call the Sync() method of the execution provider, which must
// wait for the work pending in the user stream (a non-blocking stream is not synchronised with the null stream).
TEST_F(MIGraphXUserStreamTest, SyncWaitsForUserStream) {
  auto session = CreateSession(stream_);
  Ort::Value input = DeviceTensor(d_a_, kM, kK);
  Ort::Value output = DeviceTensor(d_y_, kM, kN);
  Ort::IoBinding binding(session);
  binding.BindInput("A", input);
  binding.BindOutput("Y", output);

  int delay_ms = 200;
  for (bool inputs : {true, false}) {
    ASSERT_EQ(hipSuccess, hipLaunchHostFunc(stream_, SleepHostFunction, &delay_ms));
    ASSERT_EQ(hipSuccess, hipMemsetD32Async(d_a_, static_cast<int>(FloatBits(1.f)), kM * kK, stream_));
    EXPECT_EQ(hipErrorNotReady, hipStreamQuery(stream_)) << "the user stream should still be busy";
    if (inputs) {
      binding.SynchronizeInputs();
    } else {
      binding.SynchronizeOutputs();
    }
    EXPECT_EQ(hipSuccess, hipStreamQuery(stream_))
        << "the user stream is still busy after Synchronize" << (inputs ? "Inputs()" : "Outputs()");
  }
}

// The session selects the device of the user stream on the thread that runs it, even if a different device is
// current on that thread.
TEST_F(MIGraphXUserStreamTest, RunFromThreadOnAnotherDevice) {
  int device_count = 0;
  ASSERT_EQ(hipSuccess, hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "This test needs at least 2 HIP devices.";
  }
  auto session = CreateSession(stream_);
  std::string error;
  std::thread thread([&]() {
    try {
      // make a different device current on this thread
      if (hipSetDevice(1) != hipSuccess) {
        error = "hipSetDevice(1) failed";
        return;
      }
      Ort::RunOptions run_options;
      run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
      for (int i = 0; i < 3; ++i) {
        int delay_ms = 1;
        EnqueueAndRun(session, run_options, static_cast<float>(i + 1), &delay_ms);
        CheckResult(static_cast<float>(i + 1));
      }
    } catch (const Ort::Exception& e) {
      error = e.what();
    }
  });
  thread.join();
  EXPECT_TRUE(error.empty()) << error;
}

// The user stream is not owned by the session: it remains valid after the session has been destroyed.
TEST_F(MIGraphXUserStreamTest, UserStreamOutlivesSession) {
  {
    auto session = CreateSession(stream_);
    Ort::RunOptions run_options;
    run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
    int delay_ms = 1;
    EnqueueAndRun(session, run_options, 1.f, &delay_ms);
    CheckResult(1.f);
  }
  ASSERT_EQ(hipSuccess, hipMemsetD32Async(d_a_, 0, kM * kK, stream_));
  ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream_));
  ASSERT_EQ(hipSuccess, hipStreamDestroy(stream_));
  stream_ = nullptr;
}

// Several sessions, each bound to its own user stream, run concurrently from different threads.
TEST_F(MIGraphXUserStreamTest, ConcurrentSessionsWithUserStreams) {
  constexpr int kThreads = 4;
  std::vector<std::thread> threads;
  std::vector<std::string> errors(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      if (hipSetDevice(0) != hipSuccess) {
        errors[t] = "hipSetDevice failed";
        return;
      }
      hipStream_t stream = nullptr;
      float* d_a = nullptr;
      float* d_y = nullptr;
      std::vector<float> h_y(kM * kN);
      if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess ||
          hipMalloc(&d_a, kM * kK * sizeof(float)) != hipSuccess ||
          hipMalloc(&d_y, kM * kN * sizeof(float)) != hipSuccess) {
        errors[t] = "HIP setup failed";
        return;
      }
      try {
        auto session = CreateSession(stream);
        Ort::RunOptions run_options;
        run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
        Ort::Value input = DeviceTensor(d_a, kM, kK);
        Ort::Value output = DeviceTensor(d_y, kM, kN);
        const char* input_names[] = {"A"};
        const char* output_names[] = {"Y"};
        for (int i = 0; i < 20 && errors[t].empty(); ++i) {
          const float v = static_cast<float>(t * 100 + i);
          (void)hipMemsetD32Async(d_a, static_cast<int>(FloatBits(v)), kM * kK, stream);
          session.Run(run_options, input_names, &input, 1, output_names, &output, 1);
          (void)hipMemcpyAsync(h_y.data(), d_y, h_y.size() * sizeof(float), hipMemcpyDeviceToHost, stream);
          if (hipStreamSynchronize(stream) != hipSuccess) {
            errors[t] = "hipStreamSynchronize failed";
          }
          for (int64_t j = 0; j < kM * kN && errors[t].empty(); ++j) {
            const float expected = static_cast<float>(kK) * v + static_cast<float>(j % kN);
            if (h_y[j] != expected) {
              errors[t] = "iteration " + std::to_string(i) + ": Y[" + std::to_string(j / kN) + "][" +
                          std::to_string(j % kN) + "] = " + std::to_string(h_y[j]) + ", expected " +
                          std::to_string(expected);
            }
          }
        }
      } catch (const Ort::Exception& e) {
        errors[t] = e.what();
      }
      (void)hipFree(d_y);
      (void)hipFree(d_a);
      (void)hipStreamDestroy(stream);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (int t = 0; t < kThreads; ++t) {
    EXPECT_TRUE(errors[t].empty()) << "thread " << t << ": " << errors[t];
  }
}

// A user stream that does not belong to the device used by the execution provider is rejected.
TEST_F(MIGraphXUserStreamTest, UserStreamOnAnotherDevice) {
  int device_count = 0;
  ASSERT_EQ(hipSuccess, hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "This test needs at least 2 HIP devices.";
  }
  hipStream_t other_stream = nullptr;
  ASSERT_EQ(hipSuccess, hipSetDevice(1));
  ASSERT_EQ(hipSuccess, hipStreamCreateWithFlags(&other_stream, hipStreamNonBlocking));
  ASSERT_EQ(hipSuccess, hipSetDevice(0));
  // when creating the execution provider directly ...
  const ProviderOptions options{{"device_id", "0"}, {"user_compute_stream", StreamAddress(other_stream)}};
  EXPECT_ANY_THROW(CreateProviderAndGetOptions(options));
  // ... and through the session options
  EXPECT_THROW(CreateSession(other_stream, 0), Ort::Exception);
  ASSERT_EQ(hipSuccess, hipStreamDestroy(other_stream));
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
