// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for recording ONNX Runtime sessions into a caller-initiated ("external") CUDA graph
// capture.
//
// The scenario these tests protect is a pipeline built from several small sessions that must
// run back to back on one caller-owned stream, exchanging device buffers, with a single graph
// launch and no host round trip in between. Before this support the caller could not wrap
// Session::Run() in its own cudaStreamBeginCapture/cudaStreamEndCapture: ORT replayed its own
// captured graph inside the capture window and CUDA rejected cudaGraphLaunch on a capturing
// stream with error 900 (cudaErrorStreamCaptureUnsupported). ORT now detects the caller's
// capture and records the run instead.
//
// Sessions A and B both run testdata/mul_1.onnx (Y = X * X), chained as
// in -> A -> mid -> B -> out, so out == in^4 and `mid` is a stable device buffer shared
// between the two sessions with no host copy in between.

#ifdef USE_CUDA

#include <array>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_run_options_config_keys.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

#ifdef _WIN32
constexpr const wchar_t* kSquareModelUri = L"testdata/mul_1.onnx";
#else
constexpr const char* kSquareModelUri = "testdata/mul_1.onnx";
#endif

constexpr size_t kNumElements = 6;  // mul_1.onnx has a fixed [3, 2] input
const std::array<int64_t, 2> kShape = {3, 2};

// RAII wrapper for a caller-owned CUDA stream.
class OwnedStream {
 public:
  OwnedStream() { EXPECT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess); }
  ~OwnedStream() {
    if (stream_ != nullptr) {
      // Drain before destroying so teardown cannot race work recorded during a test.
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
  }

  OwnedStream(const OwnedStream&) = delete;
  OwnedStream& operator=(const OwnedStream&) = delete;

  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

// RAII wrapper for a caller-owned device allocation. Addresses stay stable for the lifetime of
// the object, which is what a captured graph requires of every buffer it references.
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t num_elements) : num_elements_(num_elements) {
    EXPECT_EQ(cudaMalloc(&ptr_, num_elements * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMemset(ptr_, 0, num_elements * sizeof(float)), cudaSuccess);
  }
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  float* data() const { return static_cast<float*>(ptr_); }

  void Write(const std::vector<float>& host) const {
    ASSERT_EQ(host.size(), num_elements_);
    ASSERT_EQ(cudaMemcpy(ptr_, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);
  }

  std::vector<float> Read() const {
    std::vector<float> host(num_elements_);
    EXPECT_EQ(cudaMemcpy(host.data(), ptr_, host.size() * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return host;
  }

 private:
  void* ptr_ = nullptr;
  size_t num_elements_ = 0;
};

// Create a session bound to `stream` as its user compute stream. Recording into a caller's graph
// only requires the session to run on the caller's stream; ORT-managed capture is independent and
// is exercised both ways.
Ort::Session CreateSessionOnStream(cudaStream_t stream, bool enable_ort_cuda_graph = true) {
  Ort::SessionOptions session_options;
  Ort::CUDAProviderOptions cuda_options;
  std::unordered_map<std::string, std::string> options_map = {
      {"enable_cuda_graph", enable_ort_cuda_graph ? "1" : "0"},
      {"has_user_compute_stream", "1"},
      {"user_compute_stream", std::to_string(reinterpret_cast<uintptr_t>(stream))},
  };
  cuda_options.Update(options_map);
  session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
  return Ort::Session(*ort_env, kSquareModelUri, session_options);
}

// Bind `input` and `output` device pointers to `session`. The binding is reused across runs so
// that the recorded graph references addresses that stay valid.
Ort::IoBinding CreateBinding(Ort::Session& session, const DeviceBuffer& input, const DeviceBuffer& output) {
  Ort::MemoryInfo mem_info("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  Ort::IoBinding binding(session);
  binding.BindInput("X",
                    Ort::Value::CreateTensor(mem_info, input.data(), kNumElements, kShape.data(), kShape.size()));
  binding.BindOutput("Y",
                     Ort::Value::CreateTensor(mem_info, output.data(), kNumElements, kShape.data(), kShape.size()));
  return binding;
}

// Run both sessions once so the arenas are populated and any ORT-managed capture has completed.
// A capturing stream cannot service a cudaMalloc, so every allocation must already have happened
// before the caller starts recording.
void WarmUp(Ort::Session& a, Ort::IoBinding& binding_a, Ort::Session& b, Ort::IoBinding& binding_b,
            cudaStream_t stream) {
  Ort::RunOptions run_options;
  for (int i = 0; i < 3; ++i) {
    a.Run(run_options, binding_a);
    b.Run(run_options, binding_b);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
}

// Runs `fn` and reports whether it threw an Ort::Exception whose message contains `needle`, so a
// negative test cannot pass because of an unrelated failure.
template <typename Fn>
void ExpectThrowsContaining(Fn&& fn, const std::string& needle) {
  try {
    fn();
  } catch (const Ort::Exception& ex) {
    EXPECT_NE(std::string(ex.what()).find(needle), std::string::npos)
        << "expected the error to mention \"" << needle << "\" but got: " << ex.what();
    return;
  } catch (const std::exception& ex) {
    ADD_FAILURE() << "expected an Ort::Exception mentioning \"" << needle << "\" but got: " << ex.what();
    return;
  }
  ADD_FAILURE() << "expected an Ort::Exception mentioning \"" << needle << "\" but nothing was thrown";
}

bool IsStreamCapturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  EXPECT_EQ(cudaStreamIsCapturing(stream, &status), cudaSuccess);
  return status != cudaStreamCaptureStatusNone;
}

// Expected values are produced by running the same two sessions eagerly on the same input,
// rather than assuming what testdata/mul_1.onnx computes.
std::vector<float> RunEagerChain(Ort::Session& a, Ort::IoBinding& binding_a, Ort::Session& b,
                                 Ort::IoBinding& binding_b, cudaStream_t stream,
                                 const DeviceBuffer& in, const DeviceBuffer& out,
                                 const std::vector<float>& input) {
  in.Write(input);
  Ort::RunOptions run_options;
  a.Run(run_options, binding_a);
  b.Run(run_options, binding_b);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  return out.Read();
}

}  // namespace

// Two sessions sharing one caller-owned stream are recorded into a single caller graph, and the
// replayed graph produces exactly the values eager execution produces.
TEST(CudaExternalGraphCaptureTest, TwoSessionsRecordIntoOneCallerGraph) {
  OwnedStream stream;
  DeviceBuffer in(kNumElements), mid(kNumElements), out(kNumElements);

  Ort::Session session_a = CreateSessionOnStream(stream.get());
  Ort::Session session_b = CreateSessionOnStream(stream.get());
  Ort::IoBinding binding_a = CreateBinding(session_a, in, mid);
  Ort::IoBinding binding_b = CreateBinding(session_b, mid, out);

  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> input2 = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  const std::vector<float> input3 = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};

  in.Write(input);
  ASSERT_NO_FATAL_FAILURE(WarmUp(session_a, binding_a, session_b, binding_b, stream.get()));

  // Eager references for every input the recorded graph will later be replayed with.
  const std::vector<float> expected2 =
      RunEagerChain(session_a, binding_a, session_b, binding_b, stream.get(), in, out, input2);
  const std::vector<float> expected3 =
      RunEagerChain(session_a, binding_a, session_b, binding_b, stream.get(), in, out, input3);
  ASSERT_NE(expected2, expected3) << "inputs must produce distinguishable outputs";

  in.Write(input);

  // Record both runs into one caller-owned graph. ThreadLocal mode keeps the capture from
  // interfering with unrelated CUDA activity on other threads of the test process.
  ASSERT_EQ(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal), cudaSuccess);
  EXPECT_TRUE(IsStreamCapturing(stream.get()));

  Ort::RunOptions run_options;
  run_options.AddConfigEntry(kOrtRunOptionsConfigExternalDeviceGraphCapture, "1");
  ASSERT_NO_THROW(session_a.Run(run_options, binding_a));
  ASSERT_NO_THROW(session_b.Run(run_options, binding_b));

  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(stream.get(), &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);

  cudaGraphExec_t graph_exec = nullptr;
  ASSERT_EQ(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0), cudaSuccess);

  // A fresh input proves the replay recomputes rather than replaying a stale result, and that
  // both sessions are inside the one graph: a graph holding only session A would leave `out`
  // holding session A's output instead of the chained result.
  in.Write(input2);
  ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  EXPECT_EQ(out.Read(), expected2);

  // Repeated replay of the same graph stays correct and needs no re-capture.
  in.Write(input3);
  for (int i = 0; i < 5; ++i) {
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.get()), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  EXPECT_EQ(out.Read(), expected3);

  ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  // Sessions and stream are destroyed after the graph, in reverse order of dependency.
}

// The caller may record, destroy, and re-record without the sessions getting stuck in a bad state.
TEST(CudaExternalGraphCaptureTest, RepeatedCaptureAndReplayOnOneStream) {
  OwnedStream stream;
  DeviceBuffer in(kNumElements), mid(kNumElements), out(kNumElements);

  Ort::Session session_a = CreateSessionOnStream(stream.get());
  Ort::Session session_b = CreateSessionOnStream(stream.get());
  Ort::IoBinding binding_a = CreateBinding(session_a, in, mid);
  Ort::IoBinding binding_b = CreateBinding(session_b, mid, out);

  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  in.Write(input);
  ASSERT_NO_FATAL_FAILURE(WarmUp(session_a, binding_a, session_b, binding_b, stream.get()));

  // Eager references for each round's input, plus the final non-capturing run.
  constexpr int kRounds = 3;
  std::vector<std::vector<float>> expected_per_round;
  for (int round = 0; round < kRounds; ++round) {
    const std::vector<float> round_input(kNumElements, static_cast<float>(round) + 1.0f);
    expected_per_round.push_back(
        RunEagerChain(session_a, binding_a, session_b, binding_b, stream.get(), in, out, round_input));
  }
  const std::vector<float> final_input = {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f};
  const std::vector<float> expected_final =
      RunEagerChain(session_a, binding_a, session_b, binding_b, stream.get(), in, out, final_input);

  Ort::RunOptions run_options;
  run_options.AddConfigEntry(kOrtRunOptionsConfigExternalDeviceGraphCapture, "1");

  for (int round = 0; round < kRounds; ++round) {
    ASSERT_EQ(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal), cudaSuccess);
    ASSERT_NO_THROW(session_a.Run(run_options, binding_a));
    ASSERT_NO_THROW(session_b.Run(run_options, binding_b));

    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(stream.get(), &graph), cudaSuccess) << "round " << round;
    ASSERT_NE(graph, nullptr);

    cudaGraphExec_t graph_exec = nullptr;
    ASSERT_EQ(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0), cudaSuccess);

    const std::vector<float> round_input(kNumElements, static_cast<float>(round) + 1.0f);
    in.Write(round_input);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
    EXPECT_EQ(out.Read(), expected_per_round[round]) << "round " << round;

    ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  }

  // After the caller stops capturing, the sessions fall back to their normal behavior.
  in.Write(final_input);
  Ort::RunOptions plain_options;
  ASSERT_NO_THROW(session_a.Run(plain_options, binding_a));
  ASSERT_NO_THROW(session_b.Run(plain_options, binding_b));
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  EXPECT_EQ(out.Read(), expected_final);
}

// Negative: asking for record-only mode without an active capture is a caller bug and must be
// reported, not silently ignored.
TEST(CudaExternalGraphCaptureTest, RequireExternalCaptureFailsWithoutActiveCapture) {
  OwnedStream stream;
  DeviceBuffer in(kNumElements), out(kNumElements);

  Ort::Session session = CreateSessionOnStream(stream.get());
  Ort::IoBinding binding = CreateBinding(session, in, out);
  in.Write(std::vector<float>(kNumElements, 2.0f));

  Ort::RunOptions run_options;
  run_options.AddConfigEntry(kOrtRunOptionsConfigExternalDeviceGraphCapture, "1");

  ASSERT_FALSE(IsStreamCapturing(stream.get()));
  ExpectThrowsContaining([&] { session.Run(run_options, binding); },
                         "no caller-initiated device graph capture is active");
}

// Negative: a capture on some other stream must not be mistaken for a capture on the stream this
// session runs on. This catches the common wiring mistake of capturing on a stream the session
// was never bound to.
TEST(CudaExternalGraphCaptureTest, RequireExternalCaptureFailsWhenSessionIsOnAnotherStream) {
  OwnedStream session_stream;
  OwnedStream other_stream;
  DeviceBuffer in(kNumElements), out(kNumElements);

  Ort::Session session = CreateSessionOnStream(session_stream.get());
  Ort::IoBinding binding = CreateBinding(session, in, out);
  in.Write(std::vector<float>(kNumElements, 2.0f));

  Ort::RunOptions warmup_options;
  ASSERT_NO_THROW(session.Run(warmup_options, binding));
  ASSERT_EQ(cudaStreamSynchronize(session_stream.get()), cudaSuccess);

  // Relaxed mode: this thread must stay free to drive the session's own (non-capturing) stream,
  // which ThreadLocal and Global modes would forbid.
  ASSERT_EQ(cudaStreamBeginCapture(other_stream.get(), cudaStreamCaptureModeRelaxed), cudaSuccess);

  Ort::RunOptions run_options;
  run_options.AddConfigEntry(kOrtRunOptionsConfigExternalDeviceGraphCapture, "1");
  ExpectThrowsContaining([&] { session.Run(run_options, binding); },
                         "no caller-initiated device graph capture is active");

  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(other_stream.get(), &graph), cudaSuccess);
  if (graph != nullptr) {
    cudaGraphDestroy(graph);
  }
}

// Negative: the run option only accepts "0" and "1".
TEST(CudaExternalGraphCaptureTest, RequireExternalCaptureRejectsInvalidValue) {
  OwnedStream stream;
  DeviceBuffer in(kNumElements), out(kNumElements);

  Ort::Session session = CreateSessionOnStream(stream.get());
  Ort::IoBinding binding = CreateBinding(session, in, out);
  in.Write(std::vector<float>(kNumElements, 2.0f));

  Ort::RunOptions run_options;
  run_options.AddConfigEntry(kOrtRunOptionsConfigExternalDeviceGraphCapture, "yes");
  ExpectThrowsContaining([&] { session.Run(run_options, binding); }, "expected \"0\" or \"1\"");
}

// A session whose stream is not capturing keeps its existing ORT-managed capture/replay behavior
// even while another stream in the process is being captured.
TEST(CudaExternalGraphCaptureTest, UnrelatedSessionKeepsOrtManagedCapture) {
  OwnedStream session_stream;
  OwnedStream other_stream;
  DeviceBuffer in(kNumElements), out(kNumElements);

  Ort::Session session = CreateSessionOnStream(session_stream.get());
  Ort::IoBinding binding = CreateBinding(session, in, out);

  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> input2 = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  // Establish the expected output for input2 (and let the session capture its own graph), then
  // leave the buffer holding a different value so the check below cannot pass on a stale result.
  Ort::RunOptions run_options;
  in.Write(input2);
  ASSERT_NO_THROW(session.Run(run_options, binding));
  ASSERT_EQ(cudaStreamSynchronize(session_stream.get()), cudaSuccess);
  const std::vector<float> expected2 = out.Read();

  in.Write(input);
  ASSERT_NO_THROW(session.Run(run_options, binding));
  ASSERT_EQ(cudaStreamSynchronize(session_stream.get()), cudaSuccess);
  const std::vector<float> expected1 = out.Read();
  ASSERT_NE(expected1, expected2) << "inputs must produce distinguishable outputs";

  ASSERT_EQ(cudaStreamBeginCapture(other_stream.get(), cudaStreamCaptureModeRelaxed), cudaSuccess);
  // Replay on the session's own (non-capturing) stream is still allowed and still correct, and it
  // must recompute: `out` currently holds the result for `input`, not for `input2`.
  in.Write(input2);
  ASSERT_NO_THROW(session.Run(run_options, binding));
  ASSERT_EQ(cudaStreamSynchronize(session_stream.get()), cudaSuccess);

  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(other_stream.get(), &graph), cudaSuccess);
  if (graph != nullptr) {
    cudaGraphDestroy(graph);
  }

  EXPECT_EQ(out.Read(), expected2);
}

// Recording works without the explicit run option too: ORT detects the caller's capture on its
// own, which is what keeps existing callers from having to thread a new flag through their stack.
TEST(CudaExternalGraphCaptureTest, RecordingIsDetectedWithoutRunOption) {
  OwnedStream stream;
  DeviceBuffer in(kNumElements), mid(kNumElements), out(kNumElements);

  Ort::Session session_a = CreateSessionOnStream(stream.get());
  Ort::Session session_b = CreateSessionOnStream(stream.get());
  Ort::IoBinding binding_a = CreateBinding(session_a, in, mid);
  Ort::IoBinding binding_b = CreateBinding(session_b, mid, out);

  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> input2(kNumElements, 3.0f);
  in.Write(input);
  ASSERT_NO_FATAL_FAILURE(WarmUp(session_a, binding_a, session_b, binding_b, stream.get()));

  const std::vector<float> expected2 =
      RunEagerChain(session_a, binding_a, session_b, binding_b, stream.get(), in, out, input2);
  // Leave `out` holding the result for a different input, so a graph that records nothing (or
  // only session A) cannot make the final check pass on a stale value.
  const std::vector<float> expected1 =
      RunEagerChain(session_a, binding_a, session_b, binding_b, stream.get(), in, out, input);
  ASSERT_NE(expected1, expected2) << "inputs must produce distinguishable outputs";

  ASSERT_EQ(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal), cudaSuccess);
  Ort::RunOptions run_options;  // no external_device_graph_capture entry
  ASSERT_NO_THROW(session_a.Run(run_options, binding_a));
  ASSERT_NO_THROW(session_b.Run(run_options, binding_b));

  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(stream.get(), &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);

  cudaGraphExec_t graph_exec = nullptr;
  ASSERT_EQ(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0), cudaSuccess);

  in.Write(input2);
  ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  EXPECT_EQ(out.Read(), expected2);

  ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

// Recording into the caller's graph must not require the session to also enable ORT-managed CUDA
// graph capture: the caller owns the graph, so `enable_cuda_graph` is irrelevant to it.
TEST(CudaExternalGraphCaptureTest, RecordsWithoutOrtManagedCudaGraph) {
  OwnedStream stream;
  DeviceBuffer in(kNumElements), mid(kNumElements), out(kNumElements);

  Ort::Session session_a = CreateSessionOnStream(stream.get(), /*enable_ort_cuda_graph=*/false);
  Ort::Session session_b = CreateSessionOnStream(stream.get(), /*enable_ort_cuda_graph=*/false);
  Ort::IoBinding binding_a = CreateBinding(session_a, in, mid);
  Ort::IoBinding binding_b = CreateBinding(session_b, mid, out);

  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> input2 = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  in.Write(input);
  ASSERT_NO_FATAL_FAILURE(WarmUp(session_a, binding_a, session_b, binding_b, stream.get()));

  const std::vector<float> expected2 =
      RunEagerChain(session_a, binding_a, session_b, binding_b, stream.get(), in, out, input2);
  const std::vector<float> expected1 =
      RunEagerChain(session_a, binding_a, session_b, binding_b, stream.get(), in, out, input);
  ASSERT_NE(expected1, expected2) << "inputs must produce distinguishable outputs";

  ASSERT_EQ(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal), cudaSuccess);
  Ort::RunOptions run_options;
  run_options.AddConfigEntry(kOrtRunOptionsConfigExternalDeviceGraphCapture, "1");
  ASSERT_NO_THROW(session_a.Run(run_options, binding_a));
  ASSERT_NO_THROW(session_b.Run(run_options, binding_b));

  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(stream.get(), &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);

  cudaGraphExec_t graph_exec = nullptr;
  ASSERT_EQ(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0), cudaSuccess);

  in.Write(input2);
  ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  EXPECT_EQ(out.Read(), expected2);

  ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_CUDA
