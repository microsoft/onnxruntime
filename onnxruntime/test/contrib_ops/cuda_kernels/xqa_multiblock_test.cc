// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_bf16.h>
#include "gtest/gtest.h"
#include "contrib_ops/cuda/bert/xqa/xqa_loader.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime::contrib::cuda::test {
namespace {

// Decode problems of the streams run concurrently, so CTAs of different launches interleave on the SMs. That
// perturbs the order in which the CTAs of one sequence reach the cross-CTA reduction.
constexpr int kNumStreams = 4;
constexpr int kCapacity = 2503;
constexpr int kTileTokens = 256;  // KV tokens consumed by a CTA per iteration
constexpr float kScale = 0.125f;
constexpr int kSplits[] = {1, 4};
constexpr int kLengths[] = {505, 1025, kCapacity};

struct DeviceDeleter {
  void operator()(void* buffer) const { cudaFree(buffer); }
};
struct PinnedHostDeleter {
  void operator()(void* buffer) const { cudaFreeHost(buffer); }
};
struct StreamDeleter {
  void operator()(cudaStream_t stream) const { cudaStreamDestroy(stream); }
};
struct EventDeleter {
  void operator()(cudaEvent_t event) const { cudaEventDestroy(event); }
};
struct GraphDeleter {
  void operator()(cudaGraph_t graph) const { cudaGraphDestroy(graph); }
};
struct GraphExecDeleter {
  void operator()(cudaGraphExec_t executable) const { cudaGraphExecDestroy(executable); }
};

using DevicePtr = std::unique_ptr<void, DeviceDeleter>;
using PinnedHostPtr = std::unique_ptr<void, PinnedHostDeleter>;
using StreamPtr = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, StreamDeleter>;
using EventPtr = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, EventDeleter>;
using GraphPtr = std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, GraphDeleter>;
using GraphExecPtr = std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, GraphExecDeleter>;

DevicePtr DeviceBuffer(size_t bytes, const void* source = nullptr) {
  void* buffer = nullptr;
  CUDA_CALL_THROW(cudaMalloc(&buffer, bytes));
  DevicePtr result(buffer);
  if (source != nullptr) {
    CUDA_CALL_THROW(cudaMemcpy(buffer, source, bytes, cudaMemcpyHostToDevice));
  }
  return result;
}

PinnedHostPtr PinnedHostBuffer(size_t bytes) {
  void* buffer = nullptr;
  CUDA_CALL_THROW(cudaHostAlloc(&buffer, bytes, cudaHostAllocDefault));
  return PinnedHostPtr(buffer);
}

// Read-only KV cache storage. With `zero_copy`, the cache lives in mapped host memory that the kernel reads over
// PCIe. Those loads are slow and highly variable, which widens the window in which the CTAs of a sequence finish out
// of order. Devices that cannot map host memory fall back to device memory.
class CacheBuffer {
 public:
  CacheBuffer(size_t bytes, bool zero_copy) : bytes_(bytes) {
    if (zero_copy) {
      void* host = nullptr;
      void* device = nullptr;
      if (cudaHostAlloc(&host, bytes, cudaHostAllocMapped) == cudaSuccess) {
        host_.reset(host);
        if (cudaHostGetDevicePointer(&device, host, 0) == cudaSuccess) {
          device_pointer_ = device;
          return;
        }
        host_.reset();
      }
      (void)cudaGetLastError();
    }
    device_ = DeviceBuffer(bytes);
    device_pointer_ = device_.get();
  }

  void Upload(const void* source) {
    if (host_ != nullptr) {
      std::memcpy(host_.get(), source, bytes_);
    } else {
      CUDA_CALL_THROW(cudaMemcpy(device_.get(), source, bytes_, cudaMemcpyHostToDevice));
    }
  }

  const void* device_pointer() const { return device_pointer_; }

 private:
  size_t bytes_;
  PinnedHostPtr host_;
  DevicePtr device_;
  void* device_pointer_ = nullptr;
};

struct Config {
  int head_size = 64;
  int kv_heads = 24;   // with `batch`, the number of independent decode problems per launch
  int group_size = 1;  // query heads per KV head
  int batch = 1;
  int window = -1;  // sliding-window length, or -1 for global attention
  bool sinks = false;
  int repeats = 100;  // rounds per (split count, sequence length) pair

  int num_heads() const { return kv_heads * group_size; }
  size_t output_elements() const { return static_cast<size_t>(batch) * num_heads() * head_size; }
};

// 24 independent decode problems per launch, each reduced across several CTAs.
Config StressConfig() { return {}; }

Config GroupedQueryConfig() {
  Config config;
  config.head_size = 128;
  config.kv_heads = 6;
  config.group_size = 4;
  config.batch = 4;
  config.repeats = 30;
  return config;
}

// Head size 256 has a single GEMM1 warp group, so the CTA output is staged in a different shared-memory buffer than
// for head sizes 64 and 128.
Config HeadSize256Config() {
  Config config;
  config.head_size = 256;
  config.kv_heads = 12;
  config.group_size = 4;
  config.batch = 2;
  config.repeats = 30;
  return config;
}

// The short window leaves some sub-sequences without any token, and every head has its own attention sink.
Config SlidingWindowSinkConfig() {
  Config config;
  config.kv_heads = 12;
  config.group_size = 2;
  config.batch = 2;
  config.window = 300;
  config.sinks = true;
  config.repeats = 30;
  return config;
}

// A token's tile index modulo four selects the sub-sequence that owns it when four sub-sequences are used, so every
// partial result has a distinct mean and none equals the full attention result.
float ValueAt(int batch_index, int kv_head, int token, int channel) {
  return 1.0f + static_cast<float>((token / kTileTokens) % 4) + static_cast<float>(channel % 8) / 16.0f +
         0.125f * static_cast<float>((batch_index + kv_head) % 4);
}

std::vector<float> SinkValues(const Config& config) {
  std::vector<float> sinks;
  if (config.sinks) {
    for (int head = 0; head < config.num_heads(); ++head) {
      sinks.push_back(4.0f + 0.5f * static_cast<float>(head % 5));
    }
  }
  return sinks;
}

// Offset of a KV head in the cache layout [batch, kv_head, capacity, head_size].
size_t HeadCacheOffset(const Config& config, int batch_index, int kv_head) {
  return (static_cast<size_t>(batch_index) * config.kv_heads + kv_head) * kCapacity * config.head_size;
}

// Q and K are zero, so every logit is zero and the attention weights are uniform over the visible tokens. The output
// is the mean of V, damped by exp(sink) in the denominator when attention sinks are used.
template <typename T>
std::vector<double> ExpectedOutput(const Config& config, const std::vector<T>& value, int length,
                                   const std::vector<float>& sinks) {
  const int first_token = config.window > 0 ? std::max(0, length - config.window) : 0;
  const double visible_tokens = static_cast<double>(length - first_token);
  std::vector<double> expected(config.output_elements());
  std::vector<double> sums(config.head_size);
  for (int batch_index = 0; batch_index < config.batch; ++batch_index) {
    for (int kv_head = 0; kv_head < config.kv_heads; ++kv_head) {
      const size_t base = HeadCacheOffset(config, batch_index, kv_head);
      std::fill(sums.begin(), sums.end(), 0.0);
      for (int token = first_token; token < length; ++token) {
        for (int channel = 0; channel < config.head_size; ++channel) {
          sums[channel] += static_cast<float>(value[base + static_cast<size_t>(token) * config.head_size + channel]);
        }
      }
      for (int member = 0; member < config.group_size; ++member) {
        const int head = kv_head * config.group_size + member;
        const double denominator = visible_tokens + (sinks.empty() ? 0.0 : std::exp(static_cast<double>(sinks[head])));
        const size_t offset = (static_cast<size_t>(batch_index) * config.num_heads() + head) * config.head_size;
        for (int channel = 0; channel < config.head_size; ++channel) {
          expected[offset + channel] = sums[channel] / denominator;
        }
      }
    }
  }
  return expected;
}

std::string DescribeElement(const Config& config, size_t index) {
  const size_t head_size = static_cast<size_t>(config.head_size);
  const size_t num_heads = static_cast<size_t>(config.num_heads());
  return "batch " + std::to_string(index / (num_heads * head_size)) + ", head " +
         std::to_string(index / head_size % num_heads) + ", channel " + std::to_string(index % head_size);
}

template <typename T>
void VerifyOutput(const T* actual, const std::vector<double>& expected, const Config& config) {
  constexpr double tolerance = std::is_same_v<T, __nv_bfloat16> ? 0.03 : 0.003;
  for (size_t i = 0; i < expected.size(); ++i) {
    const float value = static_cast<float>(actual[i]);
    ASSERT_TRUE(std::isfinite(value)) << DescribeElement(config, i);
    ASSERT_NEAR(value, expected[i], tolerance) << DescribeElement(config, i);
  }
}

// Every captured launch must use the forced number of sub-sequences, otherwise the test silently stops covering the
// multi-block reduction.
void VerifyKernelNodes(cudaGraph_t graph, unsigned int splits, const Config& config) {
  size_t node_count = 0;
  CUDA_CALL_THROW(cudaGraphGetNodes(graph, nullptr, &node_count));
  std::vector<cudaGraphNode_t> nodes(node_count);
  CUDA_CALL_THROW(cudaGraphGetNodes(graph, nodes.data(), &node_count));
  int kernel_count = 0;
  for (auto node : nodes) {
    cudaGraphNodeType type;
    CUDA_CALL_THROW(cudaGraphNodeGetType(node, &type));
    if (type == cudaGraphNodeTypeKernel) {
      cudaKernelNodeParams params{};
      CUDA_CALL_THROW(cudaGraphKernelNodeGetParams(node, &params));
      EXPECT_EQ(params.gridDim.x, splits);
      EXPECT_EQ(params.gridDim.y, static_cast<unsigned int>(config.kv_heads));
      EXPECT_EQ(params.gridDim.z, static_cast<unsigned int>(config.batch));
      ++kernel_count;
    }
  }
  EXPECT_EQ(kernel_count, kNumStreams);
}

// Per-stream state. Every stream decodes its own problems with its own workspace and output buffers.
struct StreamResources {
  DevicePtr query;
  DevicePtr output;
  DevicePtr lengths;
  DevicePtr workspace;
  PinnedHostPtr host_output;
};

class XqaMultiBlockTest : public ::testing::Test {
 protected:
  XqaMultiBlockTest() = default;

  void SetUp() override {
#ifdef _WIN32
    char* original_splits = nullptr;
    size_t original_splits_size = 0;
    ASSERT_EQ(_dupenv_s(&original_splits, &original_splits_size, "XQA_NB_SUB_SEQ"), 0);
    std::unique_ptr<char, decltype(&std::free)> original_splits_owner(original_splits, &std::free);
    original_splits_ = original_splits == nullptr ? "" : original_splits;
#endif
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA device unavailable";
    }
    int device = 0;
    CUDA_CALL_THROW(cudaGetDevice(&device));
    CUDA_CALL_THROW(cudaGetDeviceProperties(&properties_, device));
    if (properties_.major < 8 || properties_.multiProcessorCount < 4) {
      GTEST_SKIP() << "XQA multi-block regression requires SM80+ with at least four SMs";
    }
    for (int i = 0; i < kNumStreams; ++i) {
      cudaStream_t stream = nullptr;
      CUDA_CALL_THROW(cudaStreamCreate(&stream));
      streams_.emplace_back(stream);
    }
    fork_event_ = MakeEvent();
    for (int i = 1; i < kNumStreams; ++i) {
      join_events_.push_back(MakeEvent());
    }
  }

  void TearDown() override {
#ifdef _WIN32
    EXPECT_EQ(_putenv_s("XQA_NB_SUB_SEQ", original_splits_.c_str()), 0);
#endif
  }

  static EventPtr MakeEvent() {
    cudaEvent_t event = nullptr;
    CUDA_CALL_THROW(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return EventPtr(event);
  }

  // Records the work of all streams into one graph. The first stream forks the others and joins them again, which
  // makes the graph run the problems of all streams concurrently.
  template <typename EnqueueFn>
  GraphPtr CaptureRound(EnqueueFn&& enqueue) {
    cudaStream_t origin = streams_[0].get();
    CUDA_CALL_THROW(cudaStreamBeginCapture(origin, cudaStreamCaptureModeThreadLocal));
    try {
      CUDA_CALL_THROW(cudaEventRecord(fork_event_.get(), origin));
      for (int i = 1; i < kNumStreams; ++i) {
        CUDA_CALL_THROW(cudaStreamWaitEvent(streams_[i].get(), fork_event_.get(), 0));
      }
      for (int i = 0; i < kNumStreams; ++i) {
        enqueue(i);
      }
      for (int i = 1; i < kNumStreams; ++i) {
        CUDA_CALL_THROW(cudaEventRecord(join_events_[i - 1].get(), streams_[i].get()));
        CUDA_CALL_THROW(cudaStreamWaitEvent(origin, join_events_[i - 1].get(), 0));
      }
    } catch (...) {
      // Do not leave the stream in capture mode, which would fail every later CUDA call of the process.
      cudaGraph_t partial = nullptr;
      if (cudaStreamEndCapture(origin, &partial) == cudaSuccess) {
        GraphPtr discard(partial);
      }
      throw;
    }
    cudaGraph_t graph = nullptr;
    CUDA_CALL_THROW(cudaStreamEndCapture(origin, &graph));
    return GraphPtr(graph);
  }

  template <typename T>
  void Run(const Config& config, bool use_graph) {
    const int num_heads = config.num_heads();
    const size_t shared_memory =
        GetXQARequiredSharedMemoryBytes(properties_, config.head_size, num_heads, config.kv_heads);
    if (shared_memory > properties_.sharedMemPerBlockOptin) {
      GTEST_SKIP() << "XQA kernel shared memory exceeds the device limit";
    }

    const size_t cache_elements =
        static_cast<size_t>(config.batch) * config.kv_heads * kCapacity * static_cast<size_t>(config.head_size);
    const size_t cache_bytes = cache_elements * sizeof(T);
    std::vector<T> key(cache_elements, static_cast<T>(0.0f));
    std::vector<T> value(cache_elements);
    for (int batch_index = 0; batch_index < config.batch; ++batch_index) {
      for (int kv_head = 0; kv_head < config.kv_heads; ++kv_head) {
        const size_t base = HeadCacheOffset(config, batch_index, kv_head);
        for (int token = 0; token < kCapacity; ++token) {
          for (int channel = 0; channel < config.head_size; ++channel) {
            value[base + static_cast<size_t>(token) * config.head_size + channel] =
                static_cast<T>(ValueAt(batch_index, kv_head, token, channel));
          }
        }
      }
    }
    const bool zero_copy = properties_.canMapHostMemory != 0;
    CacheBuffer device_key(cache_bytes, zero_copy);
    CacheBuffer device_value(cache_bytes, zero_copy);
    device_key.Upload(key.data());
    device_value.Upload(value.data());

    const std::vector<float> sinks = SinkValues(config);
    DevicePtr device_sinks;
    if (!sinks.empty()) {
      device_sinks = DeviceBuffer(sinks.size() * sizeof(float), sinks.data());
    }

    // GetXQAScratchSize derives the number of sub-sequences from the SM count and ignores XQA_NB_SUB_SEQ, so size the
    // workspace for a device that has exactly the SM count that yields each forced split count.
    size_t workspace_bytes = 0;
    for (int splits : kSplits) {
      cudaDeviceProp forced_properties = properties_;
      forced_properties.multiProcessorCount = splits * config.batch * config.kv_heads;
      workspace_bytes = std::max(
          workspace_bytes, GetXQAScratchSize(forced_properties, config.batch, num_heads, config.kv_heads,
                                             config.head_size, kCapacity, XqaQuantType::kNone,
                                             std::is_same_v<T, __nv_bfloat16>));
    }
    const size_t output_bytes = config.output_elements() * sizeof(T);
    std::vector<StreamResources> resources(kNumStreams);
    for (auto& stream_resources : resources) {
      const std::vector<T> query(config.output_elements(), static_cast<T>(0.0f));
      stream_resources.query = DeviceBuffer(output_bytes, query.data());
      stream_resources.output = DeviceBuffer(output_bytes);
      stream_resources.lengths = DeviceBuffer(static_cast<size_t>(config.batch) * sizeof(int));
      stream_resources.workspace = DeviceBuffer(workspace_bytes);
      stream_resources.host_output = PinnedHostBuffer(output_bytes);
    }
    CUDA_CALL_THROW(cudaDeviceSynchronize());

    const auto enqueue = [&](int stream_index) {
      StreamResources& stream_resources = resources[stream_index];
      cudaStream_t stream = streams_[stream_index].get();
      CUDA_CALL_THROW(cudaMemsetAsync(stream_resources.output.get(), 0xff, output_bytes, stream));
      ORT_THROW_IF_ERROR(LaunchXQAKernel<T>(
          properties_, stream, stream_resources.query.get(), device_key.device_pointer(),
          device_value.device_pointer(), stream_resources.output.get(), config.batch, num_heads, config.kv_heads,
          config.head_size, kCapacity, kScale, config.window, false,
          static_cast<const int*>(stream_resources.lengths.get()), static_cast<const float*>(device_sinks.get()),
          nullptr, nullptr, XqaQuantType::kNone, stream_resources.workspace.get(), workspace_bytes));
      CUDA_CALL_THROW(cudaMemcpyAsync(stream_resources.host_output.get(), stream_resources.output.get(), output_bytes,
                                      cudaMemcpyDeviceToHost, stream));
    };

    for (int splits : kSplits) {
      SCOPED_TRACE("splits " + std::to_string(splits));
      onnxruntime::test::ScopedEnvironmentVariables scoped_env_vars{{{"XQA_NB_SUB_SEQ", std::to_string(splits)}}};
#ifdef _WIN32
      // XQA uses getenv, whose CRT environment is separate from the Win32 environment.
      ASSERT_EQ(_putenv_s("XQA_NB_SUB_SEQ", std::to_string(splits).c_str()), 0);
#endif
      const auto set_length = [&](int length) {
        const std::vector<int> past_lengths(config.batch, length - 1);
        for (auto& stream_resources : resources) {
          CUDA_CALL_THROW(cudaMemcpy(stream_resources.lengths.get(), past_lengths.data(),
                                     past_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
        }
      };

      // The first launch of a kernel configures it, which is not allowed while capturing.
      set_length(kLengths[0]);
      for (int i = 0; i < kNumStreams; ++i) {
        enqueue(i);
      }
      CUDA_CALL_THROW(cudaDeviceSynchronize());

      // Capture one round for every variant to check its grid. Only the graph variants run the captured graph.
      GraphPtr graph = CaptureRound(enqueue);
      ASSERT_NO_FATAL_FAILURE(VerifyKernelNodes(graph.get(), static_cast<unsigned int>(splits), config));
      GraphExecPtr executable;
      if (use_graph) {
        cudaGraphExec_t instantiated = nullptr;
        CUDA_CALL_THROW(cudaGraphInstantiate(&instantiated, graph.get(), nullptr, nullptr, 0));
        executable.reset(instantiated);
      }
      graph.reset();

      for (int length : kLengths) {
        SCOPED_TRACE("length " + std::to_string(length));
        set_length(length);
        const std::vector<double> expected = ExpectedOutput(config, value, length, sinks);
        std::vector<T> first_output;
        for (int repeat = 0; repeat < config.repeats; ++repeat) {
          SCOPED_TRACE("repeat " + std::to_string(repeat));
          if (use_graph) {
            CUDA_CALL_THROW(cudaGraphLaunch(executable.get(), streams_[0].get()));
            CUDA_CALL_THROW(cudaStreamSynchronize(streams_[0].get()));
          } else {
            for (int i = 0; i < kNumStreams; ++i) {
              enqueue(i);
            }
            for (int i = 0; i < kNumStreams; ++i) {
              CUDA_CALL_THROW(cudaStreamSynchronize(streams_[i].get()));
            }
          }
          for (int i = 0; i < kNumStreams; ++i) {
            SCOPED_TRACE("stream " + std::to_string(i));
            const T* actual = static_cast<const T*>(resources[i].host_output.get());
            ASSERT_NO_FATAL_FAILURE(VerifyOutput(actual, expected, config));
            if (first_output.empty()) {
              first_output.assign(actual, actual + config.output_elements());
            } else {
              ASSERT_EQ(std::memcmp(actual, first_output.data(), output_bytes), 0);
            }
          }
        }
      }
    }
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(XqaMultiBlockTest);

  cudaDeviceProp properties_{};
  std::vector<StreamPtr> streams_;
  EventPtr fork_event_;
  std::vector<EventPtr> join_events_;
#ifdef _WIN32
  std::string original_splits_;
#endif
};

TEST_F(XqaMultiBlockTest, Float16) { Run<half>(StressConfig(), false); }
TEST_F(XqaMultiBlockTest, Float16GraphReplay) { Run<half>(StressConfig(), true); }
TEST_F(XqaMultiBlockTest, BFloat16) { Run<__nv_bfloat16>(StressConfig(), false); }
TEST_F(XqaMultiBlockTest, BFloat16GraphReplay) { Run<__nv_bfloat16>(StressConfig(), true); }
TEST_F(XqaMultiBlockTest, Float16GroupedQueryHeadSize128) { Run<half>(GroupedQueryConfig(), false); }
TEST_F(XqaMultiBlockTest, BFloat16GroupedQueryHeadSize128) { Run<__nv_bfloat16>(GroupedQueryConfig(), false); }
TEST_F(XqaMultiBlockTest, Float16HeadSize256) { Run<half>(HeadSize256Config(), false); }
TEST_F(XqaMultiBlockTest, BFloat16HeadSize256GraphReplay) { Run<__nv_bfloat16>(HeadSize256Config(), true); }
TEST_F(XqaMultiBlockTest, Float16SlidingWindowAttentionSinks) { Run<half>(SlidingWindowSinkConfig(), true); }
TEST_F(XqaMultiBlockTest, BFloat16SlidingWindowAttentionSinks) {
  Run<__nv_bfloat16>(SlidingWindowSinkConfig(), true);
}

}  // namespace
}  // namespace onnxruntime::contrib::cuda::test
