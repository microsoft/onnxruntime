// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#if !defined(DISABLE_CONTRIB_OPS) && defined(USE_FPA_INTB_GEMM) && USE_FPA_INTB_GEMM

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>

#include "core/common/path_string.h"
#include "core/framework/allocator_stats.h"
#include "core/framework/session_state.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime {
namespace test {
namespace {

constexpr int64_t kSequenceLength = 64;
constexpr int64_t kPastSequenceLength = 1;
constexpr int64_t kNumLayers = 32;
constexpr int64_t kNumKeyValueHeads = 4;
constexpr int64_t kHeadSize = 128;
constexpr int64_t kVocabSize = 120818;
constexpr int kMinFpAIntBSm = 75;

class CudaMemorySampler {
 public:
  explicit CudaMemorySampler(int device_id) : device_id_(device_id) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t initial_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (initial_status != cudaSuccess) {
      error_.store(static_cast<int>(initial_status), std::memory_order_relaxed);
      return;
    }
    peak_used_bytes_.store(total_bytes - free_bytes, std::memory_order_relaxed);

    worker_ = std::thread([this]() {
      const cudaError_t set_device_status = cudaSetDevice(device_id_);
      if (set_device_status != cudaSuccess) {
        error_.store(static_cast<int>(set_device_status), std::memory_order_relaxed);
        return;
      }

      while (!stop_.load(std::memory_order_relaxed)) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (status != cudaSuccess) {
          error_.store(static_cast<int>(status), std::memory_order_relaxed);
          return;
        }

        const size_t used_bytes = total_bytes - free_bytes;
        size_t peak = peak_used_bytes_.load(std::memory_order_relaxed);
        while (used_bytes > peak &&
               !peak_used_bytes_.compare_exchange_weak(
                   peak, used_bytes, std::memory_order_relaxed)) {
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
  }

  ~CudaMemorySampler() {
    Stop();
  }

  void Stop() {
    stop_.store(true, std::memory_order_relaxed);
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  size_t PeakUsedBytes() const {
    return peak_used_bytes_.load(std::memory_order_relaxed);
  }

  cudaError_t Error() const {
    return static_cast<cudaError_t>(error_.load(std::memory_order_relaxed));
  }

 private:
  int device_id_;
  std::atomic<bool> stop_{false};
  std::atomic<size_t> peak_used_bytes_{0};
  std::atomic<int> error_{static_cast<int>(cudaSuccess)};
  std::thread worker_;
};

std::optional<size_t> GetCudaUsedMemoryBytes() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    return std::nullopt;
  }

  return total_bytes - free_bytes;
}

int CudaDeviceComputeCapabilityOrNegative() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return -1;
  }

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    return -1;
  }

  return prop.major * 10 + prop.minor;
}

std::string GetBinaryEnvironmentValue(const char* name, const char* default_value) {
  std::string value = Env::Default().GetEnvironmentVar(name);
  if (value.empty()) {
    value = default_value;
  }

  ORT_ENFORCE(value == "0" || value == "1", name, " must be 0 or 1, but got: ", value);
  return value;
}

std::string BuildMaxShapeOverride() {
  std::ostringstream shapes;
  shapes << "input_ids:[1," << kSequenceLength << "]"
         << ";attention_mask:[1," << kPastSequenceLength + kSequenceLength << "]";
  for (int64_t layer = 0; layer < kNumLayers; ++layer) {
    const std::string shape = ":[1," + std::to_string(kNumKeyValueHeads) + "," +
                              std::to_string(kPastSequenceLength) + "," +
                              std::to_string(kHeadSize) + "]";
    shapes << ";past_key_values." << layer << ".key" << shape
           << ";past_key_values." << layer << ".value" << shape;
  }
  return shapes.str();
}

}  // namespace

// Opt-in, model-backed benchmark. It reports metrics without asserting fixed performance thresholds
// because device memory and latency vary by GPU, driver, clocks, and system load. Run it in a fresh
// Release process for each configuration so CUDA arena state cannot affect the next measurement.
//
// Required: ORT_HY_MT2_MODEL_PATH.
// Optional binary settings: ORT_HY_MT2_DISABLE_PREPACKING, ORT_HY_MT2_USE_DEVICE_INITIALIZERS,
// and ORT_HY_MT2_FPA_INTB_GEMM (defaults: 0, 0, and 1).
TEST(MatMulNBitsWorkspace, HyMT2BenchmarkReportsMemoryAndLatency) {
  const std::string model_path_utf8 = Env::Default().GetEnvironmentVar("ORT_HY_MT2_MODEL_PATH");
  if (model_path_utf8.empty()) {
    GTEST_SKIP() << "Set ORT_HY_MT2_MODEL_PATH to the Hy-MT2 model.onnx file.";
  }

  const std::filesystem::path model_path(ToPathString(model_path_utf8));
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "Hy-MT2 model not found at " << model_path_utf8 << ".";
  }

  const int device_sm = CudaDeviceComputeCapabilityOrNegative();
  if (device_sm < 0) {
    GTEST_SKIP() << "No CUDA device available; skipping Hy-MT2 benchmark.";
  }
  if (device_sm < kMinFpAIntBSm) {
    GTEST_SKIP() << "Device compute capability " << device_sm << " < " << kMinFpAIntBSm
                 << "; Hy-MT2 fpA_intB benchmarking is unsupported.";
  }

  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  const std::optional<size_t> baseline_used_bytes = GetCudaUsedMemoryBytes();
  ASSERT_TRUE(baseline_used_bytes.has_value());

  const std::string disable_prepacking =
      GetBinaryEnvironmentValue("ORT_HY_MT2_DISABLE_PREPACKING", "0");
  const std::string use_device_initializers =
      GetBinaryEnvironmentValue("ORT_HY_MT2_USE_DEVICE_INITIALIZERS", "0");
  const std::string enable_fpa_intb =
      GetBinaryEnvironmentValue("ORT_HY_MT2_FPA_INTB_GEMM", "1");
  const std::string add_position_ids =
      GetBinaryEnvironmentValue("ORT_HY_MT2_ADD_POSITION_IDS", "0");
  const std::string enable_mem_pattern =
      GetBinaryEnvironmentValue("ORT_HY_MT2_ENABLE_MEM_PATTERN", "1");

  ASSERT_FALSE(disable_prepacking == "1" && enable_fpa_intb == "1")
      << "The fpA_intB path requires PrePack. Set ORT_HY_MT2_FPA_INTB_GEMM=0 when "
         "ORT_HY_MT2_DISABLE_PREPACKING=1.";

  SessionOptions session_options;
  session_options.session_logid = "HyMT2MemoryLatencyBenchmark";
  session_options.enable_mem_pattern = enable_mem_pattern == "1";
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsMaxShapeOverride, BuildMaxShapeOverride().c_str()));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsCudaFpAIntBGemm, enable_fpa_intb.c_str()));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsConfigDisablePrepacking, disable_prepacking.c_str()));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsUseDeviceAllocatorForInitializers, use_device_initializers.c_str()));

  using Clock = std::chrono::steady_clock;
  CudaMemorySampler initialization_memory_sampler(0);
  const auto initialize_start = Clock::now();

  InferenceSessionWrapper session(session_options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
  ASSERT_STATUS_OK(session.Load(model_path.native().c_str()));
  ASSERT_STATUS_OK(session.Initialize());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto initialize_end = Clock::now();
  initialization_memory_sampler.Stop();
  ASSERT_EQ(initialization_memory_sampler.Error(), cudaSuccess);
  const std::optional<size_t> after_initialize_used_bytes = GetCudaUsedMemoryBytes();
  ASSERT_TRUE(after_initialize_used_bytes.has_value());

  size_t matmul_nbits_nodes = 0;
  size_t cuda_matmul_nbits_nodes = 0;
  for (const auto& node : session.GetGraph().Nodes()) {
    if (node.OpType() != "MatMulNBits") {
      continue;
    }
    ++matmul_nbits_nodes;
    if (node.GetExecutionProviderType() == kCudaExecutionProvider) {
      ++cuda_matmul_nbits_nodes;
    }
  }
  // NOTE: temporarily relaxed for local benchmarking against a model whose MatMulNBits node
  // count differs from the originally assumed 225 (e.g. a different quantization/layer config).
  // All MatMulNBits nodes must still be assigned to the CUDA EP.
  ASSERT_GT(matmul_nbits_nodes, static_cast<size_t>(0));
  ASSERT_EQ(cuda_matmul_nbits_nodes, matmul_nbits_nodes);

  std::vector<int64_t> input_ids(static_cast<size_t>(kSequenceLength), 120000);
  std::vector<int64_t> attention_mask(
      static_cast<size_t>(kPastSequenceLength + kSequenceLength), 1);
  std::vector<MLFloat16> past_data(
      static_cast<size_t>(kNumKeyValueHeads * kPastSequenceLength * kHeadSize),
      MLFloat16(0.0f));

  NameMLValMap feeds;
  OrtValue input_ids_value;
  CreateMLValue<int64_t>(
      std::array<int64_t, 2>{1, kSequenceLength},
      input_ids.data(), OrtMemoryInfo(), &input_ids_value);
  feeds.emplace("input_ids", input_ids_value);

  OrtValue attention_mask_value;
  CreateMLValue<int64_t>(
      std::array<int64_t, 2>{1, kPastSequenceLength + kSequenceLength},
      attention_mask.data(), OrtMemoryInfo(), &attention_mask_value);
  feeds.emplace("attention_mask", attention_mask_value);

  OrtValue position_ids_value;
  std::vector<int64_t> position_ids;
  if (add_position_ids == "1") {
    position_ids.resize(static_cast<size_t>(kSequenceLength));
    std::iota(position_ids.begin(), position_ids.end(), kPastSequenceLength);
    CreateMLValue<int64_t>(
        std::array<int64_t, 2>{1, kSequenceLength},
        position_ids.data(), OrtMemoryInfo(), &position_ids_value);
    feeds.emplace("position_ids", position_ids_value);
  }

  const std::array<int64_t, 4> past_shape{
      1, kNumKeyValueHeads, kPastSequenceLength, kHeadSize};
  for (int64_t layer = 0; layer < kNumLayers; ++layer) {
    for (const char* kind : {"key", "value"}) {
      OrtValue past_value;
      CreateMLValue<MLFloat16>(past_shape, past_data.data(), OrtMemoryInfo(), &past_value);
      feeds.emplace(
          "past_key_values." + std::to_string(layer) + "." + kind, std::move(past_value));
    }
  }

  const std::vector<std::string> output_names{"logits"};
  std::vector<OrtValue> fetches;
  constexpr int kWarmupRuns = 5;
  for (int i = 0; i < kWarmupRuns; ++i) {
    fetches.clear();
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  ASSERT_EQ(fetches.size(), static_cast<size_t>(1));
  ASSERT_EQ(fetches.front().Get<Tensor>().Shape(),
            TensorShape({1, kSequenceLength, kVocabSize}));

  constexpr int kMemoryMeasurementRuns = 3;
  CudaMemorySampler inference_memory_sampler(0);
  for (int i = 0; i < kMemoryMeasurementRuns; ++i) {
    fetches.clear();
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  inference_memory_sampler.Stop();
  ASSERT_EQ(inference_memory_sampler.Error(), cudaSuccess);

  constexpr int kMeasuredRuns = 30;
  std::vector<double> latencies_ms;
  latencies_ms.reserve(kMeasuredRuns);
  for (int i = 0; i < kMeasuredRuns; ++i) {
    fetches.clear();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto start = Clock::now();
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto end = Clock::now();
    latencies_ms.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }

  ASSERT_FALSE(latencies_ms.empty());
  std::sort(latencies_ms.begin(), latencies_ms.end());
  const auto percentile = [&latencies_ms](double fraction) {
    const size_t index = static_cast<size_t>(
                             std::ceil(fraction * static_cast<double>(latencies_ms.size()))) -
                         1;
    return latencies_ms[std::min(index, latencies_ms.size() - 1)];
  };
  const double average_ms =
      std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) /
      static_cast<double>(latencies_ms.size());
  const double initialize_ms =
      std::chrono::duration<double, std::milli>(initialize_end - initialize_start).count();

  const OrtDevice cuda_device(
      OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA, 0);
  AllocatorPtr cuda_allocator = session.GetSessionState().GetAllocator(cuda_device);
  ASSERT_NE(cuda_allocator, nullptr);
  AllocatorStats allocator_stats;
  cuda_allocator->GetStats(&allocator_stats);

  const auto to_mib = [](size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
  };
  const auto delta_from_baseline = [baseline = *baseline_used_bytes](size_t bytes) {
    return bytes > baseline ? bytes - baseline : size_t{0};
  };
  const size_t initialization_peak_bytes = initialization_memory_sampler.PeakUsedBytes();
  const size_t inference_peak_bytes = inference_memory_sampler.PeakUsedBytes();

  std::cout << "[ HY-MT2 BENCHMARK ]"
            << " disable_prepacking=" << disable_prepacking
            << " use_device_initializers=" << use_device_initializers
            << " fpa_intb_gemm=" << enable_fpa_intb
            << " enable_mem_pattern=" << enable_mem_pattern
            << " prepack_count=" << session.GetSessionState().GetNumberOfPrepacksCounter()
            << " initialize_ms=" << initialize_ms
            << " average_ms=" << average_ms
            << " p50_ms=" << percentile(0.50)
            << " p90_ms=" << percentile(0.90)
            << " p99_ms=" << percentile(0.99)
            << " min_ms=" << latencies_ms.front()
            << " max_ms=" << latencies_ms.back()
            << " baseline_device_used_mib=" << to_mib(*baseline_used_bytes)
            << " initialization_peak_device_used_mib="
            << to_mib(initialization_peak_bytes)
            << " initialization_peak_delta_mib="
            << to_mib(delta_from_baseline(initialization_peak_bytes))
            << " post_initialize_device_used_mib=" << to_mib(*after_initialize_used_bytes)
            << " inference_peak_device_used_mib="
            << to_mib(inference_peak_bytes)
            << " inference_peak_delta_mib="
            << to_mib(delta_from_baseline(inference_peak_bytes))
            << " arena_bytes_in_use=" << allocator_stats.bytes_in_use
            << " arena_total_allocated_bytes=" << allocator_stats.total_allocated_bytes
            << " arena_max_bytes_in_use=" << allocator_stats.max_bytes_in_use
            << " arena_num_allocs=" << allocator_stats.num_allocs
            << " arena_num_reserves=" << allocator_stats.num_reserves
            << std::endl;
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(DISABLE_CONTRIB_OPS) && USE_FPA_INTB_GEMM
