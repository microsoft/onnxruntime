// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(_WIN32) && defined(USE_WEBGPU) && !defined(DISABLE_CONTRIB_OPS)

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <dxgi1_4.h>
#include <wrl/client.h>

#pragma comment(lib, "dxgi.lib")

#include "core/common/path_string.h"
#include "core/framework/allocator.h"
#include "core/framework/allocator_stats.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/framework/session_state.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime {
namespace test {
namespace {

using Microsoft::WRL::ComPtr;

constexpr int64_t kSequenceLength = 1024;
constexpr int64_t kPastSequenceLength = 1;
constexpr int64_t kNumLayers = 28;
constexpr int64_t kNumKeyValueHeads = 2;
constexpr int64_t kHeadSize = 128;
constexpr int64_t kVocabSize = 151936;
constexpr int64_t kBosTokenId = 151643;

std::string BuildMaxShapeOverride() {
  std::ostringstream shapes;
  shapes << "input_ids:[1," << kSequenceLength << "]"
         << ";attention_mask:[1," << kPastSequenceLength + kSequenceLength << "]";
  const std::string cache_shape =
      ":[1," + std::to_string(kNumKeyValueHeads) + "," +
      std::to_string(kPastSequenceLength) + "," + std::to_string(kHeadSize) + "]";
  for (int64_t layer = 0; layer < kNumLayers; ++layer) {
    shapes << ";past_key_values." << layer << ".key" << cache_shape
           << ";past_key_values." << layer << ".value" << cache_shape;
  }

  return shapes.str();
}

std::string GetBinaryEnvironmentValue(const char* name, const char* default_value) {
  std::string value = Env::Default().GetEnvironmentVar(name);
  if (value.empty()) {
    value = default_value;
  }

  ORT_ENFORCE(value == "0" || value == "1", name, " must be 0 or 1, but got: ", value);
  return value;
}

ComPtr<IDXGIAdapter3> GetDxgiAdapterForWebGpu() {
  const auto& adapter_info = webgpu::WebGpuContextFactory::DefaultContext().AdapterInfo();
  ComPtr<IDXGIFactory1> factory;
  ORT_ENFORCE(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))),
              "Failed to create a DXGI factory for WDDM memory sampling.");

  std::vector<ComPtr<IDXGIAdapter3>> matches;
  for (UINT adapter_index = 0;; ++adapter_index) {
    ComPtr<IDXGIAdapter1> adapter;
    const HRESULT enum_result = factory->EnumAdapters1(adapter_index, &adapter);
    if (enum_result == DXGI_ERROR_NOT_FOUND) {
      break;
    }
    ORT_ENFORCE(SUCCEEDED(enum_result), "Failed to enumerate DXGI adapter ", adapter_index,
                " for WDDM memory sampling. HRESULT=", enum_result);

    DXGI_ADAPTER_DESC1 description{};
    ORT_ENFORCE(SUCCEEDED(adapter->GetDesc1(&description)),
                "Failed to query DXGI adapter ", adapter_index, ".");
    if (description.VendorId != adapter_info.vendorID ||
        (adapter_info.deviceID != 0 && description.DeviceId != adapter_info.deviceID)) {
      continue;
    }

    ComPtr<IDXGIAdapter3> adapter3;
    ORT_ENFORCE(SUCCEEDED(adapter.As(&adapter3)),
                "The WebGPU adapter does not expose IDXGIAdapter3.");
    matches.push_back(std::move(adapter3));
  }

  ORT_ENFORCE(matches.size() == 1,
              "Expected exactly one DXGI adapter matching WebGPU vendor ID ",
              adapter_info.vendorID, " and device ID ", adapter_info.deviceID,
              ", but found ", matches.size(), ".");
  return std::move(matches.front());
}

size_t GetWddmLocalUsageBytes(IDXGIAdapter3* adapter) {
  ORT_ENFORCE(adapter != nullptr);
  DXGI_QUERY_VIDEO_MEMORY_INFO memory_info{};
  const HRESULT result = adapter->QueryVideoMemoryInfo(
      0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory_info);
  ORT_ENFORCE(SUCCEEDED(result),
              "IDXGIAdapter3::QueryVideoMemoryInfo failed. HRESULT=", result);
  return static_cast<size_t>(memory_info.CurrentUsage);
}

class WddmMemorySampler {
 public:
  explicit WddmMemorySampler(ComPtr<IDXGIAdapter3> adapter)
      : adapter_(std::move(adapter)) {
    ORT_ENFORCE(adapter_ != nullptr);
    peak_used_bytes_.store(
        GetWddmLocalUsageBytes(adapter_.Get()), std::memory_order_relaxed);

    worker_ = std::thread([this]() {
      while (!stop_.load(std::memory_order_relaxed)) {
        DXGI_QUERY_VIDEO_MEMORY_INFO memory_info{};
        const HRESULT result = adapter_->QueryVideoMemoryInfo(
            0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory_info);
        if (FAILED(result)) {
          error_.store(result, std::memory_order_relaxed);
          return;
        }

        const size_t used_bytes = static_cast<size_t>(memory_info.CurrentUsage);
        size_t peak = peak_used_bytes_.load(std::memory_order_relaxed);
        while (used_bytes > peak &&
               !peak_used_bytes_.compare_exchange_weak(
                   peak, used_bytes, std::memory_order_relaxed)) {
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
  }

  ~WddmMemorySampler() {
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

  HRESULT Error() const {
    return error_.load(std::memory_order_relaxed);
  }

 private:
  ComPtr<IDXGIAdapter3> adapter_;
  std::atomic<bool> stop_{false};
  std::atomic<size_t> peak_used_bytes_{0};
  std::atomic<HRESULT> error_{S_OK};
  std::thread worker_;
};

}  // namespace

// Run baseline and planned configurations in separate processes by setting
// ORT_WEBGPU_WORKSPACE_BENCHMARK_PREALLOCATION to 0 or 1.
TEST(MatMulNBitsWorkspace, WebGpuQwen25WorkspacePreallocationBenchmark) {
  std::string model_path_utf8 =
      Env::Default().GetEnvironmentVar("ORT_WEBGPU_WORKSPACE_BENCHMARK_MODEL_PATH");
  if (model_path_utf8.empty()) {
    model_path_utf8 =
        "C:\\Users\\lochi\\.foundry\\cache\\models\\Microsoft\\"
        "qwen2.5-1.5b-instruct-cuda-gpu-4\\v4\\model.onnx";
  }

  const std::filesystem::path model_path(ToPathString(model_path_utf8));
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "Qwen 2.5 1.5B model not found at " << model_path_utf8
                 << "; set ORT_WEBGPU_WORKSPACE_BENCHMARK_MODEL_PATH.";
  }

  const std::string enable_workspace_preallocation =
      GetBinaryEnvironmentValue("ORT_WEBGPU_WORKSPACE_BENCHMARK_PREALLOCATION", "0");

  SessionOptions session_options;
  session_options.session_logid = "WebGpuQwen25WorkspacePreallocationBenchmark";
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsMaxShapeOverride, BuildMaxShapeOverride().c_str()));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsEnableStaticWorkspacePreallocation,
      enable_workspace_preallocation.c_str()));

  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  ComPtr<IDXGIAdapter3> dxgi_adapter = GetDxgiAdapterForWebGpu();
  const size_t wddm_baseline_local_bytes = GetWddmLocalUsageBytes(dxgi_adapter.Get());

  using Clock = std::chrono::steady_clock;
  WddmMemorySampler initialization_wddm_sampler(dxgi_adapter);
  const auto initialize_start = Clock::now();

  InferenceSessionWrapper session(session_options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(webgpu_ep)));
  ASSERT_STATUS_OK(session.Load(model_path.native().c_str()));
  ASSERT_STATUS_OK(session.Initialize());

  const auto initialize_end = Clock::now();
  initialization_wddm_sampler.Stop();
  ASSERT_TRUE(SUCCEEDED(initialization_wddm_sampler.Error()));
  const size_t wddm_after_initialize_local_bytes =
      GetWddmLocalUsageBytes(dxgi_adapter.Get());

  AllocatorPtr webgpu_allocator =
      session.GetSessionState().GetAllocator(webgpu::WebGpuDevice);
  ASSERT_NE(webgpu_allocator, nullptr);
  AllocatorStats post_initialize_stats;
  webgpu_allocator->GetStats(&post_initialize_stats);

  size_t matmul_nbits_nodes = 0;
  size_t webgpu_matmul_nbits_nodes = 0;
  for (const auto& node : session.GetGraph().Nodes()) {
    if (node.OpType() != "MatMulNBits") {
      continue;
    }

    ++matmul_nbits_nodes;
    if (node.GetExecutionProviderType() == kWebGpuExecutionProvider) {
      ++webgpu_matmul_nbits_nodes;
    }
  }
  ASSERT_GT(matmul_nbits_nodes, static_cast<size_t>(0));
  ASSERT_EQ(webgpu_matmul_nbits_nodes, matmul_nbits_nodes);

  const SequentialExecutionPlan* execution_plan =
      session.GetSessionState().GetExecutionPlan();
  ASSERT_NE(execution_plan, nullptr);
  size_t planned_workspace_nodes = 0;
  size_t planned_workspace_slots = 0;
  size_t largest_workspace_bytes = 0;
  size_t aggregate_workspace_bytes = 0;
  for (const auto& [node_index, workspace_plans] :
       execution_plan->workspace_allocation_plan) {
    ORT_UNUSED_PARAMETER(node_index);
    ++planned_workspace_nodes;
    planned_workspace_slots += workspace_plans.size();
    for (const auto& workspace_plan : workspace_plans) {
      largest_workspace_bytes =
          std::max(largest_workspace_bytes, workspace_plan.size_bytes);
      aggregate_workspace_bytes += workspace_plan.size_bytes;
    }
  }
  if (enable_workspace_preallocation == "1") {
    ASSERT_GT(planned_workspace_nodes, static_cast<size_t>(0));
  } else {
    ASSERT_EQ(planned_workspace_nodes, static_cast<size_t>(0));
  }

  std::vector<int64_t> input_ids(
      static_cast<size_t>(kSequenceLength), kBosTokenId);
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

  const std::array<int64_t, 4> past_shape{
      1, kNumKeyValueHeads, kPastSequenceLength, kHeadSize};
  for (int64_t layer = 0; layer < kNumLayers; ++layer) {
    for (const char* kind : {"key", "value"}) {
      OrtValue past_value;
      CreateMLValue<MLFloat16>(
          past_shape, past_data.data(), OrtMemoryInfo(), &past_value);
      feeds.emplace(
          "past_key_values." + std::to_string(layer) + "." + kind,
          std::move(past_value));
    }
  }

  const std::vector<std::string> output_names{"logits"};
  std::vector<OrtValue> fetches;
  constexpr int kWarmupRuns = 5;
  for (int i = 0; i < kWarmupRuns; ++i) {
    fetches.clear();
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
  }
  ASSERT_EQ(fetches.size(), static_cast<size_t>(1));
  ASSERT_EQ(fetches.front().Get<Tensor>().Shape(),
            TensorShape({1, kSequenceLength, kVocabSize}));
  fetches.clear();

  size_t workspace_pattern_peak_bytes = 0;
  if (const MemoryPatternGroup* workspace_patterns =
          session.GetSessionState().GetWorkspaceMemoryPatternGroup()) {
    for (const auto& pattern : workspace_patterns->patterns) {
      workspace_pattern_peak_bytes += pattern.PeakSize();
    }
  }
  if (enable_workspace_preallocation == "1") {
    ASSERT_GT(workspace_pattern_peak_bytes, static_cast<size_t>(0));
  }

  const size_t wddm_before_inference_local_bytes =
      GetWddmLocalUsageBytes(dxgi_adapter.Get());
  AllocatorStats before_memory_measurement;
  webgpu_allocator->GetStats(&before_memory_measurement);
  auto* webgpu_buffer_allocator =
      static_cast<webgpu::GpuBufferAllocator*>(webgpu_allocator.get());
  webgpu_buffer_allocator->ResetPeakStats();

  constexpr int kMemoryMeasurementRuns = 3;
  WddmMemorySampler inference_wddm_sampler(dxgi_adapter);
  for (int i = 0; i < kMemoryMeasurementRuns; ++i) {
    fetches.clear();
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
  }
  inference_wddm_sampler.Stop();
  ASSERT_TRUE(SUCCEEDED(inference_wddm_sampler.Error()));
  AllocatorStats after_memory_measurement;
  webgpu_allocator->GetStats(&after_memory_measurement);

  constexpr int kMeasuredRuns = 30;
  std::vector<double> latencies_ms;
  latencies_ms.reserve(kMeasuredRuns);
  for (int i = 0; i < kMeasuredRuns; ++i) {
    fetches.clear();
    const auto start = Clock::now();
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
    const auto end = Clock::now();
    latencies_ms.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }

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
      std::chrono::duration<double, std::milli>(
          initialize_end - initialize_start)
          .count();
  const auto to_mib = [](size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
  };
  const auto positive_delta = [](size_t value, size_t baseline) {
    return value > baseline ? value - baseline : size_t{0};
  };

  const size_t wddm_initialization_peak_local_bytes =
      initialization_wddm_sampler.PeakUsedBytes();
  const size_t wddm_inference_peak_local_bytes =
      inference_wddm_sampler.PeakUsedBytes();
  const int64_t measurement_allocation_calls =
      after_memory_measurement.num_allocs - before_memory_measurement.num_allocs;
  const int64_t measurement_peak_bytes = after_memory_measurement.max_bytes_in_use;
  const int64_t measurement_peak_delta_bytes =
      measurement_peak_bytes -
      std::min(measurement_peak_bytes, before_memory_measurement.bytes_in_use);

  std::cout << "[ WEBGPU WORKSPACE BENCHMARK ]"
            << " model=qwen2.5-1.5b"
            << " workspace_preallocation=" << enable_workspace_preallocation
            << " planned_workspace_nodes=" << planned_workspace_nodes
            << " planned_workspace_slots=" << planned_workspace_slots
            << " aggregate_workspace_bytes=" << aggregate_workspace_bytes
            << " largest_workspace_bytes=" << largest_workspace_bytes
            << " workspace_pattern_peak_bytes=" << workspace_pattern_peak_bytes
            << " initialize_ms=" << initialize_ms
            << " average_ms=" << average_ms
            << " p50_ms=" << percentile(0.50)
            << " p90_ms=" << percentile(0.90)
            << " p99_ms=" << percentile(0.99)
            << " min_ms=" << latencies_ms.front()
            << " max_ms=" << latencies_ms.back()
            << " wddm_baseline_local_mib=" << to_mib(wddm_baseline_local_bytes)
            << " wddm_initialization_peak_local_mib="
            << to_mib(wddm_initialization_peak_local_bytes)
            << " wddm_initialization_peak_delta_mib="
            << to_mib(positive_delta(
                   wddm_initialization_peak_local_bytes,
                   wddm_baseline_local_bytes))
            << " wddm_post_initialize_local_mib="
            << to_mib(wddm_after_initialize_local_bytes)
            << " wddm_pre_inference_local_mib="
            << to_mib(wddm_before_inference_local_bytes)
            << " wddm_inference_peak_local_mib="
            << to_mib(wddm_inference_peak_local_bytes)
            << " wddm_inference_peak_delta_mib="
            << to_mib(positive_delta(
                   wddm_inference_peak_local_bytes,
                   wddm_before_inference_local_bytes))
            << " post_initialize_bytes_in_use="
            << post_initialize_stats.bytes_in_use
            << " post_initialize_max_bytes_in_use="
            << post_initialize_stats.max_bytes_in_use
            << " post_initialize_allocation_calls="
            << post_initialize_stats.num_allocs
            << " measurement_allocation_calls=" << measurement_allocation_calls
            << " measurement_peak_bytes=" << measurement_peak_bytes
            << " measurement_peak_delta_bytes=" << measurement_peak_delta_bytes
            << " final_bytes_in_use=" << after_memory_measurement.bytes_in_use
            << " final_max_bytes_in_use=" << after_memory_measurement.max_bytes_in_use
            << std::endl;
}

}  // namespace test
}  // namespace onnxruntime

#endif
