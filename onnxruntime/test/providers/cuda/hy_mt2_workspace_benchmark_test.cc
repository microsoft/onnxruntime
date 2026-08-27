// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32

#include "gtest/gtest.h"

#if !defined(DISABLE_CONTRIB_OPS) && defined(USE_FPA_INTB_GEMM) && USE_FPA_INTB_GEMM

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#pragma comment(lib, "dxgi.lib")

#include "core/common/path_string.h"
#include "core/framework/allocator.h"
#include "core/framework/allocator_stats.h"
#include "core/framework/session_state.h"
#include "core/providers/cuda/cuda_provider_options.h"
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

constexpr int64_t kSequenceLength = 64;
constexpr int64_t kPastSequenceLength = 1;
constexpr int kMinFpAIntBSm = 75;

struct ModelProfile {
  const char* name;
  const char* default_model_path;
  int64_t num_layers;
  int64_t num_key_value_heads;
  int64_t head_size;
  int64_t vocab_size;
  int64_t bos_token_id;
  size_t matmul_nbits_nodes;
};

struct ArenaMemoryBreakdown {
  int64_t total_allocated_bytes;
  int64_t reserved_bytes;
  int64_t bfc_region_bytes;
  int64_t bytes_in_use;
  int64_t internal_fragmentation_bytes;
  int64_t arena_slack_bytes;
  double internal_fragmentation_ratio;
};

ArenaMemoryBreakdown GetArenaMemoryBreakdown(const AllocatorStats& stats) {
  ORT_ENFORCE(stats.total_allocated_bytes >= stats.reserved_bytes);
  ORT_ENFORCE(stats.bytes_in_use >= stats.reserved_bytes);
  ORT_ENFORCE(stats.bytes_in_use >= stats.bytes_requested_in_use);

  const int64_t bfc_region_bytes = stats.total_allocated_bytes - stats.reserved_bytes;
  const int64_t bfc_bytes_in_use = stats.bytes_in_use - stats.reserved_bytes;
  ORT_ENFORCE(bfc_region_bytes >= bfc_bytes_in_use);

  const int64_t internal_fragmentation_bytes =
      stats.bytes_in_use - stats.bytes_requested_in_use;
  return {
      stats.total_allocated_bytes,
      stats.reserved_bytes,
      bfc_region_bytes,
      stats.bytes_in_use,
      internal_fragmentation_bytes,
      bfc_region_bytes - bfc_bytes_in_use,
      stats.bytes_in_use == 0
          ? 0.0
          : static_cast<double>(internal_fragmentation_bytes) /
                static_cast<double>(stats.bytes_in_use),
  };
}

constexpr ModelProfile kHyMT2Profile{
    "hy-mt2-1.8b",
    "C:\\Users\\lochi\\repos\\onnxruntime\\Hy-MT2-1.8B-ONNX\\Q4_KQuant_tie\\cuda\\model.onnx",
    32, 4, 128, 120818, 120000, 225};

constexpr ModelProfile kQwen25Profile{
    "qwen2.5-1.5b",
    "C:\\Users\\lochi\\.foundry\\cache\\models\\Microsoft\\qwen2.5-1.5b-instruct-cuda-gpu-4\\v4\\model.onnx",
    28, 2, 128, 151936, 151643, 141};

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

ComPtr<IDXGIAdapter3> GetDxgiAdapterForCudaDevice(int device_id) {
  cudaDeviceProp device_properties{};
  const cudaError_t properties_result =
      cudaGetDeviceProperties(&device_properties, device_id);
  ORT_ENFORCE(properties_result == cudaSuccess,
              "cudaGetDeviceProperties failed for CUDA device ", device_id,
              ": ", cudaGetErrorString(properties_result));
  static_assert(sizeof(device_properties.luid) == sizeof(LUID));

  ComPtr<IDXGIFactory1> factory;
  ORT_ENFORCE(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))),
              "Failed to create a DXGI factory for WDDM memory sampling.");

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
    if (std::memcmp(device_properties.luid, &description.AdapterLuid, sizeof(LUID)) != 0) {
      continue;
    }

    ComPtr<IDXGIAdapter3> adapter3;
    ORT_ENFORCE(SUCCEEDED(adapter.As(&adapter3)),
                "CUDA device ", device_id, " does not expose IDXGIAdapter3.");
    return adapter3;
  }

  ORT_THROW("No DXGI adapter matched CUDA device ", device_id,
            " LUID for WDDM memory sampling.");
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

const ModelProfile& GetModelProfile() {
  std::string model_name = Env::Default().GetEnvironmentVar("ORT_WORKSPACE_BENCHMARK_MODEL");
  if (model_name.empty() || model_name == kHyMT2Profile.name) {
    return kHyMT2Profile;
  }
  if (model_name == kQwen25Profile.name) {
    return kQwen25Profile;
  }

  ORT_THROW("ORT_WORKSPACE_BENCHMARK_MODEL must be ", kHyMT2Profile.name, " or ",
            kQwen25Profile.name, ", but got: ", model_name);
}

std::string BuildMaxShapeOverride(const ModelProfile& profile) {
  std::ostringstream shapes;
  shapes << "input_ids:[1," << kSequenceLength << "]"
         << ";attention_mask:[1," << kPastSequenceLength + kSequenceLength << "]";
  for (int64_t layer = 0; layer < profile.num_layers; ++layer) {
    const std::string shape = ":[1," + std::to_string(profile.num_key_value_heads) + "," +
                              std::to_string(kPastSequenceLength) + "," +
                              std::to_string(profile.head_size) + "]";
    shapes << ";past_key_values." << layer << ".key" << shape
           << ";past_key_values." << layer << ".value" << shape;
  }
  return shapes.str();
}

}  // namespace

// Run each configuration in a fresh process so CUDA arena state from one measurement cannot
// affect the other. ORT_WORKSPACE_BENCHMARK_PREALLOCATION selects the baseline (0) or planned (1)
// configuration. The benchmark reports metrics without fixed thresholds because device memory
// and latency vary by GPU, driver, clocks, and system load.
TEST(MatMulNBitsWorkspace, ModelWorkspacePreallocationBenchmark) {
  const ModelProfile& profile = GetModelProfile();
  std::string model_path_utf8 = Env::Default().GetEnvironmentVar("ORT_WORKSPACE_BENCHMARK_MODEL_PATH");
  if (model_path_utf8.empty()) {
    model_path_utf8 = profile.default_model_path;
  }

  const std::filesystem::path model_path(ToPathString(model_path_utf8));
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << profile.name << " model not found at " << model_path_utf8
                 << "; set ORT_WORKSPACE_BENCHMARK_MODEL_PATH to model.onnx.";
  }

  const int device_sm = CudaDeviceComputeCapabilityOrNegative();
  if (device_sm < 0) {
    GTEST_SKIP() << "No CUDA device available; skipping " << profile.name << " benchmark.";
  }
  if (device_sm < kMinFpAIntBSm) {
    GTEST_SKIP() << "Device compute capability " << device_sm << " < " << kMinFpAIntBSm
                 << "; fpA_intB benchmarking is unsupported.";
  }

  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  const std::optional<size_t> baseline_used_bytes = GetCudaUsedMemoryBytes();
  ASSERT_TRUE(baseline_used_bytes.has_value());
  ComPtr<IDXGIAdapter3> dxgi_adapter = GetDxgiAdapterForCudaDevice(0);
  const size_t wddm_baseline_local_bytes =
      GetWddmLocalUsageBytes(dxgi_adapter.Get());

  const std::string enable_workspace_preallocation =
      GetBinaryEnvironmentValue("ORT_WORKSPACE_BENCHMARK_PREALLOCATION", "0");
  const std::string disable_prepacking =
      GetBinaryEnvironmentValue("ORT_WORKSPACE_BENCHMARK_DISABLE_PREPACKING", "0");
  const std::string use_device_initializers =
      GetBinaryEnvironmentValue("ORT_WORKSPACE_BENCHMARK_USE_DEVICE_INITIALIZERS", "0");
  const std::string enable_fpa_intb =
      GetBinaryEnvironmentValue("ORT_WORKSPACE_BENCHMARK_FPA_INTB_GEMM", "1");

  ASSERT_FALSE(disable_prepacking == "1" && enable_fpa_intb == "1")
      << "The fpA_intB path requires PrePack. Set ORT_WORKSPACE_BENCHMARK_FPA_INTB_GEMM=0 when "
         "ORT_WORKSPACE_BENCHMARK_DISABLE_PREPACKING=1.";
  ASSERT_FALSE(enable_workspace_preallocation == "1" && enable_fpa_intb == "0")
      << "This model only declares workspace through the fpA_intB path. Enable "
         "ORT_WORKSPACE_BENCHMARK_FPA_INTB_GEMM when benchmarking workspace preallocation.";

  SessionOptions session_options;
  session_options.session_logid = "ModelWorkspacePreallocationBenchmark";
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsMaxShapeOverride, BuildMaxShapeOverride(profile).c_str()));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsCudaFpAIntBGemm, enable_fpa_intb.c_str()));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsConfigDisablePrepacking, disable_prepacking.c_str()));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsUseDeviceAllocatorForInitializers, use_device_initializers.c_str()));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsEnableStaticWorkspacePreallocation,
      enable_workspace_preallocation.c_str()));

  using Clock = std::chrono::steady_clock;
  CudaMemorySampler initialization_memory_sampler(0);
  WddmMemorySampler initialization_wddm_sampler(dxgi_adapter);
  const auto initialize_start = Clock::now();

  OrtCUDAProviderOptionsV2 cuda_options{};
  cuda_options.arena_extend_strategy = ArenaExtendStrategy::kSameAsRequested;
  cuda_options.do_copy_in_default_stream = true;
  cuda_options.use_tf32 = false;
  auto cuda_ep = CudaExecutionProviderWithOptions(&cuda_options);
  ASSERT_NE(cuda_ep, nullptr);

  InferenceSessionWrapper session(session_options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda_ep)));
  ASSERT_STATUS_OK(session.Load(model_path.native().c_str()));
  ASSERT_STATUS_OK(session.Initialize());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto initialize_end = Clock::now();
  initialization_memory_sampler.Stop();
  initialization_wddm_sampler.Stop();
  ASSERT_EQ(initialization_memory_sampler.Error(), cudaSuccess);
  ASSERT_TRUE(SUCCEEDED(initialization_wddm_sampler.Error()));
  const std::optional<size_t> after_initialize_used_bytes = GetCudaUsedMemoryBytes();
  ASSERT_TRUE(after_initialize_used_bytes.has_value());
  const size_t wddm_after_initialize_local_bytes =
      GetWddmLocalUsageBytes(dxgi_adapter.Get());
  const OrtDevice cuda_device(
      OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA, 0);
  AllocatorPtr cuda_allocator = session.GetSessionState().GetAllocator(cuda_device);
  ASSERT_NE(cuda_allocator, nullptr);
  IArena* cuda_arena = IArena::SafeArenaCast(cuda_allocator.get());
  ASSERT_NE(cuda_arena, nullptr);
  AllocatorStats post_initialize_stats;
  cuda_allocator->GetStats(&post_initialize_stats);

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
  ASSERT_EQ(matmul_nbits_nodes, profile.matmul_nbits_nodes);
  ASSERT_EQ(cuda_matmul_nbits_nodes, matmul_nbits_nodes);

  const SequentialExecutionPlan* execution_plan = session.GetSessionState().GetExecutionPlan();
  ASSERT_NE(execution_plan, nullptr);
  size_t planned_workspace_nodes = 0;
  size_t largest_workspace_bytes = 0;
  for (const auto& [node_index, workspace_plans] : execution_plan->workspace_allocation_plan) {
    static_cast<void>(node_index);
    ASSERT_EQ(workspace_plans.size(), static_cast<size_t>(1));
    ++planned_workspace_nodes;
    largest_workspace_bytes =
        std::max(largest_workspace_bytes, workspace_plans.front().allocation_bytes);
  }
  if (enable_workspace_preallocation == "1") {
    ASSERT_GT(planned_workspace_nodes, static_cast<size_t>(0));
    ASSERT_GT(largest_workspace_bytes, static_cast<size_t>(0));
  } else {
    ASSERT_EQ(planned_workspace_nodes, static_cast<size_t>(0));
    ASSERT_EQ(largest_workspace_bytes, static_cast<size_t>(0));
  }

  std::vector<int64_t> input_ids(static_cast<size_t>(kSequenceLength), profile.bos_token_id);
  std::vector<int64_t> attention_mask(
      static_cast<size_t>(kPastSequenceLength + kSequenceLength), 1);
  std::vector<MLFloat16> past_data(
      static_cast<size_t>(profile.num_key_value_heads * kPastSequenceLength * profile.head_size),
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
      1, profile.num_key_value_heads, kPastSequenceLength, profile.head_size};
  for (int64_t layer = 0; layer < profile.num_layers; ++layer) {
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
            TensorShape({1, kSequenceLength, profile.vocab_size}));
  fetches.clear();

  AllocatorStats post_warmup_stats;
  cuda_allocator->GetStats(&post_warmup_stats);
  ASSERT_STATUS_OK(cuda_arena->Shrink());
  AllocatorStats post_shrink_stats;
  cuda_allocator->GetStats(&post_shrink_stats);
  ASSERT_GE(post_warmup_stats.total_allocated_bytes, post_shrink_stats.total_allocated_bytes);
  const int64_t shrink_reclaimed_bytes =
      post_warmup_stats.total_allocated_bytes - post_shrink_stats.total_allocated_bytes;
  const size_t wddm_before_inference_local_bytes =
      GetWddmLocalUsageBytes(dxgi_adapter.Get());
  AllocatorStats before_memory_measurement;
  cuda_allocator->GetStats(&before_memory_measurement);

  constexpr int kMemoryMeasurementRuns = 3;
  CudaMemorySampler inference_memory_sampler(0);
  WddmMemorySampler inference_wddm_sampler(dxgi_adapter);
  for (int i = 0; i < kMemoryMeasurementRuns; ++i) {
    fetches.clear();
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  inference_memory_sampler.Stop();
  inference_wddm_sampler.Stop();
  ASSERT_EQ(inference_memory_sampler.Error(), cudaSuccess);
  ASSERT_TRUE(SUCCEEDED(inference_wddm_sampler.Error()));
  AllocatorStats after_memory_measurement;
  cuda_allocator->GetStats(&after_memory_measurement);
  const int64_t measurement_new_arena_bytes =
      after_memory_measurement.total_allocated_bytes -
      before_memory_measurement.total_allocated_bytes;

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

  AllocatorStats allocator_stats;
  cuda_allocator->GetStats(&allocator_stats);
  const ArenaMemoryBreakdown post_initialize_breakdown =
      GetArenaMemoryBreakdown(post_initialize_stats);
  const ArenaMemoryBreakdown post_warmup_breakdown =
      GetArenaMemoryBreakdown(post_warmup_stats);
  const ArenaMemoryBreakdown post_shrink_breakdown =
      GetArenaMemoryBreakdown(post_shrink_stats);
  const ArenaMemoryBreakdown final_breakdown =
      GetArenaMemoryBreakdown(allocator_stats);

  const auto to_mib = [](size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
  };
  const auto delta_from_baseline = [baseline = *baseline_used_bytes](size_t bytes) {
    return bytes > baseline ? bytes - baseline : size_t{0};
  };
  const size_t initialization_peak_bytes = initialization_memory_sampler.PeakUsedBytes();
  const size_t inference_peak_bytes = inference_memory_sampler.PeakUsedBytes();
  const size_t wddm_initialization_peak_local_bytes =
      initialization_wddm_sampler.PeakUsedBytes();
  const size_t wddm_inference_peak_local_bytes =
      inference_wddm_sampler.PeakUsedBytes();
  const uintmax_t serialized_model_bytes = std::filesystem::file_size(model_path);
  std::filesystem::path external_data_path = model_path;
  external_data_path += L".data";
  const uintmax_t serialized_external_data_bytes =
      std::filesystem::exists(external_data_path)
          ? std::filesystem::file_size(external_data_path)
          : uintmax_t{0};

  std::cout << "[ MODEL WORKSPACE BENCHMARK ]"
            << " model=" << profile.name
            << " workspace_preallocation=" << enable_workspace_preallocation
            << " arena_extend_strategy=same_as_requested"
            << " disable_prepacking=" << disable_prepacking
            << " use_device_initializers=" << use_device_initializers
            << " fpa_intb_gemm=" << enable_fpa_intb
            << " planned_workspace_nodes=" << planned_workspace_nodes
            << " largest_workspace_bytes=" << largest_workspace_bytes
            << " serialized_model_bytes=" << serialized_model_bytes
            << " serialized_external_data_bytes=" << serialized_external_data_bytes
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
            << " wddm_baseline_local_mib=" << to_mib(wddm_baseline_local_bytes)
            << " wddm_initialization_peak_local_mib="
            << to_mib(wddm_initialization_peak_local_bytes)
            << " wddm_initialization_peak_delta_mib="
            << to_mib(wddm_initialization_peak_local_bytes -
                      std::min(wddm_initialization_peak_local_bytes,
                               wddm_baseline_local_bytes))
            << " wddm_post_initialize_local_mib="
            << to_mib(wddm_after_initialize_local_bytes)
            << " wddm_pre_inference_local_mib="
            << to_mib(wddm_before_inference_local_bytes)
            << " wddm_inference_peak_local_mib="
            << to_mib(wddm_inference_peak_local_bytes)
            << " wddm_inference_peak_delta_mib="
            << to_mib(wddm_inference_peak_local_bytes -
                      std::min(wddm_inference_peak_local_bytes,
                               wddm_before_inference_local_bytes))
            << " post_initialize_total_allocated_bytes="
            << post_initialize_breakdown.total_allocated_bytes
            << " post_initialize_reserved_bytes=" << post_initialize_breakdown.reserved_bytes
            << " post_initialize_bfc_region_bytes=" << post_initialize_breakdown.bfc_region_bytes
            << " post_initialize_bytes_in_use=" << post_initialize_breakdown.bytes_in_use
            << " post_initialize_arena_slack_bytes=" << post_initialize_breakdown.arena_slack_bytes
            << " post_initialize_internal_fragmentation_bytes="
            << post_initialize_breakdown.internal_fragmentation_bytes
            << " post_initialize_internal_fragmentation_ratio="
            << post_initialize_breakdown.internal_fragmentation_ratio
            << " post_warmup_total_allocated_bytes="
            << post_warmup_breakdown.total_allocated_bytes
            << " post_warmup_arena_slack_bytes=" << post_warmup_breakdown.arena_slack_bytes
            << " shrink_reclaimed_bytes=" << shrink_reclaimed_bytes
            << " post_shrink_total_allocated_bytes="
            << post_shrink_breakdown.total_allocated_bytes
            << " post_shrink_arena_slack_bytes=" << post_shrink_breakdown.arena_slack_bytes
            << " measurement_new_arena_bytes=" << measurement_new_arena_bytes
            << " arena_bytes_in_use=" << allocator_stats.bytes_in_use
            << " arena_total_allocated_bytes=" << allocator_stats.total_allocated_bytes
            << " arena_reserved_bytes=" << final_breakdown.reserved_bytes
            << " arena_bfc_region_bytes=" << final_breakdown.bfc_region_bytes
            << " arena_slack_bytes=" << final_breakdown.arena_slack_bytes
            << " arena_internal_fragmentation_bytes="
            << final_breakdown.internal_fragmentation_bytes
            << " arena_internal_fragmentation_ratio="
            << final_breakdown.internal_fragmentation_ratio
            << " arena_max_bytes_in_use=" << allocator_stats.max_bytes_in_use
            << " arena_num_allocs=" << allocator_stats.num_allocs
            << " arena_num_reserves=" << allocator_stats.num_reserves
            << std::endl;
}

}  // namespace test
}  // namespace onnxruntime

#endif
#endif
