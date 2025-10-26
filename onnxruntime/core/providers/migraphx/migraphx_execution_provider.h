// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/execution_provider.h"
#include "core/framework/provider_options_utils.h"
#include "core/providers/migraphx/migraphx_execution_provider_info.h"
#include "core/providers/migraphx/migraphx_call.h"

using namespace std::literals::string_view_literals;

namespace onnxruntime {

namespace migraphx_env_vars {
constexpr auto kFP16Enable = "ORT_MIGRAPHX_FP16_ENABLE"sv;
constexpr auto kBF16Enable = "ORT_MIGRAPHX_BF16_ENABLE"sv;
constexpr auto kFP8Enable = "ORT_MIGRAPHX_FP8_ENABLE"sv;
constexpr auto kINT8Enable = "ORT_MIGRAPHX_INT8_ENABLE"sv;
constexpr auto kDumpModelOps = "ORT_MIGRAPHX_DUMP_MODEL_OPS"sv;
constexpr auto kINT8CalibrationTableName = "ORT_MIGRAPHX_INT8_CALIBRATION_TABLE_NAME"sv;
constexpr auto kCachePath = "ORT_MIGRAPHX_CACHE_PATH"sv;
constexpr auto kINT8UseNativeMIGraphXCalibrationTable = "ORT_MIGRAPHX_INT8_USE_NATIVE_CALIBRATION_TABLE"sv;
constexpr auto kExhaustiveTune = "ORT_MIGRAPHX_EXHAUSTIVE_TUNE"sv;
constexpr auto kModelCachePath = "ORT_MIGRAPHX_MODEL_CACHE_PATH"sv;
}  // namespace migraphx_env_vars

// Information to construct kernel function state.
struct MIGraphXFuncState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocate_handle = nullptr;
  migraphx::program prog{};
  std::string onnx_string;
  migraphx::onnx_options options;
  migraphx::target t{};
  std::unordered_map<std::string, std::size_t> input_name_indexes;
  std::mutex* mgx_mu_ptr = nullptr;
  bool no_input_shape = false;
  bool fp16_enable = false;
  bool bf16_enable = false;
  bool fp8_enable = false;
  bool int8_enable = false;
  bool int8_calibration_cache_available = false;
  std::unordered_map<std::string, float> dynamic_range_map;
  std::filesystem::path model_cache_dir;
  bool dump_model_ops = false;
  bool exhaustive_tune = false;
};

// Logical device representation.
class MIGraphXExecutionProvider : public IExecutionProvider {
 public:
  explicit MIGraphXExecutionProvider(const MIGraphXExecutionProviderInfo& info);
  ~MIGraphXExecutionProvider() override = default;

  Status Sync() const override;

  Status OnRunStart(const onnxruntime::RunOptions& run_options) override;

  Status OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) override;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const IKernelLookup& /*kernel_lookup*/,
                const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                IResourceAccountant* /* resource_accountant */) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  std::unique_ptr<IndexedSubGraph> GetSubGraph(const std::vector<std::size_t>& graph_nodes_index, const GraphViewer& graph) const;
  void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry, AllocatorMap& allocators) const override;
  OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  int GetDeviceId() const override { return device_id_; }
  ProviderOptions GetProviderOptions() const override {
    return {
        {std::string{migraphx_provider_option::kDeviceId}, MakeStringWithClassicLocale(device_id_)},
        {std::string{migraphx_provider_option::kFp16Enable}, MakeStringWithClassicLocale(fp16_enable_)},
        {std::string{migraphx_provider_option::kBf16Enable}, MakeStringWithClassicLocale(bf16_enable_)},
        {std::string{migraphx_provider_option::kFp8Enable}, MakeStringWithClassicLocale(fp8_enable_)},
        {std::string{migraphx_provider_option::kInt8Enable}, MakeStringWithClassicLocale(int8_enable_)},
        {std::string{migraphx_provider_option::kInt8CalibTable}, MakeStringWithClassicLocale(int8_calibration_table_name_)},
        {std::string{migraphx_provider_option::kInt8UseNativeCalibTable}, MakeStringWithClassicLocale(int8_use_native_calibration_table_)},
        {std::string{migraphx_provider_option::kExhaustiveTune}, MakeStringWithClassicLocale(exhaustive_tune_)},
        {std::string{migraphx_provider_option::kMemLimit}, MakeStringWithClassicLocale(mem_limit_)},
        {std::string{migraphx_provider_option::kArenaExtendStrategy}, EnumToName(arena_extend_strategy_mapping, arena_extend_strategy_)},
        {std::string{migraphx_provider_option::kGpuExternalAlloc}, MakeStringWithClassicLocale(external_alloc_)},
        {std::string{migraphx_provider_option::kGpuExternalFree}, MakeStringWithClassicLocale(external_free_)},
        {std::string{migraphx_provider_option::kGpuExternalEmptyCache}, MakeStringWithClassicLocale(external_empty_cache_)},
        {std::string{migraphx_provider_option::kModelCacheDir}, MakeStringWithClassicLocale(model_cache_path_)}};
  }

 private:
  OrtDevice::DeviceId device_id_{0};
  bool fp16_enable_ = false;
  bool bf16_enable_ = false;
  bool fp8_enable_ = false;
  bool int8_enable_ = false;
  std::string int8_calibration_table_name_;
  bool int8_calibration_cache_available_ = false;
  bool int8_use_native_calibration_table_ = false;
  std::filesystem::path calibration_cache_path_{};
  std::unordered_map<std::string, float> dynamic_range_map_;
  std::filesystem::path model_cache_path_{};
  std::set<std::string> session_input_names;
  bool dump_model_ops_ = false;
  migraphx::target t_;
  std::mutex mgx_mu_;
  hipStream_t stream_ = nullptr;
  hipDeviceProp_t device_prop_{};
  bool exhaustive_tune_ = false;
  mutable std::filesystem::path model_path_{};
  size_t mem_limit_{std::numeric_limits<size_t>::max()};
  ArenaExtendStrategy arena_extend_strategy_{ArenaExtendStrategy::kSameAsRequested};

  std::unordered_map<std::string, migraphx::program> map_progs_;
  std::unordered_map<std::string, std::string> map_onnx_string_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::size_t>> map_input_index_;
  std::unordered_map<std::string, bool> map_no_input_shape_;

  AllocatorPtr allocator_;
  std::unique_ptr<ModelMetadefIdGenerator> metadef_id_generator_;
  void* external_alloc_{nullptr};
  void* external_free_{nullptr};
  void* external_empty_cache_{nullptr};
};

}  // namespace onnxruntime
