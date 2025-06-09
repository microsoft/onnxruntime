// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include "core/framework/arena_extend_strategy.h"
#include "core/framework/execution_provider.h"
#include <mutex>
#include "core/providers/migraphx/migraphx_execution_provider_info.h"
#include "core/providers/migraphx/migraphx_call.h"

#include <map>
#include <unordered_map>
#include <filesystem>

namespace onnxruntime {

namespace migraphx_env_vars {
constexpr auto kFP16Enable = "ORT_MIGRAPHX_FP16_ENABLE";
constexpr auto kFP8Enable = "ORT_MIGRAPHX_FP8_ENABLE";
constexpr auto kINT8Enable = "ORT_MIGRAPHX_INT8_ENABLE";
constexpr auto kDumpModelOps = "ORT_MIGRAPHX_DUMP_MODEL_OPS";
constexpr auto kINT8CalibrationTableName = "ORT_MIGRAPHX_INT8_CALIBRATION_TABLE_NAME";
constexpr auto kCachePath = "ORT_MIGRAPHX_CACHE_PATH";
constexpr auto kINT8UseNativeMIGraphXCalibrationTable = "ORT_MIGRAPHX_INT8_USE_NATIVE_CALIBRATION_TABLE";
constexpr auto kModelCachePath = "ORT_MIGRAPHX_MODEL_CACHE_PATH";
constexpr auto kExhaustiveTune = "ORT_MIGRAPHX_EXHAUSTIVE_TUNE";
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
  bool fp8_enable = false;
  bool int8_enable = false;
  bool int8_calibration_cache_available = false;
  std::unordered_map<std::string, float> dynamic_range_map;
  std::filesystem::path model_cache_dir;
  bool dump_model_ops = false;
  bool exhaustive_tune = false;
};

// Logical device representation.
class MIGraphXExecutionProvider final : public IExecutionProvider {
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

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  static AllocatorPtr CreateMIGraphXAllocator(OrtDevice::DeviceId device_id, size_t migx_mem_limit, ArenaExtendStrategy arena_extend_strategy,
                                              MIGraphXExecutionProviderExternalAllocatorInfo external_alloc_info, const OrtArenaCfg* arena_cfg);

  std::unique_ptr<IndexedSubGraph> GetSubGraph(const std::vector<std::size_t>& graph_nodes_index, const GraphViewer& graph) const;
  void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry, AllocatorMap& allocators) const override;
  OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  int GetDeviceId() const override { return device_id_; }
  ProviderOptions GetProviderOptions() const override {
     return {{migraphx_provider_option::kDeviceId, MakeStringWithClassicLocale(device_id_)},
       {migraphx_provider_option::kFp16Enable, MakeStringWithClassicLocale(fp16_enable_)},
       {migraphx_provider_option::kInt8Enable, MakeStringWithClassicLocale(int8_enable_)},
       {migraphx_provider_option::kModelCacheDir, MakeStringWithClassicLocale(model_cache_path_)}
   };
  }

 private:
  OrtDevice::DeviceId device_id_{0};
  bool fp16_enable_ = false;
  bool fp8_enable_ = false;
  bool int8_enable_ = false;
  std::string int8_calibration_cache_name_;
  bool int8_calibration_cache_available_ = false;
  bool int8_use_native_migraphx_calibration_table_ = false;
  std::filesystem::path calibration_cache_path_{};
  std::unordered_map<std::string, float> dynamic_range_map_;
  std::set<std::string> session_input_names;
  std::filesystem::path model_cache_path_{};
  bool dump_model_ops_ = false;
  migraphx::target t_;
  std::mutex mgx_mu_;
  hipStream_t stream_ = nullptr;
  hipDeviceProp_t device_prop_;
  bool exhaustive_tune_ = false;
  mutable std::filesystem::path model_path_{};

  std::unordered_map<std::string, migraphx::program> map_progs_;
  std::unordered_map<std::string, std::string> map_onnx_string_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::size_t>> map_input_index_;
  std::unordered_map<std::string, bool> map_no_input_shape_;

  AllocatorPtr allocator_;
  std::unique_ptr<ModelMetadefIdGenerator> metadef_id_generator_;
};

}  // namespace onnxruntime
