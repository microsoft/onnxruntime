// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <miopen/miopen.h>
#include <rocblas/rocblas.h>

#include "core/framework/arena_extend_strategy.h"
#include "core/framework/execution_provider.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/migraphx/migraphx_execution_provider_info.h"
#include "core/providers/migraphx/migraphx_inc.h"

#include <map>
#include <unordered_map>
// TODO: find a better way to share this
// #include "core/providers/cuda/rocm_stream_handle.h"

namespace onnxruntime {

namespace migraphx_env_vars {
static const char kFP16Enable[] = "ORT_MIGRAPHX_FP16_ENABLE";
static const char kINT8Enable[] = "ORT_MIGRAPHX_INT8_ENABLE";
static const char dumpModelOps[] = "ORT_MIGRAPHX_DUMP_MODEL_OPS";
static const char kINT8CalibrationTableName[] = "ORT_MIGRAPHX_INT8_CALIBRATION_TABLE_NAME";
static const char kCachePath[] = "ORT_MIGRAPHX_CACHE_PATH";
static const char kINT8UseNativeMIGraphXCalibrationTable[] = "ORT_MIGRAPHX_INT8_USE_NATIVE_CALIBRATION_TABLE";
};  // namespace migraphx_env_vars

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
  OrtMutex* mgx_mu_ptr = nullptr;
  bool no_input_shape = false;
  bool fp16_enable = false;
  bool int8_enable = false;
  bool int8_calibration_cache_available = false;
  std::unordered_map<std::string, float> dynamic_range_map;
  bool dump_model_ops = false;
};

// Logical device representation.
class MIGraphXExecutionProvider : public IExecutionProvider {
 public:
  explicit MIGraphXExecutionProvider(const MIGraphXExecutionProviderInfo& info);
  ~MIGraphXExecutionProvider();

#ifdef MIGRAPHX_STREAM_SYNC
  Status Sync() const override;

  Status OnRunStart(const onnxruntime::RunOptions& run_options) override;

  Status OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) override;
#endif

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const IKernelLookup& /*kernel_lookup*/) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  std::unique_ptr<IndexedSubGraph> GetSubGraph(const std::vector<std::size_t>& graph_nodes_index, const GraphViewer& graph) const;
  void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry, AllocatorMap& allocators) const override;
  OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

 private:
  bool fp16_enable_ = false;
  bool int8_enable_ = false;
  std::string int8_calibration_cache_name_;
  bool int8_calibration_cache_available_ = false;
  bool int8_use_native_migraphx_calibration_table_ = false;
  std::string calibration_cache_path_;
  std::unordered_map<std::string, float> dynamic_range_map;
  bool dump_model_ops_ = false;
  int device_id_;
  migraphx::target t_;
  OrtMutex mgx_mu_;
  hipStream_t stream_ = nullptr;

  std::unordered_map<std::string, migraphx::program> map_progs_;
  std::unordered_map<std::string, std::string> map_onnx_string_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::size_t>> map_input_index_;
  std::unordered_map<std::string, bool> map_no_input_shape_;

  AllocatorPtr allocator_;
  miopenHandle_t external_miopen_handle_ = nullptr;
  rocblas_handle external_rocblas_handle_ = nullptr;
  std::unique_ptr<ModelMetadefIdGenerator> metadef_id_generator_;
};

}  // namespace onnxruntime
