// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
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

using namespace std::literals::string_view_literals;  // NOLINT(build/namespaces_literals)

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
constexpr auto kModelMaxDynamicBatch = "ORT_MIGRAPHX_MAX_DYNAMIC_BATCH"sv;
constexpr auto kCompileBatches = "ORT_MIGRAPHX_COMPILE_BATCHES"sv;
}  // namespace migraphx_env_vars

// Tracks which dimensions are symbolic for a given input
struct SymbolicDimInfo {
  int dim_index;                // The dimension index (0 = batch, 1, 2, ...)
  std::string dim_param;        // The symbolic parameter name (e.g., "batch", "sequence_length")
};

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
  bool defer_compilation = false;
  bool fp16_enable = false;
  bool bf16_enable = false;
  bool fp8_enable = false;
  bool int8_enable = false;
  bool int8_calibration_cache_available = false;
  std::unordered_map<std::string, float> dynamic_range_map;
  std::filesystem::path model_cache_dir;
  bool dump_model_ops = false;
  bool exhaustive_tune = false;
  size_t max_dynamic_batch;
  // Reference to the cached programs map for this node (keyed by input shape hash)
  std::optional<std::reference_wrapper<std::unordered_map<std::string, migraphx::program>>> cached_programs_ref = std::nullopt;

  // Dynamic batch support
  bool has_dynamic_batch = false;
  std::vector<std::size_t> compiled_batch_sizes;

  // Padded input buffers for dynamic batching (allocated on GPU)
  struct PaddedBuffer {
    void* data = nullptr;          // GPU buffer pointer
    std::size_t size_bytes = 0;    // Buffer size in bytes
    migraphx::shape mgx_shape;     // Padded MIGraphX shape
  };
  std::vector<PaddedBuffer> padded_input_buffers;  // One per input when padding is active

  // Track last batch sizes to avoid re-allocation when batch size is unchanged
  std::size_t last_original_batch_size = 0;  // Original batch size from last run
  std::size_t last_padded_batch_size = 0;    // Padded batch size from last run

  // ═══════════════════════════════════════════════════════════════════════════
  // PERFORMANCE CACHES - Avoid redundant MIGraphX API calls per inference
  // ═══════════════════════════════════════════════════════════════════════════

  // Cached input parameter info (name as const char*, ORT index, MIGraphX shape)
  struct CachedInputParam {
    std::string name;              // Parameter name (owns the string)
    std::size_t ort_index;         // ORT input index
    migraphx::shape mgx_shape;     // MIGraphX shape for this input
  };

  // Cached output parameter info (name as const char*, output index, MIGraphX shape)
  struct CachedOutputParam {
    std::string name;              // Parameter name (owns the string)
    int output_index;              // ORT output index
    migraphx::shape mgx_shape;     // MIGraphX shape for this output
  };

  // Separated input/output parameter lists for O(1) iteration without map lookups
  std::vector<CachedInputParam> cached_inputs;
  std::vector<CachedOutputParam> cached_outputs;

  // Pre-allocated output shapes in ORT format (avoids vector allocation per inference)
  std::vector<std::vector<int64_t>> cached_output_ort_shapes;

  // Cached program_parameters object for ultra-fast rebinding
  std::optional<migraphx::program_parameters> cached_prog_params;

  // Cached output indices for pre-allocated outputs (used by run_migraphx_program)
  std::vector<std::size_t> cached_prog_output_indices;

  // Last input shapes for quick comparison (avoids hash computation in ultra-fast path)
  std::vector<std::int64_t> last_input_shapes_raw;

  // Last input shape hash (only computed when shapes change, used for cache lookup)
  std::string last_input_shape_hash;

  // Flag indicating caches are valid
  bool caches_valid = false;

  // ═══════════════════════════════════════════════════════════════════════════
  // OPTIMIZATION: Cached MIGraphX API results (avoid redundant API calls)
  // ═══════════════════════════════════════════════════════════════════════════

  // Cached program parameter shapes (from prog.get_parameter_shapes())
  std::optional<migraphx::program_parameter_shapes> cached_mgx_param_shapes;

  // Cached output shapes (from prog.get_output_shapes())
  std::optional<migraphx::shapes> cached_mgx_output_shapes;

  // Flag indicating ultra-fast caches are populated (avoid redundant populate calls)
  bool ultra_fast_caches_populated = false;

  // Track which program hash the cached shapes belong to (invalidate when program changes)
  std::string cached_program_hash;

  // ═══════════════════════════════════════════════════════════════════════════
  // OPTIMIZATION: Reusable temporary output buffers (for slicing mode)
  // ═══════════════════════════════════════════════════════════════════════════

  // Temporary output buffers for slicing (allocated at padded size)
  struct TempOutputBuffer {
    void* data = nullptr;           // GPU buffer pointer
    std::size_t size_bytes = 0;     // Buffer size in bytes
    migraphx::shape mgx_shape;      // Padded MIGraphX shape
  };
  std::vector<TempOutputBuffer> temp_output_buffers;

  // Track padded batch size for temp output buffers
  std::size_t temp_output_padded_batch_size = 0;
};

// Logical device representation.
class MIGraphXExecutionProvider final : public IExecutionProvider {
 public:
  explicit MIGraphXExecutionProvider(const MIGraphXExecutionProviderInfo& info);
  ~MIGraphXExecutionProvider() override = default;

  Status Sync() const override;

  Status OnRunStart(const RunOptions& run_options) override;

  Status OnRunEnd(bool sync_stream, const RunOptions& run_options) override;

  void dump_model_as_onnx(const std::string& onnx_buffer,
                          const std::string& model_name) const;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const GraphViewer& graph_viewer,
                const IKernelLookup& kernel_lookup,
                const GraphOptimizerRegistry& graph_optimizer_registry,
                IResourceAccountant* resource_accountant) const override;

  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

  std::unique_ptr<IndexedSubGraph> GetSubGraph(const std::vector<std::size_t>& graph_nodes_index, const GraphViewer& graph, bool is_graph_split) const;
  void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry, AllocatorMap& allocators) const override;
  OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  ProviderOptions GetProviderOptions() const override {
    return {
        {std::string{migraphx_provider_option::kDeviceId}, MakeStringWithClassicLocale(GetDeviceId())},
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
        {std::string{migraphx_provider_option::kModelCacheDir}, MakeStringWithClassicLocale(model_cache_path_)},
        {std::string{migraphx_provider_option::kModelMaxDynamicBatch}, MakeStringWithClassicLocale(max_dynamic_batch_)},
        {std::string{migraphx_provider_option::kCompileBatches}, compile_batches_}};
   }

 private:
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
  // Map of model input names per node (excludes weights/constants)
  std::unordered_map<std::string, std::set<std::string>> map_session_input_names_;
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
  std::unordered_map<std::string, bool> map_defer_compilation_;
  // Map of cached programs per node: node_name -> (input_shape_hash -> program)
  std::unordered_map<std::string, std::unordered_map<std::string, migraphx::program>> cached_programs_;

  AllocatorPtr allocator_;
  std::unique_ptr<ModelMetadefIdGenerator> metadef_id_generator_;
  void* external_alloc_{nullptr};
  void* external_free_{nullptr};
  void* external_empty_cache_{nullptr};
  bool first_start_ = true;
  size_t max_dynamic_batch_{0};
  std::string compile_batches_{};  // Comma-separated list of batch sizes to compile, e.g. "1,4,8,16,32"
};

}; // namespace onnxruntime
