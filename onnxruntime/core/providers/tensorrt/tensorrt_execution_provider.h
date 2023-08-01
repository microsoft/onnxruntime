// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <ctime>
#include <cudnn.h>
#include <cublas_v2.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/cuda/cuda_graph.h"
#include "tensorrt_execution_provider_info.h"

namespace onnxruntime {

namespace tensorrt_env_vars {
static const std::string kMaxPartitionIterations = "ORT_TENSORRT_MAX_PARTITION_ITERATIONS";
static const std::string kMinSubgraphSize = "ORT_TENSORRT_MIN_SUBGRAPH_SIZE";
static const std::string kMaxWorkspaceSize = "ORT_TENSORRT_MAX_WORKSPACE_SIZE";
static const std::string kFP16Enable = "ORT_TENSORRT_FP16_ENABLE";
static const std::string kINT8Enable = "ORT_TENSORRT_INT8_ENABLE";
static const std::string kINT8CalibrationTableName = "ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME";
static const std::string kINT8UseNativeTensorrtCalibrationTable = "ORT_TENSORRT_INT8_USE_NATIVE_CALIBRATION_TABLE";
static const std::string kDLAEnable = "ORT_TENSORRT_DLA_ENABLE";
static const std::string kDLACore = "ORT_TENSORRT_DLA_CORE";
static const std::string kDumpSubgraphs = "ORT_TENSORRT_DUMP_SUBGRAPHS";
static const std::string kEngineCacheEnable = "ORT_TENSORRT_ENGINE_CACHE_ENABLE";
static const std::string kCachePath = "ORT_TENSORRT_CACHE_PATH";
static const std::string kDecryptionEnable = "ORT_TENSORRT_ENGINE_DECRYPTION_ENABLE";
static const std::string kDecryptionLibPath = "ORT_TENSORRT_ENGINE_DECRYPTION_LIB_PATH";
static const std::string kForceSequentialEngineBuild = "ORT_TENSORRT_FORCE_SEQUENTIAL_ENGINE_BUILD";
static const std::string kContextMemorySharingEnable = "ORT_TENSORRT_CONTEXT_MEMORY_SHARING_ENABLE";
static const std::string kLayerNormFP32Fallback = "ORT_TENSORRT_LAYER_NORM_FP32_FALLBACK";
static const std::string kTimingCacheEnable = "ORT_TENSORRT_TIMING_CACHE_ENABLE";
static const std::string kForceTimingCache = "ORT_TENSORRT_FORCE_TIMING_CACHE_ENABLE";
static const std::string kDetailedBuildLog = "ORT_TENSORRT_DETAILED_BUILD_LOG_ENABLE";
static const std::string kBuildHeuristics = "ORT_TENSORRT_BUILD_HEURISTICS_ENABLE";
static const std::string kSparsityEnable = "ORT_TENSORRT_SPARSITY_ENABLE";
static const std::string kBuilderOptimizationLevel = "ORT_TENSORRT_BUILDER_OPTIMIZATION_LEVEL";
static const std::string kAuxiliaryStreams = "ORT_TENSORRT_AUXILIARY_STREAMS";
static const std::string kTacticSources = "ORT_TENSORRT_TACTIC_SOURCES";
static const std::string kExtraPluginLibPaths = "ORT_TENSORRT_EXTRA_PLUGIN_LIB_PATHS";
static const std::string kProfilesMinShapes = "ORT_TENSORRT_PROFILE_MIN_SHAPES";
static const std::string kProfilesMaxShapes = "ORT_TENSORRT_PROFILE_MAX_SHAPES";
static const std::string kProfilesOptShapes = "ORT_TENSORRT_PROFILE_OPT_SHAPES";
static const std::string kCudaGraphEnable = "ORT_TENSORRT_CUDA_GRAPH_ENABLE";
// Old env variable for backward compatibility
static const std::string kEngineCachePath = "ORT_TENSORRT_ENGINE_CACHE_PATH";
}  // namespace tensorrt_env_vars

class TensorrtLogger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity verbosity_;

 public:
  TensorrtLogger(Severity verbosity = Severity::kWARNING)
      : verbosity_(verbosity) {}
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= verbosity_) {
      time_t rawtime = std::time(0);
      struct tm stm;
#ifdef _MSC_VER
      gmtime_s(&stm, &rawtime);
#else
      gmtime_r(&rawtime, &stm);
#endif
      char buf[256];
      strftime(&buf[0], 256,
               "%Y-%m-%d %H:%M:%S",
               &stm);
      const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" : severity == Severity::kERROR ? "  ERROR"
                                                                            : severity == Severity::kWARNING ? "WARNING"
                                                                            : severity == Severity::kINFO    ? "   INFO"
                                                                                                             : "UNKNOWN");
      if (severity <= Severity::kERROR) {
        LOGS_DEFAULT(ERROR) << "[" << buf << " " << sevstr << "] " << msg;
      } else {
        LOGS_DEFAULT(WARNING) << "[" << buf << " " << sevstr << "] " << msg;
      }
    }
  }
};

namespace tensorrt_ptr {

struct TensorrtInferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

template <typename T>
using unique_pointer = std::unique_ptr<T, TensorrtInferDeleter>;
};  // namespace tensorrt_ptr

// Information to construct kernel function state.
struct TensorrtFuncState {
  AllocateFunc test_allocate_func = nullptr;
  DestroyFunc test_release_func = nullptr;
  AllocatorHandle allocator = nullptr;
  tensorrt_ptr::unique_pointer<nvonnxparser::IParser>* parser = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine>* engine = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext>* context = nullptr;
  std::unique_ptr<nvinfer1::IBuilder>* builder = nullptr;
  std::unique_ptr<nvinfer1::INetworkDefinition>* network = nullptr;
  std::vector<std::unordered_map<std::string, size_t>> input_info;
  std::vector<std::unordered_map<std::string, size_t>> output_info;
  std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>> input_shape_ranges;
  OrtMutex* tensorrt_mu_ptr = nullptr;
  bool fp16_enable = false;
  bool int8_enable = false;
  bool int8_calibration_cache_available = false;
  bool dla_enable = false;
  int dla_core = 0;
  size_t* max_workspace_size_ptr = nullptr;
  std::string trt_node_name_with_precision;
  bool engine_cache_enable = false;
  std::string engine_cache_path;
  nvinfer1::IRuntime* runtime = nullptr;
  std::vector<nvinfer1::IOptimizationProfile*> profiles;
  bool context_memory_sharing_enable = false;
  size_t* max_context_mem_size_ptr = nullptr;
  std::unordered_map<std::string, float> dynamic_range_map;
  bool engine_decryption_enable = false;
  int (*engine_decryption)(const char*, char*, size_t*) = nullptr;
  int (*engine_encryption)(const char*, char*, size_t) = nullptr;
  bool timing_cache_enable = true;
  bool force_timing_cache = false;
  bool detailed_build_log = false;
  bool build_heuristics_enable = false;
  bool sparsity_enable = false;
  int builder_optimization_level = 3;
  int auxiliary_streams = -1;
  bool filter_tactic_sources = false;
  nvinfer1::TacticSources tactic_sources;
  bool cuda_graph_enable = 0;
};

// Holds important information for building valid ORT graph.
struct SubGraphContext {
  std::unordered_set<std::string> output_args;
  std::unordered_map<std::string, const NodeArg*> inputs_and_initializers;
  std::unordered_map<std::string, const NodeArg*> manually_added_graph_inputs;
};

using SubGraphContextMap = std::unordered_map<std::string, std::unique_ptr<SubGraphContext>>;

// Logical device representation.
class TensorrtExecutionProvider : public IExecutionProvider {
 public:
  explicit TensorrtExecutionProvider(const TensorrtExecutionProviderInfo& info);
  virtual ~TensorrtExecutionProvider();

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const GraphViewer& graph,
                const IKernelLookup& /*kernel_lookup*/) const override;

  int GetDeviceId() const { return device_id_; }

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  Status OnRunStart() override;
  Status OnRunEnd(bool sync_stream) override;

  ProviderOptions GetProviderOptions() const override {
    return TensorrtExecutionProviderInfo::ToProviderOptions(info_);
  }

  void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry, AllocatorMap& allocators) const override;

  void GetCustomOpDomainList(std::vector<OrtCustomOpDomain*>& custom_op_domain_list) const override;

  OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const override;

  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  bool IsGraphCaptureEnabled() const override;
  bool IsGraphCaptured() const override;
  Status ReplayGraph() override;

 private:
  TensorrtExecutionProviderInfo info_;
  bool external_stream_ = false;
  cudaStream_t stream_ = nullptr;
  int max_partition_iterations_ = 1000;
  size_t min_subgraph_size_ = 1;
  size_t max_workspace_size_ = 1 << 30;  // 1GB
  bool fp16_enable_ = false;
  bool int8_enable_ = false;
  bool dla_enable_ = false;
  int dla_core_ = 0;
  bool force_sequential_engine_build_ = false;
  std::string int8_calibration_cache_name_;
  bool int8_calibration_cache_available_ = false;
  bool int8_use_native_tensorrt_calibration_table_ = false;
  bool dump_subgraphs_ = false;
  bool engine_cache_enable_ = false;
  bool build_heuristics_enable_ = false;
  bool sparsity_enable_ = false;
  int builder_optimization_level_ = 3;
  int auxiliary_streams_ = -1;
  std::string tactic_sources_;
  std::string cache_path_, engine_decryption_lib_path_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
  OrtMutex tensorrt_mu_;
  int device_id_;
  bool context_memory_sharing_enable_ = false;
  bool layer_norm_fp32_fallback_ = false;
  size_t max_ctx_mem_size_ = 0;
  IAllocatorUniquePtr<void> context_memory_ = nullptr;
  mutable char model_path_[4096] = {};  // Reserved for max path length
  bool engine_decryption_enable_ = false;
  int (*engine_decryption_)(const char*, char*, size_t*) = nullptr;
  int (*engine_encryption_)(const char*, char*, size_t) = nullptr;
  bool timing_cache_enable_ = false;
  bool force_timing_cache_match_ = false;
  bool detailed_build_log_ = false;
  bool cuda_graph_enable_ = false;

  // The OrtAllocator object will be get during ep compute time
  // and should be kept for the lifetime of TRT EP object.
  OrtAllocator* alloc_ = nullptr;

  std::unique_ptr<CUDAGraph> cuda_graph_;  // ORT TRT only supports CUDA graph when whole model is supported by TRT, so simply maintaining a CUDAGraph pointer is enough (no need to maintain one CUDAGraph pointer per TRT subgraph)
  bool is_graph_captured_ = false;
  int regular_run_count_before_graph_capture_ = 0;
  const int min_num_runs_before_cuda_graph_capture_ = 1;  // required min regular runs before graph capture for the necessary memory allocations.

  std::unordered_set<std::string> control_flow_op_set_ = {"If", "Loop", "Scan"};
  mutable std::unordered_map<std::string, std::unique_ptr<SubGraphContext>> subgraph_context_map_;
  std::unordered_map<std::string, tensorrt_ptr::unique_pointer<nvonnxparser::IParser>> parsers_;
  std::unordered_map<std::string, std::unique_ptr<nvinfer1::ICudaEngine>> engines_;
  std::unordered_map<std::string, std::unique_ptr<nvinfer1::IExecutionContext>> contexts_;
  std::unordered_map<std::string, std::unique_ptr<nvinfer1::IBuilder>> builders_;
  std::unordered_map<std::string, std::unique_ptr<nvinfer1::INetworkDefinition>> networks_;
  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, size_t>>> input_info_;
  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, size_t>>> output_info_;
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> profile_min_shapes_;
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> profile_max_shapes_;
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> profile_opt_shapes_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>>> input_shape_ranges_;
  std::unordered_map<std::string, std::vector<nvinfer1::IOptimizationProfile*>> profiles_;

  // for external stream, we need to create its cudnn/cublass handle before cuda EP enable cuda graph capture
  cudnnHandle_t external_cudnn_handle_ = nullptr;
  cublasHandle_t external_cublas_handle_ = nullptr;

  /**Get IndexedSubGraph based on node list of the subgraph*/
  std::unique_ptr<IndexedSubGraph> GetSubGraph(SubGraph_t graph_nodes_index,
                                               const GraphViewer& graph, const HashValue& model_hash, int subgraph_index) const;

  /**
  Get TensorRT supported node lists by calling Onnx-TensorRT parser recursively. Since each time the parser
  can only detect first unsupported node failure, it needs to wait for Onnxruntime to partition the graph
  and then detect next failure again. If there are too many iterations, which means many nodes in the graph
  are not supported by TensorRT, the process will be terminated and the whole graph is simply assigned to
  other execution provider.
  */
  SubGraphCollection_t GetSupportedList(SubGraphCollection_t supported_nodes_list, int iterations, const int max_iterations,
                                        const GraphViewer& graph, bool* early_termination) const;

  bool DetectTensorRTGraphCycles(SubGraphCollection_t& supported_nodes_vector, const GraphViewer& graph, const HashValue& model_hash, bool remove_cycles = true) const;

  /**
  Get a unique_lock object to control the concurrency behavior.
  Every api call not in the thread-safe operations(https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading)
  should be protected by a lock when invoked by multiple threads concurrently.
  */
  std::unique_lock<OrtMutex> GetApiLock() const;

  /**Check the graph is the subgraph of control flow op*/
  bool IsSubGraphOfControlFlowOp(const GraphViewer& graph) const;

  /**Check whether all the nodes of the graph are assigned to specific ep*/
  bool AllNodesAssignedToSpecificEP(const GraphViewer& graph, const std::string& provider_type) const;

  /**Check whether all the nodes of subgraph are supported*/
  bool IsSubGraphFullySupported(SubGraphCollection_t supported_nodes_vector, const int number_of_ort_nodes) const;

  /**
   * Set inputs, initializers and outputs for all subgraphs during TensorrtExecutionProvider::GetSupportedList()
   * and save those information in subgraph context data structure. It's useful for building a valid graph and
   * make Graph::Resolve() happy especially when dealing with nested control-flow op graph.
   */
  void BuildSubGraphContext(const Graph& build_graph) const;

  /**
   * Set outer scope values for subgraphs and add thoes values as top-level graph's inputs if needed.
   */
  void SetGraphOuterScopeValuesAndInputs(Graph& build_graph, const Graph& graph) const;

  /**
   * If ORT TRT manually sets graph input in TensorrtExecutionProvider::SetGraphOuterScopeValuesAndInputs(),
   * we have to manully set all the graph inputs in order to pass Graph::Resolve().
   */
  void SetAllGraphInputs(Graph& graph) const;

  /**
   * The newly-built graph has not yet being resolved by Graph::Resolve(), so we can't leverage
   * Graph::ResolveContext::IsInputInitializerOrOutput(). We have to implement this fuction again.
   */
  bool IsInputInitializerOrOutput(const Graph& graph, const std::string& name, bool check_ancestors) const;

  /**
   * The newly-built graph has not yet being resolved by Graph::Resolve(), so we can't leverage
   * Graph::ResolveContext::IsOuterScopeValue(). We have to implement this fuction again.
   */
  bool IsOuterScopeValue(const Graph& graph, const std::string& name) const;

  /**
   * The newly-built graph has not yet being resolved by Graph::Resolve(), so we can't leverage
   * Graph::ResolveContext::IsLocalValue(). We have to implement this fuction again.
   */
  bool IsLocalValue(const Graph& graph, const std::string& name) const;

  bool IsGraphCaptureAllowed() const;
  void CaptureBegin();
  void CaptureEnd();
  void IncrementRegularRunCountBeforeGraphCapture();
};
}  // namespace onnxruntime
