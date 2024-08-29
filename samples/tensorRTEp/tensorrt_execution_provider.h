#pragma once
#include <ctime>
#include <string>
#include <unordered_set>
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/provider_options.h"
#include "tensorrt_execution_provider_info.h"
#include "nv_includes.h"

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

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
static const std::string kWeightStrippedEngineEnable = "ORT_TENSORRT_WEIGHT_STRIPPED_ENGINE_ENABLE";
static const std::string kOnnxModelFolderPath = "ORT_TENSORRT_ONNX_MODEL_FOLDER_PATH";
// As a timing cache can be used across multiple ONNX files it makes sense to have a separate cache path
static const std::string kTimingCachePath = "ORT_TENSORRT_GLOBAL_CACHE_PATH";
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
static const std::string kDumpEpContextModel = "ORT_DUMP_EP_CONTEXT_MODEL";
static const std::string kEpContextEmbedMode = "ORT_EP_CONTEXT_EMBED_MODE";
static const std::string kEpContextComputeCapabilityEnable = "ORT_EP_CONTEXT_COMPUTE_CAPABILITY_ENABLE";
static const std::string kEngineCachePrefix = "ORT_TENSORRT_CACHE_PREFIX";
// Old env variable for backward compatibility
static const std::string kEngineCachePath = "ORT_TENSORRT_ENGINE_CACHE_PATH";
}  // namespace tensorrt_env_vars

using HashValue = uint64_t;
using AllocateFunc = void* (*)(void*, size_t, size_t);
using DestroyFunc = void (*)(void*, void*);

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
//        LOGS_DEFAULT(ERROR) << "[" << buf << " " << sevstr << "] " << msg;
      } else {
//        LOGS_DEFAULT(WARNING) << "[" << buf << " " << sevstr << "] " << msg;
      }
    }
  }
  void set_level(Severity verbosity) {
    verbosity_ = verbosity;
  }
  Severity get_level() const {
    return verbosity_;
  }
};

namespace tensorrt_ptr {

struct TensorrtInferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      delete obj;
    }
  }
};

template <typename T>
using unique_pointer = std::unique_ptr<T, TensorrtInferDeleter>;
};  // namespace tensorrt_ptr

class OutputAllocator : public nvinfer1::IOutputAllocator {
 public:
#if NV_TENSORRT_MAJOR >= 10
  void* reallocateOutputAsync(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment, cudaStream_t stream) noexcept override;
#else
  void* reallocateOutput(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment) noexcept override;
#endif
  void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override;

  void* getBuffer() {
    return outputPtr;
  }

  std::vector<int64_t>& getOutputShape() {
    return output_shapes;
  }

  uint64_t getSize() {
    return allocated_size;
  }

  ~OutputAllocator() override {
    cudaFree(outputPtr);
  }

 private:
  void* outputPtr{nullptr};
  uint64_t allocated_size = 0;
  std::vector<int64_t> output_shapes;
};

using ShapeRangesMap = std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>>;

struct TensorrtFuncState {
  AllocateFunc test_allocate_func = nullptr;
  DestroyFunc test_release_func = nullptr;
  void* allocator = nullptr;
  std::string fused_node_name;
  nvinfer1::IBuilder* builder;
  tensorrt_ptr::unique_pointer<nvonnxparser::IParser>* parser = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine>* engine = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext>* context = nullptr;
  std::unique_ptr<nvinfer1::INetworkDefinition>* network = nullptr;
  std::vector<std::unordered_map<std::string, size_t>> input_info;
  std::vector<std::unordered_map<std::string, size_t>> output_info;
  std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>> input_shape_ranges;
//  OrtMutex* tensorrt_mu_ptr = nullptr;
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
  std::string timing_cache_path;
  bool force_timing_cache = false;
  bool detailed_build_log = false;
  bool build_heuristics_enable = false;
  bool sparsity_enable = false;
  int builder_optimization_level = 3;
  int auxiliary_streams = -1;
  bool filter_tactic_sources = false;
  nvinfer1::TacticSources tactic_sources;
  bool cuda_graph_enable = 0;
  std::string cache_prefix;
  std::string cache_suffix;
  bool engine_hw_compatible = false;
};

// Minimum information to construct kernel function state for direct engine load code path
struct TensorrtShortFuncState {
  AllocateFunc test_allocate_func = nullptr;
  DestroyFunc test_release_func = nullptr;
  void* allocator = nullptr;
  std::string fused_node_name;
  std::unique_ptr<nvinfer1::ICudaEngine>* engine = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext>* context = nullptr;
  std::vector<std::unordered_map<std::string, size_t>> input_info;
  std::vector<std::unordered_map<std::string, size_t>> output_info;
  bool context_memory_sharing_enable = false;
  size_t* max_context_mem_size_ptr = nullptr;
//  OrtMutex* tensorrt_mu_ptr = nullptr;
};

using DDSOutputAllocatorMap = std::unordered_map<std::string, std::unique_ptr<OutputAllocator>>;
std::string GetWeightRefittedEnginePath(std::string engine_cache_path);

struct TensorrtExecutionProvider : public OrtExecutionProvider {
    TensorrtExecutionProvider(const char* ep_type, const ProviderOptions& provider_options);
    bool IsGraphCaptured(int graph_annotation_id) const { return false; }
    static OrtStatusPtr RefitEngine(std::string onnx_model_filename,
                                      std::string& onnx_model_folder_path,
                                      std::string& weight_stripped_engine_cath_path,
                                      bool path_check,
                                      nvinfer1::ICudaEngine* trt_engine,
                                      bool serialize_refitted_engine,
                                      bool detailed_build_log);

    std::unique_ptr<OrtIndexedSubGraph> GetSubGraph(SubGraph_t graph_nodes_index,
                                               const OrtGraphViewer* graph, const HashValue& model_hash, int subgraph_index) const;
    SubGraphCollection_t GetSupportedList(SubGraphCollection_t supported_nodes_list, int iterations, const int max_iterations,
                                          const OrtGraphViewer* graph, bool* early_termination) const;

    bool DetectTensorRTGraphCycles(SubGraphCollection_t& supported_nodes_vector, const OrtGraphViewer* graph, const HashValue& model_hash, bool remove_cycles = true) const;

    /**Check the graph is the subgraph of control flow op*/
    bool IsSubGraphOfControlFlowOp(const OrtGraphViewer* graph) const;

    /**Check whether all the nodes of the graph are assigned to specific ep*/
    bool AllNodesAssignedToSpecificEP(const OrtGraphViewer* graph, const std::string& provider_type) const;

    /**Check whether all the nodes of subgraph are supported*/
    bool IsSubGraphFullySupported(SubGraphCollection_t supported_nodes_vector, const int number_of_ort_nodes) const;

    static const OrtApi* api_;
    std::string trt_node_name_with_precision_;
    std::unordered_map<std::string, float> dynamic_range_map_;
    std::string cache_suffix_;
private:
  mutable TensorrtExecutionProviderInfo info_;
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
  bool weight_stripped_engine_enable_ = false;
  bool weight_stripped_engine_refit_ = false;
  std::string onnx_model_folder_path_;
  bool build_heuristics_enable_ = false;
  bool sparsity_enable_ = false;
  int builder_optimization_level_ = 3;
  int auxiliary_streams_ = -1;
  std::string tactic_sources_;
  std::string global_cache_path_, cache_path_, engine_decryption_lib_path_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
//  OrtMutex tensorrt_mu_;
  int device_id_;
  std::string compute_capability_;
  bool context_memory_sharing_enable_ = false;
  bool layer_norm_fp32_fallback_ = false;
  size_t max_ctx_mem_size_ = 0;
//  IAllocatorUniquePtr<void> context_memory_ = nullptr;
  mutable char model_path_[4096] = {};  // Reserved for max path length
  bool engine_decryption_enable_ = false;
  int (*engine_decryption_)(const char*, char*, size_t*) = nullptr;
  int (*engine_encryption_)(const char*, char*, size_t) = nullptr;
  bool timing_cache_enable_ = false;
  bool force_timing_cache_match_ = false;
  bool detailed_build_log_ = false;
  bool cuda_graph_enable_ = false;
  std::string cache_prefix_;
  bool engine_hw_compatible_ = false;

  // The OrtAllocator object will be get during ep compute time
  // and should be kept for the lifetime of TRT EP object.
  OrtAllocator* alloc_ = nullptr;

  // For create/dump EP context node model
  bool dump_ep_context_model_ = false;
  std::string ep_context_file_path_;
  int ep_context_embed_mode_ = 0;
  std::string ctx_model_path_;
  std::string ep_cache_context_attr_;
  std::string engine_cache_relative_path_to_context_model_dir;
//  std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto_ = ONNX_NAMESPACE::ModelProto::Create();

  std::unordered_set<std::string> control_flow_op_set_ = {"If", "Loop", "Scan"};
//  mutable std::unordered_map<std::string, std::unique_ptr<SubGraphContext>> subgraph_context_map_;

  mutable std::unique_ptr<nvinfer1::IBuilder> builder_;

  // Following maps that hold TRT objects will be accessible by different threads if ORT is using multithreading.
  // In general, TensorRT objects are not thread safe; accesses to an object from different threads must be serialized by the client.
  // But there are still some thread safe operations, please see here https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
  // For those non thread safe operations, TRT EP uses (1) lock_guard or (2) PerThreadContext to make sure synchronization.
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
  std::unordered_map<std::string, ShapeRangesMap> input_shape_ranges_;  // The profile shape ranges that the engine is built with
  std::unordered_map<std::string, std::vector<nvinfer1::IOptimizationProfile*>> profiles_;
  std::unordered_map<std::string, DDSOutputAllocatorMap> dds_output_allocator_maps_;

  // for external stream, we need to create its cudnn/cublass handle before cuda EP enable cuda graph capture
//  cudnnHandle_t external_cudnn_handle_ = nullptr;
//  cublasHandle_t external_cublas_handle_ = nullptr;

  // Call cudaStreamSynchronize() after TRT enqueueV3()
  mutable bool sync_stream_after_enqueue_ = true;

//  CUDAGraph cuda_graph_;
//  bool is_graph_captured_ = false;
  int regular_run_count_before_graph_capture_ = 0;
  // There is chance (currently only happens in CUDA EP) that the second regular run allocates GPU memory for causes like:
  // (1) memory pattern is enabled. (2) arena allocation for stream.
  // Since no GPU memory allocation is allowed during graph capturing, we need at least two regular runs
  // to allocate enough memory in Arena before graph capturing.
  const int min_num_runs_before_cuda_graph_capture_ = 1;  // required min regular runs before graph capture for the necessary memory allocations.

  OrtStatusPtr CreateNodeComputeInfoFromPrecompiledEngine(const OrtGraphViewer* graph_body_viewer, const OrtNode* fused_node,
                                                    std::unordered_map<std::string, size_t>& input_map,
                                                    std::unordered_map<std::string, size_t>& output_map,
                                                    OrtNodeComputeInfo** node_compute_funcs);

  OrtStatusPtr CreateNodeComputeInfoFromGraph(const OrtGraphViewer* graph_body_viewer,
                                        const OrtNode* fused_node,
                                        std::unordered_map<std::string, size_t>& input_map,
                                        std::unordered_map<std::string, size_t>& output_map,
                                        OrtNodeComputeInfo** node_compute_funcs);

  bool IsGraphCaptureAllowed() const { return false; };

  nvinfer1::IBuilder* GetBuilder(TensorrtLogger& trt_logger) const;
};

struct TensorrtExecutionProviderFactory : public OrtExecutionProviderFactory {
    TensorrtExecutionProviderFactory();
};
}

#ifdef __cplusplus
extern "C" {
#endif

EXPORT_API OrtExecutionProviderFactory* RegisterCustomEp();

#ifdef __cplusplus
}
#endif
