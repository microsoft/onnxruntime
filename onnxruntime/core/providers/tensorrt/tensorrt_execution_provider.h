// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <ctime>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "core/platform/ort_mutex.h"
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
static const std::string kDumpSubgraphs = "ORT_TENSORRT_DUMP_SUBGRAPHS";
static const std::string kEngineCacheEnable = "ORT_TENSORRT_ENGINE_CACHE_ENABLE";
static const std::string kCachePath = "ORT_TENSORRT_CACHE_PATH";
static const std::string kForceSequentialEngineBuild= "ORT_TENSORRT_FORCE_SEQUENTIAL_ENGINE_BUILD";
// Old env variable for backward compatibility
static const std::string kEngineCachePath = "ORT_TENSORRT_ENGINE_CACHE_PATH";
static const std::string kDecryptionEnable = "ORT_TENSORRT_ENGINE_DECRYPTION_ENABLE";
static const std::string kDecryptionLibPath = "ORT_TENSORRT_ENGINE_DECRYPTION_LIB_PATH";
static const std::string kDLAEnable = "ORT_TENSORRT_DLA_ENABLE";
static const std::string kDLACore = "ORT_TENSORRT_DLA_CORE";
}  // namespace tensorrt_env_vars

class TensorrtLogger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity verbosity_;

 public:
  TensorrtLogger(Severity verbosity = Severity::kWARNING)
      : verbosity_(verbosity) {}
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= verbosity_) {
      time_t rawtime = std::time(0);
      char buf[256];
      strftime(&buf[0], 256,
               "%Y-%m-%d %H:%M:%S",
               std::gmtime(&rawtime));
      const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" : severity == Severity::kERROR ? "  ERROR"
                                                                            : severity == Severity::kWARNING ? "WARNING"
                                                                            : severity == Severity::kINFO    ? "   INFO"
                                                                                                             : "UNKNOWN");
      if (severity <= Severity::kERROR)
        LOGS_DEFAULT(ERROR) << "[" << buf << " " << sevstr << "] " << msg;
      else
        LOGS_DEFAULT(WARNING) << "[" << buf << " " << sevstr << "] " << msg;
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
  tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>* engine = nullptr;
  tensorrt_ptr::unique_pointer<nvinfer1::IExecutionContext>* context = nullptr;
  tensorrt_ptr::unique_pointer<nvinfer1::IBuilder>* builder = nullptr;
  tensorrt_ptr::unique_pointer<nvinfer1::INetworkDefinition>* network = nullptr;
  std::vector<std::unordered_map<std::string, int>> input_info;
  std::vector<std::unordered_map<std::string, int>> output_info;
  std::unordered_map<std::string, std::unordered_map<int, std::pair<int64_t, int64_t>>> input_shape_ranges;
  OrtMutex* tensorrt_mu_ptr = nullptr;
  bool fp16_enable;
  bool int8_enable;
  bool dla_enable;
  int dla_core;
  size_t* max_workspace_size_ptr = nullptr;
  std::string trt_node_name_with_precision;
  bool engine_cache_enable;
  std::string engine_cache_path;
  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::IOptimizationProfile* trt_profile = nullptr;
  AllocatorPtr scratch_allocator;
  std::unordered_map<std::string, float> dynamic_range_map;
  bool engine_decryption_enable;
  int (*engine_decryption)(const char*, char*, size_t*);
};

// Logical device representation.
class TensorrtExecutionProvider : public IExecutionProvider {
 public:
  explicit TensorrtExecutionProvider(const TensorrtExecutionProviderInfo& info);
  virtual ~TensorrtExecutionProvider();

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  int GetDeviceId() const { return device_id_; }

  common::Status Compile(const std::vector<Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;

  void RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager) override;

  Status OnRunEnd() override;

  Status SetComputeStream(void* stream) override;

  void* GetComputeStream() const override { return static_cast<void*>(stream_); }

 private:
  bool external_stream_ = false;
  cudaStream_t stream_ = nullptr;
  int max_partition_iterations_ = 1000;
  int min_subgraph_size_ = 1;
  size_t max_workspace_size_ = 1 << 30;  // 1GB
  bool fp16_enable_ = false;
  bool int8_enable_ = false;
  bool dla_enable_ = false;
  int dla_core_ = 0;
  bool force_sequential_engine_build_ = false;
  std::string int8_calibration_cache_name_ = "INT8_calibration_table";
  bool int8_use_native_tensorrt_calibration_table_ = false;
  bool dump_subgraphs_ = false;
  bool engine_cache_enable_ = false;
  std::string cache_path_;
  tensorrt_ptr::unique_pointer<nvinfer1::IRuntime> runtime_ = nullptr;
  OrtMutex tensorrt_mu_;
  int device_id_;
  AllocatorPtr allocator_;
  mutable char model_path_[4096];  // Reserved for max path length
  bool engine_decryption_enable_ = false;
  int (*engine_decryption_)(const char*, char*, size_t*);

  std::unordered_map<std::string, tensorrt_ptr::unique_pointer<nvonnxparser::IParser>> parsers_;
  std::unordered_map<std::string, tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>> engines_;
  std::unordered_map<std::string, tensorrt_ptr::unique_pointer<nvinfer1::IExecutionContext>> contexts_;
  std::unordered_map<std::string, tensorrt_ptr::unique_pointer<nvinfer1::IBuilder>> builders_;
  std::unordered_map<std::string, tensorrt_ptr::unique_pointer<nvinfer1::INetworkDefinition>> networks_;
  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, int>>> input_info_;
  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, int>>> output_info_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<int, std::pair<int64_t, int64_t>>>> input_shape_ranges_;

  /**Get IndexedSubGraph based on node list of the subgraph*/
  std::unique_ptr<IndexedSubGraph> GetSubGraph(SubGraph_t graph_nodes_index,
                                               const GraphViewer& graph) const;

  /**
  Get TensorRT supported node lists by calling Onnx-TensorRT parser recursively. Since each time the parser
  can only detect first unsupported node failure, it needs to wait for Onnxruntime to partition the graph
  and then detect next failure again. If there are too many iterations, which means many nodes in the graph
  are not supported by TensorRT, the process will be terminated and the whole graph is simply assigned to
  other execution provider.
  */
  SubGraphCollection_t GetSupportedList(SubGraphCollection_t supported_nodes_list, int iterations, const int max_iterations,
                                        const GraphViewer& graph, bool* early_termination) const;

  void RemoveTensorRTGraphCycles(SubGraphCollection_t& supported_nodes_vector, const GraphViewer& graph) const;

  /** 
  Get a unique_lock object to control the concurrency behavior of TensorRT engine building. When force_sequential_engine_build
  is set to true, the lock object is associated with a mutex shared across all providers to enforce sequential engine build. 
  Otherwise, the constructed unique_lock is not associated with any mutex therefore no locking/unlocking will happen.
  */
  std::unique_lock<OrtMutex> GetEngineBuildLock() const;
};
}  // namespace onnxruntime
