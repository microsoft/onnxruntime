// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <ctime>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {

namespace tensorrt_env_vars {
static const std::string kMaxPartitionIterations = "ORT_TENSORRT_MAX_PARTITION_ITERATIONS";
static const std::string kMinSubgraphSize = "ORT_TENSORRT_MIN_SUBGRAPH_SIZE";
static const std::string kMaxWorkspaceSize = "ORT_TENSORRT_MAX_WORKSPACE_SIZE";
static const std::string kFP16Enable = "ORT_TENSORRT_FP16_ENABLE";
static const std::string kDumpSubgraphs = "ORT_TENSORRT_DUMP_SUBGRAPHS";
static const std::string kEngineCacheEnable = "ORT_TENSORRT_ENGINE_CACHE_ENABLE";
static const std::string kEngineCachePath = "ORT_TENSORRT_ENGINE_CACHE_PATH";
}  // namespace tensorrt_env_vars

class TensorrtLogger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity verbosity_;

 public:
  TensorrtLogger(Severity verbosity = Severity::kWARNING)
      : verbosity_(verbosity) {}
  void log(Severity severity, const char* msg) override {
    if (severity <= verbosity_) {
      time_t rawtime = std::time(0);
      char buf[256];
      strftime(&buf[0], 256,
               "%Y-%m-%d %H:%M:%S",
               std::gmtime(&rawtime));
      const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" : severity == Severity::kERROR ? "  ERROR" : severity == Severity::kWARNING ? "WARNING" : severity == Severity::kINFO ? "   INFO" : "UNKNOWN");
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

// Information needed to construct trt execution providers.
struct TensorrtExecutionProviderInfo {
  int device_id{0};
};

// Information to construct kernel function state.
struct TensorrtFuncState {
  AllocateFunc test_allocate_func = nullptr;
  DestroyFunc test_release_func = nullptr;
  AllocatorHandle allocator = nullptr;
  nvonnxparser::IParser* parser = nullptr;
  tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>* engine = nullptr;
  tensorrt_ptr::unique_pointer<nvinfer1::IExecutionContext>* context = nullptr;
  nvinfer1::IBuilder* builder = nullptr;
  nvinfer1::INetworkDefinition* network = nullptr;
  std::vector<std::unordered_map<std::string, int>> input_info;
  std::vector<std::unordered_map<std::string, int>> output_info;
  std::unordered_map<std::string, std::unordered_map<int, std::pair<int64_t, int64_t>>> input_shape_ranges;
  OrtMutex* tensorrt_mu_ptr = nullptr;
  bool* fp16_enable_ptr = nullptr;
  size_t* max_workspace_size_ptr = nullptr;
};

// Logical device representation.
class TensorrtExecutionProvider : public Provider_IExecutionProvider {
 public:
  explicit TensorrtExecutionProvider(const TensorrtExecutionProviderInfo& info);
  virtual ~TensorrtExecutionProvider();

  virtual std::shared_ptr<Provider_KernelRegistry> Provider_GetKernelRegistry() const override;
  std::unique_ptr<Provider_IDataTransfer> Provider_GetDataTransfer() const override;

  std::vector<std::unique_ptr<Provider_ComputeCapability>>
  Provider_GetCapability(const Provider_GraphViewer& graph,
                         const std::vector<const Provider_KernelRegistry*>& /*kernel_registries*/) const override;

  int GetDeviceId() const { return device_id_; }

  common::Status Provider_Compile(const std::vector<Provider_Node*>& fused_nodes,
                                  std::vector<NodeComputeInfo>& node_compute_funcs) override;

  Provider_AllocatorPtr Provider_GetAllocator(int id, OrtMemType mem_type) const override;

 private:
  size_t max_workspace_size_ = 1 << 30;  // 1GB
  int max_partition_iterations_ = 1000;
  int min_subgraph_size_ = 1;
  bool fp16_enable_ = false;
  bool dump_subgraphs_ = false;
  bool engine_cache_enable_ = false;
  std::string engine_cache_path_;
  nvinfer1::IRuntime* runtime_ = nullptr;

  OrtMutex tensorrt_mu_;
  int device_id_;
  std::unordered_map<std::string, tensorrt_ptr::unique_pointer<nvonnxparser::IParser>> parsers_;
  std::unordered_map<std::string, tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>> engines_;
  std::unordered_map<std::string, tensorrt_ptr::unique_pointer<nvinfer1::IExecutionContext>> contexts_;
  std::unordered_map<std::string, tensorrt_ptr::unique_pointer<nvinfer1::IBuilder>> builders_;
  std::unordered_map<std::string, tensorrt_ptr::unique_pointer<nvinfer1::INetworkDefinition>> networks_;
  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, int>>> input_info_;
  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, int>>> output_info_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<int, std::pair<int64_t, int64_t>>>> input_shape_ranges_;

  /**Get IndexedSubGraph based on node list of the subgraph*/
  std::unique_ptr<Provider_IndexedSubGraph> GetSubGraph(SubGraph_t graph_nodes_index, int& kernels_index,
                                                        const onnxruntime::Provider_GraphViewer& graph) const;

  /**
  Get TensorRT supported node lists by calling Onnx-TensorRT parser recursively. Since each time the parser
  can only detect first unsupported node failure, it needs to wait for Onnxruntime to partition the graph
  and then detect next failure again. If there are too many iterations, which means many nodes in the graph
  are not supported by TensorRT, the process will be terminated and the whole graph is simply assigned to
  other execution provider.
  */
  SubGraphCollection_t GetSupportedList(SubGraphCollection_t supported_nodes_list, int iterations, const int max_iterations,
                                        const onnxruntime::Provider_GraphViewer& graph, bool* early_termination) const;

  void RemoveTensorRTGraphCycles(SubGraphCollection_t& supported_nodes_vector, const onnxruntime::Provider_GraphViewer& graph) const;

  Provider_AllocatorPtr allocator_;
};

}  // namespace onnxruntime
