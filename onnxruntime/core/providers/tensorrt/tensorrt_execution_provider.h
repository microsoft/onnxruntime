// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <ctime>
#include "core/common/logging/logging.h"
#include "core/framework/op_kernel.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {

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
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;
  std::vector<std::vector<int>> input_info;
  std::vector<std::vector<int>> output_info;
  std::vector<std::vector<int64_t>> output_shapes;
  OrtMutex* tensorrt_mu_ptr = nullptr;
};

// Logical device representation.
class TensorrtExecutionProvider : public IExecutionProvider {
 public:
  explicit TensorrtExecutionProvider(const TensorrtExecutionProviderInfo& info);
  virtual ~TensorrtExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  int GetDeviceId() const { return device_id_; }

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  void SetMaxBatchSize(const int batch_size) {
    max_batch_size_ = batch_size;
  }

  void SetMaxWorkspaceSize(const size_t workspace_size) {
    max_workspace_size_ = workspace_size;
  }

 private:
  int max_batch_size_ = 1;
  size_t max_workspace_size_ = 1 << 30;  // 1GB
  int max_parser_iterations_ = 6;

  struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
      if (obj) {
        obj->destroy();
      }
    }
  };

  template <typename T>
  using unique_pointer = std::unique_ptr<T, InferDeleter>;

  OrtMutex tensorrt_mu_;
  int device_id_;
  std::unordered_map<std::string, unique_pointer<nvonnxparser::IParser>> parsers_;
  std::unordered_map<std::string, unique_pointer<nvinfer1::ICudaEngine>> engines_;
  std::unordered_map<std::string, unique_pointer<nvinfer1::IExecutionContext>> contexts_;
  std::unordered_map<std::string, std::vector<std::vector<int>>> input_info_;
  std::unordered_map<std::string, std::vector<std::vector<int>>> output_info_;
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> output_shapes_;

  /**Get IndexedSubGraph based on node list of the subgraph*/
  std::unique_ptr<IndexedSubGraph> GetSubGraph(SubGraph_t graph_nodes_index, int& kernels_index,
                                               const onnxruntime::GraphViewer& graph) const;

  /**
  Get TensorRT supported node lists by calling Onnx-TensorRT parser recursively. Since each time the parser
  can only detect first unsupported node failure, it needs to wait for Onnxruntime to partition the graph
  and then detect next failure again. If there are too many iterations, which means many nodes in the graph
  are not supported by TensorRT, the process will be terminated and the whole graph is simply assigned to
  other execution provider.
  */
  SubGraphCollection_t GetSupportedList(SubGraphCollection_t supported_nodes_list, int iterations, const int max_iterations,
                                        const onnxruntime::GraphViewer& graph, bool* early_termination) const;
};

}  // namespace onnxruntime
