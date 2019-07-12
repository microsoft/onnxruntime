// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/logging/logging.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"
#include "core/platform/ort_mutex.h"
#include "core/framework/op_kernel.h"
namespace onnxruntime {
struct NodeComputeInfo;
class TRTLogger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity verbosity_;

 public:
  TRTLogger(Severity verbosity = Severity::kWARNING)
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

std::vector<std::unique_ptr<ComputeCapability>> CheckTensorRtCapability(const onnxruntime::GraphViewer& graph);

// Information to construct kernel function state.
struct TRTFuncState {
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
  cudaStream_t stream;
};

class TRTCompiler {
 public:
  TRTCompiler();
  common::Status Compile(const onnxruntime::Node* fused_nodes, cudaStream_t stream,
                         NodeComputeInfo& node_compute_funcs);

 private:
  OrtMutex tensorrt_mu_;
  size_t max_batch_size_ = 10;
  size_t max_workspace_size_ = 1 << 30;
  std::unordered_map<std::string, unique_pointer<nvonnxparser::IParser>> parsers_;
  std::unordered_map<std::string, unique_pointer<nvinfer1::ICudaEngine>> engines_;
  std::unordered_map<std::string, unique_pointer<nvinfer1::IExecutionContext>> contexts_;
  std::unordered_map<std::string, std::vector<std::vector<int>>> input_info_;
  std::unordered_map<std::string, std::vector<std::vector<int>>> output_info_;
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> output_shapes_;
};
}  // namespace onnxruntime
