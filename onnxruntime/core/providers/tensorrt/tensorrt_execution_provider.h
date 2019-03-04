// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <ctime>
#include "core/framework/op_kernel.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

namespace onnxruntime {

static const int kBatchSize = 1;
static const int max_batch_size = 16;
static const int max_workspace_size = 16 << 20;

#define CHECK(status)                       \
  do {                                      \
    auto ret = (status);                    \
    if (ret != 0) {                         \
      std::cout << "Cuda failure: " << ret; \
      abort();                              \
    }                                       \
  } while (0)

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

class TensorrtLogger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity verbosity_;
  std::ostream* ostream_;

 public:
  TensorrtLogger(Severity verbosity = Severity::kWARNING,
            std::ostream& ostream = std::cout)
      : verbosity_(verbosity), ostream_(&ostream) {}
  void log(Severity severity, const char* msg) override {
    ORT_UNUSED_PARAMETER(severity);
    ORT_UNUSED_PARAMETER(msg);
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
};

// Logical device representation.
class TensorrtExecutionProvider : public IExecutionProvider {
 public:
  TensorrtExecutionProvider();
  virtual ~TensorrtExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

 private:
  int device_id_;
  std::unordered_map<std::string, unique_pointer<nvonnxparser::IParser>> parsers_;
  std::unordered_map<std::string, unique_pointer<nvinfer1::ICudaEngine>> engines_;
  std::unordered_map<std::string, unique_pointer<nvinfer1::IExecutionContext>> contexts_;
  std::unordered_map<std::string, std::vector<std::vector<int>>> input_info_;
  std::unordered_map<std::string, std::vector<std::vector<int>>> output_info_;
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> output_shapes_;
};

}  // namespace onnxruntime

