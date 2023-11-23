// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <filesystem>

#include "NvInfer.h"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {

static const std::string EPCONTEXT_OP = "EPContext";
static const std::string MAIN_CONTEXT = "main_context";
static const std::string EMBED_MODE = "embed_mode";
static const std::string EP_CACHE_CONTEXT = "ep_cache_context";
static const std::string EP_SDK_VER = "ep_sdk_version";
static const std::string COMPUTE_CAPABILITY = "hardware_arch";
static const std::string PARTITION_NAME = "partition_name";
static const std::string SOURCE = "source";
static const std::string EPCONTEXT_OP_DOMAIN = "com.microsoft";

bool GraphHasCtxNode(const GraphViewer& graph_viewer);
const onnxruntime::Path& GetModelPath(const GraphViewer& graph_viewer);
std::filesystem::path LocateEngineRelativeToPath(std::string engine_cache_path, const onnxruntime::Path& path);
ONNX_NAMESPACE::ModelProto* CreateCtxNodeModel(const GraphViewer& graph_viewer,
                                               const std::string engine_cache_path,
                                               char* engine_data,
                                               size_t size,
                                               const int64_t embed_mode,
                                               const logging::Logger* logger);
void DumpCtxNodeModel(ONNX_NAMESPACE::ModelProto* model_proto,
                      const std::string engine_cache_path);
void UpdateCtxNodeModelEngineContext(ONNX_NAMESPACE::ModelProto* model_proto,
                                     char* engine_data,
                                     size_t size);

class TensorRTCacheModelHandler {
 public:
  TensorRTCacheModelHandler(std::unique_ptr<nvinfer1::ICudaEngine>* trt_engine,
                            nvinfer1::IRuntime* trt_runtime,
                            int device_id = 0) : trt_engine_(trt_engine), trt_runtime_(trt_runtime), device_id_(device_id) {
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TensorRTCacheModelHandler);

  bool ValidateEPCtxNode(const GraphViewer& graph_viewer);

  Status GetEpContextFromGraph(const GraphViewer& graph_viewer);

 private:
  std::unique_ptr<nvinfer1::ICudaEngine>* trt_engine_;
  nvinfer1::IRuntime* trt_runtime_;
  std::filesystem::path engine_cache_path_;
  int device_id_ = 0;
};  // TRTCacheModelHandler
}  // namespace onnxruntime
