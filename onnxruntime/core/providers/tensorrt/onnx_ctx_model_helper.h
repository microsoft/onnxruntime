// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <filesystem>

#include "core/providers/tensorrt/nv_includes.h"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {

static const std::string EPCONTEXT_OP = "EPContext";
static const std::string EMBED_MODE = "embed_mode";
static const std::string EP_CACHE_CONTEXT = "ep_cache_context";
static const std::string COMPUTE_CAPABILITY = "hardware_architecture";
static const std::string EPCONTEXT_OP_DOMAIN = "com.microsoft";
static const std::string EPCONTEXT_WARNING =
    "It's suggested to set the ORT graph optimization level to 0 and  \
                                              make \"embed_mode\" to 0 (\"ep_cache_context\" is the cache path)\
                                              for the best model loading time";

bool GraphHasCtxNode(const GraphViewer& graph_viewer);
const onnxruntime::Path& GetModelPath(const GraphViewer& graph_viewer);
std::filesystem::path GetPathOrParentPathOfCtxModel(const std::string& ep_context_file_path);
ONNX_NAMESPACE::ModelProto* CreateCtxModel(const GraphViewer& graph_viewer,
                                           const std::string engine_cache_path,
                                           char* engine_data,
                                           size_t size,
                                           const int64_t embed_mode,
                                           std::string compute_capability,
                                           const logging::Logger* logger);
std::string GetCtxModelPath(const std::string& ep_context_file_path,
                            const std::string& original_model_path);
bool IsAbsolutePath(std::string& path_string);
bool IsRelativePathToParentPath(std::string& path_string);
void DumpCtxModel(ONNX_NAMESPACE::ModelProto* model_proto,
                  const std::string& ctx_model_path);
void UpdateCtxNodeModelEngineContext(ONNX_NAMESPACE::ModelProto* model_proto,
                                     char* engine_data,
                                     size_t size);

class TensorRTCacheModelHandler {
 public:
  TensorRTCacheModelHandler(std::unique_ptr<nvinfer1::ICudaEngine>* trt_engine,
                            nvinfer1::IRuntime* trt_runtime,
                            std::string ep_context_model_path,
                            std::string compute_capability) : trt_engine_(trt_engine), trt_runtime_(trt_runtime), ep_context_model_path_(ep_context_model_path), compute_capability_(compute_capability) {
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TensorRTCacheModelHandler);

  bool ValidateEPCtxNode(const GraphViewer& graph_viewer);

  Status GetEpContextFromGraph(const GraphViewer& graph_viewer);

 private:
  std::unique_ptr<nvinfer1::ICudaEngine>* trt_engine_;
  nvinfer1::IRuntime* trt_runtime_;
  std::string ep_context_model_path_;  // If using context model, it implies context model and engine cache is in the same directory
  std::string compute_capability_;
};  // TRTCacheModelHandler
}  // namespace onnxruntime
