// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <filesystem>
#include <memory>

#include "core/providers/nv_tensorrt_rtx/nv_includes.h"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {

static const std::string EPCONTEXT_OP = "EPContext";
static const std::string EMBED_MODE = "embed_mode";
static const std::string EP_CACHE_CONTEXT = "ep_cache_context";
static const std::string COMPUTE_CAPABILITY = "hardware_architecture";
static const std::string ONNX_MODEL_FILENAME = "onnx_model_filename";
static const std::string EPCONTEXT_OP_DOMAIN = "com.microsoft";
static const std::string EPCONTEXT_WARNING =
    "It's suggested to set the ORT graph optimization level to 0 and  \
                                              make \"embed_mode\" to 0 (\"ep_cache_context\" is the cache path)\
                                              for the best model loading time";

bool GraphHasCtxNode(const GraphViewer& graph_viewer);
const std::filesystem::path& GetModelPath(const GraphViewer& graph_viewer);
std::filesystem::path GetPathOrParentPathOfCtxModel(const std::string& ep_context_file_path);
ONNX_NAMESPACE::ModelProto* CreateCtxModel(const GraphViewer& graph_viewer,
                                           const std::string engine_cache_path,
                                           char* engine_data,
                                           size_t size,
                                           const int64_t embed_mode,
                                           const std::string compute_capability,
                                           const std::string onnx_model_path,
                                           const logging::Logger* logger);
std::string GetCtxModelPath(const std::string& ep_context_file_path,
                            const std::string& original_model_path);
bool IsAbsolutePath(const std::string& path_string);
bool IsRelativePathToParentPath(const std::string& path_string);
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
                            std::string compute_capability,
                            bool weight_stripped_engine_refit,
                            std::string onnx_model_folder_path,
                            const void* onnx_model_bytestream,
                            size_t onnx_model_bytestream_size,
                            bool detailed_build_log)
      : trt_engine_(trt_engine),
        trt_runtime_(trt_runtime),
        ep_context_model_path_(ep_context_model_path),
        compute_capability_(compute_capability),
        weight_stripped_engine_refit_(weight_stripped_engine_refit),
        onnx_model_folder_path_(onnx_model_folder_path),
        onnx_model_bytestream_(onnx_model_bytestream),
        onnx_model_bytestream_size_(onnx_model_bytestream_size),
        detailed_build_log_(detailed_build_log) {
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TensorRTCacheModelHandler);

  bool ValidateEPCtxNode(const GraphViewer& graph_viewer);

  Status GetEpContextFromGraph(const GraphViewer& graph_viewer);

 private:
  std::unique_ptr<nvinfer1::ICudaEngine>* trt_engine_;
  nvinfer1::IRuntime* trt_runtime_;
  std::string ep_context_model_path_;  // If using context model, it implies context model and engine cache is in the same directory
  std::string compute_capability_;
  bool weight_stripped_engine_refit_;
  std::string onnx_model_folder_path_;
  const void* onnx_model_bytestream_;
  size_t onnx_model_bytestream_size_;
  bool detailed_build_log_;
};  // TRTCacheModelHandler
}  // namespace onnxruntime
