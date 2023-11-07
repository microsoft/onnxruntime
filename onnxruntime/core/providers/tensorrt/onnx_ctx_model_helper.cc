// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <filesystem>

#include "onnx_ctx_model_helper.h"
#include "tensorrt_execution_provider_utils.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime {

/*
 *  Check whether the graph has the EP context contrib op.
 *  The op can contain the precompiled engine info for TRT EP to directly load the engine.
 *
 *  Note: Please see more details about "EPContext" contrib op in contrib_defs.cc
 */
bool GraphHasCtxNode(const GraphViewer& graph_viewer) {
  for (int i = 0; i < graph_viewer.MaxNodeIndex(); ++i) {
    auto node = graph_viewer.GetNode(i);
    if (node != nullptr && node->OpType() == EPCONTEXT_OP) {
      return true;
    }
  }
  return false;
}

const onnxruntime::Path& GetModelPath(const GraphViewer& graph_viewer) {
  // find the top level graph
  const Graph* cur_graph = &graph_viewer.GetGraph();
  while (cur_graph->IsSubgraph()) {
    cur_graph = cur_graph->ParentGraph();
  }

  const Graph& main_graph = *cur_graph;
  return main_graph.ModelPath();
}

std::filesystem::path LocateEngineRelativeToPath(std::string engine_cache_path, const onnxruntime::Path& path) {
  std::filesystem::path base_path(path.ToPathString());
  std::filesystem::path parent_path = base_path.parent_path();
  std::filesystem::path engine_path = parent_path.append(engine_cache_path);
  return engine_path;
}

Status TensorRTCacheModelHandler::GetEpContextFromGraph(const GraphViewer& graph_viewer) {
  if (!ValidateEPCtxNode(graph_viewer)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "It's not a valid EP Context node");
  }
  auto node = graph_viewer.GetNode(0);
  auto& attrs = node->GetAttributes();

  const int64_t embed_mode = attrs.at(EMBED_MODE).i();
  if (embed_mode) {
    // Get engine from byte stream
    const std::string& context_binary = attrs.at(EP_CACHE_CONTEXT).s();
    *(trt_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(trt_runtime_->deserializeCudaEngine(const_cast<char*>(context_binary.c_str()),
                                                                                                static_cast<size_t>(context_binary.length())));
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Read engine as binary data from \"ep_cache_context\" attribute of ep context node and deserialized it";
    if (!(*trt_engine_)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP could not deserialize engine from binary data");
    }
  } else {
    // Get engine from cache file
    std::ifstream engine_file(engine_cache_path_.string(), std::ios::binary | std::ios::in);
    engine_file.seekg(0, std::ios::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engine_buf{new char[engine_size]};
    engine_file.read((char*)engine_buf.get(), engine_size);
    *(trt_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(trt_runtime_->deserializeCudaEngine(engine_buf.get(), engine_size));
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + engine_cache_path_.string();
    if (!(*trt_engine_)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP could not deserialize engine from cache: " + engine_cache_path_.string());
    }
  }
  return Status::OK();
}

/*
 * The sanity check for EP context contrib op.
 */
bool TensorRTCacheModelHandler::ValidateEPCtxNode(const GraphViewer& graph_viewer) {
  assert(graph_viewer.NumberOfNodes() == 1);
  assert(graph_viewer.GetNode(0)->OpType() == EPCONTEXT_OP);
  auto node = graph_viewer.GetNode(0);
  auto& attrs = node->GetAttributes();

  // Check "compute_capability" if it's present
  if (attrs.count(COMPUTE_CAPABILITY) > 0) {
    std::string model_compute_capability = attrs.at(COMPUTE_CAPABILITY).s();
    cudaDeviceProp prop;
    CUDA_CALL_THROW(cudaGetDeviceProperties(&prop, device_id_));
    if (model_compute_capability != GetComputeCapacity(prop)) {
      LOGS_DEFAULT(ERROR) << "The compute capability of the engine cache doesn't match with the GPU's compute capability";
      return false;
    }
  }

  // "embed_mode" attr and "ep_cache_context" attr should be present
  if (attrs.count(EMBED_MODE) > 0 && attrs.count(EP_CACHE_CONTEXT) > 0) {
    // ep_cache_context: payload of the execution provider context if embed_mode=1, or path to the context file if embed_mode=0
    const int64_t embed_mode = attrs.at(EMBED_MODE).i();

    // engine cache path
    if (embed_mode == 0) {
      // First assume engine cache path is relatvie to model path,
      // If not, then assume the engine cache path is an absolute path.
      engine_cache_path_ = LocateEngineRelativeToPath(attrs.at(EP_CACHE_CONTEXT).s(), GetModelPath(graph_viewer));
      auto default_engine_cache_path_ = engine_cache_path_;
      if (!std::filesystem::exists(engine_cache_path_)) {
        engine_cache_path_.assign(attrs.at(EP_CACHE_CONTEXT).s());
        if (!std::filesystem::exists(engine_cache_path_)) {
          LOGS_DEFAULT(ERROR) << "Can't find " << default_engine_cache_path_.string() << " or " << engine_cache_path_.string() << " TensorRT engine";
          return false;
        }
      }
    }
  }
  return true;
}
}  // namespace onnxruntime
