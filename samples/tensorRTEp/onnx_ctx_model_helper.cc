#include <cassert>
#include <iostream>
#include <fstream>
#include "onnx_ctx_model_helper.h"
#include "tensorrt_execution_provider.h"

namespace onnxruntime {

bool GraphHasCtxNode(const OrtGraphViewer* graph_viewer) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  int maxNodeIndex = 0;
  api->OrtGraph_MaxNodeIndex(graph_viewer, &maxNodeIndex);
  for (int i = 0; i < maxNodeIndex; ++i) {
    const OrtNode* node = nullptr;
    api->OrtGraph_GetOrtNode(graph_viewer, i, &node);
    if (node == nullptr) {
      continue;
    }
    const char* opType = nullptr;
    api->OrtNode_GetOpType(node, &opType);
    if (strcmp(opType, EPCONTEXT_OP.c_str()) == 0) {
      return true;
    }
  }
  return false;
}

/*
 * Return the directory where the ep context model locates
 */
std::filesystem::path GetPathOrParentPathOfCtxModel(const std::string& ep_context_file_path) {
  if (ep_context_file_path.empty()) {
    return std::filesystem::path();
  }
  std::filesystem::path ctx_path(ep_context_file_path);
  if (std::filesystem::is_directory(ep_context_file_path)) {
    return ctx_path;
  } else {
    return ctx_path.parent_path();
  }
}

std::string GetCtxModelPath(const std::string& ep_context_file_path,
                            const std::string& original_model_path) {
  std::string ctx_model_path;

  if (!ep_context_file_path.empty() && !std::filesystem::is_directory(ep_context_file_path)) {
    ctx_model_path = ep_context_file_path;
  } else {
    std::filesystem::path model_path = original_model_path;
    std::filesystem::path model_name_stem = model_path.stem();  // model_name.onnx -> model_name
    std::string ctx_model_name = model_name_stem.string() + "_ctx.onnx";

    if (std::filesystem::is_directory(ep_context_file_path)) {
      std::filesystem::path model_directory = ep_context_file_path;
      ctx_model_path = model_directory.append(ctx_model_name).string();
    } else {
      ctx_model_path = ctx_model_name;
    }
  }
  return ctx_model_path;
}

bool IsAbsolutePath(const std::string& path_string) {
#ifdef _WIN32
  onnxruntime::PathString ort_path_string = onnxruntime::ToPathString(path_string);
  auto path = std::filesystem::path(ort_path_string.c_str());
  return path.is_absolute();
#else
  if (!path_string.empty() && path_string[0] == '/') {
    return true;
  }
  return false;
#endif
}

// Like "../file_path"
bool IsRelativePathToParentPath(const std::string& path_string) {
#ifdef _WIN32
  onnxruntime::PathString ort_path_string = onnxruntime::ToPathString(path_string);
  auto path = std::filesystem::path(ort_path_string.c_str());
  auto relative_path = path.lexically_normal().make_preferred().wstring();
  if (relative_path.find(L"..", 0) != std::string::npos) {
    return true;
  }
  return false;
#else
  if (!path_string.empty() && path_string.find("..", 0) != std::string::npos) {
    return true;
  }
  return false;
#endif
}

/*
 * Get the weight-refitted engine cache path from a weight-stripped engine cache path
 *
 * Weight-stipped engine:
 * An engine with weights stripped and its size is smaller than a regualr engine.
 * The cache name of weight-stripped engine is TensorrtExecutionProvider_TRTKernel_XXXXX.stripped.engine
 *
 * Weight-refitted engine:
 * An engine that its weights have been refitted and it's simply a regular engine.
 * The cache name of weight-refitted engine is TensorrtExecutionProvider_TRTKernel_XXXXX.engine
 */
std::string GetWeightRefittedEnginePath(std::string stripped_engine_cache) {
  std::filesystem::path stripped_engine_cache_path(stripped_engine_cache);
  std::string refitted_engine_cache_path = stripped_engine_cache_path.stem().stem().string() + ".engine";
  return refitted_engine_cache_path;
}

bool IsWeightStrippedEngineCache(std::filesystem::path& engine_cache_path) {
  // The weight-stripped engine cache has the naming of xxx.stripped.engine
  return engine_cache_path.stem().extension().string() == ".stripped";
}

OrtStatusPtr TensorRTCacheModelHandler::GetEpContextFromGraph(const OrtGraphViewer* graph_viewer) {
  if (!ValidateEPCtxNode(graph_viewer)) {
    return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL, "It's not a valid EP Context node");
  }
  const OrtNode* node = nullptr;
  api_->OrtGraph_GetOrtNode(graph_viewer, 0, &node);

  const int64_t embed_mode = api_->OrtNode_GetAttributeInt(node, EMBED_MODE.c_str());
  if (embed_mode) {
    // Get engine from byte stream.
    const std::string& context_binary(api_->OrtNode_GetAttributeStr(node, EP_CACHE_CONTEXT.c_str()));
    *(trt_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(trt_runtime_->deserializeCudaEngine(const_cast<char*>(context_binary.c_str()),
                                                                                                static_cast<size_t>(context_binary.length())));
//    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Read engine as binary data from \"ep_cache_context\" attribute of ep context node and deserialized it";
    if (!(*trt_engine_)) {
      return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL, "TensorRT EP could not deserialize engine from binary data");
    }
  } else {
    // Get engine from cache file.
    std::string cache_path(api_->OrtNode_GetAttributeStr(node, EP_CACHE_CONTEXT.c_str()));

    // For security purpose, in the case of running context model, TRT EP won't allow
    // engine cache path to be the relative path like "../file_path" or the absolute path.
    // It only allows the engine cache to be in the same directory or sub directory of the context model.
    if (IsAbsolutePath(cache_path)) {
      return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL, std::string("For security purpose, the ep_cache_context attribute should be set with a relative path, but it is an absolute path:  " + cache_path).c_str());
    }
    if (IsRelativePathToParentPath(cache_path)) {
      return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL, "The file path in ep_cache_context attribute has '..'. For security purpose, it's not allowed to point outside the directory.");
    }

    // The engine cache and context model (current model) should be in the same directory
    std::filesystem::path ctx_model_dir(GetPathOrParentPathOfCtxModel(ep_context_model_path_));
    auto engine_cache_path = ctx_model_dir.append(cache_path);
//    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] GetEpContextFromGraph engine_cache_path: " + engine_cache_path.string();

    // If it's a weight-stripped engine cache, it needs to be refitted even though the refit flag is not enabled
    if (!weight_stripped_engine_refit_) {
      weight_stripped_engine_refit_ = IsWeightStrippedEngineCache(engine_cache_path);
    }

    // If the serialized refitted engine is present, use it directly without refitting the engine again
    if (weight_stripped_engine_refit_) {
      const std::filesystem::path refitted_engine_cache_path = GetWeightRefittedEnginePath(engine_cache_path.string());
      if (std::filesystem::exists(refitted_engine_cache_path)) {
//        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] " + refitted_engine_cache_path.string() + " exists.";
        engine_cache_path = refitted_engine_cache_path.string();
        weight_stripped_engine_refit_ = false;
      }
    }

    if (!std::filesystem::exists(engine_cache_path)) {
      return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL,
                             std::string("TensorRT EP can't find engine cache: " + engine_cache_path.string() +
                                 ". Please make sure engine cache is in the same directory or sub-directory of context model.").c_str());
    }

    std::ifstream engine_file(engine_cache_path.string(), std::ios::binary | std::ios::in);
    engine_file.seekg(0, std::ios::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engine_buf{new char[engine_size]};
    engine_file.read((char*)engine_buf.get(), engine_size);
    *(trt_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(trt_runtime_->deserializeCudaEngine(engine_buf.get(), engine_size));
    if (!(*trt_engine_)) {
      return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL,
                             std::string("TensorRT EP could not deserialize engine from cache: " + engine_cache_path.string()).c_str());
    }
//    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + engine_cache_path.string();

    if (weight_stripped_engine_refit_) {
      const std::string onnx_model_filename(api_->OrtNode_GetAttributeStr(node, ONNX_MODEL_FILENAME.c_str()));
      std::string weight_stripped_engine_cache = engine_cache_path.string();
      auto status = TensorrtExecutionProvider::RefitEngine(onnx_model_filename,
                                                           onnx_model_folder_path_,
                                                           weight_stripped_engine_cache,
                                                           true /* path check for security */,
                                                           (*trt_engine_).get(),
                                                           true /* serialize refitted engine to disk */,
                                                           detailed_build_log_);
      if (status != nullptr) {
        return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL, api_->GetErrorMessage(status));
      }
    }
  }
  return nullptr;
}

bool TensorRTCacheModelHandler::ValidateEPCtxNode(const OrtGraphViewer* graph_viewer) {
  assert(api_->OrtGraph_NumberOfNodes(graph_viewer) == 1);
  const OrtNode* node = nullptr;
  api_->OrtGraph_GetOrtNode(graph_viewer, 0, &node);
  const char* opType = nullptr;
  api_->OrtNode_GetOpType(node, &opType);
  assert(strcmp(opType, EPCONTEXT_OP.c_str()) == 0);

  size_t key_count = 0;
  api_->OrtNode_GetAttributeKeyCount(node, COMPUTE_CAPABILITY.c_str(), &key_count);
  // Show the warning if compute capability is not matched
  if (key_count > 0) {
    const char* model_compute_capability = api_->OrtNode_GetAttributeStr(node, COMPUTE_CAPABILITY.c_str());
    // Verify if engine was compiled with ampere+ hardware compatibility enabled
    if (strcmp(model_compute_capability, "80+") == 0) {
//      if (std::stoi(compute_capability_) < 80) {
//        LOGS_DEFAULT(WARNING) << "[TensorRT EP] However, this GPU doesn't match. The compute capability of the GPU: " << compute_capability_;
//      }
    } else if (strcmp(model_compute_capability, compute_capability_.c_str()) != 0) {
//      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Engine was compiled for a different compatibility level and might not work or perform suboptimal";
//      LOGS_DEFAULT(WARNING) << "[TensorRT EP] The compute capability of the engine: " << model_compute_capability;
//      LOGS_DEFAULT(WARNING) << "[TensorRT EP] The compute capability of the GPU: " << compute_capability_;
    }
  }

  // "embed_mode" attr and "ep_cache_context" attr should be present
  api_->OrtNode_GetAttributeKeyCount(node, EMBED_MODE.c_str(), &key_count);
  assert(key_count > 0);
  api_->OrtNode_GetAttributeKeyCount(node, EP_CACHE_CONTEXT.c_str(), &key_count);
  assert(key_count > 0);

  const int64_t embed_mode = api_->OrtNode_GetAttributeInt(node, EMBED_MODE.c_str());
  if (embed_mode == 1) {
    // engine binary data
//    LOGS_DEFAULT(WARNING) << EPCONTEXT_WARNING;
  }

  return true;
}
}
