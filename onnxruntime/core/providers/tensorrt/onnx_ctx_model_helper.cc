// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <filesystem>

#include "onnx_ctx_model_helper.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/framework/execution_provider.h"
#include "tensorrt_execution_provider.h"

namespace onnxruntime {
extern TensorrtLogger& GetTensorrtLogger(bool verbose_log);

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

const std::filesystem::path& GetModelPath(const GraphViewer& graph_viewer) {
  // find the top level graph
  const Graph* cur_graph = &graph_viewer.GetGraph();
  while (cur_graph->IsSubgraph()) {
    cur_graph = cur_graph->ParentGraph();
  }

  const Graph& main_graph = *cur_graph;
  return main_graph.ModelPath();
}

/*
 * Update ep_cache_context attribute of the EP context node with the given engine binary data
 */
void UpdateCtxNodeModelEngineContext(ONNX_NAMESPACE::ModelProto* model_proto,
                                     char* engine_data,
                                     size_t size) {
  ONNX_NAMESPACE::GraphProto* graph_proto = model_proto->mutable_graph();
  ONNX_NAMESPACE::NodeProto* node_proto = graph_proto->mutable_node(0);

  for (int i = 0; i < node_proto->attribute_size(); ++i) {
    ONNX_NAMESPACE::AttributeProto* attribute_proto = node_proto->mutable_attribute(i);
    if (attribute_proto->name() == EP_CACHE_CONTEXT) {
      std::string engine_data_str = "";
      if (size > 0) {
        engine_data_str.assign(engine_data, size);
      }
      attribute_proto->set_s(engine_data_str);
    }
  }
}

/*
 * Create "EP context node" model where engine information is embedded
 */
ONNX_NAMESPACE::ModelProto* CreateCtxModel(const GraphViewer& graph_viewer,
                                           const std::string engine_cache_path,
                                           char* engine_data,
                                           size_t size,
                                           const int64_t embed_mode,
                                           const std::string compute_capability,
                                           const std::string onnx_model_path,
                                           const logging::Logger* logger) {
  auto model_build = graph_viewer.CreateModel(*logger);
  auto& graph_build = model_build->MainGraph();

  // Get graph inputs and outputs
  std::vector<onnxruntime::NodeArg*> inputs, outputs;
  for (auto input : graph_viewer.GetInputs()) {
    auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
    inputs.push_back(&n_input);
  }

  for (auto output : graph_viewer.GetOutputs()) {
    auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
    outputs.push_back(&n_output);
  }

  // Create EP context node attributes
  auto attr_0 = ONNX_NAMESPACE::AttributeProto::Create();  // embed_mode
  auto attr_1 = ONNX_NAMESPACE::AttributeProto::Create();  // ep_cache_context
  auto attr_2 = ONNX_NAMESPACE::AttributeProto::Create();  // hardware_architecture
  auto attr_3 = ONNX_NAMESPACE::AttributeProto::Create();  // onnx_model_filename
  std::string engine_data_str = "";
  attr_0->set_name(EMBED_MODE);
  attr_0->set_type(onnx::AttributeProto_AttributeType_INT);
  attr_0->set_i(embed_mode);
  attr_1->set_name(EP_CACHE_CONTEXT);
  attr_1->set_type(onnx::AttributeProto_AttributeType_STRING);
  if (embed_mode) {
    if (size > 0) {
      engine_data_str.assign(engine_data, size);
    }
    attr_1->set_s(engine_data_str);
    LOGS_DEFAULT(WARNING) << EPCONTEXT_WARNING;
  } else {
    attr_1->set_s(engine_cache_path);
  }
  attr_2->set_name(COMPUTE_CAPABILITY);
  attr_2->set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_2->set_s(compute_capability);
  attr_3->set_name(ONNX_MODEL_FILENAME);
  attr_3->set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_3->set_s(std::filesystem::path(onnx_model_path).filename().string());

  auto node_attributes = ONNX_NAMESPACE::NodeAttributes::Create();
  constexpr int num_attributes = 4;
  node_attributes->reserve(num_attributes);
  node_attributes->emplace(EMBED_MODE, *attr_0);
  node_attributes->emplace(EP_CACHE_CONTEXT, *attr_1);
  node_attributes->emplace(COMPUTE_CAPABILITY, *attr_2);
  node_attributes->emplace(ONNX_MODEL_FILENAME, *attr_3);

  // Create EP context node
  graph_build.AddNode(EPCONTEXT_OP, EPCONTEXT_OP, "", inputs, outputs, node_attributes.get(), EPCONTEXT_OP_DOMAIN);
  ORT_ENFORCE(graph_build.Resolve().IsOK());

  // Serialize modelproto to string
  auto new_graph_viewer = graph_build.CreateGraphViewer();
  auto& metadata = graph_viewer.GetGraph().GetModel().MetaData();
  auto model = new_graph_viewer->CreateModel(*logger, metadata);
  auto model_proto = model->ToProto();
  new_graph_viewer->ToProto(*model_proto->mutable_graph(), true, true);
  model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  return model_proto.release();
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

/*
 * Get "EP context" model path.
 *
 * Function logic:
 * If ep_context_file_path is provided,
 *     - If ep_context_file_path is a file, return "ep_context_file_path".
 *     - If ep_context_file_path is a directory, return "ep_context_file_path/original_model_name_ctx.onnx".
 * If ep_context_file_path is not provided,
 *     - Return "original_model_name_ctx.onnx".
 *
 * TRT EP has rules about context model path and engine cache path (see tensorrt_execution_provider.cc):
 * - If dump_ep_context_model_ and engine_cache_enabled_ is enabled, TRT EP will dump context model and save engine cache
 *   to the same directory provided by ep_context_file_path_. (i.e. engine_cache_path_ = ep_context_file_path_)
 *
 * Example 1:
 * ep_context_file_path = "/home/user/ep_context_model_directory"
 * original_model_path = "model.onnx"
 * => return "/home/user/ep_context_model_folder/model_ctx.onnx"
 *
 * Example 2:
 * ep_context_file_path = "my_ctx_model.onnx"
 * original_model_path = "model.onnx"
 * => return "my_ctx_model.onnx"
 *
 * Example 3:
 * ep_context_file_path = "/home/user2/ep_context_model_directory/my_ctx_model.onnx"
 * original_model_path = "model.onnx"
 * => return "/home/user2/ep_context_model_directory/my_ctx_model.onnx"
 *
 */
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

/*
 * Dump "EP context" model
 *
 */
void DumpCtxModel(ONNX_NAMESPACE::ModelProto* model_proto,
                  const std::string& ctx_model_path) {
  std::fstream dump(ctx_model_path, std::ios::out | std::ios::trunc | std::ios::binary);
  model_proto->SerializeToOstream(dump);
  LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Dumped " + ctx_model_path;
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

Status TensorRTCacheModelHandler::GetEpContextFromGraph(const GraphViewer& graph_viewer) {
  if (!ValidateEPCtxNode(graph_viewer)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "It's not a valid EP Context node");
  }
  auto node = graph_viewer.GetNode(0);
  auto& attrs = node->GetAttributes();

  const int64_t embed_mode = attrs.at(EMBED_MODE).i();
  // Only make path checks if model not provided as byte buffer
  bool make_secure_path_checks = !GetModelPath(graph_viewer).empty();

  if (embed_mode) {
    // Get engine from byte stream.
    const std::string& context_binary = attrs.at(EP_CACHE_CONTEXT).s();
    *(trt_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(trt_runtime_->deserializeCudaEngine(const_cast<char*>(context_binary.c_str()),
                                                                                                static_cast<size_t>(context_binary.length())));
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Read engine as binary data from \"ep_cache_context\" attribute of ep context node and deserialized it";
    if (!(*trt_engine_)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP could not deserialize engine from binary data");
    }

    if (weight_stripped_engine_refit_) {
      const std::string onnx_model_filename = attrs.at(ONNX_MODEL_FILENAME).s();
      std::string placeholder;
      auto status = TensorrtExecutionProvider::RefitEngine(onnx_model_filename,
                                                           onnx_model_folder_path_,
                                                           placeholder,
                                                           make_secure_path_checks,
                                                           onnx_model_bytestream_,
                                                           onnx_model_bytestream_size_,
                                                           (*trt_engine_).get(),
                                                           false /* serialize refitted engine to disk */,
                                                           detailed_build_log_);
      if (status != Status::OK()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
      }
    }
  } else {
    // Get engine from cache file.
    std::string cache_path = attrs.at(EP_CACHE_CONTEXT).s();

    // For security purpose, in the case of running context model, TRT EP won't allow
    // engine cache path to be the relative path like "../file_path" or the absolute path.
    // It only allows the engine cache to be in the same directory or sub directory of the context model.
    if (IsAbsolutePath(cache_path)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "For security purpose, the ep_cache_context attribute should be set with a relative path, but it is an absolute path:  " + cache_path);
    }
    if (IsRelativePathToParentPath(cache_path)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "The file path in ep_cache_context attribute has '..'. For security purpose, it's not allowed to point outside the directory.");
    }

    // The engine cache and context model (current model) should be in the same directory
    std::filesystem::path ctx_model_dir(GetPathOrParentPathOfCtxModel(ep_context_model_path_));
    auto engine_cache_path = ctx_model_dir.append(cache_path);
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] GetEpContextFromGraph engine_cache_path: " + engine_cache_path.string();

    // If it's a weight-stripped engine cache, it needs to be refitted even though the refit flag is not enabled
    if (!weight_stripped_engine_refit_) {
      weight_stripped_engine_refit_ = IsWeightStrippedEngineCache(engine_cache_path);
    }

    // If the serialized refitted engine is present, use it directly without refitting the engine again
    if (weight_stripped_engine_refit_) {
      const std::filesystem::path refitted_engine_cache_path = GetWeightRefittedEnginePath(engine_cache_path.string());
      if (std::filesystem::exists(refitted_engine_cache_path)) {
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] " + refitted_engine_cache_path.string() + " exists.";
        engine_cache_path = refitted_engine_cache_path.string();
        weight_stripped_engine_refit_ = false;
      }
    }

    if (!std::filesystem::exists(engine_cache_path)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP can't find engine cache: " + engine_cache_path.string() +
                                 ". Please make sure engine cache is in the same directory or sub-directory of context model.");
    }

    std::ifstream engine_file(engine_cache_path.string(), std::ios::binary | std::ios::in);
    engine_file.seekg(0, std::ios::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engine_buf{new char[engine_size]};
    engine_file.read((char*)engine_buf.get(), engine_size);
    *(trt_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(trt_runtime_->deserializeCudaEngine(engine_buf.get(), engine_size));
    if (!(*trt_engine_)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP could not deserialize engine from cache: " + engine_cache_path.string());
    }
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + engine_cache_path.string();

    if (weight_stripped_engine_refit_) {
      const std::string onnx_model_filename = attrs.at(ONNX_MODEL_FILENAME).s();
      std::string weight_stripped_engine_cache = engine_cache_path.string();
      auto status = TensorrtExecutionProvider::RefitEngine(onnx_model_filename,
                                                           onnx_model_folder_path_,
                                                           weight_stripped_engine_cache,
                                                           make_secure_path_checks,
                                                           onnx_model_bytestream_,
                                                           onnx_model_bytestream_size_,
                                                           (*trt_engine_).get(),
                                                           true /* serialize refitted engine to disk */,
                                                           detailed_build_log_);
      if (status != Status::OK()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
      }
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

  // Show the warning if compute capability is not matched
  if (attrs.count(COMPUTE_CAPABILITY) > 0) {
    std::string model_compute_capability = attrs.at(COMPUTE_CAPABILITY).s();
    // Verify if engine was compiled with ampere+ hardware compatibility enabled
    if (model_compute_capability == "80+") {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Engine is compatible to all Ampere+ GPU (except Jetson)";
      if (std::stoi(compute_capability_) < 80) {
        LOGS_DEFAULT(WARNING) << "[TensorRT EP] However, this GPU doesn't match. The compute capability of the GPU: " << compute_capability_;
      }
    } else if (model_compute_capability != compute_capability_) {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Engine was compiled for a different compatibility level and might not work or perform suboptimal";
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] The compute capability of the engine: " << model_compute_capability;
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] The compute capability of the GPU: " << compute_capability_;
    }
  }

  // "embed_mode" attr and "ep_cache_context" attr should be present
  assert(attrs.count(EMBED_MODE) > 0);
  assert(attrs.count(EP_CACHE_CONTEXT) > 0);

  const int64_t embed_mode = attrs.at(EMBED_MODE).i();
  if (embed_mode == 1) {
    // engine binary data
    LOGS_DEFAULT(WARNING) << EPCONTEXT_WARNING;
  }

  return true;
}
}  // namespace onnxruntime
