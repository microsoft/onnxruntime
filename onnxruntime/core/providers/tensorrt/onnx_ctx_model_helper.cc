// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <filesystem>

#include "onnx_ctx_model_helper.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/framework/execution_provider.h"

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
ONNX_NAMESPACE::ModelProto* CreateCtxNodeModel(const GraphViewer& graph_viewer,
                                               const std::string engine_cache_path,
                                               char* engine_data,
                                               size_t size,
                                               const int64_t embed_mode,
                                               std::string compute_capability,
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

  auto node_attributes = ONNX_NAMESPACE::NodeAttributes::Create();
  int num_attributes = 3;
  node_attributes->reserve(num_attributes);
  node_attributes->emplace(EMBED_MODE, *attr_0);
  node_attributes->emplace(EP_CACHE_CONTEXT, *attr_1);
  node_attributes->emplace(COMPUTE_CAPABILITY, *attr_2);

  // Create EP context node
  graph_build.AddNode(EPCONTEXT_OP, EPCONTEXT_OP, "", inputs, outputs, node_attributes.get(), EPCONTEXT_OP_DOMAIN);
  ORT_ENFORCE(graph_build.Resolve().IsOK());

  // Serialize modelproto to string
  auto new_graph_viewer = graph_build.CreateGraphViewer();
  auto model = new_graph_viewer->CreateModel(*logger);
  auto model_proto = model->ToProto();
  new_graph_viewer->ToProto(*model_proto->mutable_graph(), true, true);
  model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  return model_proto.release();
}

/*
 * Get "EP context node" model path
 *
 *
 * If ep_context_file_path is provided:
 *     - If ep_context_file_path is a file:
 *         - If it's a file name without any path associated with it, return "engine_cache_path/ep_context_file_path".
           - If it's a file name with path associated with it, return "ep_context_file_path".
 *     - If ep_context_file_path is a directory, return "ep_context_file_path/original_model_name_ctx.onnx".
 * If ep_context_file_path is not provided:
 *     - Return "engine_cache_path/original_model_name_ctx.onnx".
 *
 *
 * Example 1:
 * ep_context_file_path = "/home/user/ep_context_model_foler"
 * engine_cache_path = "trt_engine.engine"
 * original_model_path = "model.onnx"
 * => return "/home/user/ep_context_model_folder/model_ctx.onnx"
 *
 * Example 2:
 * ep_context_file_path = "my_ctx_model.onnx"
 * engine_cache_path = "/home/user/cache_folder/trt_engine.engine"
 * original_model_path = "model.onnx"
 * => return "/home/user/cache_folder/my_ctx_model.onnx"
 *
 * Example 3:
 * ep_context_file_path = "/home/user2/ep_context_model_foler/my_ctx_model.onnx"
 * engine_cache_path = "trt_engine.engine"
 * original_model_path = "model.onnx"
 * => return "/home/user2/ep_context_model_foler/my_ctx_model.onnx"
 *
 * Example 4:
 * ep_context_file_path = ""
 * engine_cache_path = "/home/user3/cache_folder/trt_engine.engine"
 * original_model_path = "model.onnx"
 * => return "/home/user3/cache_folder/model_ctx.onnx"
 *
 */
std::string GetCtxNodeModelPath(const std::string& ep_context_file_path,
                                const std::string& engine_cache_path,
                                const std::string& original_model_path) {
  std::string ctx_model_path;

  if (!ep_context_file_path.empty() && !std::filesystem::is_directory(ep_context_file_path)) {
    std::filesystem::path ctx_model_file_path = ep_context_file_path;
    if (ctx_model_file_path.filename().string() == ep_context_file_path) {
      std::filesystem::path cache_path = engine_cache_path;
      if (cache_path.has_parent_path()) {
        ctx_model_path = cache_path.parent_path().append(ep_context_file_path).string();
      } else {
        ctx_model_path = ep_context_file_path;
      }
    } else {
      ctx_model_path = ep_context_file_path;
    }
  } else {
    std::filesystem::path model_path = original_model_path;
    std::filesystem::path model_name_stem = model_path.stem();  // model_name.onnx -> model_name
    std::string ctx_model_name = model_name_stem.string() + "_ctx.onnx";

    if (std::filesystem::is_directory(ep_context_file_path)) {
      std::filesystem::path model_directory = ep_context_file_path;
      ctx_model_path = model_directory.append(ctx_model_name).string();
    } else {
      std::filesystem::path cache_path = engine_cache_path;
      if (cache_path.has_parent_path()) {
        ctx_model_path = cache_path.parent_path().append(ctx_model_name).string();
      } else {
        ctx_model_path = ctx_model_name;
      }
    }
  }
  return ctx_model_path;
}

/*
 * Dump "EP context node" model
 *
 */
void DumpCtxNodeModel(ONNX_NAMESPACE::ModelProto* model_proto,
                      const std::string& ctx_model_path) {
  std::fstream dump(ctx_model_path, std::ios::out | std::ios::trunc | std::ios::binary);
  model_proto->SerializeToOstream(dump);
  LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Dumped " + ctx_model_path;
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

  // Show the warning if compute capability is not matched
  if (attrs.count(COMPUTE_CAPABILITY) > 0) {
    std::string model_compute_capability = attrs.at(COMPUTE_CAPABILITY).s();
    if (model_compute_capability != compute_capability_) {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Engine was compiled for a different compatibility level and might not work or perform suboptimal";
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] The compute capability of the engine: " << model_compute_capability;
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] The compute capability of the GPU: " << compute_capability_;
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
    } else if (embed_mode == 1) {
      LOGS_DEFAULT(WARNING) << EPCONTEXT_WARNING;
    }
  }
  return true;
}
}  // namespace onnxruntime
