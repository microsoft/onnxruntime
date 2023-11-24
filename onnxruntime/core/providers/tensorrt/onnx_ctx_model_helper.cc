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
 * Get compute capability
 *
 */
std::string GetComputeCapacityString(const cudaDeviceProp& prop) {
  const std::string compute_capability = std::to_string(prop.major) + "." + std::to_string(prop.minor);
  return compute_capability;
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
                                               bool compute_capability_enable,
                                               int device_id,
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
  auto attr_2 = ONNX_NAMESPACE::AttributeProto::Create();  // hardware_arch
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
  } else {
    attr_1->set_s(engine_cache_path);
  }
  auto node_attributes = ONNX_NAMESPACE::NodeAttributes::Create();
  int num_attributes = compute_capability_enable ? 3 : 2;
  node_attributes->reserve(num_attributes);
  node_attributes->emplace(EMBED_MODE, *attr_0);
  node_attributes->emplace(EP_CACHE_CONTEXT, *attr_1);

  if (compute_capability_enable) {
    cudaDeviceProp prop;
    CUDA_CALL_THROW(cudaGetDeviceProperties(&prop, device_id));
    attr_2->set_name(COMPUTE_CAPABILITY);
    attr_2->set_type(onnx::AttributeProto_AttributeType_STRING);
    attr_2->set_s(GetComputeCapacityString(prop));
    node_attributes->emplace(COMPUTE_CAPABILITY, *attr_2);
  }

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
 * Dump "EP context node" model
 *
 */
void DumpCtxNodeModel(ONNX_NAMESPACE::ModelProto* model_proto,
                      const std::string engine_cache_path) {
  std::string string_buf;
  model_proto->SerializeToString(string_buf);

  // Dump EP context node model to disk
  std::fstream dump(engine_cache_path + "_wrapper.onnx", std::ios::out | std::ios::trunc | std::ios::binary);
  model_proto->SerializeToOstream(dump);
  LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialized " + engine_cache_path + "_wrapper.onnx";
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

  // Check hardware_arch(compute_capability) if it's present as an attribute
  if (attrs.count(COMPUTE_CAPABILITY) > 0) {
    std::string model_compute_capability = attrs.at(COMPUTE_CAPABILITY).s();
    cudaDeviceProp prop;
    CUDA_CALL_THROW(cudaGetDeviceProperties(&prop, device_id_));
    if (model_compute_capability != GetComputeCapacityString(prop)) {
      LOGS_DEFAULT(ERROR) << "The compute capability of the engine cache doesn't match with the GPU's compute capability";
      LOGS_DEFAULT(ERROR) << "The compute capability of the engine cache: " << model_compute_capability;
      LOGS_DEFAULT(ERROR) << "The compute capability of the GPU: " << GetComputeCapacityString(prop);
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
