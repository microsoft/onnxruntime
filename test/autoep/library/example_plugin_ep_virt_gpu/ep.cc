// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include <gsl/span>
#include <array>
#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ep_factory.h"
#include "../plugin_ep_utils.h"

/// <summary>
/// Empty (not implemented) computation functor for a compiled Add.
/// This EP only supports a virtual GPU device that cannot run inference, but can create a compiled model.
/// </summary>
struct VirtualCompiledAdd {
  VirtualCompiledAdd(const OrtApi& ort_api, const OrtLogger& logger) : ort_api(ort_api), logger(logger) {}

  OrtStatus* Compute(OrtKernelContext* /*kernel_ctx*/) {
    RETURN_IF_ERROR(ort_api.Logger_LogMessage(&logger,
                                              OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                              "VirtualCompiledAdd::Compute", ORT_FILE, __LINE__, __FUNCTION__));
    return ort_api.CreateStatus(ORT_NOT_IMPLEMENTED, "EP only supports a virtual GPU that cannot run ops.");
  }

  const OrtApi& ort_api;
  const OrtLogger& logger;
};

/// <summary>
/// Example OrtNodeComputeInfo that represents the computation function for a compiled OrtGraph.
/// </summary>
struct AddNodeComputeInfo : OrtNodeComputeInfo {
  explicit AddNodeComputeInfo(EpVirtualGpu& ep);

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state);
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                             OrtKernelContext* kernel_context);
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state);

  EpVirtualGpu& ep;
};

EpVirtualGpu::EpVirtualGpu(EpFactoryVirtualGpu& factory, const EpVirtualGpu::Config& config,
                           const OrtLogger& logger)
    : OrtEp{},  // explicitly call the struct ctor to ensure all optional values are default initialized
      config_{config},
      ort_api_{factory.GetOrtApi()},
      ep_api_{factory.GetEpApi()},
      model_editor_api_{factory.GetModelEditorApi()},
      name_{factory.GetEpName()},
      logger_{logger} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.

  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;

  IGNORE_ORTSTATUS(ort_api_.Logger_LogMessage(&logger_,
                                              OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                              ("EpVirtualGpu has been created with name " + name_).c_str(),
                                              ORT_FILE, __LINE__, __FUNCTION__));
}

EpVirtualGpu::~EpVirtualGpu() = default;

/*static*/
const char* ORT_API_CALL EpVirtualGpu ::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const EpVirtualGpu*>(this_ptr);
  return ep->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL EpVirtualGpu::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                                        OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    EpVirtualGpu* ep = static_cast<EpVirtualGpu*>(this_ptr);

    Ort::ConstGraph graph{ort_graph};
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();

    if (nodes.empty()) {
      return nullptr;  // No nodes to process
    }

    std::vector<Ort::ConstNode> supported_nodes;

    for (const auto& node : nodes) {
      auto op_type = node.GetOperatorType();

      if (op_type == "Add") {
        supported_nodes.push_back(node);  // Only support a single Add for now.
        break;
      }
    }

    if (supported_nodes.empty()) {
      return nullptr;
    }

    // Create (optional) fusion options for the supported nodes to fuse.
    OrtNodeFusionOptions node_fusion_options = {};
    node_fusion_options.ort_version_supported = ORT_API_VERSION;
    node_fusion_options.drop_constant_initializers = false;

    RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                                                  reinterpret_cast<const OrtNode* const*>(supported_nodes.data()),
                                                                  supported_nodes.size(),
                                                                  &node_fusion_options));
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL EpVirtualGpu::CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** ort_graphs,
                                                  _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                                  _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                                  _Out_writes_(count) OrtNode** ep_context_nodes) noexcept {
  try {
    if (count != 1) {
      Ort::Status status("Expected to compile a single graph", ORT_EP_FAIL);
      return status.release();
    }

    EpVirtualGpu* ep = static_cast<EpVirtualGpu*>(this_ptr);

    Ort::ConstGraph graph{ort_graphs[0]};

    std::vector<Ort::ConstNode> nodes = graph.GetNodes();
    if (nodes.size() != 1) {
      Ort::Status status("Expected to compile a single Add node", ORT_EP_FAIL);
      return status.release();
    }

    auto node_op_type = nodes[0].GetOperatorType();
    if (node_op_type != "Add") {
      Ort::Status status("Expected to compile a single Add node", ORT_EP_FAIL);
      return status.release();
    }

    // Now we know we're compiling a single Add node. Create a computation kernel.
    Ort::ConstNode fused_node{fused_nodes[0]};
    auto ep_name = fused_node.GetEpName();
    if (ep_name != ep->name_) {
      Ort::Status status("The fused node is expected to assigned to this EP to run on", ORT_EP_FAIL);
      return status.release();
    }

    // Associate the name of the fused node with our VirtualCompiledAdd.
    auto fused_node_name = fused_node.GetName();
    ep->compiled_subgraphs_.emplace(std::move(fused_node_name),
                                    std::make_unique<VirtualCompiledAdd>(ep->ort_api_, ep->logger_));

    // Update the OrtNodeComputeInfo associated with the graph.
    auto node_compute_info = std::make_unique<AddNodeComputeInfo>(*ep);
    node_compute_infos[0] = node_compute_info.release();

    // Create EpContext nodes for the fused nodes we compiled (if enabled by user via session options).
    if (ep->config_.enable_ep_context) {
      assert(ep_context_nodes != nullptr);
      RETURN_IF_ERROR(ep->CreateEpContextNodes(gsl::span<const OrtNode*>(fused_nodes, count),
                                               gsl::span<OrtNode*>(ep_context_nodes, count)));
    }
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

// Creates EPContext nodes from the given fused nodes.
// This is an example implementation that can be used to generate an EPContext model. However, this example EP
// cannot currently run the EPContext model.
OrtStatus* EpVirtualGpu::CreateEpContextNodes(gsl::span<const OrtNode*> fused_nodes,
                                              /*out*/ gsl::span<OrtNode*> ep_context_nodes) {
  try {
    assert(fused_nodes.size() == ep_context_nodes.size());

    // Helper to collect input or output names from an array of OrtValueInfo instances.
    auto collect_input_output_names = [&](gsl::span<Ort::ConstValueInfo const> value_infos,
                                          std::vector<std::string>& result) {
      std::vector<std::string> value_names;
      value_names.reserve(value_infos.size());

      for (const auto& vi : value_infos) {
        value_names.push_back(vi.GetName());
      }

      result = std::move(value_names);
    };

    // Create an "EPContext" node for every fused node.
    for (size_t i = 0; i < fused_nodes.size(); ++i) {
      Ort::ConstNode fused_node{fused_nodes[i]};
      auto fused_node_name = fused_node.GetName();

      std::vector<Ort::ConstValueInfo> fused_node_inputs = fused_node.GetInputs();
      std::vector<Ort::ConstValueInfo> fused_node_outputs = fused_node.GetOutputs();

      std::vector<std::string> input_names;
      std::vector<std::string> output_names;

      collect_input_output_names(fused_node_inputs, /*out*/ input_names);
      collect_input_output_names(fused_node_outputs, /*out*/ output_names);

      int64_t is_main_context = (i == 0);
      int64_t embed_mode = 1;

      // Create node attributes. The CreateNode() function copies the attributes.
      std::array<Ort::OpAttr, 6> attributes = {};
      std::string ep_ctx = "binary_data";
      attributes[0] = Ort::OpAttr("ep_cache_context", ep_ctx.data(), static_cast<int>(ep_ctx.size()),
                                  ORT_OP_ATTR_STRING);

      attributes[1] = Ort::OpAttr("main_context", &is_main_context, 1, ORT_OP_ATTR_INT);
      attributes[2] = Ort::OpAttr("embed_mode", &embed_mode, 1, ORT_OP_ATTR_INT);
      attributes[3] = Ort::OpAttr("ep_sdk_version", "1", 1, ORT_OP_ATTR_STRING);
      attributes[4] = Ort::OpAttr("partition_name", fused_node_name.data(), static_cast<int>(fused_node_name.size()),
                                  ORT_OP_ATTR_STRING);

      attributes[5] = Ort::OpAttr("source", this->name_.data(), static_cast<int>(this->name_.size()),
                                  ORT_OP_ATTR_STRING);

      std::vector<const char*> c_input_names;
      std::transform(input_names.begin(), input_names.end(), std::back_inserter(c_input_names),
                     [](const std::string& s) { return s.c_str(); });
      std::vector<const char*> c_output_names;
      std::transform(output_names.begin(), output_names.end(), std::back_inserter(c_output_names),
                     [](const std::string& s) { return s.c_str(); });

      OrtOpAttr** op_attrs = reinterpret_cast<OrtOpAttr**>(attributes.data());
      RETURN_IF_ERROR(model_editor_api_.CreateNode("EPContext", "com.microsoft", fused_node_name.c_str(),
                                                   c_input_names.data(), c_input_names.size(),
                                                   c_output_names.data(), c_output_names.size(),
                                                   op_attrs, attributes.size(),
                                                   &ep_context_nodes[i]));
    }
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL EpVirtualGpu::ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                            OrtNodeComputeInfo** node_compute_infos,
                                                            size_t num_node_compute_infos) noexcept {
  (void)this_ptr;
  for (size_t i = 0; i < num_node_compute_infos; i++) {
    delete static_cast<AddNodeComputeInfo*>(node_compute_infos[i]);
  }
}

//
// Implementation of AddNodeComputeInfo
//
AddNodeComputeInfo::AddNodeComputeInfo(EpVirtualGpu& ep) : ep(ep) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

OrtStatus* AddNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                               OrtNodeComputeContext* compute_context,
                                               void** compute_state) {
  auto* node_compute_info = static_cast<AddNodeComputeInfo*>(this_ptr);
  EpVirtualGpu& ep = node_compute_info->ep;

  std::string fused_node_name = ep.GetEpApi().NodeComputeContext_NodeName(compute_context);
  auto subgraph_it = ep.GetCompiledSubgraphs().find(fused_node_name);
  if (subgraph_it == ep.GetCompiledSubgraphs().end()) {
    std::string message = "Unable to get compiled subgraph for fused node with name " + fused_node_name;
    return ep.GetOrtApi().CreateStatus(ORT_EP_FAIL, message.c_str());
  }

  VirtualCompiledAdd& add_impl = *subgraph_it->second;
  *compute_state = &add_impl;
  return nullptr;
}

OrtStatus* AddNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                           OrtKernelContext* kernel_context) {
  (void)this_ptr;
  VirtualCompiledAdd& add_impl = *reinterpret_cast<VirtualCompiledAdd*>(compute_state);
  return add_impl.Compute(kernel_context);
}

void AddNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state) {
  (void)this_ptr;
  VirtualCompiledAdd& add_impl = *reinterpret_cast<VirtualCompiledAdd*>(compute_state);
  (void)add_impl;
  // Do nothing for this example.
}
