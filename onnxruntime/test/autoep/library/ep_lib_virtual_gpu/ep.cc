// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include <gsl/gsl>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ep_factory.h"
#include "../plugin_ep_utils.h"

/// <summary>
/// Example implementation of ONNX Add. Does not handle many things like broadcasting.
/// </summary>
struct AddImpl {
  AddImpl(const OrtApi& ort_api, const OrtLogger& logger) : ort_api(ort_api), logger(logger) {}

  void GetInputDataAndShape(Ort::KernelContext kernel_context, size_t index,
                            /*out*/ gsl::span<const float>& data,
                            /*out*/ std::vector<int64_t>& shape) const {
    Ort::ConstValue input = kernel_context.GetInput(index);
    auto type_shape = input.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType elem_type = type_shape.GetElementType();
    if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      throw Ort::Exception("EP Expected float32 inputs", ORT_EP_FAIL);

    const float* float_data = input.GetTensorData<float>();
    size_t num_elems = type_shape.GetElementCount();
    data = gsl::span<const float>(float_data, num_elems);
    shape = type_shape.GetShape();
  }

  OrtStatus* Compute(OrtKernelContext* kernel_ctx) {
    RETURN_IF_ERROR(ort_api.Logger_LogMessage(&logger,
                                              OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                              "MulKernel::Compute", ORT_FILE, __LINE__, __FUNCTION__));
    Ort::KernelContext kernel_context(kernel_ctx);
    try {
      gsl::span<const float> input0;
      gsl::span<const float> input1;
      std::vector<int64_t> shape0;
      std::vector<int64_t> shape1;

      size_t num_inputs = kernel_context.GetInputCount();
      if (num_inputs != 2) {
        throw Ort::Exception("Expected 2 inputs for AddImpl", ORT_INVALID_ARGUMENT);
      }

      GetInputDataAndShape(kernel_context, 0, input0, shape0);
      GetInputDataAndShape(kernel_context, 1, input1, shape1);

      if (shape0 != shape1) {
        throw Ort::Exception("Expected same dimensions for both inputs", ORT_INVALID_ARGUMENT);
      }

      size_t num_outputs = kernel_context.GetOutputCount();
      if (num_outputs != 1) {
        throw Ort::Exception("Expected 1 output for AddImpl", ORT_INVALID_ARGUMENT);
      }

      auto output = kernel_context.GetOutput(0, shape0);
      float* output_data = output.GetTensorMutableData<float>();

      for (size_t i = 0; i < input0.size(); ++i) {
        output_data[i] = input0[i] + input1[i];
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

  const OrtApi& ort_api;
  const OrtLogger& logger;
};

/// <summary>
/// Example OrtNodeComputeInfo that represents the computation function for a compiled OrtGraph.
/// </summary>
struct ExampleNodeComputeInfo : OrtNodeComputeInfo {
  explicit ExampleNodeComputeInfo(EpVirtualGpu& ep);

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state);
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                             OrtKernelContext* kernel_context);
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state);

  EpVirtualGpu& ep;
};

EpVirtualGpu::EpVirtualGpu(EpFactoryVirtualGpu& factory, const OrtLogger& logger)
    : OrtEp{},  // explicitly call the struct ctor to ensure all optional values are default initialized
      factory_{factory},
      ort_api_{factory.GetOrtApi()},
      ep_api_{factory.GetEpApi()},
      name_{factory.GetEpName()},
      logger_{logger} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.

  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;

  auto status = ort_api_.Logger_LogMessage(&logger_,
                                           OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                           ("EpVirtualGpu has been created with name " + name_).c_str(),
                                           ORT_FILE, __LINE__, __FUNCTION__);
  // ignore status for now
  (void)status;
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
        // Check that Add has inputs/output of type float
        std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
        std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();

        RETURN_IF(inputs.size() != 2 || outputs.size() != 1, ep->ort_api_, "Add should have 2 inputs and 1 output");

        std::array<bool, 3> is_float = {false, false, false};
        IsFloatTensor(inputs[0], is_float[0]);
        IsFloatTensor(inputs[1], is_float[1]);
        IsFloatTensor(outputs[0], is_float[2]);
        if (!is_float[0] || !is_float[1] || !is_float[2]) {
          continue;  // Input or output is not of type float
        }

        {
          const auto input_0_shape = GetTensorShape(inputs[0]),
                     input_1_shape = GetTensorShape(inputs[1]);

          if (!input_0_shape.has_value() || !input_1_shape.has_value()) {
            continue;  // unable to get input shape
          }

          const auto is_static_shape = [](gsl::span<const int64_t> shape) -> bool {
            return std::all_of(shape.begin(), shape.end(), [](int64_t dim) { return dim >= 0; });
          };

          if (!is_static_shape(*input_0_shape) || !is_static_shape(*input_1_shape)) {
            continue;  // input shape has dynamic dimensions
          }

          if (*input_0_shape != *input_1_shape) {
            continue;  // input shapes do not match (no broadcasting support for now)
          }
        }

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
                                                  _Out_writes_(count) OrtNode** /*ep_context_nodes*/) noexcept {
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

    // Associate the name of the fused node with our AddImpl.
    auto fused_node_name = fused_node.GetName();
    ep->compiled_subgraphs_.emplace(std::move(fused_node_name),
                                    std::make_unique<AddImpl>(ep->ort_api_, ep->logger_));

    // Update the OrtNodeComputeInfo associated with the graph.
    auto node_compute_info = std::make_unique<ExampleNodeComputeInfo>(*ep);
    node_compute_infos[0] = node_compute_info.release();
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
    delete node_compute_infos[i];
  }
}

//
// Implementation of ExampleNodeComputeInfo
//
ExampleNodeComputeInfo::ExampleNodeComputeInfo(EpVirtualGpu& ep) : ep(ep) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

OrtStatus* ExampleNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                   OrtNodeComputeContext* compute_context,
                                                   void** compute_state) {
  auto* node_compute_info = static_cast<ExampleNodeComputeInfo*>(this_ptr);
  EpVirtualGpu& ep = node_compute_info->ep;

  std::string fused_node_name = ep.GetEpApi().NodeComputeContext_NodeName(compute_context);
  auto subgraph_it = ep.GetCompiledSubgraphs().find(fused_node_name);
  if (subgraph_it == ep.GetCompiledSubgraphs().end()) {
    std::string message = "Unable to get compiled subgraph for fused node with name " + fused_node_name;
    return ep.GetOrtApi().CreateStatus(ORT_EP_FAIL, message.c_str());
  }

  AddImpl& add_impl = *subgraph_it->second;
  *compute_state = &add_impl;
  return nullptr;
}

OrtStatus* ExampleNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                               OrtKernelContext* kernel_context) {
  (void)this_ptr;
  AddImpl& add_impl = *reinterpret_cast<AddImpl*>(compute_state);
  return add_impl.Compute(kernel_context);
}

void ExampleNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state) {
  (void)this_ptr;
  AddImpl& add_impl = *reinterpret_cast<AddImpl*>(compute_state);
  (void)add_impl;
  // Do nothing for this example.
}
