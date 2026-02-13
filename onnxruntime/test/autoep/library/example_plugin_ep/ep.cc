// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ep_factory.h"
#include "ep_stream_support.h"

const FloatInitializer* MulKernel::TryGetSavedInitializer(const std::string& name) const {
  auto iter = float_initializers.find(name);
  return iter != float_initializers.end() ? &iter->second : nullptr;
}

void MulKernel::GetInputDataAndShape(Ort::KernelContext kernel_context, size_t index,
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

OrtStatus* MulKernel::Compute(OrtKernelContext* kernel_ctx) {
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

    if (num_inputs == 2) {
      // Both inputs are non-constant. Get them from ORT's KernelContext.
      GetInputDataAndShape(kernel_context, 0, input0, shape0);
      GetInputDataAndShape(kernel_context, 1, input1, shape1);
    } else if (num_inputs == 1) {
      // ORT is only providing one non-constant input because this EP chose not to request constant initializer inputs.
      // Get the constant input from the initializers saved by the EP.
      // Refer to "NodeFusionOptions_DropConstantInitializers()".

      if (const FloatInitializer* const_input0 = TryGetSavedInitializer(input0_name); const_input0 != nullptr) {
        GetInputDataAndShape(kernel_context, 0, input1, shape1);
        input0 = gsl::span<const float>(const_input0->data);
        shape0 = const_input0->shape;
      } else if (const FloatInitializer* const_input1 = TryGetSavedInitializer(input1_name); const_input1 != nullptr) {
        GetInputDataAndShape(kernel_context, 0, input0, shape0);
        input1 = gsl::span<const float>(const_input1->data);
        shape1 = const_input1->shape;
      }
    } else {
      // Both inputs are constant. Should never happen unless all ORT optimizations (specifically constant-folding)
      // are disabled.
      const FloatInitializer* const_input0 = TryGetSavedInitializer(input0_name);
      const FloatInitializer* const_input1 = TryGetSavedInitializer(input1_name);
      RETURN_IF(const_input0 == nullptr || const_input1 == nullptr, ort_api,
                "Expected 2 initializer inputs to be saved by EP");

      input0 = gsl::span<const float>(const_input0->data);
      input1 = gsl::span<const float>(const_input1->data);
      shape0 = const_input0->shape;
      shape1 = const_input1->shape;
    }

    if (shape0 != shape1) {
      throw Ort::Exception("Expected same dimensions for both inputs", ORT_INVALID_ARGUMENT);
    }

    size_t num_outputs = kernel_context.GetOutputCount();
    if (num_outputs != 1) {
      throw Ort::Exception("Expected 1 output for MulKernel", ORT_INVALID_ARGUMENT);
    }

    auto output = kernel_context.GetOutput(0, shape0);
    float* output_data = output.GetTensorMutableData<float>();

    for (size_t i = 0; i < input0.size(); ++i) {
      output_data[i] = input0[i] * input1[i];
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

OrtStatus* EpContextKernel::Compute(OrtKernelContext* /*kernel_ctx*/) {
  // This example EP does not fully support EPContext inference.
  // A production EP would:
  // 1. Deserialize state from ep_cache_context attribute during Compile
  // 2. Use that state here to perform actual computation
  //
  // Session creation succeeds for metadata access and compatibility testing,
  // but inference requires deserializing ep_cache_context (not implemented).
  return ort_api.CreateStatus(
      ORT_NOT_IMPLEMENTED,
      "EPContext inference is not fully implemented in this example EP. "
      "Session creation succeeds for metadata access and compatibility testing, "
      "but inference requires deserializing ep_cache_context (not implemented). "
      "A production EP would restore compiled state from the EPContext node's attributes.");
}

/// <summary>
/// Intermediate base class with virtual destructor for proper polymorphic deletion.
/// This allows ReleaseNodeComputeInfosImpl to delete any derived type correctly
/// without manual type dispatch.
/// </summary>
struct NodeComputeInfoBase : OrtNodeComputeInfo {
  virtual ~NodeComputeInfoBase() = default;
};

/// <summary>
/// Example OrtNodeComputeInfo that represents the computation function for a compiled OrtGraph.
/// </summary>
struct ExampleNodeComputeInfo : NodeComputeInfoBase {
  explicit ExampleNodeComputeInfo(ExampleEp& ep);

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state);
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                             OrtKernelContext* kernel_context);
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state);

  ExampleEp& ep;
};

/// <summary>
/// OrtNodeComputeInfo for EPContext nodes - delegates to EpContextKernel.
/// </summary>
struct EpContextNodeComputeInfo : NodeComputeInfoBase {
  explicit EpContextNodeComputeInfo(ExampleEp& ep);

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state);
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                             OrtKernelContext* kernel_context);
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state);

  ExampleEp& ep;
};

ExampleEp::ExampleEp(ExampleEpFactory& factory, const std::string& name, const Config& config, const OrtLogger& logger)
    : OrtEp{},  // explicitly call the struct ctor to ensure all optional values are default initialized
      ApiPtrs{static_cast<const ApiPtrs&>(factory)},
      factory_{factory},
      name_{name},
      config_{config},
      logger_{logger} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.

  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  CreateAllocator = CreateAllocatorImpl;                                      // optional. can be nullptr
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;                  // optional. can be nullptr
  GetCompiledModelCompatibilityInfo = GetCompiledModelCompatibilityInfoImpl;  // compatibility info for compiled models

  IGNORE_ORTSTATUS(ort_api.Logger_LogMessage(&logger_,
                                             OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                             ("ExampleEp has been created with name " + name_).c_str(),
                                             ORT_FILE, __LINE__, __FUNCTION__));
}

ExampleEp::~ExampleEp() = default;

/*static*/
const char* ORT_API_CALL ExampleEp ::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const ExampleEp*>(this_ptr);
  return ep->name_.c_str();
}

OrtStatus* ExampleEp::SaveConstantInitializers(const OrtGraph* ort_graph) {
  Ort::ConstGraph graph{ort_graph};

  try {
    std::vector<Ort::ConstValueInfo> initializers = graph.GetInitializers();

    for (const auto& initializer : initializers) {
      const bool is_constant = initializer.IsConstantInitializer();

      if (is_constant) {
        auto name = initializer.GetName();
        Ort::ConstValue value;
        auto status = initializer.GetInitializer(value);
        if (!status.IsOK())
          return status.release();

        auto type_shape = value.GetTensorTypeAndShapeInfo();
        const size_t num_elems = type_shape.GetElementCount();
        const ONNXTensorElementDataType elem_type = type_shape.GetElementType();
        if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
          return Ort::Status("Expected float32 initializers", ORT_INVALID_ARGUMENT).release();

        std::vector<int64_t> dims = type_shape.GetShape();
        const float* data = value.GetTensorData<float>();

        FloatInitializer ep_initializer = {std::move(dims), std::vector<float>(data, data + num_elems)};
        float_initializers_.emplace(std::move(name), std::move(ep_initializer));
      }
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
OrtStatus* ORT_API_CALL ExampleEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                                     OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    ExampleEp* ep = static_cast<ExampleEp*>(this_ptr);

    Ort::ConstGraph graph{ort_graph};
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();
    if (nodes.empty()) {
      return nullptr;  // No nodes to process
    }

    // Single array for all supported node types.
    // This EP only supports compiling one node at a time (a documented limitation).
    std::vector<Ort::ConstNode> supported_nodes;

    for (const auto& node : nodes) {
      auto op_type = node.GetOperatorType();
      auto domain = node.GetDomain();

      // Check for EPContext nodes that belong to this EP (from compiled models).
      // This is needed to handle loading pre-compiled models with EPContext nodes.
      if (op_type == "EPContext" && domain == "com.microsoft") {
        // Check if this EPContext node belongs to this EP via the "source" attribute
        Ort::ConstOpAttr source_attr;
        Ort::Status status = node.GetAttributeByName("source", source_attr);
        if (status.IsOK()) {
          std::string source_value;
          status = source_attr.GetValue(source_value);
          if (status.IsOK() && source_value == ep->name_) {
            // This EPContext node was created by this EP - add to supported nodes
            supported_nodes.push_back(node);
            break;  // Only support one node at a time
          }
        }
        continue;  // Don't process further, EPContext is a special case
      }

      if (op_type == "Mul") {
        // Check that Mul has inputs/output of type float
        std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
        std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();

        RETURN_IF(inputs.size() != 2 || outputs.size() != 1, ep->ort_api, "Mul should have 2 inputs and 1 output");

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

          // Don't support broadcasting and dynamic dimensions for now.
          if (!AreShapesStaticAndEqual(*input_0_shape, *input_1_shape)) {
            continue;
          }
        }

        supported_nodes.push_back(node);
        break;  // Only support a single Mul for now.
      } else if (op_type == "Custom_Mul" && domain == "test") {
        supported_nodes.push_back(node);
        break;  // Only support one node at a time (consistent with Mul/EPContext handling).
      }
    }

    // Return early if no supported nodes
    if (supported_nodes.empty()) {
      return nullptr;
    }

    // Unified dispatch based on node type
    const auto& node = supported_nodes[0];
    auto op_type = node.GetOperatorType();

    if (op_type == "Custom_Mul") {
      // Custom_Mul has concrete kernel implementation - no fusion needed.
      // Calls EpGraphSupportInfo_AddSingleNode() to inform ORT that the custom node should NOT be fused or compiled.
      RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddSingleNode(graph_support_info, node));
    } else {
      // Both EPContext and Mul use AddNodesToFuse
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;

      // Set "drop constant initializers" to true if the compiling EP doesn't need ORT to provide constant initializers
      // as inputs to the fused/compiled node at inference time. This allows ORT to release unused initializers.
      // This example EP sets this to true and saves initializers during the call to OrtEp::Compile for use
      // during inference.
      node_fusion_options.drop_constant_initializers = true;
      RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          reinterpret_cast<const OrtNode* const*>(supported_nodes.data()),
          supported_nodes.size(),
          &node_fusion_options));
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
OrtStatus* ORT_API_CALL ExampleEp::CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** ort_graphs,
                                               _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                               _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                               _Out_writes_(count) OrtNode** ep_context_nodes) noexcept {
  try {
    if (count != 1) {
      Ort::Status status("Expected to compile a single graph", ORT_EP_FAIL);
      return status.release();
    }

    ExampleEp* ep = static_cast<ExampleEp*>(this_ptr);

    Ort::ConstGraph graph{ort_graphs[0]};

    // In GetCapability(), this EP specified that it doesn't need ORT to provide constant initializers during inference.
    // So, this EP saves constant initializers so that they're available during inference, but an actual EP
    // implementation could transfer the weights to device memory.
    ep->SaveConstantInitializers(graph);

    std::vector<Ort::ConstNode> nodes = graph.GetNodes();
    if (nodes.size() != 1) {
      Ort::Status status("Expected to compile a single node", ORT_EP_FAIL);
      return status.release();
    }

    auto node_op_type = nodes[0].GetOperatorType();
    auto node_domain = nodes[0].GetDomain();

    // Check if this is an EPContext node (from loading a pre-compiled model)
    bool is_ep_context_node = (node_op_type == "EPContext" && node_domain == "com.microsoft");

    // Validate configuration: cannot enable EPContext generation when loading a compiled model.
    // This is a configuration error - you cannot re-compile an already compiled model.
    if (ep->config_.enable_ep_context && is_ep_context_node) {
      Ort::Status status(
          "Invalid configuration: 'enable_ep_context' is true but model already contains "
          "EPContext nodes. Cannot re-compile an already compiled model. Either:\n"
          "  1. Use the original (uncompiled) model as input, or\n"
          "  2. Disable ep_context generation when loading a compiled model.",
          ORT_INVALID_ARGUMENT);
      return status.release();
    }

    if (node_op_type != "Mul" && !is_ep_context_node) {
      Ort::Status status("Expected to compile a Mul node or EPContext node", ORT_EP_FAIL);
      return status.release();
    }

    Ort::ConstNode fused_node{fused_nodes[0]};
    auto ep_name = fused_node.GetEpName();
    if (ep_name != ep->name_) {
      Ort::Status status("The fused node is expected to assigned to this EP to run on", ORT_EP_FAIL);
      return status.release();
    }

    auto fused_node_name = fused_node.GetName();

    if (is_ep_context_node) {
      // Create EpContextKernel for EPContext nodes - clearly separates from MulKernel
      ep->ep_context_kernels_.emplace(fused_node_name,
                                      std::make_unique<EpContextKernel>(ep->ort_api, ep->logger_));

      // Use EpContextNodeComputeInfo for EPContext nodes
      auto node_compute_info = std::make_unique<EpContextNodeComputeInfo>(*ep);
      node_compute_infos[0] = node_compute_info.release();
    } else {
      // For Mul nodes during initial compilation, we need exactly 2 inputs
      std::vector<Ort::ConstValueInfo> node_inputs = nodes[0].GetInputs();
      if (node_inputs.size() != 2) {
        std::string err_msg = "Mul node should have 2 inputs, got " + std::to_string(node_inputs.size());
        Ort::Status status(err_msg.c_str(), ORT_EP_FAIL);
        return status.release();
      }

      // Create MulKernel for Mul nodes
      ep->mul_kernels_.emplace(fused_node_name,
                               std::make_unique<MulKernel>(ep->ort_api, ep->logger_,
                                                           ep->float_initializers_,
                                                           node_inputs[0].GetName(),
                                                           node_inputs[1].GetName()));

      // Use ExampleNodeComputeInfo for Mul nodes
      auto node_compute_info = std::make_unique<ExampleNodeComputeInfo>(*ep);
      node_compute_infos[0] = node_compute_info.release();

      // Create EpContext nodes for the fused nodes we compiled (only for Mul, not EPContext).
      if (ep->config_.enable_ep_context) {
        assert(ep_context_nodes != nullptr);
        RETURN_IF_ERROR(ep->CreateEpContextNodes(gsl::span<const OrtNode*>(fused_nodes, count),
                                                 gsl::span<OrtNode*>(ep_context_nodes, count)));
      }
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
void ORT_API_CALL ExampleEp::ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                         OrtNodeComputeInfo** node_compute_infos,
                                                         size_t num_node_compute_infos) noexcept {
  (void)this_ptr;
  for (size_t i = 0; i < num_node_compute_infos; i++) {
    // All node compute info types derive from NodeComputeInfoBase which has a virtual destructor.
    // This ensures correct polymorphic deletion without manual type dispatch.
    delete static_cast<NodeComputeInfoBase*>(node_compute_infos[i]);
  }
}

// Creates EPContext nodes from the given fused nodes.
// This is an example implementation that can be used to generate an EPContext model. However, this example EP
// cannot currently run the EPContext model.
OrtStatus* ExampleEp::CreateEpContextNodes(gsl::span<const OrtNode*> fused_nodes,
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
      RETURN_IF_ERROR(model_editor_api.CreateNode("EPContext", "com.microsoft", fused_node_name.c_str(),
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
OrtStatus* ORT_API_CALL ExampleEp::CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                       _In_ const OrtMemoryInfo* memory_info,
                                                       _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept {
  // A per-session allocator could be created here.
  // Logging of any issues should use ep->logger_ which is the session logger.

  ExampleEp* ep = static_cast<ExampleEp*>(this_ptr);

  // for simplicity in this example we use the factory implementation.
  return ep->factory_.CreateAllocator(&ep->factory_, memory_info, nullptr, allocator);
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEp::CreateSyncStreamForDeviceImpl(_In_ OrtEp* this_ptr,
                                                                 _In_ const OrtMemoryDevice* memory_device,
                                                                 _Outptr_ OrtSyncStreamImpl** stream) noexcept {
  // A per-session OrtSyncStreamImpl can be created here if the session options affect the implementation.
  // Logging of any issues should use logger_ which is the session logger.

  ExampleEp* ep = static_cast<ExampleEp*>(this_ptr);

  // we only create streams for the default device memory.
  if (auto mem_type = ep->factory_.ep_api.MemoryDevice_GetMemoryType(memory_device);
      mem_type != OrtDeviceMemoryType_DEFAULT) {
    std::string error = "Invalid OrtMemoryDevice. Expected OrtDeviceMemoryType_DEFAULT(0). Got ";
    error += std::to_string(mem_type);
    return ep->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, error.c_str());
  }

  auto sync_stream = std::make_unique<StreamImpl>(ep->factory_, ep, nullptr);
  *stream = sync_stream.release();

  return nullptr;
}

//
// Implementation of ExampleNodeComputeInfo
//
ExampleNodeComputeInfo::ExampleNodeComputeInfo(ExampleEp& ep) : ep(ep) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

OrtStatus* ExampleNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                   OrtNodeComputeContext* compute_context,
                                                   void** compute_state) {
  auto* node_compute_info = static_cast<ExampleNodeComputeInfo*>(this_ptr);
  ExampleEp& ep = node_compute_info->ep;

  std::string fused_node_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
  auto kernel_it = ep.MulKernels().find(fused_node_name);
  if (kernel_it == ep.MulKernels().end()) {
    std::string message = "Unable to get MulKernel for fused node with name " + fused_node_name;
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
  }

  MulKernel& kernel = *kernel_it->second;
  *compute_state = &kernel;
  return nullptr;
}

OrtStatus* ExampleNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                               OrtKernelContext* kernel_context) {
  (void)this_ptr;
  MulKernel& kernel = *reinterpret_cast<MulKernel*>(compute_state);
  return kernel.Compute(kernel_context);
}

void ExampleNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state) {
  (void)this_ptr;
  MulKernel& kernel = *reinterpret_cast<MulKernel*>(compute_state);
  (void)kernel;
  // Do nothing for this example.
}

//
// Implementation of EpContextNodeComputeInfo
//
EpContextNodeComputeInfo::EpContextNodeComputeInfo(ExampleEp& ep) : ep(ep) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

OrtStatus* EpContextNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                     OrtNodeComputeContext* compute_context,
                                                     void** compute_state) {
  auto* node_compute_info = static_cast<EpContextNodeComputeInfo*>(this_ptr);
  ExampleEp& ep = node_compute_info->ep;

  std::string fused_node_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
  auto kernel_it = ep.EpContextKernels().find(fused_node_name);
  if (kernel_it == ep.EpContextKernels().end()) {
    std::string message = "Unable to get EpContextKernel for fused node with name " + fused_node_name;
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
  }

  EpContextKernel& kernel = *kernel_it->second;
  *compute_state = &kernel;
  return nullptr;
}

OrtStatus* EpContextNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                                 OrtKernelContext* kernel_context) {
  (void)this_ptr;
  EpContextKernel& kernel = *reinterpret_cast<EpContextKernel*>(compute_state);
  return kernel.Compute(kernel_context);
}

void EpContextNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state) {
  (void)this_ptr;
  (void)compute_state;
  // Do nothing for this example.
}

//
// Implementation of GetCompiledModelCompatibilityInfo
//
/*static*/
const char* ORT_API_CALL ExampleEp::GetCompiledModelCompatibilityInfoImpl(OrtEp* this_ptr,
                                                                          const OrtGraph* graph) noexcept {
  // Suppress unused parameter warning. The ORT_UNUSED_PARAMETER macro is in internal headers
  // (core/common/common.h) which are not available to plugin EPs using only public APIs.
  // A real EP would inspect the graph for model-specific compatibility info.
  (void)graph;
  auto* ep = static_cast<ExampleEp*>(this_ptr);

  // Generate a compatibility string that includes:
  // - EP name
  // - EP version (from factory)
  // - ORT API version
  //
  // In a real EP, this might include driver versions, hardware IDs, etc.
  // The string format is EP-defined and should be parseable by ValidateCompiledModelCompatibilityInfo.
  ep->compatibility_info_ = ep->name_ + ";version=" + ep->factory_.GetEpVersionString() + ";ort_api_version=" +
                            std::to_string(ORT_API_VERSION);

  IGNORE_ORTSTATUS(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                 OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                 ("GetCompiledModelCompatibilityInfo returning: " + ep->compatibility_info_).c_str(),
                                                 ORT_FILE, __LINE__, __FUNCTION__));

  return ep->compatibility_info_.c_str();
}
