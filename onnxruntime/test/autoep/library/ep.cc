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

/// <summary>
/// Example implementation of ONNX Mul. Does not handle many things like broadcasting.
/// </summary>
struct MulKernel {
  MulKernel(const OrtApi& ort_api, const OrtLogger& logger,
            const std::unordered_map<std::string, FloatInitializer>& float_initializers,
            std::string input0_name, std::string input1_name)
      : ort_api(ort_api),
        logger(logger),
        float_initializers(float_initializers),
        input0_name(input0_name),
        input1_name(input1_name) {}

  const FloatInitializer* TryGetSavedInitializer(const std::string& name) const {
    auto iter = float_initializers.find(name);
    return iter != float_initializers.end() ? &iter->second : nullptr;
  }

  OrtStatus* GetInputDataAndShape(OrtKernelContext* kernel_context, size_t index,
                                  /*out*/ gsl::span<const float>& data,
                                  /*out*/ std::vector<int64_t>& shape) const {
    const OrtValue* input = nullptr;
    RETURN_IF_ERROR(ort_api.KernelContext_GetInput(kernel_context, index, &input));

    OrtTensorTypeAndShapeInfo* type_shape = nullptr;
    DeferOrtRelease<OrtTensorTypeAndShapeInfo> release_type(&type_shape, ort_api.ReleaseTensorTypeAndShapeInfo);

    RETURN_IF_ERROR(ort_api.GetTensorTypeAndShape(input, &type_shape));

    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    RETURN_IF_ERROR(ort_api.GetTensorElementType(type_shape, &elem_type));
    RETURN_IF(elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ort_api, "Expected float32 inputs");

    size_t num_elems = 0;
    RETURN_IF_ERROR(ort_api.GetTensorShapeElementCount(type_shape, &num_elems));

    size_t num_dims = 0;
    RETURN_IF_ERROR(ort_api.GetDimensionsCount(type_shape, &num_dims));

    shape.resize(num_dims, 0);
    RETURN_IF_ERROR(ort_api.GetDimensions(type_shape, shape.data(), shape.size()));

    const void* raw_data = nullptr;
    RETURN_IF_ERROR(ort_api.GetTensorData(input, &raw_data));

    const float* float_data = static_cast<const float*>(raw_data);
    data = gsl::span<const float>(float_data, num_elems);
    return nullptr;
  }

  OrtStatus* Compute(OrtKernelContext* kernel_context) {
    RETURN_IF_ERROR(ort_api.Logger_LogMessage(&logger,
                                              OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                              "MulKernel::Compute", ORT_FILE, __LINE__, __FUNCTION__));
    gsl::span<const float> input0;
    gsl::span<const float> input1;
    std::vector<int64_t> shape0;
    std::vector<int64_t> shape1;

    size_t num_inputs = 0;
    RETURN_IF_ERROR(ort_api.KernelContext_GetInputCount(kernel_context, &num_inputs));

    if (num_inputs == 2) {
      // Both inputs are non-constant. Get them from ORT's KernelContext.
      RETURN_IF_ERROR(GetInputDataAndShape(kernel_context, 0, input0, shape0));
      RETURN_IF_ERROR(GetInputDataAndShape(kernel_context, 1, input1, shape1));
    } else if (num_inputs == 1) {
      // ORT is only providing one non-constant input because this EP chose not to request constant initializer inputs.
      // Get the constant input from the initializers saved by the EP.
      // Refer to "NodeFusionOptions_DropConstantInitializers()".

      if (const FloatInitializer* const_input0 = TryGetSavedInitializer(input0_name); const_input0 != nullptr) {
        RETURN_IF_ERROR(GetInputDataAndShape(kernel_context, 0, input1, shape1));
        input0 = gsl::span<const float>(const_input0->data);
        shape0 = const_input0->shape;
      } else if (const FloatInitializer* const_input1 = TryGetSavedInitializer(input1_name); const_input1 != nullptr) {
        RETURN_IF_ERROR(GetInputDataAndShape(kernel_context, 0, input0, shape0));
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

    RETURN_IF(shape0 != shape1, ort_api, "Expected same dimensions for both inputs");  // No broadcasting.

    size_t num_outputs = 0;
    RETURN_IF_ERROR(ort_api.KernelContext_GetOutputCount(kernel_context, &num_outputs));
    RETURN_IF(num_outputs != 1, ort_api, "Expected 1 output for MulKernel");

    OrtValue* output = nullptr;
    float* output_data = nullptr;
    RETURN_IF_ERROR(ort_api.KernelContext_GetOutput(kernel_context, 0, shape0.data(), shape0.size(), &output));
    RETURN_IF_ERROR(ort_api.GetTensorMutableData(output, reinterpret_cast<void**>(&output_data)));

    for (size_t i = 0; i < input0.size(); ++i) {
      output_data[i] = input0[i] * input1[i];
    }

    return nullptr;
  }

  const OrtApi& ort_api;
  const OrtLogger& logger;
  const std::unordered_map<std::string, FloatInitializer>& float_initializers;
  std::string input0_name;
  std::string input1_name;
};

/// <summary>
/// Example OrtNodeComputeInfo that represents the computation function for a compiled OrtGraph.
/// </summary>
struct ExampleNodeComputeInfo : OrtNodeComputeInfo {
  explicit ExampleNodeComputeInfo(ExampleEp& ep);

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
  CreateAllocator = CreateAllocatorImpl;                      // optional. can be nullptr
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;  // optional. can be nullptr

  auto status = ort_api.Logger_LogMessage(&logger_,
                                          OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                          ("ExampleEp has been created with name " + name_).c_str(),
                                          ORT_FILE, __LINE__, __FUNCTION__);
  // ignore status for now
  (void)status;
}

ExampleEp::~ExampleEp() = default;

/*static*/
const char* ORT_API_CALL ExampleEp ::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const ExampleEp*>(this_ptr);
  return ep->name_.c_str();
}

OrtStatus* ExampleEp::SaveConstantInitializers(const OrtGraph* graph) {
  size_t num_initializers = 0;
  RETURN_IF_ERROR(ort_api.Graph_GetNumInitializers(graph, &num_initializers));

  std::vector<const OrtValueInfo*> initializers(num_initializers);
  RETURN_IF_ERROR(ort_api.Graph_GetInitializers(graph, initializers.data(), initializers.size()));

  for (const OrtValueInfo* initializer : initializers) {
    bool is_constant = false;
    RETURN_IF_ERROR(ort_api.ValueInfo_IsConstantInitializer(initializer, &is_constant));

    if (is_constant) {
      const char* name = nullptr;
      const OrtValue* value = nullptr;
      OrtTensorTypeAndShapeInfo* type_shape = nullptr;
      DeferOrtRelease<OrtTensorTypeAndShapeInfo> release_type(&type_shape, ort_api.ReleaseTensorTypeAndShapeInfo);
      size_t num_elems = 0;

      RETURN_IF_ERROR(ort_api.GetValueInfoName(initializer, &name));
      RETURN_IF_ERROR(ort_api.ValueInfo_GetInitializerValue(initializer, &value));
      RETURN_IF_ERROR(ort_api.GetTensorTypeAndShape(value, &type_shape));
      RETURN_IF_ERROR(ort_api.GetTensorShapeElementCount(type_shape, &num_elems));

      ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      RETURN_IF_ERROR(ort_api.GetTensorElementType(type_shape, &elem_type));
      RETURN_IF(elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ort_api, "Expected float32 initializers");

      size_t num_dims = 0;
      RETURN_IF_ERROR(ort_api.GetDimensionsCount(type_shape, &num_dims));

      std::vector<int64_t> dims(num_dims, 0);
      RETURN_IF_ERROR(ort_api.GetDimensions(type_shape, dims.data(), dims.size()));

      const float* data = nullptr;
      RETURN_IF_ERROR(ort_api.GetTensorMutableData(const_cast<OrtValue*>(value), (void**)&data));

      FloatInitializer ep_initializer = {std::move(dims), std::vector<float>(data, data + num_elems)};
      float_initializers_.emplace(name, std::move(ep_initializer));
    }
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                     OrtEpGraphSupportInfo* graph_support_info) noexcept {
  ExampleEp* ep = static_cast<ExampleEp*>(this_ptr);

  size_t num_nodes = 0;
  RETURN_IF_ERROR(ep->ort_api.Graph_GetNumNodes(graph, &num_nodes));

  if (num_nodes == 0) {
    return nullptr;  // No nodes to process
  }

  std::vector<const OrtNode*> nodes(num_nodes);
  RETURN_IF_ERROR(ep->ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));

  std::vector<const OrtNode*> supported_nodes;

  for (const OrtNode* node : nodes) {
    const char* op_type = nullptr;
    RETURN_IF_ERROR(ep->ort_api.Node_GetOperatorType(node, &op_type));

    if (std::strncmp(op_type, "Mul", 4) == 0) {
      // Check that Mul has inputs/output of type float
      size_t num_inputs = 0;
      size_t num_outputs = 0;
      RETURN_IF_ERROR(ep->ort_api.Node_GetNumInputs(node, &num_inputs));
      RETURN_IF_ERROR(ep->ort_api.Node_GetNumOutputs(node, &num_outputs));
      RETURN_IF(num_inputs != 2 || num_outputs != 1, ep->ort_api, "Mul should have 2 inputs and 1 output");

      std::vector<const OrtValueInfo*> inputs(num_inputs);
      std::vector<const OrtValueInfo*> outputs(num_outputs);
      RETURN_IF_ERROR(ep->ort_api.Node_GetInputs(node, inputs.data(), inputs.size()));
      RETURN_IF_ERROR(ep->ort_api.Node_GetOutputs(node, outputs.data(), outputs.size()));

      std::array<bool, 3> is_float = {false, false, false};
      RETURN_IF_ERROR(IsFloatTensor(ep->ort_api, inputs[0], is_float[0]));
      RETURN_IF_ERROR(IsFloatTensor(ep->ort_api, inputs[1], is_float[1]));
      RETURN_IF_ERROR(IsFloatTensor(ep->ort_api, outputs[0], is_float[2]));
      if (!is_float[0] || !is_float[1] || !is_float[2]) {
        continue;  // Input or output is not of type float
      }

      supported_nodes.push_back(node);  // Only support a single Mul for now.
      break;
    }
  }

  // Create (optional) fusion options for the supported nodes to fuse.
  OrtNodeFusionOptions node_fusion_options = {};
  node_fusion_options.ort_version_supported = ORT_API_VERSION;

  // Set "drop constant initializers" to true if the compiling EP doesn't need ORT to provide constant initializers
  // as inputs to the fused/compiled node at inference time. This allows ORT to release unused initializers.
  // This example EP sets this to true and saves initializers during the call to OrtEp::Compile for use
  // during inference.
  node_fusion_options.drop_constant_initializers = true;
  RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info, supported_nodes.data(),
                                                               supported_nodes.size(), &node_fusion_options));

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEp::CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** graphs,
                                               _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                               _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                               _Out_writes_(count) OrtNode** ep_context_nodes) noexcept {
  ExampleEp* ep = static_cast<ExampleEp*>(this_ptr);
  const OrtApi& ort_api = ep->ort_api;

  if (count != 1) {
    return ort_api.CreateStatus(ORT_EP_FAIL, "Expected to compile a single graph");
  }

  // In GetCapability(), this EP specified that it doesn't need ORT to provide constant initializers during inference.
  // So, this EP saves constant initializers so that they're available during inference, but an actual EP
  // implementation could transfer the weights to device memory.
  ep->SaveConstantInitializers(graphs[0]);

  size_t num_nodes = 0;
  RETURN_IF_ERROR(ep->ort_api.Graph_GetNumNodes(graphs[0], &num_nodes));

  std::vector<const OrtNode*> nodes(num_nodes);
  RETURN_IF_ERROR(ep->ort_api.Graph_GetNodes(graphs[0], nodes.data(), nodes.size()));

  if (num_nodes != 1) {
    return ort_api.CreateStatus(ORT_EP_FAIL, "Expected to compile a single Mul node");
  }

  const char* node_op_type = nullptr;
  RETURN_IF_ERROR(ort_api.Node_GetOperatorType(nodes[0], &node_op_type));

  if (std::strncmp(node_op_type, "Mul", 4) != 0) {
    return ort_api.CreateStatus(ORT_EP_FAIL, "Expected to compile a single Mul node");
  }

  // Now we know we're compiling a single Mul node. Create a computation kernel.
  std::array<const OrtValueInfo*, 2> node_inputs = {};
  std::array<const char*, 2> node_input_names = {};

  RETURN_IF_ERROR(ort_api.Node_GetInputs(nodes[0], node_inputs.data(), node_inputs.size()));
  RETURN_IF_ERROR(ort_api.GetValueInfoName(node_inputs[0], &node_input_names[0]));
  RETURN_IF_ERROR(ort_api.GetValueInfoName(node_inputs[1], &node_input_names[1]));

  const char* ep_name = nullptr;
  RETURN_IF_ERROR(ort_api.Node_GetEpName(fused_nodes[0], &ep_name));
  if (std::strncmp(ep_name, "example_ep", 11) != 0) {
    return ort_api.CreateStatus(ORT_EP_FAIL, "The fused node is expected to assigned to this EP to run on");
  }

  // Associate the name of the fused node with our MulKernel.
  const char* fused_node_name = nullptr;
  RETURN_IF_ERROR(ort_api.Node_GetName(fused_nodes[0], &fused_node_name));

  ep->kernels_.emplace(std::string(fused_node_name), std::make_unique<MulKernel>(ep->ort_api, ep->logger_,
                                                                                 ep->float_initializers_,
                                                                                 node_input_names[0],
                                                                                 node_input_names[1]));

  // Update the OrtNodeComputeInfo associated with the graph.
  auto node_compute_info = std::make_unique<ExampleNodeComputeInfo>(*ep);
  node_compute_infos[0] = node_compute_info.release();

  // Create EpContext nodes for the fused nodes we compiled.
  if (ep->config_.enable_ep_context) {
    assert(ep_context_nodes != nullptr);
    RETURN_IF_ERROR(ep->CreateEpContextNodes(gsl::span<const OrtNode*>(fused_nodes, count),
                                             gsl::span<OrtNode*>(ep_context_nodes, count)));
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL ExampleEp::ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                         OrtNodeComputeInfo** node_compute_infos,
                                                         size_t num_node_compute_infos) noexcept {
  (void)this_ptr;
  for (size_t i = 0; i < num_node_compute_infos; i++) {
    delete node_compute_infos[i];
  }
}

// Creates EPContext nodes from the given fused nodes.
// This is an example implementation that can be used to generate an EPContext model. However, this example EP
// cannot currently run the EPContext model.
OrtStatus* ExampleEp::CreateEpContextNodes(gsl::span<const OrtNode*> fused_nodes,
                                           /*out*/ gsl::span<OrtNode*> ep_context_nodes) {
  assert(fused_nodes.size() == ep_context_nodes.size());

  // Helper to collect input or output names from an array of OrtValueInfo instances.
  auto collect_input_output_names = [&](gsl::span<const OrtValueInfo* const> value_infos,
                                        std::vector<const char*>& result) -> OrtStatus* {
    size_t num_values = value_infos.size();
    std::vector<const char*> value_names(num_values);

    for (size_t i = 0; i < num_values; ++i) {
      const OrtValueInfo* value_info = value_infos[i];
      RETURN_IF_ERROR(ort_api.GetValueInfoName(value_info, &value_names[i]));
    }

    result = std::move(value_names);
    return nullptr;
  };

  // Create an "EPContext" node for every fused node.
  for (size_t i = 0; i < fused_nodes.size(); ++i) {
    const OrtNode* fused_node = fused_nodes[i];
    const char* fused_node_name = nullptr;

    RETURN_IF_ERROR(ort_api.Node_GetName(fused_node, &fused_node_name));

    size_t num_fused_node_inputs = 0;
    size_t num_fused_node_outputs = 0;
    RETURN_IF_ERROR(ort_api.Node_GetNumInputs(fused_node, &num_fused_node_inputs));
    RETURN_IF_ERROR(ort_api.Node_GetNumOutputs(fused_node, &num_fused_node_outputs));

    std::vector<const OrtValueInfo*> fused_node_inputs(num_fused_node_inputs);
    std::vector<const OrtValueInfo*> fused_node_outputs(num_fused_node_outputs);
    RETURN_IF_ERROR(ort_api.Node_GetInputs(fused_node, fused_node_inputs.data(), fused_node_inputs.size()));
    RETURN_IF_ERROR(ort_api.Node_GetOutputs(fused_node, fused_node_outputs.data(), fused_node_outputs.size()));

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    RETURN_IF_ERROR(collect_input_output_names(fused_node_inputs, /*out*/ input_names));
    RETURN_IF_ERROR(collect_input_output_names(fused_node_outputs, /*out*/ output_names));

    int64_t is_main_context = (i == 0);
    int64_t embed_mode = 1;

    // Create node attributes. The CreateNode() function copies the attributes, so we have to release them.
    std::array<OrtOpAttr*, 6> attributes = {};
    DeferOrtRelease<OrtOpAttr> defer_release_attrs(attributes.data(), attributes.size(), ort_api.ReleaseOpAttr);

    std::string ep_ctx = "binary_data";
    RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_cache_context", ep_ctx.c_str(), static_cast<int>(ep_ctx.length()),
                                         ORT_OP_ATTR_STRING, &attributes[0]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr("main_context", &is_main_context, 1, ORT_OP_ATTR_INT, &attributes[1]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr("embed_mode", &embed_mode, 1, ORT_OP_ATTR_INT, &attributes[2]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_sdk_version", "1", 1, ORT_OP_ATTR_STRING, &attributes[3]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr("partition_name", fused_node_name, static_cast<int>(strlen(fused_node_name)),
                                         ORT_OP_ATTR_STRING, &attributes[4]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr("source", this->name_.c_str(), static_cast<int>(this->name_.length()),
                                         ORT_OP_ATTR_STRING, &attributes[5]));

    RETURN_IF_ERROR(model_editor_api.CreateNode("EPContext", "com.microsoft", fused_node_name,
                                                input_names.data(), input_names.size(),
                                                output_names.data(), output_names.size(),
                                                attributes.data(), attributes.size(),
                                                &ep_context_nodes[i]));
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


  // This creates an OrtSyncStreamImpl that is used is a specific session. 
    // The 

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
  auto kernel_it = ep.Kernels().find(fused_node_name);
  if (kernel_it == ep.Kernels().end()) {
    std::string message = "Unable to get kernel for fused node with name " + fused_node_name;
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
