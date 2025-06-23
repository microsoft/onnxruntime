#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <gsl/gsl>
#include <cassert>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define RETURN_IF_ERROR(fn)   \
  do {                        \
    OrtStatus* status = (fn); \
    if (status != nullptr) {  \
      return status;          \
    }                         \
  } while (0)

#define RETURN_IF(cond, ort_api, msg)                    \
  do {                                                   \
    if ((cond)) {                                        \
      return (ort_api).CreateStatus(ORT_EP_FAIL, (msg)); \
    }                                                    \
  } while (0)

struct ExampleEp;

// Helper to release an Ort object at the end of its scope.
template <typename T>
struct DeferOrtRelease {
  DeferOrtRelease(T** obj_ptr, std::function<void(T*)> release_func) : obj_ptr_(obj_ptr), release_func_(release_func) {}
  ~DeferOrtRelease() {
    if (obj_ptr_ != nullptr && *obj_ptr_ != nullptr) {
      release_func_(*obj_ptr_);
      *obj_ptr_ = nullptr;
    }
  }
  T** obj_ptr_ = nullptr;
  std::function<void(T*)> release_func_ = nullptr;
};

struct FloatInitializer {
  std::vector<int64_t> shape;
  std::vector<float> data;
};

/// <summary>
/// Example implementation of ONNX Mul. Does not handle many things like broadcasting.
/// </summary>
struct MulKernel {
  MulKernel(ExampleEp* ep, const char* input0_name, const char* input1_name)
      : ort_api(ep->ort_api),
        logger(ep->logger_),
        float_initializers(ep->float_initializers),
        input0_name(input0_name),
        input1_name(input1_name) {}

  const FloatInitializer* TryGetSavedInitializer(const char* name) const {
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

    const float* raw_data = nullptr;
    RETURN_IF_ERROR(ort_api.GetTensorMutableData(const_cast<OrtValue*>(input), (void**)&raw_data));  // No const-correct API?

    data = gsl::span<const float>(raw_data, num_elems);
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
      } else if (const FloatInitializer* const_input1 = TryGetSavedInitializer(input0_name); const_input1 != nullptr) {
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
  const char* input0_name = nullptr;
  const char* input1_name = nullptr;
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

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

static OrtStatus* IsFloatTensor(const OrtApi& ort_api, const OrtValueInfo* value_info, bool& result) {
  result = false;

  const OrtTypeInfo* type_info = nullptr;
  RETURN_IF_ERROR(ort_api.GetValueInfoTypeInfo(value_info, &type_info));

  ONNXType onnx_type = ONNX_TYPE_UNKNOWN;
  RETURN_IF_ERROR(ort_api.GetOnnxTypeFromTypeInfo(type_info, &onnx_type));
  if (onnx_type != ONNX_TYPE_TENSOR) {
    return nullptr;
  }

  const OrtTensorTypeAndShapeInfo* type_shape = nullptr;
  RETURN_IF_ERROR(ort_api.CastTypeInfoToTensorInfo(type_info, &type_shape));

  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  RETURN_IF_ERROR(ort_api.GetTensorElementType(type_shape, &elem_type));
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return nullptr;
  }

  result = true;
  return nullptr;
}

/// <summary>
/// Example EP that can compile a single Mul operator.
/// </summary>
struct ExampleEp : OrtEp, ApiPtrs {
  ExampleEp(ApiPtrs apis, const std::string& name, const OrtSessionOptions& session_options, const OrtLogger& logger)
      : ApiPtrs(apis), name_{name}, logger_{logger} {
    // Initialize the execution provider.
    auto status = ort_api.Logger_LogMessage(&logger_,
                                            OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                            ("ExampleEp has been created with name " + name_).c_str(),
                                            ORT_FILE, __LINE__, __FUNCTION__);
    // ignore status for now
    (void)status;

    // Can get configurations from the session options.
    // Note: should not store a direct reference to the session options object as its lifespan is not guaranteed.
    // EP should copy any configurations or settings it needs.
    (void)session_options;

    ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
    GetName = GetNameImpl;
    GetCapability = GetCapabilityImpl;
    Compile = CompileImpl;
    ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  }

  ~ExampleEp() {
    // Clean up the execution provider
  }

  OrtStatus* SaveConstantInitializers(const OrtGraph* graph) {
    OrtArrayOfConstObjects* initializers = nullptr;
    DeferOrtRelease<OrtArrayOfConstObjects> release_initializers(&initializers, ort_api.ReleaseArrayOfConstObjects);
    size_t num_initializers = 0;

    RETURN_IF_ERROR(ort_api.Graph_GetInitializers(graph, &initializers));
    RETURN_IF_ERROR(ort_api.ArrayOfConstObjects_GetSize(initializers, &num_initializers));

    for (size_t i = 0; i < num_initializers; ++i) {
      const OrtValueInfo* initializer = nullptr;
      RETURN_IF_ERROR(ort_api.ArrayOfConstObjects_GetElementAt(initializers, i,
                                                               reinterpret_cast<const void**>(&initializer)));

      bool is_constant = false;
      RETURN_IF_ERROR(ort_api.ValueInfo_IsConstantInitializer(initializer, &is_constant));

      if (is_constant) {
        const char* name = nullptr;
        const OrtValue* value = nullptr;
        OrtTensorTypeAndShapeInfo* type_shape = nullptr;
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
        float_initializers.emplace(name, std::move(ep_initializer));
      }
    }

    return nullptr;
  }

  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) {
    const auto* ep = static_cast<const ExampleEp*>(this_ptr);
    return ep->name_.c_str();
  }

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) {
    ExampleEp* ep = static_cast<ExampleEp*>(this_ptr);

    OrtArrayOfConstObjects* nodes_array = nullptr;
    DeferOrtRelease<OrtArrayOfConstObjects> release_nodes_array(&nodes_array, ep->ort_api.ReleaseArrayOfConstObjects);

    size_t num_nodes = 0;

    RETURN_IF_ERROR(ep->ort_api.Graph_GetNodes(graph, &nodes_array));
    RETURN_IF_ERROR(ep->ort_api.ArrayOfConstObjects_GetSize(nodes_array, &num_nodes));

    if (num_nodes == 0) {
      return nullptr;  // No nodes to process
    }

    const void* const* nodes_data = nullptr;
    RETURN_IF_ERROR(ep->ort_api.ArrayOfConstObjects_GetData(nodes_array, &nodes_data));
    auto nodes_span = gsl::span<const OrtNode* const>(reinterpret_cast<const OrtNode* const*>(nodes_data), num_nodes);

    std::vector<const OrtNode*> supported_nodes;

    for (const OrtNode* node : nodes_span) {
      const char* op_type = nullptr;
      RETURN_IF_ERROR(ep->ort_api.Node_GetOperatorType(node, &op_type));

      if (std::strncmp(op_type, "Mul", 4) == 0) {
        // Check that Mul has inputs/output of type float
        OrtArrayOfConstObjects* inputs_array = nullptr;
        OrtArrayOfConstObjects* outputs_array = nullptr;
        DeferOrtRelease<OrtArrayOfConstObjects> release_inputs(&inputs_array, ep->ort_api.ReleaseArrayOfConstObjects);
        DeferOrtRelease<OrtArrayOfConstObjects> release_outputs(&outputs_array, ep->ort_api.ReleaseArrayOfConstObjects);

        RETURN_IF_ERROR(ep->ort_api.Node_GetInputs(node, &inputs_array));
        RETURN_IF_ERROR(ep->ort_api.Node_GetOutputs(node, &outputs_array));

        size_t num_inputs = 0;
        size_t num_outputs = 0;
        RETURN_IF_ERROR(ep->ort_api.ArrayOfConstObjects_GetSize(inputs_array, &num_inputs));
        RETURN_IF_ERROR(ep->ort_api.ArrayOfConstObjects_GetSize(outputs_array, &num_outputs));
        RETURN_IF(num_inputs != 2 || num_outputs != 1, ep->ort_api, "Mul should have 2 inputs and 1 output");

        const void* const* inputs_data = nullptr;
        const void* const* outputs_data = nullptr;
        RETURN_IF_ERROR(ep->ort_api.ArrayOfConstObjects_GetData(inputs_array, &inputs_data));
        RETURN_IF_ERROR(ep->ort_api.ArrayOfConstObjects_GetData(outputs_array, &outputs_data));

        std::array<bool, 3> is_float = {false, false, false};
        RETURN_IF_ERROR(IsFloatTensor(ep->ort_api, static_cast<const OrtValueInfo*>(inputs_data[0]), is_float[0]));
        RETURN_IF_ERROR(IsFloatTensor(ep->ort_api, static_cast<const OrtValueInfo*>(inputs_data[1]), is_float[1]));
        RETURN_IF_ERROR(IsFloatTensor(ep->ort_api, static_cast<const OrtValueInfo*>(outputs_data[0]), is_float[2]));
        if (!is_float[0] || !is_float[1] || !is_float[2]) {
          continue;  // Input or output is not of type float
        }

        supported_nodes.push_back(node);  // Only support a single Mul for now.
        break;
      }
    }
    OrtNodeFusionOptions* node_fusion_options = nullptr;
    RETURN_IF_ERROR(ep->ep_api.CreateNodeFusionOptions(supported_nodes.data(), supported_nodes.size(),
                                                       &node_fusion_options));

    // Set "drop constant initializers" to true if the compiling EP doesn't need ORT to provide constant initializers
    // as inputs to the fused/compiled node at inference time. This allows ORT to release unused initializers.
    // This example EP sets this to true and saves initializers during the call to OrtEp::Compile for use
    // during inference.
    RETURN_IF_ERROR(ep->ep_api.NodeFusionOptions_DropConstantInitializers(node_fusion_options, true));

    // Note: ORT takes ownership of node_fusion_options.
    RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info, node_fusion_options));
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs, const OrtNode** fused_nodes,
                                             size_t count, OrtNodeComputeInfo** node_compute_infos) {
    ExampleEp* ep = static_cast<ExampleEp*>(this_ptr);
    const OrtApi& ort_api = ep->ort_api;

    if (count != 1) {
      return ep->ort_api.CreateStatus(ORT_EP_FAIL, "Expected to compile a single graph");
    }

    // In GetCapability(), this EP specified that it doesn't need ORT to provide constant initializers during inference.
    // So, this EP saves constant initializers so that they're available during inference, but an actual EP
    // implementation could transfer the weights to device memory.
    ep->SaveConstantInitializers(graphs[0]);

    OrtArrayOfConstObjects* nodes_array = nullptr;
    DeferOrtRelease<OrtArrayOfConstObjects> release_nodes(&nodes_array, ort_api.ReleaseArrayOfConstObjects);
    size_t num_nodes = 0;

    RETURN_IF_ERROR(ort_api.Graph_GetNodes(graphs[0], &nodes_array));
    RETURN_IF_ERROR(ort_api.ArrayOfConstObjects_GetSize(nodes_array, &num_nodes));

    if (num_nodes != 1) {
      return ort_api.CreateStatus(ORT_EP_FAIL, "Expected to compile a single Mul node");
    }

    const OrtNode* node_to_compile = nullptr;
    RETURN_IF_ERROR(ort_api.ArrayOfConstObjects_GetElementAt(nodes_array, 0,
                                                             reinterpret_cast<const void**>(&node_to_compile)));

    const char* node_op_type = nullptr;
    RETURN_IF_ERROR(ort_api.Node_GetOperatorType(node_to_compile, &node_op_type));

    if (std::strncmp(node_op_type, "Mul", 4) != 0) {
      return ort_api.CreateStatus(ORT_EP_FAIL, "Expected to compile a single Mul node");
    }

    // Now we know we're compiling a single Mul node. Create a computation kernel.
    OrtArrayOfConstObjects* inputs = nullptr;
    DeferOrtRelease<OrtArrayOfConstObjects> release_inputs(&inputs, ep->ort_api.ReleaseArrayOfConstObjects);

    RETURN_IF_ERROR(ep->ort_api.Node_GetInputs(node_to_compile, &inputs));
    const OrtValueInfo* input0 = nullptr;
    const OrtValueInfo* input1 = nullptr;

    RETURN_IF_ERROR(ort_api.ArrayOfConstObjects_GetElementAt(inputs, 0, reinterpret_cast<const void**>(&input0)));
    RETURN_IF_ERROR(ort_api.ArrayOfConstObjects_GetElementAt(inputs, 1, reinterpret_cast<const void**>(&input1)));

    const char* input0_name = nullptr;
    const char* input1_name = nullptr;
    RETURN_IF_ERROR(ort_api.GetValueInfoName(input0, &input0_name));
    RETURN_IF_ERROR(ort_api.GetValueInfoName(input1, &input1_name));

    // Associate the name of the fused node with our MulKernel.
    const char* fused_node_name = nullptr;
    RETURN_IF_ERROR(ep->ort_api.Node_GetName(fused_nodes[0], &fused_node_name));

    ep->kernels.emplace(std::string(fused_node_name), std::make_unique<MulKernel>(ep, input0_name, input1_name));

    // Update the OrtNodeComputeInfo associated with the graph.
    auto node_compute_info = std::make_unique<ExampleNodeComputeInfo>(*ep);
    node_compute_infos[0] = node_compute_info.release();

    return nullptr;
  }

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) {
    (void)this_ptr;
    for (size_t i = 0; i < num_node_compute_infos; i++) {
      delete node_compute_infos[i];
    }
  }

  std::string name_;
  const OrtLogger& logger_;
  std::unordered_map<std::string, std::unique_ptr<MulKernel>> kernels;
  std::unordered_map<std::string, FloatInitializer> float_initializers;
};

//
// Implementation of ExampleNodeComuteInfo
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
  auto kernel_it = ep.kernels.find(fused_node_name);
  if (kernel_it == ep.kernels.end()) {
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

/// <summary>
/// Example EP factory that can create an OrtEp and return information about the supported hardware devices.
/// </summary>
struct ExampleEpFactory : OrtEpFactory, ApiPtrs {
  ExampleEpFactory(const char* ep_name, ApiPtrs apis) : ApiPtrs(apis), ep_name_{ep_name} {
    ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
  }

  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) {
    const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
    return factory->ep_name_.c_str();
  }

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) {
    const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
    return factory->vendor_.c_str();
  }

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* p_num_ep_devices) {
    size_t& num_ep_devices = *p_num_ep_devices;
    auto* factory = static_cast<ExampleEpFactory*>(this_ptr);

    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      // C API
      const OrtHardwareDevice& device = *devices[i];
      if (factory->ort_api.HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
        // these can be returned as nullptr if you have nothing to add.
        OrtKeyValuePairs* ep_metadata = nullptr;
        OrtKeyValuePairs* ep_options = nullptr;
        factory->ort_api.CreateKeyValuePairs(&ep_metadata);
        factory->ort_api.CreateKeyValuePairs(&ep_options);

        // random example using made up values
        factory->ort_api.AddKeyValuePair(ep_metadata, "version", "0.1");
        factory->ort_api.AddKeyValuePair(ep_options, "run_really_fast", "true");

        // OrtEpDevice copies ep_metadata and ep_options.
        auto* status = factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, ep_metadata, ep_options,
                                                                   &ep_devices[num_ep_devices++]);

        factory->ort_api.ReleaseKeyValuePairs(ep_metadata);
        factory->ort_api.ReleaseKeyValuePairs(ep_options);

        if (status != nullptr) {
          return status;
        }
      }

      // C++ API equivalent. Throws on error.
      //{
      //  Ort::ConstHardwareDevice device(devices[i]);
      //  if (device.Type() == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      //    Ort::KeyValuePairs ep_metadata;
      //    Ort::KeyValuePairs ep_options;
      //    ep_metadata.Add("version", "0.1");
      //    ep_options.Add("run_really_fast", "true");
      //    Ort::EpDevice ep_device{*this_ptr, device, ep_metadata.GetConst(), ep_options.GetConst()};
      //    ep_devices[num_ep_devices++] = ep_device.release();
      //  }
      //}
    }

    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                              _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                              _In_ size_t num_devices,
                                              _In_ const OrtSessionOptions* session_options,
                                              _In_ const OrtLogger* logger,
                                              _Out_ OrtEp** ep) {
    auto* factory = static_cast<ExampleEpFactory*>(this_ptr);
    *ep = nullptr;

    if (num_devices != 1) {
      // we only registered for CPU and only expected to be selected for one CPU
      // if you register for multiple devices (e.g. CPU, GPU and maybe NPU) you will get an entry for each device
      // the EP has been selected for.
      return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                           "Example EP only supports selection for one device.");
    }

    // Create the execution provider
    RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(logger,
                                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                       "Creating Example EP", ORT_FILE, __LINE__, __FUNCTION__));

    // use properties from the device and ep_metadata if needed
    // const OrtHardwareDevice* device = devices[0];
    // const OrtKeyValuePairs* ep_metadata = ep_metadata[0];

    auto dummy_ep = std::make_unique<ExampleEp>(*factory, factory->ep_name_, *session_options, *logger);

    *ep = dummy_ep.release();
    return nullptr;
  }

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) {
    ExampleEp* dummy_ep = static_cast<ExampleEp*>(ep);
    delete dummy_ep;
  }

  const std::string ep_name_;            // EP name
  const std::string vendor_{"Contoso"};  // EP vendor name
};

// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

extern "C" {
//
// Public symbols
//
EXPORT_SYMBOL OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                                           OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ort_ep_api = ort_api->GetEpApi();

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<OrtEpFactory> factory = std::make_unique<ExampleEpFactory>(registration_name,
                                                                             ApiPtrs{*ort_api, *ort_ep_api});

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete factory;
  return nullptr;
}

}  // extern "C"
