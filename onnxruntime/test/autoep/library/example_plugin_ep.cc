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

/// <summary>
/// Example implementation of ONNX Mul. Does not handle many things like broadcasting.
/// </summary>
struct MulKernel {
  MulKernel(const OrtApi& ort_api, const OrtLogger& logger) : ort_api(ort_api), logger(logger) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) {
    RETURN_IF_ERROR(ort_api.Logger_LogMessage(&logger,
                                              OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                              "MulKernel::Compute", ORT_FILE, __LINE__, __FUNCTION__));
    size_t num_inputs = 0;
    RETURN_IF_ERROR(ort_api.KernelContext_GetInputCount(kernel_context, &num_inputs));
    RETURN_IF(num_inputs != 2, ort_api, "Expected 2 inputs for MulKernel");

    size_t num_outputs = 0;
    RETURN_IF_ERROR(ort_api.KernelContext_GetOutputCount(kernel_context, &num_outputs));
    RETURN_IF(num_outputs != 1, ort_api, "Expected 1 output for MulKernel");

    const OrtValue* input0 = nullptr;
    const OrtValue* input1 = nullptr;
    RETURN_IF_ERROR(ort_api.KernelContext_GetInput(kernel_context, 0, &input0));
    RETURN_IF_ERROR(ort_api.KernelContext_GetInput(kernel_context, 1, &input1));

    OrtTensorTypeAndShapeInfo* type_shape0 = nullptr;
    OrtTensorTypeAndShapeInfo* type_shape1 = nullptr;
    DeferOrtRelease<OrtTensorTypeAndShapeInfo> release_type0(&type_shape0, ort_api.ReleaseTensorTypeAndShapeInfo);
    DeferOrtRelease<OrtTensorTypeAndShapeInfo> release_type1(&type_shape1, ort_api.ReleaseTensorTypeAndShapeInfo);

    RETURN_IF_ERROR(ort_api.GetTensorTypeAndShape(input0, &type_shape0));
    RETURN_IF_ERROR(ort_api.GetTensorTypeAndShape(input1, &type_shape1));

    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    RETURN_IF_ERROR(ort_api.GetTensorElementType(type_shape0, &elem_type));
    RETURN_IF(elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ort_api, "Expected float32 inputs");

    size_t num_dims0 = 0;
    size_t num_dims1 = 0;
    RETURN_IF_ERROR(ort_api.GetDimensionsCount(type_shape0, &num_dims0));
    RETURN_IF_ERROR(ort_api.GetDimensionsCount(type_shape1, &num_dims1));
    RETURN_IF((num_dims0 == 0) || (num_dims1 == 0), ort_api, "Input has 0 dimensions");
    RETURN_IF(num_dims0 != num_dims1, ort_api, "Expected same dimensions for both inputs");  // No broadcasting

    std::vector<int64_t> dims0(num_dims0, 0);
    std::vector<int64_t> dims1(num_dims1, 0);
    RETURN_IF_ERROR(ort_api.GetDimensions(type_shape0, dims0.data(), dims0.size()));
    RETURN_IF_ERROR(ort_api.GetDimensions(type_shape1, dims1.data(), dims1.size()));
    RETURN_IF(dims0 != dims1, ort_api, "Expected same dimensions for both inputs");  // No broadcasting.

    const float* input_data0 = nullptr;
    const float* input_data1 = nullptr;
    RETURN_IF_ERROR(ort_api.GetTensorMutableData(const_cast<OrtValue*>(input0), (void**)&input_data0));  // No const-correct API?
    RETURN_IF_ERROR(ort_api.GetTensorMutableData(const_cast<OrtValue*>(input1), (void**)&input_data1));

    OrtValue* output = nullptr;
    RETURN_IF_ERROR(ort_api.KernelContext_GetOutput(kernel_context, 0, dims0.data(), dims0.size(), &output));

    float* output_data = nullptr;
    RETURN_IF_ERROR(ort_api.GetTensorMutableData(output, reinterpret_cast<void**>(&output_data)));

    int64_t num_elems = 1;
    for (int64_t dim : dims0) {
      RETURN_IF(dim < 0, ort_api, "Invalid dimension: negative value detected");
      num_elems *= dim;
    }

    for (size_t i = 0; i < static_cast<size_t>(num_elems); ++i) {
      output_data[i] = input_data0[i] * input_data1[i];
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
    RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info, supported_nodes.data(),
                                                                 supported_nodes.size()));
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs, const OrtNode** fused_nodes,
                                             size_t count, OrtNodeComputeInfo** node_compute_infos) {
    ExampleEp* ep = static_cast<ExampleEp*>(this_ptr);

    if (count != 1) {
      return ep->ort_api.CreateStatus(ORT_EP_FAIL, "Expected to compile a single graph");
    }

    OrtArrayOfConstObjects* nodes_array = nullptr;
    DeferOrtRelease<OrtArrayOfConstObjects> release_nodes(&nodes_array, ep->ort_api.ReleaseArrayOfConstObjects);
    size_t num_nodes = 0;

    RETURN_IF_ERROR(ep->ort_api.Graph_GetNodes(graphs[0], &nodes_array));
    RETURN_IF_ERROR(ep->ort_api.ArrayOfConstObjects_GetSize(nodes_array, &num_nodes));

    if (num_nodes != 1) {
      return ep->ort_api.CreateStatus(ORT_EP_FAIL, "Expected to compile a single Mul node");
    }

    const OrtNode* node_to_compile = nullptr;
    RETURN_IF_ERROR(ep->ort_api.ArrayOfConstObjects_GetElementAt(nodes_array, 0,
                                                                 reinterpret_cast<const void**>(&node_to_compile)));

    const char* node_op_type = nullptr;
    RETURN_IF_ERROR(ep->ort_api.Node_GetOperatorType(node_to_compile, &node_op_type));

    if (std::strncmp(node_op_type, "Mul", 4) != 0) {
      return ep->ort_api.CreateStatus(ORT_EP_FAIL, "Expected to compile a single Mul node");
    }

    // Now we know we're compiling a single Mul node.
    // Associate the name of the fused node with our MulKernel.
    const char* fused_node_name = nullptr;
    RETURN_IF_ERROR(ep->ort_api.Node_GetName(fused_nodes[0], &fused_node_name));

    ep->kernels.emplace(std::string(fused_node_name), std::make_unique<MulKernel>(ep->ort_api, ep->logger_));

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
  std::vector<const OrtHardwareDevice*> hardware_devices_;
  const OrtLogger& logger_;
  std::unordered_map<std::string, std::unique_ptr<MulKernel>> kernels;
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
