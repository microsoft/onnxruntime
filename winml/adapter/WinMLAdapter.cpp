// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"
#include "WinMLAdapter.h"
#include "WinMLAdapterErrors.h"
#include "CustomRegistryHelper.h"
#include "PheonixSingleton.h"
#include "LotusEnvironment.h"
#include "AbiCustomRegistryImpl.h"

#ifdef USE_DML
#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "core/providers/dml/GraphTransformers/GraphTransformerHelpers.h"
#include "core/providers/dml/OperatorAuthorHelper/SchemaInferenceOverrider.h"
#include "DmlOrtSessionBuilder.h"
#endif USE_DML

#include "LearningModelDevice.h"
#include "TensorFeatureDescriptor.h"
#include "ImageFeatureDescriptor.h"
#include "api.image/inc/D3DDeviceCache.h"
#include "Common/inc/WinMLTelemetryHelper.h"

#include "CpuOrtSessionBuilder.h"

#include <io.h>
#include <fcntl.h>

#include "ZeroCopyInputStreamWrapper.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

#include "FeatureDescriptorFactory.h"
#include "core\framework\utils.h"
#include "core\framework\session_state.h"
#include "core/providers/winml/winml_provider_factory.h"

using namespace winrt::Windows::AI::MachineLearning;

namespace Windows::AI::MachineLearning::Adapter {

// Define winml trace logging provider with WinML GUID
TRACELOGGING_DEFINE_PROVIDER(
    winml_trace_logging_provider,
    WINML_PROVIDER_DESC,
    WINML_PROVIDER_GUID);

// ORT intentionally requires callers derive from their session class to access
// the protected methods used below.
class InferenceSessionProtectedLoadAccessor : public onnxruntime::InferenceSession {
 public:
  onnxruntime::common::Status
  Load(std::unique_ptr<ONNX_NAMESPACE::ModelProto> p_model_proto) {
    return onnxruntime::InferenceSession::Load(std::move(p_model_proto));
  }
  const onnxruntime::SessionState& GetSessionState() {
    return *session_state_;
  }
};

class ModelProto : public Microsoft::WRL::RuntimeClass<
                       Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                       IModelProto> {
 public:
  ModelProto::ModelProto(onnx::ModelProto* model_proto) : model_proto_(model_proto) {
  }

  onnx::ModelProto* STDMETHODCALLTYPE get() noexcept override {
    return model_proto_.get();
  }

  onnx::ModelProto* STDMETHODCALLTYPE detach() noexcept override {
    return model_proto_.release();
  }

 private:
  std::unique_ptr<onnx::ModelProto> model_proto_;
};  // class ModelProto

class ModelInfo : public Microsoft::WRL::RuntimeClass<
                      Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                      IModelInfo> {
 private:
  std::string author_;
  std::string name_;
  std::string domain_;
  std::string description_;
  int64_t version_;
  std::unordered_map<std::string, std::string> model_metadata_;
  wfc::IVector<winml::ILearningModelFeatureDescriptor> input_features_;
  wfc::IVector<winml::ILearningModelFeatureDescriptor> output_features_;

 public:
  ModelInfo(const onnx::ModelProto* model_proto) {
    Initialize(model_proto);
  }

  const char* STDMETHODCALLTYPE author() noexcept override {
    return author_.c_str();
  }

  const char* STDMETHODCALLTYPE name() noexcept override {
    return name_.c_str();
  }

  const char* STDMETHODCALLTYPE domain() noexcept override {
    return domain_.c_str();
  }

  const char* STDMETHODCALLTYPE description() noexcept override {
    return description_.c_str();
  }

  int64_t STDMETHODCALLTYPE version() noexcept override {
    return version_;
  }

  HRESULT STDMETHODCALLTYPE GetModelMetadata(
      ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING>** metadata) override try {
    *metadata = nullptr;
    std::unordered_map<winrt::hstring, winrt::hstring> map_copy;
    for (auto& pair : model_metadata_) {
      auto key = WinML::Strings::HStringFromUTF8(pair.first);
      auto map_value = WinML::Strings::HStringFromUTF8(pair.second);
      map_copy.emplace(std::move(key), std::move(map_value));
    }
    auto out = winrt::single_threaded_map<winrt::hstring, winrt::hstring>(
        std::move(map_copy));

    winrt::copy_to_abi(out.GetView(), *(void**)metadata);
    return S_OK;
  }
  WINMLA_CATCH_ALL_COM

  HRESULT STDMETHODCALLTYPE GetInputFeatures(
      ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features) override try {
    *features = nullptr;
    winrt::copy_to_abi(input_features_.GetView(), *(void**)features);
    return S_OK;
  }
  WINMLA_CATCH_ALL_COM

  HRESULT STDMETHODCALLTYPE GetOutputFeatures(
      ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features) override try {
    *features = nullptr;
    winrt::copy_to_abi(output_features_.GetView(), *(void**)features);
    return S_OK;
  }
  WINMLA_CATCH_ALL_COM

  static std::vector<const char*>
  GetAllNodeOutputs(const onnx::ModelProto& model_proto) {
    std::vector<const char*> nodes_outputs;
    auto& graph = model_proto.graph();
    auto& nodes = graph.node();
    for (auto& node : nodes) {
      for (auto& node_output : node.output()) {
        nodes_outputs.push_back(node_output.c_str());
      }
    }
    return nodes_outputs;
  }

  static std::vector<const char*>
  GetInitializers(const onnx::ModelProto& model_proto) {
    std::vector<const char*> initializers;
    auto& graph = model_proto.graph();
    auto& graph_initializers = graph.initializer();
    for (auto& initializer : graph_initializers) {
      initializers.push_back(initializer.name().c_str());
    }
    return initializers;
  }

  static std::vector<const onnx::ValueInfoProto*>
  GetInputsWithoutInitializers(const onnx::ModelProto& model_proto) {
    auto initializers = GetInitializers(model_proto);

    std::vector<const onnx::ValueInfoProto*> inputs_without_initializers;
    auto& graph = model_proto.graph();
    auto& inputs = graph.input();
    for (auto& input : inputs) {
      if (input.has_name() && input.has_type()) {
        auto found_it = std::find_if(
            std::begin(initializers),
            std::end(initializers),
            [&](auto& initializer) {
              return std::strcmp(initializer, input.name().c_str()) == 0;
            });

        auto is_initializer = found_it != std::end(initializers);
        if (!is_initializer) {
          inputs_without_initializers.push_back(&input);
        }
      }
    }
    return inputs_without_initializers;
  }

  static std::vector<const onnx::ValueInfoProto*> GetOutputs(const onnx::ModelProto& model_proto) {
    std::vector<const onnx::ValueInfoProto*> outputs_with_name;
    auto& graph = model_proto.graph();
    auto& outputs = graph.output();
    for (auto& output : outputs) {
      if (output.has_name() && output.has_type()) {
        outputs_with_name.push_back(&output);
      }
    }
    return outputs_with_name;
  }

 private:
  void Initialize(const onnx::ModelProto* model_proto) {
    // metadata
    for (auto& prop : model_proto->metadata_props()) {
      model_metadata_[prop.key()] = prop.value();
    }

    WinML::FeatureDescriptorFactory builder(model_metadata_);

    // Create inputs
    auto inputs = GetInputsWithoutInitializers(*model_proto);
    input_features_ = builder.CreateDescriptorsFromValueInfoProtos(inputs);

    // Create outputs
    auto outputs = GetOutputs(*model_proto);
    output_features_ = builder.CreateDescriptorsFromValueInfoProtos(outputs);

    // author
    auto has_producer_name = model_proto->has_producer_name();
    author_ = has_producer_name
                  ? model_proto->producer_name()
                  : "";

    // domain
    auto has_domain = model_proto->has_domain();
    domain_ = has_domain
                  ? model_proto->domain()
                  : "";

    // name
    auto has_graph = model_proto->has_graph();
    auto graph_has_name = model_proto->graph().has_name();
    auto is_name_available = has_graph && graph_has_name;
    name_ = is_name_available
                ? model_proto->graph().name()
                : "";

    // description
    auto has_description = model_proto->has_doc_string();
    description_ = has_description
                       ? model_proto->doc_string()
                       : "";

    // version
    auto has_version = model_proto->has_model_version();
    version_ = has_version
                   ? model_proto->model_version()
                   : 0;
  }
};  // class ModelInfo

class WinMLAdapter : public Microsoft::WRL::RuntimeClass<
                         Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                         IWinMLAdapter> {
 public:
  // factory methods for creating an ort model from a path
  HRESULT STDMETHODCALLTYPE CreateModelProto(
      const char* path,
      IModelProto** model_proto) override try {
    int file_descriptor;
    _set_errno(0);  // clear errno
    _sopen_s(
        &file_descriptor,
        path,
        O_RDONLY | _O_SEQUENTIAL | _O_BINARY,
        _SH_DENYWR,
        _S_IREAD | _S_IWRITE);

    errno_t err = 0;
    _get_errno(&err);
    THROW_HR_IF_MSG(
        __HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND),
        err == ENOENT,
        "File not found: %s",
        path);

    THROW_HR_IF_MSG(
        E_FAIL,
        0 > file_descriptor,
        "Failed");  //errno

    auto stream = google::protobuf::io::FileInputStream(file_descriptor);
    stream.SetCloseOnDelete(true);

    auto model_proto_inner = new onnx::ModelProto();
    THROW_HR_IF_MSG(
        E_INVALIDARG,
        model_proto_inner->ParseFromZeroCopyStream(&stream) == false,
        "The stream failed to parse.");

    auto model_proto_outer = wil::MakeOrThrow<ModelProto>(model_proto_inner);
    return model_proto_outer.CopyTo(__uuidof(IModelProto), reinterpret_cast<void**>(model_proto));
  }
  WINMLA_CATCH_ALL_COM

  // factory methods for creating an ort model from a stream
  HRESULT STDMETHODCALLTYPE CreateModelProto(
      ABI::Windows::Storage::Streams::IRandomAccessStreamReference* stream_reference,
      IModelProto** model_proto) override try {
    ZeroCopyInputStreamWrapper wrapper(stream_reference);

    auto model_proto_inner = std::make_unique<onnx::ModelProto>();
    THROW_HR_IF_MSG(
        E_INVALIDARG,
        model_proto_inner->ParseFromZeroCopyStream(&wrapper) == false,
        "The stream failed to parse.");

    auto model_proto_outer = wil::MakeOrThrow<ModelProto>(model_proto_inner.release());
    return model_proto_outer.CopyTo(__uuidof(IModelProto), reinterpret_cast<void**>(model_proto));
  }
  WINMLA_CATCH_ALL_COM

  // factory methods for creating an ort model from a model_proto
  HRESULT STDMETHODCALLTYPE CreateModelProto(IModelProto* model_proto_in, IModelProto** model_proto) override try {
    auto model_proto_inner = new onnx::ModelProto(*model_proto_in->get());
    auto model_proto_outer = wil::MakeOrThrow<ModelProto>(model_proto_inner);
    return model_proto_outer.CopyTo(__uuidof(IModelProto), reinterpret_cast<void**>(model_proto));
  }
  WINMLA_CATCH_ALL_COM

  HRESULT STDMETHODCALLTYPE CreateModelInfo(IModelProto* model_proto, IModelInfo** model_info) override try {
    auto model_info_outer = wil::MakeOrThrow<ModelInfo>(model_proto->get());
    return model_info_outer.CopyTo(__uuidof(IModelInfo), reinterpret_cast<void**>(model_info));
  }
  WINMLA_CATCH_ALL_COM

  void STDMETHODCALLTYPE EnableDebugOutput() override try {
    WinML::CWinMLLogSink::EnableDebugOutput();
  }
  WINMLA_CATCH_ALL_DONOTHING

  static bool IsFeatureDescriptorFp16(
      winml::ILearningModelFeatureDescriptor descriptor) {
    if (auto imageFeatureDescriptor = descriptor.try_as<winml::IImageFeatureDescriptor2>()) {
      return TensorKind::Float16 == imageFeatureDescriptor.TensorKind();
    }

    if (auto tensorFeatureDescriptor = descriptor.try_as<winml::ITensorFeatureDescriptor>()) {
      return TensorKind::Float16 == tensorFeatureDescriptor.TensorKind();
    }

    return false;
  }

  HRESULT STDMETHODCALLTYPE EnsureModelDeviceCompatibility(
      winml::LearningModel const& model,
      IModelProto* p_model_proto,
      bool is_float16_supported) override try {
    if (!is_float16_supported) {
      auto& graph = p_model_proto->get()->graph();

      // The model will not contain fp16 operations if:
      // 1. The model has no fp16 inputs
      // 2. The model has no fp16 initializers
      // 3. The model does not create any fp16 intermediary tensors via the Cast (to float16) operator
      // 4. The model does not have any fp16 outputs

      // 1. Ensure that The model has no fp16 inputs
      for (auto descriptor : model.InputFeatures()) {
        THROW_HR_IF_MSG(
            DXGI_ERROR_UNSUPPORTED,
            IsFeatureDescriptorFp16(descriptor),
            "The model contains a 16-bit input (%ls), but the current device does not support 16-bit float.",
            descriptor.Name().c_str());
      }

      // 2. Ensure that the model has no fp16 initializers
      for (int i = 0; i < graph.node_size(); i++) {
        auto node = graph.node(i);
        if (node.op_type() == "Cast" && node.domain().empty()) {
          for (int attribIndex = 0; attribIndex < node.attribute_size(); attribIndex++) {
            auto attribute = node.attribute(attribIndex);
            if (attribute.name() == "to") {
              THROW_HR_IF_MSG(
                  DXGI_ERROR_UNSUPPORTED,
                  attribute.i() == onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16,
                  "The model contains a 16-bit float Cast Op (%s), but the current device does not support 16-bit float.",
                  node.name().c_str());
            }
          }
        }
      }

      // 3. Ensure that the model does not create any fp16 intermediary
      //    tensors via the Cast (to float16) operator
      for (int i = 0; i < graph.initializer_size(); i++) {
        auto initializer = graph.initializer(i);

        THROW_HR_IF_MSG(
            DXGI_ERROR_UNSUPPORTED,
            initializer.data_type() == onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16,
            "The model contains a 16-bit float initializer (%s), but the current device does not support 16-bit float.",
            initializer.name().c_str());
      }

      // 4. Ensure that the model does not have any fp16 outputs
      for (auto descriptor : model.OutputFeatures()) {
        THROW_HR_IF_MSG(
            DXGI_ERROR_UNSUPPORTED,
            IsFeatureDescriptorFp16(descriptor),
            "The model contains a 16-bit output (%ls), but the current device does not support 16-bit float.",
            descriptor.Name().c_str());
      }
    }
    return S_OK;
  }
  WINMLA_CATCH_ALL_COM

  ID3D12Resource* STDMETHODCALLTYPE GetD3D12ResourceFromAllocation(onnxruntime::IExecutionProvider* provider, void* allocation) override try {
#ifdef USE_DML
    auto d3dResource =
        Dml::GetD3D12ResourceFromAllocation(
            provider->GetAllocator(0, ::OrtMemType::OrtMemTypeDefault).get(),
            allocation);
    return d3dResource;
#else
    return nullptr;
#endif USE_DML
  } catch (...) {
    return nullptr;
  }

  static onnxruntime::MLDataType GetType(winml::TensorKind kind) {
    switch (kind) {
      case winml::TensorKind::Float:
        return onnxruntime::DataTypeImpl::GetType<float>();
      case winml::TensorKind::Float16:
        return onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>();
    };
    return nullptr;
  }

  // factory method for creating an ortsessionbuilder from a device
  HRESULT STDMETHODCALLTYPE CreateOrtSessionBuilder(
      ID3D12Device* device,
      ID3D12CommandQueue* queue,
      IOrtSessionBuilder** session_builder) override try {
    if (device == nullptr) {
      auto builder = wil::MakeOrThrow<OnnxruntimeCpuSessionBuilder>();
      return builder.CopyTo(__uuidof(IOrtSessionBuilder), reinterpret_cast<void**>(session_builder));
    }
#ifdef USE_DML
    else {
      auto builder = wil::MakeOrThrow<OnnxruntimeDmlSessionBuilder>(device, queue);
      return builder.CopyTo(__uuidof(IOrtSessionBuilder), reinterpret_cast<void**>(session_builder));
    }
#else
    return E_NOTIMPL;
#endif USE_DML
  }
  WINMLA_CATCH_ALL_COM

  HRESULT STDMETHODCALLTYPE GetMapType(const OrtValue* ort_value, ONNXTensorElementDataType* key_type, ONNXTensorElementDataType* value_type) override try {
    *key_type = *value_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    auto type = ort_value->Type();
    if (type == onnxruntime::DataTypeImpl::GetType<onnxruntime::MapStringToString>()) {
      *key_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
      *value_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    } else if (type == onnxruntime::DataTypeImpl::GetType<onnxruntime::MapStringToInt64>()) {
      *key_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
      *value_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    } else if (type == onnxruntime::DataTypeImpl::GetType<onnxruntime::MapStringToFloat>()) {
      *key_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
      *value_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    } else if (type == onnxruntime::DataTypeImpl::GetType<onnxruntime::MapStringToDouble>()) {
      *key_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
      *value_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    } else if (type == onnxruntime::DataTypeImpl::GetType<onnxruntime::MapInt64ToString>()) {
      *key_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      *value_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    } else if (type == onnxruntime::DataTypeImpl::GetType<onnxruntime::MapInt64ToInt64>()) {
      *key_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      *value_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    } else if (type == onnxruntime::DataTypeImpl::GetType<onnxruntime::MapInt64ToFloat>()) {
      *key_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      *value_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    } else if (type == onnxruntime::DataTypeImpl::GetType<onnxruntime::MapInt64ToDouble>()) {
      *key_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      *value_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    }
    return S_OK;
  }
  WINMLA_CATCH_ALL_COM

  HRESULT STDMETHODCALLTYPE GetVectorMapType(const OrtValue* ort_value, ONNXTensorElementDataType* key_type, ONNXTensorElementDataType* value_type) override try {
    *key_type = *value_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    auto type = ort_value->Type();
    if (type == onnxruntime::DataTypeImpl::GetType<onnxruntime::VectorMapStringToFloat>()) {
      *key_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
      *value_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    } else if (type == onnxruntime::DataTypeImpl::GetType<onnxruntime::VectorMapInt64ToFloat>()) {
      *key_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      *value_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    return S_OK;
  }
  WINMLA_CATCH_ALL_COM

  HRESULT STDMETHODCALLTYPE GetCustomRegistry(IMLOperatorRegistry** registry) override try {
#ifdef USE_DML
    auto impl = wil::MakeOrThrow<AbiCustomRegistryImpl>();
    *registry = impl.Detach();
    return S_OK;
#else
    return E_NOTIMPL;
#endif USE_DML
  }
  WINMLA_CATCH_ALL_COM

  HRESULT STDMETHODCALLTYPE GetOperatorRegistry(ILearningModelOperatorProviderNative* operator_provider_native, IMLOperatorRegistry** registry) override try {
#ifdef USE_DML
    // Retrieve the "operator abi" registry.
    winrt::com_ptr<IMLOperatorRegistry> operator_registry;
    THROW_IF_FAILED(operator_provider_native->GetRegistry(operator_registry.put()));
    *registry = operator_registry.detach();
    return S_OK;
#else
    return E_NOTIMPL;
#endif USE_DML
  }
  WINMLA_CATCH_ALL_COM

  void* STDMETHODCALLTYPE CreateGPUAllocationFromD3DResource(ID3D12Resource* pResource) override try {
#ifdef USE_DML
    return Dml::CreateGPUAllocationFromD3DResource(pResource);
#else
    return nullptr;
#endif USE_DML
  } catch (...) {
    return nullptr;
  }

  void STDMETHODCALLTYPE FreeGPUAllocation(void* ptr) override try {
#ifdef USE_DML
    Dml::FreeGPUAllocation(ptr);
#endif USE_DML
  }
  WINMLA_CATCH_ALL_DONOTHING

  HRESULT STDMETHODCALLTYPE CopyTensor(
      onnxruntime::IExecutionProvider* provider,
      OrtValue* src,
      OrtValue* dst) override try {
#ifdef USE_DML
    ORT_THROW_IF_ERROR(Dml::CopyTensor(provider, *(src->GetMutable<onnxruntime::Tensor>()), *(dst->GetMutable<onnxruntime::Tensor>())));
    return S_OK;
#else
    return E_NOTIMPL;
#endif USE_DML
  }
  WINMLA_CATCH_ALL_COM

  // Override select shape inference functions which are incomplete in ONNX with versions that are complete,
  // and are also used in DML kernel registrations.  Doing this avoids kernel and shader creation being
  // deferred until first evaluation.  It also prevents a situation where inference functions in externally
  // registered schema are reachable only after upstream schema have been revised in a later OS release,
  // which would be a compatibility risk.
  HRESULT STDMETHODCALLTYPE OverrideSchemaInferenceFunctions() override try {
#ifdef USE_DML
    static std::once_flag schema_override_once_flag;
    std::call_once(schema_override_once_flag, []() {
      SchemaInferenceOverrider::OverrideSchemaInferenceFunctions();
    });
    return S_OK;
#else
    return S_OK;  // needs to return S_OK otherwise everything breaks because this gets called from the learningmodel constructor
#endif USE_DML
  }
  WINMLA_CATCH_ALL_COM

  HRESULT STDMETHODCALLTYPE GetProviderMemoryInfo(
      onnxruntime::IExecutionProvider* provider,
      OrtMemoryInfo** memory_info) override try {
    auto allocator = provider->GetAllocator(0, ::OrtMemType::OrtMemTypeDefault);

    const auto& info = allocator->Info();
    *memory_info = new OrtMemoryInfo(info.name, info.type, info.device, info.id, info.mem_type);
    if (*memory_info == nullptr) {
      return E_OUTOFMEMORY;
    }
    return S_OK;
  }
  WINMLA_CATCH_ALL_COM

  HRESULT STDMETHODCALLTYPE GetValueMemoryInfo(const OrtValue* ort_value, OrtMemoryInfo** memory_info) override try {
    const auto& tensor = ort_value->Get<onnxruntime::Tensor>();
    auto info = tensor.Location();
    *memory_info = new OrtMemoryInfo(info.name, info.type, info.device, info.id, info.mem_type);
    if (*memory_info == nullptr) {
      return E_OUTOFMEMORY;
    }
    return S_OK;
  }
  WINMLA_CATCH_ALL_COM

  struct AllocatorWrapper : public OrtAllocator {
   public:
    AllocatorWrapper(onnxruntime::AllocatorPtr impl) : impl_(impl) {
      version = ORT_API_VERSION;
      Alloc = AllocImpl;
      Free = FreeImpl;
      Info = InfoImpl;
    }

    static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size) {
      return static_cast<AllocatorWrapper*>(this_)->impl_->Alloc(size);
    }
    static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p) {
      return static_cast<AllocatorWrapper*>(this_)->impl_->Free(p);
    }
    static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
      return &(static_cast<const AllocatorWrapper*>(this_)->impl_->Info());
    }

   private:
    onnxruntime::AllocatorPtr impl_;
  };

  HRESULT STDMETHODCALLTYPE GetProviderAllocator(
      onnxruntime::IExecutionProvider* provider,
      OrtAllocator** allocator) override try {
    auto allocator_ptr = provider->GetAllocator(0, ::OrtMemType::OrtMemTypeDefault);
    *allocator = new (std::nothrow) AllocatorWrapper(allocator_ptr);
    if (*allocator == nullptr) {
      return E_OUTOFMEMORY;
    }

    return S_OK;
  }
  WINMLA_CATCH_ALL_COM

  HRESULT STDMETHODCALLTYPE FreeProviderAllocator(
      OrtAllocator* allocator) override try {
    delete static_cast<AllocatorWrapper*>(allocator);
    return S_OK;
  }
  WINMLA_CATCH_ALL_COM
};

extern "C" HRESULT STDMETHODCALLTYPE OrtGetWinMLAdapter(IWinMLAdapter** adapter) try {
  // make an adapter instance
  Microsoft::WRL::ComPtr<WinMLAdapter> adapterptr = wil::MakeOrThrow<WinMLAdapter>();
  return adapterptr.CopyTo(__uuidof(IWinMLAdapter), reinterpret_cast<void**>(adapter));
}
WINMLA_CATCH_ALL_COM

// InferenceSession
// ================

InferenceSession::InferenceSession(onnxruntime::InferenceSession* session) : session_(session) {
}

void STDMETHODCALLTYPE InferenceSession::RegisterGraphTransformers() try {
#ifdef USE_DML
  // Bug 22973884 : Fix issues with BatchNorm + Add and BatchNorm + Mul handling implicit inputs, and move from Winml to ORT
  GraphTransformerHelpers::RegisterGraphTransformers(session_.get());
#endif USE_DML
}
WINMLA_CATCH_ALL_DONOTHING

HRESULT STDMETHODCALLTYPE InferenceSession::StartProfiling() try {
  this->session_->StartProfiling(PheonixSingleton<WinML::LotusEnvironment>()->GetDefaultLogger());
  return S_OK;
}
WINMLA_CATCH_ALL_COM

HRESULT STDMETHODCALLTYPE InferenceSession::EndProfiling() try {
  this->session_->EndProfiling();
  return S_OK;
}
WINMLA_CATCH_ALL_COM

HRESULT STDMETHODCALLTYPE
InferenceSession::LoadModel(
    IModelProto* model_proto) try {
  auto session_protected_load_accessor =
      static_cast<InferenceSessionProtectedLoadAccessor*>(session_.get());
  // session's like to have their very own copy of the model_proto, use detach()
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto_ptr(model_proto->detach());
  ORT_THROW_IF_ERROR(session_protected_load_accessor->Load(std::move(model_proto_ptr)));
  return S_OK;
}
WINMLA_CATCH_ALL_COM

HRESULT STDMETHODCALLTYPE
InferenceSession::RegisterCustomRegistry(
    IMLOperatorRegistry* registry) try {
  RETURN_HR_IF(S_OK, registry == nullptr);

#ifdef USE_DML
  auto custom_registries = GetLotusCustomRegistries(registry);

  // Register
  for (auto& custom_registry : custom_registries) {
    ORT_THROW_IF_ERROR(session_->RegisterCustomRegistry(custom_registry));
  }
#endif USE_DML

  return S_OK;
}
WINMLA_CATCH_ALL_COM

void STDMETHODCALLTYPE InferenceSession::FlushContext(onnxruntime::IExecutionProvider* dml_provider) try {
#ifdef USE_DML
  Dml::FlushContext(dml_provider);
#endif USE_DML
}
WINMLA_CATCH_ALL_DONOTHING

void STDMETHODCALLTYPE InferenceSession::TrimUploadHeap(onnxruntime::IExecutionProvider* dml_provider) try {
#ifdef USE_DML
  Dml::TrimUploadHeap(dml_provider);
#endif USE_DML
}
WINMLA_CATCH_ALL_DONOTHING

void STDMETHODCALLTYPE InferenceSession::ReleaseCompletedReferences(onnxruntime::IExecutionProvider* dml_provider) try {
#ifdef USE_DML
  Dml::ReleaseCompletedReferences(dml_provider);
#endif USE_DML
}
WINMLA_CATCH_ALL_DONOTHING

HRESULT STDMETHODCALLTYPE InferenceSession::CopyOneInputAcrossDevices(
    const char* input_name,
    const OrtValue* orig_mlvalue,
    OrtValue** new_mlvalue) try {
  auto session_protected_load_accessor =
      static_cast<InferenceSessionProtectedLoadAccessor*>(session_.get());
  const onnxruntime::SessionState& sessionState = session_protected_load_accessor->GetSessionState();
  auto temp_mlvalue = std::make_unique<OrtValue>();
  ORT_THROW_IF_ERROR(onnxruntime::utils::CopyOneInputAcrossDevices(sessionState, input_name, *orig_mlvalue, *temp_mlvalue.get()));
  *new_mlvalue = temp_mlvalue.release();
  return S_OK;
}
WINMLA_CATCH_ALL_COM

}  // namespace Windows::AI::MachineLearning::Adapter